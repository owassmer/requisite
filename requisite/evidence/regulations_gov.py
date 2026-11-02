"""The regulations.gov v4 API, read only (spec §7.3: "regulatory comment dockets (the regulations.gov API)"; spec §14:
public sources through their official APIs; decisions, "Public sources are reached through their official APIs").

Every request goes through the cassettes (requisite/cassettes.py) as provider "regulations_gov", keyed by the endpoint
path and the query parameters, so a replay reads exactly what the live run read and never touches the network. The key
travels in the X-Api-Key header and is read only inside a live call, so it never reaches a URL, a cassette key, a log or
a replay (decisions: "The regulations.gov key goes in the X-Api-Key header, never in a URL"). Every request names
Requisite by its User-Agent (decisions: the conventional self-identifying form, which the file CDN requires).

A response is kept as the bytes the source returned (text where they are UTF-8, base64 otherwise), so the artifact a
replay commits is the artifact the live run committed (spec I2).

Pacing. The key allows 1,000 requests in any rolling hour (api.data.gov: "The hourly counters for your API key reset on
a rolling basis"; research/sources/api-data-gov-developer-manual-web-service-rate-limits.md). Every response states how
many remain (x-ratelimit-remaining). One Pacer, shared by every thread of a run, lets a request go while more than a
small reserve remains beyond those in flight, and otherwise waits until the oldest of its own requests leaves the hour,
which frees a slot: a long read slows to the rate the hour frees requests and never runs the key dry. The key's hour is
also spent by others (another process, an earlier run), whose requests leave it unseen by this one, so a run that waits
lets one request go each probe interval with nothing in flight while others hold part of the hour (its limit, less what
remains, less this run's own requests in it), and that request's response states the count afresh
(November 1: a run that began with the key nearly spent by an earlier run stayed at that small share each hour after the
earlier requests had left it). A 429 is waited out and retried, never dropped; a refused request spends nothing. Pacing applies only to live calls: a replay has nothing to pace.

Listings. The API serves at most 20 pages of at most 250 results per query (its OpenAPI specification;
research/sources/gsa-the-regulations-gov-api-v4-openapi-specification-commit-.md). A listing is sorted by
lastModifiedDate,documentId and, after 20 full pages, continued from the last lastModifiedDate seen, inclusively, as the
API's documentation gives for large result sets, dropping the entries the seam repeats
(research/sources/gsa-open-technology-regulations-gov-api-documentation.md). Every page of every continuation is kept.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlsplit
from zoneinfo import ZoneInfo

import httpx

from requisite import cassettes
from requisite.evidence.fetched import USER_AGENT, Fetched, fetched, free_budget, kept
from requisite.secrets import secret

PROVIDER = "regulations_gov"
SOURCE = "regulations.gov"
API = "https://api.regulations.gov/v4/"
PAGE_SIZE = 250  # the API's largest page (OpenAPI: page[size] "between, and including, 5 and 250")
MAX_PAGES = 20  # the API's last page of a query (OpenAPI: page[number] "between, and including, 1 and 20")
EASTERN = ZoneInfo("America/New_York")  # the API reads a lastModifiedDate filter in Eastern time (its documentation)
WINDOW_S = 3600.0  # api.data.gov's rolling hour
RESERVE = 10  # requests left unused beyond those in flight, so responses that arrive out of order cannot overdraw the key
PROBE_S = 60.0  # how long a waiting run goes before one request reads the key's count afresh (others spend it too)
RETRIES = 6  # attempts after the first, for a server error (a 5xx) that may pass
# A request that gets no response at all (the network down, a handshake timing out) is retried, with backoff, for this
# long before the run stops: a frame reads for hours, and on November 2 a handshake timeout lasting a couple of minutes
# ended one after 3,600 requests. Longer than any network blip, short enough that a lost connection is said.
NETWORK_WAIT_S = 30 * 60.0
PASSES = 3  # listings of one query, where the record changes under the first (Listing)
LIMIT_WAIT_S = 2 * WINDOW_S  # a 429 lasting longer than this is no rate limit the rolling hour explains: it raises
TRANSIENT = frozenset({500, 502, 503, 504})
TIMEOUT = httpx.Timeout(120.0, connect=20.0)
KEPT_HEADERS = ("content-type", "content-length", "last-modified", "etag", "x-ratelimit-limit", "x-ratelimit-remaining")


class RegulationsGovError(RuntimeError):
    """A request the API (or its file host) did not answer after its retries, or answered with an error that is ours to
    look at (a malformed request, a refused key). Nothing is skipped: the run stops here."""


@dataclass(frozen=True)
class Entry:
    """One resource of a listing (a document or a comment), as the last page that served it gave it, and that page."""

    data: Mapping[str, Any]
    page: int

    @property
    def id(self) -> str:
        return str(self.data["id"])

    @property
    def attributes(self) -> Mapping[str, Any]:
        return self.data.get("attributes") or {}


def identity(resource: Mapping[str, Any]) -> tuple[str, str, str]:
    """A listed resource's identity: its id, with its type and posting date, which a modification does not change. The
    id alone is not unique: regulations.gov lists two different documents under one id (EPA-R03-OAR-2019-0295-0002, a
    proposed rule and a transmittal letter, in one response on November 2, 2026), so keyed by id one would be lost and
    the listing read as changing under itself."""
    attributes = resource.get("attributes") or {}
    return (str(resource["id"]), str(attributes.get("documentType")), str(attributes.get("postedDate")))


@dataclass(frozen=True)
class Listing:
    """A whole listing: its base request (path and filters, no page, no continuation), every page it took, continuations
    included, and each distinct resource once, in the order first served."""

    request: str
    pages: tuple[Fetched, ...]
    entries: tuple[Entry, ...]


def instant(value: str) -> datetime:
    """An API timestamp (ISO 8601 with its offset, e.g. 2026-05-19T04:00:00Z) as an aware instant."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def eastern_seam(last_modified: str) -> str:
    """The lastModifiedDate filter value that restarts a listing at `last_modified`, inclusively: the instant in Eastern
    time as yyyy-MM-dd HH:mm:ss (the API's documentation). Where that wall time occurs twice (the autumn hour), it is
    given an hour earlier, so whichever occurrence the API reads it as, nothing after the seam is skipped: the entries
    repeated are dropped as any seam's are."""
    at = instant(last_modified).astimezone(EASTERN)
    if at.fold == 0 and at.replace(fold=1).utcoffset() != at.utcoffset():
        at = (at - timedelta(hours=1)).astimezone(EASTERN)
    return at.strftime("%Y-%m-%d %H:%M:%S")


def locator(path: str, params: Mapping[str, str]) -> str:
    """The URL of an API request, without the key (which travels in a header)."""
    query = urlencode(sorted(params.items()), safe="[]")
    return f"{API}{path}" + (f"?{query}" if query else "")


class Pacer:
    """One run's pacing state, shared by its threads (see the module docstring). `clock` is monotonic seconds and `sleep`
    waits; both are injectable, so a test does not wait in real time."""

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        *,
        window: float = WINDOW_S,
        reserve: int = RESERVE,
        probe: float = PROBE_S,
    ):
        self.clock, self.sleep = clock, sleep
        self.window, self.reserve, self.probe = window, reserve, probe
        self._lock = threading.Lock()
        self._remaining: int | None = None  # as the API last stated it, less what has been sent since
        self._sent: deque[float] = deque()  # this run's requests within the last window
        self._inflight = 0
        self._resume: float | None = None  # a 429's Retry-After: no request goes before it
        self._probe_at: float | None = None  # while waiting: when one request goes to read the count afresh
        self._limit: int | None = None  # the key's hourly limit, as the API states it

    def _expire(self, now: float, *, frees: bool) -> None:
        while self._sent and self._sent[0] <= now - self.window:
            self._sent.popleft()
            if frees and self._remaining is not None:
                self._remaining += 1  # a request of ours left the hour: its slot is free again

    def _others(self) -> bool:
        """Whether requests not this run's hold part of the hour, so the count can rise without this run's own leaving
        it: unknown counts as yes (the count is then read afresh, never assumed)."""
        if self._remaining is None or self._limit is None:
            return True
        return self._limit - self._remaining - len(self._sent) > 0

    def _wait(self, now: float) -> float:
        """How long until a request may go: until the oldest of this run's requests leaves the hour, freeing a slot, or,
        where others hold part of the hour, until the probe interval has passed, whichever is sooner."""
        until = self._sent[0] + self.window - now if self._sent else None
        # Requests in flight state the count themselves when they settle, so a probe waits for none to be in flight.
        if self._inflight == 0 and (self._others() or until is None):
            if self._probe_at is None:
                self._probe_at = now + self.probe
            probe = self._probe_at - now
            until = probe if until is None else min(until, probe)
        else:
            self._probe_at = None
        if until is None:  # nothing of this run's in the hour, and requests in flight: wait a probe interval for them
            until = self.probe
        return max(until, 0.0)

    def acquire(self) -> float:
        """Wait until a request may go without spending the key's last requests, then count it as sent; returns when."""
        while True:
            with self._lock:
                now = self.clock()
                if self._resume is not None and now < self._resume:
                    wait = self._resume - now
                else:
                    self._resume = None
                    self._expire(now, frees=True)
                    room = self._remaining is None or self._remaining - self._inflight > self.reserve
                    probe = self._inflight == 0 and self._probe_at is not None and now >= self._probe_at
                    if room or probe:
                        self._probe_at = None
                        self._inflight += 1
                        self._sent.append(now)
                        return now
                    wait = self._wait(now)
            self.sleep(wait)

    def settle(self, remaining: int | None, limit: int | None = None) -> None:
        """A request came back, stating (or not) how many requests remain in the hour, and the hour's limit."""
        with self._lock:
            if limit is not None:
                self._limit = limit
            self._inflight -= 1
            self._expire(self.clock(), frees=False)  # the count just stated already excludes what has left the hour
            if remaining is not None:
                self._remaining = remaining

    def refused(self, sent_at: float, retry_after: float | None) -> None:
        """The API answered 429. The refused request spent nothing. Where the API said when to retry, nothing goes before
        then and the next request reads the count afresh; otherwise nothing remains until the hour frees a slot."""
        with self._lock:
            if sent_at in self._sent:
                self._sent.remove(sent_at)
            if retry_after is not None:
                self._resume = max(self._resume or 0.0, self.clock() + retry_after)
                self._remaining = None
            else:
                self._remaining = 0


def _stated(response: httpx.Response, header: str) -> int | None:
    value = response.headers.get(header)
    return int(value) if value is not None and value.strip().isdigit() else None


def _remaining(response: httpx.Response) -> int | None:
    return _stated(response, "x-ratelimit-remaining")


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return max(float(value), 0.0) if value is not None else None
    except ValueError:
        return None


def _backoff(attempt: int) -> float:
    return float(min(2**attempt, 60))


def _fetched(where: str, recorded: cassettes.Recorded) -> Fetched:
    return fetched(SOURCE, where, recorded)


@dataclass
class RegulationsGov:
    """The client. `transport` is httpx's (tests serve recorded responses through one); `key` returns the API key and is
    called only inside a live request; `root` is the cassette directory (default: the repository's)."""

    transport: httpx.BaseTransport | None = None
    key: Callable[[], str] = field(default=lambda: secret("GOV_API_KEY"))
    root: Path | None = None
    pacer: Pacer = field(default_factory=Pacer)
    budget: cassettes.Budget = field(default_factory=free_budget)

    def __post_init__(self) -> None:
        self._http = httpx.Client(
            transport=self.transport, timeout=TIMEOUT, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )

    # ---------------------------------------------------------------- one request

    def _send(self, url: str, params: Mapping[str, str], *, api: bool, keep: frozenset[int]) -> dict[str, Any]:
        """One live request, paced (API requests), retried while a failure may pass, and kept as the source returned it.
        `keep`: statuses besides 200 that are the source's answer about the resource (a record no longer served) rather
        than a failure of the request."""
        failures = 0
        refusals = 0
        first_refusal: float | None = None
        first_unreached: float | None = None
        unreached = 0
        while True:
            sent_at = self.pacer.acquire() if api else None
            try:
                headers = {"X-Api-Key": self.key()} if api else None
                response = self._http.get(url, params=dict(params), headers=headers)
            except httpx.TransportError as e:
                if api:
                    self.pacer.settle(None)
                now = self.pacer.clock()
                first_unreached = now if first_unreached is None else first_unreached
                if now - first_unreached > NETWORK_WAIT_S:
                    raise RegulationsGovError(f"{url}: no response for {now - first_unreached:.0f}s: {e!r}") from e
                unreached += 1
                self.pacer.sleep(_backoff(unreached))
                continue
            if api:
                self.pacer.settle(_remaining(response), _stated(response, "x-ratelimit-limit"))
            status = response.status_code
            if status == 200 or status in keep:
                return kept(response, KEPT_HEADERS)
            if status == 429:
                now = self.pacer.clock()
                first_refusal = now if first_refusal is None else first_refusal
                if now - first_refusal > LIMIT_WAIT_S:
                    raise RegulationsGovError(f"{url}: still refused for its rate limit after {now - first_refusal:.0f}s")
                refusals += 1
                if sent_at is not None:  # the pacer waits it out before this request goes again
                    self.pacer.refused(sent_at, _retry_after(response))
                else:
                    wait = _retry_after(response)
                    self.pacer.sleep(_backoff(refusals) if wait is None else wait)
                continue
            if status in TRANSIENT:
                failures += 1
                if failures > RETRIES:
                    raise RegulationsGovError(f"{url}: HTTP {status} after {RETRIES} retries")
                self.pacer.sleep(_backoff(failures))
                continue
            raise RegulationsGovError(f"{url}: HTTP {status}: {response.text[:300]}")

    def api(
        self, path: str, params: Mapping[str, str | int], *, keep: frozenset[int] = frozenset(), attempt: int = 1
    ) -> Fetched:
        """GET an API path (relative to /v4/), through the cassettes, keyed by the path and the query; a listing's later
        pass (`attempt` > 1) is another request of the same query, so its key names the pass."""
        query = {k: str(v) for k, v in sorted(params.items())}
        recorded = cassettes.call(
            PROVIDER,
            f"v4/{path}",
            query if attempt == 1 else {"query": query, "pass": attempt},
            lambda: (self._send(API + path, query, api=True, keep=keep), 0.0),
            budget=self.budget,
            estimate_usd=0.0,
            root=self.root,
        )
        return _fetched(locator(path, query), recorded)

    def download(self, url: str) -> Fetched:
        """A file the API links to (an attachment), through the cassettes. The file host is no api.data.gov service: no
        key is sent and no hourly limit applies. A file the host no longer serves (403, 404, 410) is its answer, kept."""
        parts = urlsplit(url)
        if parts.scheme != "https" or not parts.netloc:
            raise RegulationsGovError(f"not an https file URL: {url!r}")
        recorded = cassettes.call(
            PROVIDER,
            url,
            {},
            lambda: (self._send(url, {}, api=False, keep=frozenset({403, 404, 410})), 0.0),
            budget=self.budget,
            estimate_usd=0.0,
            root=self.root,
        )
        return _fetched(url, recorded)

    # ---------------------------------------------------------------- the endpoints

    def listing(self, path: str, filters: Mapping[str, str]) -> Listing:
        """Every resource a search of `path` ("documents" or "comments") returns under `filters`, page after page and
        continued past the API's 20 pages (module docstring).

        Pages are offsets into a record that can change while it is read. A new resource sorts last and shifts nothing
        already read; but a resource modified after its page was read moves to the end, and the rest move up a place, so
        one resource can pass unread from the next page to the last one read. A pass that sees a resource again with a
        changed lastModifiedDate, or ends with fewer distinct resources than its first page counted, may have missed one
        and is listed again, merged, until a pass is undisturbed (ours)."""
        base = {**filters, "sort": "lastModifiedDate,documentId", "page[size]": str(PAGE_SIZE)}
        pages: list[Fetched] = []
        entries: dict[tuple[str, str, str], Entry] = {}
        for attempt in range(1, PASSES + 1):
            if not self._pass(path, base, attempt, pages, entries):
                return Listing(locator(path, base), tuple(pages), tuple(entries.values()))
        raise RegulationsGovError(f"{locator(path, base)}: the record changed under each of {PASSES} listings of it")

    def _pass(
        self, path: str, base: Mapping[str, str], attempt: int, pages: list[Fetched], entries: dict[tuple[str, str, str], Entry]
    ) -> bool:
        """One pass through a listing, merged into `pages` and `entries`; whether it was disturbed."""
        seen: dict[tuple[str, str, str], Any] = {}
        disturbed = False
        total: int | None = None
        since: str | None = None
        while True:
            query = base if since is None else {**base, "filter[lastModifiedDate][ge]": since}
            for number in range(1, MAX_PAGES + 1):
                page = self.api(path, {**query, "page[number]": str(number)}, attempt=attempt)
                body = page.json()
                data: Sequence[Mapping[str, Any]] = body["data"]
                total = int(body["meta"]["totalElements"]) if total is None else total
                pages.append(page)
                for d in data:
                    rid, modified = identity(d), (d.get("attributes") or {}).get("lastModifiedDate")
                    disturbed |= rid in seen and seen[rid] != modified
                    seen[rid] = modified
                    entries[rid] = Entry(d, len(pages) - 1)  # the latest version, at its first position
                if not body["meta"]["hasNextPage"]:
                    return disturbed or len(seen) < total
                if not data:
                    raise RegulationsGovError(f"{page.locator}: an empty page, yet the API says more follow")
            seam = eastern_seam(str(data[-1]["attributes"]["lastModifiedDate"]))
            if seam == since:
                raise RegulationsGovError(
                    f"{locator(path, base)}: more than {PAGE_SIZE * MAX_PAGES} results share lastModifiedDate "
                    f"{data[-1]['attributes']['lastModifiedDate']}, past which the API's continuation cannot page"
                )
            since = seam

    def documents(self, filters: Mapping[str, str]) -> Listing:
        return self.listing("documents", filters)

    def comments(self, filters: Mapping[str, str]) -> Listing:
        return self.listing("comments", filters)

    def comment(self, comment_id: str) -> Fetched:
        """A comment's record with its attachments; a 404 is kept as the source's answer that it no longer serves it."""
        return self.api(f"comments/{quote(comment_id, safe='')}", {"include": "attachments"}, keep=frozenset({404}))

    def document(self, document_id: str) -> Fetched:
        return self.api(f"documents/{quote(document_id, safe='')}", {"include": "attachments"}, keep=frozenset({404}))

    def docket(self, docket_id: str) -> Fetched:
        return self.api(f"dockets/{quote(docket_id, safe='')}", {}, keep=frozenset({404}))

    def close(self) -> None:
        self._http.close()
