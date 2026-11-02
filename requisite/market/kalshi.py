"""Kalshi's public trade API v2, read only (spec §9.3, §10: "Use public market-data endpoints only; the build holds no
trading credentials"; spec §14: "Market adapters expose no write endpoints"; WP13).

Every request is a GET to one of the read paths in READS, through the cassettes (requisite/cassettes.py) as provider
"kalshi", keyed by the path and the query parameters, so a replay reads exactly what the live run read and never touches
the network. A client may serve an instant (`snapshot`): the record a request returns changes over time (a market
settles, the archive's cutoff advances, the series list grows), so the same request made for another instant is
another request, and the instant enters its key (ours; the regulations.gov client keys a listing's later pass the same
way). No key or credential is read or sent: these endpoints are public. Every request names Requisite by its
User-Agent (decisions, "Public sources are reached through their official APIs").

Pacing. Kalshi meters reads by a token bucket that refills continuously; a request costs 10 tokens and the Basic tier's
read budget is 200 a second (research/sources/kalshi-api-documentation-rate-limits-and-tiers.md), so live requests
start at most 20 a second. A 429 carries no Retry-After (the same source: "Apply exponential backoff on 429"), so it is
waited out with exponential backoff and retried, never dropped; one lasting longer than LIMIT_WAIT_S raises. A dropped
connection or a 5xx is retried RETRIES times, then raises: nothing is skipped.

Listings page by cursor until the response carries none (research/sources/kalshi-api-documentation-understanding-
pagination.md: "Continue until the cursor is `null`"; the recorded KXCAGOAT response ends with "cursor": ""), 1,000
markets to a page, the API's largest (its OpenAPI: limit "Maximum value is 1000").

Numbers are read exactly: a JSON number (a strike) is read as a Decimal, never a float. A quote is a set of prices in
dollars per contract with its sizes and the time it was retrieved (spec §6.5: "Market quotes are stored as prices: bid,
ask, last, size and timestamp"); nothing here reads a price as a probability (I9).
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import quote as urlquote
from urllib.parse import urlencode, urlsplit

import httpx

from requisite import cassettes
from requisite.evidence.fetched import USER_AGENT, Fetched, fetched, free_budget, kept

PROVIDER = "kalshi"
SOURCE = "Kalshi"
# The OpenAPI specification's production Trade API server (docs.kalshi.com/openapi.yaml, servers[0]).
API = "https://external-api.kalshi.com/trade-api/v2"
# Every path this client requests: Kalshi's public market-data reads, each a GET (spec §9.3; the OpenAPI specification).
READS = (
    "/series",
    "/series/{series_ticker}",
    "/markets",
    "/markets/{ticker}",
    "/historical/markets",
    "/historical/markets/{ticker}",
    "/historical/cutoff",
)
PAGE = 1000  # the API's largest page of markets (OpenAPI MarketLimitQuery: "Maximum value is 1000")
RATE = 20.0  # live requests a second: the Basic read budget, 200 tokens a second, over 10 tokens a request
RETRIES = 6  # attempts after the first, for a failure that may pass (a dropped connection, a 5xx)
LIMIT_WAIT_S = 600.0  # a 429 lasting longer than this is no rate limit a refilling bucket explains: it raises
TRANSIENT = frozenset({500, 502, 503, 504})
TIMEOUT = httpx.Timeout(120.0, connect=20.0)
KEPT_HEADERS = ("content-type", "content-length", "last-modified", "etag")


class KalshiError(RuntimeError):
    """A request Kalshi did not answer after its retries, or answered with an error that is ours to look at. Nothing is
    skipped: the run stops here."""


def instant(value: str) -> datetime:
    """A Kalshi timestamp (RFC 3339, e.g. 2026-09-19T15:41:04.503741Z) as an aware instant."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def body(response: Fetched) -> Any:
    """A JSON response read exactly: its numbers as Decimal, never float."""
    return json.loads(response.content, parse_float=Decimal)


def locator(path: str, params: Mapping[str, str]) -> str:
    query = urlencode(sorted(params.items()))
    return f"{API}{path}" + (f"?{query}" if query else "")


class Pacer:
    """Live requests start at most `rate` a second, shared by a run's threads. `clock` and `sleep` are injectable, so a
    test does not wait in real time. A replay has nothing to pace."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep, *,
                 rate: float = RATE):  # fmt: skip
        self.clock, self.sleep, self.interval = clock, sleep, 1.0 / rate
        self._lock = threading.Lock()
        self._next: float | None = None

    def acquire(self) -> None:
        with self._lock:
            now = self.clock()
            start = now if self._next is None else max(now, self._next)
            self._next = start + self.interval
        if start > now:
            self.sleep(start - now)


def _backoff(attempt: int) -> float:
    return float(min(2**attempt, 60))


@dataclass(frozen=True)
class Listing:
    """Every page of one listing (its path and filters, without a cursor), in order, and the entries they served."""

    request: str
    pages: tuple[Fetched, ...]
    entries: tuple[tuple[Mapping[str, Any], int], ...]  # (entry, index of the page that served it)


@dataclass
class Kalshi:
    """The client. `transport` is httpx's (tests serve recorded or synthetic responses through one); `root` is the
    cassette directory (default: the repository's); `snapshot` the instant the reads serve (module docstring)."""

    transport: httpx.BaseTransport | None = None
    root: Path | None = None
    snapshot: datetime | None = None
    pacer: Pacer = field(default_factory=Pacer)
    budget: cassettes.Budget = field(default_factory=free_budget)

    def __post_init__(self) -> None:
        if self.snapshot is not None and self.snapshot.utcoffset() is None:
            raise ValueError("a snapshot is an instant with its time zone")
        self._http = httpx.Client(
            transport=self.transport, timeout=TIMEOUT, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )

    def close(self) -> None:
        self._http.close()

    # ---------------------------------------------------------------- one request

    def _send(self, url: str, params: Mapping[str, str], *, keep: frozenset[int]) -> dict[str, Any]:
        """One live GET, paced, retried while a failure may pass, and kept as the source returned it. `keep`: statuses
        besides 200 that are the source's answer about the resource (a market no longer in the live set) rather than a
        failure of the request."""
        failures = 0
        refusals = 0
        first_refusal: float | None = None
        while True:
            self.pacer.acquire()
            try:
                response = self._http.get(url, params=dict(params))
            except httpx.TransportError as e:
                failures += 1
                if failures > RETRIES:
                    raise KalshiError(f"{url}: no response after {RETRIES} retries: {e!r}") from e
                self.pacer.sleep(_backoff(failures))
                continue
            status = response.status_code
            if status == 200 or status in keep:
                return kept(response, KEPT_HEADERS)
            if status == 429:
                now = self.pacer.clock()
                first_refusal = now if first_refusal is None else first_refusal
                if now - first_refusal > LIMIT_WAIT_S:
                    raise KalshiError(f"{url}: still refused for its rate limit after {now - first_refusal:.0f}s")
                refusals += 1
                self.pacer.sleep(_backoff(refusals))
                continue
            if status in TRANSIENT:
                failures += 1
                if failures > RETRIES:
                    raise KalshiError(f"{url}: HTTP {status} after {RETRIES} retries")
                self.pacer.sleep(_backoff(failures))
                continue
            raise KalshiError(f"{url}: HTTP {status}: {response.text[:300]}")

    def _key(self, params: Mapping[str, str]) -> dict[str, Any]:
        """A request's cassette body: its query, and the instant it serves where the client serves one, in UTC, so one
        instant written with another offset is the same request."""
        if self.snapshot is None:
            return {"params": dict(params)}
        return {"params": dict(params), "as_of": self.snapshot.astimezone(UTC).isoformat()}

    def get(self, path: str, params: Mapping[str, str | int] | None = None, *, keep: frozenset[int] = frozenset()) -> Fetched:
        """GET an API path (relative to /trade-api/v2), through the cassettes, keyed by the path and the query."""
        query = {k: str(v) for k, v in sorted((params or {}).items())}
        recorded = cassettes.call(
            PROVIDER, path, self._key(query), lambda: (self._send(API + path, query, keep=keep), 0.0),
            budget=self.budget, estimate_usd=0.0, root=self.root,
        )  # fmt: skip
        return fetched(SOURCE, locator(path, query), recorded)

    def download(self, url: str) -> Fetched:
        """A document the API links to (a filed contract, its current terms), through the cassettes; refused unless https.
        A document the host does not serve is an error: a filing that cannot be read is not skipped."""
        parts = urlsplit(url)
        if parts.scheme != "https" or not parts.netloc:
            raise KalshiError(f"not an https document URL: {url!r}")
        recorded = cassettes.call(
            PROVIDER, url, self._key({}), lambda: (self._send(url, {}, keep=frozenset()), 0.0),
            budget=self.budget, estimate_usd=0.0, root=self.root,
        )  # fmt: skip
        return fetched(SOURCE, url, recorded)

    def listing(self, path: str, params: Mapping[str, str | int], key: str) -> Listing:
        """Every page of a listing of `key` entries ("markets", "series"), following the cursor until none is given."""
        base = {k: str(v) for k, v in params.items()}
        pages: list[Fetched] = []
        entries: list[tuple[Mapping[str, Any], int]] = []
        seen: set[str] = set()
        cursor = ""
        while True:
            page = self.get(path, {**base, "cursor": cursor} if cursor else base)
            data = body(page)
            pages.append(page)
            entries.extend((e, len(pages) - 1) for e in data.get(key) or [])
            cursor = str(data.get("cursor") or "")
            if not cursor:
                return Listing(locator(path, base), tuple(pages), tuple(entries))
            if cursor in seen:
                raise KalshiError(f"{page.locator}: the cursor {cursor[:24]}... came back again; the listing would not end")
            seen.add(cursor)

    # ---------------------------------------------------------------- the read endpoints

    def series_list(self) -> Listing:
        """Every series (GET /series). It is served whole today; a cursor, if one is given, is followed."""
        return self.listing("/series", {}, "series")

    def series(self, ticker: str) -> Fetched:
        return self.get(f"/series/{urlquote(ticker, safe='')}")

    def markets(self, **filters: str | int) -> Listing:
        """Markets in the live set (GET /markets) under the API's filters: series_ticker, event_ticker, tickers,
        min_settled_ts, max_settled_ts and the rest, 1,000 to a page."""
        return self.listing("/markets", {"limit": PAGE, **filters}, "markets")

    def market(self, ticker: str) -> Fetched:
        """One market in the live set; a 404 is kept, the source's answer that the market is not (or no longer) there."""
        return self.get(f"/markets/{urlquote(ticker, safe='')}", keep=frozenset({404}))

    def historical_markets(self, **filters: str | int) -> Listing:
        """Markets archived before the cutoff (GET /historical/markets): one filter of series_ticker, event_ticker or
        tickers (the API's filters are mutually exclusive), 1,000 to a page."""
        if len({"series_ticker", "event_ticker", "tickers"} & set(filters)) > 1:
            raise ValueError("the historical markets endpoint takes one filter at a time")
        return self.listing("/historical/markets", {"limit": PAGE, **filters}, "markets")

    def historical_market(self, ticker: str) -> Fetched:
        """One archived market (Kalshi API: Get Historical Market); a 404 is kept, as for the live set."""
        return self.get(f"/historical/markets/{urlquote(ticker, safe='')}", keep=frozenset({404}))

    def historical_cutoff(self) -> Fetched:
        """The archive's cutoffs; markets settled before market_settled_ts are only in the historical endpoints."""
        return self.get("/historical/cutoff")


# ---------------------------------------------------------------- what a market record says

# The fields of a market record that state the terms it was listed under: its identity and titles, its rules text, its
# strike vocabulary, its notional and its times (spec §9.3: "Each market carries rules_primary, rules_secondary,
# early_close_condition and a small strike vocabulary: strike_type, floor_strike, cap_strike, functional_strike and
# custom_strike"; the OpenAPI's Market). Never its outcome (status, result, expiration_value, settlement_ts,
# settlement_value_dollars), against which a reading of its terms is checked, nor its prices and volumes, which say
# nothing of its terms (I9). A field Kalshi adds is outside this list until it is read and placed here.
LISTING_FIELDS = (
    "ticker", "event_ticker", "market_type", "title", "subtitle", "yes_sub_title", "no_sub_title", "rules_primary",
    "rules_secondary", "early_close_condition", "can_close_early", "strike_type", "floor_strike", "cap_strike",
    "functional_strike", "custom_strike", "notional_value_dollars", "open_time", "close_time", "expiration_time",
    "expected_expiration_time", "latest_expiration_time", "occurrence_datetime", "settlement_timer_seconds",
)  # fmt: skip


def listing(market: Mapping[str, Any]) -> dict[str, Any]:
    """A market record's listing fields (LISTING_FIELDS), those it carries."""
    return {k: market[k] for k in LISTING_FIELDS if k in market}



@dataclass(frozen=True)
class Settled:
    """A settled market's outcome as Kalshi recorded it: the known outcome F1's regression compares the evaluator
    against (spec §9.3: "settled markets supplying known outcomes"). result is the OpenAPI's enum ('yes', 'no',
    'scalar', ''); settled_at is settlement_ts ("Timestamp when the market was settled"); expiration_value is "The value
    that was considered for the settlement", as given; settlement_value the YES side's settlement in dollars."""

    result: str
    settled_at: datetime
    expiration_value: str
    settlement_value: Decimal | None


@dataclass(frozen=True)
class Quote:
    """A market's prices as one response served them (spec §6.5; I9): dollars per contract, never probabilities, with
    the sizes in contracts and the time of the retrieval."""

    ticker: str
    yes_bid: Decimal
    yes_ask: Decimal
    no_bid: Decimal
    no_ask: Decimal
    last: Decimal
    yes_bid_size: Decimal
    yes_ask_size: Decimal
    volume: Decimal
    open_interest: Decimal
    at: datetime
    locator: str


def settled_at(market: Mapping[str, Any]) -> datetime | None:
    """When the market settled, as recorded, or None for a market not settled (settlement_ts "Only filled for settled
    markets")."""
    value = market.get("settlement_ts")
    return instant(str(value)) if value else None


def settled(market: Mapping[str, Any]) -> Settled | None:
    """The known outcome of a settled market, or None for one not settled."""
    at = settled_at(market)
    if at is None:
        return None
    value = market.get("settlement_value_dollars")
    return Settled(
        result=str(market.get("result") or ""),
        settled_at=at,
        expiration_value=str(market.get("expiration_value") or ""),
        settlement_value=Decimal(str(value)) if value is not None else None,
    )


def strike_type(market: Mapping[str, Any]) -> str | None:
    """The market's strike_type, or None where it has none (the field is optional: KXCAGOAT's market carries none)."""
    value = market.get("strike_type")
    return str(value) if value else None


def quote(market: Mapping[str, Any], served: Fetched) -> Quote:
    """The prices a market record carries, as served by `served` (the response it came from)."""

    def money(name: str) -> Decimal:
        return Decimal(str(market[name]))

    return Quote(
        ticker=str(market["ticker"]),
        yes_bid=money("yes_bid_dollars"),
        yes_ask=money("yes_ask_dollars"),
        no_bid=money("no_bid_dollars"),
        no_ask=money("no_ask_dollars"),
        last=money("last_price_dollars"),
        yes_bid_size=money("yes_bid_size_fp"),
        yes_ask_size=money("yes_ask_size_fp"),
        volume=money("volume_fp"),
        open_interest=money("open_interest_fp"),
        at=served.retrieved_at,
        locator=served.locator,
    )


def markets_of(listings: Sequence[Listing]) -> list[Mapping[str, Any]]:
    """The markets several listings served, each once by ticker; where two served one market (a market in both the live
    set and the archive while it is handed over), the first listing's copy stands."""
    out: dict[str, Mapping[str, Any]] = {}
    for listing in listings:
        for entry, _ in listing.entries:
            out.setdefault(str(entry["ticker"]), entry)
    return list(out.values())
