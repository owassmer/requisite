"""F1's filed contracts (spec §9.3, §12 F1): the draw over Kalshi's filed record at a pinned instant, and the harvest of
what F1 reads into a case's store.

The rule (design/decisions.md, "F1's filed contracts are drawn by a rule over the filed record, fixed here before the
draw runs"), as built. The Western Grazers contract, Kalshi series KXCAGOAT, is in by name (spec §9.3 names it as the
reference specimen); it is apart from the draw: it counts toward neither the cap nor any type, and the walk passes over
it. The others are drawn from the frame, Kalshi's series that carry filed contract terms (a non-empty
contract_terms_url), ordered by sha256("requisite-F1|" + ticker), which makes the choice independent of anyone's view of
which contracts are interesting. A series is eligible when one of its markets settled in the LOOKBACK before the
instant; its strike types are those of the markets of its latest settled event, the event of its market settled last
by the instant (a tie in settlement time goes to the smaller event ticker), counting that event's markets settled by
the instant; a series is eligible under each of its types. Each strike type's draw is the first eligible series in hash
order that carries it, and the walk through the frame stops once CAP types are drawn, so at most CAP series are drawn.
A market counts as settled by the instant when its recorded settlement_ts is at or before it, decided by
core.observable_at over the listing that served it, as a retrospective reconstruction: the record is read after the
instant and every response the draw rests on was retrieved at or after it (a record read before the instant cannot
show what settled by it, and such a draw is refused).

Where the rule as built departs from decisions.md's wording, and why (the worker on task 001-2, November 1; measured
that day): the record is too large for the wording's scope to be read. All 15,589 series in Kalshi's series list carry
contract_terms_url; one series alone (an hourly one) settled about 1,000 markets in six hours, and the archive held
about 1,000 combination markets settled in forty minutes, so the settled markets since 2021 number in the millions and
their pages in the gigabytes, which no run can read or cassette. Kalshi gives no count and no strike filter, and its
archive has no time filter (its OpenAPI specification). So:
- the cap is on the first CAP types reached in hash order, not on "the four with the most settled markets", which only
  a count of every settled market could decide; the walk reads only the frame's prefix up to the series that brings
  the CAP-th type, and the evidence lists each type's eligible series in hash order within that prefix;
- a series' strike types are its latest settled event's, not those of all its markets settled by the instant, which
  for a large series is millions of markets; the latest event is found by reading windows of its settled markets ending
  at the instant, doubling from FIRST_PROBE, so each read is bounded by about one event's markets;
- eligibility asks a market settled in the LOOKBACK before the instant, not at any time before it, so that no read of
  the archive (which has no time filter, so can be read only whole) is needed while the lookback ends after the
  archive's cutoff, and so that the settled outcomes F1 compares were settled recently, under terms most likely to be
  the current ones contract_terms_url links to. LOOKBACK is 45 days: longer than a month, so a monthly series that
  settles on time is eligible, and shorter than the archive's lag (its cutoff was 2026-09-02 when read on 2026-11-01).
  Where the lookback does reach before the cutoff, the series' archive is read whole, which is exact and only slower.
The per-type choice is the wording's: the eligible series with the smallest hash.

The harvest. For each harvested series (the named one and each drawn one): its series record, its filed contract
(contract_url) and current terms (contract_terms_url) as the bytes Kalshi serves, and each of its markets' record (Get
Market, or Get Historical Market for an archived one: spec §9.3), which carries the market's rules text (rules_primary,
rules_secondary), its strike, and, for a market settled by the instant, its known outcome (result, settlement_ts,
expiration_value). A drawn series' markets are those of the event that put it in the draw; the named series' are all of
its markets. Each harvested series is committed as a core.FiledSeries naming those artifacts and the draw's instant, so
a later step reads the harvest, and which markets settled by the instant, from committed state alone. Every response
the draw rests on is committed too, so the draw's evidence is frozen beside what it chose. Text extraction of the
filings is the evidence package's (requisite/evidence/extract.py); here they are committed as retrieved.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from requisite.core import Artifact, FiledSeries, observable_at
from requisite.evidence.artifacts import LISTING_BASIS, commit, describe, listed
from requisite.evidence.fetched import Fetched
from requisite.market.kalshi import (
    Kalshi,
    KalshiError,
    Listing,
    Settled,
    body,
    instant,
    settled,
    settled_at,
    strike_type,
)
from requisite.store import Refused, Store

NAMED = "KXCAGOAT"  # decisions, F1's draw rule: "The Western Grazers contract is in by name"
SALT = "requisite-F1|"  # decisions: sha256("requisite-F1|" + ticker)
CAP = 4  # decisions: at most four strike types drawn; spec §12 asks for 2-5 contracts with the named one
LOOKBACK = timedelta(days=45)  # module docstring
# The instant F1's draw is pinned at, fixed before the live draw ran (decisions, F1's draw rule: "the first midnight UTC
# after the rule was fixed"). F1 is this draw; another instant is another draw, whose responses a replay does not hold.
INSTANT = datetime(2026, 11, 2, tzinfo=UTC)
FIRST_PROBE = timedelta(seconds=60)  # the first window of the latest-event search; each next one doubles it
MODE = "retrospective_reconstruction"  # the record is read after the instant (module docstring; addendum A11 §11.3)
SETTLED = "the market's settlement_ts in the listing: when its result was determined"
RECORD_BASIS = "none stated: a series record has no publication date of its own; its retrieval evidences it"
FILING_BASIS = "none stated by the API: a filing's own dates are in its text; its retrieval evidences it"
MARKET_SETTLED = "the market's settlement_ts: when its result was determined"
MARKET_OPEN = "none stated: the market had not settled when it was retrieved; its retrieval evidences it"


def rank(ticker: str) -> str:
    """A series' place in the draw's order (decisions: sha256("requisite-F1|" + ticker))."""
    return hashlib.sha256(f"{SALT}{ticker}".encode()).hexdigest()


@dataclass(frozen=True)
class Found:
    """What the walk knows of one series at the instant: whether it is eligible and why, its latest settled event and
    that event's markets (each with the endpoint that served it: "live" or "historical"), which of them settled by the
    instant, and the strike types of those."""

    eligible: bool
    reason: str
    event: str | None = None
    latest: str | None = None
    latest_at: datetime | None = None
    event_markets: tuple[tuple[str, str], ...] = ()
    settled: tuple[str, ...] = ()
    types: tuple[str, ...] = ()
    responses: tuple[Fetched, ...] = ()


@dataclass(frozen=True)
class Walked:
    position: int
    ticker: str
    hash: str
    found: Found
    drawn_for: tuple[str, ...]


@dataclass(frozen=True)
class Draw:
    """The draw at an instant, with the rule's evidence: the frame's size, the named series apart, every series walked
    in hash order with what was found of it, each drawn type's eligible series in hash order within the walk (the first
    is the draw), and every response the draw rests on."""

    as_of: datetime
    lookback: timedelta
    cap: int
    named: str
    named_in_frame: bool
    frame: int
    without_terms: int
    cutoff: datetime | None
    walked: tuple[Walked, ...]
    types: Mapping[str, tuple[str, ...]]
    exhausted: bool
    responses: tuple[Fetched, ...]

    @property
    def drawn(self) -> tuple[Walked, ...]:
        return tuple(w for w in self.walked if w.drawn_for)


def walk(order: Sequence[tuple[str, str]], find: Callable[[str], Found], cap: int = CAP
         ) -> tuple[tuple[Walked, ...], dict[str, tuple[str, ...]], bool]:  # fmt: skip
    """The rule over a frame in hash order: each strike type's draw is the first eligible series that carries it; the
    walk stops once `cap` types are drawn. Returns the series walked, each type's eligible series in walk order, and
    whether the frame was exhausted before the cap was reached."""
    drawn: dict[str, str] = {}
    carrying: dict[str, list[str]] = {}
    walked: list[Walked] = []
    for position, (ticker, h) in enumerate(order, 1):
        found = find(ticker)
        types = found.types if found.eligible else ()
        for t in types:
            carrying.setdefault(t, []).append(ticker)
        new = tuple(t for t in types if t not in drawn)
        drawn.update(dict.fromkeys(new, ticker))
        walked.append(Walked(position, ticker, h, found, new))
        if len(drawn) >= cap:
            return tuple(walked), {t: tuple(v) for t, v in carrying.items()}, False
    return tuple(walked), {t: tuple(v) for t, v in carrying.items()}, True


# ---------------------------------------------------------------- reading the record at the instant


def _below(t: datetime) -> int:
    """A Unix second strictly before t, so a filter whose bound may be inclusive or not still admits everything after t
    (the API does not say which; each market is then decided by its recorded settlement_ts)."""
    return math.floor(t.timestamp()) - 1


def _above(t: datetime) -> int:
    return math.floor(t.timestamp()) + 1


def _settled_by(listing: Listing, as_of: datetime, after: datetime | None = None) -> list[Mapping[str, Any]]:
    """The markets a listing served that settled by as_of (and after `after`, where given): each judged by
    core.observable_at, its settlement_ts as its publication in the listing page that served it."""
    pages = [describe(p, published_at=None, basis=LISTING_BASIS) for p in listing.pages]
    out = []
    for market, page in listing.entries:
        at = settled_at(market)
        if at is None or (after is not None and not at > after):
            continue
        if observable_at(listed(pages[page], at, SETTLED), as_of, listing.pages[page].retrieved_at, MODE):
            out.append(market)
    return out


def find(client: Kalshi, ticker: str, as_of: datetime, lookback: timedelta, cutoff: datetime | None) -> Found:
    """One series at the instant (module docstring): eligible when a market settled in (as_of - lookback, as_of];
    characterized by the markets of its latest settled event."""
    since = as_of - lookback
    responses: list[Fetched] = []
    window: dict[str, str | int] = {"series_ticker": ticker, "max_settled_ts": _above(as_of)}
    probe = client.get("/markets", {**window, "min_settled_ts": _below(since), "limit": 1})
    responses.append(probe)
    found: list[Mapping[str, Any]] = []
    if body(probe).get("markets"):
        width = FIRST_PROBE
        while not found:
            low = max(as_of - width, since)
            listing = client.markets(**window, min_settled_ts=_below(low))
            responses.extend(listing.pages)
            found = _settled_by(listing, as_of, after=low)
            if low == since:
                break
            width *= 2
    if not found and cutoff is not None and cutoff > since:
        archived = client.historical_markets(series_ticker=ticker)
        responses.extend(archived.pages)
        found = _settled_by(archived, as_of, after=since)
    if not found:
        return Found(False, f"no market settled in ({since.isoformat()}, {as_of.isoformat()}]", responses=tuple(responses))
    stamped = [(at, m) for m in found if (at := settled_at(m)) is not None]
    last = max(at for at, _ in stamped)
    event = min(str(m["event_ticker"]) for at, m in stamped if at == last)
    latest = min(str(m["ticker"]) for at, m in stamped if at == last and str(m["event_ticker"]) == event)
    live = client.markets(event_ticker=event)
    archived_event = client.historical_markets(event_ticker=event)
    responses.extend(live.pages + archived_event.pages)
    served: dict[str, str] = {}
    for listing, where in ((live, "live"), (archived_event, "historical")):
        for market, _ in listing.entries:
            served.setdefault(str(market["ticker"]), where)
    settled_markets = {str(m["ticker"]): m for m in _settled_by(live, as_of) + _settled_by(archived_event, as_of)}
    if latest not in settled_markets:
        raise KalshiError(f"{ticker}: market {latest} settled by {as_of.isoformat()} is missing from its event {event}'s "
                          "listings; the record changed under the read")  # fmt: skip
    types = sorted({t for m in settled_markets.values() if (t := strike_type(m)) is not None})
    return Found(
        eligible=True,
        reason=f"latest settled event {event}: market {latest} settled {last.isoformat()}",
        event=event,
        latest=latest,
        latest_at=last,
        event_markets=tuple(sorted(served.items())),
        settled=tuple(sorted(settled_markets)),
        types=tuple(types),
        responses=tuple(responses),
    )


def draw(client: Kalshi, as_of: datetime, *, named: str = NAMED, lookback: timedelta = LOOKBACK, cap: int = CAP) -> Draw:
    """The draw at `as_of` over the record the client reads (module docstring). The client serves `as_of`, so its
    requests are keyed by it and a replay of this draw reads exactly what its live run read."""
    if as_of.utcoffset() is None:
        raise ValueError("the draw's instant needs its time zone")
    if client.snapshot != as_of:
        raise ValueError(f"the client serves {client.snapshot}, not the draw's instant {as_of.isoformat()}")
    if as_of > datetime.now(UTC):
        raise ValueError(f"the draw's instant {as_of.isoformat()} is still to come: a record read before it cannot show "
                         "what settled by it")  # fmt: skip
    series = client.series_list()
    cutoff_read = client.historical_cutoff()
    cutoff_value = body(cutoff_read).get("market_settled_ts")
    cutoff = instant(str(cutoff_value)) if cutoff_value else None
    entries: dict[str, Mapping[str, Any]] = {}
    for entry, _ in series.entries:
        entries.setdefault(str(entry["ticker"]), entry)
    with_terms = [t for t, e in entries.items() if e.get("contract_terms_url")]
    order = sorted(((t, rank(t)) for t in with_terms if t != named), key=lambda x: x[1])
    walked, carrying, exhausted = walk(order, lambda t: find(client, t, as_of, lookback, cutoff), cap)
    responses = (*series.pages, cutoff_read, *(f for w in walked for f in w.found.responses))
    early = [f for f in responses if f.retrieved_at < as_of]
    if early:
        raise ValueError(f"{early[0].locator} was retrieved {early[0].retrieved_at.isoformat()}, before the draw's "
                         f"instant {as_of.isoformat()}: a record read then cannot show what settled by the instant")  # fmt: skip
    drawn_types = {t for w in walked for t in w.drawn_for}
    return Draw(
        as_of=as_of,
        lookback=lookback,
        cap=cap,
        named=named,
        named_in_frame=named in with_terms,
        frame=len(order),
        without_terms=len(entries) - len(with_terms),
        cutoff=cutoff,
        walked=walked,
        types={t: carrying[t] for t in sorted(drawn_types)},
        exhausted=exhausted,
        responses=responses,
    )


def render(d: Draw) -> str:
    """The draw with the rule's evidence, as the CLI prints it."""
    out = [
        f"F1 draw at {d.as_of.isoformat()} (design/decisions.md, F1's draw rule, as built in requisite/market/f1.py)",
        f"frame: {d.frame} Kalshi series with filed contract terms, ordered by sha256(\"{SALT}\" + ticker); "
        f"{d.without_terms} listed series carry no contract_terms_url",
        f"eligible: a market settled in the {d.lookback.days} days before the instant; strike types: those of the "
        "latest settled event's markets settled by the instant",
        f"archive cutoff as read: {d.cutoff.isoformat() if d.cutoff else 'none given'}",
        f"named: {d.named} (Western Grazers), in by name as spec §9.3's reference specimen, apart from the draw: it counts "
        f"toward neither the cap nor any type{'' if d.named_in_frame else ' (not in the series list with filed terms)'}",
        "",
        "walk (hash order):",
    ]
    for w in d.walked:
        f = w.found
        if not f.eligible:
            out.append(f"  {w.position:>5}  {w.ticker}  {w.hash[:12]}  not eligible: {f.reason}")
            continue
        types = ", ".join(f.types) or "none (its markets carry no strike_type)"
        drawn = f"  -> drawn for {', '.join(w.drawn_for)}" if w.drawn_for else ""
        out.append(f"  {w.position:>5}  {w.ticker}  {w.hash[:12]}  eligible: {f.reason}; "
                   f"{len(f.settled)} market(s) settled by the instant; types: {types}{drawn}")  # fmt: skip
    out += ["", "each drawn strike type's eligible series in hash order (the first is drawn):"]
    out += [f"  {t}: {', '.join(series)}" for t, series in d.types.items()]
    out += ["", "drawn:"]
    out += [f"  {w.ticker}  for {', '.join(w.drawn_for)}  (event {w.found.event})" for w in d.drawn]
    if d.exhausted:
        out.append(f"the frame was exhausted after {len(d.walked)} series with {len(d.types)} strike type(s) drawn")
    else:
        out.append(f"stopped at position {len(d.walked)} of {d.frame}: {len(d.types)} strike types drawn (cap {d.cap})")
    out.append(f"{len(d.responses)} responses read")
    return "\n".join(out)


# ---------------------------------------------------------------- the harvest


@dataclass(frozen=True)
class HarvestedMarket:
    ticker: str
    event: str
    strike_type: str | None
    artifact: Artifact
    outcome: Settled | None  # its known outcome where it settled by the instant


@dataclass(frozen=True)
class HarvestedSeries:
    ticker: str
    why: str
    record: Artifact
    contract: Artifact | None
    terms: Artifact
    markets: tuple[HarvestedMarket, ...]


@dataclass(frozen=True)
class Harvest:
    draw: Draw
    series: tuple[HarvestedSeries, ...]
    evidence: tuple[Artifact, ...]


def _filing(client: Kalshi, store: Store, url: str) -> Artifact:
    doc = client.download(url)
    if not doc.content.startswith(b"%PDF-"):
        raise KalshiError(f"{url}: served {doc.media_type}, not the PDF a filing is")
    return commit(store, doc, published_at=None, basis=FILING_BASIS)


def _market(client: Kalshi, store: Store, ticker: str, where: str, as_of: datetime) -> HarvestedMarket:
    """One market's record (Get Market, or Get Historical Market for an archived one), committed, with its known outcome
    where it settled by the instant."""
    response = client.market(ticker) if where == "live" else client.historical_market(ticker)
    if response.status == 404 and where == "live":  # handed to the archive since its listing was read
        response = client.historical_market(ticker)
    if response.status != 200:
        raise KalshiError(f"{response.locator}: HTTP {response.status}; the market its listing served is not served")
    market = body(response)["market"]
    at = settled_at(market)
    artifact = commit(store, response, published_at=at, basis=MARKET_SETTLED if at else MARKET_OPEN)
    by_instant = at is not None and observable_at(artifact, as_of, response.retrieved_at, MODE)
    return HarvestedMarket(ticker, str(market["event_ticker"]), strike_type(market), artifact,
                           settled(market) if by_instant else None)  # fmt: skip


def _series(client: Kalshi, store: Store, ticker: str, why: str, markets: Sequence[tuple[str, str]], as_of: datetime
            ) -> HarvestedSeries:  # fmt: skip
    response = client.series(ticker)
    record = commit(store, response, published_at=None, basis=RECORD_BASIS)
    series = body(response)["series"]
    terms_url = series.get("contract_terms_url")
    if not terms_url:
        raise KalshiError(f"{ticker}: its series record carries no contract_terms_url; F1 reads filed terms")
    contract_url = series.get("contract_url")
    harvested = HarvestedSeries(
        ticker=ticker,
        why=why,
        record=record,
        contract=_filing(client, store, str(contract_url)) if contract_url else None,
        terms=_filing(client, store, str(terms_url)),
        markets=tuple(_market(client, store, t, where, as_of) for t, where in markets),
    )
    filed = FiledSeries(series=ticker, as_of=as_of, why=why, record=record.id,
                        contract=harvested.contract.id if harvested.contract else None, terms=harvested.terms.id,
                        markets={m.ticker: m.artifact.id for m in harvested.markets})  # fmt: skip
    store.commit("filed_series", filed.key, filed)
    return harvested


def filed_series(store: Store, series: str, as_of: datetime | None = None) -> FiledSeries:
    """A series' committed harvest: the one at `as_of`, or, given no instant, its only one."""
    found = [f for f in store.all("filed_series", FiledSeries) if f.series == series]
    if as_of is not None:
        found = [f for f in found if f.as_of == as_of]
    if len(found) != 1:
        at = f" at {as_of.isoformat()}" if as_of is not None else ""
        instants = ", ".join(f.as_of.isoformat() for f in found)
        raise Refused(f"{series} is harvested {len(found)} times{at} in this case" + (f" ({instants}); name the draw's "
                       "instant" if found else "; retrieve it with `market f1-retrieve`"))  # fmt: skip
    return found[0]


def retrieve(client: Kalshi, store: Store, d: Draw) -> Harvest:
    """The harvest of a draw into a case's store (module docstring): the draw's evidence, then the named series and
    each drawn series, with their filings and their markets' records."""
    evidence = tuple(commit(store, f, published_at=None, basis=LISTING_BASIS) for f in d.responses)
    live = client.markets(series_ticker=d.named)
    archived = client.historical_markets(series_ticker=d.named)
    named_markets: dict[str, str] = {}
    for listing, where in ((live, "live"), (archived, "historical")):
        for market, _ in listing.entries:
            named_markets.setdefault(str(market["ticker"]), where)
    for f in (*live.pages, *archived.pages):
        commit(store, f, published_at=None, basis=LISTING_BASIS)
    harvested = [_series(client, store, d.named, "in by name (spec §9.3's reference specimen)",
                         sorted(named_markets.items()), d.as_of)]  # fmt: skip
    for w in d.drawn:
        why = f"drawn for {', '.join(w.drawn_for)} (position {w.position}; event {w.found.event})"
        harvested.append(_series(client, store, w.ticker, why, w.found.event_markets, d.as_of))
    return Harvest(d, tuple(harvested), evidence)


def known_outcome(store: Store, artifact: str, as_of: datetime) -> Settled | None:
    """A committed market record's known outcome at the instant, read back from the store: its result, settlement time,
    expiration value and settlement value where it settled by as_of, else None."""
    a = store.get("artifact", artifact, Artifact)
    if a is None:
        raise KeyError(f"artifact {artifact} is not committed")
    market = json.loads(store.content(artifact))["market"]
    if a.published_at is None or not observable_at(a, as_of, a.observed_at, MODE):
        return None
    return settled(market)


def render_harvest(h: Harvest, store: Store) -> str:
    """What was committed, with each settled market's known outcome as read back from the store."""
    out = [f"F1 harvest at {h.draw.as_of.isoformat()}: {len(h.series)} series; {len(h.evidence)} responses of the draw "
           "committed as its evidence"]  # fmt: skip
    for s in h.series:
        out.append(f"{s.ticker}: {s.why}")
        out.append(f"  series record {s.record.id}")
        out.append(f"  filed contract {s.contract.id} ({s.contract.size} bytes) {s.contract.locator}" if s.contract
                   else "  filed contract: the series record carries no contract_url")  # fmt: skip
        out.append(f"  contract terms {s.terms.id} ({s.terms.size} bytes) {s.terms.locator}")
        for m in s.markets:
            outcome = known_outcome(store, m.artifact.id, h.draw.as_of)
            kind = m.strike_type or "no strike_type"
            if outcome is None:
                out.append(f"  market {m.ticker} ({kind}) {m.artifact.id}: not settled by the instant")
            else:
                value = f"; settlement value ${outcome.settlement_value}" if outcome.settlement_value is not None else ""
                out.append(f"  market {m.ticker} ({kind}) {m.artifact.id}: settled {outcome.settled_at.isoformat()}, "
                           f"result {outcome.result or '(none given)'}, expiration value {outcome.expiration_value!r}"
                           f"{value}")  # fmt: skip
    return "\n".join(out)
