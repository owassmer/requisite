"""F1's draw rule (design/decisions.md, F1's draw rule, as built in requisite/market/f1.py) on synthetic series written
here, served through the fake Kalshi: eligibility, one series per strike type, the hash order, the cap, the named series
in by name and apart, the latest settled event, the archive, and a record read before the instant refused.

The hash order is computed here from its definition, sha256("requisite-F1|" + ticker), independently of the product."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from fake_kalshi import FakeKalshi, market_entry, series_entry

from requisite.market import f1
from requisite.market.f1 import Found

AS_OF = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
SINCE = AS_OF - timedelta(days=45)


def by_hash(tickers: list[str]) -> list[str]:
    return sorted(tickers, key=lambda t: hashlib.sha256(f"requisite-F1|{t}".encode()).hexdigest())


def iso(t: datetime) -> str:
    return t.isoformat().replace("+00:00", "Z")


def event(api: FakeKalshi, series: str, name: str, at: datetime | None, *strikes: str | None) -> None:
    """One event of `series` whose markets carry the given strike types (None: no strike_type), settled at `at`."""
    for i, s in enumerate(strikes):
        api.add_market(series, market_entry(f"{series}-{name}-M{i}", f"{series}-{name}", settled=iso(at) if at else None,
                                            strike=s))  # fmt: skip


def run(api: FakeKalshi, make_client, tmp, *, named: str = f1.NAMED, as_of: datetime = AS_OF, **kw) -> f1.Draw:
    c = make_client(api, tmp / "cassettes", tmp, snapshot=as_of)
    return f1.draw(c, as_of, named=named, **kw)


@pytest.fixture
def scene():
    """Twelve series in hash order, each built to exercise one clause."""
    t = by_hash([f"KXS{i:02d}" for i in range(12)])
    api = FakeKalshi()
    for ticker in t:
        api.add_series(series_entry(ticker, terms=ticker != t[1]))
    event(api, t[0], "E", AS_OF - timedelta(days=1), "functional")  # the named series: first in hash order, typed
    event(api, t[1], "E", AS_OF - timedelta(days=1), "greater")  # no filed terms: not in the frame
    event(api, t[2], "E", None, "greater")  # nothing settled
    event(api, t[3], "E", AS_OF + timedelta(milliseconds=500), "greater")  # settled just after the instant
    event(api, t[4], "E", SINCE - timedelta(milliseconds=500), "greater")  # settled just before the lookback
    event(api, t[5], "E", AS_OF, "greater")  # settled exactly at the instant
    event(api, t[6], "E", AS_OF - timedelta(days=2), "greater")  # a second greater series, later in hash order
    event(api, t[7], "E", AS_OF - timedelta(days=3), "between", "between", "less", None)  # two types in one event
    event(api, t[8], "OLD", AS_OF - timedelta(days=9), "custom")  # an earlier event's type does not count
    event(api, t[8], "NEW", AS_OF - timedelta(days=4), "structured")
    event(api, t[9], "E", AS_OF - timedelta(days=1), "greater_or_equal")  # past the cap: never read
    for ticker in t[10:]:
        event(api, ticker, "E", AS_OF - timedelta(days=1), "less_or_equal")
    return api, t


def test_the_rule_clause_by_clause(scene, live, make_client):
    api, t = scene
    d = run(api, make_client, live, named=t[0])
    walked = {w.ticker: w for w in d.walked}
    # the frame: series with filed terms, the named one apart, in hash order
    assert d.frame == 10 and d.without_terms == 1 and d.named_in_frame
    assert [w.ticker for w in d.walked] == t[2:9]
    assert [w.hash for w in d.walked] == sorted(w.hash for w in d.walked)
    assert t[0] not in walked and t[1] not in walked
    # eligibility: a market settled in the lookback, by the instant inclusive
    assert not walked[t[2]].found.eligible
    assert not walked[t[3]].found.eligible
    assert not walked[t[4]].found.eligible
    assert walked[t[5]].found.eligible and walked[t[5]].found.latest_at == AS_OF
    # one series per strike type: the first eligible in hash order; a series is eligible under each of its types
    assert walked[t[5]].drawn_for == ("greater",)
    assert walked[t[6]].found.eligible and walked[t[6]].drawn_for == ()
    assert walked[t[7]].found.types == ("between", "less") and walked[t[7]].drawn_for == ("between", "less")
    # the latest settled event's types, not an earlier event's
    assert walked[t[8]].found.event == f"{t[8]}-NEW" and walked[t[8]].found.types == ("structured",)
    # the cap: the walk stops at the fourth type; nothing after it is read
    assert not d.exhausted and [w.ticker for w in d.drawn] == [t[5], t[7], t[8]]
    assert dict(d.types) == {"between": (t[7],), "greater": (t[5], t[6]), "less": (t[7],), "structured": (t[8],)}
    assert not any(t[9] in p for r in api.requests for p in [str(r.url)])
    # the named series: in by name, apart: its type is no type of the draw's
    assert "functional" not in d.types and d.named == t[0]
    text = f1.render(d)
    assert f"named: {t[0]} (Western Grazers), in by name" in text and "greater: " + f"{t[5]}, {t[6]}" in text


def test_the_named_series_is_kxcagoat_unless_said_otherwise(scene, live, make_client):
    api, t = scene
    api.kxcagoat()
    d = run(api, make_client, live)
    assert d.named == "KXCAGOAT" and d.named_in_frame
    assert "KXCAGOAT" not in [w.ticker for w in d.walked]
    assert d.walked[0].ticker == t[0]  # the series that was named in the other test is drawn here
    assert d.walked[0].drawn_for == ("functional",)


def test_a_frame_with_fewer_types_than_the_cap_is_walked_whole(live, make_client):
    api = FakeKalshi()
    t = by_hash(["KXA", "KXB", "KXC"])
    for ticker in t:
        api.add_series(series_entry(ticker))
    event(api, t[0], "E", AS_OF - timedelta(hours=5), "greater")
    event(api, t[1], "E", AS_OF - timedelta(hours=5), None)  # eligible, but no strike type
    event(api, t[2], "E", AS_OF - timedelta(hours=5), "between")
    d = run(api, make_client, live)
    assert d.exhausted and [w.ticker for w in d.walked] == t
    assert [w.ticker for w in d.drawn] == [t[0], t[2]]
    assert d.walked[1].found.eligible and d.walked[1].found.types == ()


def test_a_tie_in_settlement_goes_to_the_smaller_event_ticker(live, make_client):
    api = FakeKalshi()
    api.add_series(series_entry("KXT"))
    event(api, "KXT", "B", AS_OF - timedelta(hours=1), "less")
    event(api, "KXT", "A", AS_OF - timedelta(hours=1), "greater")
    d = run(api, make_client, live)
    assert d.walked[0].found.event == "KXT-A" and d.walked[0].found.types == ("greater",)


def test_a_dense_series_is_characterized_from_a_bounded_read(live, make_client):
    """A series settling forty markets an hour for forty days: the latest event is found from windows ending at the
    instant, and the reads hold about one event's markets, never the series' 38,400."""
    api = FakeKalshi()
    api.add_series(series_entry("KXDENSE"))
    start = AS_OF - timedelta(days=40, minutes=30)
    for h in range(40 * 24):
        at = start + timedelta(hours=h)
        for k in range(40):
            api.add_market("KXDENSE", market_entry(f"KXDENSE-H{h:04d}-T{k:02d}", f"KXDENSE-H{h:04d}", settled=iso(at),
                                                   strike="greater" if k else "less"))  # fmt: skip
    d = run(api, make_client, live)
    found = d.walked[0].found
    assert found.event == "KXDENSE-H0959" and found.latest_at == start + timedelta(hours=959)
    assert found.types == ("greater", "less") and len(found.settled) == 40
    served = sum(len(f.json().get("markets", [])) for f in found.responses)
    assert served <= 3 * 40 + 1


def test_the_archive_is_read_where_the_lookback_reaches_before_its_cutoff(live, make_client):
    api = FakeKalshi(cutoff=iso(AS_OF - timedelta(days=20)))
    t = by_hash(["KXOLD", "KXOLDER", "KXLIVE"])
    for ticker in t:
        api.add_series(series_entry(ticker))
    event(api, "KXOLD", "E", AS_OF - timedelta(days=30), "between")  # archived, inside the lookback
    event(api, "KXOLDER", "E", AS_OF - timedelta(days=50), "greater")  # archived, before the lookback
    event(api, "KXLIVE", "E", AS_OF - timedelta(days=2), "less")
    d = run(api, make_client, live)
    walked = {w.ticker: w.found for w in d.walked}
    assert walked["KXOLD"].eligible and walked["KXOLD"].types == ("between",)
    assert walked["KXOLD"].event_markets == (("KXOLD-E-M0", "historical"),)
    assert not walked["KXOLDER"].eligible
    assert walked["KXLIVE"].eligible and "/historical/markets" in api.paths()


def test_a_record_read_before_the_instant_is_refused(scene, live, make_client, monkeypatch):
    api, t = scene
    later = datetime.now(UTC) + timedelta(days=1)
    with pytest.raises(ValueError, match="still to come"):
        run(api, make_client, live, named=t[0], as_of=later)
    assert api.requests == []  # refused before anything was read
    # a recording made before the instant, replayed once the instant has passed, is refused too
    soon = datetime.now(UTC) + timedelta(seconds=30)
    monkeypatch.setattr(f1, "datetime", _Clock(soon + timedelta(seconds=1)))
    with pytest.raises(ValueError, match="before the draw's instant"):
        run(api, make_client, live, named=t[0], as_of=soon)


class _Clock:
    """datetime as f1 sees it, with now() fixed: the time a replay runs, after a recording made earlier."""

    def __init__(self, now: datetime):
        self._now = now

    def now(self, tz=None) -> datetime:
        return self._now


def test_the_client_serves_the_draws_instant(scene, live, make_client):
    api, _ = scene
    c = make_client(api, live / "cassettes", live, snapshot=AS_OF - timedelta(days=1))
    with pytest.raises(ValueError, match="not the draw's instant"):
        f1.draw(c, AS_OF)


def test_the_draw_replays_from_its_cassettes_alone(scene, live, make_client, monkeypatch):
    api, t = scene
    first = run(api, make_client, live, named=t[0])
    monkeypatch.setenv("REQUISITE_REPLAY_ONLY", "1")
    again = f1.draw(make_client(None, live / "cassettes", live, snapshot=AS_OF), AS_OF, named=t[0])
    assert f1.render(again) == f1.render(first)
    assert [r.content for r in again.responses] == [r.content for r in first.responses]


# ---------------------------------------------------------------- the rule alone, over what was found


def _found(*types: str, eligible: bool = True) -> Found:
    return Found(eligible=eligible, reason="test", types=types)


def test_walk_draws_each_type_once_in_order_and_stops_at_the_cap():
    found = {"a": _found("x"), "b": _found("x", "y"), "c": _found(eligible=False), "d": _found("z", "w", "v"),
             "e": _found("u")}  # fmt: skip
    order = [(k, k) for k in "abcde"]
    walked, carrying, exhausted = f1.walk(order, found.__getitem__, cap=4)
    assert [w.drawn_for for w in walked] == [("x",), ("y",), (), ("z", "w", "v")]
    assert not exhausted and carrying == {"x": ("a", "b"), "y": ("b",), "z": ("d",), "w": ("d",), "v": ("d",)}
    assert sum(bool(w.drawn_for) for w in walked) <= 4  # at most CAP series, though one may bring several types


def test_walk_ignores_the_types_of_an_ineligible_series():
    walked, carrying, exhausted = f1.walk([("a", "a")], {"a": _found("x", eligible=False)}.__getitem__)
    assert walked[0].drawn_for == () and carrying == {} and exhausted
