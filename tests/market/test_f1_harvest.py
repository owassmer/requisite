"""F1's harvest against the recorded KXCAGOAT responses (fixtures/kalshi/KXCAGOAT), served through the fake Kalshi:
the named series' filings and market committed as the bytes served, its settled outcome kept and read back from the
store, the drawn series' event markets beside it, the draw's evidence committed, and the CLI's second run of each
command reading only cassettes."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from fake_kalshi import FIXTURES, FakeKalshi, load, market_entry, series_entry

from requisite import cassettes, cli
from requisite.core import Artifact, FiledSeries
from requisite.market import f1, kalshi
from requisite.store import Refused, Store

AS_OF = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
SETTLED = datetime(2026, 9, 19, 15, 41, 4, 503741, tzinfo=UTC)


def test_the_settled_outcome_is_read_from_kxcagoats_market():
    market = load("markets.json")["body"]["markets"][0]
    outcome = kalshi.settled(market)
    assert outcome == kalshi.Settled(result="yes", settled_at=SETTLED, expiration_value="Yes",
                                     settlement_value=Decimal("1.0000"))  # fmt: skip
    assert kalshi.strike_type(market) is None  # its market carries no strike_type
    unsettled = {k: v for k, v in market.items() if k != "settlement_ts"}
    assert kalshi.settled(unsettled) is None


@pytest.fixture
def record() -> FakeKalshi:
    api = FakeKalshi()
    api.kxcagoat()
    api.add_series(series_entry("KXRANGE"))
    for i, (strike, floor, cap) in enumerate([("less", None, "10"), ("between", "10", "20"), ("greater", "20", None)]):
        api.add_market("KXRANGE", market_entry(f"KXRANGE-26OCT19-B{i}", "KXRANGE-26OCT19",
                                               settled="2026-10-19T20:00:00Z", strike=strike, floor=floor, cap=cap))  # fmt: skip
    api.add_market("KXRANGE", market_entry("KXRANGE-26OCT18-B0", "KXRANGE-26OCT18", settled="2026-10-18T20:00:00Z",
                                           strike="less", cap="10"))  # fmt: skip
    return api


def harvest(api, make_client, tmp, store) -> f1.Harvest:
    c = make_client(api, tmp / "cassettes", tmp, snapshot=AS_OF)
    return f1.retrieve(c, store, f1.draw(c, AS_OF))


def test_the_harvest_commits_filings_rules_and_known_outcomes(record, live, make_client, store):
    h = harvest(record, make_client, live, store)
    named, drawn = h.series
    assert named.ticker == "KXCAGOAT" and named.why.startswith("in by name")
    assert store.content(named.contract.id) == (FIXTURES / "contract.pdf").read_bytes()
    assert store.content(named.terms.id) == (FIXTURES / "contract_terms.pdf").read_bytes()
    assert named.terms.locator == "https://assets.kalshi.com/contract_terms/CAGOATWAGE.pdf"
    assert json.loads(store.content(named.record.id))["series"]["ticker"] == "KXCAGOAT"
    (m,) = named.markets
    assert m.ticker == "KXCAGOAT-26SEP30" and m.strike_type is None
    kept = json.loads(store.content(m.artifact.id))["market"]
    assert kept["rules_primary"].startswith("If California authorizes an alternative wage")
    assert kept["rules_secondary"].startswith('A "Qualifying Action" means')
    assert m.artifact.locator.endswith("/markets/KXCAGOAT-26SEP30") and m.artifact.published_at == SETTLED
    # the known outcome, kept and read back from the store
    assert f1.known_outcome(store, m.artifact.id, AS_OF) == kalshi.Settled("yes", SETTLED, "Yes", Decimal("1.0000"))
    assert f1.known_outcome(store, m.artifact.id, SETTLED - timedelta(microseconds=1)) is None
    assert f1.known_outcome(store, m.artifact.id, SETTLED) is not None
    # the drawn series: its latest settled event's markets, each with its strike and outcome
    assert drawn.ticker == "KXRANGE" and drawn.why.startswith("drawn for between, greater, less")
    assert [x.ticker for x in drawn.markets] == [f"KXRANGE-26OCT19-B{i}" for i in range(3)]
    assert [x.strike_type for x in drawn.markets] == ["less", "between", "greater"]
    assert all(x.outcome is not None and x.outcome.result == "yes" for x in drawn.markets)
    floor = json.loads(store.content(drawn.markets[1].artifact.id))["market"]["floor_strike"]
    assert floor == 10.0
    # each harvested series is committed with the draw's instant, so a later step reads the harvest from the store
    kept_named = f1.filed_series(store, "KXCAGOAT")
    assert kept_named == FiledSeries(series="KXCAGOAT", as_of=AS_OF, why=named.why, record=named.record.id,
                                     contract=named.contract.id, terms=named.terms.id,
                                     markets={"KXCAGOAT-26SEP30": m.artifact.id})  # fmt: skip
    assert kept_named.key == "KXCAGOAT|2026-10-20T12:00:00+00:00"
    assert f1.filed_series(store, "KXRANGE", AS_OF).markets == {x.ticker: x.artifact.id for x in drawn.markets}
    with pytest.raises(Refused, match="KXNONE is harvested 0 times in this case"):
        f1.filed_series(store, "KXNONE")
    # every response the draw rests on is committed, as its evidence
    assert all(store.get("artifact", a.id, Artifact) == a for a in h.evidence)
    assert len(h.evidence) == len(h.draw.responses)
    text = f1.render_harvest(h, store)
    assert "market KXCAGOAT-26SEP30 (no strike_type)" in text and "result yes, expiration value 'Yes'" in text


def test_an_archived_market_is_read_from_the_historical_endpoint(record, live, make_client, store):
    record.cutoff = "2026-10-01T00:00:00Z"  # KXCAGOAT's market, settled September 19, is now archived
    h = harvest(record, make_client, live, store)
    (m,) = h.series[0].markets
    assert "/historical/markets/KXCAGOAT-26SEP30" in m.artifact.locator
    assert m.outcome == kalshi.Settled("yes", SETTLED, "Yes", Decimal("1.0000"))


def test_a_filing_that_is_not_a_pdf_is_refused(record, live, make_client, store):
    record.files["https://assets.kalshi.com/contract_terms/KXRANGE.pdf"] = b"<html>not found</html>"
    with pytest.raises(kalshi.KalshiError, match="not the PDF"):
        harvest(record, make_client, live, store)


def test_a_replay_commits_the_same_harvest(record, live, make_client, store, monkeypatch, tmp_path):
    first = harvest(record, make_client, live, store)
    monkeypatch.setenv("REQUISITE_REPLAY_ONLY", "1")
    other = Store(tmp_path / "other", case="f1")
    again = harvest(None, make_client, live, other)
    ids = lambda h: [(s.record.id, s.terms.id, [m.artifact.id for m in s.markets]) for s in h.series]  # noqa: E731
    assert ids(again) == ids(first)
    assert [a.id for a in again.evidence] == [a.id for a in first.evidence]


def test_the_cli_draws_and_retrieves_from_cassettes_alone(record, live, make_client, monkeypatch, capsys, tmp_path):
    """The CLI's commands over cassettes recorded through the fake (as the coordinator's live run records them), with
    replay only: the second run of each reads nothing but cassettes."""
    root = live / "cassettes"
    c = make_client(record, root, live, snapshot=AS_OF)
    f1.retrieve(c, Store(tmp_path / "recording", case="f1"), f1.draw(c, AS_OF))
    monkeypatch.setenv("REQUISITE_REPLAY_ONLY", "1")
    monkeypatch.setattr(cassettes, "CASSETTES", root)
    assert cli.main(["market", "f1-draw", "--as-of", "2026-10-20T12:00:00Z"]) == 0
    out = capsys.readouterr().out
    assert "named: KXCAGOAT (Western Grazers), in by name" in out and "KXRANGE  for between, greater, less" in out
    case = tmp_path / "case"
    assert cli.main(["market", "f1-retrieve", "--case-dir", str(case), "--case", "f1",
                     "--as-of", "2026-10-20T12:00:00Z"]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "KXCAGOAT-26SEP30 (no strike_type)" in out and "settled 2026-09-19T15:41:04.503741+00:00, result yes" in out
    with pytest.raises(SystemExit):
        cli.main(["market", "f1-draw", "--as-of", "2026-10-20T12:00:00"])  # an instant without its zone
