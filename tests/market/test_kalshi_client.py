"""The Kalshi client against a fake that serves Kalshi's documented shapes: paging by cursor, retrying what may pass and
raising after its retries, waiting out rate limits, pacing, every request a GET naming Requisite, and a second run
that reads only cassettes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from fake_kalshi import FakeKalshi, FakeTime, market_entry, series_entry

from requisite.market import kalshi
from requisite.market.kalshi import KalshiError


def _fake_with_markets(n: int) -> FakeKalshi:
    api = FakeKalshi()
    api.add_series(series_entry("KXSYN"))
    for i in range(n):
        api.add_market("KXSYN", market_entry(f"KXSYN-E-{i:04d}", "KXSYN-E", settled="2026-10-01T00:00:00Z"))
    return api


def test_listings_follow_the_cursor_until_none_is_given(live, make_client):
    api = _fake_with_markets(2345)
    c = make_client(api, live / "cassettes", live)
    listing = c.markets(series_ticker="KXSYN")
    assert len(listing.pages) == 3  # 1,000 to a page, the API's largest
    assert [e["ticker"] for e, _ in listing.entries] == [f"KXSYN-E-{i:04d}" for i in range(2345)]
    assert [p for e, p in listing.entries][-1] == 2
    asked = [dict(r.url.params) for r in api.requests]
    assert [a.get("cursor") for a in asked] == [None, "1000", "2000"]
    assert all(a["limit"] == "1000" for a in asked)


def test_a_paged_series_list_is_followed(live, make_client):
    api = FakeKalshi(series_page=2)
    for t in ("KXA", "KXB", "KXC", "KXD", "KXE"):
        api.add_series(series_entry(t))
    listing = make_client(api, live / "cassettes", live).series_list()
    assert [e["ticker"] for e, _ in listing.entries] == ["KXA", "KXB", "KXC", "KXD", "KXE"]
    assert len(listing.pages) == 3


def test_a_cursor_that_comes_back_raises_rather_than_looping(live, make_client):
    api = _fake_with_markets(3)

    def stuck(request):
        import httpx

        return httpx.Response(200, json={"markets": [], "cursor": "same"})

    api.handle = stuck  # type: ignore[method-assign]
    with pytest.raises(KalshiError, match="came back again"):
        make_client(api, live / "cassettes", live).markets(series_ticker="KXSYN")


def test_transient_failures_are_retried_and_then_answered(live, make_client):
    api = _fake_with_markets(1)
    api.script["/markets"] = [503, "drop", 500, 502]
    time = FakeTime()
    listing = make_client(api, live / "cassettes", live, time=time).markets(series_ticker="KXSYN")
    assert len(listing.entries) == 1
    assert len(api.requests) == 5  # four failures, then the answer
    assert time.slept[-4:] == [2.0, 4.0, 8.0, 16.0]


def test_a_failure_that_does_not_pass_raises_after_the_retries(live, make_client):
    api = _fake_with_markets(1)
    api.script["/markets"] = [503] * (kalshi.RETRIES + 1)
    with pytest.raises(KalshiError, match="HTTP 503 after"):
        make_client(api, live / "cassettes", live).markets(series_ticker="KXSYN")
    api.script["/markets"] = ["drop"] * (kalshi.RETRIES + 1)
    with pytest.raises(KalshiError, match="no response after"):
        make_client(api, live / "cassettes", live).markets(series_ticker="KXSYN", event_ticker="other")


def test_a_rate_limit_is_waited_out_with_backoff_never_dropped(live, make_client):
    api = _fake_with_markets(1)
    api.script["/markets"] = [429] * 9
    time = FakeTime()
    listing = make_client(api, live / "cassettes", live, time=time).markets(series_ticker="KXSYN")
    assert len(listing.entries) == 1  # answered once the refusals ended: nothing skipped
    backoffs = [s for s in time.slept if s >= 1.0]
    assert backoffs == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0, 60.0, 60.0]


def test_a_rate_limit_that_never_ends_raises(live, make_client):
    api = _fake_with_markets(1)
    api.script["/markets"] = [429] * 100
    with pytest.raises(KalshiError, match="still refused"):
        make_client(api, live / "cassettes", live).markets(series_ticker="KXSYN")


def test_other_errors_raise_with_what_the_api_said(live, make_client):
    api = _fake_with_markets(1)
    api.script["/series/KXSYN"] = [400]
    with pytest.raises(KalshiError, match="HTTP 400"):
        make_client(api, live / "cassettes", live).series("KXSYN")


def test_live_requests_are_paced_at_the_read_budget():
    time = FakeTime()
    pacer = kalshi.Pacer(time.clock, time.sleep)
    for _ in range(5):
        pacer.acquire()
    assert time.slept == pytest.approx([0.05] * 4)  # 20 a second: 200 tokens a second over 10 tokens a request


def test_the_second_run_reads_only_cassettes(live, make_client, monkeypatch):
    api = _fake_with_markets(1500)
    root = live / "cassettes"
    first = make_client(api, root, live).markets(series_ticker="KXSYN")
    monkeypatch.setenv("REQUISITE_REPLAY_ONLY", "1")
    again = make_client(None, root, live).markets(series_ticker="KXSYN")  # no transport of ours: a replay or nothing
    assert [p.content for p in again.pages] == [p.content for p in first.pages]
    assert [p.retrieved_at for p in again.pages] == [p.retrieved_at for p in first.pages]
    assert len(api.requests) == 2


def test_an_instant_served_is_part_of_each_key(live, make_client, monkeypatch):
    api = _fake_with_markets(1)
    root = live / "cassettes"
    one = datetime(2026, 10, 1, tzinfo=UTC)
    make_client(api, root, live, snapshot=one).series_list()
    make_client(api, root, live, snapshot=datetime(2026, 10, 2, tzinfo=UTC)).series_list()
    assert len(api.requests) == 2  # the same request for another instant is another request
    monkeypatch.setenv("REQUISITE_REPLAY_ONLY", "1")
    make_client(None, root, live, snapshot=one).series_list()
    eastern = one.astimezone(timezone(timedelta(hours=-4)))  # the same instant, written with another offset
    make_client(None, root, live, snapshot=eastern).series_list()
    with pytest.raises(kalshi.cassettes.CassetteMissing):
        make_client(None, root, live, snapshot=datetime(2026, 10, 3, tzinfo=UTC)).series_list()


def test_documents_are_fetched_whole_and_only_over_https(live, make_client):
    api = FakeKalshi()
    api.add_series(series_entry("KXSYN"))
    c = make_client(api, live / "cassettes", live)
    doc = c.download("https://assets.kalshi.com/contract_terms/KXSYN.pdf")
    assert doc.content.startswith(b"%PDF-") and doc.source == "Kalshi"
    with pytest.raises(KalshiError, match="https"):
        c.download("http://assets.kalshi.com/contract_terms/KXSYN.pdf")
    api.script["/contract_terms/KXGONE.pdf"] = [404]
    with pytest.raises(KalshiError, match="HTTP 404"):
        c.download("https://assets.kalshi.com/contract_terms/KXGONE.pdf")


def test_the_archive_takes_one_filter_at_a_time(live, make_client):
    with pytest.raises(ValueError, match="one filter"):
        make_client(FakeKalshi(), live / "cassettes", live).historical_markets(series_ticker="A", event_ticker="B")


def test_quotes_are_prices_in_dollars_never_probabilities(live, make_client):
    api = FakeKalshi()
    api.kxcagoat()
    c = make_client(api, live / "cassettes", live)
    response = c.market("KXCAGOAT-26SEP30")
    q = kalshi.quote(kalshi.body(response)["market"], response)
    assert (q.yes_bid, q.yes_ask, q.no_bid, q.no_ask, q.last) == (
        Decimal("0.0000"), Decimal("1.0000"), Decimal("0.0000"), Decimal("1.0000"), Decimal("0.9900"))
    assert q.volume == Decimal("655736.60") and q.at == response.retrieved_at
    assert not any("prob" in name for name in kalshi.Quote.__dataclass_fields__)


def test_strikes_are_read_exactly():
    import json

    from requisite.evidence.fetched import Fetched

    raw = json.dumps({"market": {"floor_strike": 86299.99}}).encode()
    f = Fetched("Kalshi", "x", 200, "application/json", raw, datetime(2026, 11, 1, tzinfo=UTC))
    assert kalshi.body(f)["market"]["floor_strike"] == Decimal("86299.99")
