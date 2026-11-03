"""Kalshi's public trade API v2 served offline, through an httpx transport, with the documented behaviour the client
relies on (docs.kalshi.com/openapi.yaml and its guides): GET only; /series whole; /series/{ticker}; /markets with
limit (at most 1,000), cursor, series_ticker, event_ticker, tickers, min_settled_ts and max_settled_ts (Unix seconds,
read here strictly: "settled after" and "settled before"); /markets/{ticker}; the archive (/historical/markets with one
filter at a time, /historical/markets/{ticker}, /historical/cutoff), where a market settled before the cutoff is served
only; and the filed documents by URL. Tests add series and markets in the recorded shapes (fixtures/kalshi/KXCAGOAT),
and script failures (statuses, dropped connections) before a path answers.

Nothing here is the product: it stands in for the network, which the sandbox and CI do not have. Its data are test
inputs written in the tests or read from the recorded fixtures, never cassettes."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import httpx

from requisite.evidence.fetched import USER_AGENT

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures" / "kalshi" / "KXCAGOAT"
PREFIX = "/trade-api/v2"


def load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())


def when(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def series_entry(ticker: str, *, terms: bool = True) -> dict[str, Any]:
    """A series in the recorded series shape (fixtures/kalshi/KXCAGOAT/series.json)."""
    entry = copy.deepcopy(load("series.json")["body"]["series"])
    entry.update(ticker=ticker, title=f"series {ticker}",
                 contract_url=f"https://assets.kalshi.com/regulatory/product-certifications/{ticker}.pdf",
                 contract_terms_url=f"https://assets.kalshi.com/contract_terms/{ticker}.pdf" if terms else "")  # fmt: skip
    return entry


def market_entry(ticker: str, event: str, *, settled: str | None, strike: str | None = None,
                 floor: str | None = None, cap: str | None = None) -> dict[str, Any]:  # fmt: skip
    """A market in the recorded market shape (fixtures/kalshi/KXCAGOAT/markets.json), settled at `settled` or not."""
    m = copy.deepcopy(load("markets.json")["body"]["markets"][0])
    m.update(ticker=ticker, event_ticker=event, title=f"market {ticker}", rules_primary=f"Rules of {ticker}.",
             rules_secondary=f"Secondary rules of {ticker}.")  # fmt: skip
    for k in ("strike_type", "floor_strike", "cap_strike"):
        m.pop(k, None)
    if strike is not None:
        m["strike_type"] = strike
    if floor is not None:
        m["floor_strike"] = float(floor)
    if cap is not None:
        m["cap_strike"] = float(cap)
    if settled is None:
        for k in ("settlement_ts", "settlement_value_dollars"):
            m.pop(k, None)
        m.update(status="active", result="", expiration_value="")
    else:
        m.update(settlement_ts=settled, status="finalized", result="yes", expiration_value="Yes",
                 settlement_value_dollars="1.0000")  # fmt: skip
    return m


def pdf(text: str) -> bytes:
    return f"%PDF-1.4\n% {text}\n%%EOF\n".encode()


@dataclass
class FakeKalshi:
    cutoff: str = "2026-09-02T00:00:00Z"
    series: dict[str, dict[str, Any]] = field(default_factory=dict)
    markets: dict[str, dict[str, Any]] = field(default_factory=dict)  # ticker -> market, with "series" kept apart
    of_series: dict[str, str] = field(default_factory=dict)  # market ticker -> series ticker
    files: dict[str, bytes] = field(default_factory=dict)
    script: dict[str, list[int | str]] = field(default_factory=dict)  # path -> statuses (or "drop") before it answers
    series_page: int | None = None  # serve /series in pages of this size, with a cursor
    requests: list[httpx.Request] = field(default_factory=list)

    # ---------------------------------------------------------------- the record

    def add_series(self, entry: dict[str, Any], *, documents: bool = True) -> None:
        self.series[entry["ticker"]] = entry
        if documents:
            for key in ("contract_url", "contract_terms_url"):
                if entry.get(key):
                    self.files[entry[key]] = pdf(f"{key} of {entry['ticker']}")

    def add_market(self, series: str, market: dict[str, Any]) -> None:
        self.markets[market["ticker"]] = market
        self.of_series[market["ticker"]] = series

    def kxcagoat(self) -> None:
        """The recorded KXCAGOAT series, its market and its two filed PDFs, as fixtures/kalshi/KXCAGOAT holds them."""
        entry = load("series.json")["body"]["series"]
        self.add_series(entry, documents=False)
        self.files[entry["contract_url"]] = (FIXTURES / "contract.pdf").read_bytes()
        self.files[entry["contract_terms_url"]] = (FIXTURES / "contract_terms.pdf").read_bytes()
        for m in load("markets.json")["body"]["markets"]:
            self.add_market("KXCAGOAT", m)

    def _archived(self, m: dict[str, Any]) -> bool:
        return bool(m.get("settlement_ts")) and when(m["settlement_ts"]) < when(self.cutoff)

    # ---------------------------------------------------------------- serving

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        assert request.method == "GET", f"the client sent {request.method}"
        assert request.headers["user-agent"] == USER_AGENT
        self.requests.append(request)
        url = urlsplit(str(request.url))
        path = url.path.removeprefix(PREFIX)
        params = dict(parse_qsl(url.query))
        scripted = self.script.get(path)
        if scripted:
            step = scripted.pop(0)
            if step == "drop":
                raise httpx.ConnectError("connection dropped", request=request)
            return httpx.Response(int(step), json={"error": "scripted"})
        if f"{url.scheme}://{url.netloc}{url.path}" in self.files:
            return httpx.Response(200, content=self.files[f"{url.scheme}://{url.netloc}{url.path}"],
                                  headers={"content-type": "application/pdf"})  # fmt: skip
        if path == "/series":
            return self._series_list(params)
        if path.startswith("/series/"):
            entry = self.series.get(path.removeprefix("/series/"))
            return httpx.Response(200, json={"series": entry}) if entry else httpx.Response(404, json={"error": "nf"})
        if path == "/historical/cutoff":
            return httpx.Response(200, json={"market_settled_ts": self.cutoff, "trades_created_ts": self.cutoff,
                                             "orders_updated_ts": self.cutoff,
                                             "market_positions_last_updated_ts": self.cutoff})  # fmt: skip
        if path in ("/markets", "/historical/markets"):
            return self._listing(path, params)
        for prefix, archived in (("/markets/", False), ("/historical/markets/", True)):
            if path.startswith(prefix):
                m = self.markets.get(path.removeprefix(prefix))
                if m is None or self._archived(m) != archived:
                    return httpx.Response(404, json={"error": "not found"})
                return httpx.Response(200, json={"market": m})
        return httpx.Response(404, json={"error": f"no route {path}"})

    def _series_list(self, params: dict[str, str]) -> httpx.Response:
        everything = list(self.series.values())
        if self.series_page is None:
            return httpx.Response(200, json={"series": everything})
        start = int(params.get("cursor") or 0)
        end = start + self.series_page
        return httpx.Response(200, json={"series": everything[start:end],
                                         "cursor": str(end) if end < len(everything) else ""})  # fmt: skip

    def _listing(self, path: str, params: dict[str, str]) -> httpx.Response:
        archive = path == "/historical/markets"
        filters = {k for k in ("series_ticker", "event_ticker", "tickers") if k in params}
        if archive and len(filters) > 1:
            return httpx.Response(400, json={"error": "filters are mutually exclusive"})
        limit = int(params.get("limit", 100))
        if not 0 < limit <= 1000:
            return httpx.Response(400, json={"error": "limit"})
        chosen = []
        for ticker in sorted(self.markets):
            m = self.markets[ticker]
            if self._archived(m) != archive:
                continue
            if "series_ticker" in params and self.of_series[ticker] != params["series_ticker"]:
                continue
            if "event_ticker" in params and m["event_ticker"] != params["event_ticker"]:
                continue
            if "tickers" in params and ticker not in params["tickers"].split(","):
                continue
            if "min_settled_ts" in params or "max_settled_ts" in params:
                if not m.get("settlement_ts"):
                    continue
                at = when(m["settlement_ts"]).timestamp()
                if "min_settled_ts" in params and not at > int(params["min_settled_ts"]):
                    continue
                if "max_settled_ts" in params and not at < int(params["max_settled_ts"]):
                    continue
            chosen.append(m)
        start = int(params.get("cursor") or 0)
        end = start + limit
        return httpx.Response(200, json={"markets": chosen[start:end],
                                         "cursor": str(end) if end < len(chosen) else ""})  # fmt: skip

    def paths(self) -> list[str]:
        return [urlsplit(str(r.url)).path.removeprefix(PREFIX) for r in self.requests]


@dataclass
class FakeTime:
    """A clock and a sleep for the pacer: sleeping advances the clock, and nothing waits in real time."""

    now: float = 0.0
    slept: list[float] = field(default_factory=list)

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds
