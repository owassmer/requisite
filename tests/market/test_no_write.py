"""No write endpoint is reachable from a market adapter (spec §13 WP13; spec §14: "No order placement, RFQ submission,
account linking or trading credentials. Market adapters expose no write endpoints").

Read from the package's syntax, never by running it: every module of requisite/market is parsed, and
- no network library but httpx is imported, nor the secrets module (a public read needs no credential);
- no call anywhere in the package is named for another HTTP method (post, put, patch, delete, request, stream, send,
  head, options), and httpx's own request functions are never called: the one call that sends is the client's
  `self._http.get`, inside Kalshi._send;
- every API path the package requests (the first argument of a get or a listing that names a path) is one of the read
  paths in kalshi.READS, each a GET of Kalshi's public market data (its OpenAPI specification: GetSeriesList,
  GetSeries, GetMarkets, GetMarket, GetHistoricalMarkets, GetHistoricalMarket, GetHistoricalCutoff);
- no header that carries a Kalshi credential is named.
The check is shown to catch each of these on sources built to break them."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from requisite.market import kalshi

PACKAGE = Path(kalshi.__file__).resolve().parent
NETWORK = {"requests", "urllib.request", "urllib3", "http", "http.client", "socket", "ssl", "aiohttp", "websockets"}
OTHER_METHODS = {"post", "put", "patch", "delete", "request", "stream", "send", "head", "options"}
CREDENTIALS = ("authorization", "kalshi-access-key", "kalshi-access-signature", "kalshi-access-timestamp")
# Kalshi's public market-data reads, each a GET in its OpenAPI specification (docs.kalshi.com/openapi.yaml).
EXPECTED_READS = {"/series", "/series/{}", "/markets", "/markets/{}", "/historical/markets",
                  "/historical/markets/{}", "/historical/cutoff"}  # fmt: skip


def _shape(node: ast.expr) -> str | None:
    """A path argument as a pattern: a string as written, an f-string with each value as {}; None if not a path."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        text = node.value
    elif isinstance(node, ast.JoinedStr):
        text = "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)  # type: ignore[misc]
    else:
        return None
    return text if text.startswith("/") else None


def _reads() -> set[str]:
    return {r.split("{")[0] + "{}" if "{" in r else r for r in kalshi.READS}


def violations(source: str, name: str) -> tuple[list[str], set[str]]:
    """What in this module could send anything but a GET to a read path, and the paths it requests."""
    tree = ast.parse(source, filename=name)
    found: list[str] = []
    paths: set[str] = set()
    parents: dict[ast.AST, ast.AST] = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}

    def function_of(node: ast.AST) -> str | None:
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                return node.name
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
        else:
            mods = []
        for m in mods:
            if m in NETWORK or m.split(".")[0] in {"requests", "urllib3", "socket", "aiohttp", "websockets"}:
                found.append(f"{name}:{node.lineno}: imports {m}")
            if m == "requisite.secrets" or m.startswith("requisite.secrets."):
                found.append(f"{name}:{node.lineno}: imports the secrets module")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(c in node.value.lower() for c in CREDENTIALS):
                found.append(f"{name}:{node.lineno}: names a credential header ({node.value[:40]!r})")
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        attr, receiver = node.func.attr, node.func.value
        if attr in OTHER_METHODS:
            found.append(f"{name}:{node.lineno}: calls .{attr}()")
        if isinstance(receiver, ast.Name) and receiver.id == "httpx" and attr.islower():
            found.append(f"{name}:{node.lineno}: calls httpx.{attr}() outside the client")
        is_http = isinstance(receiver, ast.Attribute) and receiver.attr == "_http"
        if attr == "get" and is_http and function_of(node) != "_send":
            found.append(f"{name}:{node.lineno}: sends from {function_of(node)}, not Kalshi._send")
        if attr in ("get", "listing") and node.args and (shape := _shape(node.args[0])) is not None:
            paths.add(shape)
            if shape not in _reads():
                found.append(f"{name}:{node.lineno}: requests {shape}, not a read path")
    return found, paths


def test_the_read_paths_are_kalshis_public_market_data_gets():
    assert _reads() == EXPECTED_READS


@pytest.mark.covers("WP13.no_write")
def test_no_write_endpoint_is_reachable_from_a_market_adapter():
    modules = sorted(PACKAGE.glob("*.py"))
    assert {m.name for m in modules} >= {"kalshi.py", "f1.py"}
    found: list[str] = []
    paths: set[str] = set()
    sends = 0
    for module in modules:
        source = module.read_text()
        v, p = violations(source, module.name)
        found += v
        paths |= p
        sends += source.count("self._http.get(")
    assert found == []
    assert sends == 1  # the one place a request leaves the package
    assert paths == EXPECTED_READS  # every read path in use, and nothing else


@pytest.mark.parametrize("bad, why", [
    ("def place(c):\n    c._http.post('/portfolio/orders', json={})\n", "calls .post()"),
    ("def f(c):\n    c.request('DELETE', '/portfolio/orders/1')\n", "calls .request()"),
    ("import httpx\ndef f():\n    httpx.get('https://external-api.kalshi.com/trade-api/v2/markets')\n", "httpx.get()"),
    ("class K:\n    def other(self):\n        self._http.get('x')\n", "not Kalshi._send"),
    ("def f(c):\n    c.get('/portfolio/balance')\n", "/portfolio/balance, not a read path"),
    ("def f(c, t):\n    c.get(f'/portfolio/orders/{t}')\n", "/portfolio/orders/{}, not a read path"),
    ("def f(c):\n    c.listing('/communications/rfqs', {}, 'rfqs')\n", "not a read path"),
    ("import requests\n", "imports requests"),
    ("from requisite.secrets import secret\n", "imports the secrets module"),
    ("H = {'KALSHI-ACCESS-KEY': 'x'}\n", "credential header"),
])  # fmt: skip
def test_the_check_catches_what_it_claims(bad, why):
    found, _ = violations(bad, "bad.py")
    assert any(why in f for f in found), found
