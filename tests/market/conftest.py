"""Shared by the market tests: a live-allowed mode in which the client records through the fake Kalshi into a fresh
cassette directory (never the repository's), and a client over it."""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pytest
from fake_kalshi import FakeKalshi, FakeTime

from requisite import cassettes, secrets
from requisite.market.kalshi import Kalshi, Pacer
from requisite.store import Store


@pytest.fixture
def live(monkeypatch, tmp_path):
    """Live calls allowed, as in the recording run outside the sandbox: only the fake answers them, and Kalshi's public
    endpoints take no secret, so none is looked up to redact."""
    monkeypatch.delenv("REQUISITE_REPLAY_ONLY", raising=False)
    monkeypatch.setattr(secrets, "known_values", lambda: [])
    return tmp_path


def budget(tmp: Path) -> cassettes.Budget:
    return cassettes.Budget(run_usd=0.0, run_calls=sys.maxsize, day_usd=float("inf"), log=tmp / "spend.jsonl")


def client(api: FakeKalshi | None, root: Path, tmp: Path, *, snapshot: datetime | None = None,
           time: FakeTime | None = None) -> Kalshi:  # fmt: skip
    """A client over the fake (or, with None, over no transport of ours: the real network, which a replay never
    reaches), recording into `root`."""
    time = time or FakeTime()
    return Kalshi(transport=api.transport() if api else None, root=root, snapshot=snapshot,
                  pacer=Pacer(time.clock, time.sleep), budget=budget(tmp))  # fmt: skip


@pytest.fixture
def make_client() -> Callable[..., Kalshi]:
    return client


@pytest.fixture
def store(tmp_path) -> Store:
    s = Store(tmp_path / "case", case="f1")
    yield s
    s.close()
