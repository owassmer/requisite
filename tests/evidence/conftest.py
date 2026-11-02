"""Shared by the evidence tests: a live-allowed mode in which the client records through the fake API into a fresh
cassette directory (never the repository's), and the scanned page the OCR path reads."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from fake_api import KEY, FakeAPI, FakeTime, fixture_pdf, rasterized

from requisite import cassettes, secrets
from requisite.evidence.regulations_gov import Pacer, RegulationsGov
from requisite.store import Store


@pytest.fixture
def live(monkeypatch, tmp_path):
    """Live calls allowed, as the recording run outside the sandbox: only the fake API answers them, and the only secret
    there is the fake key, which redaction removes from anything kept."""
    monkeypatch.delenv("REQUISITE_REPLAY_ONLY", raising=False)
    monkeypatch.setattr(secrets, "known_values", lambda: [KEY])
    return tmp_path


def budget(tmp: Path) -> cassettes.Budget:
    return cassettes.Budget(run_usd=0.0, run_calls=sys.maxsize, day_usd=float("inf"), log=tmp / "spend.jsonl")


def client(api: FakeAPI | None, root: Path, tmp: Path, time: FakeTime | None = None) -> RegulationsGov:
    """A client over the fake API (or, with None, over no transport of ours: the real network, which a replay never
    reaches), recording into `root`."""
    time = time or (api.time if api is not None else FakeTime())

    def key() -> str:
        if api is None:
            raise AssertionError("a replay asked for the key")
        return KEY

    return RegulationsGov(transport=api.transport() if api else None, key=key, root=root,
                          pacer=Pacer(time.clock, time.sleep), budget=budget(tmp))  # fmt: skip


@pytest.fixture
def make_client() -> Callable[..., RegulationsGov]:
    return client


@pytest.fixture
def store(tmp_path) -> Store:
    s = Store(tmp_path / "case", case="case")
    yield s
    s.close()


@pytest.fixture(scope="session")
def scan() -> bytes:
    """United's letter as a scan: its page rendered and wrapped as an image-only PDF (no text layer)."""
    return rasterized(fixture_pdf())
