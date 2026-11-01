"""Every test runs replay-only unless it is marked live: no test spends or varies (spec §14)."""

import pytest


@pytest.fixture(autouse=True)
def _replay_only(request, monkeypatch):
    if "live" not in request.keywords:
        monkeypatch.setenv("REQUISITE_REPLAY_ONLY", "1")
