"""Recorded provider responses: CI and workers only replay; live calls are recorded once, redacted, and kept.

A cassette is keyed by the exact request (provider, endpoint, body), so a changed question, state rendering or config is
a new request and a missing cassette, never a stale answer (spec §6.2: record and replay; Handoff's judgment cache keys
the same way). A cassette keeps the request and response with every known secret value redacted (spec §14). Spend is
checked against the run's and the day's caps before a live call (spec §6.6) and logged after it.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from requisite.secrets import ReplayOnly, redact, replay_only

ROOT = Path(__file__).resolve().parent.parent
CASSETTES = Path(os.environ.get("REQUISITE_CASSETTES", ROOT / "cassettes"))
SPEND_LOG = Path(os.environ.get("REQUISITE_SPEND_LOG", ROOT / "var" / "spend.jsonl"))


class CassetteMissing(ReplayOnly):
    pass


class OverBudget(RuntimeError):
    pass


def request_key(provider: str, endpoint: str, body: Any) -> str:
    canon = json.dumps({"provider": provider, "endpoint": endpoint, "body": body}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


@dataclass
class Budget:
    """Caps on judge calls and dollars for one run, and dollars for the day (spec §6.6). Owen set the caps flexible on
    November 1; the defaults are generous and set by REQUISITE_RUN_USD_CAP, REQUISITE_RUN_CALL_CAP, REQUISITE_DAY_USD_CAP."""

    run_usd: float = float(os.environ.get("REQUISITE_RUN_USD_CAP", "25"))
    run_calls: int = int(os.environ.get("REQUISITE_RUN_CALL_CAP", "20000"))
    day_usd: float = float(os.environ.get("REQUISITE_DAY_USD_CAP", "100"))
    spent_usd: float = 0.0
    calls: int = 0
    reserved_usd: float = 0.0
    reserved_calls: int = 0
    log: Path = field(default_factory=lambda: SPEND_LOG)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    day_before: float | None = field(default=None, repr=False, compare=False)

    def day_spent(self, today: date | None = None) -> float:
        today = today or datetime.now(UTC).date()
        if not self.log.exists():
            return 0.0
        total = 0.0
        for line in self.log.read_text().splitlines():
            e = json.loads(line)
            if e["at"][:10] == today.isoformat():
                total += float(e["cost_usd"])
        return total

    def reserve(self, estimate_usd: float) -> None:
        """Check the caps for one more call and hold its estimate, atomically: calls made at once from several threads
        cannot together pass a cap that each alone would respect."""
        with self.lock:
            if self.calls + self.reserved_calls + 1 > self.run_calls:
                raise OverBudget(f"run call cap {self.run_calls} reached")
            if self.spent_usd + self.reserved_usd + estimate_usd > self.run_usd:
                raise OverBudget(f"run cap ${self.run_usd:.2f} would be passed (spent ${self.spent_usd:.2f})")
            if self.day_before is None:  # what earlier runs spent today, read once; this run's spend is added to it
                self.day_before = self.day_spent()
            if self.day_before + self.spent_usd + self.reserved_usd + estimate_usd > self.day_usd:
                raise OverBudget(f"day cap ${self.day_usd:.2f} would be passed")
            self.reserved_usd += estimate_usd
            self.reserved_calls += 1

    def settle(self, provider: str, estimate_usd: float, cost_usd: float | None) -> None:
        """Release a reservation and charge what the call cost (None: it failed before the provider billed)."""
        with self.lock:
            self.reserved_usd -= estimate_usd
            self.reserved_calls -= 1
            if cost_usd is None:
                return
            self.calls += 1
            self.spent_usd += cost_usd
            self.log.parent.mkdir(parents=True, exist_ok=True)
            with self.log.open("a") as f:
                entry = {"at": datetime.now(UTC).isoformat(), "provider": provider, "cost_usd": cost_usd}
                f.write(json.dumps(entry) + "\n")


@dataclass(frozen=True)
class Recorded:
    response: Any
    latency_ms: int
    cost_usd: float
    recorded_at: str
    replayed: bool


def path_for(provider: str, key: str, root: Path | None = None) -> Path:
    return (root or CASSETTES) / provider / f"{key}.json"


def call(
    provider: str,
    endpoint: str,
    body: Any,
    live: Callable[[], tuple[Any, float]],
    *,
    budget: Budget | None = None,
    estimate_usd: float = 0.01,
    root: Path | None = None,
) -> Recorded:
    """The response to this exact request: replayed from its cassette, or, where live calls are allowed, fetched once by
    ``live`` (which returns the response and its cost in dollars), redacted, and recorded."""
    key = request_key(provider, endpoint, body)
    path = path_for(provider, key, root)
    if path.exists():
        kept = json.loads(path.read_text())
        return Recorded(kept["response"], kept["latency_ms"], kept["cost_usd"], kept["recorded_at"], replayed=True)
    if replay_only():
        raise CassetteMissing(f"no cassette for {provider} {endpoint} request {key[:12]}; record it outside the sandbox")
    budget = budget or Budget()
    budget.reserve(estimate_usd)
    started = time.monotonic()
    try:
        response, cost = live()
    except BaseException:
        budget.settle(provider, estimate_usd, None)
        raise
    latency = int((time.monotonic() - started) * 1000)
    budget.settle(provider, estimate_usd, cost)
    kept = {
        "provider": provider,
        "endpoint": endpoint,
        "request": body,
        "response": response,
        "latency_ms": latency,
        "cost_usd": cost,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    text = redact(json.dumps(kept, indent=1, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written whole, then moved into place: a thread replaying the same request meanwhile reads all of it or none.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text + "\n")
    tmp.replace(path)
    # The response as kept, so the live call and every replay of it read the same value (ours: a value redaction changed
    # would otherwise differ between the run that recorded it and the runs that replay it).
    return Recorded(json.loads(text)["response"], latency, cost, kept["recorded_at"], replayed=False)
