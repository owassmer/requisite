"""Credentials, read at run time and never printed, logged, cassetted or fixtured (spec §14).

Where a key is found, in order: the environment; this checkout's ``.env``; Slope's ``.env``; the macOS Keychain under
``handoff.<NAME>`` (Owen, November 1: Requisite uses the same keys as Slope and Handoff). A process with
``REQUISITE_REPLAY_ONLY=1`` (CI, every worker) is never given a key: it has no business calling a provider.
"""

from __future__ import annotations

import os
import re
import subprocess  # noqa: S404  (the Keychain's own command, fixed arguments)
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOTENVS = (ROOT / ".env", Path.home() / "dev" / "Slope_Sparse_Events" / ".env")
NAMES = (
    "OPENAI_DECISIONS_API_KEY",
    "OPENROUTER_API_KEY",
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_API_TOKEN",
    "GOV_API_KEY",
)


class MissingSecret(RuntimeError):
    pass


class ReplayOnly(RuntimeError):
    """A live call was attempted where only recorded responses may be used."""


def replay_only() -> bool:
    return os.environ.get("REQUISITE_REPLAY_ONLY", "") not in ("", "0")


def _dotenv(path: Path, name: str) -> str | None:
    try:
        text = path.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        m = re.match(rf"^\s*(?:export\s+)?{re.escape(name)}\s*=\s*(.*?)\s*$", line)
        if m:
            return m.group(1).strip().strip("\"'") or None
    return None


def _keychain(name: str) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603
            ["/usr/bin/security", "find-generic-password", "-s", f"handoff.{name}", "-w"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


@cache
def secret(name: str) -> str:
    if replay_only():
        raise ReplayOnly(f"{name} requested in a replay-only process; record the cassette outside the sandbox")
    if os.environ.get(name):
        return os.environ[name]
    for path in DOTENVS:
        if value := _dotenv(path, name):
            return value
    if value := _keychain(name):
        return value
    raise MissingSecret(f"{name} is not set in the environment, .env, Slope's .env or the Keychain")


def known_values() -> list[str]:
    """Every secret value this process could read, for redaction. Never raises, never prints."""
    if replay_only():
        return [v for n in NAMES if (v := os.environ.get(n))]
    values = []
    for n in NAMES:
        try:
            values.append(secret(n))
        except (MissingSecret, ReplayOnly):
            continue
    return values


def redact(text: str, values: list[str] | None = None) -> str:
    """The text with every known secret value replaced. Applied to everything a cassette or log keeps."""
    for v in values if values is not None else known_values():
        if v and len(v) >= 8:
            text = text.replace(v, "[REDACTED]")
    return text
