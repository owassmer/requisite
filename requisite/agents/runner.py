"""Running Requisite's own agents (spec §3, §13 WP15): Claude Opus 5.5 through the Claude Code CLI on Owen's subscription
(decisions, "Providers and keys"), each call a bounded task with a structured answer.

An agent proposes; it never writes. Its only output is a value of the role's output schema, which code turns into
Proposals and submits through the store's single write path (spec I1). It has no tools: it reads only what the
orchestrating code hands it, so everything it relied on is in its recorded input. Each call is recorded like a provider
response (requisite/cassettes.py), keyed by the exact request (role prompt, input, schema, model, effort), so a run
replays offline and a replay-only process (CI, a worker's sandbox) can never call out.

The role's prompt is its system prompt, read from requisite/agents/prompts/<role>.md: what the agent reads, shown whole in
a change's review (tools/roles/requisite-reads).
"""

from __future__ import annotations

import json
import os
import subprocess  # noqa: S404  (the Claude Code CLI, with fixed arguments)
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from requisite import cassettes
from requisite.core import canonical_json
from requisite.secrets import replay_only

PROMPTS = Path(__file__).resolve().parent / "prompts"
MODEL = "claude-opus-5-5"
TIMEOUT_S = 900


class AgentError(RuntimeError):
    """The agent's call failed, or its answer is not a value of its output schema."""


@dataclass(frozen=True)
class AgentResult[T]:
    """An agent's answer, validated against its schema, with the cassette key of the call that produced it (the run's
    identity: Proposals it leads to name it as their author)."""

    role: str
    output: T
    run: str
    model: str
    effort: str
    replayed: bool


def prompt(role: str) -> str:
    path = PROMPTS / f"{role}.md"
    if not path.exists():
        raise AgentError(f"no prompt for agent role {role!r} at {path}")
    return path.read_text()


def _oauth_token() -> str | None:
    """The Claude Code OAuth token, from the environment or the macOS Keychain, as the claude-cli extension reads it."""
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return os.environ["CLAUDE_CODE_OAUTH_TOKEN"]
    try:
        out = subprocess.run(  # noqa: S603
            ["/usr/bin/security", "find-generic-password", "-w", "-a", os.environ.get("USER", ""), "-s",
             "claude-code-oauth-token"],
            capture_output=True, text=True, timeout=10, check=False,
        )  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def _claude(system: str, message: str, schema: Mapping[str, Any], effort: str) -> tuple[Any, float]:
    if replay_only():
        raise cassettes.CassetteMissing("an agent call in a replay-only process; record it outside the sandbox")
    env = {k: v for k, v in os.environ.items() if not k.startswith("REQUISITE_")}
    token = _oauth_token()
    if token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    env.update(CLAUDE_CODE_DISABLE_CLAUDE_MDS="1", CLAUDE_CODE_DISABLE_AUTO_MEMORY="1")
    args = [
        "claude", "-p", message, "--model", MODEL, "--effort", effort, "--output-format", "json",
        "--no-session-persistence", "--tools", "", "--setting-sources", "", "--strict-mcp-config",
        "--system-prompt", system, "--json-schema", json.dumps(schema),
    ]  # fmt: skip
    try:
        done = subprocess.run(args, capture_output=True, text=True, env=env, timeout=TIMEOUT_S, check=False)  # noqa: S603
    except subprocess.TimeoutExpired as e:
        raise AgentError(f"the agent did not answer within {TIMEOUT_S}s") from e
    if done.returncode != 0:
        raise AgentError(f"claude exited {done.returncode}: {done.stderr[-500:]}")
    data = json.loads(done.stdout)
    if data.get("is_error") or "structured_output" not in data:
        raise AgentError(f"the agent's call ended {data.get('subtype')!r} without a structured answer")
    kept = {k: data.get(k) for k in ("structured_output", "modelUsage", "usage", "total_cost_usd", "duration_ms")}
    # On the subscription a call adds no dollars; total_cost_usd is the API-equivalent figure, kept in the response.
    return kept, 0.0


def run[O: BaseModel](
    role: str, input: Mapping[str, Any], output: type[O], *, effort: str = "high", root: Path | None = None
) -> AgentResult[O]:
    """One agent call: the role's prompt, the input as canonical JSON, an answer of type `output`."""
    system = prompt(role)
    schema = output.model_json_schema()
    message = canonical_json(dict(input))
    body = {"role": role, "model": MODEL, "effort": effort, "system": system, "input": json.loads(message),
            "schema": schema}  # fmt: skip
    recorded = cassettes.call(
        "claude_agent", role, body, lambda: _claude(system, message, schema, effort), estimate_usd=0.0, root=root
    )
    try:
        value = output.model_validate(recorded.response["structured_output"])
    except Exception as e:  # pydantic ValidationError and anything malformed: the answer is not one
        raise AgentError(f"{role}'s answer is not a {output.__name__}: {e}") from e
    return AgentResult(role, value, cassettes.request_key("claude_agent", role, body), MODEL, effort, recorded.replayed)


__all__ = ["MODEL", "AgentError", "AgentResult", "prompt", "run"]
