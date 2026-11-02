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
from functools import partial
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

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


def json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The answer type's JSON Schema in a form both the Claude Code CLI's strict local check and the API's draft 2020-12
    accept: pydantic's OpenAPI-only "discriminator" keyword dropped (the oneOf it annotates already says which branch
    each value takes by its tag), and a tuple's "prefixItems" (which the CLI's check refuses) written as "items" with a
    fixed length, the items' schemas as alternatives where they differ. The answer is validated by its pydantic type
    afterwards, which holds each position to its own type."""

    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            out = {k: clean(v) for k, v in node.items() if k != "discriminator"}
            if "prefixItems" in out:
                positions = out.pop("prefixItems")
                distinct = [p for i, p in enumerate(positions) if p not in positions[:i]]
                out["items"] = distinct[0] if len(distinct) == 1 else {"anyOf": distinct}
                out["minItems"] = out["maxItems"] = len(positions)
            return out
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node

    cleaned: dict[str, Any] = clean(model.model_json_schema())
    return cleaned


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


ATTEMPTS = 3


def run[O: BaseModel](
    role: str, input: Mapping[str, Any], output: type[O], *, effort: str = "high", root: Path | None = None
) -> AgentResult[O]:
    """One agent call: the role's prompt, the input as canonical JSON, an answer of type `output`. An answer that its
    type refuses is returned to the agent with the validator's errors, as spec §13 has agents run the checks before they
    submit, up to ATTEMPTS times; each attempt is its own recorded call, so a replay takes the same path."""
    system = prompt(role)
    schema = json_schema(output)
    feedback: list[dict[str, Any]] = []
    for _ in range(ATTEMPTS):
        payload = dict(input) if not feedback else {**input, "earlier_attempts_refused": feedback}
        message = canonical_json(payload)
        body = {"role": role, "model": MODEL, "effort": effort, "system": system, "input": json.loads(message),
                "schema": schema}  # fmt: skip
        recorded = cassettes.call(
            "claude_agent", role, body, partial(_claude, system, message, schema, effort), estimate_usd=0.0, root=root
        )
        answer = recorded.response["structured_output"]
        try:
            # JSON mode, as the answer arrived: a record's tuple fields take JSON arrays there (strict Python mode
            # refuses a list for a tuple).
            value = output.model_validate_json(json.dumps(answer))
        except ValidationError as e:
            errors = [{"at": ".".join(str(x) for x in err["loc"]), "error": err["msg"]} for err in e.errors()]
            feedback.append({"answer": answer, "refused_because": errors})
            continue
        run_id = cassettes.request_key("claude_agent", role, body)
        return AgentResult(role, value, run_id, MODEL, effort, recorded.replayed)
    raise AgentError(f"{role}'s answer was refused {ATTEMPTS} times; the last refusal: {feedback[-1]['refused_because']}")


__all__ = ["MODEL", "AgentError", "AgentResult", "prompt", "run"]
