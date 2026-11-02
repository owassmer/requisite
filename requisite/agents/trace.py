"""An agent run's trace, committed by code beside the Proposals it made from the run's answer (spec §13 WP15: "each run
leaves a trace of Proposals and gate results"). The run itself is its cassette (requisite/agents/runner.py), so the
trace names it and the hashes of its input and answer.

A run is identified by its request, so a step run again replays the same run and gets the same answer. The proposals
that answer led to are already committed, and its trace names them: a step returns those (`made_from`) rather than
proposing the same answer again, which would commit a second proposal of one run and then a second, different trace
under the run's one id."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from requisite import core
from requisite.agents.runner import AgentResult
from requisite.store import Store


def made_from(store: Store, result: AgentResult[Any]) -> list[core.Proposal]:
    """The proposals this run's answer already led to in this case, each as it now stands; none for a run not yet
    recorded here."""
    run = store.get("agent_run", result.run[:16], core.AgentRun)
    if run is None:
        return []
    found = [store.proposal(p) for p in run.proposals]
    return [p for p in found if p is not None]


def record_run(store: Store, result: AgentResult[Any], input: Mapping[str, Any], proposals: Sequence[str]) -> None:
    output = result.output.model_dump(mode="json") if isinstance(result.output, BaseModel) else result.output
    run = core.AgentRun(
        id=result.run[:16],
        role=result.role,
        model=result.model,
        effort=result.effort,
        input_hash=core.sha256(core.canonical_json(dict(input))),
        output_hash=core.sha256(core.canonical_json(output)),
        response=result.run,
        proposals=list(proposals),
        at=store.clock(),
    )
    store.commit("agent_run", run.id, run)
