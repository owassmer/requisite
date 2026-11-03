"""An agent run's trace, committed by code beside the Proposals it made from the run's answer (spec §13 WP15: "each run
leaves a trace of Proposals and gate results"). The run itself is its cassette (requisite/agents/runner.py), so the
trace names it and the hashes of its input and answer.

A run is identified by its request, so a step run again replays the same run and gets the same answer. Where a proposal
that answer led to is still proposed or active, a step returns it (`made_from`) rather than proposing the same answer
again. Where every one was rejected, a step run again proposes the answer anew: the user re-runs it because the cause
was fixed (a gate that did not exist, a check that was wrong), and its gates judge it again. Each submission leaves its
own trace, keyed by the run and the submission (`record_run`), so the run's history keeps every proposal it led to."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from requisite import core
from requisite.agents.runner import AgentResult
from requisite.store import Store


def _traces(store: Store, run: str) -> list[core.AgentRun]:
    return [t for t in store.all("agent_run", core.AgentRun) if t.id == run[:16]]


def made_from(store: Store, result: AgentResult[Any]) -> list[core.Proposal]:
    """The proposals this run's answer already led to in this case that still stand (proposed or active), each as it
    now stands; none where the run is not recorded here or every proposal it led to was rejected."""
    found = [store.proposal(p) for t in _traces(store, result.run) for p in t.proposals]
    return [p for p in found if p is not None and p.status != "rejected"]


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
    earlier = len(_traces(store, result.run))
    store.commit("agent_run", run.id if not earlier else f"{run.id}.{earlier + 1}", run)
