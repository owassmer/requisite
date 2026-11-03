"""An agent run's trace (requisite/agents/trace.py): a step run again returns the proposal its answer led to while that
proposal stands, and proposes the answer anew where it was rejected, each submission leaving its own trace."""

import sys
from datetime import UTC, datetime
from pathlib import Path

from requisite import core
from requisite.agents.runner import AgentResult
from requisite.agents.trace import made_from, record_run
from requisite.store import Store

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from core_examples import examples  # noqa: E402

T = datetime(2026, 11, 1, 20, 0, tzinfo=UTC)


def test_a_rejected_proposal_is_proposed_anew_and_a_standing_one_is_returned(tmp_path):
    store = Store(tmp_path / "case", case="case", clock=lambda: T)
    result = AgentResult("frame_drafter", {"answer": 1}, "run" + "0" * 61, "model", "high", True)
    assert made_from(store, result) == []
    gate = {"passed": False}
    store.register_gate("frame", "tests", lambda s, p: core.GateResult(gate="tests", passed=gate["passed"], detail="t"))
    payload = next(x for x in examples() if type(x) is core.FrameCriteria).model_dump(mode="json")

    def submit(version: int) -> core.Proposal:
        p = store.submit_proposal(core.Proposal(id=f"frame-{version}", case="case", kind="frame",
                                                payload={**payload, "version": version}, author=result.run[:16],
                                                rationale="r", status="proposed"))  # fmt: skip
        record_run(store, result, {"in": 1}, [p.id])
        return p

    first = submit(1)
    assert first.status == "rejected" and made_from(store, result) == []  # rejected: the step proposes anew
    gate["passed"] = True
    second = submit(2)
    assert second.status == "active" and [p.id for p in made_from(store, result)] == ["frame-2"]
    traces = store.all("agent_run", core.AgentRun)
    assert [list(t.proposals) for t in traces] == [["frame-1"], ["frame-2"]] and {t.id for t in traces} == {result.run[:16]}
