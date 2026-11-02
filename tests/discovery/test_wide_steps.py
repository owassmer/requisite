"""A wide step's agent run, replayed when the step runs again, is the proposal it already made (requisite/agents/trace.py):
the step returns it rather than proposing the same answer a second time."""

from datetime import UTC, date, datetime

from requisite import core
from requisite.agents import runner
from requisite.agents.answers import ObjectiveDraft
from requisite.agents.runner import AgentResult
from requisite.discovery import wide
from requisite.store import Store

DRAFT = ObjectiveDraft(
    population="Small businesses under 500 employees.", events="pending cost measures",
    horizon=core.TimeWindow(start=datetime(2026, 11, 1, tzinfo=UTC), end=None, start_closed=True, end_closed=False),
    loss_definition="higher costs", estimand="share of items", open_questions=[],
)  # fmt: skip


def test_an_objective_drafted_again_from_the_same_run_is_the_proposal_it_made(tmp_path, monkeypatch):
    calls = []

    def recorded(role, input, output, *, effort):
        calls.append(input)
        return AgentResult(role, DRAFT, "run-" + core.sha256(core.canonical_json(input)), "test-double", effort, True)

    monkeypatch.setattr(runner, "run", recorded)
    store = Store(tmp_path / "case", case="wide")
    first = wide.draft_objective(store, "labor costs", date(2026, 11, 1))
    again = wide.draft_objective(store, "labor costs", date(2026, 11, 1))
    assert len(calls) == 2 and again == first and first.status == "proposed"
    assert [p.id for p in store.all("proposal", core.Proposal)] == [first.id]
    (run,) = store.all("agent_run", core.AgentRun)
    assert run.proposals == (first.id,)
    # Another brief is another run, and its own proposal.
    other = wide.draft_objective(store, "input costs", date(2026, 11, 1))
    assert other.id != first.id and len(store.all("agent_run", core.AgentRun)) == 2


def test_a_human_gate_decision_is_recorded_and_replays_from_a_clean_clone(tmp_path, monkeypatch):
    # A G1 approval is the human's recorded answer, replayed like a provider's: a fresh store replaying it commits the
    # same approval at the same instant, and a decision given differently from the recording is refused.
    import pytest

    from requisite import cassettes, secrets

    monkeypatch.delenv("REQUISITE_REPLAY_ONLY", raising=False)
    monkeypatch.setattr(secrets, "known_values", lambda: [])
    monkeypatch.setattr(cassettes, "CASSETTES", tmp_path / "cassettes")
    monkeypatch.setattr(runner, "run", lambda role, input, output, *, effort: AgentResult(
        role, DRAFT, "run-" + core.sha256(core.canonical_json(input)), "test-double", effort, True))

    def decided(root, clock):
        store = Store(tmp_path / root, case="wide", clock=lambda: clock)
        p = wide.draft_objective(store, "labor costs", date(2026, 11, 1))
        return store, wide.approve(store, p.id, gate="G1", approver="owen", decision="approved", reason="as drafted",
                                   scope="this case")  # fmt: skip

    first, p = decided("a", datetime(2026, 11, 1, 22, tzinfo=UTC))
    assert p.status == "active"
    monkeypatch.setenv("REQUISITE_REPLAY_ONLY", "1")
    again, q = decided("b", datetime(2026, 11, 9, tzinfo=UTC))  # another day: the recorded instant stands
    assert q.status == "active" and again.all("approval", core.Approval) == first.all("approval", core.Approval)
    store = Store(tmp_path / "c", case="wide")
    r = wide.draft_objective(store, "labor costs", date(2026, 11, 1))
    with pytest.raises(wide.Refused, match="recorded decision at G1 .* differs from the one given: decision"):
        wide.approve(store, r.id, gate="G1", approver="owen", decision="rejected", reason="as drafted", scope="this case")
