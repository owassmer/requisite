"""Committed state (requisite/store): the single write path (I1), immutable evidence (I2), synthetic quarantine in the
store (I4), and acceptance records written at decision time (spec §11)."""

import ast
from datetime import UTC, datetime, timedelta
from decimal import Decimal as D
from fractions import Fraction as F
from pathlib import Path

import pytest

from requisite import core as c
from requisite.core.evidence import artifact_id, claim_id, synthetic_artifact_id
from requisite.store import Conflict, Refused, Store

NOW = datetime(2026, 11, 1, 18, 0, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "case1", case="case1", clock=lambda: NOW)
    yield s
    s.close()


def artifact(content: bytes, predecessor=None) -> c.Artifact:
    return c.Artifact(
        id=artifact_id(content), locator="https://example.gov/doc", source="example.gov", media_type="text/plain",
        size=len(content), published_at=NOW, published_at_basis="postedDate", observed_at=NOW, availability="retrieved",
        predecessor=predecessor,
    )  # fmt: skip


def question_proposal(pid="p1", **payload_changes) -> c.Proposal:
    payload = c.AtomicQuestion(
        id="q", version=1, task="interpret", target="Does the order cover temporary workers in 2027?",
        answers=[c.AnswerValue(value="yes", meaning="m", status="affirmative"),
                 c.AnswerValue(value="ne", meaning="m", status="not_established")],
    ).model_dump(mode="json")  # fmt: skip
    payload.update(payload_changes)
    return c.Proposal(id=pid, case="case1", kind="question", payload=payload, author="run-1", rationale="r",
                      status="proposed")  # fmt: skip


@pytest.mark.covers("I1")
def test_a_proposal_becomes_active_only_when_its_required_gates_pass(store):
    # A question's required gates are schema and relevance (core.REQUIRED_GATES). With no relevance check registered,
    # it cannot become active: an unchecked gate is a failed gate.
    p = store.submit_proposal(question_proposal("p1"))
    assert p.status == "rejected" and [g.gate for g in p.gates] == ["schema", "relevance"]
    assert not p.gates[-1].passed and "no relevance check" in p.gates[-1].detail
    # A malformed payload stops at the schema gate.
    bad = store.submit_proposal(question_proposal("p2", answers=[]))
    assert bad.status == "rejected" and [g.gate for g in bad.gates] == ["schema"]
    # Once relevance is decided, by a check a package registers, the proposal can become active, and only then.
    store.register_gate(
        "question", "relevance", lambda st, pr: c.GateResult(gate="relevance", passed=True, detail="reaches D")
    )
    ok = store.submit_proposal(question_proposal("p3"))
    assert ok.status == "active" and all(g.passed for g in ok.gates)
    assert store.proposal("p3").status == "active"
    # A proposal arrives as proposed with no gate results: the gates are the store's to run, never the author's.
    with pytest.raises(Refused):
        store.submit_proposal(question_proposal("p4").model_copy(update={"status": "active"}))
    with pytest.raises(Refused):
        store.submit_proposal(
            question_proposal("p5").model_copy(update={"gates": [c.GateResult(gate="schema", passed=True, detail="")]})
        )
    # A gate is registered only for a kind that passes it.
    with pytest.raises(Refused):
        store.register_gate("assumption", "tests", lambda st, pr: c.GateResult(gate="tests", passed=True, detail=""))


def test_a_kinds_registered_schema_check_runs_once_its_record_validates(store):
    # A kind whose validator is more than its record's (spec §5: the contract layer's IR type check and evaluator)
    # registers a schema check; it decides the schema gate only for a payload that is already a valid record.
    seen = []

    def beyond(st, pr):
        seen.append(pr.id)
        return c.GateResult(gate="schema", passed=False, detail="the evaluator refuses it")

    store.register_gate("question", "schema", beyond)
    malformed = store.submit_proposal(question_proposal("s1", answers=[]))
    assert [(g.gate, g.passed) for g in malformed.gates] == [("schema", False)] and seen == []
    assert "the evaluator refuses it" not in malformed.gates[0].detail
    refused = store.submit_proposal(question_proposal("s2"))
    assert seen == ["s2"] and refused.status == "rejected"
    assert [(g.gate, g.passed, g.detail) for g in refused.gates] == [("schema", False, "the evaluator refuses it")]


def test_a_human_gate_waits_for_an_approval_of_that_exact_payload(store):
    spec = c.DecisionSpec(
        id="d", version=1, entity="e", decision_time=NOW,
        horizon=c.TimeWindow(start=NOW, end=NOW + timedelta(days=90), start_closed=True, end_closed=False),
        loss_definition="incremental cash cost", actions=["buy", "none"], objective="terminal_residual", alpha=None,
        tolerance=D("100"), liquidity_floor=[], currency="USD", information=[], recourse="none",
    )  # fmt: skip
    proposal = c.Proposal(id="g1", case="case1", kind="decision_spec", payload=spec.model_dump(mode="json"),
                          author="run-2", rationale="r", status="proposed")  # fmt: skip
    waiting = store.submit_proposal(proposal)
    assert waiting.status == "proposed" and waiting.gates[-1].gate == "human"
    lineage = c.sha256(c.canonical_json(proposal.payload))
    store.commit_approval(c.Approval(id="a1", gate="G1", subject=c.Ref(id="g1", version=1), lineage_hash=lineage,
                                     approver="owen", at=NOW, decision="approved", reason="as drafted", scope="case1"))  # fmt: skip
    assert store.resubmit_after_approval("g1").status == "active"


@pytest.mark.covers("I2")
def test_evidence_is_content_addressed_and_never_overwritten(store):
    v1 = b"Comment text, first retrieval."
    a1 = artifact(v1)
    store.commit_artifact(a1, v1)
    assert store.content(a1.id) == v1
    # Committing the same artifact again is the same commit; committing other bytes or other metadata under its id is
    # refused.
    store.commit_artifact(a1, v1)
    with pytest.raises(Refused):
        store.commit_artifact(a1, b"other bytes")
    with pytest.raises(Conflict):
        store.commit_artifact(a1.model_copy(update={"source": "elsewhere"}), v1)
    # A changed source is a new artifact, linked to its predecessor, which must be committed.
    v2 = b"Comment text, as revised."
    store.commit_artifact(artifact(v2, predecessor=a1.id), v2)
    assert store.get("artifact", artifact_id(v2), c.Artifact).predecessor == a1.id
    with pytest.raises(Refused):
        store.commit_artifact(artifact(b"x", predecessor=artifact_id(b"never committed")), b"x")
    # A blob altered on disk is caught, not trusted.
    blob = store.root / "blobs" / a1.id[4:6] / a1.id[4:]
    blob.write_bytes(b"tampered")
    with pytest.raises(Conflict):
        store.commit_artifact(a1, v1)


def test_a_claim_is_its_artifacts_text_at_its_span(store):
    body = b"Our costs rose 40% under the proposed fee."
    a = artifact(body)
    store.commit_artifact(a, body)
    store.commit_extraction(a.id, "plain-v1", body.decode())
    span, text = (4, 25), body.decode()[4:25]
    store.commit_claim(c.Claim(id=claim_id(a.id, "plain-v1", span, text), artifact=a.id, extraction="plain-v1",
                               span=span, text=text, authority="a commenter"))  # fmt: skip
    wrong = "costs fell 40% under t"[: span[1] - span[0]]
    with pytest.raises(Refused):
        store.commit_claim(c.Claim(id=claim_id(a.id, "plain-v1", span, wrong), artifact=a.id, extraction="plain-v1",
                                   span=span, text=wrong, authority="x"))  # fmt: skip


def test_real_and_synthetic_cases_hold_only_their_own_evidence(store, tmp_path):
    body = b"scenario packet"
    syn = c.SyntheticArtifact(id=synthetic_artifact_id(body), scenario="s1", generator="run-3", media_type="text/plain",
                              size=len(body), created_at=NOW)  # fmt: skip
    with pytest.raises(Refused):
        store.commit_synthetic_artifact(syn, body)
    fuzz = Store(tmp_path / "fuzz", case="fuzz1", synthetic=True, clock=lambda: NOW)
    fuzz.commit_synthetic_artifact(syn, body)
    with pytest.raises(Refused):
        fuzz.commit_artifact(artifact(b"real"), b"real")
    # A store's root holds one case, of one kind.
    with pytest.raises(Refused):
        Store(tmp_path / "fuzz", case="fuzz1", synthetic=False)
    fuzz.close()


@pytest.mark.covers("WP14.acceptance_record")
def test_an_acceptance_record_is_immutable_and_written_at_decision_time(tmp_path):
    record = c.AcceptanceRecord(
        case="case1", decided_at=NOW, objective=c.Ref(id="d", version=1), decision="buy", chosen="y",
        alternatives=["none"], assessed_domain="scenarios v1", price_policy=c.Ref(id="indicative", version=1),
        scenario_probabilities={"s1": F(1)}, accepted_uncovered=[], outstanding_uncertainties=[], budget=D("20"),
        approvals=["a1"], approval_scope="case1",
    )  # fmt: skip
    at_decision = Store(tmp_path / "a", case="case1", clock=lambda: NOW + timedelta(seconds=30))
    at_decision.commit_acceptance(record)
    with pytest.raises(Conflict):
        at_decision.commit_acceptance(record.model_copy(update={"chosen": "none"}))
    with pytest.raises(Exception, match="frozen"):
        record.chosen = "none"  # type: ignore[misc]
    later = Store(tmp_path / "b", case="case1", clock=lambda: NOW + timedelta(days=2))
    with pytest.raises(Refused, match="when the decision is made"):
        later.commit_acceptance(record)


def test_only_the_store_opens_the_database():
    # Every write reaches committed state through requisite/store (spec I1): no other module imports sqlite3.
    offenders = []
    for path in (ROOT / "requisite").rglob("*.py"):
        if "store" in path.relative_to(ROOT / "requisite").parts[:1]:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) else [])  # fmt: skip
            if any(n.split(".")[0] == "sqlite3" for n in names):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
