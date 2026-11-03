"""Proposals, gates and human records (spec §3 I1's gate table, §11)."""

import typing
from decimal import Decimal

import pytest
from core_examples import REF, T0
from pydantic import ValidationError

from requisite import core as c
from requisite.core.governance import ProposalKind


def test_every_proposal_kind_has_its_required_gates():
    kinds = set(typing.get_args(ProposalKind))
    assert set(c.REQUIRED_GATES) == kinds
    assert all(gates[0] == "schema" for gates in c.REQUIRED_GATES.values())
    assert "human" in c.required_gates("decision_spec", {})  # G1
    assert c.required_gates("question", {"replaces": "q.parent"}) == ("schema", "stability", "relevance")


@pytest.mark.covers("WP14.override_reason")
def test_an_override_without_a_reason_is_rejected():
    fields = dict(subject=REF, field="application_state", old="review", new="yes", approver="owen", at=T0)
    for reason in ("", "   "):
        with pytest.raises(ValidationError, match="without a reason"):
            c.Override(**fields, reason=reason)
    assert c.Override(**fields, reason="read the order").reason


def test_a_proposal_cannot_claim_a_status_its_gates_did_not_give():
    base = dict(id="p", case="c", kind="question", payload={"id": "q"}, author="run", rationale="r")
    ok = c.GateResult(gate="schema", passed=True, detail="")
    relevant = c.GateResult(gate="relevance", passed=True, detail="")
    with pytest.raises(ValidationError, match="passed every required gate"):
        c.Proposal(**base, status="active")
    with pytest.raises(ValidationError, match="passed every required gate"):
        c.Proposal(**base, status="active", gates=[ok])
    assert c.Proposal(**base, status="active", gates=[ok, relevant]).status == "active"
    with pytest.raises(ValidationError, match="in order"):
        c.Proposal(**base, status="rejected", gates=[c.GateResult(gate="relevance", passed=False, detail="")])
    with pytest.raises(ValidationError, match="ends at the gate that failed"):
        c.Proposal(**base, status="rejected", gates=[ok])


def test_a_witness_shows_divergent_actions_or_a_material_value_difference():
    fields = dict(question=REF, decision=REF, tolerance=Decimal("10"), shared_answer="no", states=["none covered", "half covered"], nodes=["n"])
    c.RefinementWitness(**fields, acceptable_actions=[["buy"], ["buy"]], value_discrepancy=Decimal("90"))
    with pytest.raises(ValidationError, match="materially"):
        c.RefinementWitness(**fields, acceptable_actions=[["buy"], ["buy"]], value_discrepancy=None)
