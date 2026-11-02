"""The definition's rules that a record could otherwise break (spec §2.1, §4, §9; addendum A04, A05, A07, A09)."""

from datetime import date
from decimal import Decimal as D
from fractions import Fraction as F

import pytest
from core_examples import CONTRACT, JUDGMENT, examples
from pydantic import ValidationError

from requisite import core as c


def one(cls):
    return next(x for x in examples() if type(x) is cls)


def test_a_comparison_value_keeps_its_type_through_a_round_trip():
    for value in (c.Number(value=D("10")), c.Category(value="10"), c.Flag(value=False), c.OnDay(value=date(2027, 1, 1))):
        cmp = c.Cmp(id="x", observable="o", op="==", value=value)
        assert c.Cmp.model_validate_json(cmp.model_dump_json()).value == value
    # The value's type is its observable's.
    with pytest.raises(ValidationError, match="compares rain, a number, with a text"):
        c.ContractIR(**{**CONTRACT.model_dump(), "trigger": c.Cmp(id="t", observable="rain", op="==",
                                                                      value=c.Category(value="wet")).model_dump()})  # fmt: skip


def test_a_contested_question_is_held_until_a_human_resolves_it():
    rj = one(c.ResolvedJudgment).model_dump()
    contested = {**rj, "stable": False, "primary_diagnosis": "contested"}
    with pytest.raises(ValidationError, match="held as not established"):
        c.ResolvedJudgment(**{**contested, "evidence_status": "established", "application_state": "yes"})
    assert c.ResolvedJudgment(**{**contested, "evidence_status": "not_established", "application_state": "review"})
    assert c.ResolvedJudgment(**{**contested, "evidence_status": "established", "application_state": "yes",
                                 "approvals": ["g2-approval"]})  # fmt: skip


def test_promotion_needs_a_g4_approval_after_the_required_gates():
    base = dict(id="p", case="c", kind="template", payload={}, author="run", rationale="r")
    schema, tests = (c.GateResult(gate=g, passed=True, detail="") for g in ("schema", "tests"))
    with pytest.raises(ValidationError, match="G4"):
        c.Proposal(**base, status="promoted", gates=[schema, tests])
    g4 = c.GateResult(gate="human", passed=True, detail="G4 approved")
    assert c.Proposal(**base, status="promoted", gates=[schema, tests, g4]).status == "promoted"


def test_mechanical_policies_are_complete():
    with pytest.raises(ValidationError, match="two or more sources"):
        c.ConflictPolicy(procedure="require_agreement")
    with pytest.raises(ValidationError, match="names no sources"):
        c.ConflictPolicy(procedure="precedence_first_available", sources=["a"])
    payout = one(c.Payout).model_dump()
    with pytest.raises(ValidationError):
        c.Payout(**{k: v for k, v in payout.items() if k != "rounding_mode"})


def test_a_met_obligation_is_a_pair_of_two_scenarios():
    with pytest.raises(ValidationError, match="two distinct"):
        c.CoverageObligation(condition="A", status="met", witnesses=["s1", "s1"], proof=None)
    with pytest.raises(ValidationError, match="states its proof"):
        c.CoverageObligation(condition="A", status="redundant", witnesses=[], proof=None)
    assert c.CoverageObligation(condition="A", status="uncovered", witnesses=[], proof=None)


def test_sampling_sizes_and_scenario_probabilities_are_coherent():
    with pytest.raises(ValidationError, match="part of its phase-1 items"):
        c.SubStratum(name="s", parent="p", rule="r", phase1_count=2, sample_size=3)
    with pytest.raises(ValidationError, match="part of its phase-1 items"):
        c.SubStratum(name="s", parent="p", rule="r", phase1_count=2, sample_size=-1)
    record = one(c.AcceptanceRecord).model_dump()
    with pytest.raises(ValidationError, match="never normalized"):
        c.AcceptanceRecord(**{**record, "scenario_probabilities": {"s1": F(1, 2), "s2": F(1, 3)}})


def test_a_judgment_corrects_only_round_off_and_records_it():
    base = JUDGMENT.model_dump()
    assert c.Judgment(**{**base, "distribution": {"yes": 0.7, "no": 0.2, "ne": 0.1}, "roundoff": 0.00015,
                         "precision": 0.0001})  # fmt: skip
    with pytest.raises(ValidationError, match="half a reporting step"):
        c.Judgment(**{**base, "roundoff": 0.01, "precision": 0.0001})
