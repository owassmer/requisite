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
        c.CoverageObligation(conditions=["A"], status="met", witnesses=["s1", "s1"], proof=None)
    with pytest.raises(ValidationError, match="states its proof"):
        c.CoverageObligation(conditions=["A"], status="redundant", witnesses=[], proof=None)
    assert c.CoverageObligation(conditions=["A"], status="uncovered", witnesses=[], proof=None)
    with pytest.raises(ValidationError, match="single condition"):
        c.CoverageObligation(conditions=["A", "B"], status="redundant", witnesses=[], proof="UNSAT")


def test_sampling_sizes_and_scenario_probabilities_are_coherent():
    with pytest.raises(ValidationError, match="part of its phase-1 items"):
        c.SubStratum(name="s", parent="p", rule="r", phase1_count=2, sample_size=3)
    with pytest.raises(ValidationError, match="part of its phase-1 items"):
        c.SubStratum(name="s", parent="p", rule="r", phase1_count=2, sample_size=-1)
    record = one(c.AcceptanceRecord).model_dump()
    with pytest.raises(ValidationError, match="never normalized"):
        c.AcceptanceRecord(**{**record, "scenario_probabilities": {"s1": F(1, 2), "s2": F(1, 3)}})


def test_a_judgment_corrects_only_round_off_and_keeps_what_was_reported():
    base = JUDGMENT.model_dump()
    assert c.Judgment(**{**base, "reported": {"yes": 0.7, "no": 0.2, "ne": 0.1}, "precision": 0.1})
    # Only the correction the definition gives: a distribution elsewhere within the intervals is not it.
    with pytest.raises(ValidationError, match="round-off correction"):
        c.Judgment(**{**base, "reported": {"yes": 0.7, "no": 0.2, "ne": 0.1},
                      "distribution": {"yes": 0.74, "no": 0.16, "ne": 0.1}, "precision": 0.1})  # fmt: skip
    # (0.1, 0.1, 0.9) at step 0.1 sums to 1.1: proportional scaling would move 0.9 to 9/11, outside its interval.
    with pytest.raises(ValidationError, match="round-off correction"):
        c.Judgment(**{**base, "reported": {"yes": 0.9, "no": 0.1, "ne": 0.1},
                      "distribution": {"yes": 9 / 11, "no": 1 / 11, "ne": 1 / 11}, "precision": 0.1})  # fmt: skip
    with pytest.raises(ValidationError, match="correction of the reported"):
        c.Judgment(**{**base, "reported": None})


def test_fuzz_reports_claim_only_what_their_scenarios_support():
    report = one(c.FuzzReport).model_dump()
    with pytest.raises(ValidationError, match="every hard gate"):
        c.FuzzReport(**{**report, "hard_gates": []})
    with pytest.raises(ValidationError, match="over a suite"):
        c.FuzzReport(**{**report, "equivalence": "tested_equivalent_on_suite"})
    with pytest.raises(ValidationError, match="declares its finite domain"):
        c.FuzzReport(**{**report, "equivalence": "exhaustive_equivalent_on_declared_finite_domain"})


def test_every_unit_has_a_positive_chance_of_gold():
    design = one(c.SamplingDesign).model_dump()
    starved = [{**design["phase2"][0], "sample_size": 0}, design["phase2"][1]]
    with pytest.raises(ValidationError, match="at least"):
        c.SamplingDesign(**{**design, "phase2": starved})


def test_round_off_is_decided_exactly_even_at_tiny_steps():
    from requisite.core.judgment import correct_roundoff

    # Two values of 0.5000000000004 at a step of 1e-13: no distribution is within half a step of both.
    assert correct_roundoff([0.5000000000004, 0.5000000000004], 1e-13) is None
    assert correct_roundoff([0.5, 0.5], 1e-13) == [0.5, 0.5]
    # The distribution is the correction itself, exactly: no tolerance wider than a tiny step lets another through.
    base = JUDGMENT.model_dump()
    with pytest.raises(ValidationError, match="round-off correction"):
        c.Judgment(**{**base, "reported": {"yes": 0.5, "no": 0.5, "ne": 0.0},
                      "distribution": {"yes": 0.5000000000004, "no": 0.4999999999996, "ne": 0.0}, "precision": 1e-13})  # fmt: skip
    thirds = correct_roundoff([0.33, 0.33, 0.33], 0.01)
    assert thirds == [1 / 3, 1 / 3, 1 / 3] or abs(sum(thirds) - 1) < 1e-15


def test_a_time_zone_is_a_real_zone():
    with pytest.raises(ValidationError, match="IANA"):
        c.TimeWindow(start=None, end=None, start_closed=True, end_closed=True, tz="Mars/Olympus_Mons")
    timing = one(c.Timing).model_dump()
    with pytest.raises(ValidationError, match="IANA"):
        c.Timing(**{**timing, "tz": "Eastern"})


def test_transfers_agree_as_dated_amounts_in_any_order():
    from datetime import UTC, datetime

    from requisite.core.results import same_transfers

    scenario = one(c.FuzzScenario)
    a = c.CashTransfer(at=datetime(2027, 1, 5, tzinfo=UTC), amount=D("50"), currency="USD", basis="payout")
    b = c.CashTransfer(at=datetime(2027, 2, 5, tzinfo=UTC), amount=D("50.00"), currency="USD", basis="payout")
    fields = scenario.model_dump()
    assert same_transfers(c.FuzzScenario(**{**fields, "ir_transfers": [a, b], "prose_transfers": [b, a]}))
    from datetime import timedelta, timezone

    same_instant = c.CashTransfer(at=datetime(2027, 1, 4, 19, tzinfo=timezone(timedelta(hours=-5))), amount=D("50.0"),
                                  currency="USD", basis="payout")  # fmt: skip
    assert same_transfers(c.FuzzScenario(**{**fields, "ir_transfers": [a], "prose_transfers": [same_instant]}))
    cent = c.CashTransfer(at=a.at, amount=D("50.0000000000000000000000000001"), currency="USD", basis="payout")
    assert not same_transfers(c.FuzzScenario(**{**fields, "ir_transfers": [a], "prose_transfers": [cent]}))
    late = a.model_copy(update={"at": datetime(2027, 1, 6, tzinfo=UTC)})
    assert not same_transfers(c.FuzzScenario(**{**fields, "ir_transfers": [a, b], "prose_transfers": [late, b]}))


def test_a_split_payment_is_the_same_cash():
    from datetime import UTC, datetime

    from requisite.core.results import same_transfers

    fields = one(c.FuzzScenario).model_dump()
    at = datetime(2027, 1, 5, tzinfo=UTC)
    whole = c.CashTransfer(at=at, amount=D("100"), currency="USD", basis="payout")
    half = c.CashTransfer(at=at, amount=D("50"), currency="USD", basis="payout")
    assert same_transfers(c.FuzzScenario(**{**fields, "ir_transfers": [whole], "prose_transfers": [half, half]}))


def test_an_independent_labelled_sample_is_not_nested():
    design = one(c.SamplingDesign).model_dump()
    labelled = {"name": "labelled", "parent": "all", "rule": "srs", "phase1_count": 10, "sample_size": 3}
    independent = {**design, "sample": "independent_two_sample", "phase2": [labelled]}
    assert c.SamplingDesign(**independent)
    # Independence removes the nesting, never the labelled sample: a stratum with items must have one.
    with pytest.raises(ValidationError, match="no labelled sample"):
        c.SamplingDesign(**{**independent, "phase2": [{**labelled, "sample_size": 0}]})
