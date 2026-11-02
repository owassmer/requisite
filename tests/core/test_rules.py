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
    # What the prose pays stands only on terms read as stated, and an unread payment is None, never an empty list that
    # says it pays nothing (external evaluation and definition review, November 2).
    scenario = one(c.FuzzScenario).model_dump()  # YES, paying 100 at T1 under the IR and the prose alike
    with pytest.raises(ValidationError, match="no discrepancy computed"):
        c.FuzzScenario(**{**scenario, "prose_transfers": None})
    unread = c.FuzzScenario(**{**scenario, "prose_transfers": None, "amount_error": None,
                                "liquidity_discrepancy": None}).model_dump()  # fmt: skip
    base = one(c.TermReading).model_dump()

    def read(term: str, as_: str) -> dict:
        others = [a for a in ("states", "states_otherwise", "not_stated") if a != as_]
        return {**base, "term": term, "read": as_,
                "answers": {as_: 0.9, others[0]: 0.05, others[1]: 0.05}}  # fmt: skip

    def terms(**reads: str) -> list[dict]:
        return [read(t, r) for t, r in reads.items()]

    # A reading is its answers' top: tied or empty answers read as ambiguous, never another answer.
    with pytest.raises(ValidationError, match="pooled answers' top"):
        c.TermReading(**{**read("payout_yes", "states"), "read": "states_otherwise"})
    assert c.TermReading(**{**base, "answers": {}, "read": "ambiguous"}).read == "ambiguous"
    with pytest.raises(ValidationError, match="among"):
        c.TermReading(**{**base, "answers": {"states": 2.0, "invented": -1.0}})
    with pytest.raises(ValidationError, match="a distribution"):
        c.TermReading(**{**base, "answers": {"states": 2.0, "not_stated": -1.0}})
    paid = {**report, "scenarios": [scenario], "equivalence": "tested_equivalent_on_suite"}
    stated = terms(payout_yes="states", payout_no="states", payment_date="states")
    assert c.FuzzReport(**{**paid, "terms": stated}).equivalence == "tested_equivalent_on_suite"
    with pytest.raises(ValidationError, match="rest on terms not read as stated"):
        c.FuzzReport(**{**paid, "terms": terms(payout_yes="states_otherwise", payment_date="states")})
    with pytest.raises(ValidationError, match="each payment term is read once"):
        c.FuzzReport(**{**paid, "terms": [*stated, read("payout_yes", "states")]})
    # A payment may be unestablished though its terms were read, as where its amount is not yet determined (A05 §5.3).
    assert c.FuzzReport(**{**paid, "terms": stated, "scenarios": [unread],
                           "equivalence": "unsupported_or_unassessed"}).terms  # fmt: skip
    # A term read as stated otherwise leaves the payment unestablished, never a counterexample: the whole rulebook's
    # reading cannot say this scenario pays differently, since the prose may state the term otherwise only where it does
    # not apply here (an exception, a rate on a zero quantity, another split of the lags).
    differ = terms(payout_yes="states_otherwise", payout_no="states", payment_date="states")
    with pytest.raises(ValidationError, match="needs what the prose pays read on every scenario"):
        c.FuzzReport(**{**paid, "terms": differ, "scenarios": [unread]})
    with pytest.raises(ValidationError, match="a counterexample is a scenario on which prose and IR disagree"):
        c.FuzzReport(**{**paid, "terms": differ, "scenarios": [unread], "equivalence": "counterexample_found"})
    assert c.FuzzReport(**{**paid, "terms": differ, "scenarios": [unread],
                           "equivalence": "unsupported_or_unassessed"}).terms  # fmt: skip
    # A payment unread for want of a reading (not stated, or none) is neither: no equivalence, no counterexample.
    unstated = terms(payout_yes="not_stated", payment_date="states")
    with pytest.raises(ValidationError, match="needs what the prose pays read on every scenario"):
        c.FuzzReport(**{**paid, "terms": unstated, "scenarios": [unread]})
    with pytest.raises(ValidationError, match="a counterexample is a scenario on which prose and IR disagree"):
        c.FuzzReport(**{**paid, "terms": unstated, "scenarios": [unread], "equivalence": "counterexample_found"})
    # Paying nothing rests on its own term: a NO that pays nothing is established only where the prose was read to state
    # what NO pays. A term of another outcome does not bear on it: what YES pays, or when anything is paid, read as
    # stated otherwise, is no disagreement on a NO scenario that pays nothing.
    nothing = {**scenario, "ir_outcome": "resolved_no", "prose_outcome": "resolved_no", "ir_transfers": [],
               "prose_transfers": [], "amount_error": None, "liquidity_discrepancy": None}  # fmt: skip
    with pytest.raises(ValidationError, match="rest on terms not read as stated: \\['payout_no'\\]"):
        c.FuzzReport(**{**paid, "terms": terms(payout_yes="states"), "scenarios": [nothing]})
    others_differ = terms(payout_yes="states_otherwise", payout_no="states", payment_date="states_otherwise")
    assert c.FuzzReport(**{**paid, "terms": others_differ, "scenarios": [nothing]}).equivalence == "tested_equivalent_on_suite"
    no_unread = {**nothing, "prose_transfers": None}
    assert c.FuzzReport(**{**paid, "terms": others_differ, "scenarios": [no_unread],
                           "equivalence": "unsupported_or_unassessed"}).terms  # fmt: skip
    void = {**nothing, "ir_outcome": "void", "prose_outcome": "void"}
    with pytest.raises(ValidationError, match="rest on terms not read as stated: \\['void_refund'\\]"):
        c.FuzzReport(**{**paid, "terms": stated, "scenarios": [void]})
    with pytest.raises(ValidationError, match="determines no payment"):
        c.FuzzScenario(**{**nothing, "prose_outcome": "ambiguous"})
    # A pending or extended outcome pays nothing yet only as the payment date states it.
    pending = {**nothing, "ir_outcome": "pending", "prose_outcome": "pending"}
    with pytest.raises(ValidationError, match="rest on terms not read as stated: \\['payment_date'\\]"):
        c.FuzzReport(**{**paid, "terms": terms(payout_yes="states"), "scenarios": [pending]})
    assert c.FuzzReport(**{**paid, "terms": stated, "scenarios": [pending]}).equivalence == "tested_equivalent_on_suite"
    # A boundary point is infeasible only with its proof, and an unconstructed one says what was tried.
    with pytest.raises(ValidationError, match="says why"):
        c.BoundaryPoint(condition="z_ge_501", point="501", status="unconstructed")
    with pytest.raises(ValidationError, match="realizes only a feasible point"):
        c.BoundaryPoint(condition="z_ge_501", point="501", status="infeasible", excluded_by=["k1"], scenario="b01")


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


def test_a_frame_is_prospective_only_when_retrieved_by_its_as_of():
    from datetime import timedelta

    frame = one(c.Frame).model_dump()
    late = frame["as_of"] + timedelta(minutes=5)
    with pytest.raises(ValidationError, match="retrospective reconstruction"):
        c.Frame(**{**frame, "retrieval_ended": late})
    assert c.Frame(**{**frame, "retrieval_ended": late, "mode": "retrospective_reconstruction"})


def test_a_parameter_is_sourced_or_assumed_and_a_register_row_is_traced():
    from core_examples import CLM, H

    with pytest.raises(ValidationError, match="exactly one"):
        c.Parameter(name="p", value=D("1"), unit="x")
    with pytest.raises(ValidationError, match="exactly one"):
        c.Parameter(name="p", value=D("1"), unit="x", claims=[CLM],
                    assumption=c.Assumption(text="t", binds={}, reason="r"))  # fmt: skip
    row = one(c.RegisterRow).model_dump()
    for empty in ("claims", "judgments", "nodes", "diagnostics"):
        with pytest.raises(ValidationError, match="traces|diagnostics"):
            c.RegisterRow(**{**row, empty: []})
    with pytest.raises(ValidationError, match="interval"):
        c.RegisterRow(**{**row, "expected_impact": (F(5), F(1))})
    liq = one(c.LiquidityProfile).model_dump()
    two = {**liq["seasonality"][0], "value": "2"}
    with pytest.raises(ValidationError, match="averaging exactly one"):
        c.LiquidityProfile(**{**liq, "seasonality": [*liq["seasonality"][:11], two]})
    # Exactly: a sum beyond twelve by less than Decimal's default precision is still beyond twelve.
    hair = {**liq["seasonality"][0], "value": "1.00000000000000000000000000001"}
    with pytest.raises(ValidationError, match="averaging exactly one"):
        c.LiquidityProfile(**{**liq, "seasonality": [*liq["seasonality"][:11], hair]})
    assert H


def test_informational_is_scoped_to_its_decision_and_tolerance():
    d = one(c.QuestionDiagnostic).model_dump()
    with pytest.raises(ValidationError, match="tolerance"):
        c.QuestionDiagnostic(**{**d, "informational": True})
    assert c.QuestionDiagnostic(**{**d, "regret": F(5), "informational": True})
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        c.QuestionDiagnostic(**{**d, "s_prob": 1.2})
    with pytest.raises(ValidationError, match="non-negative"):
        c.QuestionDiagnostic(**{**d, "regret": F(-1), "tolerance": D("0"), "informational": True})


def test_a_kept_docket_has_its_comment_count():
    d = one(c.FrameDocket).model_dump()
    with pytest.raises(ValidationError, match="kept docket"):
        c.FrameDocket(**{**d, "comments": None})
    assert c.FrameDocket(**{**d, "kept": False, "comments": None}).comments is None


def test_a_familys_market_irs_differ_from_its_representative_only_at_their_bound_values():
    family = one(c.ContractFamily)
    rep = family.representative
    assert rep is not None
    trigger = rep.trigger.model_copy(update={"value": rep.trigger.value + 1})  # the bound threshold, another market's
    other = rep.model_copy(update={"id": "KXWET-26DEC31-T21", "trigger": trigger})
    assert c.ContractFamily(**{**dict(family), "markets": [rep, other]}).markets[1] == other
    policy = rep.resolution.model_copy(update={"on_unknown_at_expiry": "yes"})  # no binding varies the procedure
    stray = rep.model_copy(update={"id": "KXWET-26DEC31-T22", "resolution": policy})
    with pytest.raises(ValidationError, match="beyond the values their bindings fill"):
        c.ContractFamily(**{**dict(family), "markets": [rep, stray]})


def test_an_interpretive_parameter_is_compared_even_where_it_looks_like_a_bound_node():
    family = one(c.ContractFamily)
    rep = family.representative
    assert rep is not None

    def with_parameter(value: str, ir_id: str) -> c.ContractIR:
        q = c.Interpretive(id="q", question_template="t", bindings={"p": {"id": rep.trigger.id, "value": value}})
        return rep.model_copy(update={"id": ir_id, "trigger": c.And(id="all", args=[rep.trigger, q])})

    a, b = with_parameter("A", rep.id), with_parameter("B", "KXWET-26DEC31-T23")
    with pytest.raises(ValidationError, match="beyond the values their bindings fill"):
        c.ContractFamily(**{**dict(family), "representative": a, "markets": [a, b]})


def test_a_rulebooks_sections_are_spec_8_5s_each_once_in_its_order():
    rb = one(c.Rulebook)
    sections = list(rb.sections)
    with pytest.raises(ValidationError, match="in its order"):
        c.Rulebook(contract=rb.contract, sections=[sections[2], *sections[:2], *sections[3:]])
    with pytest.raises(ValidationError, match="each once"):
        c.Rulebook(contract=rb.contract, sections=[*sections, sections[-1]])
    with pytest.raises(ValidationError, match=r"lacks \['timing'\]"):
        c.Rulebook(contract=rb.contract, sections=[x for x in sections if x.name != "timing"])
    assert c.Rulebook(contract=rb.contract, sections=[x for x in sections if x.name != "examples"])
    drafted_example = sections[3].model_copy(update={"text": "If it rains 0.2 inches, it pays."})
    with pytest.raises(ValidationError, match="never drafted"):
        c.Rulebook(contract=rb.contract, sections=[*sections[:3], drafted_example, *sections[4:]])
    with pytest.raises(ValidationError, match="holds its documents"):
        c.ScenarioPacket(**{**one(c.ScenarioPacket).model_dump(), "documents": []})


def test_a_runs_families_partition_its_dockets_and_its_estimate_reads_the_completions_hull():
    fam = one(c.EventFamilies)
    twice = fam.families[0].model_copy(update={"id": "other"})
    with pytest.raises(ValidationError, match="one family only"):
        c.EventFamilies(frame=fam.frame, families=[fam.families[0], twice])
    scope = one(c.ScopeEstimate)
    empty = scope.lowest.model_copy(update={"interval": []})
    with pytest.raises(ValidationError, match="None exactly where either set is empty"):
        c.ScopeEstimate(**{**dict(scope), "lowest": empty})
    assert c.ScopeEstimate(**{**dict(scope), "lowest": empty, "read": None}).read is None
    with pytest.raises(ValidationError, match="runs from the lowest"):
        c.ScopeEstimate(**{**dict(scope), "read": (0.0, 0.0)})
    leads = one(c.LeadList)
    with pytest.raises(ValidationError, match="ranked 1, 2"):
        c.LeadList(estimate=leads.estimate, leads=[leads.leads[0].model_copy(update={"rank": 2})])
