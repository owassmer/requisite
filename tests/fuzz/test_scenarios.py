"""What a fuzz run tests, before any model is asked (requisite/fuzz/scenarios.py; spec §8.6 steps 1-3; addendum A09).

The planner's claims are checked against the evaluator and a brute force written here: every scenario's outcome is the
evaluator's on its own inputs (I8); every coverage or boundary scenario's inputs mean its state (each condition evaluated
alone); every obligation met is met by two of its scenarios that differ in that condition alone and on which the
evaluator's trigger differs; the coverage scenarios are as few as any set meeting the obligations can be; a boundary
point given a scenario is realized there; a point or obligation it cannot meet is recorded with why."""

import itertools
from datetime import timedelta
from decimal import Decimal as D

import western_grazers as wg
from hypothesis import HealthCheck, given, settings
from irgen import contracts
from irkit import EXPIRY, T0, at, contract, num, observable, procedure

from requisite import core as c
from requisite.contract.evaluate import evaluate
from requisite.core.kleene import Kleene
from requisite.fuzz import coverage as cov
from requisite.fuzz.run import determined, forced, total
from requisite.fuzz.scenarios import EXACT_UP_TO, STRESS, plan

W = T0 + timedelta(days=5)


def truths(s) -> dict[str, str]:
    return {k: v.value for k, v in s.truths.items()}


def check(p) -> None:
    """The planner's claims on one plan (see the module docstring)."""
    by = {s.id: s for s in p.scenarios}
    for s in p.scenarios:
        assert s.evaluation == evaluate(p.ir, s.observations, s.interpretations, at=p.at), s.id
        assert {o.artifact for o in s.observations} <= {s.artifact}
        assert c.evidence.synthetic_artifact_id(s.content) == s.artifact
    for o in p.obligations:
        if o.status == "met":
            a, b = (by[w] for w in o.witnesses)
            outside = [k for k in p.lowered.conditions if k not in o.conditions]
            assert all(a.truths[k] is b.truths[k] for k in outside), o
            if len(o.conditions) == 1:
                assert a.truths[o.conditions[0]] is not b.truths[o.conditions[0]]
            assert {a.evaluation.trigger_truth, b.evaluation.trigger_truth} == {Kleene.TRUE, Kleene.FALSE}
        if o.status == "redundant":
            assert o.proof
    for b in p.boundaries:
        if b.scenario is not None:
            assert b.feasible and any(pin.shown() == b.point for pin in by[b.scenario].pins)
        if not b.feasible:
            assert b.excluded_by, b


def minimum(p) -> int:
    """The fewest feasible states meeting every met obligation, by brute force over sets of feasible states."""
    conds = list(p.account.conditions)
    with cov.Feasible(p.account) as feas:
        states = [dict(s) for s, _ in feas.states()]
    fires = [cov.truth(p.lowered.trigger, s) for s in states]
    needs = []
    for o in p.obligations:
        if o.status != "met":
            continue
        g = set(o.conditions)
        pairs = [(i, j) for i, j in itertools.permutations(range(len(states)), 2) if not fires[i] and fires[j]
                 and all(states[i][k] == states[j][k] for k in conds if k not in g)
                 and (len(g) > 1 or states[i][o.conditions[0]] != states[j][o.conditions[0]])]  # fmt: skip
        needs.append(pairs)
    for n in range(len(states) + 1):
        for chosen in itertools.combinations(range(len(states)), n):
            pick = set(chosen)
            if all(any(i in pick and j in pick for i, j in pairs) for pairs in needs):
                return n
    raise AssertionError("no set of feasible states meets the obligations")


def test_western_grazers_is_covered_by_its_fewest_scenarios_each_meaning_its_state():
    p = plan(wg.ir(wg.AB187), reference=wg.rates())
    check(p)
    coverage = [s for s in p.scenarios if s.role == "coverage"]
    # Eight conditions, each with an independent effect under the IR's constraints: N + 1 scenarios meet them all (the
    # six interpretive conditions and the cost test conjoined, the cost test or "enjoins entirely").
    assert len(p.lowered.conditions) == 8 and len(coverage) == 9
    assert {o.status for o in p.obligations} == {"met"} and len(p.obligations) == 8
    yes = [s for s in coverage if s.evaluation.procedure_status == "resolved_yes"]
    assert len(yes) == 2 and all(s.evaluation.unresolved for s in yes)  # its early date is Rule 7.2's, named
    # Its cost test is computed from readings answered as bands, so no scenario can put the cost exactly at the
    # threshold: each boundary point is infeasible, with that reason, never a scenario of a value no history has.
    assert len(p.boundaries) == 3
    assert all(not b.feasible and "answered as bands" in b.excluded_by[0] for b in p.boundaries)
    # The stress classes the IR bears: revised published rates; the interpretive presentations; evidence missing. Its
    # rates have one source, so no conflicting sources.
    stressed = {s.stress for s in p.scenarios if s.role == "stress"}
    assert stressed == set(STRESS) - {"conflicting_sources"}
    # The cost test's inputs are the reference rates, as published; readings the walk left nearest the threshold.
    rates = {(o.observable, o.value.value) for o in wg.rates()}
    assert {(f.observable, f.value.value) for s in coverage for f in s.facts} == rates
    assert not p.notes


@settings(max_examples=60, deadline=None, suppress_health_check=list(HealthCheck), derandomize=True)
@given(contracts(decidable=True))
def test_every_plan_meets_its_claims_on_random_contracts(ir):
    p = plan(ir)
    check(p)
    coverage = [s for s in p.scenarios if s.role == "coverage"]
    if p.whole_trigger == "ok" and not p.notes and len(p.account.conditions) <= min(EXACT_UP_TO, 5):
        assert len(coverage) == minimum(p)
    for o in p.obligations:  # an obligation left unmet says why
        assert o.status != "uncovered" or any(o.conditions[0] in n for n in p.notes)


def test_a_linked_threshold_is_reported_redundant_and_no_witness_is_fabricated():
    linked = contract(c.And(id="both", args=[at("x", ">", 10, W, nid="above10"), at("x", ">", 5, W, nid="above5")]))
    p = plan(linked)
    check(p)
    status = {o.conditions[0]: o for o in p.obligations}
    assert status["above10"].status == "met" and status["above5"].status == "redundant"
    assert "exhaustive" in (status["above5"].proof or "")
    # No scenario has x above 10 without x above 5: every one means a state a history has.
    assert all(not (s.truths["above10"] is Kleene.TRUE and s.truths["above5"] is Kleene.FALSE) for s in p.scenarios)


def test_boundary_values_are_realized_exactly_where_a_value_can_sit():
    p = plan(contract(at("x", ">", 10, W, nid="A")))
    check(p)
    values = {b.point.rsplit(": ", 1)[1]: b.scenario for b in p.boundaries}
    assert set(values) == {"9.99", "10.00", "10.01"} and all(values.values())
    by = {s.id: s for s in p.scenarios}
    for v, sid in values.items():
        assert [f.value.value for f in by[sid].facts if f.at == W] == [D(v)]


def test_a_boundary_reached_only_by_moving_two_inputs_is_built_and_one_no_search_builds_is_unconstructed():
    # z = 2x + 3y >= 501 over integer-precision inputs. From any reference point, moving x alone steps z by 2 and y alone
    # by 3, and the parity of the reference decides whether either reaches z = 501; moving both does (x = 99, y = 101).
    # A point the search builds is feasible with its scenario; a point it cannot build is unconstructed, counted unmet,
    # never infeasible: only a proof makes a point infeasible (external evaluation, November 2).
    whole = {"precision": D("1")}
    ir = contract(c.NumCmp(id="z", op=">=", value=D(501), expr=c.Computed(
        id="f", node=c.Ref(id="tests.contract.weighted", version=1),
        inputs={"x": c.ValueAt(id="vx", observable="x", at=W), "y": c.ValueAt(id="vy", observable="y", at=W)})),
        observables=[observable("x", **whole), observable("y", **whole)])  # fmt: skip
    p = plan(ir)
    check(p)
    by = {s.id: s for s in p.scenarios}
    for b in p.boundaries:
        assert b.status in ("feasible", "unconstructed"), b  # no proof of infeasibility exists here
        if b.status == "feasible" and b.scenario is not None:
            facts = {f.observable: f.value.value for f in by[b.scenario].facts}
            assert 2 * facts["x"] + 3 * facts["y"] == D(b.point.rsplit(": ", 1)[1]), b
    at_501 = [b for b in p.boundaries if b.point.endswith(": 501")]
    assert at_501 and all(b.status == "feasible" and b.scenario for b in at_501)


def test_stress_classes_follow_the_contracts_own_policies():
    two = observable("x", sources=("agency", "backup"))
    precedence = contract(at("x", ">", 10, W, nid="A"), observables=[two], resolution=procedure(precedence=["agency", "backup"]))
    agreement = contract(at("x", ">", 10, W, nid="A"), observables=[two], resolution=procedure(
        precedence=["agency", "backup"], conflict_policy=c.ConflictPolicy(procedure="require_agreement",
                                                                          sources=["agency", "backup"])))  # fmt: skip
    for ir, conflict in ((precedence, "kept"), (agreement, "disputed")):
        p = plan(ir)
        check(p)
        base = {s.id: s for s in p.scenarios}
        for s in p.scenarios:
            if s.stress == "conflicting_sources":
                assert len({f.source for f in s.facts}) == 2
                if conflict == "kept":  # the higher-ranked source settles
                    assert truths(s) == truths(base[s.base]) and s.evaluation.procedure_status == base[s.base].evaluation.procedure_status
                else:  # the sources that must agree disagree
                    assert s.truths["A"] is Kleene.UNKNOWN and s.evaluation.procedure_status == "disputed"
    for policy in ("first_print", "as_of_expiry"):
        p = plan(contract(at("x", ">", 10, W, nid="A"), observables=[observable("x", policy=policy)]))
        revised = [s for s in p.scenarios if s.stress == "revised_value"]
        assert revised
        base = {s.id: s for s in p.scenarios}
        for s in revised:  # two vintages of the deciding value; the policy selects the base's
            assert len({f.published_at for f in s.facts if f.at == W}) == 2
            assert truths(s) == truths(base[s.base])
    p = plan(contract(at("x", ">", 10, W, nid="A"), resolution=procedure(on_unknown_at_expiry="void")))
    missing = [s for s in p.scenarios if s.stress == "evidence_missing_at_expiry"]
    assert missing and all(s.truths["A"] is Kleene.UNKNOWN and s.evaluation.procedure_status == "void" for s in missing)
    assert all(s.withdrawn == "A" for s in missing)


def test_a_forced_outcome_is_the_evaluators_on_the_scenarios_own_inputs():
    count = c.CountTrue(id="wet", expr=c.Cmp(id="rain", observable="x", op=">=", value=num("0.10")),
                        days=c.TimeWindow(start=T0, end=T0 + timedelta(days=10), start_closed=True, end_closed=False))  # fmt: skip
    payout = c.Payout(kind="per_unit", unit_value=D("10.00"), quantity=count, currency="USD", rounding_quantum=D("0.01"),
                      rounding_mode="half_even", payment="at_settlement")  # fmt: skip
    ir = contract(c.NumCmp(id="days", expr=count, op=">=", value=D(3)), payout=payout)
    p = plan(ir)
    check(p)
    for s in p.scenarios:
        yes = forced(ir, Kleene.TRUE, s.observations, s.interpretations, p.at)
        no = forced(ir, Kleene.FALSE, s.observations, s.interpretations, p.at)
        assert yes.procedure_status == "resolved_yes" and no.procedure_status == "resolved_no"
        assert determined(no, ir) and total(no) == 0
        # The quantity is the scenario's own count, as the evaluator computes it for the IR's outcome; where its
        # evidence is withdrawn the count is an interval, and a YES pays no determined amount (A05 §5.3).
        assert yes.quantity == s.evaluation.quantity
        assert determined(yes, ir) == (yes.quantity is not None and yes.quantity[0] == yes.quantity[1])
        assert determined(yes, ir) == (s.stress != "evidence_missing_at_expiry")
        if s.evaluation.procedure_status == "resolved_yes":
            assert total(yes) == total(s.evaluation)
            assert [t.at for t in yes.transfers] == [t.at for t in s.evaluation.transfers]
    assert EXPIRY == ir.timing.expiration
