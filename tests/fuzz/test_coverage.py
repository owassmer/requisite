"""Feasible MC/DC coverage and boundary values (requisite/fuzz/coverage.py; spec §8.6 step 1; addendum A09 §10.2-10.3).

The kernel's account is checked against a brute-force enumeration written here, independent of it: its own evaluator,
its own feasible states, its own search over every subset of the redundant conditions.
"""

import itertools
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from requisite.fuzz.coverage import (
    ENUMERATE_UP_TO,
    And,
    Exhaustive,
    Not,
    Or,
    Scenario,
    Unsat,
    Var,
    _minimal_sets,
    count_points,
    coverage,
    cutoff_points,
    meets,
    threshold_points,
    truth,
)

A, B, C = Var("A"), Var("B"), Var("C")


def implies(p, q):
    return Or(Not(p), q)


def iff(p, q):
    return And(implies(p, q), implies(q, p))


BOTH = pytest.mark.parametrize("up_to", [0, 64], ids=["sat", "enumeration"])


# ---------------------------------------------------------------- the brute force the kernel is checked against


def ev(e, s):
    if isinstance(e, Var):
        return s[e.name]
    if isinstance(e, Not):
        return not ev(e.item, s)
    if isinstance(e, And):
        return all(ev(x, s) for x in e.items)
    if isinstance(e, Or):
        return any(ev(x, s) for x in e.items)
    raise TypeError(e)


def names_in(e, out=None):
    out = [] if out is None else out
    if isinstance(e, Var):
        if e.name not in out:
            out.append(e.name)
    elif isinstance(e, Not):
        names_in(e.item, out)
    else:
        for x in e.items:
            names_in(x, out)
    return out


def brute(trigger, constraints):
    names = names_in(trigger)
    states = [dict(zip(names, bits, strict=True)) for bits in itertools.product([False, True], repeat=len(names))]
    feasible = [s for s in states if all(ev(c, s) for c in constraints.values())]
    if not feasible:
        return {"finding": "no_feasible_state"}
    fires = [ev(trigger, s) for s in feasible]
    if len(set(fires)) == 1:
        return {"finding": "always_fires" if fires[0] else "never_fires"}

    def changes(group):
        seen = {}
        for s, f in zip(feasible, fires, strict=True):
            seen.setdefault(tuple(s[c] for c in names if c not in group), set()).add(f)
        return any(len(v) == 2 for v in seen.values())

    witness = {c: changes({c}) for c in names}
    redundant = [c for c in names if not witness[c]]
    changing = [
        frozenset(g) for k in range(len(redundant) + 1) for g in itertools.combinations(redundant, k) if changes(set(g))
    ]
    groups = {g for g in changing if not any(h < g for h in changing)}
    return {"finding": None, "witness": witness, "groups": groups}


def feasible(state, constraints):
    return all(ev(c, dict(state)) for c in constraints.values())


def check_against_brute_force(trigger, constraints, cov):
    expected = brute(trigger, constraints)
    assert set(cov.conditions) == set(names_in(trigger))
    proof = Unsat if cov.method == "sat" else Exhaustive
    if expected["finding"] is not None:
        assert cov.finding is not None and cov.finding.kind == expected["finding"]
        assert isinstance(cov.finding.proof, proof)
        assert cov.accounts == () and cov.groups == ()
        return
    assert cov.finding is None
    assert {a.condition: not a.redundant for a in cov.accounts} == expected["witness"]
    assert {g.conditions for g in cov.groups} == expected["groups"]
    for a in cov.accounts:
        if a.redundant:
            assert isinstance(a.redundancy, proof)
            continue
        # A unique-cause pair: both states feasible, differing in this condition alone, and the outcome flips.
        lo, hi = dict(a.witness.not_fired), dict(a.witness.fired)
        assert feasible(lo.items(), constraints) and feasible(hi.items(), constraints)
        assert [c for c in cov.conditions if lo[c] != hi[c]] == [a.condition]
        assert ev(trigger, lo) is False and ev(trigger, hi) is True
    for g in cov.groups:
        lo, hi = dict(g.pair.not_fired), dict(g.pair.fired)
        assert feasible(lo.items(), constraints) and feasible(hi.items(), constraints)
        assert all(lo[c] == hi[c] for c in cov.conditions if c not in g.conditions)
        assert ev(trigger, lo) is False and ev(trigger, hi) is True


NAMES = [f"c{i}" for i in range(ENUMERATE_UP_TO + 2)]


def exprs(names, max_leaves):
    leaf = st.sampled_from(names).map(Var)
    return st.recursive(
        leaf,
        lambda kids: st.one_of(
            kids.map(Not),
            st.lists(kids, max_size=3).map(lambda xs: And(*xs)),
            st.lists(kids, max_size=3).map(lambda xs: Or(*xs)),
        ),
        max_leaves=max_leaves,
    )


@st.composite
def problems(draw, sizes, every=False):
    """A random trigger over up to n conditions (exactly n when every), with random constraints over its conditions."""
    names = NAMES[: draw(sizes)]
    trigger = draw(exprs(names, 10))
    if every:
        missing = [x for x in names if x not in names_in(trigger)]
        if missing:
            rest = [draw(st.sampled_from([Var(x), Not(Var(x))])) for x in missing]
            inner = draw(st.sampled_from([And, Or]))(*rest)
            trigger = draw(st.sampled_from([And, Or]))(trigger, inner)
    conds = names_in(trigger)
    pieces = exprs(conds, 4) if conds else st.sampled_from([And(), Or()])
    constraints = {f"k{i}": e for i, e in enumerate(draw(st.lists(pieces, max_size=3)))}
    return trigger, constraints


# ---------------------------------------------------------------- the oracle and the amended acceptance test


@pytest.mark.covers("O11")
@BOTH
def test_feasibility_constraints_eliminate_an_independence_witness(up_to):
    # Oracle check 11: F = A AND B with A implies B. A has a feasible witness; B has none, being redundant in A AND B
    # over the feasible states.
    cov = coverage(And(A, B), {"A_implies_B": implies(A, B)}, enumerate_up_to=up_to)
    a, b = cov.accounts
    assert a.condition == "A" and a.witness is not None
    assert b.condition == "B" and b.witness is None


@pytest.mark.covers("A.T15")
@BOTH
def test_no_fabricated_witness_and_infeasible_obligations_recorded(up_to):
    # Addendum §13 row 15: A AND B with A implies B. The pair that would show B's effect needs (A, not B), which the
    # constraint excludes: no witness is fabricated for B, B is recorded as redundant under the constraint with its
    # proof, and B's obligation is recorded as infeasible rather than met or unmet. N + 1 scenarios are not required.
    cov = coverage(And(A, B), {"A_implies_B": implies(A, B)}, enumerate_up_to=up_to)
    assert cov.finding is None
    a, b = cov.accounts
    assert dict(a.witness.not_fired) == {"A": False, "B": True}
    assert dict(a.witness.fired) == {"A": True, "B": True}
    assert b.witness is None
    assert isinstance(b.redundancy, Unsat if up_to == 0 else Exhaustive)
    if up_to:
        # Every state was evaluated: 3 of the 4 are feasible, and the one pair differing only in B has equal outcomes.
        assert b.redundancy == Exhaustive(states=4, feasible=3, pairs=1)
    # Removing B alone leaves the trigger unchanged on the feasible states (it equals A there): no group.
    assert cov.groups == ()

    # Scenarios: the impossible (A, not B) history meets nothing, even beside a state that would pair with it.
    s = [
        Scenario("a_only", {"A": True, "B": False}),
        Scenario("both", {"A": True, "B": True}),
        Scenario("b_only", {"A": False, "B": True}),
    ]
    met = meets(cov, s)
    assert met.infeasible_scenarios == ("a_only",)
    assert met.independence_met == {"A": ("b_only", "both")}
    assert met.independence_unmet == ()
    assert met.independence_infeasible == ("B",)
    # Without b_only, A's obligation is uncovered, and the infeasible scenario does not stand in for it.
    met = meets(cov, s[:2])
    assert met.independence_met == {} and met.independence_unmet == ("A",)


# ---------------------------------------------------------------- redundancy and coupled groups


@BOTH
def test_two_conditions_each_redundant_but_not_removable_together(up_to):
    # A == B under A AND B: each is redundant given the other, and removing both changes the trigger.
    cov = coverage(And(A, B), {"same": iff(A, B)}, enumerate_up_to=up_to)
    assert cov.redundant == ("A", "B")
    (g,) = cov.groups
    assert g.conditions == {"A", "B"}
    assert dict(g.pair.not_fired) == {"A": False, "B": False} and dict(g.pair.fired) == {"A": True, "B": True}


@BOTH
def test_three_equal_conditions_form_one_group_of_three(up_to):
    # A == B == C under A AND B AND C: any two can go, given the third; all three cannot.
    cov = coverage(And(A, B, C), {"ab": iff(A, B), "bc": iff(B, C)}, enumerate_up_to=up_to)
    assert cov.redundant == ("A", "B", "C")
    assert [g.conditions for g in cov.groups] == [frozenset({"A", "B", "C"})]


@BOTH
def test_a_linked_threshold_is_redundant_through_its_constraint(up_to):
    # Rain >= 0.25 implies rain >= 0.10, so in (rain >= 0.10 OR rain >= 0.25) the higher threshold has no effect, and
    # removing it changes nothing: redundant, in no group.
    r10, r25 = Var("rain_ge_0.10"), Var("rain_ge_0.25")
    cov = coverage(Or(r10, r25), {"linked": implies(r25, r10)}, enumerate_up_to=up_to)
    assert cov.redundant == ("rain_ge_0.25",)
    assert cov.groups == ()


@BOTH
def test_a_redundant_condition_is_not_removable_beside_a_coupled_one(up_to):
    # (A AND B) OR C with A == B: A and B form a group; C has a witness.
    cov = coverage(Or(And(A, B), C), {"same": iff(A, B)}, enumerate_up_to=up_to)
    assert cov.redundant == ("A", "B")
    assert [g.conditions for g in cov.groups] == [frozenset({"A", "B"})]
    assert cov.accounts[2].witness is not None


# ---------------------------------------------------------------- findings about the whole trigger


@BOTH
def test_constraints_with_no_feasible_state_are_one_finding_not_removable_conditions(up_to):
    cov = coverage(And(A, B), {"a": A, "not_a": Not(A), "b": B}, enumerate_up_to=up_to)
    assert cov.finding.kind == "no_feasible_state"
    assert cov.finding.excluded_by == (frozenset({"a", "not_a"}),)
    assert cov.accounts == () and cov.groups == () and cov.redundant == ()


@BOTH
def test_a_constant_trigger_is_one_finding_not_removable_conditions(up_to):
    never = coverage(And(A, Not(A), B), enumerate_up_to=up_to)
    assert never.finding.kind == "never_fires" and never.accounts == () and never.groups == ()
    always = coverage(Or(A, B), {"a": A}, enumerate_up_to=up_to)
    assert always.finding.kind == "always_fires" and always.accounts == ()
    assert truth(Or(A, B), dict(always.finding.example)) is True
    assert meets(always, [Scenario("s", {"A": True, "B": False})]).independence_infeasible == ("A", "B")


# ---------------------------------------------------------------- agreement with the brute force


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(problems(st.integers(0, 6)))
def test_every_verdict_agrees_with_brute_force_on_both_paths(problem):
    trigger, constraints = problem
    for up_to in (0, 64):
        cov = coverage(trigger, constraints, enumerate_up_to=up_to)
        assert cov.method == ("sat" if up_to == 0 and cov.conditions else "enumeration")
        check_against_brute_force(trigger, constraints, cov)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(problems(st.sampled_from([ENUMERATE_UP_TO, ENUMERATE_UP_TO + 1]), every=True))
def test_every_verdict_agrees_on_both_sides_of_the_size_where_sat_takes_over(problem):
    trigger, constraints = problem
    cov = coverage(trigger, constraints)  # the default size
    assert cov.method == ("enumeration" if len(cov.conditions) <= ENUMERATE_UP_TO else "sat")
    check_against_brute_force(trigger, constraints, cov)


@given(st.integers(0, 6).flatmap(lambda n: st.lists(st.lists(st.integers(0, n - 1), max_size=n), max_size=5)), st.integers(0, 6))
def test_minimal_sets_are_found_completely(family, n):
    # The predicate "contains some member of the family" is closed under supersets; its minimal sets are the family's
    # minimal members.
    sets = [frozenset(f) for f in family if all(x < n for x in f)]
    expected = {s for s in sets if not any(t < s for t in sets)}
    found = _minimal_sets(tuple(range(n)), lambda s: any(f <= s for f in sets))
    assert set(found) == expected and len(found) == len(expected)


# ---------------------------------------------------------------- boundary values


def test_threshold_points_sit_at_the_declared_precision():
    pts = threshold_points("rain_ge_0.10", ">=", Decimal("0.10"), Decimal("0.01"))
    assert [(p.offset, p.value, p.holds) for p in pts] == [
        (-1, Decimal("0.09"), False),
        (0, Decimal("0.10"), True),
        (1, Decimal("0.11"), True),
    ]
    assert [p.holds for p in threshold_points("x", "==", 5, 1)] == [False, True, False]
    assert [p.holds for p in threshold_points("x", ">", Decimal("100.00"), Decimal("0.01"))] == [False, False, True]
    with pytest.raises(TypeError):
        threshold_points("x", ">=", 0.1, Decimal("0.01"))
    with pytest.raises(ValueError, match="declared precision"):
        threshold_points("x", ">=", Decimal("0.105"), Decimal("0.01"))
    with pytest.raises(ValueError):
        threshold_points("x", ">=", Decimal("1"), Decimal("0"))
    # A domain bound: rain is never negative.
    pts = threshold_points("rain_gt_0", ">", Decimal("0.00"), Decimal("0.01"), domain=(Decimal("0"), None))
    assert [p.in_domain for p in pts] == [False, True, True]


def test_cutoff_points_are_the_exact_instant_and_its_neighbours():
    cutoff = datetime(2027, 6, 30, 23, 59, 59, tzinfo=ZoneInfo("America/New_York"))
    inclusive = cutoff_points("by_cutoff", "<=", cutoff, timedelta(seconds=1))
    exclusive = cutoff_points("by_cutoff", "<", cutoff, timedelta(seconds=1))
    assert [p.value for p in inclusive] == [cutoff - timedelta(seconds=1), cutoff, cutoff + timedelta(seconds=1)]
    assert [p.holds for p in inclusive] == [True, True, False]
    assert [p.holds for p in exclusive] == [True, False, False]
    with pytest.raises(ValueError, match="time zone"):
        cutoff_points("c", "<=", datetime(2027, 6, 30), timedelta(seconds=1))
    days = cutoff_points("on_or_before", "<=", date(2027, 6, 30), timedelta(days=1))
    assert [p.value for p in days] == [date(2027, 6, 29), date(2027, 6, 30), date(2027, 7, 1)]
    with pytest.raises(ValueError, match="whole days"):
        cutoff_points("on_or_before", "<=", date(2027, 6, 30), timedelta(hours=1))


def test_cutoff_neighbours_are_instants_across_a_daylight_saving_change():
    ny = ZoneInfo("America/New_York")
    # Fall back: 01:30 EDT is followed an hour later by 01:30 EST, the same wall-clock time.
    cutoff = datetime(2027, 11, 7, 1, 30, tzinfo=ny)
    pts = cutoff_points("by_cutoff", "<=", cutoff, timedelta(hours=1))
    assert [p.value.astimezone(UTC).hour for p in pts] == [4, 5, 6]
    assert pts[2].value.replace(tzinfo=None) == cutoff.replace(tzinfo=None)  # same wall clock, an hour later
    assert [p.holds for p in pts] == [True, True, False]
    # Spring forward: the instant an hour before 03:00 EDT is 01:00 EST; 02:00 does not exist that day.
    pts = cutoff_points("by_cutoff", "<", datetime(2027, 3, 14, 3, 0, tzinfo=ny), timedelta(hours=1))
    assert pts[0].value.astimezone(UTC) == datetime(2027, 3, 14, 6, 0, tzinfo=UTC)
    assert (pts[0].value.hour, pts[0].value.utcoffset()) == (1, timedelta(hours=-5))


def test_count_points_are_k_minus_one_k_and_k_plus_one_within_the_window():
    pts = count_points("wet_days_ge_10", ">=", 10, 91)
    assert [(p.value, p.holds, p.in_domain) for p in pts] == [(9, False, True), (10, True, True), (11, True, True)]
    assert [p.in_domain for p in count_points("any_wet_day", ">=", 0, 91)] == [False, True, True]
    assert [p.in_domain for p in count_points("every_day_wet", ">=", 91, 91)] == [True, True, False]


@BOTH
def test_an_infeasible_boundary_point_records_every_minimal_set_of_constraints_excluding_it(up_to):
    # A cannot be true: k1 and k2 together exclude it, and so does k3 alone.
    pts = threshold_points("A", ">=", 10, 1)
    constraints = {"k1": implies(A, B), "k2": Not(B), "k3": Not(A)}
    cov = coverage(Or(A, C), constraints, pts, enumerate_up_to=up_to)
    below, at, above = cov.points
    assert below.feasible and dict(below.example)["A"] is False
    assert not at.feasible and at.excluded_by == (frozenset({"k3"}), frozenset({"k1", "k2"}))
    assert not above.feasible and above.excluded_by == at.excluded_by
    # Outside the domain a point is infeasible for that reason alone, with no constraint blamed.
    cov = coverage(Or(A, C), {}, count_points("A", ">=", 0, 5), enumerate_up_to=up_to)
    assert not cov.points[0].feasible and cov.points[0].excluded_by == ()


def test_scenarios_meet_boundary_points_by_their_values():
    r = Var("rain_ge_0.10")
    pts = threshold_points("rain_ge_0.10", ">=", Decimal("0.10"), Decimal("0.01"), domain=(Decimal("0"), None))
    cov = coverage(And(r, B), {}, pts)
    s = [
        Scenario("dry", {"rain_ge_0.10": False, "B": True}, {"rain_ge_0.10": Decimal("0.09")}),
        Scenario("wet", {"rain_ge_0.10": True, "B": True}, {"rain_ge_0.10": Decimal("0.1")}),
    ]
    met = meets(cov, s)
    assert met.points_met == {pts[0]: "dry", pts[1]: "wet"}
    assert met.points_unmet == (pts[2],)
    assert met.independence_met == {"rain_ge_0.10": ("dry", "wet")}
    assert met.independence_unmet == ("B",)
    # A scenario whose value contradicts its own condition is refused.
    with pytest.raises(ValueError, match="gives the opposite"):
        meets(cov, [Scenario("bad", {"rain_ge_0.10": True, "B": True}, {"rain_ge_0.10": Decimal("0.09")})])
    with pytest.raises(ValueError, match="every condition"):
        meets(cov, [Scenario("partial", {"B": True})])


def test_malformed_problems_are_refused():
    with pytest.raises(ValueError, match="not conditions of the trigger"):
        coverage(A, {"k": implies(A, B)})
    with pytest.raises(ValueError, match="two comparisons"):
        coverage(A, {}, threshold_points("A", ">=", 1, 1) + threshold_points("A", "<", 1, 1))
    with pytest.raises(ValueError):
        Var("")
