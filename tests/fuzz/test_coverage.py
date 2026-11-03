"""Feasible MC/DC coverage and boundary values (requisite/fuzz/coverage.py; spec §8.6 step 1; addendum A09 §10.2-10.3).

The kernel's account is checked against a brute-force enumeration written here, independent of it: its own evaluator,
its own feasible states (a state is feasible when some values of the conditions only the constraints mention satisfy
every constraint), its own search over every subset of the redundant conditions and of the constraints. Every feasible
state the kernel shows is checked by evaluating the constraints on it and its completion.
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


def others_in(trigger, constraints):
    names, out = names_in(trigger), []
    for c in constraints.values():
        out = [o for o in names_in(c, out) if o not in names]
    return out


def holds_everywhere(constraints, full):
    return all(ev(c, full) for c in constraints.values())


def feasible(state, trigger, constraints):
    """Some values of the conditions only the constraints mention satisfy every constraint in this state."""
    others = others_in(trigger, constraints)
    return any(
        holds_everywhere(constraints, {**dict(state), **dict(zip(others, bits, strict=True))})
        for bits in itertools.product([False, True], repeat=len(others))
    )


def brute(trigger, constraints):
    names = names_in(trigger)
    states = [dict(zip(names, bits, strict=True)) for bits in itertools.product([False, True], repeat=len(names))]
    states = [s for s in states if feasible(s, trigger, constraints)]
    if not states:
        return {"finding": "no_feasible_state"}
    fires = [ev(trigger, s) for s in states]
    if len(set(fires)) == 1:
        return {"finding": "always_fires" if fires[0] else "never_fires"}

    def changes(group):
        seen = {}
        for s, f in zip(states, fires, strict=True):
            seen.setdefault(tuple(s[c] for c in names if c not in group), set()).add(f)
        return any(len(v) == 2 for v in seen.values())

    witness = {c: changes({c}) for c in names}
    redundant = [c for c in names if not witness[c]]
    changing = [
        frozenset(g) for k in range(len(redundant) + 1) for g in itertools.combinations(redundant, k) if changes(set(g))
    ]
    groups = {g for g in changing if not any(h < g for h in changing)}
    return {"finding": None, "witness": witness, "groups": groups}


def minimal_excluding(trigger, constraints, fixed):
    """Every minimal set of constraint ids that no assignment with the fixed values satisfies, by trying every subset."""
    names = names_in(trigger) + others_in(trigger, constraints)
    full = [dict(zip(names, bits, strict=True)) for bits in itertools.product([False, True], repeat=len(names))]
    full = [f for f in full if all(f[c] == v for c, v in fixed.items())]
    ids = list(constraints)
    excluding = [
        frozenset(g)
        for k in range(len(ids) + 1)
        for g in itertools.combinations(ids, k)
        if not any(all(ev(constraints[i], f) for i in g) for f in full)
    ]
    return {g for g in excluding if not any(h < g for h in excluding)}


def check_points(trigger, constraints, cov):
    for pa in cov.points:
        fixed = {pa.point.condition: pa.point.holds}
        reasons = minimal_excluding(trigger, constraints, fixed)
        assert pa.feasible == (not reasons)
        if pa.feasible:
            assert certified(pa.example, pa.completion, trigger, constraints, cov)
            assert dict(pa.example)[pa.point.condition] is pa.point.holds
        else:
            assert set(pa.excluded_by) == reasons and len(pa.excluded_by) == len(reasons)


def certified(state, completion, trigger, constraints, cov):
    """The state is over the trigger's conditions, and its completion assigns exactly the other conditions so that
    every constraint holds: the certificate of feasibility, checked by evaluation."""
    assert [c for c, _ in state] == list(cov.conditions)
    assert [c for c, _ in completion] == list(cov.others) == others_in(trigger, constraints)
    assert holds_everywhere(constraints, {**dict(state), **dict(completion)})
    return True


def check_pair(pair, trigger, constraints, cov):
    lo, hi = dict(pair.not_fired), dict(pair.fired)
    assert certified(pair.not_fired, pair.completions[0], trigger, constraints, cov)
    assert certified(pair.fired, pair.completions[1], trigger, constraints, cov)
    assert ev(trigger, lo) is False and ev(trigger, hi) is True
    return lo, hi


def check_against_brute_force(trigger, constraints, cov):
    expected = brute(trigger, constraints)
    assert cov.conditions == tuple(names_in(trigger))
    proof = Unsat if cov.method == "sat" else Exhaustive
    if expected["finding"] is not None:
        assert cov.finding is not None and cov.finding.kind == expected["finding"]
        assert isinstance(cov.finding.proof, proof)
        assert cov.accounts == () and cov.groups == ()
        if cov.finding.kind == "no_feasible_state":
            assert set(cov.finding.excluded_by) == minimal_excluding(trigger, constraints, {})
        if cov.finding.example is not None:
            assert certified(cov.finding.example, cov.finding.completion, trigger, constraints, cov)
            assert ev(trigger, dict(cov.finding.example)) is (cov.finding.kind == "always_fires")
        return
    assert cov.finding is None
    assert {a.condition: not a.redundant for a in cov.accounts} == expected["witness"]
    assert {g.conditions for g in cov.groups} == expected["groups"]
    for a in cov.accounts:
        if a.redundant:
            assert isinstance(a.redundancy, proof)
            continue
        # A unique-cause pair: both states feasible, differing in this condition alone, and the outcome flips.
        lo, hi = check_pair(a.witness, trigger, constraints, cov)
        assert [c for c in cov.conditions if lo[c] != hi[c]] == [a.condition]
    for g in cov.groups:
        lo, hi = check_pair(g.pair, trigger, constraints, cov)
        assert all(lo[c] == hi[c] for c in cov.conditions if c not in g.conditions)


NAMES = [f"c{i}" for i in range(ENUMERATE_UP_TO + 2)]
OTHERS = ["o0", "o1"]  # conditions only the constraints mention


def exprs(names, max_leaves, constants=False):
    """Random expressions over the names; with constants, the empty And (true) and Or (false) are rare leaves."""
    constant = st.sampled_from([And(), Or()])
    if not names:
        return constant
    var = st.sampled_from(names).map(Var)
    leaf = st.one_of(var, var, var, constant) if constants else var
    return st.recursive(
        leaf,
        lambda kids: st.one_of(
            kids.map(Not),
            st.lists(kids, min_size=2, max_size=3).map(lambda xs: And(*xs)),
            st.lists(kids, min_size=2, max_size=3).map(lambda xs: Or(*xs)),
        ),
        max_leaves=max_leaves,
    )


def excludes(p, q):
    return Not(And(p, q))


def relations(names):
    """World constraints in the shapes they take (a linked threshold implies, a repeated predicate is equivalent,
    states exclude each other), and arbitrary expressions besides."""
    if not names:
        return exprs(names, 1, constants=True)
    two = st.tuples(st.sampled_from(names).map(Var), st.sampled_from(names).map(Var))
    return st.one_of(
        two.map(lambda pq: implies(*pq)),
        two.map(lambda pq: iff(*pq)),
        two.map(lambda pq: excludes(*pq)),
        exprs(names, 4, constants=True),
    )


def over_every(draw, names, repeats=3):
    """A random trigger that mentions every one of the names, some more than once: a random tree over a shuffled cover
    of them, with random connectives and negations."""
    leaves = draw(st.permutations([*names, *draw(st.lists(st.sampled_from(names), max_size=repeats))]))

    def build(xs):
        if len(xs) == 1:
            return draw(st.sampled_from([Var(xs[0]), Not(Var(xs[0]))]))
        k = draw(st.integers(2, min(3, len(xs))))
        cuts = sorted(draw(st.lists(st.integers(1, len(xs) - 1), min_size=k - 1, max_size=k - 1, unique=True)))
        parts = [xs[a:b] for a, b in zip([0, *cuts], [*cuts, len(xs)], strict=True)]
        node = draw(st.sampled_from([And, Or]))(*(build(part) for part in parts))
        return Not(node) if draw(st.integers(0, 3)) == 0 else node

    return build(leaves)


def linked(draw, names, others):
    """Constraints with structure: alias groups (a repeated predicate, an equivalence chain), implications (linked
    thresholds), links through conditions the trigger does not mention, and arbitrary relations besides."""
    pieces = []
    rest = list(draw(st.permutations(names)))
    while rest:
        size = draw(st.sampled_from([2, 1, 2, 3]))  # one alone: no alias
        group, rest = rest[:size], rest[size:]
        pieces += [iff(Var(group[0]), Var(x)) for x in group[1:]]
    for o in others:
        x, y = draw(st.permutations(names))[:2] if len(names) > 1 else (names[0], names[0])
        pieces += [implies(Var(x), Var(o)), draw(st.sampled_from([excludes, implies]))(Var(o), Var(y))]
    pieces += draw(st.lists(relations(list(names) + list(others)), max_size=2))
    return pieces


@st.composite
def problems(draw, sizes):
    """A random trigger over up to n conditions, with random constraints over its conditions and up to two others:
    either arbitrary, or with the structure that makes conditions redundant and coupled."""
    others = OTHERS[: draw(st.integers(0, len(OTHERS)))]
    if draw(st.booleans()):
        trigger = draw(exprs(NAMES[: draw(sizes)], 10))
        pool = names_in(trigger) + others
        pieces = draw(st.lists(relations(pool), max_size=5))
    else:
        names = NAMES[: draw(sizes)] or NAMES[:1]
        trigger = over_every(draw, names)
        pieces = linked(draw, names, others)
    return trigger, {f"k{i}": e for i, e in enumerate(pieces)}


@st.composite
def sized_problems(draw, total):
    """A random trigger over exactly n conditions and constraints mentioning exactly m others, n + m = total."""
    m = draw(st.integers(0, len(OTHERS)))
    names, others = NAMES[: total - m], OTHERS[:m]
    return over_every(draw, names), {f"k{i}": e for i, e in enumerate(linked(draw, names, others))}


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
        assert b.redundancy == Exhaustive(assignments=4, feasible=3, pairs=1)
    # Removing B alone leaves the trigger unchanged on the feasible states (it equals A there): no group.
    assert cov.groups == ()

    # Scenarios: the impossible (A, not B) history meets nothing, even beside a state that would pair with it.
    s = [
        Scenario("a_only", {"A": True, "B": False}),
        Scenario("both", {"A": True, "B": True}),
        Scenario("b_only", {"A": False, "B": True}),
    ]
    met = meets(cov, s)
    assert met.infeasible_scenarios == {"a_only": (frozenset({"A_implies_B"}),)}
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
    # Neither condition has an independent effect to cover; the group's joint effect is the obligation, so one
    # scenario does not cover the trigger.
    both, neither = Scenario("both", {"A": True, "B": True}), Scenario("neither", {"A": False, "B": False})
    met = meets(cov, [both])
    assert met.independence_infeasible == ("A", "B") and met.independence_unmet == ()
    assert met.groups_met == {} and met.groups_unmet == (frozenset({"A", "B"}),)
    met = meets(cov, [both, neither])
    assert met.groups_met == {frozenset({"A", "B"}): ("neither", "both")} and met.groups_unmet == ()


@BOTH
def test_three_equal_conditions_form_one_group_of_three(up_to):
    # A == B == C under A AND B AND C: any two can go, given the third; all three cannot.
    cov = coverage(And(A, B, C), {"ab": iff(A, B), "bc": iff(B, C)}, enumerate_up_to=up_to)
    assert cov.redundant == ("A", "B", "C")
    assert [g.conditions for g in cov.groups] == [frozenset({"A", "B", "C"})]


@BOTH
def test_two_coupled_pairs_are_two_groups(up_to):
    # (A AND B) OR (C AND D) with A == B and C == D: every condition is redundant given the others; each pair's joint
    # removal changes the trigger, and so two groups, while any one condition from each pair can go together.
    D = Var("D")
    cov = coverage(Or(And(A, B), And(C, D)), {"ab": iff(A, B), "cd": iff(C, D)}, enumerate_up_to=up_to)
    assert cov.redundant == ("A", "B", "C", "D")
    assert [g.conditions for g in cov.groups] == [frozenset({"A", "B"}), frozenset({"C", "D"})]


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


# ---------------------------------------------------------------- constraints over conditions the trigger does not mention


@BOTH
def test_a_condition_outside_the_trigger_links_two_of_its_conditions(up_to):
    # A implies B and B excludes C, where B is no condition of the trigger: through B, A excludes C. In A AND NOT C the
    # state (A, C) is infeasible, so C has no independent effect: NOT C is implied by A, and the trigger equals A on the
    # feasible states. The witness for A fires with A true, which forces B true in its completion.
    constraints = {"a_implies_b": implies(A, B), "b_excludes_c": Not(And(B, C))}
    cov = coverage(And(A, Not(C)), constraints, enumerate_up_to=up_to)
    assert cov.conditions == ("A", "C") and cov.others == ("B",)
    assert cov.method == ("sat" if up_to == 0 else "enumeration")
    a, c = cov.accounts
    assert dict(a.witness.not_fired) == {"A": False, "C": False} and dict(a.witness.fired) == {"A": True, "C": False}
    assert dict(a.witness.completions[1]) == {"B": True}
    assert c.redundant
    if up_to:
        # Every assignment of A, C and B was evaluated: three of the four trigger states are feasible, and the one pair
        # differing in C alone, with A false, has equal outcomes.
        assert c.redundancy == Exhaustive(assignments=8, feasible=3, pairs=1)
    assert cov.groups == ()
    # A scenario is feasible when some value of B satisfies both constraints; one that states B is checked with it.
    s = [
        Scenario("a_and_c", {"A": True, "C": True}),
        Scenario("a_without_b", {"A": True, "C": False, "B": False}),
        Scenario("a", {"A": True, "C": False}),
        Scenario("nothing", {"A": False, "C": False}),
    ]
    met = meets(cov, s)
    assert met.infeasible_scenarios == {
        "a_and_c": (frozenset({"a_implies_b", "b_excludes_c"}),),
        "a_without_b": (frozenset({"a_implies_b"}),),
    }
    assert met.independence_met == {"A": ("nothing", "a")}
    assert met.independence_infeasible == ("C",)


@BOTH
def test_the_trigger_can_be_constant_through_a_condition_it_does_not_mention(up_to):
    # A implies B and B excludes C: A AND C can never fire, a finding about the whole trigger.
    cov = coverage(And(A, C), {"a_implies_b": implies(A, B), "b_excludes_c": Not(And(B, C))}, enumerate_up_to=up_to)
    assert cov.finding.kind == "never_fires" and cov.accounts == ()
    assert truth(And(implies(A, B), Not(And(B, C))), {**dict(cov.finding.example), **dict(cov.finding.completion)})


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


@settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(problems(st.sampled_from(range(8))))
def test_every_verdict_agrees_with_brute_force_on_both_paths(problem):
    # Every condition's verdict and every group, and, for boundary points on the first condition (one at each of its
    # values), feasibility and every minimal set of constraints excluding them.
    trigger, constraints = problem
    conditions = names_in(trigger)
    points = threshold_points(conditions[0], ">=", 1, 1) if conditions else ()
    for up_to in (0, 64):
        cov = coverage(trigger, constraints, points, enumerate_up_to=up_to)
        assert cov.method == ("sat" if up_to == 0 and (cov.conditions or cov.others) else "enumeration")
        check_against_brute_force(trigger, constraints, cov)
        check_points(trigger, constraints, cov)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(st.sampled_from([ENUMERATE_UP_TO, ENUMERATE_UP_TO + 1]).flatmap(sized_problems))
def test_every_verdict_agrees_on_both_sides_of_the_size_where_sat_takes_over(problem):
    # The size is every condition the trigger and its constraints mention: ENUMERATE_UP_TO is enumerated, one more is
    # solved, at the default setting.
    trigger, constraints = problem
    cov = coverage(trigger, constraints)
    size = len(cov.conditions) + len(cov.others)
    assert size in (ENUMERATE_UP_TO, ENUMERATE_UP_TO + 1)
    assert cov.method == ("enumeration" if size <= ENUMERATE_UP_TO else "sat")
    check_against_brute_force(trigger, constraints, cov)


families = st.integers(0, 6).flatmap(
    lambda n: st.tuples(st.just(n), st.lists(st.sets(st.integers(0, max(n - 1, 0)), max_size=n), max_size=5))
)


@given(families)
def test_minimal_sets_are_found_completely(problem):
    # The predicate "contains some member of the family" is closed under supersets; its minimal sets are the family's
    # minimal members.
    n, family = problem
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
    assert truth(And(*constraints.values()), {**dict(below.example), **dict(below.completion)})
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
    with pytest.raises(ValueError, match="not a condition of the trigger"):
        coverage(A, {"k": implies(A, B)}, threshold_points("B", ">=", 1, 1))
    with pytest.raises(ValueError, match="every condition"):
        meets(coverage(A, {"k": implies(A, B)}), [Scenario("stray", {"A": True, "Z": False})])
    with pytest.raises(ValueError, match="two comparisons"):
        coverage(A, {}, threshold_points("A", ">=", 1, 1) + threshold_points("A", "<", 1, 1))
    with pytest.raises(ValueError):
        Var("")
