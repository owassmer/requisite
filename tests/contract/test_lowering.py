"""Lowering a trigger for coverage (requisite.contract.evaluate.lower; spec §8.6 step 1; addendum A09 §10.2-10.3): the
trigger's boolean skeleton over its atomic conditions, one variable per condition named by its IR node id, with each
comparison's boundary points and the constraints the IR itself places between conditions. The coverage kernel works on
that skeleton only; a scenario's outcome is always the evaluator's, and the two agree."""

from datetime import timedelta
from decimal import Decimal as D

import western_grazers as wg
from hypothesis import given, settings
from hypothesis import strategies as st
from irgen import contracts, worlds
from irkit import EXPIRY, T0, at, contract, num, obs, observable, reading

from requisite import core as c
from requisite.contract.evaluate import evaluate, lower
from requisite.core.kleene import Kleene
from requisite.fuzz import coverage as cov

W = T0 + timedelta(days=5)


def points(lowered, condition):
    return {comp.kind: [p.value for p in comp.points] for comp in lowered.comparisons if comp.condition == condition}


def test_the_skeleton_keeps_and_or_not_over_one_variable_per_atomic_condition():
    count = c.CountTrue(id="count", expr=c.Cmp(id="hot", observable="x", op=">", value=num(10)),
                        days=c.TimeWindow(start=T0, end=T0 + timedelta(days=10), start_closed=True, end_closed=False))  # fmt: skip
    trigger = c.And(id="all", args=[
        at("x", ">", 10, W, nid="high"),
        c.Not(id="not", arg=c.Interpretive(id="stayed", question_template="t", bindings={})),
        c.Or(id="either", args=[c.NumCmp(id="days", expr=count, op=">=", value=D(3)),
                                c.NumCmp(id="read", expr=reading(), op="<", value=D(10))]),
    ])  # fmt: skip
    lowered = lower(contract(trigger))
    assert lowered.trigger == cov.And(cov.Var("high"), cov.Not(cov.Var("stayed")), cov.Or(cov.Var("days"), cov.Var("read")))
    assert lowered.conditions == {"high": "at", "stayed": "interpretive", "days": "num_cmp", "read": "num_cmp"}
    # Boundary points: a threshold at the observable's declared precision (one cent), a count at k - 1, k, k + 1 within
    # the days the window allows, and a reading's threshold at its stated precision within its bands' range.
    assert points(lowered, "high") == {"threshold": [D("9.99"), D("10.00"), D("10.01")]}
    assert points(lowered, "days") == {"count": [2, 3, 4]}
    assert points(lowered, "read") == {"threshold": [D(9), D(10), D(11)]}
    assert lowered.constraints == {}
    # The kernel takes them, and finds a unique-cause witness for every condition.
    account = cov.coverage(lowered.trigger, lowered.constraints, lowered.rounds()[0])
    assert account.finding is None and account.redundant == ()


def test_temporal_cutoffs_are_lowered_at_the_comparisons_resolution_with_their_inclusion():
    by = c.By(id="by", expr=c.Cmp(id="hot", observable="x", op=">", value=num(10)), since=T0, since_closed=False,
              deadline=EXPIRY, closed=True)  # fmt: skip
    lowered = lower(contract(by), resolution=timedelta(seconds=1))
    cutoffs = [comp for comp in lowered.comparisons if comp.kind == "cutoff"]
    assert [(comp.points[0].op, comp.points[0].threshold) for comp in cutoffs] == [("<=", EXPIRY), (">", T0)]
    assert [p.value for p in cutoffs[0].points] == [EXPIRY - timedelta(seconds=1), EXPIRY, EXPIRY + timedelta(seconds=1)]
    assert [p.holds for p in cutoffs[0].points] == [True, True, False]  # the deadline is included
    assert [p.holds for p in cutoffs[1].points] == [False, False, True]  # the start is not
    # A By's deadline and its threshold are two comparisons of one condition: the kernel takes them in two rounds.
    assert [len({p.condition for p in r}) for r in lowered.rounds()] == [1, 1, 1]


def feasible(lowered) -> set[tuple[bool, ...]]:
    """The states of the trigger's conditions, in condition order, that the lowered constraints allow, as the kernel
    finds them."""
    names = list(lowered.conditions)
    states = [tuple(bool(n >> i & 1) for i in range(len(names))) for n in range(2 ** len(names))]
    scenarios = [cov.Scenario(id=str(k), state=dict(zip(names, s, strict=True))) for k, s in enumerate(states)]
    account = cov.coverage(lowered.trigger, lowered.constraints)
    infeasible = cov.meets(account, scenarios).infeasible_scenarios
    return {s for k, s in enumerate(states) if str(k) not in infeasible}


def realized(ir, records) -> tuple[bool, ...] | None:
    """Each atomic condition's truth on these records, evaluated alone; None where one is not established."""
    out = []
    for cid in lower(ir).conditions:
        node = next(n for n in c.walk(ir.trigger) if n.id == cid)
        t = evaluate(ir.model_copy(update={"trigger": node}), records, {}, at=EXPIRY).trigger_truth
        if t is Kleene.UNKNOWN:
            return None
        out.append(t is Kleene.TRUE)
    return tuple(out)


def test_conditions_over_one_quantity_carry_the_constraints_its_values_impose():
    # x > 10 and x > 5 at one instant: x > 10 without x > 5 is impossible (A09 §10.2: a linked threshold), and the kernel
    # is told so, so it fabricates no witness from that state. It is exactly what the evaluator can realize: every value
    # at the observable's precision gives an allowed state, and each allowed state is given by some value.
    linked = contract(c.And(id="both", args=[at("x", ">", 10, W, nid="above10"), at("x", ">", 5, W, nid="above5")]))
    allowed = feasible(lower(linked))
    assert allowed == {(False, False), (False, True), (True, True)}
    assert {realized(linked, [obs(D(cents) / 100, W)]) for cents in range(300, 1500, 7)} == allowed
    account = cov.coverage(lower(linked).trigger, lower(linked).constraints)
    assert [a.condition for a in account.accounts if a.redundant] == ["above5"]
    # Two nodes with one meaning are one condition in every state; two bands of one reading split its range exactly.
    repeated = contract(c.Or(id="o", args=[at("x", ">", 10, W, nid="first"), at("x", ">", 10, W, nid="second")]))
    assert feasible(lower(repeated)) == {(False, False), (True, True)}
    r = reading()
    split = c.Or(id="o", args=[c.NumCmp(id="low", expr=r, op="<", value=D(10)),
                               c.NumCmp(id="high", expr=r, op=">=", value=D(10))])  # fmt: skip
    assert feasible(lower(contract(split, observables=[observable()]))) == {(True, False), (False, True)}
    # A comparison no published value can meet is constant: x == 10.005 where x is published to the cent.
    never = contract(c.Or(id="o", args=[at("x", "==", "10.005", W, nid="odd"), at("x", ">", 0, W, nid="pos")]))
    assert {s[0] for s in feasible(lower(never))} == {False}


def test_conditions_over_one_observable_at_different_instants_and_ranges_are_linked_exactly():
    hot = c.Cmp(id="hot", observable="x", op=">", value=num(10))
    cold = c.Cmp(id="cold", observable="x", op="<", value=num(5))
    d = [T0 + timedelta(days=i) for i in range(11)]
    # A value shown by day 5 is shown by day 10; one established by day 10 need not be by day 5.
    early = c.By(id="early", expr=hot, since=d[0], since_closed=True, deadline=d[5], closed=True)
    late = c.By(id="late", expr=hot, since=d[0], since_closed=True, deadline=d[10], closed=True)
    nested = contract(c.And(id="both", args=[early, late]))
    assert feasible(lower(nested)) == {(False, False), (False, True), (True, True)}
    examples = {(False, False): [obs(1, d[2])], (False, True): [obs(1, d[2]), obs(11, d[8])],
                (True, True): [obs(11, d[2])]}  # fmt: skip
    assert {realized(nested, records) for records in examples.values()} == set(examples)
    # A value at day 3 inside a window throughout which x is below 5 cannot be above 10; outside the window it can.
    for when, linked in ((d[3], True), (d[9], False)):
        point = c.At(id="point", expr=hot, at=when)
        window = c.Throughout(id="window", expr=cold, window=c.TimeWindow(start=d[1], end=d[6], start_closed=True,
                                                                          end_closed=True))  # fmt: skip
        ir = contract(c.And(id="both", args=[point, window]))
        assert ((True, True) in feasible(lower(ir))) is not linked
        assert realized(ir, [obs(11, when), obs(1, d[2])]) == ((True, False) if linked else (True, True))
    # By and Throughout over one window: below 5 throughout excludes above 10 at some instant within it.
    by_hot = c.By(id="by_hot", expr=hot, since=d[1], since_closed=True, deadline=d[6], closed=True)
    all_cold = c.Throughout(id="all_cold", expr=cold, window=c.TimeWindow(start=d[1], end=d[6], start_closed=True,
                                                                          end_closed=True))  # fmt: skip
    assert feasible(lower(contract(c.Or(id="o", args=[by_hot, all_cold])))) == {(False, False), (True, False),
                                                                                (False, True)}  # fmt: skip
    # Between two named instants any number of instants can be published: above 10 at one and not at another.
    some = c.By(id="some", expr=hot, since=d[1], since_closed=False, deadline=d[2], closed=False)
    every = c.Throughout(id="every", expr=hot, window=c.TimeWindow(start=d[1], end=d[2], start_closed=False,
                                                                    end_closed=False))  # fmt: skip
    both = contract(c.And(id="both", args=[some, every]))
    assert feasible(lower(both)) == {(False, False), (True, False), (True, True)}
    assert realized(both, [obs(11, d[1] + timedelta(hours=6)), obs(1, d[1] + timedelta(hours=12))]) == (True, False)


@settings(max_examples=400, deadline=None)
@given(data=st.data())
def test_no_state_the_evaluator_can_realize_is_excluded_by_the_lowered_constraints(data):
    # Random conditions over one observable at instants and ranges of a ten-day grid, and random observations: whenever
    # every condition is established, the state they give is one the constraints allow.
    d = [T0 + timedelta(days=i) for i in range(11)]
    ids = iter(range(100))

    def condition():
        n = next(ids)
        cmp = c.Cmp(id=f"c{n}", observable="x", op=data.draw(st.sampled_from(["<", "<=", "==", "!=", ">=", ">"])),
                    value=num(data.draw(st.integers(0, 6))))  # fmt: skip
        i = data.draw(st.integers(0, 10))
        j = data.draw(st.integers(i, 10))
        kind = data.draw(st.sampled_from(["at", "by", "throughout", "value_at"]))
        if kind == "at":
            return c.At(id=f"a{n}", expr=cmp, at=d[i])
        if kind == "value_at":
            return c.NumCmp(id=f"v{n}", expr=c.ValueAt(id=f"w{n}", observable="x", at=d[i]), op=cmp.op, value=cmp.value.value)
        if kind == "by":
            return c.By(id=f"b{n}", expr=cmp, since=d[i], since_closed=data.draw(st.booleans()), deadline=d[j],
                        closed=data.draw(st.booleans()))  # fmt: skip
        return c.Throughout(id=f"t{n}", expr=cmp, window=c.TimeWindow(start=d[i], end=d[j],
                            start_closed=data.draw(st.booleans()), end_closed=data.draw(st.booleans())))  # fmt: skip

    ir = contract(c.And(id="all", args=[condition() for _ in range(data.draw(st.integers(2, 3)))]),
                  observables=[observable(precision=D(1))])  # fmt: skip
    # A value on every day of the grid, so that most conditions are established, and some between the days.
    instants = sorted({4 * i for i in range(11)} | set(data.draw(st.lists(st.integers(0, 40), max_size=8))))
    records = [obs(data.draw(st.integers(0, 7)), T0 + timedelta(hours=6 * k)) for k in instants]
    state = realized(ir, records)
    if state is not None:
        assert state in feasible(lower(ir))


@settings(max_examples=300, deadline=None)
@given(data=st.data())
def test_the_skeleton_over_each_conditions_evaluated_truth_is_the_evaluators_trigger(data):
    # For random contracts and worlds in which every input is known: evaluate each atomic condition alone, and the
    # skeleton over those truths is the trigger's truth. So coverage reasons about the same boolean function the
    # evaluator settles, and only the evaluator reads observations.
    ir = data.draw(contracts(decidable=True))
    ir = ir.model_copy(update={"resolution": ir.resolution.model_copy(update={
        "early_expiration": "none", "early_expiration_at": None, "early_expiration_rule": None})})  # fmt: skip
    (observations, given), _ = data.draw(worlds(ir, complete=True))
    when = T0 + timedelta(days=12)
    lowered = lower(ir)
    nodes = {n.id: n for n in c.walk(ir.trigger)}
    state = {}
    for cid in lowered.conditions:
        alone = ir.model_copy(update={"trigger": nodes[cid]})
        needed = {k: v for k, v in given.items() if k in {n.id for n in c.walk(nodes[cid])}}
        state[cid] = evaluate(alone, observations, needed, at=when).trigger_truth is Kleene.TRUE
    whole = evaluate(ir, observations, given, at=when).trigger_truth
    assert whole is (Kleene.TRUE if cov.truth(lowered.trigger, state) else Kleene.FALSE)


def test_the_western_grazers_coverage_witnesses_are_settled_by_the_evaluator():
    # The kernel finds a unique-cause witness for each of the eight conditions; each witness's two abstract states,
    # realized as interpretations and readings, are settled by the evaluator, never by the skeleton: the state that
    # fires resolves YES and the one that does not resolves NO.
    k = wg.ir(wg.AB187)
    lowered = lower(k)
    account = cov.coverage(lowered.trigger, lowered.constraints, lowered.rounds()[0] if lowered.rounds() else ())
    assert len(account.accounts) == 8 and account.redundant == ()
    assert points(lowered, "net_monthly_cost_at_most_threshold") == {"threshold": [D("-0.01"), D("0.00"), D("0.01")]}
    passes = wg.ab187()
    fails = {**passes, "read_sheepherder_framework": wg.FLAG[0], "read_monthly_wage": (D(10000), D(15000))}

    def realized(state):
        given = dict(passes if dict(state)["net_monthly_cost_at_most_threshold"] else fails)
        for cid, value in state:
            if cid != "net_monthly_cost_at_most_threshold":
                given[cid] = Kleene.TRUE if value else Kleene.FALSE
        return evaluate(k, wg.rates(), given, at=wg.EXPIRATION).procedure_status

    for a in account.accounts:
        assert realized(a.witness.fired) == "resolved_yes", a.condition
        assert realized(a.witness.not_fired) == "resolved_no", a.condition


def test_a_durations_boundary_points_lie_within_its_window():
    # A duration lies between zero and its window's length: 25 hours in a 24-hour window is no value it can take, so
    # that boundary point is infeasible, never a scenario (A09 §10.3).
    window = c.TimeWindow(start=T0, end=T0 + timedelta(days=1), start_closed=True, end_closed=False)
    hot = c.DurationIn(id="hot", expr=c.Cmp(id="warm", observable="x", op=">", value=num(30)), window=window, unit="hour")
    lowered = lower(contract(c.NumCmp(id="day", expr=hot, op="!=", value=D(24))))
    (comp,) = lowered.comparisons
    assert [(p.value, p.in_domain) for p in comp.points] == [(D(23), True), (D(24), True), (D(25), False)]
    account = cov.coverage(lowered.trigger, lowered.constraints, comp.points)
    assert [pa.feasible for pa in account.points] == [True, True, False]


@settings(max_examples=200, deadline=None)
@given(contracts(decidable=True))
def test_every_feasible_state_of_an_observables_conditions_is_realized_by_its_layout(ir):
    # The lowering's claim: any assignment its constraints allow is realized by placing one value of each class present
    # in each region, at an instant of that region. Every feasible state the kernel finds, so placed, is settled by the
    # evaluator to exactly that state for each condition of the layout.
    lowered = lower(ir)
    nodes = {n.id: n for n in c.walk(ir.trigger)}
    for lay in lowered.layouts.values():
        if len(lay.conditions) < 2:
            continue
        constraints = {k: v for k, v in lowered.constraints.items() if k.startswith(f"ir:{lay.observable}:")}
        group = cov.And(*(cov.Var(x) for x in lay.conditions))  # any expression mentioning exactly the conditions
        account = cov.coverage(group, constraints)
        with cov.Feasible(account) as feas:
            for state, completion in feas.states():
                done = dict(completion)
                placed = {i: [k for k in range(len(lay.keys)) if done.get(lay.variable(i, k))] for i in lay.used}
                records = [obs(lay.classes[k].value, t, oid=lay.observable, src=src, synthetic=False)
                           for i, ks in placed.items() if ks for k, t in zip(ks, lay.instants(i, len(ks)), strict=True)
                           for src in (["a", "b"] if lay.observable == "y" else ["a"])]  # fmt: skip
                for cid, want in state:
                    alone = ir.model_copy(update={"trigger": nodes[cid]})
                    got = evaluate(alone, records, {}, at=T0 + timedelta(days=12)).trigger_truth
                    assert got is (Kleene.TRUE if want else Kleene.FALSE), (cid, state, done)
