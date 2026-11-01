"""Properties of the liquidity optimizer over generated problems, at its limits (addendum A06 §6.4-6.5, C01 §7.2).

The generator reaches the edges: one scenario or one date, zero-probability scenarios, alpha 0 and 0.99, a zero budget,
zero caps, free instruments, premiums paid after the first date, a reserve above the cash, and money from cents to
millions.
"""

from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal as D
from fractions import Fraction as F

from hypothesis import given, settings
from hypothesis import strategies as st

from requisite.optimize import Candidate, LiquidityProblem, solve, try_candidate

UNITS = [D("0.01"), D(1), D(1000), D(10**6)]
LEVELS = [F(0), F(1, 2), F(9, 10), F(95, 100), F(99, 100)]


def amount(draw, unit, lo, hi):
    """Money with two decimals of the unit, between lo and hi units."""
    return D(draw(st.integers(lo * 100, hi * 100))) * unit / 100


@st.composite
def businesses(draw):
    n_s, n_t = draw(st.integers(1, 4)), draw(st.integers(1, 4))
    unit = draw(st.sampled_from(UNITS))
    weights = draw(st.lists(st.integers(0, 5), min_size=n_s, max_size=n_s).filter(any))
    dates = tuple(date(2026, 1, 1) + timedelta(days=30 * i) for i in range(n_t))
    return {
        "dates": dates,
        "probabilities": tuple(F(w, sum(weights)) for w in weights),
        "initial_cash": amount(draw, unit, -50, 100),
        "operating": [[amount(draw, unit, -20, 20) for _ in dates] for _ in weights],
        "loss": [[amount(draw, unit, 0, 100) if draw(st.booleans()) else D(0) for _ in dates] for _ in weights],
        "reserve": [amount(draw, unit, 0, 50) if draw(st.booleans()) else D(0) for _ in dates],
        "alpha": draw(st.one_of(st.sampled_from(LEVELS), st.fractions(0, F(99, 100), max_denominator=1000))),
        "budget": amount(draw, unit, 0, 100),
    }, unit


def instrument(draw, ref, business, unit, dominated=False):
    """A candidate on the business's grid. A dominated one (addendum §6.5) costs something and its cumulative receipts
    never exceed what has been paid for it, in any scenario at any date: at most zero before its premium date, at most
    its price from then on."""
    dates, n_s = business["dates"], len(business["probabilities"])
    paid = draw(st.integers(0, len(dates) - 1))
    price = amount(draw, unit, 1, 50) if dominated else amount(draw, unit, 0, 50)
    if dominated:
        receipts = []
        for _ in range(n_s):
            held = [amount(draw, unit, -10, 0) if t < paid else draw(st.sampled_from([price, D(0)])) - amount(draw, unit, 0, 10) for t in range(len(dates))]  # fmt: skip
            receipts.append([held[0], *(b - a for a, b in zip(held, held[1:], strict=False))])
    else:
        receipts = [[amount(draw, unit, 0, 200) if draw(st.booleans()) else D(0) for _ in dates] for _ in range(n_s)]
    cap = D(draw(st.integers(0, 500))) / 100
    return Candidate(ref, price, dates[paid], receipts, cap)


@st.composite
def problems(draw, extra=0, dominated=0):
    """A business with up to three free candidates; then `extra` more, and `dominated` dominated ones, returned apart."""
    business, unit = draw(businesses())
    free = [instrument(draw, f"c{j}", business, unit) for j in range(draw(st.integers(0, 3)))]
    more = [instrument(draw, f"e{j}", business, unit) for j in range(extra)]
    worse = [instrument(draw, f"d{j}", business, unit, dominated=True) for j in range(dominated)]
    return LiquidityProblem(**business, candidates=tuple(free)), more, worse


def tail_mass_cvar(losses, probabilities, alpha):
    room, total = 1 - alpha, F(0)
    for x, p in sorted(zip(losses, probabilities, strict=True), reverse=True):
        take = min(p, room)
        total += take * x
        room -= take
    return total / (1 - alpha)


def recompute(problem, quantities):
    """Cash paths and deficits from the inputs and the quantities, in Fractions, by this test's own accounting."""
    cash, deficits = [], []
    for s in range(len(problem.probabilities)):
        balance, received, path = F(problem.initial_cash), [F(0)] * len(quantities), []
        for t, day in enumerate(problem.dates):
            balance += F(problem.operating[s][t]) - F(problem.loss[s][t])
            hedge = F(0)
            for j, (c, q) in enumerate(zip(problem.candidates, quantities, strict=True)):
                received[j] += F(c.receipts[s][t])
                hedge += F(q) * (received[j] - (F(c.price) if day >= c.premium_date else 0))
            path.append(balance + hedge)
        cash.append(path)
        deficits.append(max([F(0), *(F(r) - c for r, c in zip(problem.reserve, path, strict=True))]))
    return cash, deficits


@settings(deadline=None)
@given(problems(dominated=2))
def test_a_dominated_payoff_is_bought_at_zero(case):
    # Addendum §6.5: receipts that never exceed acquisition cost in any state-date, no other benefit, and an objective
    # monotone in cash shortfall: holding any of it only lowers cash, so the premium tie-break holds none, exactly.
    problem, _, dominated = case
    for k in range(1, len(dominated) + 1):
        solved = solve(replace(problem, candidates=(*problem.candidates, *dominated[:k])))
        assert solved.quantities[len(problem.candidates) :] == (0,) * k


@settings(deadline=None)
@given(problems(extra=1))
def test_adding_a_candidate_never_raises_the_optimum_by_more_than_the_tolerance(case):
    problem, (extra,), _ = case
    without, with_it = solve(problem), solve(problem.with_candidate(extra))
    assert with_it.scale == without.scale  # the scale is the business's, whatever the candidates
    assert with_it.primary_objective <= without.primary_objective + without.tolerance


@settings(deadline=None)
@given(problems())
def test_the_optimum_is_the_cvar_of_the_chosen_cash_and_the_caps_and_budget_hold(case):
    problem, _, _ = case
    solved = solve(problem)
    q = solved.quantities
    cash, deficits = recompute(problem, q)
    p, alpha = problem.probabilities, problem.alpha
    assert [[F(x) for x in row] for row in solved.hedged.cash] == cash
    assert [F(x) for x in solved.hedged.deficit] == deficits
    assert solved.hedged.cvar == tail_mass_cvar(deficits, p, alpha)
    assert abs(solved.hedged.cvar - F(solved.primary_objective)) <= solved.tolerance
    assert all(0 <= x <= c.cap for x, c in zip(q, problem.candidates, strict=True))
    assert sum(F(c.price) * F(x) for c, x in zip(problem.candidates, q, strict=True)) <= problem.budget


@settings(deadline=None)
@given(problems(extra=1))
def test_the_reduced_cost_bounds_the_measured_improvement(case):
    # LP duality (C01 §7.2): with the new instrument held at t units, the optimum is at least the old one plus t times
    # its reduced cost under any optimal dual, so the measured improvement never exceeds -t * reduced cost (beyond the
    # two solves' tolerances), and a nonnegative reduced cost admits none. The converse does not hold: a negative
    # reduced cost promises nothing, which is why the kernel re-solves.
    problem, (extra,), _ = case
    trial = try_candidate(solve(problem), extra)
    held = trial.after.quantities[-1]
    slack = trial.before.tolerance + trial.after.tolerance
    assert trial.improvement <= -F(trial.reduced_cost) * F(held) + F(slack)
