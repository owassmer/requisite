"""The liquidity optimizer (spec §8.3, WP10; addendum A06 §6.1-6.6 and C01 §7.2-7.4): the oracle's checks 6-9 reproduced
to its tolerances, and the addendum's fixtures that replace the spec's own WP10 tests."""

from dataclasses import fields
from datetime import date, timedelta
from decimal import Decimal as D
from fractions import Fraction as F

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from requisite.optimize import (
    Candidate,
    CandidateTrial,
    CashEvaluation,
    LiquidityProblem,
    LiquiditySolution,
    LPResiduals,
    discrete_cvar,
    evaluate,
    reduced_cost,
    solve,
    try_candidate,
)

GRID = tuple(date(2026, 1, 1) + timedelta(days=30 * i) for i in range(4))  # day 0, 30, 60, 90


def business(probabilities, initial_cash, loss, *, alpha, budget, candidates=()):
    """Scenarios whose only flow is the modeled loss, with a zero floor, on the first len(loss[0]) grid dates."""
    n_t = len(loss[0])
    return LiquidityProblem(
        dates=GRID[:n_t],
        probabilities=tuple(F(p) for p in probabilities),
        initial_cash=D(initial_cash),
        operating=[[0] * n_t for _ in loss],
        loss=[[D(x) for x in row] for row in loss],
        reserve=[0] * n_t,
        alpha=F(alpha),
        budget=D(budget),
        candidates=tuple(candidates),
    )


def instrument(ref, price, receipts, cap, premium_date=GRID[0]):
    return Candidate(ref, D(price), premium_date, [[D(x) for x in row] for row in receipts], D(cap))


def isclose(a, b, atol=1e-8):
    """numpy.isclose, the oracle's comparison: |a - b| <= atol + 1e-5 |b|."""
    return abs(float(a) - float(b)) <= atol + 1e-5 * abs(float(b))


def allclose(a, b, atol=1e-8):
    if isinstance(b, list | tuple):
        return len(a) == len(b) and all(allclose(x, y, atol) for x, y in zip(a, b, strict=True))
    return isclose(a, b, atol)


# ------------------------------------------------------------------ the oracle (spec/verify_math.py, checks 6-9)
# The recorded values are spec/verified_synthetic_results.json's, which is not in git. The oracle asserts its LP outputs
# with np.isclose(atol=1e-7), and the rest with np.isclose's defaults; the kernel is held to the same.


@pytest.mark.covers("O06")
def test_the_oracle_check_6_zero_covariance_tail_hedge():
    loss, payoff, p = [0, 1, 2], [1, 0, 1], [F(1, 3)] * 3
    problem = business(
        p, "1.5", [[0, x] for x in loss], alpha=F(2, 3), budget=10,
        candidates=[instrument("z", "0.75", [[0, z] for z in payoff], 2)],
    )  # fmt: skip
    s = solve(problem)
    one_unit = evaluate(problem, [1])
    mean = lambda xs: sum(F(x) * px for x, px in zip(xs, p, strict=True))  # noqa: E731
    recorded = {
        "covariance": 0.0,
        "initial_cash": 1.5,
        "terminal_residual_cvar_unhedged": 2.0,
        "terminal_residual_cvar_one_unit": 1.75,
        "liquidity_cvar_unhedged": 0.5,
        "quantity": [0.9999999959999997],
        "premium": 0.7499999969999998,
        "cash": [[0.7500000030000002, 1.749999999], [0.7500000030000002, -0.24999999699999975],
                 [0.7500000030000002, -0.2500000010000001]],
        "shortfall": [0.0, 0.24999999699999975, 0.2500000010000001],
        "cvar": 0.2500000010000001,
        "primary_value": 0.25,
        "cash_dual_beta": [[0.0, 0.0], [0.0, 0.25], [0.0, 0.75]],
        "budget_dual_mu": 0.0,
    }  # fmt: skip
    ours = {
        "covariance": mean([x * z for x, z in zip(loss, payoff, strict=True)]) - mean(loss) * mean(payoff),
        "initial_cash": problem.initial_cash,
        "terminal_residual_cvar_unhedged": discrete_cvar(s.unhedged.residual, p, problem.alpha),
        "terminal_residual_cvar_one_unit": discrete_cvar(one_unit.residual, p, problem.alpha),
        "liquidity_cvar_unhedged": s.unhedged.cvar,
        "quantity": s.quantities,
        "premium": s.hedged.premium,
        "cash": s.hedged.cash,
        "shortfall": s.hedged.deficit,
        "cvar": s.hedged.cvar,
        "primary_value": s.primary_objective,
        "cash_dual_beta": s.cash_duals,
        "budget_dual_mu": s.budget_dual,
    }
    for key, value in recorded.items():
        assert allclose(ours[key], value, atol=1e-7), key


@pytest.mark.covers("O07")
def test_the_oracle_check_7_flat_objective_uses_premium_tiebreak():
    problem = business([1], 100, [[0, 0]], alpha="0.95", budget=50, candidates=[instrument("x", 10, [[0, 0]], 5)])
    s = solve(problem)
    recorded = {
        "quantity": [0.0],
        "premium": 0.0,
        "cash": [[100.0, 100.0]],
        "shortfall": [0.0],
        "cvar": 0.0,
        "primary_value": 0.0,
        "cash_dual_beta": [[0.0, 0.0]],
        "budget_dual_mu": 0.0,
    }
    ours = {
        "quantity": s.quantities,
        "premium": s.hedged.premium,
        "cash": s.hedged.cash,
        "shortfall": s.hedged.deficit,
        "cvar": s.hedged.cvar,
        "primary_value": s.primary_objective,
        "cash_dual_beta": s.cash_duals,
        "budget_dual_mu": s.budget_dual,
    }
    for key, value in recorded.items():
        assert allclose(ours[key], value), key


@pytest.mark.covers("O08")
def test_the_oracle_check_8_dated_receipts():
    # Cash 100, a loss of 120 at day 30, a premium of 10 at day 0; the payout of 120 at day 60 or at day 30.
    unhedged = business([1], 100, [[0, 120, 0]], alpha="0.9", budget=10)
    late = evaluate(unhedged.with_candidate(instrument("late", 10, [[0, 0, 120]], 1)), [1])
    early = evaluate(unhedged.with_candidate(instrument("early", 10, [[0, 120, 0]], 1)), [1])
    assert late.cash == ((90, -30, 90),) and early.cash == ((90, 90, 90),)
    assert late.deficit == (30,) and early.deficit == (0,)
    assert late.premium == early.premium == 10


@pytest.mark.covers("O09")
def test_the_oracle_check_9_dual_guided_candidate():
    master = solve(business(["0.8", "0.2"], 100, [[0, 0], [0, 120]], alpha="0.9", budget=20))
    proposed = instrument("y", 10, [[0, 0], [0, 120]], 1)
    rc = reduced_cost(master, proposed)
    sliver = solve(master.problem.with_candidate(instrument("y", 10, [[0, 0], [0, 120]], "0.00001")))
    derivative = (sliver.primary_objective - master.primary_objective) / D("0.00001")
    trial = try_candidate(master, proposed)
    after = trial.after
    assert rc < 0
    assert isclose(rc, -110.0) and isclose(derivative, -110.0000000000989, atol=1e-6)
    assert isclose(rc, derivative, atol=1e-6)
    assert isclose(master.hedged.cvar, 20.0)
    recorded = {
        "quantity": [0.1818181818090909],
        "premium": 1.818181818090909,
        "cash": [[98.1818181819091, 98.1818181819091], [98.1818181819091, -1.000000082740371e-09]],
        "shortfall": [0.0, 1.000000082740371e-09],
        "cvar": 1.000000082740371e-09,
        "primary_value": 0.0,
        "cash_dual_beta": [[0.0, 0.0], [0.0, 0.0]],
        "budget_dual_mu": 0.0,
    }
    ours = {
        "quantity": after.quantities,
        "premium": after.hedged.premium,
        "cash": after.hedged.cash,
        "shortfall": after.hedged.deficit,
        "cvar": after.hedged.cvar,
        "primary_value": after.primary_objective,
        "cash_dual_beta": after.cash_duals,
        "budget_dual_mu": after.budget_dual,
    }
    for key, value in recorded.items():
        assert allclose(ours[key], value, atol=1e-7), key
    # q = 2/11 at premium 20/11, and the deficit to zero: the oracle's own figure, 1e-9, is its tolerance at work.
    assert isclose(after.quantities[0], 2 / 11, atol=1e-7) and isclose(after.hedged.premium, 20 / 11, atol=1e-7)
    assert after.hedged.cvar <= after.tolerance


# ------------------------------------------------------------------ the addendum's fixtures (addendum §13)


def test_a_perfect_payout_after_the_cash_loss_leaves_the_timing_gap():
    # Addendum §6.6: cash 100, a loss of 120 at day 30, a hedge costing 10 at day 0 that pays the whole 120. (A.T06 is
    # covered by tests/contract/test_timing_gap.py, where the payout's dates are the settlement evaluator's.)
    unhedged = business([1], 100, [[0, 120, 0]], alpha="0.9", budget=10)
    late = instrument("late", 10, [[0, 0, 120]], 1)
    early = instrument("early", 10, [[0, 120, 0]], 1)
    before = solve(unhedged)
    assert before.hedged.deficit == (20,)
    # Held, the late payout matches the loss at the horizon: the terminal residual is the premium and nothing more...
    held = evaluate(unhedged.with_candidate(late), [1])
    assert held.loss == held.receipts == (120,) and held.residual == (held.premium,) == (10,)
    # ...and the deficit, at day 30, is 30: worse than having no hedge.
    assert held.deficit == (30,) and held.breach_dates == ((GRID[1],),)
    # Offered to the optimizer, the late payout buys nothing and the gap at day 30 stays; the same payout at day 30
    # closes it. The reduced costs alone do not separate them: the shortfall persists to day 60, where the late payout
    # would help, so a dual that weighs day 60 prices both at -110. The re-solves measure the difference.
    with_late = try_candidate(before, late)
    assert with_late.after.quantities == (0,) and with_late.improvement == 0
    assert with_late.after.hedged.deficit == (20,) and GRID[1] in with_late.after.hedged.breach_dates[0]
    with_early = try_candidate(before, early)
    assert with_early.after.hedged.cvar <= with_early.after.tolerance
    assert with_early.after.hedged.breach_dates == ((),) and with_early.after.hedged.breach_probability == 0


@pytest.mark.covers("A.T07")
def test_a_fully_funded_business_buys_no_irrelevant_protection():
    # Addendum §6.5: cash never falls below the floor, so the objective is flat in the quantities, and the premium
    # tie-break buys nothing, whether the instrument is dominated (its receipts never return its cost) or pays well in a
    # state where the business needs nothing.
    funded = business(
        ["0.5", "0.5"], 1000, [[0, 300, 0], [0, 600, 100]], alpha="0.9", budget=1000,
        candidates=[
            instrument("dominated", 10, [[0, 4, 4], [0, 0, 6]], 5),
            instrument("windfall", 10, [[0, 0, 100], [0, 0, 0]], 5),
        ],
    )  # fmt: skip
    s = solve(funded)
    assert s.unhedged.cvar == 0
    assert evaluate(funded, [5, 5]).cvar == 0  # flat: holding both to their caps leaves the objective at zero too
    assert s.quantities == (0, 0) and s.hedged.premium == 0 and s.hedged.cvar == 0


@pytest.mark.covers("A.T08")
def test_a_zero_covariance_tail_hedge_is_bought_where_it_lowers_the_tail():
    # Addendum §6.5: losses (0, 1, 2) and payoff (1, 0, 1), equally likely, price 0.75; cash 1.5, the premium paid at
    # once, alpha 2/3 (the worst third).
    p, loss, payoff = [F(1, 3)] * 3, [0, 1, 2], [1, 0, 1]
    mean = lambda xs: sum(F(x) * px for x, px in zip(xs, p, strict=True))  # noqa: E731
    assert mean([x * z for x, z in zip(loss, payoff, strict=True)]) == mean(loss) * mean(payoff)  # Cov(L, Z) = 0
    problem = business(
        p, "1.5", [[0, x] for x in loss], alpha=F(2, 3), budget=10,
        candidates=[instrument("z", "0.75", [[0, z] for z in payoff], 2)],
    )  # fmt: skip
    s = solve(problem)
    assert s.quantities[0] > 0
    assert s.unhedged.cvar == F(1, 2) and abs(s.hedged.cvar - F(1, 4)) <= s.tolerance
    # One unit exactly: liquidity CVaR 1/2 to 1/4, and the terminal residual's tail 2 to 7/4.
    one = evaluate(problem, [1])
    assert one.cvar == F(1, 4)
    assert discrete_cvar(s.unhedged.residual, p, problem.alpha) == 2
    assert discrete_cvar(one.residual, p, problem.alpha) == F(7, 4)


@pytest.mark.covers("A.T10")
def test_a_negative_reduced_cost_is_re_solved_and_its_improvement_measured():
    # Two equally likely scenarios, each 10 short at day 30; at alpha 1/2 the tail is either one. X costs 1 at day 0 and
    # pays 4 at day 30 in the first scenario only; Y mirrors it in the second. However the duals split the tail between
    # the scenarios (beta_1 + beta_2 = 1), rc_X + rc_Y = 2 - 4 = -2, so at least one is negative; yet either alone adds
    # its premium to the other scenario's deficit, so neither alone improves the CVaR at all.
    base = business(["0.5", "0.5"], 10, [[0, 20], [0, 20]], alpha="0.5", budget=100)
    x = instrument("X", 1, [[0, 4], [0, 0]], 10)
    y = instrument("Y", 1, [[0, 0], [0, 4]], 10)
    before = solve(base)
    assert before.hedged.cvar == 10
    alone = [try_candidate(before, c) for c in (x, y)]
    assert abs(sum(t.reduced_cost for t in alone) + 2) <= D("1e-9")  # the duals are HiGHS's, feasible to 1e-10
    assert min(t.reduced_cost for t in alone) < 0
    for t in alone:
        assert t.after.quantities == (0,) and t.improvement == 0
    # Beside X, Y's reduced cost is negative whatever the split (X at zero needs beta_1 <= 1/4), and here the re-solve
    # finds the improvement: five of each fund both scenarios.
    paired = try_candidate(solve(base.with_candidate(x)), y)
    assert paired.reduced_cost <= -2 + D("1e-9")
    assert paired.improvement == paired.before.hedged.cvar - paired.after.hedged.cvar
    assert abs(paired.improvement - 10) <= paired.after.tolerance
    assert all(abs(q - 5) <= D("1e-6") for q in paired.after.quantities)
    # What holding a candidate would do to cash is all the kernel reports: none of its results has a field that could
    # say whether an instrument exists, is listed or can be bought (addendum §7.2).
    names = {
        f.name
        for cls in (Candidate, CandidateTrial, LiquiditySolution, CashEvaluation, LPResiduals)
        for f in fields(cls)
    }
    claims = ("avail", "tradab", "execut", "listed", "quot", "buy", "order", "recommend", "exist")
    assert not [n for n in names if any(c in n for c in claims)]


@pytest.mark.covers("WP10.budget")
def test_the_premium_budget_is_respected():
    # Check 9's business and instrument: unconstrained it buys 2/11 of a unit for 20/11. With a budget of 1 it can buy a
    # tenth: the budget binds, its dual is the CVaR bought per unit of budget (110 / 10 = 11), and the CVaR falls only
    # to 20 - 11 = 9.
    def problem(budget):
        return business(
            ["0.8", "0.2"], 100, [[0, 0], [0, 120]], alpha="0.9", budget=budget,
            candidates=[instrument("y", 10, [[0, 0], [0, 120]], 1)],
        )  # fmt: skip

    tight = solve(problem(1))
    assert tight.hedged.premium <= 1
    assert abs(tight.hedged.cvar - 9) <= tight.tolerance
    assert abs(tight.budget_dual - 11) <= D("1e-9")
    loose = solve(problem(20))
    assert loose.budget_dual == 0 and abs(loose.hedged.premium - D(20) / 11) <= D("1e-6")


@pytest.mark.covers("WP10.late_payout")
@settings(deadline=None)
@given(
    cash=st.integers(0, 99),
    loss=st.integers(100, 1000),
    payouts=st.lists(
        st.tuples(st.integers(0, 50), st.integers(1, 10_000), st.sampled_from([2, 3]), st.integers(0, 3)),
        min_size=1,
        max_size=3,
    ),
)
def test_a_late_settling_payout_flags_a_breach(cash, loss, payouts):
    # A loss at day 30 that the cash on hand cannot meet, and instruments (price, payout, settlement date, cap) whose
    # payouts, however large, settle after it: any portfolio of them, the optimizer's included, breaches at day 30.
    problem = business(
        [1], cash, [[0, loss, 0, 0]], alpha="0.9", budget=10**6,
        candidates=[
            instrument(f"late{i}", price, [[amount if t == settle else 0 for t in range(4)]], cap)
            for i, (price, amount, settle, cap) in enumerate(payouts)
        ],
    )  # fmt: skip
    solved = solve(problem)
    held = evaluate(problem, [c.cap for c in problem.candidates])
    for portfolio in (solved.unhedged, solved.hedged, held):
        assert GRID[1] in portfolio.breach_dates[0]
        assert portfolio.deficit[0] >= loss - cash and portfolio.breach_probability == 1
