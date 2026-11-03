"""Discrete CVaR by the Rockafellar-Uryasev minimization (addendum A06 §6.3), exact at its limits."""

from datetime import date
from decimal import Decimal as D
from fractions import Fraction as F

import pytest
from hypothesis import given
from hypothesis import strategies as st

from requisite.optimize import LiquidityProblem, discrete_cvar, expected_liquidity_shortfall, solve


def tail_mass_cvar(losses, probabilities, alpha):
    """An independent computation: the worst 1 - alpha of probability mass, taken from the top, an atom at the boundary
    contributing only the part of its mass that fits, averaged."""
    room, total = 1 - F(alpha), F(0)
    for x, p in sorted(zip(map(F, losses), map(F, probabilities), strict=True), reverse=True):
        take = min(p, room)
        total += take * x
        room -= take
    return total / (1 - F(alpha))


@pytest.mark.covers("O05")
def test_the_oracle_check_5_atom_at_the_var():
    # verify_math.py check 5, verified_synthetic_results.json "discrete_cvar_at_atom": 50.000000000000014 (float), with
    # the oracle's np.isclose: the 90% tail holds all of the 100-loss mass and as much again of the zero-loss mass.
    cv = discrete_cvar([D(0), D(100)], [D("0.95"), D("0.05")], D("0.90"))
    assert cv == 50
    assert float(cv) == pytest.approx(50.000000000000014, rel=1e-5, abs=1e-8)


@pytest.mark.covers("A.T09")
def test_an_atom_at_the_var_enters_the_objective_with_its_fractional_tail_mass():
    # Deficits 0, 50, 100 with probabilities 0.5, 0.3, 0.2 at alpha 0.75: the worst 25% is all of the 100 atom (20%)
    # and a quarter of the 50 atom (5%), which sits at the VaR. CVaR = (0.2 * 100 + 0.05 * 50) / 0.25 = 90.
    p, alpha = [F(1, 2), F(3, 10), F(1, 5)], F(3, 4)
    deficits = [D(0), D(50), D(100)]
    assert discrete_cvar(deficits, p, alpha) == 90
    # Not the conditional mean at or above the VaR (all of the 50 atom: 70), nor strictly above it (100).
    assert (F(3, 10) * 50 + F(1, 5) * 100) / (F(3, 10) + F(1, 5)) == 70
    # The liquidity LP's objective, with nothing to buy, is the same functional of the same deficits: cash 0, -50,
    # -100 at one date with a zero floor.
    business = LiquidityProblem(
        dates=(date(2026, 1, 1),),
        probabilities=tuple(p),
        initial_cash=D(0),
        operating=[[0], [0], [0]],
        loss=[[d] for d in deficits],
        reserve=[0],
        alpha=alpha,
        budget=D(0),
    )
    solved = solve(business)
    assert abs(solved.primary_objective - 90) <= solved.tolerance
    assert solved.hedged.cvar == 90


exact_losses = st.lists(st.integers(-(10**6), 10**6).map(lambda i: D(i).scaleb(-2)), min_size=1, max_size=8)


@st.composite
def loss_distributions(draw):
    losses = draw(exact_losses)
    weights = draw(st.lists(st.integers(0, 6), min_size=len(losses), max_size=len(losses)).filter(any))
    p = [F(w, sum(weights)) for w in weights]
    # Atoms are likely: repeat some losses so that ties straddle the quantile.
    if draw(st.booleans()):
        losses = [losses[0] if draw(st.booleans()) else x for x in losses]
    alpha = draw(st.one_of(st.sampled_from([F(0), F(1, 2), F(9, 10), F(99, 100)]), st.fractions(0, F(99, 100))))
    return losses, p, alpha


@given(loss_distributions())
def test_the_minimization_equals_the_tail_mass_at_every_level(case):
    losses, p, alpha = case
    assert discrete_cvar(losses, p, alpha) == tail_mass_cvar(losses, p, alpha)


@given(loss_distributions())
def test_the_limits_of_the_level(case):
    losses, p, _ = case
    # alpha = 0 is the mean; once the tail 1 - alpha is no more than the mass of the worst atom, the worst loss.
    assert discrete_cvar(losses, p, 0) == sum(F(x) * px for x, px in zip(losses, p, strict=True))
    worst = max(x for x, px in zip(losses, p, strict=True) if px)
    top = sum(px for x, px in zip(losses, p, strict=True) if x == worst)
    assert discrete_cvar(losses, p, 1 - top) == worst


def test_expected_liquidity_shortfall_is_the_mean_deficit():
    assert expected_liquidity_shortfall([D(0), D(100)], [D("0.95"), D("0.05")]) == 5
    with pytest.raises(ValueError, match="nonnegative"):
        expected_liquidity_shortfall([D(-1), D(100)], [D("0.95"), D("0.05")])


def test_probabilities_are_exact_and_never_normalized():
    # Addendum §4.2: incoherent forecasts are rejected, never silently normalized; a float cannot be checked to sum to 1.
    with pytest.raises(ValueError, match="sum to exactly 1"):
        discrete_cvar([D(0), D(100)], [D("0.95"), D("0.06")], D("0.9"))
    with pytest.raises(TypeError, match="exact"):
        discrete_cvar([D(0), D(100)], [0.95, 0.05], D("0.9"))
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        discrete_cvar([D(0), D(100)], [D("0.95"), D("0.05")], 1)
