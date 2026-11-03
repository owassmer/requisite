"""Discrete CVaR and the expected liquidity shortfall, in exact arithmetic (addendum A06 §6.3).

Losses are money (Decimal) and probabilities are exact (Fraction, Decimal or int): a float cannot be checked to sum to
exactly one, and incoherent probabilities are refused, never normalized (addendum §4.2). Results are Fractions, since an
expectation under rational probabilities is rational (a third of a dollar has no finite Decimal).
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from fractions import Fraction

Exact = Decimal | Fraction | int


def exact(x: object, what: str) -> Fraction:
    """x as an exact rational; a float, a bool or a non-finite Decimal is refused (its exact value is not what was meant)."""
    if isinstance(x, bool) or not isinstance(x, Decimal | Fraction | int):
        raise TypeError(f"{what} must be exact (Decimal, Fraction or int), not {type(x).__name__}")
    if isinstance(x, Decimal) and not x.is_finite():
        raise ValueError(f"{what} must be finite, not {x}")
    return Fraction(x)


def distribution(probabilities: Sequence[Exact]) -> tuple[Fraction, ...]:
    """Scenario probabilities: nonnegative and summing to exactly one (addendum §4.2: never silently normalized)."""
    p = tuple(exact(x, "a probability") for x in probabilities)
    if not p:
        raise ValueError("a distribution needs at least one scenario")
    if any(x < 0 for x in p):
        raise ValueError("probabilities must be nonnegative")
    if sum(p) != 1:
        raise ValueError(f"probabilities must sum to exactly 1, not {sum(p)}; they are never normalized here")
    return p


def tail_level(alpha: Exact) -> Fraction:
    a = exact(alpha, "alpha")
    if not 0 <= a < 1:
        raise ValueError(f"alpha must lie in [0, 1), not {a}")
    return a


def discrete_cvar(losses: Sequence[Exact], probabilities: Sequence[Exact], alpha: Exact) -> Fraction:
    """CVaR_alpha of a finite-support loss by the Rockafellar-Uryasev minimization (addendum §6.3; R2):

        CVaR_alpha(L) = min over zeta of  zeta + E[(L - zeta)_+] / (1 - alpha).

    The function minimized is convex and piecewise linear with its breakpoints at the atoms, so the minimum is taken
    over the atoms, each evaluated exactly. An atom straddling the alpha-quantile then contributes only the fraction of
    its mass that lies in the tail; E[L | L >= VaR] would count all of it.
    """
    p = distribution(probabilities)
    loss = tuple(exact(x, "a loss") for x in losses)
    if len(loss) != len(p):
        raise ValueError(f"{len(loss)} losses for {len(p)} probabilities")
    a = tail_level(alpha)
    # Atoms from the largest down, with the mass and the first moment strictly above each one.
    atoms: dict[Fraction, Fraction] = {}
    for x, px in zip(loss, p, strict=True):
        atoms[x] = atoms.get(x, Fraction(0)) + px
    above_mass, above_moment = Fraction(0), Fraction(0)
    best: Fraction | None = None
    for z in sorted(atoms, reverse=True):
        value = z + (above_moment - z * above_mass) / (1 - a)
        best = value if best is None or value < best else best
        above_mass += atoms[z]
        above_moment += atoms[z] * z
    assert best is not None  # a distribution has at least one atom
    return best


def expected_liquidity_shortfall(deficits: Sequence[Exact], probabilities: Sequence[Exact]) -> Fraction:
    """E[D], the probability-weighted mean of the scenario deficits D_s >= 0 (addendum §6.3: named for what it is, not
    "expected shortfall", which conventionally means CVaR)."""
    p = distribution(probabilities)
    d = tuple(exact(x, "a deficit") for x in deficits)
    if len(d) != len(p):
        raise ValueError(f"{len(d)} deficits for {len(p)} probabilities")
    if any(x < 0 for x in d):
        raise ValueError("a deficit is nonnegative: D_s = max(0, max_t (r_t - C_st))")
    return sum((px * x for px, x in zip(p, d, strict=True)), Fraction(0))
