"""Tie-robust relevance: min-max conditional regret over feasible answers and a finite action menu (addendum A03 §3.3).

For a question Q with feasible answers a and a decision with a finite action menu, ``costs[a][d]`` is J_a(d): the
decision's cost objective for action d under the joint model conditioned on Q = a (the caller conditions; this kernel
takes the resulting numbers). Then

    regret_a(d) = J_a(d) - min_d' J_a(d')
    A_eps(a)    = {d : regret_a(d) <= eps}                      the eps-optimal actions under answer a
    rho(Q, D)   = min_d max_a regret_a(d)

rho = 0 means one action is optimal under every answer; rho <= eps means one action is acceptably good under every
answer, and those actions are exactly the intersection of the A_eps(a). Two answers whose optimum sets merely differ
(a solver picking different tied optima) are not a decision flip while their eps-optimal sets share an action (A.T04).

Rows are the feasible answers only: not_established is an evidence status, not a state of the world (A03 §3.3), so the
caller passes the substantive answers and the model compatible with each. Costs and eps are taken as exact values
(int, float, Decimal or Fraction, each converted without rounding) and the result is exact: no float rounding moves an
action across the tolerance.

A cost may be unbounded, given as None: under that answer the action's objective has no finite bound (a consequence
interval open at one end, addendum A04 §4.1), so its regret there is unbounded and it is in no eps-optimal set. Where an
answer leaves every action unbounded, no action has a bounded regret under it, and rho is unbounded (None) for the
question. An unbounded value is never replaced by a finite stand-in.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

Number = int | float | Decimal | Fraction


@dataclass(frozen=True)
class Regret:
    """Min-max conditional regret of a decision over a question's feasible answers."""

    regret: Fraction | None  # rho: min over actions of the worst regret over answers (None: unbounded)
    epsilon: Fraction  # the economic tolerance the sets below are taken at
    acceptable: frozenset[str]  # the common acceptable actions: worst regret <= eps, the intersection of the A_eps(a)
    minimax: frozenset[str]  # the actions attaining rho
    worst_regret: Mapping[str, Fraction | None]  # each action's worst regret over the answers (None: unbounded)
    optimal_sets: Mapping[str, frozenset[str]]  # each answer's eps-optimal actions A_eps(a)


def _exact(x: Number, what: str) -> Fraction:
    if isinstance(x, bool) or not isinstance(x, int | float | Decimal | Fraction):
        raise TypeError(f"{what} is {x!r}: expected a number")
    if (isinstance(x, float) and not math.isfinite(x)) or (isinstance(x, Decimal) and not x.is_finite()):
        raise ValueError(f"{what} is {x!r}: expected a finite number")
    return Fraction(x)


def minmax_regret(costs: Mapping[str, Mapping[str, Number | None]], *, epsilon: Number) -> Regret:
    """rho(Q, D) and the actions acceptable under every feasible answer, from costs[answer][action] = J_a(d)."""
    eps = _exact(epsilon, "epsilon")
    if eps < 0:
        raise ValueError(f"epsilon is {epsilon!r}: a tolerance is non-negative")
    if not costs:
        raise ValueError("no feasible answers")
    rows = list(costs.items())
    actions = frozenset(rows[0][1])
    if not actions:
        raise ValueError("an empty action menu")
    for answer, row in rows:
        if frozenset(row) != actions:
            raise ValueError(f"answer {answer!r} is costed over {sorted(row)}, not the menu {sorted(actions)}")
    regrets: dict[str, dict[str, Fraction | None]] = {}
    for answer, row in rows:
        exact = {d: None if c is None else _exact(c, f"cost of {d!r} under {answer!r}") for d, c in row.items()}
        bounded = [c for c in exact.values() if c is not None]
        best = min(bounded) if bounded else None
        regrets[answer] = {d: None if c is None or best is None else c - best for d, c in exact.items()}
    worst: dict[str, Fraction | None] = {}
    for d in actions:
        rs = [r[d] for r in regrets.values()]
        worst[d] = None if any(x is None for x in rs) else max(x for x in rs if x is not None)
    finite = [w for w in worst.values() if w is not None]
    rho = min(finite) if finite else None
    return Regret(
        regret=rho,
        epsilon=eps,
        acceptable=frozenset(d for d, w in worst.items() if w is not None and w <= eps),
        minimax=frozenset(d for d, w in worst.items() if w == rho),
        worst_regret=worst,
        optimal_sets={a: frozenset(d for d, r in rs.items() if r is not None and r <= eps) for a, rs in regrets.items()},
    )
