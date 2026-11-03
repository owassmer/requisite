"""The liquidity optimizer and CVaR kernel (spec §8.3, WP10; addendum A06 and C01 §7.2): pure functions over numbers,
with no provider calls and no database. Construction reads its choice, its residual table with liquidity timing and its
dual diagnostics from here, and nothing else in Requisite computes them (spec I7)."""

from requisite.optimize.cvar import discrete_cvar, expected_liquidity_shortfall
from requisite.optimize.liquidity import (
    DEFAULT_EPSILON,
    SOLVER,
    Candidate,
    CandidateTrial,
    CashEvaluation,
    LiquidityProblem,
    LiquiditySolution,
    LPResiduals,
    SolveFailed,
    evaluate,
    monetary_scale,
    reduced_cost,
    solve,
    try_candidate,
)

__all__ = [
    "DEFAULT_EPSILON",
    "SOLVER",
    "Candidate",
    "CandidateTrial",
    "CashEvaluation",
    "LPResiduals",
    "LiquidityProblem",
    "LiquiditySolution",
    "SolveFailed",
    "discrete_cvar",
    "evaluate",
    "expected_liquidity_shortfall",
    "monetary_scale",
    "reduced_cost",
    "solve",
    "try_candidate",
]
