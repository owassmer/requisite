"""The liquidity optimizer: CVaR of the maximum dated deficit over scenarios, a dated grid and candidate instruments
(addendum A06 §6.1-6.4; spec §8.3), with the primary solve's duals and each candidate's reduced cost (addendum C01 §7.2).

Accounting (§6.1). Every flow is dated on the grid, and the cash at a grid date counts every flow dated on or before it:

    B_st  = B_0 + sum_{tau<=t} (o_s,tau - l_s,tau)        baseline cash, operating flow taken before the modeled loss
    A_stj = sum_{tau<=t} h_j,s,tau - c_j * 1[t >= t_j]     per-unit net cash of instrument j, its premium paid on t_j
    C_st  = B_st + sum_j A_stj q_j                         cash with the hedge
    D_s   = max(0, max_t (r_t - C_st))                     the deficit below the reserve floor r_t (with r_t = 0, the
                                                           maximum funding deficit)

LP (§6.2): minimize zeta + sum_s p_s u_s / (1 - alpha) subject to d_s >= r_t - C_st for every s and t, d_s >= 0,
u_s >= d_s - zeta, u_s >= 0, c.q <= budget and 0 <= q_j <= cap_j. A second solve minimizes premium among portfolios
whose primary objective is within the declared solver tolerance of the optimum (§6.4). The chosen quantities are then
rechecked through the cash accounting above, done exactly in Decimal from the inputs: the reported cash paths, deficits
and CVaR are that recomputation, never the LP's epigraph variables, which can carry slack outside the tail (§6.4).

Quantities are continuous and nonnegative; minimum lots, collateral, borrowing and quantity-dependent prices are outside
this program (§6.2). Nothing here says whether an instrument exists, is listed, or can be bought (§6.2, §7.2): a
candidate is a payoff and a price the caller supplies, and the kernel reports what holding it would do to cash.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import (
    MAX_EMAX,
    MAX_PREC,
    MIN_EMIN,
    ROUND_FLOOR,
    Context,
    Decimal,
    DivisionByZero,
    Inexact,
    InvalidOperation,
    Overflow,
    localcontext,
)
from fractions import Fraction
from functools import cached_property
from typing import Any

import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

from requisite.optimize.cvar import discrete_cvar, distribution, expected_liquidity_shortfall, tail_level

# Money arithmetic is exact: + and * never round at this precision, and a rounding anywhere raises (Inexact).
_EXACT = Context(
    prec=MAX_PREC, Emax=MAX_EMAX, Emin=MIN_EMIN, traps=[InvalidOperation, DivisionByZero, Overflow, Inexact]
)

# The solver and its own tolerances, which are HiGHS's tightest: its answers are rechecked exactly in any case.
_METHOD = "highs-ds"
_HIGHS = {"presolve": True, "primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}
SOLVER = (
    f"HiGHS dual simplex through scipy {scipy.__version__} linprog(method={_METHOD!r}), "
    f"primal and dual feasibility tolerance {_HIGHS['primal_feasibility_tolerance']:g}"
)
DEFAULT_EPSILON = Decimal("1e-9")  # the oracle's (spec/verify_math.py); the addendum gives no value of its own


class SolveFailed(RuntimeError):
    """HiGHS did not reach an optimum, or its answer did not survive the exact recheck: there is no result to report."""


def _money(x: object, what: str) -> Decimal:
    if isinstance(x, bool) or not isinstance(x, Decimal | int):
        raise TypeError(f"{what} is money and arrives as Decimal (or int), not {type(x).__name__}")
    d = Decimal(x)
    if not d.is_finite():
        raise ValueError(f"{what} must be finite, not {d}")
    return d


def _grid(rows: Sequence[Sequence[object]], n_rows: int, n_cols: int, what: str) -> tuple[tuple[Decimal, ...], ...]:
    out = tuple(tuple(_money(x, what) for x in row) for row in rows)
    if len(out) != n_rows or any(len(row) != n_cols for row in out):
        raise ValueError(f"{what}: expected {n_rows} scenarios by {n_cols} grid dates")
    return out


def _float_to_decimal(v: float) -> Decimal:
    """A solver float as the shortest Decimal that reads back as the same float (its own precision, no more)."""
    d = Decimal(repr(float(v)))
    return d if d else Decimal(0)  # no negative zero


@dataclass(frozen=True)
class Candidate:
    """One instrument a solve may hold, per unit. `ref` is the caller's identifier and is only echoed back.

    price: c_j, the acquisition cost per unit, paid on `premium_date`, a date of the grid. It is a price (a quote or an
        indicative price, which the caller labels), never a probability (spec I9).
    receipts: per scenario, the per-unit cash received on each grid date (the dated transfers h_j,s,tau of addendum
        §5.1 and §6.1, not cumulated: the kernel cumulates them).
    cap: the most units a solve may hold.
    """

    ref: str
    price: Decimal
    premium_date: date
    receipts: tuple[tuple[Decimal, ...], ...]
    cap: Decimal

    def __post_init__(self) -> None:
        price, cap = _money(self.price, "a price"), _money(self.cap, "a quantity cap")
        if price < 0:
            raise ValueError(f"{self.ref}: a price is a nonnegative acquisition cost, not {price}")
        if cap < 0:
            raise ValueError(f"{self.ref}: a quantity cap is nonnegative, not {cap}")
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "cap", cap)
        object.__setattr__(self, "receipts", tuple(tuple(_money(x, "a receipt") for x in r) for r in self.receipts))


@dataclass(frozen=True)
class LiquidityProblem:
    """The business on a dated grid, the risk objective, the budget and the candidates.

    dates: the time grid, strictly increasing; every premium-payment date is on it (§6.1).
    probabilities: p_s, exact, summing to exactly one.
    initial_cash: B_0.
    operating: per scenario, operating cash flow on each grid date, before the modeled loss (§6.1: a flow that already
        includes the loss would count it twice).
    loss: per scenario, the modeled incremental cash loss on each grid date (an outflow when positive).
    reserve: r_t, the required cash floor on each grid date.
    alpha: the CVaR level, in [0, 1).
    budget: the most premium the portfolio may cost.
    """

    dates: tuple[date, ...]
    probabilities: tuple[Fraction, ...]
    initial_cash: Decimal
    operating: tuple[tuple[Decimal, ...], ...]
    loss: tuple[tuple[Decimal, ...], ...]
    reserve: tuple[Decimal, ...]
    alpha: Fraction
    budget: Decimal
    candidates: tuple[Candidate, ...] = ()

    def __post_init__(self) -> None:
        dates = tuple(self.dates)
        if not dates:
            raise ValueError("the time grid needs at least one date")
        if not all(isinstance(d, date) for d in dates):
            raise TypeError("the time grid is made of dates")
        if any(not a < b for a, b in zip(dates, dates[1:], strict=False)):
            raise ValueError("the time grid must be strictly increasing")
        p = distribution(self.probabilities)
        n, t = len(p), len(dates)
        reserve = tuple(_money(x, "a reserve") for x in self.reserve)
        if len(reserve) != t:
            raise ValueError(f"{len(reserve)} reserves for {t} grid dates")
        budget = _money(self.budget, "the budget")
        if budget < 0:
            raise ValueError(f"the budget is nonnegative, not {budget}")
        candidates = tuple(self.candidates)
        refs = [c.ref for c in candidates]
        if len(set(refs)) != len(refs):
            raise ValueError("candidate refs must be distinct")
        for c in candidates:
            self._check(c, n, dates)
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "probabilities", p)
        object.__setattr__(self, "initial_cash", _money(self.initial_cash, "initial cash"))
        object.__setattr__(self, "operating", _grid(self.operating, n, t, "operating cash flow"))
        object.__setattr__(self, "loss", _grid(self.loss, n, t, "loss"))
        object.__setattr__(self, "reserve", reserve)
        object.__setattr__(self, "alpha", tail_level(self.alpha))
        object.__setattr__(self, "budget", budget)
        object.__setattr__(self, "candidates", candidates)

    @staticmethod
    def _check(c: Candidate, n: int, dates: tuple[date, ...]) -> None:
        if c.premium_date not in dates:
            raise ValueError(f"{c.ref}: its premium date {c.premium_date} is not on the time grid")
        if len(c.receipts) != n or any(len(r) != len(dates) for r in c.receipts):
            raise ValueError(f"{c.ref}: receipts must be {n} scenarios by {len(dates)} grid dates")

    def with_candidate(self, candidate: Candidate) -> LiquidityProblem:
        return replace(self, candidates=(*self.candidates, candidate))

    @cached_property
    def baseline(self) -> tuple[tuple[Decimal, ...], ...]:
        """B_st: the cash before any hedge."""
        out = []
        with localcontext(_EXACT):
            for o_s, l_s in zip(self.operating, self.loss, strict=True):
                cash, row = self.initial_cash, []
                for o, loss in zip(o_s, l_s, strict=True):
                    cash += o - loss
                    row.append(cash)
                out.append(tuple(row))
        return tuple(out)

    def net_cash(self, candidate: Candidate) -> tuple[tuple[Decimal, ...], ...]:
        """A_stj: one unit's cumulative receipts less its premium from the premium date on, for any candidate on this
        grid, in the solve or proposed."""
        self._check(candidate, len(self.probabilities), self.dates)
        paid = self.dates.index(candidate.premium_date)
        out = []
        with localcontext(_EXACT):
            for h_s in candidate.receipts:
                total, row = Decimal(0), []
                for t, h in enumerate(h_s):
                    total += h
                    row.append(total - candidate.price if t >= paid else total)
                out.append(tuple(row))
        return tuple(out)

    @cached_property
    def nets(self) -> tuple[tuple[tuple[Decimal, ...], ...], ...]:
        return tuple(self.net_cash(c) for c in self.candidates)


@dataclass(frozen=True)
class CashEvaluation:
    """One portfolio's cash, recomputed exactly from the inputs and the quantities: the authoritative numbers.

    premium: sum_j c_j q_j. cash: C_st per scenario and grid date. deficit: D_s. breach_tolerance: how far below the
    reserve cash must fall to count as a breach (zero for an exact evaluation; a solve's tolerance for its portfolios,
    since a premium-minimizing portfolio may sit that close to a deficit its optimum does not have). breach_dates: per
    scenario, the grid dates on which cash is below the reserve by more than breach_tolerance. loss: per scenario, the
    modeled loss summed over the grid. receipts: per scenario, the hedge's receipts summed over the grid. residual:
    loss - receipts + premium, the terminal residual cost, which ignores when cash arrives (a terminal match is not
    liquidity protection: addendum §6.6). cvar: CVaR_alpha(D). expected_liquidity_shortfall: E[D]. breach_probability:
    P(D > breach_tolerance).
    """

    quantities: tuple[Decimal, ...]
    premium: Decimal
    cash: tuple[tuple[Decimal, ...], ...]
    deficit: tuple[Decimal, ...]
    breach_tolerance: Decimal
    breach_dates: tuple[tuple[date, ...], ...]
    loss: tuple[Decimal, ...]
    receipts: tuple[Decimal, ...]
    residual: tuple[Decimal, ...]
    cvar: Fraction
    expected_liquidity_shortfall: Fraction
    breach_probability: Fraction


def evaluate(
    problem: LiquidityProblem, quantities: Sequence[Decimal | int], breach_tolerance: Decimal = Decimal(0)
) -> CashEvaluation:
    """The cash accounting of §6.1 for a held portfolio, in exact Decimal arithmetic. Any nonnegative quantities may be
    evaluated (a held position need not respect a solve's caps or budget)."""
    q = tuple(_money(x, "a quantity") for x in quantities)
    tol = _money(breach_tolerance, "the breach tolerance")
    if tol < 0:
        raise ValueError(f"the breach tolerance is nonnegative, not {tol}")
    if len(q) != len(problem.candidates):
        raise ValueError(f"{len(q)} quantities for {len(problem.candidates)} candidates")
    if any(x < 0 for x in q):
        raise ValueError("quantities are nonnegative: a price is an acquisition cost")
    held = [(c, a, x) for c, a, x in zip(problem.candidates, problem.nets, q, strict=True) if x]
    with localcontext(_EXACT):
        premium = sum((c.price * x for c, _, x in held), Decimal(0))
        cash = tuple(
            tuple(b + sum((a[s][t] * x for _, a, x in held), Decimal(0)) for t, b in enumerate(b_s))
            for s, b_s in enumerate(problem.baseline)
        )
        deficit = tuple(max([Decimal(0), *(r - c for r, c in zip(problem.reserve, c_s, strict=True))]) for c_s in cash)
        breach = tuple(
            tuple(d for d, r, c in zip(problem.dates, problem.reserve, c_s, strict=True) if r - c > tol) for c_s in cash
        )
        loss = tuple(sum(l_s, Decimal(0)) for l_s in problem.loss)
        receipts = tuple(
            sum((sum(c.receipts[s], Decimal(0)) * x for c, _, x in held), Decimal(0))
            for s in range(len(problem.probabilities))
        )
        residual = tuple(ls - rs + premium for ls, rs in zip(loss, receipts, strict=True))
    p = problem.probabilities
    return CashEvaluation(
        quantities=q,
        premium=premium,
        cash=cash,
        deficit=deficit,
        breach_tolerance=tol,
        breach_dates=breach,
        loss=loss,
        receipts=receipts,
        residual=residual,
        cvar=discrete_cvar(deficit, p, problem.alpha),
        expected_liquidity_shortfall=expected_liquidity_shortfall(deficit, p),
        breach_probability=sum((px for px, d in zip(p, deficit, strict=True) if d > tol), Fraction(0)),
    )


@dataclass(frozen=True)
class LPResiduals:
    """The largest violation, in the solver's scaled units, of each family of constraints at the point the solver
    returned (zero when every constraint holds). primary_bound is the second solve's bound on the primary objective."""

    cash: float
    tail: float
    budget: float
    bounds: float
    primary_bound: float | None


@dataclass(frozen=True)
class LiquiditySolution:
    """A solve's result: the solver's account of itself, and the chosen portfolio recomputed.

    epsilon: the solver tolerance (§6.4), absolute on the scaled primary objective; `tolerance` is it in money.
    scale: the money per LP unit (§6.4: Decimal money is scaled explicitly to a well-conditioned range).
    primary_objective: the LP's optimal CVaR, in money. secondary_objective: the LP's least premium among portfolios
        within epsilon of it, in money.
    hedged: the chosen quantities, recomputed exactly; its cvar lies within `tolerance` of primary_objective.
    unhedged: the same accounting with nothing held.
    cash_duals: beta_st >= 0, the primary solve's multipliers on the cash-shortfall constraints: the objective's
        decrease per unit of cash added in scenario s at date t, at the margin. budget_dual: mu >= 0, its decrease per
        unit of budget. Both are money per money, so the monetary scale, applied alike to the objective and to the cash,
        leaves them as HiGHS returned them; under degeneracy they are one choice among several optimal multipliers.
    reduced_costs: mu c_j - sum_st beta_st A_stj per candidate (C01 §7.2), a diagnostic only.
    """

    problem: LiquidityProblem
    epsilon: Decimal
    scale: Decimal
    solver: str
    primary_status: str
    secondary_status: str
    primary_objective: Decimal
    secondary_objective: Decimal
    primary_residuals: LPResiduals
    secondary_residuals: LPResiduals
    hedged: CashEvaluation
    unhedged: CashEvaluation
    cash_duals: tuple[tuple[Decimal, ...], ...]
    budget_dual: Decimal
    reduced_costs: tuple[Decimal, ...]

    @property
    def quantities(self) -> tuple[Decimal, ...]:
        return self.hedged.quantities

    @property
    def tolerance(self) -> Decimal:
        with localcontext(_EXACT):
            return self.epsilon * self.scale


def monetary_scale(problem: LiquidityProblem) -> Decimal:
    """The power of ten at the leading digit of the largest |B_st - r_t|, the distance between the business's cash and
    its floor that the objective measures: the scaled deficits and objective then lie below ten, and epsilon is relative
    to the business's own cash. A hedge's per-unit amounts are left to the solver's column scaling, since sizing the
    scale by them would let a generous cap widen the tolerance. With no such distance anywhere, the unit. Dividing by a
    power of ten is exact in Decimal."""
    with localcontext(_EXACT):
        m = max(abs(b - r) for b_s in problem.baseline for b, r in zip(b_s, problem.reserve, strict=True))
    if not m:
        return Decimal(1)
    k = m.adjusted()
    return Decimal(10**k) if k >= 0 else Decimal(1).scaleb(k)


@dataclass(frozen=True)
class _LP:
    objective: np.ndarray
    premium: np.ndarray
    a_ub: Any
    b_ub: np.ndarray
    bounds: list[tuple[float | None, float | None]]
    n_cash: int
    n_tail: int


def _program(problem: LiquidityProblem, scale: Decimal) -> _LP:
    """The scaled LP. Variables: q (J), d (S), zeta (1), u (S). Rows: S*T cash rows, S tail rows, the budget row."""
    p, alpha = problem.probabilities, problem.alpha
    n_s, n_t, n_j = len(p), len(problem.dates), len(problem.candidates)
    d0, z, u0 = n_j, n_j + n_s, n_j + n_s + 1
    shift = -scale.adjusted()

    def scaled(x: Decimal) -> float:
        return float(x.scaleb(shift, _EXACT))

    entries: list[tuple[int, int, float]] = []
    rhs: list[float] = []
    for s in range(n_s):
        for t in range(n_t):
            row = s * n_t + t
            entries += [(row, j, -scaled(a[s][t])) for j, a in enumerate(problem.nets) if a[s][t]]
            entries.append((row, d0 + s, -1.0))
            with localcontext(_EXACT):
                rhs.append(scaled(problem.baseline[s][t] - problem.reserve[t]))
    for s in range(n_s):
        row = n_s * n_t + s
        entries += [(row, d0 + s, 1.0), (row, z, -1.0), (row, u0 + s, -1.0)]
        rhs.append(0.0)
    budget_row = n_s * n_t + n_s
    entries += [(budget_row, j, scaled(c.price)) for j, c in enumerate(problem.candidates) if c.price]
    rhs.append(scaled(problem.budget))
    rows, cols, vals = zip(*entries, strict=True)
    n_var = n_j + 2 * n_s + 1
    objective = np.zeros(n_var)
    objective[z] = 1.0
    objective[u0:] = [float(px / (1 - alpha)) for px in p]
    premium = np.zeros(n_var)
    premium[:n_j] = [scaled(c.price) for c in problem.candidates]
    a_ub = coo_matrix((vals, (rows, cols)), shape=(budget_row + 1, n_var)).tocsr()
    bounds: list[tuple[float | None, float | None]] = [(0.0, float(c.cap)) for c in problem.candidates]
    bounds += [(0.0, None)] * n_s + [(None, None)] + [(0.0, None)] * n_s
    return _LP(objective, premium, a_ub, np.asarray(rhs), bounds, n_s * n_t, n_s)


def _highs(c: np.ndarray, a_ub: Any, b_ub: np.ndarray, bounds: list[tuple[float | None, float | None]]) -> Any:
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method=_METHOD, options=_HIGHS)
    if res.status != 0:
        raise SolveFailed(f"HiGHS reached no optimum: {res.message}")
    return res


def _residuals(lp: _LP, a_ub: Any, b_ub: np.ndarray, x: np.ndarray, bounded: bool) -> LPResiduals:
    over = np.maximum(a_ub @ x - b_ub, 0.0)
    lo = np.array([-np.inf if b[0] is None else b[0] for b in lp.bounds])
    hi = np.array([np.inf if b[1] is None else b[1] for b in lp.bounds])
    k = lp.n_cash + lp.n_tail
    return LPResiduals(
        cash=float(over[: lp.n_cash].max(initial=0.0)),
        tail=float(over[lp.n_cash : k].max(initial=0.0)),
        budget=float(over[k]),
        bounds=float(np.maximum(np.maximum(lo - x, x - hi), 0.0).max(initial=0.0)),
        primary_bound=float(over[k + 1]) if bounded else None,
    )


def _held(problem: LiquidityProblem, x: np.ndarray) -> tuple[Decimal, ...]:
    """The solver's quantities at their own precision, put back inside the caps and the budget exactly (a float answer
    can sit a rounding error outside them); the exact recheck then decides whether the result stands (§6.4)."""
    q = [min(max(_float_to_decimal(v), Decimal(0)), c.cap) for v, c in zip(x, problem.candidates, strict=True)]
    with localcontext(_EXACT):
        premium = sum((c.price * v for c, v in zip(problem.candidates, q, strict=True)), Decimal(0))
    if premium > problem.budget:
        # Scale every quantity down by budget/premium, rounding each down, so that sum c_j q_j <= budget exactly.
        with localcontext(Context(prec=34, rounding=ROUND_FLOOR)):
            q = [v * problem.budget / premium for v in q]
    return tuple(q)


def _reduced_cost(
    problem: LiquidityProblem,
    beta: tuple[tuple[Decimal, ...], ...],
    mu: Decimal,
    candidate: Candidate,
) -> Decimal:
    a = problem.net_cash(candidate)
    with localcontext(_EXACT):
        shadow = sum(
            (b * x for b_s, a_s in zip(beta, a, strict=True) for b, x in zip(b_s, a_s, strict=True)), Decimal(0)
        )
        return mu * candidate.price - shadow


def solve(problem: LiquidityProblem, epsilon: Decimal = DEFAULT_EPSILON) -> LiquiditySolution:
    """Minimize CVaR_alpha of the maximum dated deficit; then, among portfolios within epsilon of that optimum on the
    scaled objective, minimize premium (§6.4), so a flat objective never buys arbitrarily. The second solve is held to
    half of epsilon; the other half absorbs the solver's floating point, and the exact recheck confirms that the chosen
    portfolio's CVaR lies within epsilon of the optimum (both measured on the scaled objective) or the solve fails.
    Breaches, with and without the hedge, are counted beyond that same tolerance in money: the second solve may leave a
    portfolio that close to a deficit its optimum does not have."""
    eps = _money(epsilon, "the solver tolerance")
    if eps <= 0:
        raise ValueError(f"the solver tolerance is positive, not {eps}")
    scale = monetary_scale(problem)
    lp = _program(problem, scale)
    n_s, n_t, n_j = len(problem.probabilities), len(problem.dates), len(problem.candidates)

    primary = _highs(lp.objective, lp.a_ub, lp.b_ub, lp.bounds)
    a2 = vstack([lp.a_ub, lp.objective[None, :]]).tocsr()
    b2 = np.r_[lp.b_ub, primary.fun + float(eps) / 2]
    secondary = _highs(lp.premium, a2, b2, lp.bounds)

    with localcontext(_EXACT):
        j_star = _float_to_decimal(primary.fun) * scale
        tolerance = eps * scale
    hedged = evaluate(problem, _held(problem, secondary.x[:n_j]), tolerance)
    gap = hedged.cvar - Fraction(j_star)
    if abs(gap) > Fraction(tolerance):
        raise SolveFailed(
            f"the chosen portfolio's recomputed CVaR {float(hedged.cvar)!r} is {float(gap)!r} from the LP optimum "
            f"{j_star}, beyond the declared tolerance {tolerance} (epsilon {eps} at scale {scale})"
        )
    marginals = primary.ineqlin.marginals
    beta = tuple(tuple(_float_to_decimal(-marginals[s * n_t + t]) for t in range(n_t)) for s in range(n_s))
    mu = _float_to_decimal(-marginals[-1])
    with localcontext(_EXACT):
        secondary_objective = _float_to_decimal(secondary.fun) * scale
    return LiquiditySolution(
        problem=problem,
        epsilon=eps,
        scale=scale,
        solver=SOLVER,
        primary_status=str(primary.message),
        secondary_status=str(secondary.message),
        primary_objective=j_star,
        secondary_objective=secondary_objective,
        primary_residuals=_residuals(lp, lp.a_ub, lp.b_ub, primary.x, bounded=False),
        secondary_residuals=_residuals(lp, a2, b2, secondary.x, bounded=True),
        hedged=hedged,
        unhedged=evaluate(problem, [0] * n_j, tolerance),
        cash_duals=beta,
        budget_dual=mu,
        reduced_costs=tuple(_reduced_cost(problem, beta, mu, c) for c in problem.candidates),
    )


def reduced_cost(solution: LiquiditySolution, candidate: Candidate) -> Decimal:
    """mu c_j - sum_st beta_st A_stj against the solution's primary duals (C01 §7.2), for a candidate in the solve or
    one proposed on the same grid. Negative marks a candidate worth re-solving with; it guarantees no improvement under
    degeneracy and says nothing of price or availability (§7.2). `try_candidate` measures the improvement."""
    return _reduced_cost(solution.problem, solution.cash_duals, solution.budget_dual, candidate)


@dataclass(frozen=True)
class CandidateTrial:
    """A proposed candidate scored against the current duals, and the problem re-solved with it added: the reduced cost
    is the diagnostic, the re-solve is the measurement."""

    candidate: Candidate
    reduced_cost: Decimal
    before: LiquiditySolution
    after: LiquiditySolution

    @property
    def improvement(self) -> Fraction:
        """The CVaR of the portfolio chosen before less that of the portfolio chosen with the candidate, each
        recomputed exactly; each lies within its solve's tolerance of its optimum."""
        return self.before.hedged.cvar - self.after.hedged.cvar


def try_candidate(solution: LiquiditySolution, candidate: Candidate) -> CandidateTrial:
    """Score a proposed candidate against the solution's duals, then re-solve with it added at the same epsilon."""
    after = solve(solution.problem.with_candidate(candidate), solution.epsilon)
    return CandidateTrial(candidate, reduced_cost(solution, candidate), solution, after)


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
    "evaluate",
    "monetary_scale",
    "reduced_cost",
    "solve",
    "try_candidate",
]
