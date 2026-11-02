"""Feasible MC/DC coverage of a trigger under world constraints, and boundary values (spec §8.6 step 1; addendum A09).

The trigger and the world constraints arrive in a small boolean form defined here: ``Var`` for an atomic condition,
``Not``, ``And`` and ``Or``. The contract IR lowers into it, one variable per atomic condition. The trigger's conditions
are the variables it mentions. A constraint may also mention conditions the trigger does not (a proposition that links
two trigger conditions); those are the *other* conditions. A state assigns each of the trigger's conditions true or
false, and it is feasible when some assignment of the other conditions satisfies every constraint. Each feasible state
the account shows comes with such an assignment, its completion: the certificate that it is feasible, checkable by
evaluating the constraints. Witnesses, redundancy and groups are over the trigger's conditions; in each state the other
conditions take any values the constraints allow.

For each of the trigger's conditions x_i the account is exact, with no third case (A09 §10.2):

- a feasible unique-cause witness: two feasible states that differ in x_i alone and on which the trigger differs; or
- none exists. By definition that means the trigger, on the feasible states, is a function of the other trigger
  conditions: x_i has no independent effect under the stated constraints, so it is redundant under them. A linked
  threshold or a repeated predicate is one case of this, through the constraints. The proof is kept: the exhaustive
  search over every assignment, or the solver's UNSAT.

Redundancy is per condition, given the others: two conditions can each be redundant while removing both changes the
trigger (A == B under A AND B). So the account also gives, completely, the minimal groups of redundant conditions whose
joint removal changes the trigger on the feasible states, each with two feasible states that agree on every condition
outside the group and differ in outcome. A set of redundant conditions can be removed together exactly when it contains
no group. The search is exponential only in the number of redundant conditions, and always runs to completion.

Two findings about the whole trigger come first: constraints with no feasible state, and a trigger constant on the
feasible states (it never fires, or always fires). Either makes every per-condition statement vacuous, so neither is
reported as conditions being removable.

Boundary values (spec §8.6 step 1; A09 §10.3) are made for numeric thresholds at their declared precision, temporal
cutoffs at the exact instant and its neighbours at the comparison's resolution, and counts at k - 1, k and k + 1. A
boundary point fixes the value of the trigger condition its comparison defines; where no feasible state has that value
the point is infeasible, and the reason kept is every minimal set of constraints that excludes it.

Given scenarios, ``meets`` says which obligations they meet, which they do not, and which no feasible scenario can meet
(A09: store uncovered and infeasible obligations explicitly). The obligations are each witnessed condition's
independence, each group's joint effect (the coverage a coupled group can have, since none of its conditions has an
independent one), and each boundary point. An infeasible scenario meets nothing.

Search is by enumeration of every assignment while the trigger's and the other conditions number at most
``ENUMERATE_UP_TO``, and by SAT (python-sat) above. Both are exact; they differ in their proof and their cost. The
exhaustive search is its own proof, which a reader can check without trusting a solver, and its cost doubles with each
condition, while SAT's barely moves. Measured on random triggers over every condition, with n/2 implication constraints
and one boundary point the constraints exclude (so the minimal reasons are searched too), the median whole account took
3 ms by enumeration against 0.4 ms by SAT at 10 conditions, 13 ms against 0.4 ms at 12, and 260 ms against 0.6 ms at
16. Ten keeps the checkable proof for every trigger of up to ten conditions, at a few milliseconds, and stops before
the doubling makes enumeration the slow part of a fuzzing run. The size is a parameter, so either path can be forced.
"""

from __future__ import annotations

import operator
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal

from pysat.solvers import Solver

ENUMERATE_UP_TO = 10
SOLVER = "minisat22"


# ---------------------------------------------------------------- the expression form


@dataclass(frozen=True)
class Var:
    """An atomic condition."""

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"a condition is named by a non-empty string, not {self.name!r}")


@dataclass(frozen=True)
class Not:
    item: Expr


@dataclass(frozen=True, init=False)
class And:
    """True when every item is; an empty And is true."""

    items: tuple[Expr, ...]

    def __init__(self, *items: Expr) -> None:
        object.__setattr__(self, "items", tuple(items))


@dataclass(frozen=True, init=False)
class Or:
    """True when some item is; an empty Or is false."""

    items: tuple[Expr, ...]

    def __init__(self, *items: Expr) -> None:
        object.__setattr__(self, "items", tuple(items))


Expr = Var | Not | And | Or


def variables(expr: Expr) -> tuple[str, ...]:
    """The conditions an expression mentions, in order of first appearance."""
    seen: dict[str, None] = {}

    def walk(e: Expr) -> None:
        if isinstance(e, Var):
            seen.setdefault(e.name, None)
        elif isinstance(e, Not):
            walk(e.item)
        elif isinstance(e, And | Or):
            for x in e.items:
                walk(x)
        else:
            raise TypeError(f"{e!r} is not an expression of this form (Var, Not, And, Or)")

    walk(expr)
    return tuple(seen)


def truth(expr: Expr, state: Mapping[str, bool]) -> bool:
    """The expression's value in a state that assigns each of its conditions."""
    names = variables(expr)
    return _compile(expr, {c: i for i, c in enumerate(names)})(sum(1 << i for i, c in enumerate(names) if state[c]))


def _compile(expr: Expr, index: Mapping[str, int]) -> Callable[[int], bool]:
    """The expression as a function of an assignment packed into an int, bit i holding the condition indexed i."""
    if isinstance(expr, Var):
        bit = 1 << index[expr.name]
        return lambda s: bool(s & bit)
    if isinstance(expr, Not):
        f = _compile(expr.item, index)
        return lambda s: not f(s)
    fs = [_compile(x, index) for x in expr.items]
    if isinstance(expr, And):
        return lambda s: all(f(s) for f in fs)
    return lambda s: any(f(s) for f in fs)


# ---------------------------------------------------------------- results

State = tuple[tuple[str, bool], ...]  # each condition with its value, in the account's order of those conditions


@dataclass(frozen=True)
class Pair:
    """Two feasible states of the trigger's conditions on which the trigger differs: it does not fire on the first and
    fires on the second. ``completions`` gives, for each, values of the other conditions under which every constraint
    holds (empty when the constraints mention only trigger conditions)."""

    not_fired: State
    fired: State
    completions: tuple[State, State]


@dataclass(frozen=True)
class Exhaustive:
    """Shown by enumeration: every assignment was evaluated, and none (or no pair) answered the query."""

    assignments: int  # 2^(n + m): every assignment of the trigger's n conditions and the m other conditions
    feasible: int  # a pair query: the feasible states of the trigger's conditions; a state query: zero, none met it
    pairs: int  # a pair query: the pairs of feasible states agreeing where it required, each with equal outcomes


@dataclass(frozen=True)
class Unsat:
    """Shown by SAT: the solver proved the query's formula unsatisfiable."""

    solver: str
    variables: int
    clauses: int


Proof = Exhaustive | Unsat


@dataclass(frozen=True)
class ConditionAccount:
    """One condition's MC/DC account: a feasible unique-cause witness, or the proof that none exists."""

    condition: str
    witness: Pair | None  # the two states differ in this condition alone
    redundancy: Proof | None  # set exactly when there is no witness: the trigger is a function of the other conditions

    @property
    def redundant(self) -> bool:
        return self.witness is None


@dataclass(frozen=True)
class Group:
    """A minimal set of redundant conditions whose joint removal changes the trigger on the feasible states."""

    conditions: frozenset[str]
    pair: Pair  # two feasible states that agree on every trigger condition outside the group and differ in outcome


@dataclass(frozen=True)
class Finding:
    """A finding about the whole trigger, which makes every per-condition statement vacuous."""

    kind: Literal["no_feasible_state", "never_fires", "always_fires"]
    proof: Proof  # no feasible state at all, or no two feasible states with different outcomes
    example: State | None  # a constant trigger: a feasible state, on which it takes its one value
    completion: State  # the example's values of the other conditions, under which every constraint holds
    excluded_by: tuple[frozenset[str], ...]  # no feasible state: every minimal set of constraint ids with none


Value = Decimal | int | datetime | date
Op = Literal["<", "<=", ">", ">=", "==", "!="]
_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def _instant(v: Any) -> Any:
    """Values compared as what they mean: an aware datetime as its UTC instant, so that wall-clock times in one zone
    across a daylight-saving change compare by instant (Python compares same-zone datetimes by wall clock)."""
    return v.astimezone(UTC) if isinstance(v, datetime) else v


@dataclass(frozen=True)
class BoundaryPoint:
    """A value at or beside a comparison's boundary. The truth it gives the condition the comparison defines, and
    whether it lies in the comparison's domain, are computed from the point, so they cannot disagree with it."""

    condition: str
    op: Op
    threshold: Value
    offset: int  # -1, 0 or +1 steps (precision, resolution or one count) from the threshold
    value: Value
    domain: tuple[Value | None, Value | None] = (None, None)  # inclusive bounds on the values the quantity can take

    def __post_init__(self) -> None:
        if self.op not in _OPS:
            raise ValueError(f"comparison {self.op!r} is not one of {sorted(_OPS)}")

    @property
    def holds(self) -> bool:
        """The condition's value at this point: (value op threshold)."""
        return _OPS[self.op](_instant(self.value), _instant(self.threshold))

    @property
    def in_domain(self) -> bool:
        lo, hi = self.domain
        at = _instant(self.value)
        return (lo is None or at >= _instant(lo)) and (hi is None or at <= _instant(hi))


@dataclass(frozen=True)
class PointAccount:
    """Whether a boundary point can occur in a feasible state."""

    point: BoundaryPoint
    example: State | None  # a feasible state with the condition at the point's value; None when infeasible
    completion: State  # the example's values of the other conditions, under which every constraint holds
    excluded_by: tuple[frozenset[str], ...]  # infeasible in the domain: every minimal set of constraint ids excluding it

    @property
    def feasible(self) -> bool:
        return self.example is not None


@dataclass(frozen=True)
class Coverage:
    """The feasible coverage account of one trigger under its constraints."""

    trigger: Expr
    constraints: Mapping[str, Expr]
    conditions: tuple[str, ...]  # the trigger's atomic conditions, in order of first appearance
    others: tuple[str, ...]  # the conditions only the constraints mention, in order of first appearance
    method: Literal["enumeration", "sat"]
    finding: Finding | None  # a whole-trigger finding; when present, accounts and groups are empty
    accounts: tuple[ConditionAccount, ...]  # one per condition, in condition order
    groups: tuple[Group, ...]  # every minimal group of redundant conditions whose joint removal changes the trigger
    points: tuple[PointAccount, ...]

    @property
    def redundant(self) -> tuple[str, ...]:
        return tuple(a.condition for a in self.accounts if a.redundant)


# ---------------------------------------------------------------- minimal sets of a monotone predicate


def _minimal_sets[T: Hashable](ground: Sequence[T], holds: Callable[[frozenset[T]], bool]) -> tuple[frozenset[T], ...]:
    """Every minimal subset of ``ground`` on which ``holds`` is true, for a predicate closed under supersets.

    MARCO (Liffiton et al., 2016): a solver over one variable per element proposes an unexplored subset; one on which
    the predicate holds is shrunk to a minimal one and its supersets blocked, one on which it fails is grown to a maximal
    failing one and its subsets blocked. Each round blocks the subset it was given, so the search ends, and every
    minimal set is found. The result is ordered by size, then by the elements' order in ``ground``.
    """
    order = {g: k for k, g in enumerate(ground, 1)}
    if not holds(frozenset(ground)):
        return ()
    if holds(frozenset()):
        return (frozenset(),)
    found: list[frozenset[T]] = []
    with Solver(name=SOLVER) as explore:
        while explore.solve():
            chosen = {lit for lit in explore.get_model() or [] if lit > 0}
            current = {g for g in ground if order[g] in chosen}
            if holds(frozenset(current)):
                for g in ground:
                    if g in current and holds(frozenset(current - {g})):
                        current.discard(g)
                found.append(frozenset(current))
                explore.add_clause([-order[g] for g in current])
            else:
                for g in ground:
                    if g not in current and not holds(frozenset(current | {g})):
                        current.add(g)
                explore.add_clause([order[g] for g in ground if g not in current])
    return tuple(sorted(found, key=lambda s: (len(s), sorted(order[g] for g in s))))


# ---------------------------------------------------------------- the two search paths
# An assignment is an int over the trigger's n conditions (bits 0 to n - 1) and then the m other conditions. Each path
# answers two queries:
#   satisfying(fixed, ids): an assignment with the fixed values (by bit) satisfying the constraints in ids, or the
#     proof that there is none;
#   pair(free): two assignments satisfying every constraint whose trigger conditions agree outside the bit set free,
#     the first not firing and the second firing, or the proof that there are none.


class _Enumeration:
    method: Literal["enumeration", "sat"] = "enumeration"

    def __init__(
        self, n: int, m: int, trigger: Callable[[int], bool], constraints: Mapping[str, Callable[[int], bool]]
    ) -> None:
        self.n, self.size = n, 1 << (n + m)
        self.mask = (1 << n) - 1
        self.bit = {cid: 1 << j for j, cid in enumerate(constraints)}
        self.violated: list[int] = []
        for s in range(self.size):
            v = 0
            for cid, holds in constraints.items():
                if not holds(s):
                    v |= self.bit[cid]
            self.violated.append(v)
        # Each feasible state of the trigger's conditions, with the first assignment that completes it.
        self.feasible: dict[int, int] = {}
        for s, v in enumerate(self.violated):
            if v == 0:
                self.feasible.setdefault(s & self.mask, s)
        self.fires = {t: trigger(t) for t in self.feasible}

    def __enter__(self) -> _Enumeration:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def satisfying(self, fixed: Mapping[int, bool], ids: Iterable[str]) -> int | Proof:
        mask = sum(1 << i for i in fixed)
        want = sum(1 << i for i, v in fixed.items() if v)
        used = sum(self.bit[cid] for cid in ids)
        for s, v in enumerate(self.violated):
            if s & mask == want and not v & used:
                return s
        return Exhaustive(assignments=self.size, feasible=0, pairs=0)

    def pair(self, free: int) -> tuple[int, int] | Proof:
        keep = self.mask & ~free
        first: dict[tuple[int, bool], int] = {}
        for t, s in self.feasible.items():
            key, fires = t & keep, self.fires[t]
            first.setdefault((key, fires), s)
            if (key, not fires) in first:
                return (first[(key, False)], first[(key, True)])
        sizes = Counter(t & keep for t in self.feasible)
        return Exhaustive(
            assignments=self.size, feasible=len(self.feasible), pairs=sum(c * (c - 1) // 2 for c in sizes.values())
        )


class _Tseitin:
    """Clauses that give each sub-expression a variable equal to it (Tseitin's encoding), over fresh variables."""

    def __init__(self, first_free: int) -> None:
        self.top = first_free - 1
        self.clauses: list[list[int]] = []

    def fresh(self) -> int:
        self.top += 1
        return self.top

    def encode(self, expr: Expr, var: Mapping[str, int]) -> int:
        if isinstance(expr, Var):
            return var[expr.name]
        if isinstance(expr, Not):
            return -self.encode(expr.item, var)
        lits = [self.encode(x, var) for x in expr.items]
        v = self.fresh()
        if isinstance(expr, And):
            self.clauses += [[-v, lit] for lit in lits]
            self.clauses.append([v] + [-lit for lit in lits])
        else:
            self.clauses += [[-lit, v] for lit in lits]
            self.clauses.append([-v] + lits)
        return v


class _Sat:
    method: Literal["enumeration", "sat"] = "sat"

    def __init__(
        self, conditions: Sequence[str], others: Sequence[str], trigger: Expr, constraints: Mapping[str, Expr]
    ) -> None:
        names = (*conditions, *others)
        n, t = len(conditions), len(names)
        self.n = n
        # Pairs: two copies of every condition, x (firing) and y (not firing), each satisfying every constraint, with a
        # selector per trigger condition that, assumed true, makes the copies agree on it. The other conditions are
        # free in each copy.
        x = {c: 1 + i for i, c in enumerate(names)}
        y = {c: 1 + t + i for i, c in enumerate(names)}
        self.agree = [1 + 2 * t + i for i in range(n)]
        enc = _Tseitin(1 + 2 * t + n)
        for i, c in enumerate(conditions):
            enc.clauses += [[-self.agree[i], -x[c], y[c]], [-self.agree[i], x[c], -y[c]]]
        for e in constraints.values():
            enc.clauses += [[enc.encode(e, x)], [enc.encode(e, y)]]
        enc.clauses += [[enc.encode(trigger, x)], [-enc.encode(trigger, y)]]
        self.x, self.y = x, y
        self.pairs = Solver(name=SOLVER, bootstrap_with=enc.clauses)
        self.pairs_size = (enc.top, len(enc.clauses))
        # States: one copy z, with a selector per constraint that, assumed true, makes the constraint hold. A condition
        # no clause mentions is free; the solver creates it when an assumption names it.
        z = {c: 1 + i for i, c in enumerate(names)}
        self.selector = {cid: 1 + t + j for j, cid in enumerate(constraints)}
        enc = _Tseitin(1 + t + len(constraints))
        for cid, e in constraints.items():
            enc.clauses.append([-self.selector[cid], enc.encode(e, z)])
        self.z = z
        self.states = Solver(name=SOLVER, bootstrap_with=enc.clauses)
        self.states_size = (enc.top, len(enc.clauses))

    def __enter__(self) -> _Sat:
        return self

    def __exit__(self, *exc: object) -> None:
        self.pairs.delete()
        self.states.delete()

    @staticmethod
    def _assignment(model: list[int] | None, var: Mapping[str, int]) -> int:
        # A variable the model omits appears in no clause and no assumption, so it is free: false is a valid value.
        true = {lit for lit in model or [] if lit > 0}
        return sum(1 << i for i, v in enumerate(var.values()) if v in true)

    def satisfying(self, fixed: Mapping[int, bool], ids: Iterable[str]) -> int | Proof:
        zs = list(self.z.values())
        assume = [self.selector[cid] for cid in ids] + [zs[i] if v else -zs[i] for i, v in fixed.items()]
        if self.states.solve(assumptions=assume):
            return self._assignment(self.states.get_model(), self.z)
        return Unsat(solver=SOLVER, variables=self.states_size[0], clauses=self.states_size[1])

    def pair(self, free: int) -> tuple[int, int] | Proof:
        assume = [self.agree[i] for i in range(self.n) if not free >> i & 1]
        if self.pairs.solve(assumptions=assume):
            model = self.pairs.get_model()
            return (self._assignment(model, self.y), self._assignment(model, self.x))
        return Unsat(solver=SOLVER, variables=self.pairs_size[0], clauses=self.pairs_size[1])


@dataclass(frozen=True)
class _Problem:
    """A trigger and its constraints, with the search path, and the readings of an assignment as states."""

    trigger: Expr
    constraints: Mapping[str, Expr]
    conditions: tuple[str, ...]
    others: tuple[str, ...]

    @staticmethod
    def of(trigger: Expr, constraints: Mapping[str, Expr]) -> _Problem:
        conditions = variables(trigger)
        mentioned = (v for e in constraints.values() for v in variables(e) if v not in conditions)
        return _Problem(trigger, dict(constraints), conditions, tuple(dict.fromkeys(mentioned)))

    def index(self) -> dict[str, int]:
        return {c: i for i, c in enumerate((*self.conditions, *self.others))}

    def engine(self, method: Literal["enumeration", "sat"]) -> _Enumeration | _Sat:
        if method == "enumeration":
            index = self.index()
            compiled = {cid: _compile(e, index) for cid, e in self.constraints.items()}
            return _Enumeration(len(self.conditions), len(self.others), _compile(self.trigger, index), compiled)
        return _Sat(self.conditions, self.others, self.trigger, self.constraints)

    def state(self, s: int) -> State:
        return tuple((c, bool(s >> i & 1)) for i, c in enumerate(self.conditions))

    def completion(self, s: int) -> State:
        n = len(self.conditions)
        return tuple((c, bool(s >> (n + j) & 1)) for j, c in enumerate(self.others))

    def as_pair(self, found: tuple[int, int]) -> Pair:
        return Pair(
            not_fired=self.state(found[0]),
            fired=self.state(found[1]),
            completions=(self.completion(found[0]), self.completion(found[1])),
        )


def _excluded_by(engine: _Enumeration | _Sat, ids: Sequence[str], fixed: Mapping[int, bool]) -> tuple[frozenset[str], ...]:
    """Every minimal set of the constraints that no assignment with the fixed values satisfies."""
    return _minimal_sets(tuple(ids), lambda chosen: not isinstance(engine.satisfying(fixed, chosen), int))


# ---------------------------------------------------------------- the account


def coverage(
    trigger: Expr,
    constraints: Mapping[str, Expr] | None = None,
    points: Sequence[BoundaryPoint] = (),
    *,
    enumerate_up_to: int = ENUMERATE_UP_TO,
) -> Coverage:
    """The feasible MC/DC account of a trigger under its constraints (keyed by constraint id), with boundary points."""
    problem = _Problem.of(trigger, constraints or {})
    conditions, cids = problem.conditions, tuple(problem.constraints)
    index = problem.index()
    _comparisons(points, conditions)
    n = len(conditions)
    method: Literal["enumeration", "sat"] = "enumeration" if len(index) <= enumerate_up_to else "sat"
    with problem.engine(method) as engine:
        finding: Finding | None = None
        accounts: list[ConditionAccount] = []
        groups: list[Group] = []
        some = engine.satisfying({}, cids)
        if not isinstance(some, int):
            finding = Finding(
                "no_feasible_state", proof=some, example=None, completion=(), excluded_by=_excluded_by(engine, cids, {})
            )
        else:
            whole = engine.pair((1 << n) - 1)
            if not isinstance(whole, tuple):
                fires = _compile(trigger, index)(some)
                finding = Finding(
                    "always_fires" if fires else "never_fires",
                    proof=whole,
                    example=problem.state(some),
                    completion=problem.completion(some),
                    excluded_by=(),
                )
        if finding is None:
            for i, c in enumerate(conditions):
                found = engine.pair(1 << i)
                if isinstance(found, tuple):
                    accounts.append(ConditionAccount(c, witness=problem.as_pair(found), redundancy=None))
                else:
                    accounts.append(ConditionAccount(c, witness=None, redundancy=found))
            redundant = tuple(a.condition for a in accounts if a.redundant)

            def bits(group: Iterable[str]) -> int:
                return sum(1 << index[c] for c in group)

            for g in _minimal_sets(redundant, lambda group: isinstance(engine.pair(bits(group)), tuple)):
                found = engine.pair(bits(g))
                assert isinstance(found, tuple)  # the predicate held on g
                groups.append(Group(conditions=g, pair=problem.as_pair(found)))
        point_accounts = []
        for p in points:
            if not p.in_domain:
                point_accounts.append(PointAccount(p, example=None, completion=(), excluded_by=()))
                continue
            fixed = {index[p.condition]: p.holds}
            at = engine.satisfying(fixed, cids)
            if isinstance(at, int):
                point_accounts.append(
                    PointAccount(p, example=problem.state(at), completion=problem.completion(at), excluded_by=())
                )
            else:
                point_accounts.append(
                    PointAccount(p, example=None, completion=(), excluded_by=_excluded_by(engine, cids, fixed))
                )
    return Coverage(
        trigger=trigger,
        constraints=problem.constraints,
        conditions=conditions,
        others=problem.others,
        method=method,
        finding=finding,
        accounts=tuple(accounts),
        groups=tuple(groups),
        points=tuple(point_accounts),
    )


def _comparisons(points: Iterable[BoundaryPoint], conditions: Sequence[str]) -> dict[str, tuple[str, Any]]:
    """Each condition's comparison (op, threshold), as its boundary points state it; one comparison per condition."""
    out: dict[str, tuple[str, Any]] = {}
    for p in points:
        if p.condition not in conditions:
            raise ValueError(f"boundary point for {p.condition!r}, which is not a condition of the trigger")
        cmp = (p.op, _instant(p.threshold))
        if out.setdefault(p.condition, cmp) != cmp:
            raise ValueError(f"condition {p.condition!r} is given two comparisons, {out[p.condition]} and {cmp}")
    return out


# ---------------------------------------------------------------- boundary values


def _points(
    condition: str,
    op: Op,
    threshold: Value,
    step: Callable[[int], Value],
    domain: tuple[Value | None, Value | None],
) -> tuple[BoundaryPoint, ...]:
    return tuple(BoundaryPoint(condition, op, threshold, k, step(k), domain) for k in (-1, 0, 1))


def threshold_points(
    condition: str,
    op: Op,
    threshold: Decimal | int,
    precision: Decimal | int,
    *,
    domain: tuple[Decimal | int | None, Decimal | int | None] = (None, None),
) -> tuple[BoundaryPoint, ...]:
    """A numeric comparison's boundary: the threshold and one declared precision either side (A09 §10.3: monetary
    thresholds at declared precision, never a floating epsilon). Values are exact, so floats are refused."""
    for name, v in (("threshold", threshold), ("precision", precision), *(("domain", d) for d in domain)):
        if v is not None and (isinstance(v, bool) or not isinstance(v, Decimal | int)):
            raise TypeError(f"{name} is {v!r}: a Decimal or int, so that the boundary is exact")
    if precision <= 0:
        raise ValueError(f"precision is {precision!r}: expected a positive step")
    if threshold % precision != 0:
        raise ValueError(f"threshold {threshold} is not stated at its declared precision {precision}")
    return _points(condition, op, threshold, lambda k: threshold + k * precision, domain)


def cutoff_points(
    condition: str,
    op: Op,
    cutoff: datetime | date,
    resolution: timedelta,
    *,
    domain: tuple[datetime | date | None, datetime | date | None] = (None, None),
) -> tuple[BoundaryPoint, ...]:
    """A temporal comparison's boundary: the exact cutoff instant and its neighbours at the comparison's resolution.

    Inclusive and exclusive cutoffs are the comparison's op (``<=`` against ``<``). Neighbours are instants: they are
    taken in UTC, so a cutoff beside a daylight-saving change gets the instants one resolution away, not the wall-clock
    times. A date cutoff takes a whole number of days."""
    if resolution <= timedelta(0):
        raise ValueError(f"resolution is {resolution!r}: expected a positive step")
    timed = isinstance(cutoff, datetime)
    for t in (cutoff, *domain):
        if t is None:
            continue
        if isinstance(t, datetime) != timed:
            raise TypeError(f"{t!r} is not the same kind of time as the cutoff {cutoff!r}")
        if isinstance(t, datetime) and t.utcoffset() is None:
            raise ValueError(f"{t!r} has no time zone; contract times carry explicit zones (spec §14)")
    if isinstance(cutoff, datetime):
        tz = cutoff.tzinfo

        def step(k: int) -> Value:
            return cutoff if k == 0 else (cutoff.astimezone(UTC) + k * resolution).astimezone(tz)

    else:
        if resolution % timedelta(days=1):
            raise ValueError(f"a date cutoff steps by whole days, not {resolution!r}")

        def step(k: int) -> Value:
            return cutoff + k * resolution

    return _points(condition, op, cutoff, step, domain)


def count_points(condition: str, op: Op, k: int, days: int) -> tuple[BoundaryPoint, ...]:
    """A count comparison's boundary: k - 1, k and k + 1, within the counts [0, days] the window allows."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k is {k!r}: a count is a whole number")
    if isinstance(days, bool) or not isinstance(days, int) or days < 0:
        raise ValueError(f"days is {days!r}: expected a non-negative whole number")
    return threshold_points(condition, op, k, 1, domain=(0, days))


# ---------------------------------------------------------------- feasible states on request


class Feasible:
    """Queries of one account's feasible states, by the account's own search path, with its engine held open across
    queries (a context manager). A scenario generator asks it for states to build: a feasible state with some conditions
    fixed, trigger or other (a boundary point needs its condition's value and a class of value in a region), and every
    feasible state of a small trigger (to choose the fewest scenarios meeting every obligation)."""

    def __init__(self, account: Coverage) -> None:
        self.problem = _Problem.of(account.trigger, account.constraints)
        self.index = self.problem.index()
        self.cids = tuple(self.problem.constraints)
        self.method = account.method
        self._engine: _Enumeration | _Sat | None = None

    def __enter__(self) -> Feasible:
        self._engine = self.problem.engine(self.method).__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._engine is not None:
            self._engine.__exit__(*exc)
            self._engine = None

    def complete(self, fixed: Mapping[str, bool]) -> tuple[State, State] | None:
        """A feasible state of the trigger's conditions, and its completion, with the given conditions (trigger or other)
        at the given values; None where the constraints allow none."""
        assert self._engine is not None, "use Feasible as a context manager"
        unknown = sorted(set(fixed) - set(self.index))
        if unknown:
            raise ValueError(f"conditions {unknown} are neither the trigger's nor its constraints'")
        found = self._engine.satisfying({self.index[c]: v for c, v in fixed.items()}, self.cids)
        if not isinstance(found, int):
            return None
        return self.problem.state(found), self.problem.completion(found)

    def states(self) -> tuple[tuple[State, State], ...]:
        """Every feasible state of the trigger's conditions, each with a completion, in the order of their assignments
        read as binary numbers (the first condition the lowest bit)."""
        n = len(self.problem.conditions)
        out = []
        for bits in range(1 << n):
            found = self.complete({c: bool(bits >> i & 1) for i, c in enumerate(self.problem.conditions)})
            if found is not None:
                out.append(found)
        return tuple(out)


# ---------------------------------------------------------------- which obligations scenarios meet


@dataclass(frozen=True)
class Scenario:
    """An abstract scenario: each trigger condition's value (and, where it states them, other conditions' values), and
    the comparison values it uses where it sets one."""

    id: str
    state: Mapping[str, bool]
    values: Mapping[str, Value] = field(default_factory=dict)  # condition -> the value its comparison is evaluated at


@dataclass(frozen=True)
class ScenarioCoverage:
    """Which coverage obligations a set of scenarios meets, which it does not, and which no feasible set can meet."""

    infeasible_scenarios: Mapping[str, tuple[frozenset[str], ...]]  # id -> every minimal set of constraints excluding it
    independence_met: Mapping[str, tuple[str, str]]  # condition -> (not firing, firing) scenarios differing in it alone
    independence_unmet: tuple[str, ...]  # a feasible witness exists; no pair of these scenarios shows it
    independence_infeasible: tuple[str, ...]  # no feasible pair: redundant, or vacuous by a whole-trigger finding
    groups_met: Mapping[frozenset[str], tuple[str, str]]  # group -> (not firing, firing) scenarios agreeing outside it
    groups_unmet: tuple[frozenset[str], ...]
    points_met: Mapping[BoundaryPoint, str]  # a feasible scenario at the point's value
    points_unmet: tuple[BoundaryPoint, ...]
    points_infeasible: tuple[BoundaryPoint, ...]  # outside the domain, or excluded by the constraints


def meets(account: Coverage, scenarios: Sequence[Scenario]) -> ScenarioCoverage:
    """Which of the account's obligations the scenarios meet (A09: uncovered and infeasible obligations are explicit).

    A scenario is feasible when some values of the other conditions it leaves unstated satisfy every constraint; it is
    checked by the same search path the account used."""
    problem = _Problem.of(account.trigger, account.constraints)
    conditions, cids = problem.conditions, tuple(problem.constraints)
    index = problem.index()
    comparisons = _comparisons((p.point for p in account.points), conditions)
    ids = [s.id for s in scenarios]
    if len(set(ids)) != len(ids):
        raise ValueError("scenario ids repeat")
    for s in scenarios:
        if not set(conditions) <= set(s.state) <= set(index) or not all(isinstance(v, bool) for v in s.state.values()):
            raise ValueError(
                f"scenario {s.id!r} does not assign every condition {list(conditions)} true or false, with any others "
                f"from {list(problem.others)}"
            )
        for c, v in s.values.items():
            if c not in comparisons:
                raise ValueError(f"scenario {s.id!r} sets a value for {c!r}, which has no comparison here")
            op, threshold = comparisons[c]
            if _OPS[op](_instant(v), threshold) != s.state[c]:
                raise ValueError(f"scenario {s.id!r} sets {c!r} to {s.state[c]} but its value {v!r} gives the opposite")
    feasible: list[tuple[Scenario, bool]] = []
    infeasible: dict[str, tuple[frozenset[str], ...]] = {}
    fires = _compile(account.trigger, index)
    with problem.engine(account.method) as engine:
        for s in scenarios:
            fixed = {index[c]: v for c, v in s.state.items()}
            if isinstance(engine.satisfying(fixed, cids), int):
                feasible.append((s, fires(sum(1 << i for i, v in fixed.items() if v))))
            else:
                infeasible[s.id] = _excluded_by(engine, cids, fixed)

    def shown(free: frozenset[str]) -> tuple[str, str] | None:
        """Two feasible scenarios, not firing then firing, that agree on every trigger condition outside free."""
        for s, s_fires in feasible:
            for t, t_fires in feasible:
                if not s_fires and t_fires and all(s.state[d] == t.state[d] for d in conditions if d not in free):
                    return (s.id, t.id)
        return None

    met: dict[str, tuple[str, str]] = {}
    unmet: list[str] = []
    no_pair: list[str] = []
    witnessed = {a.condition for a in account.accounts if not a.redundant}
    for c in conditions:
        if c not in witnessed:
            no_pair.append(c)
        elif pair := shown(frozenset({c})):
            met[c] = pair
        else:
            unmet.append(c)
    groups_met: dict[frozenset[str], tuple[str, str]] = {}
    groups_unmet: list[frozenset[str]] = []
    for g in account.groups:
        if pair := shown(g.conditions):
            groups_met[g.conditions] = pair
        else:
            groups_unmet.append(g.conditions)

    points_met: dict[BoundaryPoint, str] = {}
    points_unmet: list[BoundaryPoint] = []
    points_infeasible: list[BoundaryPoint] = []
    for pa in account.points:
        p = pa.point
        if not pa.feasible:
            points_infeasible.append(p)
            continue
        at = next(
            (
                s.id
                for s, _ in feasible
                if p.condition in s.values and _instant(s.values[p.condition]) == _instant(p.value)
            ),
            None,
        )
        if at is None:
            points_unmet.append(p)
        else:
            points_met[p] = at
    return ScenarioCoverage(
        infeasible_scenarios=infeasible,
        independence_met=met,
        independence_unmet=tuple(unmet),
        independence_infeasible=tuple(no_pair),
        groups_met=groups_met,
        groups_unmet=tuple(groups_unmet),
        points_met=points_met,
        points_unmet=tuple(points_unmet),
        points_infeasible=tuple(points_infeasible),
    )
