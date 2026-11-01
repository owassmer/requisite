"""Feasible MC/DC coverage of a trigger under world constraints, and boundary values (spec §8.6 step 1; addendum A09).

The trigger and the world constraints arrive in a small boolean form defined here: ``Var`` for an atomic condition,
``Not``, ``And`` and ``Or``. The contract IR lowers into it, one variable per atomic condition. A state assigns each
atomic condition true or false; it is feasible when every constraint holds in it.

For each atomic condition x_i the account is exact, with no third case (A09 §10.2):

- a feasible unique-cause witness: two feasible states that differ in x_i alone and on which the trigger differs; or
- none exists. By definition that means the trigger, on the feasible states, is a function of the other conditions:
  x_i has no independent effect under the stated constraints, so it is redundant under them. A linked threshold or a
  repeated predicate is one case of this, through the constraints. The proof is kept: the exhaustive search over every
  state, or the solver's UNSAT.

Redundancy is per condition, given the others: two conditions can each be redundant while removing both changes the
trigger (A == B under A AND B). So the account also gives, completely, the minimal groups of redundant conditions whose
joint removal changes the trigger on the feasible states, each with two feasible states that agree on every other
condition and differ in outcome. Any set of redundant conditions containing no group can be removed together; no other
can. The search is exponential only in the number of redundant conditions, and always runs to completion.

Two findings about the whole trigger come first: constraints with no feasible state, and a trigger constant on the
feasible states (it never fires, or always fires). Either makes every per-condition statement vacuous, so neither is
reported as conditions being removable.

Boundary values (spec §8.6 step 1; A09 §10.3) are made for numeric thresholds at their declared precision, temporal
cutoffs at the exact instant and its neighbours at the comparison's resolution, and counts at k - 1, k and k + 1. A
boundary point fixes the value of the atomic condition its comparison defines; where no feasible state has that value
the point is infeasible, and the reason kept is every minimal set of constraints that excludes it.

Given scenarios, ``meets`` says which obligations they meet, which they do not, and which no feasible scenario can meet
(A09: store uncovered and infeasible obligations explicitly). An infeasible scenario meets nothing.

Search is by enumeration of all 2^n states up to ``ENUMERATE_UP_TO`` conditions and by SAT (python-sat) above. The
exhaustive search is its own proof, which a reader can check without trusting a solver, and up to 10 conditions it
costs a few milliseconds; past that it doubles with each condition while SAT stays under a millisecond. Measured on
random triggers over every condition with n/2 implication constraints, median of five, the whole account took 3.5 ms by
enumeration against 0.2 ms by SAT at 10 conditions, 22 ms against 0.3 ms at 12, 380 ms against 0.4 ms at 16. The size
is a parameter, so either path can be forced.
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
    """The expression as a function of a state packed into an int, bit i holding condition i."""
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

State = tuple[tuple[str, bool], ...]  # each condition with its value, in the account's condition order


@dataclass(frozen=True)
class Pair:
    """Two feasible states on which the trigger differs: it does not fire on the first and fires on the second."""

    not_fired: State
    fired: State


@dataclass(frozen=True)
class Exhaustive:
    """Shown by enumeration: every state was evaluated, and none (or no pair) answered the query."""

    states: int  # 2^n, every assignment of the n conditions
    feasible: int  # the states that met the query's requirements: zero when it asked for a state
    pairs: int  # for a pair query, the pairs of distinct feasible states it compared, each with equal outcomes


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
    pair: Pair  # two feasible states that agree on every condition outside the group and differ in outcome


@dataclass(frozen=True)
class Finding:
    """A finding about the whole trigger, which makes every per-condition statement vacuous."""

    kind: Literal["no_feasible_state", "never_fires", "always_fires"]
    proof: Proof  # no feasible state at all, or no two feasible states with different outcomes
    example: State | None  # a constant trigger: a feasible state, on which it takes its one value
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


@dataclass(frozen=True)
class BoundaryPoint:
    """A value at or beside a comparison's boundary, and the truth it gives the condition the comparison defines."""

    condition: str
    op: str
    threshold: Value
    offset: int  # -1, 0 or +1 steps (precision, resolution or one count) from the threshold
    value: Value
    holds: bool  # the condition's value at this point: (value op threshold)
    in_domain: bool  # whether the value lies in the comparison's declared domain


@dataclass(frozen=True)
class PointAccount:
    """Whether a boundary point can occur in a feasible state."""

    point: BoundaryPoint
    example: State | None  # a feasible state with the condition at the point's value; None when infeasible
    excluded_by: tuple[
        frozenset[str], ...
    ]  # infeasible in the domain: every minimal set of constraint ids excluding it

    @property
    def feasible(self) -> bool:
        return self.example is not None


@dataclass(frozen=True)
class Coverage:
    """The feasible coverage account of one trigger under its constraints."""

    trigger: Expr
    constraints: Mapping[str, Expr]
    conditions: tuple[str, ...]  # the trigger's atomic conditions, in order of first appearance
    method: Literal["enumeration", "sat"]
    finding: Finding | None  # a whole-trigger finding; when present, conditions and groups are empty
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
# States are ints, bit i holding condition i. Each path answers two queries:
#   satisfying(fixed, ids): a state with the fixed condition values satisfying the constraints in ids, or the proof
#     that there is none;
#   pair(free): two feasible states that agree on every condition outside the bit set free, the first not firing and
#     the second firing, or the proof that there are none.


class _Enumeration:
    method: Literal["enumeration", "sat"] = "enumeration"

    def __init__(self, n: int, trigger: Callable[[int], bool], constraints: Mapping[str, Callable[[int], bool]]):
        self.n = n
        bit = {cid: 1 << j for j, cid in enumerate(constraints)}
        self.bit = bit
        self.violated: list[int] = []
        self.fires: list[bool] = []
        for s in range(1 << n):
            v = 0
            for cid, holds in constraints.items():
                if not holds(s):
                    v |= bit[cid]
            self.violated.append(v)
            self.fires.append(trigger(s))
        self.feasible = [s for s in range(1 << n) if self.violated[s] == 0]

    def __enter__(self) -> _Enumeration:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def satisfying(self, fixed: Mapping[int, bool], ids: Iterable[str]) -> int | Proof:
        mask = sum(1 << i for i in fixed)
        want = sum(1 << i for i, v in fixed.items() if v)
        used = sum(self.bit[cid] for cid in ids)
        found = next((s for s in range(1 << self.n) if s & mask == want and not self.violated[s] & used), None)
        if found is not None:
            return found
        return Exhaustive(states=1 << self.n, feasible=0, pairs=0)

    def pair(self, free: int) -> tuple[int, int] | Proof:
        keep = ((1 << self.n) - 1) & ~free
        first: dict[tuple[int, bool], int] = {}
        for s in self.feasible:
            key, fires = s & keep, self.fires[s]
            first.setdefault((key, fires), s)
            if (key, not fires) in first:
                return (first[(key, False)], first[(key, True)])
        sizes = Counter(s & keep for s in self.feasible)
        return Exhaustive(
            states=1 << self.n, feasible=len(self.feasible), pairs=sum(c * (c - 1) // 2 for c in sizes.values())
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

    def __init__(self, conditions: Sequence[str], trigger: Expr, constraints: Mapping[str, Expr]):
        n = len(conditions)
        self.n = n
        # Pairs: two copies of the conditions, x (firing) and y (not firing), both feasible, with a selector e_i that,
        # assumed true, makes the copies agree on condition i.
        x = {c: 1 + i for i, c in enumerate(conditions)}
        y = {c: 1 + n + i for i, c in enumerate(conditions)}
        self.agree = [1 + 2 * n + i for i in range(n)]
        enc = _Tseitin(1 + 3 * n)
        for i, c in enumerate(conditions):
            enc.clauses += [[-self.agree[i], -x[c], y[c]], [-self.agree[i], x[c], -y[c]]]
        for e in constraints.values():
            enc.clauses += [[enc.encode(e, x)], [enc.encode(e, y)]]
        enc.clauses += [[enc.encode(trigger, x)], [-enc.encode(trigger, y)]]
        self.x, self.y = x, y
        self.pairs = Solver(name=SOLVER, bootstrap_with=enc.clauses)
        self.pairs_size = (enc.top, len(enc.clauses))
        # Feasibility: one copy z, with a selector per constraint that, assumed true, makes the constraint hold. The
        # trigger is encoded, unasserted, so that every condition is a variable of the formula.
        z = {c: 1 + i for i, c in enumerate(conditions)}
        self.selector = {cid: 1 + n + j for j, cid in enumerate(constraints)}
        enc = _Tseitin(1 + n + len(constraints))
        for cid, e in constraints.items():
            enc.clauses.append([-self.selector[cid], enc.encode(e, z)])
        enc.encode(trigger, z)
        self.z = z
        self.states = Solver(name=SOLVER, bootstrap_with=enc.clauses)
        self.states_size = (enc.top, len(enc.clauses))

    def __enter__(self) -> _Sat:
        return self

    def __exit__(self, *exc: object) -> None:
        self.pairs.delete()
        self.states.delete()

    @staticmethod
    def _state(model: list[int] | None, var: Mapping[str, int]) -> int:
        true = {lit for lit in model or [] if lit > 0}
        return sum(1 << i for i, v in enumerate(var.values()) if v in true)

    def satisfying(self, fixed: Mapping[int, bool], ids: Iterable[str]) -> int | Proof:
        zs = list(self.z.values())
        assume = [self.selector[cid] for cid in ids] + [zs[i] if v else -zs[i] for i, v in fixed.items()]
        if self.states.solve(assumptions=assume):
            return self._state(self.states.get_model(), self.z)
        return Unsat(solver=SOLVER, variables=self.states_size[0], clauses=self.states_size[1])

    def pair(self, free: int) -> tuple[int, int] | Proof:
        assume = [self.agree[i] for i in range(self.n) if not free >> i & 1]
        if self.pairs.solve(assumptions=assume):
            model = self.pairs.get_model()
            return (self._state(model, self.y), self._state(model, self.x))
        return Unsat(solver=SOLVER, variables=self.pairs_size[0], clauses=self.pairs_size[1])


# ---------------------------------------------------------------- the account


def coverage(
    trigger: Expr,
    constraints: Mapping[str, Expr] | None = None,
    points: Sequence[BoundaryPoint] = (),
    *,
    enumerate_up_to: int = ENUMERATE_UP_TO,
) -> Coverage:
    """The feasible MC/DC account of a trigger under its constraints (keyed by constraint id), with boundary points."""
    constraints = dict(constraints or {})
    conditions = variables(trigger)
    index = {c: i for i, c in enumerate(conditions)}
    for cid, e in constraints.items():
        extra = [v for v in variables(e) if v not in index]
        if extra:
            raise ValueError(f"constraint {cid!r} mentions {extra}, which are not conditions of the trigger")
    _comparisons(points, index)
    n = len(conditions)
    fires = _compile(trigger, index)
    engine: _Enumeration | _Sat
    if n <= enumerate_up_to:
        engine = _Enumeration(n, fires, {cid: _compile(e, index) for cid, e in constraints.items()})
    else:
        engine = _Sat(conditions, trigger, constraints)

    def state(s: int) -> State:
        return tuple((c, bool(s >> i & 1)) for i, c in enumerate(conditions))

    def as_pair(found: tuple[int, int]) -> Pair:
        return Pair(not_fired=state(found[0]), fired=state(found[1]))

    def excluded_by(fixed: Mapping[int, bool]) -> tuple[frozenset[str], ...]:
        return _minimal_sets(tuple(constraints), lambda ids: not isinstance(engine.satisfying(fixed, ids), int))

    with engine:
        finding: Finding | None = None
        accounts: list[ConditionAccount] = []
        groups: list[Group] = []
        some = engine.satisfying({}, constraints)
        if not isinstance(some, int):
            finding = Finding("no_feasible_state", proof=some, example=None, excluded_by=excluded_by({}))
        else:
            whole = engine.pair((1 << n) - 1)
            if not isinstance(whole, tuple):
                kind: Literal["never_fires", "always_fires"] = "always_fires" if fires(some) else "never_fires"
                finding = Finding(kind, proof=whole, example=state(some), excluded_by=())
        if finding is None:
            for i, c in enumerate(conditions):
                found = engine.pair(1 << i)
                if isinstance(found, tuple):
                    accounts.append(ConditionAccount(c, witness=as_pair(found), redundancy=None))
                else:
                    accounts.append(ConditionAccount(c, witness=None, redundancy=found))
            redundant = tuple(a.condition for a in accounts if a.redundant)

            def changes(group: frozenset[str]) -> bool:
                return isinstance(engine.pair(sum(1 << index[c] for c in group)), tuple)

            for g in _minimal_sets(redundant, changes):
                found = engine.pair(sum(1 << index[c] for c in g))
                assert isinstance(found, tuple)  # changes(g) held
                groups.append(Group(conditions=g, pair=as_pair(found)))
        point_accounts = []
        for p in points:
            if not p.in_domain:
                point_accounts.append(PointAccount(p, example=None, excluded_by=()))
                continue
            fixed = {index[p.condition]: p.holds}
            found_state = engine.satisfying(fixed, constraints)
            if isinstance(found_state, int):
                point_accounts.append(PointAccount(p, example=state(found_state), excluded_by=()))
            else:
                point_accounts.append(PointAccount(p, example=None, excluded_by=excluded_by(fixed)))
    return Coverage(
        trigger=trigger,
        constraints=constraints,
        conditions=conditions,
        method=engine.method,
        finding=finding,
        accounts=tuple(accounts),
        groups=tuple(groups),
        points=tuple(point_accounts),
    )


def _comparisons(points: Iterable[BoundaryPoint], index: Mapping[str, int]) -> dict[str, tuple[str, Any]]:
    """Each condition's comparison (op, threshold), as its boundary points state it; one comparison per condition."""
    out: dict[str, tuple[str, Any]] = {}
    for p in points:
        if p.condition not in index:
            raise ValueError(f"boundary point for {p.condition!r}, which is not a condition of the trigger")
        cmp = (p.op, _instant(p.threshold))
        if out.setdefault(p.condition, cmp) != cmp:
            raise ValueError(f"condition {p.condition!r} is given two comparisons, {out[p.condition]} and {cmp}")
    return out


# ---------------------------------------------------------------- boundary values


def _instant(v: Value) -> Any:
    """Values compared as what they mean: an aware datetime as its UTC instant, so that wall-clock times in one zone
    across a daylight-saving change compare by instant (Python compares same-zone datetimes by wall clock)."""
    return v.astimezone(UTC) if isinstance(v, datetime) else v


def _points(
    condition: str,
    op: Op,
    threshold: Value,
    step: Callable[[int], Value],
    domain: tuple[Value | None, Value | None],
) -> tuple[BoundaryPoint, ...]:
    if op not in _OPS:
        raise ValueError(f"comparison {op!r} is not one of {sorted(_OPS)}")
    lo, hi = (None if d is None else _instant(d) for d in domain)
    out = []
    for k in (-1, 0, 1):
        v = step(k)
        at = _instant(v)
        inside = (lo is None or at >= lo) and (hi is None or at <= hi)
        holds = _OPS[op](at, _instant(threshold))
        out.append(BoundaryPoint(condition, op, threshold, k, v, holds, inside))
    return tuple(out)


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
    if isinstance(cutoff, datetime):
        if cutoff.utcoffset() is None:
            raise ValueError(f"cutoff {cutoff!r} has no time zone; contract times carry explicit zones (spec §14)")
        tz = cutoff.tzinfo

        def step(k: int) -> Value:
            return cutoff if k == 0 else (cutoff.astimezone(UTC) + k * resolution).astimezone(tz)

    else:
        if resolution % timedelta(days=1):
            raise ValueError(f"a date cutoff steps by whole days, not {resolution!r}")

        def step(k: int) -> Value:
            return cutoff + k * resolution

    for d in domain:
        if d is not None and isinstance(d, datetime) != isinstance(cutoff, datetime):
            raise TypeError(f"domain bound {d!r} is not the same kind of time as the cutoff {cutoff!r}")
    return _points(condition, op, cutoff, step, domain)


def count_points(condition: str, op: Op, k: int, days: int) -> tuple[BoundaryPoint, ...]:
    """A count comparison's boundary: k - 1, k and k + 1, within the counts [0, days] the window allows."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k is {k!r}: a count is a whole number")
    if isinstance(days, bool) or not isinstance(days, int) or days < 0:
        raise ValueError(f"days is {days!r}: expected a non-negative whole number")
    return threshold_points(condition, op, k, 1, domain=(0, days))


# ---------------------------------------------------------------- which obligations scenarios meet


@dataclass(frozen=True)
class Scenario:
    """An abstract scenario: each condition's value, and the comparison values it uses where it sets one."""

    id: str
    state: Mapping[str, bool]
    values: Mapping[str, Value] = field(default_factory=dict)  # condition -> the value its comparison is evaluated at


@dataclass(frozen=True)
class ScenarioCoverage:
    """Which coverage obligations a set of scenarios meets, which it does not, and which no feasible set can meet."""

    infeasible_scenarios: tuple[str, ...]  # violate a constraint: they meet nothing
    independence_met: Mapping[str, tuple[str, str]]  # condition -> (not firing, firing) scenarios differing in it alone
    independence_unmet: tuple[str, ...]  # a feasible witness exists; no pair of these scenarios shows it
    independence_infeasible: tuple[str, ...]  # no feasible pair: redundant, or vacuous by a whole-trigger finding
    points_met: Mapping[BoundaryPoint, str]  # a feasible scenario at the point's value
    points_unmet: tuple[BoundaryPoint, ...]
    points_infeasible: tuple[BoundaryPoint, ...]  # outside the domain, or excluded by the constraints


def meets(account: Coverage, scenarios: Sequence[Scenario]) -> ScenarioCoverage:
    """Which of the account's obligations the scenarios meet (A09: uncovered and infeasible obligations are explicit)."""
    conditions = account.conditions
    index = {c: i for i, c in enumerate(conditions)}
    comparisons = _comparisons((p.point for p in account.points), index)
    ids = [s.id for s in scenarios]
    if len(set(ids)) != len(ids):
        raise ValueError("scenario ids repeat")
    feasible: list[tuple[Scenario, bool]] = []
    infeasible: list[str] = []
    for s in scenarios:
        if set(s.state) != set(conditions) or not all(isinstance(v, bool) for v in s.state.values()):
            raise ValueError(f"scenario {s.id!r} does not assign every condition {list(conditions)} true or false")
        for c, v in s.values.items():
            if c not in comparisons:
                raise ValueError(f"scenario {s.id!r} sets a value for {c!r}, which has no comparison here")
            op, threshold = comparisons[c]
            if _OPS[op](_instant(v), threshold) != s.state[c]:
                raise ValueError(f"scenario {s.id!r} sets {c!r} to {s.state[c]} but its value {v!r} gives the opposite")
        if all(truth(e, s.state) for e in account.constraints.values()):
            feasible.append((s, truth(account.trigger, s.state)))
        else:
            infeasible.append(s.id)

    met: dict[str, tuple[str, str]] = {}
    unmet: list[str] = []
    no_pair: list[str] = []
    witnessed = {a.condition for a in account.accounts if not a.redundant}
    for c in conditions:
        if c not in witnessed:
            no_pair.append(c)
            continue
        shown = None
        for s, fires in feasible:
            for t, other in feasible:
                if (
                    not fires
                    and other
                    and s.state[c] != t.state[c]
                    and all(s.state[d] == t.state[d] for d in conditions if d != c)
                ):
                    shown = (s.id, t.id)
                    break
            if shown:
                break
        if shown:
            met[c] = shown
        else:
            unmet.append(c)

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
        infeasible_scenarios=tuple(infeasible),
        independence_met=met,
        independence_unmet=tuple(unmet),
        independence_infeasible=tuple(no_pair),
        points_met=points_met,
        points_unmet=tuple(points_unmet),
        points_infeasible=tuple(points_infeasible),
    )
