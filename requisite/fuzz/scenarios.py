"""Which scenarios a fuzz run tests, each made concrete (spec §8.6 steps 1-3; addendum A09 §10.2-10.3). Code decides
which scenarios matter; nothing here asks a model anything, and every outcome is the evaluator's (I8).

Coverage. The IR is lowered by the evaluator's own lowering (requisite.contract.evaluate.lower) into the coverage
kernel (requisite.fuzz.coverage), which accounts exactly for every condition's independent effect, every coupled
group's joint effect and every boundary point under the IR's constraints, and proves what no feasible scenario can meet.
The scenarios chosen are the fewest feasible states that meet every feasible independence and group obligation: for a
trigger of up to EXACT_UP_TO conditions, a minimum found exactly, by MaxSAT (RC2) over all its feasible states; above it,
the kernel's own witness pairs with every state no obligation needs removed, which is minimal but not proved minimum.
Each feasible boundary point is then given to a chosen scenario whose state has its condition's value at the point and
which realizes the point's value with every condition kept, or else to a scenario of its own: a boundary value fixes one
value of one comparison, so points of one comparison never share a scenario.

Stress classes (spec §8.6 step 2) are variants of chosen scenarios, one where the trigger fires and one where it does
not, wherever the IR has what each bears on: conflicting sources where an observable it reads has two or more consulted
sources (a lower-ranked source, or one that must agree, publishes another value for the same instant); a revised value
wherever it reads an observable (another vintage of a deciding value, placed where the revision policy does not select
it); a renamed or recodified instrument, partial scope, and a stay or injunction wherever it has interpretive conditions,
whose truths code cannot see into (they are text), so the variant keeps its base's truths and the renderer is told to
present them under that stress, and the rendering check then holds the packet to them; and evidence missing at expiry
always (the evidence of a condition the outcome turns on is withdrawn, and the procedure's unknown-at-expiry rule
applies, as the evaluator says).

Making a scenario concrete (step 3). A state is a truth for each atomic condition; its observations and interpretations
are built kind by kind: an interpretive condition's truth is its interpretation; conditions comparing one observable's
values at instants are realized from the kernel's completion over the lowering's layout (one observation per class
present in each region: exact, as the lowering proves), at instants no other condition reads that observable for; a
reading takes a band whose comparisons have the state's truths; a count takes as many days as its comparisons allow
around the days other conditions' values already decide, one published instant on each other day; a duration a stretch
cut into the segments other conditions' values leave; a computed number's comparisons are met by solving a published
input exactly (an equality), by the readings' corner the node's declared monotonicity favours walked toward the
threshold while every comparison keeps its truth, or by a search one input at a time. Every value a count, a duration, a
compound comparison or a computation adds keeps the conditions comparing that observable at instants as the state has
them. Observations are published by the procedure's first consulted source (each of the sources that must agree, under
require_agreement), by the observation cutoff and as the revision policy requires. A value the IR reads only through a
computation is taken from the reference observations a run is given, or else is a hundred steps of its published
precision, where either keeps the state's other conditions.

Then every atomic condition is evaluated alone by the evaluator on what was built, and the scenario's intended
assignment is that: for a coverage or boundary scenario it must equal the chosen state. A chosen state whose inputs
cannot be built to mean it is left out and the fewest states are chosen again without it; an obligation or a boundary
point no buildable scenario meets stays unmet, recorded in the plan's notes with why. (The lowering's constraints are
exact among conditions reading one observable at instants, and among comparisons of one number; they do not reach
through a computation, nor between a count or duration and other conditions on its observable, so the kernel can call
such a state feasible where no history has it, and then no build means it.) The scenario's outcome and dated transfers
are the evaluator's on the whole contract, at its determination instant (its expiration, once its observation cutoff
has passed), the instant at which the resolvers are asked for the evidence available at expiration.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

from requisite import core
from requisite.compute import registry
from requisite.contract import evaluate
from requisite.contract.evaluate import Interpretation, Layout
from requisite.core.kleene import Kleene
from requisite.fuzz import coverage as cov

EXACT_UP_TO = 12

StressClass = Literal[
    "conflicting_sources",
    "revised_value",
    "renamed_or_recodified_instrument",
    "partial_scope",
    "stay_or_injunction",
    "evidence_missing_at_expiry",
]
STRESS: tuple[StressClass, ...] = (
    "conflicting_sources", "revised_value", "renamed_or_recodified_instrument", "partial_scope", "stay_or_injunction",
    "evidence_missing_at_expiry",
)  # fmt: skip  (spec §8.6 step 2, in its order)
PRESENTED: tuple[StressClass, ...] = ("renamed_or_recodified_instrument", "partial_scope", "stay_or_injunction")
GENERATOR = core.Ref(id="requisite.fuzz.scenarios", version=1)
FORCED = "requisite.fuzz.outcome"  # the node id that stands for a forced outcome (`forced`)


class FuzzError(RuntimeError):
    """A fuzz run cannot go on: a scenario's inputs do not mean its state, or the inputs it was given are not usable."""


# ---------------------------------------------------------------- records of a plan


@dataclass(frozen=True)
class Fact:
    """One published value a scenario fixes (a core.SyntheticObservation once its scenario's artifact is named)."""

    observable: str
    source: str
    at: datetime
    value: core.CmpValue
    published_at: datetime
    final: bool

    def observation(self, artifact: str) -> core.SyntheticObservation:
        return core.SyntheticObservation(observable=self.observable, source=self.source, at=self.at, value=self.value,
                                         published_at=self.published_at, final=self.final, artifact=artifact)  # fmt: skip

    def shown(self) -> dict[str, Any]:
        return {"observable": self.observable, "source": self.source, "for": self.at.isoformat(),
                "value": str(self.value.value), "published_at": self.published_at.isoformat(), "final": self.final}  # fmt: skip


@dataclass(frozen=True)
class Pin:
    """A boundary point a scenario realizes: its comparison and the value it gives the condition."""

    comparison: evaluate.Comparison
    point: cov.BoundaryPoint
    solution: tuple[tuple[str, datetime, core.CmpValue], ...] = ()  # published values a computed boundary needs

    @property
    def condition(self) -> str:
        return self.point.condition

    def shown(self) -> str:
        v = self.point.value
        return f"{self.comparison.detail}: {v.isoformat() if isinstance(v, datetime | date) else v}"


@dataclass
class _Abstract:
    id: str
    role: Literal["coverage", "boundary", "stress"]
    state: dict[str, bool]
    completion: dict[str, bool]
    pins: list[Pin] = field(default_factory=list)
    stress: StressClass | None = None
    base: str | None = None


@dataclass(frozen=True)
class Scenario:
    """A scenario made concrete. truths: each atomic condition's truth on its inputs, by the evaluator; for a coverage
    or boundary scenario, its chosen state. evaluation: the evaluator's on the whole contract at the plan's instant.
    exercised: the IR nodes whose truth the outcome turns on (the decisive atomic conditions, the nodes above them and
    the nodes inside them), and for a stress variant the condition it withdrew. artifact: the scenario's own record (its
    facts and interpretations), the synthetic artifact its observations are read from."""

    id: str
    role: Literal["coverage", "boundary", "stress"]
    stress: StressClass | None
    base: str | None
    pins: tuple[Pin, ...]
    facts: tuple[Fact, ...]
    interpretations: Mapping[str, Interpretation]
    truths: Mapping[str, Kleene]
    state_hash: str
    artifact: str
    content: bytes
    observations: tuple[core.SyntheticObservation, ...]
    evaluation: core.ResolutionEvaluation
    exercised: frozenset[str]
    withdrawn: str | None = None


@dataclass(frozen=True)
class Plan:
    """What a fuzz run tests, before any model is asked."""

    ir: core.ContractIR
    lowered: evaluate.Lowered
    account: cov.Coverage
    at: datetime
    scenarios: tuple[Scenario, ...]
    obligations: tuple[core.CoverageObligation, ...]
    boundaries: tuple[core.BoundaryPoint, ...]
    whole_trigger: Literal["ok", "no_feasible_state", "always_fires", "never_fires"]
    mandatory: frozenset[str]
    notes: tuple[str, ...] = ()

    @property
    def sources(self) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
        return sources(self.ir)


# ---------------------------------------------------------------- values compared


def _key(v: core.CmpValue) -> Any:
    if isinstance(v, core.Number):
        return Fraction(v.value)
    if isinstance(v, core.OnDay):
        return Fraction(v.value.toordinal())
    return v.value


def _holds(x: Any, op: str, t: Any) -> bool:
    if op in ("==", "!="):
        return (x == t) == (op == "==")
    return {"<": x < t, "<=": x <= t, ">": x > t, ">=": x >= t}[op]


def _truth(e: Any, values: Mapping[str, core.CmpValue]) -> bool:
    """A boolean expression over values at one instant, for choosing values only (the evaluator says what they mean)."""
    if isinstance(e, core.Cmp):
        return _holds(_key(values[e.observable]), e.op, _key(e.value))
    if isinstance(e, core.And):
        return all(_truth(a, values) for a in e.args)
    if isinstance(e, core.Or):
        return any(_truth(a, values) for a in e.args)
    if isinstance(e, core.Not):
        return not _truth(e.arg, values)
    raise TypeError(f"{getattr(e, 'id', e)!r} is not compared at an instant")


def _utc(t: datetime) -> datetime:
    return t.astimezone(UTC)


def _kleene(b: bool) -> Kleene:
    return Kleene.TRUE if b else Kleene.FALSE


# ---------------------------------------------------------------- the plan


def sources(ir: core.ContractIR) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Each observable's sources the procedure consults, in order, and those that must agree (the evaluator's)."""
    return evaluate.consulted(ir)


def plan(ir: core.ContractIR, *, reference: Sequence[core.Observation] = (), stress: bool = True) -> Plan:
    """Every scenario of a fuzz run of `ir`, made concrete, with the coverage account its scenarios meet."""
    lowered = evaluate.lower(ir)
    rounds = lowered.rounds()
    account = cov.coverage(lowered.trigger, lowered.constraints, rounds[0] if rounds else ())
    point_accounts = list(account.points) + [p for r in rounds[1:]
                                             for p in cov.coverage(lowered.trigger, lowered.constraints, r).points]  # fmt: skip
    comparison_of = {p: c for c in lowered.comparisons for p in c.points}
    b = _Builder(ir, lowered, account, reference)
    whole: Literal["ok", "no_feasible_state", "always_fires", "never_fires"] = (
        "ok" if account.finding is None else account.finding.kind
    )
    abstracts: list[_Abstract] = []
    pairs: dict[str, tuple[int, int]] = {}
    if account.finding is None:
        abstracts, pairs = b.fewest()
    elif (example := account.finding.example) is not None:
        only = _Abstract("c01", "coverage", dict(example), dict(account.finding.completion))
        why = b._builds(only)
        if why is None:
            abstracts = [only]
        else:
            b.notes.append(f"the trigger's one feasible outcome could not be built: {why}")
    unrealized: dict[cov.BoundaryPoint, Unrealized] = {}
    if whole != "no_feasible_state":
        for pa in point_accounts:
            if pa.feasible:
                reason = b.host(abstracts, Pin(comparison_of[pa.point], pa.point))
                if reason is not None:
                    unrealized[pa.point] = reason
    concrete = [b.concrete(a) for a in abstracts]
    if stress and concrete:
        concrete += b.stress(concrete)
    scenarios = tuple(concrete)
    obligations, witnesses = b.obligations(abstracts, pairs)
    pinned = {p.point: s.id for s in scenarios for p in s.pins}
    boundaries = tuple(_boundary(pa, comparison_of[pa.point], pinned, unrealized) for pa in point_accounts)
    mandatory = frozenset(witnesses) | frozenset(pinned.values())
    return Plan(ir, lowered, account, b.at, scenarios, obligations, boundaries, whole, mandatory, tuple(b.notes))


JOINT_STEPS = 256  # grid steps a second input moves either side of its reference in the joint search (ours)


@dataclass(frozen=True)
class Unrealized:
    """Why no scenario realizes a boundary point: "infeasible" where that is proved from the IR (its bands, its other
    bound, its constraints over every feasible state), "unconstructed" where the construction tried failed without a
    proof (core.BoundaryPoint)."""

    status: Literal["infeasible", "unconstructed"]
    why: str


def _boundary(pa: cov.PointAccount, comparison: evaluate.Comparison, pinned: Mapping[cov.BoundaryPoint, str],
              unrealized: Mapping[cov.BoundaryPoint, Unrealized]) -> core.BoundaryPoint:  # fmt: skip
    pin = Pin(comparison, pa.point)
    if not pa.point.in_domain:
        return core.BoundaryPoint(condition=pa.point.condition, point=pin.shown(), status="infeasible",
                                  excluded_by=("outside the values the quantity can take",))  # fmt: skip
    if not pa.feasible:
        sets = tuple(" & ".join(sorted(s)) for s in pa.excluded_by)
        return core.BoundaryPoint(condition=pa.point.condition, point=pin.shown(), status="infeasible",
                                  excluded_by=sets)  # fmt: skip
    if pa.point in unrealized:
        u = unrealized[pa.point]
        return core.BoundaryPoint(condition=pa.point.condition, point=pin.shown(), status=u.status, excluded_by=(u.why,))
    return core.BoundaryPoint(condition=pa.point.condition, point=pin.shown(), status="feasible",
                              scenario=pinned.get(pa.point))  # fmt: skip


def proof_text(p: cov.Proof) -> str:
    if isinstance(p, cov.Exhaustive):
        return (f"exhaustive: all {p.assignments} assignments evaluated; of {p.feasible} feasible states, the {p.pairs} "
                "pairs that agree on every other condition have equal outcomes")  # fmt: skip
    return f"{p.solver} proved the witness query unsatisfiable ({p.variables} variables, {p.clauses} clauses)"


class _Builder:
    def __init__(self, ir: core.ContractIR, lowered: evaluate.Lowered, account: cov.Coverage,
                 reference: Sequence[core.Observation]) -> None:  # fmt: skip
        self.ir, self.lowered, self.account = ir, lowered, account
        roots = [ir.trigger] + ([ir.payout.quantity] if ir.payout.quantity is not None else [])
        self.nodes: dict[str, Any] = {n.id: n for r in roots for n in core.walk(r)}
        self.atoms = {cid: self.nodes[cid] for cid in lowered.conditions}
        self.observables = {o.id: o for o in ir.observables}
        self.consulted, self.agree = evaluate.consulted(ir)
        self.at = evaluate.determination(ir)
        self.layout_of: dict[str, Layout] = {c: lay for lay in lowered.layouts.values() for c in lay.conditions}
        self.reference = {(r.observable, _utc(r.at)): r.value for r in reference}
        self.readings = [n for n in self.nodes.values() if isinstance(n, core.ReadValue)]
        self.parents: dict[str, str] = {}
        for r in roots:
            for n in core.walk(r):
                for child in _children(n):
                    self.parents.setdefault(child.id, n.id)
        self.count = 0
        self.notes: list[str] = []
        # Instants other conditions read an observable's value for: a layout never places a value of its own on one.
        self.reserved: dict[str, set[datetime]] = {}
        for r in roots:
            for n in core.walk(r):
                if isinstance(n, core.ValueAt):
                    self.reserved.setdefault(n.observable, set()).add(_utc(n.at))
                elif isinstance(n, core.At):
                    for o in _reads(n):
                        self.reserved.setdefault(o, set()).add(_utc(n.at))

    # ---- the fewest states meeting every obligation

    def _fires(self, state: Mapping[str, bool]) -> bool:
        return cov.truth(self.lowered.trigger, state)

    def fewest(self) -> tuple[list[_Abstract], dict[str, tuple[int, int]]]:
        """The chosen coverage states, and for each obligation (a condition, or a group's key) the pair meeting it."""
        acc = self.account
        conds = acc.conditions
        witnessed = [a.condition for a in acc.accounts if not a.redundant]
        groups = [frozenset(g.conditions) for g in acc.groups]
        if len(conds) <= EXACT_UP_TO:
            with cov.Feasible(acc) as feas:
                states = [(dict(s), dict(c)) for s, c in feas.states()]
        else:
            seen: dict[tuple[bool, ...], tuple[dict[str, bool], dict[str, bool]]] = {}
            for pair in [a.witness for a in acc.accounts if a.witness is not None] + [g.pair for g in acc.groups]:
                for st, comp in ((pair.not_fired, pair.completions[0]), (pair.fired, pair.completions[1])):
                    seen.setdefault(tuple(v for _, v in st), (dict(st), dict(comp)))
            states = list(seen.values())
        fires = [self._fires(s) for s, _ in states]
        index = {tuple(s[c] for c in conds): k for k, (s, _) in enumerate(states)}
        every: dict[str, list[tuple[int, int]]] = {}
        for c in witnessed:
            pairs = []
            for k, (s, _) in enumerate(states):
                if not fires[k]:
                    t = index.get(tuple(not s[d] if d == c else s[d] for d in conds))
                    if t is not None and fires[t]:
                        pairs.append((k, t))
            every[c] = pairs
        for g in groups:
            outside = [d for d in conds if d not in g]
            by: dict[tuple[bool, ...], list[int]] = {}
            for k, (s, _) in enumerate(states):
                by.setdefault(tuple(s[d] for d in outside), []).append(k)
            every[_group_key(g, conds)] = [(k, t) for ks in by.values() for k in ks for t in ks
                                           if not fires[k] and fires[t]]  # fmt: skip
        # The fewest buildable states: a chosen state whose inputs cannot be built is left out and the choice made again
        # without it; an obligation no buildable pair meets stays uncovered, with the reason recorded.
        unbuilt: dict[int, str] = {}
        while True:
            live = {o: [p for p in ps if p[0] not in unbuilt and p[1] not in unbuilt] for o, ps in every.items()}
            for o, ps in live.items():
                if not ps and every[o]:
                    self.notes.append(f"obligation {o}: no pair of its feasible states could be built (" + "; ".join(
                        sorted({unbuilt[k] for p in every[o] for k in p if k in unbuilt})) + ")")  # fmt: skip
            live = {o: ps for o, ps in live.items() if ps}
            if len(conds) <= EXACT_UP_TO:
                chosen = self._minimum(len(states), live, excluded=set(unbuilt))
            else:
                chosen = self._minimal(live)
            failed = {k: why for k in sorted(chosen) if (why := self._builds(_Abstract("?", "coverage", *states[k])))}
            if not failed:
                break
            unbuilt.update(failed)
        self.notes[:] = list(dict.fromkeys(self.notes))
        met = {o: next(p for p in ps if p[0] in chosen and p[1] in chosen) for o, ps in live.items()}
        order = sorted(chosen)
        ids = {k: f"c{i:02d}" for i, k in enumerate(order, 1)}
        out = [_Abstract(ids[k], "coverage", states[k][0], states[k][1]) for k in order]
        self.count = len(out)
        return out, {o: (order.index(p[0]), order.index(p[1])) for o, p in met.items()}

    def _builds(self, a: _Abstract) -> str | None:
        """Why a scenario's inputs cannot be built to mean its state, or None where they can."""
        try:
            self.concrete(a)
        except FuzzError as e:
            return f"{sorted(k for k, v in a.state.items() if v)} true: {e}"
        return None

    @staticmethod
    def _minimum(n: int, obligations: Mapping[str, list[tuple[int, int]]], excluded: set[int]) -> set[int]:
        """The fewest states including a pair for every obligation: MaxSAT, one soft clause per state left out."""
        if not obligations:
            first = next((k for k in range(n) if k not in excluded), None)
            return set() if first is None else {first}
        wcnf = WCNF()
        for k in excluded:
            wcnf.append([-(k + 1)])
        top = n
        for pairs in obligations.values():
            lits = []
            for k, t in pairs:
                top += 1
                wcnf.append([-top, k + 1])
                wcnf.append([-top, t + 1])
                lits.append(top)
            wcnf.append(lits)
        for k in range(n):
            wcnf.append([-(k + 1)], weight=1)
        with RC2(wcnf) as rc2:
            model = rc2.compute()
        if model is None:
            raise FuzzError("no set of feasible states meets every obligation, which the kernel's witnesses contradict")
        return {lit - 1 for lit in model if 0 < lit <= n}

    @staticmethod
    def _minimal(obligations: Mapping[str, list[tuple[int, int]]]) -> set[int]:
        chosen = {k for pairs in obligations.values() for k in pairs[0]}
        for k in sorted(chosen, reverse=True):
            rest = chosen - {k}
            if all(any(a in rest and b in rest for a, b in pairs) for pairs in obligations.values()):
                chosen = rest
        return chosen

    def obligations(self, abstracts: Sequence[_Abstract], pairs: Mapping[str, tuple[int, int]]
                    ) -> tuple[tuple[core.CoverageObligation, ...], set[str]]:  # fmt: skip
        """Each obligation's status (A09 §10.2), checked by the kernel's `meets` against the chosen scenarios."""
        acc = self.account
        if acc.finding is not None:
            return tuple(core.CoverageObligation(conditions=[c], status="vacuous", witnesses=(), proof=None)
                         for c in acc.conditions), set()  # fmt: skip
        coverage_scenarios = [a for a in abstracts if a.role == "coverage"]
        met = cov.meets(acc, [cov.Scenario(a.id, {**a.state, **a.completion}) for a in coverage_scenarios])
        out: list[core.CoverageObligation] = []
        witnesses: set[str] = set()
        for a in acc.accounts:
            c = a.condition
            if a.redundant:
                assert a.redundancy is not None
                coupled = sorted({d for g in acc.groups if c in g.conditions for d in g.conditions} - {c},
                                 key=acc.conditions.index)  # fmt: skip
                out.append(core.CoverageObligation(conditions=[c], status="redundant", witnesses=(),
                                                   proof=proof_text(a.redundancy), coupled_group=coupled))  # fmt: skip
            elif c in met.independence_met:
                w = met.independence_met[c]
                witnesses |= set(w)
                out.append(core.CoverageObligation(conditions=[c], status="met", witnesses=list(w), proof=None))
            else:
                out.append(core.CoverageObligation(conditions=[c], status="uncovered", witnesses=(), proof=None))
        for g in acc.groups:
            conds = sorted(g.conditions, key=acc.conditions.index)
            if g.conditions in met.groups_met:
                w = met.groups_met[g.conditions]
                witnesses |= set(w)
                out.append(core.CoverageObligation(conditions=conds, status="met", witnesses=list(w), proof=None))
            else:
                out.append(core.CoverageObligation(conditions=conds, status="uncovered", witnesses=(), proof=None))
        return tuple(out), witnesses

    # ---- boundary points

    def host(self, abstracts: list[_Abstract], pin: Pin) -> Unrealized | None:
        """Give the point to a chosen scenario that realizes it with every condition kept, or else to a new one.
        Returns why no scenario realizes the point (proved infeasible, or unconstructed with what was tried), or None
        (a feasible point no scenario this run builds realizes is recorded in the notes and stays unmet)."""
        p = pin.point
        solved = self._solve(pin)
        if isinstance(solved, Unrealized):
            return solved
        pin = solved
        why: list[str] = []
        with cov.Feasible(self.account) as feas:
            for a in abstracts:
                if a.state.get(p.condition) != p.holds or any(not self._joint(q, pin) for q in a.pins):
                    continue
                for fixed in self._pin_fixed(pin, a.state):
                    earlier = [self._pin_fixed(q, a.state)[0] for q in a.pins]
                    found = feas.complete({**a.state, **fixed, **{k: v for f in earlier for k, v in f.items()}})
                    if found is None:
                        continue
                    trial = replace(a, completion=dict(found[1]), pins=[*a.pins, pin])
                    failed = self._builds(trial)
                    if failed is None:
                        a.completion, a.pins = trial.completion, trial.pins
                        return None
                    why.append(failed)
            for fixed in self._pin_states(pin):
                found = feas.complete(fixed)
                if found is None:
                    continue
                trial = _Abstract(f"b{self.count + 1:02d}", "boundary", dict(found[0]), dict(found[1]), [pin])
                failed = self._builds(trial)
                if failed is None:
                    self.count += 1
                    abstracts.append(trial)
                    return None
                why.append(failed)
        if why:
            self.notes.append(f"boundary point {pin.shown()} of {p.condition}: no scenario could be built for it ("
                              + "; ".join(dict.fromkeys(why)) + ")")  # fmt: skip
            return None
        return Unrealized("infeasible", "no feasible state of the trigger's conditions realizes this value (every state "
                                        "this value fixes was completed against the IR's constraints)")  # fmt: skip

    def _same(self, c: str) -> list[str]:
        """The comparisons of the number condition c compares (c included), where c compares a number."""
        node = self.atoms[c]
        if not isinstance(node, core.NumCmp) or isinstance(node.expr, core.ValueAt):
            return [c]
        key = _content(node.expr)
        return [d for d, n in self.atoms.items() if isinstance(n, core.NumCmp) and _content(n.expr) == key]

    def _joint(self, q: Pin, pin: Pin) -> bool:
        """Whether two pins can sit in one scenario: never two values of one comparison, nor two values of one number
        (a reading takes one band; a count one number; a computed comparison one set of inputs)."""
        if q.comparison == pin.comparison:
            return False
        return q.condition in self.layout_of or q.condition not in self._same(pin.condition)

    def _truths_at(self, pin: Pin) -> dict[str, bool] | None:
        """The truths a pin's value gives the comparisons of its number; None where it leaves one undecided."""
        c, v = pin.condition, pin.point.value
        node = self.atoms[c]
        if not isinstance(node, core.NumCmp) or isinstance(node.expr, core.ValueAt | core.Computed):
            return {c: pin.point.holds}
        out = {}
        for d in self._same(c):
            other = self.atoms[d]
            if isinstance(node.expr, core.ReadValue):
                band = self._band_of(node.expr, Fraction(cast(Decimal, v)))
                if band is None:
                    return None
                t = _interval(node.expr, band).compare(other.op, Fraction(other.value))
                if t is Kleene.UNKNOWN:
                    return None
                out[d] = t is Kleene.TRUE
            else:
                out[d] = _holds(Fraction(cast(Decimal, v)), other.op, Fraction(other.value))
        return out

    def _pin_fixed(self, pin: Pin, state: Mapping[str, bool]) -> list[dict[str, bool]]:
        """What a pin fixes in a scenario of this state beside its condition's value, in order of preference: a class
        of value in a region of a layout; nothing, for another kind; no option where its number's other comparisons
        would take other truths."""
        c = pin.condition
        lay = self.layout_of.get(c)
        if lay is not None:
            if len(lay.conditions) == 1:
                return [{}]
            places = [{lay.variable(i, k): True} for i, k in self._pin_places(lay, pin) if i in lay.used]
            return places or [{}]
        truths = self._truths_at(pin)
        if truths is None or any(state.get(d) != t for d, t in truths.items()):
            return []
        return [{}]

    def _pin_states(self, pin: Pin) -> list[dict[str, bool]]:
        """What a new scenario for a pin fixes, in order of preference."""
        c, holds = pin.condition, pin.point.holds
        lay = self.layout_of.get(c)
        if lay is not None and len(lay.conditions) > 1:
            places = [(i, k) for i, k in self._pin_places(lay, pin) if i in lay.used]
            return [{c: holds, lay.variable(i, k): True} for i, k in places] or [{c: holds}]
        truths = self._truths_at(pin)
        return [truths] if truths is not None else []

    def _solve(self, pin: Pin) -> Pin | Unrealized:
        """The pin, with what realizing it needs solved (the published values a computed number's exact value needs),
        or why no scenario realizes it: proved from the IR, or a construction that failed."""
        node = self.atoms[pin.condition]
        lay = self.layout_of.get(pin.condition)
        if lay is not None and isinstance(pin.point.value, datetime):
            # The value at this instant holds (By) or breaks (Throughout) the comparison; it decides the condition as the
            # point says only where the condition's other bound puts the instant on the same side.
            inside = lay.within(lay.region_of(pin.point.value), pin.condition)
            gives = inside if pin.condition not in lay.every else not inside
            if gives != pin.point.holds:
                return Unrealized("infeasible", f"at {pin.point.value.isoformat()} the condition's other bound puts the "
                                  f"instant {'inside' if inside else 'outside'} its bounds, so a value there gives it "
                                  f"{gives}, not {pin.point.holds}")  # fmt: skip
            return pin
        if not isinstance(node, core.NumCmp):
            return pin
        e = node.expr
        if isinstance(e, core.ReadValue):
            if self._truths_at(pin) is None:
                band = self._band_of(e, Fraction(cast(Decimal, pin.point.value)))
                where = "no band" if band is None else f"the band {e.bands[band]}"
                return Unrealized("infeasible", f"{e.id} is read as a band, and {where} holding {pin.point.value} "
                                  f"{e.unit} decides '{node.op} {node.value}'")  # fmt: skip
            return pin
        if not isinstance(e, core.Computed):
            return pin
        banded = [r.id for r in core.walk(e) if isinstance(r, core.ReadValue) and r.bands[-1][0] != r.bands[-1][1]]
        if banded:
            return Unrealized("infeasible", f"{e.id} is computed from readings answered as bands ({', '.join(banded)}), "
                              "so its value is an interval, never one value")  # fmt: skip
        other = [n for n, sub in e.inputs.items() if not isinstance(sub, core.ReadValue | core.ValueAt)]
        if other:
            return Unrealized("unconstructed", f"{e.id}'s inputs {other} are neither readings nor published values; "
                              "this run does not solve for their exact value")  # fmt: skip
        return self._solve_computed(pin, e)

    def _solve_computed(self, pin: Pin, e: core.Computed) -> Pin | Unrealized:
        """The pin with the published values that give a computed number exactly its value (`_solve_value`). A search
        that finds none proves nothing about the node, so the point stands unconstructed."""
        solved = self._solve_value(e, Fraction(cast(Decimal, pin.point.value)), ())
        if isinstance(solved, str):
            return Unrealized("unconstructed", solved)
        return replace(pin, solution=solved)

    def _pin_places(self, lay: Layout, pin: Pin) -> list[tuple[int, int]]:
        """The (region, class) a pin's value occupies: a threshold's value in a region within the condition's bounds;
        a cutoff's instant, with a value that holds (By) or breaks (Throughout) the condition's comparison."""
        c, p = pin.condition, pin.point
        if isinstance(p.value, datetime):  # an instant against a bound (a date observable's cutoff is a value)
            i = lay.region_of(p.value)
            want = c not in lay.every
            return [(i, k) for k in self._preferred(lay, [k for k in range(len(lay.keys)) if lay.satisfies(c, k) == want])]
        value = p.value
        key = Fraction(value.toordinal()) if isinstance(value, date) and not isinstance(value, datetime) else Fraction(cast(Decimal, value))
        ks = [k for k, x in enumerate(lay.keys) if x == key]
        regions = [i for i in range(len(lay.regions)) if lay.within(i, c)]
        point = self.atoms[c].kind in ("at", "num_cmp")
        regions.sort(key=lambda i: (lay.regions[i][0] == lay.regions[i][1]) != point)
        return [(i, k) for i in regions for k in ks]

    # ---- one scenario's inputs

    def published(self, o: str, t: datetime) -> datetime:
        """When a value for instant t is published: at t, or by the latest instant the contract still considers."""
        obs, timing = self.observables[o], self.ir.timing
        latest = [_utc(timing.observation_cutoff), _utc(self.at)]
        if obs.revision_policy == "as_of_expiry" or obs.final_fallback == "as_of_expiry":
            latest.append(_utc(timing.expiration))
        if obs.revision_policy == "final" and obs.final_deadline is not None:
            latest.append(_utc(obs.final_deadline))
        return min([_utc(t), *latest])

    def facts(self, o: str, t: datetime, v: core.CmpValue) -> list[Fact]:
        sources = self.agree.get(o) or self.consulted[o][:1]
        if not sources:
            raise FuzzError(f"the procedure consults none of {o}'s sources")
        when = self.published(o, t)
        return [Fact(o, s, _utc(t), v, when, True) for s in sources]

    def _preferred(self, lay: Layout, ks: Iterable[int]) -> list[int]:
        """Classes in order of preference: a value that is not negative first (an observable states no lower bound,
        and most do not go below zero), then the order of the values."""
        return sorted(ks, key=lambda k: (isinstance(lay.keys[k], Fraction) and lay.keys[k] < 0, k))

    def _canonical(self, lay: Layout, k: int) -> int:
        """The preferred class every comparison of the layout treats as it treats class k."""
        vector = tuple(lay.satisfies(c, k) for c in lay.conditions)
        same = [m for m in range(len(lay.keys)) if tuple(lay.satisfies(c, m) for c in lay.conditions) == vector]
        return self._preferred(lay, same)[0]

    def realize(self, a: _Abstract) -> tuple[list[Fact], dict[str, Interpretation]]:
        state = a.state
        facts: list[Fact] = []
        interps: dict[str, Interpretation] = {}
        pins = {p.condition: p for p in a.pins}
        for cid, node in self.atoms.items():
            if isinstance(node, core.Interpretive):
                interps[cid] = _kleene(state[cid])
        for lay in self.lowered.layouts.values():
            facts += self._layout_facts(lay, a)
        done: set[str] = set()
        for cid, node in self.atoms.items():
            if not isinstance(node, core.NumCmp) or isinstance(node.expr, core.ValueAt) or cid in done:
                continue
            key = _content(node.expr)
            group = [d for d, n in self.atoms.items() if isinstance(n, core.NumCmp) and _content(n.expr) == key]
            done |= set(group)
            pin = next((pins[d] for d in group if d in pins), None)
            e = node.expr
            if isinstance(e, core.ReadValue):
                band = self._choose_band(e, group, state, pin)
                for r in self.readings:
                    if _content(r) == _content(e):
                        interps[r.id] = (e.bands[band][0], e.bands[band][1])
            elif isinstance(e, core.CountTrue):
                facts += self._count(e, group, state, pin, facts)
            elif isinstance(e, core.DurationIn):
                facts += self._duration(e, group, state, pin, facts)
            elif isinstance(e, core.Computed):
                got_facts, got = self._computed(group, state, pin, facts, interps)
                facts += got_facts
                interps.update(got)
        for cid, node in self.atoms.items():
            if isinstance(node, core.At | core.By | core.Throughout) and cid not in self.layout_of:
                facts += self._compound(node, state[cid], state, facts)
        if self.ir.payout.quantity is not None:
            facts += self._default_number(self.ir.payout.quantity, facts, state)
        for r in self.readings:
            interps.setdefault(r.id, (r.bands[0][0], r.bands[0][1]))
        return _dedupe(facts), interps

    def _layout_facts(self, lay: Layout, a: _Abstract) -> list[Fact]:
        o = lay.observable
        pinned = [p for p in a.pins if p.condition in lay.conditions]
        present: dict[tuple[int, int], list[datetime]] = {}
        fixed_pins: set[tuple[int, int]] = set()
        if len(lay.conditions) > 1:
            for i in lay.used:
                for k in range(len(lay.keys)):
                    if a.completion.get(lay.variable(i, k)):
                        present.setdefault((i, k), [])
        else:
            c = lay.conditions[0]
            want = a.state[c]
            placed = [self._pin_places(lay, p)[0] for p in pinned]
            if not any(lay.within(i, c) and lay.satisfies(c, k) == want for i, k in placed):
                # A value within its bounds that establishes it as the state has it (a pin may lie outside them).
                regions = [i for i in lay.used]
                regions.sort(key=lambda i: (lay.regions[i][0] == lay.regions[i][1]) != (self.atoms[c].kind in ("at", "num_cmp")))
                k = self._preferred(lay, [k for k in range(len(lay.keys)) if lay.satisfies(c, k) == want])[0]
                present[(regions[0], k)] = []
        extra: list[Fact] = []
        for p in pinned:
            places = self._pin_places(lay, p)
            chosen = next(((i, k) for i, k in places if (i, k) in present), None)
            if chosen is None and len(lay.conditions) == 1:
                chosen = places[0]
                present[chosen] = []
            if chosen is None:
                # A cutoff's instant in a region no condition's bounds include: its value bears on no condition here.
                i, k = places[0]
                at = p.point.value if isinstance(p.point.value, datetime) else lay.instant(i)
                extra += self.facts(o, at, lay.classes[k])
                continue
            fixed_pins.add(chosen)
            if isinstance(p.point.value, datetime):
                present[chosen].append(_utc(p.point.value))
        out: list[Fact] = []
        for i in sorted({i for i, _ in present}):
            here = [(k, present[(i, k)]) for (j, k) in present if j == i]
            at_pins = {t for _, ts in here for t in ts}
            loose = [k for k, ts in here if not ts]
            free = iter(self._region_instants(lay, i, len(loose), at_pins))
            for k, ts in here:
                kk = k if (i, k) in fixed_pins else self._canonical(lay, k)
                for t in ts or [next(free)]:
                    out += self.facts(o, t, lay.classes[kk])
        return out + extra

    def _region_instants(self, lay: Layout, i: int, m: int, avoid: set[datetime]) -> list[datetime]:
        """m distinct instants of region i, one per class placed there without a pinned instant, none an instant other
        conditions read the observable for or a pinned one (a region that is an instant is that instant)."""
        if m == 0:
            return []
        a, z = lay.regions[i]
        if a is not None and a == z:
            return list(lay.instants(i, m))
        taken = self.reserved.get(lay.observable, set()) | avoid
        for spread in range(m, m + 8):
            got = [t for t in lay.instants(i, spread) if t not in taken][:m]
            if len(got) == m:
                return got
        raise FuzzError(f"region {i} of {lay.observable} has no {m} instants free of those other conditions read")

    def _band_of(self, e: core.ReadValue, v: Fraction) -> int | None:
        for i in range(len(e.bands)):
            if _interval(e, i).contains(v):
                return i
        return None

    def _choose_band(self, e: core.ReadValue, group: Sequence[str], state: Mapping[str, bool], pin: Pin | None) -> int:
        if pin is not None:
            band = self._band_of(e, Fraction(cast(Decimal, pin.point.value)))
            assert band is not None
            return band
        for i in range(len(e.bands)):
            iv = _interval(e, i)
            vals = [iv.compare(self.atoms[d].op, Fraction(self.atoms[d].value)) for d in group]
            if all(v is not Kleene.UNKNOWN and (v is Kleene.TRUE) == state[d] for v, d in zip(vals, group, strict=True)):
                return i
        raise FuzzError(f"no band of {e.id} gives {[(d, state[d]) for d in group]}")

    def _compatible(self, o: str, t: datetime, v: core.CmpValue, state: Mapping[str, bool]) -> bool:
        """Whether a value of `o` at instant t keeps every condition comparing o's values at instants as the state has
        it: no value inside a false At or By satisfies it, and none inside a true Throughout breaks it."""
        lay = self.lowered.layouts.get(o)
        if lay is None:
            return True
        for c in lay.conditions:
            if c not in state or not evaluate._in(_utc(t), *lay.bounds[c]):
                continue
            op, threshold = lay.comparisons[c]
            sat = _holds(_key(v), op, threshold)
            if c in lay.every and state[c] and not sat:
                return False
            if c not in lay.every and not state[c] and sat:
                return False
        return True

    def _instant_values(self, expr: Any, want: bool, t: datetime | None = None,
                        state: Mapping[str, bool] | None = None) -> dict[str, core.CmpValue]:  # fmt: skip
        """Values, one per observable the expression compares, under which it is `want` at an instant (at t, each
        compatible with the state's other conditions)."""
        cmps = [n for n in core.walk(expr) if isinstance(n, core.Cmp)]
        by: dict[str, list[core.Cmp]] = {}
        for c in cmps:
            by.setdefault(c.observable, []).append(c)
        candidates = {o: [v for v in self._candidates(o, [c.value for c in cs])
                          if t is None or state is None or self._compatible(o, t, v, state)]
                      for o, cs in by.items()}  # fmt: skip
        space = 1
        for vs in candidates.values():
            space *= len(vs)
        if space > 100_000:
            raise FuzzError(f"{getattr(expr, 'id', expr)} compares too many values at one instant to search ({space})")
        for combo in itertools.product(*candidates.values()):
            values = dict(zip(candidates, combo, strict=True))
            if _truth(expr, values) == want:
                return values
        raise FuzzError(f"no values make {getattr(expr, 'id', expr)} {want} at {t} with the state's other conditions")

    def _candidates(self, o: str, thresholds: Sequence[core.CmpValue]) -> list[core.CmpValue]:
        obs = self.observables[o]
        if obs.value_type == "number":
            assert obs.precision is not None
            p = Fraction(obs.precision)
            reps = sorted({Fraction(int(Fraction(t.value) / p // 1) + d) * p for t in thresholds
                           if isinstance(t, core.Number) for d in (-1, 0, 1, 2)}, key=lambda x: (x < 0, x))  # fmt: skip
            return [core.Number(value=evaluate._to_decimal(x).quantize(obs.precision)) for x in reps]
        if obs.value_type == "date":
            days = sorted({t.value.toordinal() + d for t in thresholds if isinstance(t, core.OnDay) for d in (-1, 0, 1)})
            return [core.OnDay(value=date.fromordinal(x)) for x in days]
        if obs.value_type == "bool":
            return [core.Flag(value=True), core.Flag(value=False)]
        named = list(dict.fromkeys(str(t.value) for t in thresholds))
        out: list[core.CmpValue] = [core.Category(value=x) for x in named]
        return [*out, core.Category(value="a value other than " + " or ".join(named))]

    def _at(self, expr: Any, t: datetime, facts: Sequence[Fact]) -> bool | None:
        """The expression at instant t on the facts there: None where an observable it compares has no value then."""
        values = {f.observable: f.value for f in facts if f.at == _utc(t)}
        if any(c.observable not in values for c in core.walk(expr) if isinstance(c, core.Cmp)):
            return None
        return _truth(expr, values)

    def _days_shown(self, expr: Any, days: Sequence[tuple[date, datetime, datetime]],
                    facts: Sequence[Fact]) -> list[bool | None]:  # fmt: skip
        """Each counted day's truth on the facts already placed: True where some instant shows the expression, False
        where instants do and none shows it, None where no instant settles it."""
        reads = {c.observable for c in core.walk(expr) if isinstance(c, core.Cmp)}
        out: list[bool | None] = []
        for _, ds, de in days:
            instants = sorted({f.at for f in facts if f.observable in reads and ds <= f.at < de})
            vals = [self._at(expr, t, facts) for t in instants]
            out.append(True if True in vals else (False if vals and None not in vals else None))
        return out

    def _count(self, e: core.CountTrue, group: Sequence[str], state: Mapping[str, bool], pin: Pin | None,
               facts: Sequence[Fact], n: int | None = None) -> list[Fact]:  # fmt: skip
        """Days on which the count's condition holds, as many as its comparisons allow, around the values other
        conditions already placed: a day already shown keeps its truth, a day shown false can be made true by one more
        instant, and every other day gets one published instant (midday), holding or not."""
        days = evaluate.counted_days(e.days)
        shown = self._days_shown(e.expr, days, facts)
        fixed = sum(1 for v in shown if v is True)
        open_days = [j for j, v in enumerate(shown) if v is not True]
        if n is None:
            if pin is not None:
                options = [int(cast(int, pin.point.value))]
            else:
                thresholds = [Fraction(self.atoms[d].value) for d in group]
                options = sorted(range(len(days) + 1), key=lambda m: (min(abs(m - t) for t in thresholds), m))
                options = [m for m in options if all(_holds(Fraction(m), self.atoms[d].op, Fraction(self.atoms[d].value))
                                                     == state[d] for d in group)]  # fmt: skip
            options = [m for m in options if fixed <= m <= fixed + len(open_days)]
            if not options:
                raise FuzzError(f"no count of {e.id}'s {len(days)} days gives {[(d, state.get(d)) for d in group]} with "
                                f"{fixed} days already shown by other conditions' values")  # fmt: skip
            n = options[0]
        n = max(fixed, min(n, fixed + len(open_days)))
        make_true = set(sorted(open_days, key=lambda j: (shown[j] is False, j))[: n - fixed])
        out: list[Fact] = []
        for j in open_days:
            _, ds, de = days[j]
            t = ds + (de - ds) / 2
            if j in make_true:
                values = self._instant_values(e.expr, True, t, state)
            elif shown[j] is None:
                values = self._instant_values(e.expr, False, t, state)
            else:
                continue
            for o, v in values.items():
                out += self.facts(o, t, v)
        return out

    def _duration(self, e: core.DurationIn, group: Sequence[str], state: Mapping[str, bool], pin: Pin | None,
                  facts: Sequence[Fact]) -> list[Fact]:  # fmt: skip
        """A stretch of the window during which the condition holds, as long as the comparisons allow, around the values
        other conditions place. The values already placed cut the window into segments, each holding or not from its
        first instant; one more instant inside a segment changes the truth from there to the segment's end, so a segment
        that holds contributes any length up to its own above zero, one that does not any length below its own; the
        window's start gets a value where none is placed at or before it. Each new value is compatible with the state's
        other conditions."""
        w = e.window
        assert w.start is not None and w.end is not None
        start, end = _utc(w.start), _utc(w.end)
        total = evaluate.duration(start, end, e.unit, w.tz)
        if pin is not None:
            targets = [Fraction(cast(Decimal, pin.point.value))]
        else:
            options: set[Fraction] = {Fraction(0), total}
            for d in group:
                threshold = Fraction(self.atoms[d].value)
                step = Fraction(evaluate._stated_precision(self.atoms[d].value))
                options |= {threshold - step, threshold, threshold + step}
            thresholds = [Fraction(self.atoms[d].value) for d in group] or [total / 2]
            ordered = sorted((x for x in options if 0 <= x <= total), key=lambda x: (min(abs(x - y) for y in thresholds), x))
            targets = [x for x in ordered if all(_holds(x, self.atoms[d].op, Fraction(self.atoms[d].value)) == state[d]
                                                for d in group)]  # fmt: skip
        reads = set(_reads(e))
        instants = sorted({f.at for f in facts if f.observable in reads and f.at < end})
        prior = [t for t in instants if t <= start]
        inside = [t for t in instants if start < t < end]
        head = self._at(e.expr, prior[-1], facts) if prior else None
        later = [self._at(e.expr, t, facts) for t in inside]
        if any(t is None for t in later):
            raise FuzzError(f"{e.id}: a value other conditions place inside its window leaves its condition unknown")
        for target in targets:
            # The window's start is ours where no value at or before it is placed: either truth there, the one that
            # lets the stretch reach the target with the segments other conditions' values fix.
            for first in ([target > 0, not target > 0] if head is None else [head]):
                cuts = [start, *inside, end]
                truths = [first, *later]
                left = target
                plan_cuts: list[tuple[datetime, bool]] = []
                fits = True
                for (a, z), holds in zip(zip(cuts, cuts[1:], strict=False), truths, strict=True):
                    length = evaluate.duration(a, z, e.unit, w.tz)
                    if holds:
                        if left <= 0:  # a segment holding from its first instant adds more than nothing
                            fits = False
                            break
                        take = min(left, length)
                        if take < length:  # holds from a for `take`, then not
                            plan_cuts.append((self._after(a, take, e.unit, w.tz), False))
                        left -= take
                    elif 0 < left < length:  # not from a, then holds for `left` to the segment's end
                        plan_cuts.append((self._after(a, length - left, e.unit, w.tz), True))
                        left = Fraction(0)
                if not fits or left != 0:
                    continue
                try:
                    out: list[Fact] = []
                    if head is None:
                        out += [f for o, v in self._instant_values(e.expr, first, start, state).items()
                                for f in self.facts(o, start, v)]  # fmt: skip
                    for t, holds in plan_cuts:
                        out += [f for o, v in self._instant_values(e.expr, holds, t, state).items()
                                for f in self.facts(o, t, v)]  # fmt: skip
                except FuzzError:
                    continue  # no compatible value at an instant: try the other choice, or the next length
                return out
        raise FuzzError(f"no stretch of {e.id}'s window gives {[(d, state.get(d)) for d in group]} around the values "
                        "other conditions place there")  # fmt: skip

    @staticmethod
    def _after(start: datetime, length: Fraction, unit: str, tz: str) -> datetime:
        """The instant `length` of the unit after start, as the evaluator measures it."""
        if unit in ("hour", "elapsed_day"):
            micros = length * (3_600_000_000 if unit == "hour" else 86_400_000_000)
            if micros.denominator != 1:
                raise FuzzError(f"a duration of {length} {unit} is not a whole number of microseconds")
            return start + timedelta(microseconds=int(micros))
        zone = ZoneInfo(tz)
        t, left = start, length
        while True:
            local = t.astimezone(zone)
            day_end = datetime.combine(local.date() + timedelta(days=1), datetime.min.time(), tzinfo=zone).astimezone(UTC)
            whole = evaluate.duration(t, day_end, unit, tz)
            if left <= whole:
                micros = Fraction((day_end - t) // timedelta(microseconds=1)) * left / whole
                if micros.denominator != 1:
                    raise FuzzError(f"a duration of {length} calendar days ends between microseconds")
                return t + timedelta(microseconds=int(micros))
            left -= whole
            t = day_end

    def _compound(self, node: Any, want: bool, state: Mapping[str, bool], facts: Sequence[Fact]) -> list[Fact]:
        """A temporal condition over a compound comparison: values making it `want` at one instant within its bounds,
        compatible with the state's other conditions, where the facts already placed do not settle it."""
        if isinstance(node, core.At):
            t = _utc(node.at)
        elif isinstance(node, core.By):
            lo = _utc(node.since) if node.since is not None else _utc(node.deadline) - timedelta(days=2)
            t = lo + (_utc(node.deadline) - lo) / 2
        else:
            w = node.window
            hi = _utc(w.end) if w.end is not None else _utc(cast(datetime, w.start)) + timedelta(days=2)
            lo = _utc(w.start) if w.start is not None else hi - timedelta(days=2)
            t = lo + (hi - lo) / 2
        if any(f.at == t and f.observable in _reads(node) for f in facts):
            raise FuzzError(f"{node.id} reads values at {t.isoformat()} that another condition already places")
        out = []
        for o, v in self._instant_values(node.expr, want, t, state).items():
            out += self.facts(o, t, v)
        return out

    def _value_for(self, o: str, t: datetime, facts: Iterable[Fact]) -> core.CmpValue:
        have = [f.value for f in facts if f.observable == o and f.at == _utc(t)]
        if have:
            return have[0]
        if (o, _utc(t)) in self.reference:
            return self.reference[(o, _utc(t))]
        obs = self.observables[o]
        assert obs.precision is not None
        return core.Number(value=obs.precision * 100)

    def _compatible_value(self, o: str, t: datetime, state: Mapping[str, bool], facts: Iterable[Fact]) -> core.CmpValue:
        """A value for a published input nothing compares directly: its reference or default value where that keeps
        the state's other conditions, otherwise the first value of the observable's comparisons that does."""
        first = self._value_for(o, t, facts)
        if self._compatible(o, t, first, state):
            return first
        lay = self.lowered.layouts.get(o)
        thresholds = [core.Number(value=evaluate._to_decimal(x)) if isinstance(x, Fraction) and o in self.observables
                      and self.observables[o].value_type == "number" else first
                      for x in (lay.comparisons[c][1] for c in lay.conditions)] if lay is not None else []  # fmt: skip
        for cand in self._candidates(o, thresholds):
            if self._compatible(o, t, cand, state):
                return cand
        return first

    def _default_number(self, e: Any, facts: list[Fact], state: Mapping[str, bool]) -> list[Fact]:
        """Values for a number nothing compares (the payout quantity's inputs), where the scenario has none yet: a
        published value its reference or default value, a count half its days, a duration its whole window."""
        out: list[Fact] = []
        for n in core.walk(e):
            have = [*facts, *out]
            if isinstance(n, core.ValueAt) and not any(f.observable == n.observable and f.at == _utc(n.at) for f in have):
                out += self.facts(n.observable, n.at, self._value_for(n.observable, n.at, have))
            elif isinstance(n, core.CountTrue | core.DurationIn):
                window = n.days if isinstance(n, core.CountTrue) else n.window
                reads = set(_reads(n))
                lo, hi = _utc(cast(datetime, window.start)), _utc(cast(datetime, window.end))
                if not any(f.observable in reads and lo <= f.at <= hi for f in have):
                    if isinstance(n, core.CountTrue):
                        out += self._count(n, [], state, None, have, n=len(evaluate.counted_days(window)) // 2)
                    else:
                        for o, v in self._instant_values(n.expr, True, lo, state).items():
                            out += self.facts(o, lo, v)
        return out

    # ---- a computed comparison

    def _computed(self, group: Sequence[str], state: Mapping[str, bool], pin: Pin | None, facts: Sequence[Fact],
                  interps: Mapping[str, Interpretation]) -> tuple[list[Fact], dict[str, Interpretation]]:  # fmt: skip
        """Inputs giving every comparison of one computed number the state's truths. Readings are chosen among their
        bands, and published values no other condition places among their precision grid; the evaluator decides each
        trial. An equality is met by solving one published input exactly (by bisection on its grid, the node declared
        monotone in it); an ordered comparison by the readings' corner its monotonicity favours, walked toward the
        threshold while every comparison keeps its truth; failing both, by a search one input at a time."""
        e = cast(core.Computed, cast(core.NumCmp, self.atoms[group[0]]).expr)
        fn = registry.get(e.node)
        readings = [(name, sub) for name, sub in e.inputs.items() if isinstance(sub, core.ReadValue)]
        published = [(name, sub) for name, sub in e.inputs.items() if isinstance(sub, core.ValueAt)]
        other = [name for name, sub in e.inputs.items() if not isinstance(sub, core.ReadValue | core.ValueAt)]
        if other:
            raise FuzzError(f"{e.id}'s inputs {other} are neither readings nor published values; this run realizes a "
                            "computed comparison over readings and published values")  # fmt: skip
        placed = {(f.observable, f.at): f.value for f in facts}
        free = [(name, sub) for name, sub in published if (sub.observable, _utc(sub.at)) not in placed]
        values: dict[tuple[str, datetime], core.CmpValue] = {
            (sub.observable, _utc(sub.at)): self._compatible_value(sub.observable, sub.at, state, facts) for _, sub in free
        }
        if pin is not None:
            solved = self._solve_value(e, Fraction(cast(Decimal, pin.point.value)), facts)
            if isinstance(solved, str):
                raise FuzzError(f"{e.id}: {solved}, with the values this scenario places")
            values.update({(o, t): v for o, t, v in solved})
        bands = {r.id: 0 for _, r in readings}
        if pin is not None:
            bands = {r.id: len(r.bands) - 1 for _, r in readings}  # `_solve` found the values with single-value bands
        checks = [(self.atoms[d], _kleene(state[d])) for d in group]

        def trial(bs: Mapping[str, int], vs: Mapping[tuple[str, datetime], core.CmpValue]) -> int:
            new = [f for (o, t), v in vs.items() for f in self.facts(o, t, v)]
            observations = [f.observation(_PLACEHOLDER) for f in _dedupe([*facts, *new])]
            given = {**interps, **{r.id: (r.bands[bs[r.id]][0], r.bands[bs[r.id]][1]) for _, r in readings}}
            return sum(1 for node, want in checks if self.alone(node, observations, given) is want)

        def result(bs: Mapping[str, int], vs: Mapping[tuple[str, datetime], core.CmpValue]
                   ) -> tuple[list[Fact], dict[str, Interpretation]]:  # fmt: skip
            new = [f for (o, t), v in vs.items() for f in self.facts(o, t, v)]
            return new, {r.id: (r.bands[bs[r.id]][0], r.bands[bs[r.id]][1]) for _, r in readings}

        if pin is not None:
            if trial(bands, values) == len(checks):
                return result(bands, values)
            raise FuzzError(f"{e.id}: the values solved for {pin.point.value} do not give its comparisons {dict(state)}")
        equal = next((n for n, want in checks if (n.op == "==") == (want is Kleene.TRUE) and n.op in ("==", "!=")), None)
        if equal is not None and all(r.bands[-1][0] == r.bands[-1][1] for _, r in readings):
            point = {r.id: len(r.bands) - 1 for _, r in readings}
            solved = self._solve_value(e, Fraction(equal.value), facts)
            if not isinstance(solved, str):
                vs = {**values, **{(o, t): v for o, t, v in solved}}
                if trial(point, vs) == len(checks):
                    return result(point, vs)
        ordered = next(((n, want) for n, want in checks if n.op not in ("==", "!=")), None)
        if ordered is not None:
            n, want = ordered
            low = (n.op in ("<", "<=")) == (want is Kleene.TRUE)  # the computed number is wanted low

            def order(name: str, r: core.ReadValue) -> list[int]:
                direction = fn.monotone.get(name)
                if direction is None:
                    return [len(r.bands) - 1] if r.bands[-1][0] == r.bands[-1][1] else list(range(len(r.bands)))
                up = list(range(len(r.bands)))
                return up if (direction == "increasing") == low else up[::-1]

            orders = {r.id: order(name, r) for name, r in readings}
            corner = {rid: o[0] for rid, o in orders.items()}
            if trial(corner, values) == len(checks):
                for rid, o in orders.items():  # toward the threshold while every comparison keeps its truth
                    for k in o[1:]:
                        if trial({**corner, rid: k}, values) != len(checks):
                            break
                        corner = {**corner, rid: k}
                return result(corner, values)
        # A search one input at a time, for the most comparisons with their truths.
        best_b, best_v = dict(bands), dict(values)
        score = trial(best_b, best_v)
        for _ in range(4):
            before = score
            for _, r in readings:
                for k in range(len(r.bands)):
                    if (got := trial({**best_b, r.id: k}, best_v)) > score:
                        score, best_b = got, {**best_b, r.id: k}
            for _, sub in free:
                key = (sub.observable, _utc(sub.at))
                obs = self.observables[sub.observable]
                assert obs.precision is not None
                k0 = int(Fraction(cast(core.Number, best_v[key]).value) / Fraction(obs.precision))
                for k in [k0 + sign * 2**j for j in range(41) for sign in (1, -1)]:
                    v = core.Number(value=evaluate._to_decimal(Fraction(k) * Fraction(obs.precision)).quantize(obs.precision))
                    if not self._compatible(sub.observable, sub.at, v, state):
                        continue
                    if (got := trial(best_b, {**best_v, key: v})) > score:
                        score, best_v = got, {**best_v, key: v}
            if score == len(checks) or score == before:
                break
        if score == len(checks):
            return result(best_b, best_v)
        raise FuzzError(f"no readings or published values found give {e.id}'s comparisons {dict(state)} "
                        f"(the values other conditions place: {[f.shown() for f in facts if any(f.observable == s.observable for _, s in published)]})")  # fmt: skip

    def _solve_value(self, e: core.Computed, v: Fraction, facts: Iterable[Fact]
                     ) -> tuple[tuple[str, datetime, core.CmpValue], ...] | str:  # fmt: skip
        """Published values giving a computed number exactly v, its readings at their single values. Each published input
        the node is declared monotone in is searched on its precision grid by bisection (a value the node refuses being
        outside its domain), the others at their reference or default values; then, where that finds none, each such
        input is bisected again while one other free published input steps through JOINT_STEPS grid steps either side
        of its reference value (external evaluation, November 2: z = 2x + 3y reaches 501 only with both inputs moved).
        Returns the values, or what was searched where none was found, which proves nothing about the node."""
        fn = registry.get(e.node)
        args: dict[str, Fraction] = {}
        for name, sub in e.inputs.items():
            if isinstance(sub, core.ReadValue):
                args[name] = Fraction(sub.bands[-1][0])
            else:
                assert isinstance(sub, core.ValueAt)  # the callers refuse other inputs
                args[name] = Fraction(cast(core.Number, self._value_for(sub.observable, sub.at, facts)).value)
        placed = {(f.observable, f.at) for f in facts}
        free = {name: sub for name, sub in e.inputs.items()
                if isinstance(sub, core.ValueAt) and (sub.observable, _utc(sub.at)) not in placed}  # fmt: skip
        step = {name: Fraction(cast(Decimal, self.observables[sub.observable].precision)) for name, sub in free.items()}

        def value(name: str, k: int) -> core.Number:
            obs = self.observables[free[name].observable]
            return core.Number(value=evaluate._to_decimal(k * step[name]).quantize(cast(Decimal, obs.precision)))

        def bisect(name: str, base: Mapping[str, Fraction]) -> int | None:
            up = fn.monotone[name] == "increasing"

            def f(k: int) -> Fraction | None:
                try:
                    return Fraction(fn.fn(**{**base, name: k * step[name]}))
                except ValueError:
                    return None  # the node refuses this input: outside its domain

            k0 = int(base[name] / step[name])
            lo, hi, span = k0, k0, 1
            while (y := f(lo)) is not None and (y > v if up else y < v) and span < 2**62:
                lo, span = lo - span, span * 2
            span = 1
            while (y := f(hi)) is not None and (y < v if up else y > v) and span < 2**62:
                hi, span = hi + span, span * 2
            while lo < hi:
                mid = (lo + hi) // 2
                y = f(mid)
                if y is None or y == v:
                    lo = hi = mid
                elif (y < v) == up:
                    lo = mid + 1
                else:
                    hi = mid
            return lo if f(lo) == v else None

        monotone = [n for n in free if fn.monotone.get(n) is not None]
        for name in monotone:
            k = bisect(name, args)
            if k is not None:
                return ((free[name].observable, _utc(free[name].at), value(name, k)),)
        for name in monotone:
            for other in (o for o in free if o != name):
                k1 = int(args[other] / step[other])
                for d in sorted(range(-JOINT_STEPS, JOINT_STEPS + 1), key=abs):
                    if d == 0:
                        continue
                    k = bisect(name, {**args, other: (k1 + d) * step[other]})
                    if k is not None:
                        return ((free[name].observable, _utc(free[name].at), value(name, k)),
                                (free[other].observable, _utc(free[other].at), value(other, k1 + d)))
        searched = (f"each of {monotone} bisected on its precision grid, the other inputs at their reference values"
                    + (f", then with one other free input stepped {JOINT_STEPS} steps either side" if len(free) > 1 else ""))
        return f"no published value of {e.id}'s inputs found giving it exactly {v} ({searched})"

    def alone(self, node: Any, observations: Sequence[core.SyntheticObservation],
              interps: Mapping[str, Interpretation]) -> Kleene:  # fmt: skip
        """One atomic condition's truth on these inputs, by the evaluator: the contract with the condition as its
        trigger."""
        sub = self.ir.model_copy(update={"trigger": node})
        roots = [node] + ([self.ir.payout.quantity] if self.ir.payout.quantity is not None else [])
        needed = {n.id for r in roots for n in core.walk(r) if isinstance(n, core.Interpretive | core.ReadValue)}
        given = {k: v for k, v in interps.items() if k in needed}
        for r in self.readings:
            if r.id in needed and r.id not in given:
                given[r.id] = (r.bands[0][0], r.bands[0][1])
        return evaluate.evaluate(sub, observations, given, at=self.at).trigger_truth

    # ---- the concrete scenario

    def concrete(self, a: _Abstract, facts: list[Fact] | None = None, interps: dict[str, Interpretation] | None = None,
                 withdrawn: str | None = None) -> Scenario:  # fmt: skip
        if facts is None or interps is None:
            facts, interps = self.realize(a)
        facts = _dedupe(facts)
        content = core.canonical_json({
            "generator": f"{GENERATOR.id} v{GENERATOR.version}", "contract": self.ir.ref.model_dump(), "scenario": a.id,
            "stress": a.stress, "facts": [f.shown() for f in facts],
            "interpretations": {k: _shown(v) for k, v in sorted(interps.items())},
        }).encode()  # fmt: skip
        artifact = core.evidence.synthetic_artifact_id(content)
        observations = tuple(f.observation(artifact) for f in facts)
        truths = {cid: self.alone(node, observations, interps) for cid, node in self.atoms.items()}
        if a.role != "stress":
            wrong = {c: t.value for c, t in truths.items() if t is not _kleene(a.state[c])}
            if wrong:
                raise FuzzError(f"scenario {a.id}: the inputs built for state {a.state} give {wrong} by the evaluator; "
                                f"the build does not mean the state (facts: {[f.shown() for f in facts]})")  # fmt: skip
        ev = evaluate.evaluate(self.ir, observations, interps, at=self.at)
        state_hash = core.sha256(core.canonical_json({
            "contract": self.ir.ref.model_dump(), "truths": {c: t.value for c, t in sorted(truths.items())},
            "stress": a.stress,
        }))  # fmt: skip
        exercised = self._exercised(truths, withdrawn)
        return Scenario(a.id, a.role, a.stress, a.base, tuple(a.pins), tuple(facts), dict(interps), truths, state_hash,
                        artifact, content, observations, ev, exercised, withdrawn)  # fmt: skip

    def decisive(self, truths: Mapping[str, Kleene]) -> list[str]:
        """The atomic conditions the trigger's truth turns on in this state: changing one alone changes it. The
        evaluator composes the truths (the trigger with each atomic condition standing as an interpretive node given its
        truth), so an unknown condition is decisive where its value would decide the trigger."""
        base = self._composed(truths)
        out = []
        for c in self.atoms:
            if any(self._composed({**truths, c: v}) is not base for v in (Kleene.TRUE, Kleene.FALSE, Kleene.UNKNOWN)
                   if v is not truths[c]):  # fmt: skip
                out.append(c)
        return out

    def _composed(self, truths: Mapping[str, Kleene]) -> Kleene:
        def stand(e: Any) -> Any:
            if isinstance(e, core.And | core.Or):
                return e.model_copy(update={"args": [stand(a) for a in e.args]})
            if isinstance(e, core.Not):
                return e.model_copy(update={"arg": stand(e.arg)})
            return core.Interpretive(id=e.id, question_template=FORCED, bindings={})

        sub = self.ir.model_copy(update={"trigger": stand(self.ir.trigger)})
        given: dict[str, Interpretation] = dict(truths)
        for r in self.readings:
            if self.ir.payout.quantity is not None and r.id in {n.id for n in core.walk(self.ir.payout.quantity)}:
                given[r.id] = (r.bands[0][0], r.bands[-1][1])
        return evaluate.evaluate(sub, [], given, at=self.at).trigger_truth

    def _exercised(self, truths: Mapping[str, Kleene], withdrawn: str | None) -> frozenset[str]:
        conds = set(self.decisive(truths)) | ({withdrawn} if withdrawn else set())
        out: set[str] = set()
        for c in conds:
            out |= {n.id for n in core.walk(self.atoms[c])}
            p = self.parents.get(c)
            while p is not None:
                out.add(p)
                p = self.parents.get(p)
        return frozenset(out)

    # ---- stress classes

    def stress(self, base: Sequence[Scenario]) -> list[Scenario]:
        fired = [s for s in base if s.role == "coverage" and s.evaluation.trigger_truth is Kleene.TRUE]
        quiet = [s for s in base if s.role == "coverage" and s.evaluation.trigger_truth is Kleene.FALSE]
        bases = [x for x in (fired[:1] + quiet[:1])] or list(base[:1])
        out: list[Scenario] = []
        n = 0

        def add(cls: StressClass, b: Scenario, facts: list[Fact], interps: dict[str, Interpretation],
                withdrawn: str | None = None) -> None:  # fmt: skip
            nonlocal n
            n += 1
            a = _Abstract(f"x{n:02d}", "stress", {}, {}, [], cls, b.id)
            out.append(self.concrete(a, facts, interps, withdrawn))

        def variant(cls: StressClass, b: Scenario, build: Any) -> None:
            try:
                facts, interps, withdrawn = build()
            except FuzzError as e:
                self.notes.append(f"stress class {cls} on {b.id}: not built ({e})")
                return
            add(cls, b, facts, interps, withdrawn)

        read = [o for o in dict.fromkeys(n.observable for n in core.walk(self.ir.trigger)
                                         if isinstance(n, core.Cmp | core.ValueAt))]  # fmt: skip
        multi = [o for o in read if len(self.consulted[o]) >= 2]
        has_interpretive = any(isinstance(n, core.Interpretive) for n in self.atoms.values())
        for b in bases:
            o = next((x for x in multi if any(f.observable == x for f in b.facts)), None)
            if o is not None:
                variant("conflicting_sources", b, lambda b=b, o=o: (self._conflict(b, o), dict(b.interpretations), None))
        for b in bases:
            o = next((x for x in read if any(f.observable == x for f in b.facts)), None)
            if o is not None:
                variant("revised_value", b, lambda b=b, o=o: (self._revision(b, o), dict(b.interpretations), None))
        if has_interpretive:
            for cls in PRESENTED:
                for b in bases:
                    add(cls, b, list(b.facts), dict(b.interpretations))
        for b in bases:
            for c in self.decisive(b.truths):
                facts, interps = self._withdraw(b, c)
                observations = [f.observation(_PLACEHOLDER) for f in facts]
                if self.alone(self.atoms[c], observations, interps) is Kleene.UNKNOWN:
                    add("evidence_missing_at_expiry", b, facts, interps, withdrawn=c)
                    break
        return out

    def _other_value(self, b: Scenario, f: Fact) -> core.CmpValue:
        """Another value for a deciding fact: one some comparison of its observable treats differently, or, for a value
        read only through a computation, one under which a condition reading it takes another truth on this scenario's
        inputs (by the evaluator); failing both, twice the value (or one step, for zero): a vintage that changes no
        truth still asks the resolvers to take the right one."""
        o, v = f.observable, f.value
        cmps = [(n.op, n.value) for n in core.walk(self.ir.trigger) if isinstance(n, core.Cmp) and n.observable == o]
        cmps += [(n.op, core.Number(value=n.value)) for n in self.atoms.values()
                 if isinstance(n, core.NumCmp) and isinstance(n.expr, core.ValueAt) and n.expr.observable == o]  # fmt: skip

        def vector(x: core.CmpValue) -> tuple[bool, ...]:
            return tuple(_holds(_key(x), op, _key(t)) for op, t in cmps)

        if cmps:
            for cand in self._candidates(o, [t for _, t in cmps]):
                if vector(cand) != vector(v):
                    return cand
            raise FuzzError(f"no value of {o} is compared differently from {v.value}")
        obs = self.observables[o]
        if not isinstance(v, core.Number) or obs.precision is None:
            raise FuzzError(f"{o} is read only through a computation, as a number; its value is {v.value}")
        readers = [cid for cid, n in self.atoms.items() if o in _reads(n)]
        p = obs.precision
        tries = [v.value * 10, v.value / 10, Decimal(0), v.value * 2, v.value + p * 100]
        for x in tries:
            cand = core.Number(value=x.quantize(p))
            if cand.value == v.value:
                continue
            facts = [replace(g, value=cand) if g == f else g for g in b.facts]
            observations = [g.observation(_PLACEHOLDER) for g in _dedupe(facts)]
            if any(self.alone(self.atoms[c], observations, b.interpretations) is not b.truths[c] for c in readers):
                return cand
        return core.Number(value=(v.value * 2 if v.value else p).quantize(p))

    def _deciding(self, b: Scenario, o: str) -> Fact:
        return next(f for f in b.facts if f.observable == o and f.source == (self.agree.get(o) or self.consulted[o])[0])

    def _conflict(self, b: Scenario, o: str) -> list[Fact]:
        f = self._deciding(b, o)
        alt = self._other_value(b, f)
        if o in self.agree:
            other = self.agree[o][1]
            return [replace(x, value=alt) if (x.observable, x.source, x.at) == (o, other, f.at) else x for x in b.facts]
        return [*b.facts, Fact(o, self.consulted[o][1], f.at, alt, f.published_at, f.final)]

    def _revision(self, b: Scenario, o: str) -> list[Fact]:
        """Another vintage of a deciding value, where the revision policy does not select it: a later revision under
        first print, an earlier print under as-of-expiry, an earlier print not marked final under final."""
        f = self._deciding(b, o)
        alt = self._other_value(b, f)
        same = [x for x in b.facts if (x.observable, x.at) == (o, f.at)]
        rest = [x for x in b.facts if (x.observable, x.at) != (o, f.at)]
        policy = self.observables[o].revision_policy
        if policy == "first_print":
            earlier = [replace(x, published_at=x.published_at - timedelta(days=1)) for x in same]
            return rest + earlier + [replace(x, value=alt) for x in same]
        return rest + same + [replace(x, value=alt, published_at=x.published_at - timedelta(days=1), final=False)
                              for x in same]  # fmt: skip

    def _withdraw(self, b: Scenario, c: str) -> tuple[list[Fact], dict[str, Interpretation]]:
        """The scenario with the evidence of condition c withdrawn, so that it is not established at expiry: an
        interpretive condition unknown; every reading it reads its whole range; every published value it reads within
        its bounds or window gone."""
        node = self.atoms[c]
        interps = dict(b.interpretations)
        if isinstance(node, core.Interpretive):
            interps[c] = Kleene.UNKNOWN
            return list(b.facts), interps
        for r in (r for r in core.walk(node) if isinstance(r, core.ReadValue)):
            for x in self.readings:
                if _content(x) == _content(r):
                    interps[x.id] = (r.bands[0][0], r.bands[-1][1])
        reads = set(_reads(node))
        lay = self.layout_of.get(c)
        if lay is not None:
            lo, lo_closed, hi, hi_closed = lay.bounds[c]
            facts = [f for f in b.facts if not (f.observable in reads and evaluate._in(f.at, lo, lo_closed, hi, hi_closed))]
        else:
            span = _span(node)
            facts = [f for f in b.facts if not (f.observable in reads and span[0] <= f.at <= span[1])]
        return facts, interps


def _children(n: Any) -> list[Any]:
    if isinstance(n, core.And | core.Or):
        return list(n.args)
    if isinstance(n, core.Not):
        return [n.arg]
    if isinstance(n, core.By | core.At | core.Throughout | core.CountTrue | core.DurationIn | core.NumCmp):
        return [n.expr]
    if isinstance(n, core.Computed):
        return list(n.inputs.values())
    return []


def _reads(node: Any) -> list[str]:
    return list(dict.fromkeys(n.observable for n in core.walk(node) if isinstance(n, core.Cmp | core.ValueAt)))


def _span(node: Any) -> tuple[datetime, datetime]:
    """The instants a condition's evidence lies in, widely: its window, or its instant."""
    lo, hi = datetime.max.replace(tzinfo=UTC), datetime.min.replace(tzinfo=UTC)
    for n in core.walk(node):
        ts: list[datetime | None] = []
        if isinstance(n, core.At | core.ValueAt):
            ts = [n.at]
        elif isinstance(n, core.By):
            ts = [n.since or datetime.min.replace(tzinfo=UTC), n.deadline]
        elif isinstance(n, core.Throughout | core.DurationIn):
            ts = [n.window.start or datetime.min.replace(tzinfo=UTC), n.window.end or datetime.max.replace(tzinfo=UTC)]
        elif isinstance(n, core.CountTrue):
            ts = [n.days.start, n.days.end]
        for t in ts:
            if t is not None:
                lo, hi = min(lo, _utc(t)), max(hi, _utc(t))
    return lo, hi


def _group_key(g: frozenset[str], conds: Sequence[str]) -> str:
    return "group:" + "+".join(sorted(g, key=list(conds).index))


def _content(node: Any) -> str:
    return evaluate._content(node)


def _interval(e: core.ReadValue, i: int) -> evaluate.Interval:
    return evaluate._band(e, (e.bands[i][0], e.bands[i][1]))[0]


def _shown(v: Interpretation) -> str:
    return v.value if isinstance(v, Kleene) else f"[{v[0]}, {v[1]}]"


def _dedupe(facts: list[Fact]) -> list[Fact]:
    """Each published value once; two values one source publishes for one instant at one time are one value only where
    they agree (the evaluator refuses a disagreement, which a build never makes)."""
    seen: dict[tuple[str, str, datetime, datetime], Fact] = {}
    for f in facts:
        k = (f.observable, f.source, f.at, f.published_at)
        if k in seen and seen[k].value != f.value:
            raise FuzzError(f"two values of {f.observable} from {f.source} for {f.at} at {f.published_at}: "
                            f"{seen[k].value.value} and {f.value.value}; the build placed two values on one instant")  # fmt: skip
        seen.setdefault(k, f)
    return sorted(seen.values(), key=lambda f: (f.observable, f.at, f.published_at, f.source))


_PLACEHOLDER = core.evidence.synthetic_artifact_id(b"requisite.fuzz: a build's trial inputs, never committed")


__all__ = ["EXACT_UP_TO", "FORCED", "STRESS", "Fact", "FuzzError", "Pin", "Plan", "Scenario", "plan", "sources"]
