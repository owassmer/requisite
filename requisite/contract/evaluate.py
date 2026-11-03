"""The single settlement function (spec I8, never cut): what a contract's IR settles to, from the observations of its
observables and the resolved answers of its interpretive conditions, at an instant. It computes payoffs in construction,
the outcomes printed beside a rulebook's examples, ground truth in fuzzing and settlement at resolution; nothing else in
the product computes what a contract settles to (tests/contract/test_settlement_function.py checks the source).

It keeps unknown as unknown (spec I6, never cut). The trigger is evaluated in Kleene's three-valued logic (spec §6.3,
core.kleene): a required condition that is unknown never yields a definitive YES, an unknown count is the interval
[k, k + u] (addendum A05 §5.3), and the procedure's rule for an unknown at expiry decides the procedure's status without
rewriting the trigger's truth (A05 §5.2). The evaluator never asks a model anything: interpretive conditions arrive as
the truths of their resolved judgments (core.kleene.from_status: affirmative true, negative false, not_established and
contested unknown), and readings as the band of their resolved answer.

What an observation means. Each core.Observation (core.SyntheticObservation in a synthetic case, never mixed: I4) is one
vintage of one source's value of an observable for one instant (`at`, "the instant the value is for"). For each source
and instant the observable's revision policy selects a vintage (first print; the latest published by expiration; the
final one, with its deadline or fallback), and across sources the procedure's conflict policy selects the value
(precedence_first_available: the highest-ranked source in the procedure's precedence that has published for that
instant; require_agreement: the named sources must agree, or the value is disputed). Only records published by the
instant of evaluation and by the contract's observation cutoff are visible: the cutoff is the contract's own rule for
what it considers. A value no source has published by the instant it is needed is unknown, never a default.

Instants. An At or a ValueAt reads the value for exactly its instant: a value is "for" an instant (a rate in effect on a
day is recorded as the value for that day's instant). A By ranges over the instants, within its bounds, for which a
consulted source has published a value of the observables its condition compares, and a Throughout over those within
its window. Until the observation cutoff more instants can still be published, so a By is true once some instant shows
its condition and otherwise unknown, and a Throughout false once some instant breaks its condition and otherwise unknown.
After the cutoff the set is closed; a By or a Throughout over no instant at all is unknown, since nothing established
it (I6), and the procedure's rule for an unknown at expiry says what that means. CountTrue counts the calendar days of
its window, in the window's zone, wholly inside it: a day holds when some instant that day shows the condition; it fails
when, the instants closed, the instants that day show it fails; and it is unknown otherwise, so with no value for it, or
before the cutoff with none yet showing it (another value for the day may still be published). DurationIn measures the
time a condition holds, each value holding from its instant until the next one, as an interval over unknown stretches;
before the cutoff a value for any instant may still be published and split any stretch, so it is [0, the window].
These choices keep the trigger's truth monotone as information arrives: once true or false, a publication cannot reverse
it, except by a revision as_of_expiry selects before expiration. For the same reason precedence settles on a lower-ranked
source's value only once the cutoff has passed: before it, a source ranked above may still publish.

Numbers are exact (Fractions), so a third of a day or of a dollar is never rounded on the way. A numeric observable's
values are compared at its declared precision: a value not stated at that precision is refused, not rounded. A reading's
band is [low, high), the last [low, high]; an unknown reading is the whole range. A Computed node is called on its
inputs' values; over interval inputs its value is the node at the two corners of the box where every open input is
declared monotone (requisite.compute.registry), and unknown (unbounded) otherwise. A comparison over an interval is
true where every value in it satisfies the comparison, false where none does, and unknown otherwise.

The procedure (A05 §5.1-5.2; core.ResolutionProcedure). Before expiration the contract is pending, whatever its
trigger's truth so far, unless it expires early: with early_expiration "when_true" it resolves YES once its trigger is
true, at that instant ("first_true") or at a date an exchange rule sets ("per_exchange_rule"), which stays unresolved,
named. At or after expiration a true trigger resolves YES and a false one NO. An unknown trigger stays pending until the
observation cutoff has also passed; then a dispute among the values it read makes it disputed, and otherwise the
contract's rule for an unknown at expiry applies: NO, YES, void (with its stated transfer), or an extension, after which
an unknown trigger follows the contract's rule for after the extension. Every rule applied is recorded in `policies`,
as a reference to the contract clause that states it.

Transfers (A05 §5.1, §5.3) are per unit, dated, in the payout's currency: the payout by its kind on its quantity,
rounded by the contract's quantum and mode, paid at the resolution instant plus the settlement and payment lags in the
contract's day count and time zone, or per period for a per-period payment. A quantity that is still an interval at
settlement is not guessed: the contract states no missing-data rule for a quantity (its unknown-at-expiry rule is the
trigger's), so the transfer is unresolved and the evaluation says on what. So is a transfer whose date the contract
leaves to a rule outside the IR.

Lowering for coverage (spec §8.6 step 1; A09 §10.2-10.3). `lower` gives the trigger's boolean skeleton in the coverage
kernel's form (requisite.fuzz.coverage): And, Or and Not over one variable per atomic condition, named by its IR node id,
with the boundary points of each comparison and the constraints the IR itself places between its conditions (a linked
threshold, a repeated predicate, one observable read at related instants and ranges; `_implied`). A threshold over a
numeric observable is stepped at its declared precision, a count at k - 1, k and k + 1, a cutoff instant at the
comparison's resolution; a reading, a duration or a computed number declares no precision, so its threshold is stepped
at the precision it is stated to (0.00 steps by a cent), and a duration's threshold within the window's length. The
lowering also gives, for each observable its conditions compare at instants, the layout its constraints are over
(`Layout`: the regions time is cut into and the classes of value), from which the fuzzer realizes any state the kernel
finds as observations. The kernel then works on abstract condition values only; a scenario's outcome is always this
module's `evaluate`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Context, Decimal, Inexact, localcontext
from fractions import Fraction
from functools import cached_property
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo

import requisite.library.nodes  # noqa: F401  registers the library's consequence nodes, which a Computed names
from requisite.compute import registry
from requisite.contract import calendar
from requisite.contract.counts import count_true
from requisite.core import (
    And,
    At,
    By,
    CashTransfer,
    Category,
    Cmp,
    CmpValue,
    Computed,
    ContractIR,
    CountTrue,
    DurationIn,
    Exists,
    Flag,
    Interpretive,
    Not,
    Number,
    NumCmp,
    Observable,
    Observation,
    OnDay,
    Or,
    ProcedureStatus,
    ReadValue,
    Ref,
    ResolutionEvaluation,
    ScenarioExample,
    SyntheticObservation,
    Throughout,
    TimeWindow,
    ValueAt,
    walk,
)
from requisite.core.base import canonical_json, sha256
from requisite.core.contract import VALUE_KIND
from requisite.core.kleene import Kleene, conj, disj, neg, plain
from requisite.fuzz import coverage as cov

EVALUATOR = Ref(id="requisite.contract.evaluate", version=1)  # this module's semantics, recorded in every evaluation

Reading = tuple[Decimal, Decimal]  # a ReadValue's resolved band, or a run of bands, as (low, high)


@dataclass(frozen=True)
class Members:
    """An Exists node's members, an input of evaluation like the interpretations (core.Exists): each member the sources
    document, by its identity (the MEMBER binding its conditions were asked under), with each judged condition of the
    node's body resolved for it; and `complete`, TRUE where every one of the node's sources was read through the
    observation cutoff (a fact of the retrieval), so the list is every member the class holds."""

    members: Mapping[str, Mapping[str, Kleene | Reading]]
    complete: Kleene


Interpretation = Kleene | Reading | Members
Observations = Sequence[Observation] | Sequence[SyntheticObservation]


class IRError(ValueError):
    """A contract the evaluator cannot give one meaning to: a construct whose meaning the definition leaves open is
    refused rather than guessed (addendum A05 §5.5)."""


# ---------------------------------------------------------------- exact intervals


@dataclass(frozen=True)
class Interval:
    """A set of numbers between lo and hi, each end included or not; None is an unbounded end."""

    lo: Fraction | None
    hi: Fraction | None
    lo_closed: bool = True
    hi_closed: bool = True

    @staticmethod
    def point(x: Fraction) -> Interval:
        return Interval(x, x)

    @staticmethod
    def everything() -> Interval:
        return Interval(None, None, False, False)

    @property
    def is_point(self) -> bool:
        return self.lo is not None and self.lo == self.hi

    def contains(self, x: Fraction) -> bool:
        above = self.lo is None or x > self.lo or (x == self.lo and self.lo_closed)
        below = self.hi is None or x < self.hi or (x == self.hi and self.hi_closed)
        return above and below

    def compare(self, op: str, v: Fraction) -> Kleene:
        """TRUE where every number of the interval satisfies `x op v`, FALSE where none does, UNKNOWN otherwise."""
        if op == "==":
            if self.is_point:
                return Kleene.TRUE if self.lo == v else Kleene.FALSE
            return Kleene.UNKNOWN if self.contains(v) else Kleene.FALSE
        if op == "!=":
            return neg(self.compare("==", v))
        lo, hi = self.lo, self.hi
        if op in ("<", "<="):
            strict = op == "<"
            every = hi is not None and (hi < v or (hi == v and (not strict or not self.hi_closed)))
            none = lo is not None and (lo > v or (lo == v and (strict or not self.lo_closed)))
        elif op in (">", ">="):
            strict = op == ">"
            every = lo is not None and (lo > v or (lo == v and (not strict or not self.lo_closed)))
            none = hi is not None and (hi < v or (hi == v and (strict or not self.hi_closed)))
        else:
            raise IRError(f"{op!r} is not a comparison")
        if every:
            return Kleene.TRUE
        return Kleene.FALSE if none else Kleene.UNKNOWN

    def __str__(self) -> str:
        left = "(-inf" if self.lo is None else ("[" if self.lo_closed else "(") + _show(self.lo)
        right = "+inf)" if self.hi is None else _show(self.hi) + ("]" if self.hi_closed else ")")
        return f"{left}, {right}"


def _show(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else (str(_to_decimal(x)) if _terminates(x) else f"{x}")


def _terminates(x: Fraction) -> bool:
    d = x.denominator
    for p in (2, 5):
        while d % p == 0:
            d //= p
    return d == 1


_EXACT = Context(prec=1000, traps=[Inexact])


def _to_decimal(x: Fraction, rounding: str | None = None) -> Decimal:
    """A Fraction as a Decimal: exactly where it terminates, and otherwise rounded the given way at 34 digits (only
    outward bounds of an interval are ever rounded, so the interval still holds the value)."""
    if _terminates(x):
        with localcontext(_EXACT):
            return Decimal(x.numerator) / Decimal(x.denominator)
    if rounding is None:
        raise ValueError(f"{x} has no finite decimal form")
    with localcontext(Context(prec=34, rounding=rounding)):
        return Decimal(x.numerator) / Decimal(x.denominator)


def _exact_number(x: object, what: str) -> Fraction:
    if isinstance(x, bool) or isinstance(x, float) or not isinstance(x, Fraction | Decimal | int):
        raise TypeError(f"{what} is an exact number (Fraction, Decimal or int), not {type(x).__name__}")
    return Fraction(x)


# ---------------------------------------------------------------- reasons travel with values


@dataclass(frozen=True)
class _Why:
    text: str
    disputed: bool = False


@dataclass(frozen=True)
class _T:
    value: Kleene
    why: tuple[_Why, ...] = ()


@dataclass(frozen=True)
class _N:
    interval: Interval
    why: tuple[_Why, ...] = ()


def _union(groups: Iterable[tuple[_Why, ...]]) -> tuple[_Why, ...]:
    return tuple(dict.fromkeys(w for g in groups for w in g))


@dataclass(frozen=True)
class _Pick:
    kind: Literal["value", "absent", "unknown", "disputed"]
    value: CmpValue | None = None
    why: str = ""


_ABSENT = _Pick("absent")


@dataclass(frozen=True)
class _Vintage:
    published_at: datetime
    value: CmpValue
    final: bool


def _utc(t: datetime) -> datetime:
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError(f"an instant needs its time zone: {t!r}")
    return t.astimezone(UTC)


def _iso(t: datetime) -> str:
    return _utc(t).isoformat()


# ---------------------------------------------------------------- the IR's static shape


ORDERED = ("<", "<=", ">", ">=")
TEMPORAL = (By, At, Throughout, CountTrue, DurationIn)


def _level(e: Any) -> frozenset[str]:
    """The observables a condition compares at the instant an enclosing temporal operator gives: those of its Cmps not
    inside another temporal operator, NumCmp or interpretive node."""
    if isinstance(e, Cmp):
        return frozenset({e.observable})
    if isinstance(e, And | Or):
        return frozenset().union(*(_level(a) for a in e.args))
    if isinstance(e, Not):
        return _level(e.arg)
    return frozenset()


def _bounded(w: TimeWindow, what: str) -> None:
    if w.start is None or w.end is None:
        raise IRError(f"{what} ranges over a bounded window; this one is open at an end")


def _check_ir(ir: ContractIR) -> None:
    """Refuse what the evaluator cannot give one meaning to, before evaluating anything."""
    kinds = {o.id: o.value_type for o in ir.observables}

    def expr(e: Any, timed: bool) -> None:
        if isinstance(e, Cmp):
            if not timed:
                raise IRError(
                    f"{e.id} compares {e.observable} outside By, At, Throughout, CountTrue or DurationIn, so no instant "
                    "says which of its values to read; wrap it in At"
                )
            if kinds[e.observable] in ("text", "category", "bool") and e.op in ORDERED:
                raise IRError(f"{e.id} orders {e.observable}, a {kinds[e.observable]}; only == and != compare it")
        elif isinstance(e, And | Or):
            for a in e.args:
                expr(a, timed)
        elif isinstance(e, Not):
            expr(e.arg, timed)
        elif isinstance(e, By | At | Throughout):
            if not _level(e.expr):
                raise IRError(f"{e.id} reads its condition at instants, but the condition compares no observable there")
            expr(e.expr, True)
        elif isinstance(e, NumCmp):
            number(e.expr)
        elif isinstance(e, Interpretive):
            pass
        elif isinstance(e, Exists):
            expr(e.expr, timed)
        else:
            raise IRError(f"{getattr(e, 'id', e)!r} is not a trigger condition")

    def number(n: Any) -> None:
        if isinstance(n, CountTrue | DurationIn):
            if not _level(n.expr):
                raise IRError(f"{n.id} counts days on which its condition holds, but it compares no observable")
            _bounded(n.days if isinstance(n, CountTrue) else n.window, n.id)
            expr(n.expr, True)
        elif isinstance(n, Computed):
            try:
                node = registry.get(n.node)
            except registry.UnknownNode as e:
                raise IRError(f"{n.id} names node {n.node.id} v{n.node.version}, which is not registered") from e
            if set(node.inputs) != set(n.inputs):
                raise IRError(f"{n.id} gives inputs {sorted(n.inputs)}; node {node.id} takes {sorted(node.inputs)}")
            for sub in n.inputs.values():
                number(sub)
        elif not isinstance(n, ReadValue | ValueAt):
            raise IRError(f"{getattr(n, 'id', n)!r} is not a number")

    expr(ir.trigger, False)
    consulted, _ = _consulted(ir)
    roots = [ir.trigger] + ([ir.payout.quantity] if ir.payout.quantity is not None else [])
    for o in dict.fromkeys(n.observable for r in roots for n in walk(r) if isinstance(n, Cmp | ValueAt)):
        if not consulted[o]:
            raise IRError(f"{o} is read by the contract, but the procedure consults none of its sources")
    payout = ir.payout
    if payout.quantity is not None:
        number(payout.quantity)
    if payout.payment == "per_period":
        if payout.kind != "per_unit":
            raise IRError(
                f"a per-period {payout.kind} payout: the definition does not say what a binary strip pays each period, "
                "nor whether a capped strip's cap is per period or in total"
            )
        if ir.resolution.early_expiration != "none":
            raise IRError("a per-period payout with early expiration: the definition does not say what the periods "
                          "after an early expiration pay")  # fmt: skip
        for p in payout.periods:
            _bounded(p, "a payment period")


def accept(ir: ContractIR) -> None:
    """The evaluator's type check of an IR (spec §5: the contract layer's validator is the "IR type check, three-valued
    evaluator"): IRError where the evaluator cannot give it one meaning, before anything is evaluated."""
    _check_ir(ir)


def consulted(ir: ContractIR) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Each observable's sources the procedure consults, highest-ranked first, and those that must agree (`_consulted`)."""
    return _consulted(ir)


def _consulted(ir: ContractIR) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Each observable's sources the procedure consults, in order, and those that must agree. Under require_agreement
    the named sources of the observable must agree (where it has two or more of them); otherwise the procedure's
    precedence orders its sources, and a source the precedence does not name is never consulted."""
    proc = ir.resolution
    consulted: dict[str, tuple[str, ...]] = {}
    agree: dict[str, tuple[str, ...]] = {}
    for o in ir.observables:
        own = {s.id for s in o.sources}
        named = [s for s in proc.conflict_policy.sources if s in own]
        if proc.conflict_policy.procedure == "require_agreement" and len(named) >= 2:
            agree[o.id] = tuple(named)
            consulted[o.id] = tuple(named)
        elif named:
            consulted[o.id] = tuple(named)
        else:
            consulted[o.id] = tuple(s for s in proc.precedence if s in own)
    return consulted, agree


def _strip_ids(value: Any) -> Any:
    """A node's content without the ids of it and its parts: two nodes with this equal mean the same thing."""
    if isinstance(value, dict):
        return {k: _strip_ids(v) for k, v in value.items() if k != "id"}
    if isinstance(value, list):
        return [_strip_ids(v) for v in value]
    return value


def _content(node: Any) -> str:
    return canonical_json(_strip_ids(node.model_dump(mode="json")))


# ---------------------------------------------------------------- the inputs, validated and indexed


class _Inputs:
    def __init__(self, ir: ContractIR, observations: Observations, interpretations: Mapping[str, Interpretation]):
        _check_ir(ir)
        self.ir = ir
        self.observables: dict[str, Observable] = {o.id: o for o in ir.observables}
        self.observations: tuple[Observation | SyntheticObservation, ...] = tuple(observations)
        kinds = {type(o) for o in self.observations}
        if not kinds <= {Observation, SyntheticObservation}:
            raise TypeError("observations are core.Observation records, or core.SyntheticObservation in a synthetic case")
        if len(kinds) > 1:
            raise TypeError("observed and synthetic observations are never mixed (spec I4)")
        self.synthetic = kinds == {SyntheticObservation}
        self._consult_order()
        self._index()
        self._interpretations(interpretations)

    def _consult_order(self) -> None:
        self.consulted, self.agree = _consulted(self.ir)

    def _index(self) -> None:
        self.records: dict[tuple[str, str], dict[datetime, list[_Vintage]]] = {}
        publications: set[datetime] = set()
        for r in self.observations:
            o = self.observables.get(r.observable)
            if o is None:
                raise ValueError(f"an observation of {r.observable!r}, which the contract does not declare")
            if r.source not in {s.id for s in o.sources}:
                raise ValueError(f"an observation of {o.id} from {r.source!r}, which is not one of its sources")
            if r.value.kind != VALUE_KIND[o.value_type]:
                raise ValueError(f"{o.id} is a {o.value_type}; an observation gives it a {r.value.kind}")
            if isinstance(r.value, Number) and o.precision is not None and r.value.value % o.precision != 0:
                raise ValueError(
                    f"{o.id} is published at precision {o.precision}; {r.value.value} (from {r.source}, for {r.at}) "
                    "is not stated at it, and is not rounded here"
                )
            if r.source not in self.consulted[o.id]:
                continue  # a source the procedure does not consult never settles
            v = _Vintage(_utc(r.published_at), r.value, r.final)
            slot = self.records.setdefault((o.id, r.source), {}).setdefault(_utc(r.at), [])
            same = [x for x in slot if x.published_at == v.published_at]
            if same and any(_value_key(x.value) != _value_key(v.value) for x in same):
                raise ValueError(
                    f"{r.source} published two values of {o.id} for {r.at} at one instant, {r.published_at}"
                )
            if same:
                if v.final and not same[0].final:
                    slot[slot.index(same[0])] = v
                continue
            slot.append(v)
            publications.add(v.published_at)
        self.publications = tuple(sorted(publications))

    def _interpretations(self, given: Mapping[str, Interpretation]) -> None:
        roots = [self.ir.trigger] + ([self.ir.payout.quantity] if self.ir.payout.quantity is not None else [])
        every = [n for r in roots for n in walk(r)]
        within = {m.id for e in every if isinstance(e, Exists) for m in walk(e.expr)}  # judged per member, not here
        nodes = {n.id: n for n in every if isinstance(n, Interpretive | ReadValue | Exists) and n.id not in within}
        missing, extra = sorted(set(nodes) - set(given)), sorted(set(given) - set(nodes))
        if missing:
            raise ValueError(
                f"no interpretation given for {missing}: give each its resolved truth or band, and Kleene.UNKNOWN or "
                "the whole range where its question is not established"
            )
        if extra:
            raise ValueError(f"interpretations given for {extra}, which are not interpretive conditions or readings here")
        self.interps: dict[str, Kleene] = {}
        self.readings: dict[str, tuple[Interval, bool]] = {}
        self.members: dict[str, tuple[dict[str, tuple[dict[str, Kleene], dict[str, tuple[Interval, bool]]]], Kleene]] = {}
        for nid, n in nodes.items():
            v = given[nid]
            if isinstance(n, Exists):
                if not isinstance(v, Members) or not isinstance(v.complete, Kleene):
                    raise TypeError(f"{nid} is an Exists: its value is its Members, not {v!r}")
                judged = {m.id: m for m in walk(n.expr) if isinstance(m, Interpretive | ReadValue)}
                per: dict[str, tuple[dict[str, Kleene], dict[str, tuple[Interval, bool]]]] = {}
                for member, given_for in v.members.items():
                    if set(given_for) != set(judged):
                        raise ValueError(f"{nid}'s member {member}: give each judged condition of its body "
                                         f"({sorted(judged)}), not {sorted(given_for)}")  # fmt: skip
                    truths, bands = {}, {}
                    for jid, j in judged.items():
                        x = given_for[jid]
                        if isinstance(j, Interpretive):
                            if not isinstance(x, Kleene):
                                raise TypeError(f"{nid}'s member {member}: {jid} is interpretive, its value a Kleene truth")
                            truths[jid] = x
                        else:
                            bands[jid] = _band(j, x)
                    per[member] = (truths, bands)
                self.members[nid] = (per, v.complete)
                continue
            if isinstance(n, Interpretive):
                if not isinstance(v, Kleene):
                    raise TypeError(f"{nid} is interpretive: its value is a Kleene truth, not {v!r}")
                self.interps[nid] = v
            else:
                assert isinstance(n, ReadValue)
                self.readings[nid] = _band(n, v)
        self.given = dict(given)
        self.has_interpretation = bool(nodes)

    def state_hash(self, at: datetime) -> str:
        """The evaluation's inputs: its observations, its interpretations and its instant."""

        def show(v: Any) -> Any:
            if isinstance(v, Members):
                return {"complete": v.complete.value,
                        "members": {m: {k: show(x) for k, x in sorted(g.items())} for m, g in sorted(v.members.items())}}  # fmt: skip
            return v.value if isinstance(v, Kleene) else f"[{v[0]}, {v[1]}]"

        return sha256(
            canonical_json(
                {
                    "namespace": "synthetic" if self.synthetic else "observed",
                    "observations": sorted(o.content_hash() for o in self.observations),
                    "interpretations": {k: show(v) for k, v in sorted(self.given.items())},
                    "at": _iso(at),
                }
            )
        )

    @cached_property
    def has_hidden(self) -> bool:
        return any(isinstance(n, Interpretive | ReadValue | Exists) for n in walk(self.ir.trigger))


def _band(n: ReadValue, v: object) -> tuple[Interval, bool]:
    """A reading's value as an interval: one band, a run of bands, or the whole range (its question not established).
    Bands are [low, high), the last [low, high]. Returns the interval and whether it is the whole range."""
    if not (isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, Decimal) for x in v)):
        raise TypeError(f"{n.id} is a reading: its value is a band (low, high) of Decimals, not {v!r}")
    lo, hi = v
    bands = list(n.bands)
    last = len(bands) - 1
    whole = (lo, hi) == (bands[0][0], bands[-1][1])
    exact = [i for i, b in enumerate(bands) if (b[0], b[1]) == (lo, hi)]
    if exact:
        i = exact[0]
        return Interval(Fraction(lo), Fraction(hi), True, i == last), whole
    starts = [i for i, b in enumerate(bands) if b[0] == lo]
    ends = [i for i, b in enumerate(bands) if b[1] == hi]
    if not starts or not ends or max(ends) < starts[0]:
        raise ValueError(f"{n.id}: ({lo}, {hi}) is not a band of its answer set, a run of them, or its whole range")
    return Interval(Fraction(lo), Fraction(hi), True, max(ends) == last), whole


def _value_key(v: CmpValue) -> tuple[str, object]:
    if isinstance(v, Number):
        return ("number", Fraction(v.value))
    return (v.kind, v.value)


def _show_value(v: CmpValue) -> str:
    return str(v.value)


# ---------------------------------------------------------------- the procedure's instants


@dataclass(frozen=True)
class _Clock:
    expiration: datetime
    cutoff: datetime
    extended: bool = False

    @property
    def determination(self) -> datetime:
        """When the unknown-at-expiry rule can apply: expiration, once the contract's observation cutoff has passed too
        (values the contract considers may be published until its cutoff)."""
        return max(self.expiration, self.cutoff)


# ---------------------------------------------------------------- one information state


class _View:
    """The IR's values on the information available at `tau`: records published by tau and by the observation cutoff,
    and the interpretations as given (or, with interpretations=False, none of them yet)."""

    def __init__(
        self,
        inp: _Inputs,
        tau: datetime,
        clock: _Clock,
        *,
        interpretations: bool = True,
        restrict: TimeWindow | None = None,
    ):
        self.inp, self.tau, self.clock = inp, _utc(tau), clock
        self.limit = min(self.tau, _utc(clock.cutoff))
        self.closed = self.tau >= _utc(clock.cutoff)
        self.interpretations = interpretations
        self.restrict = restrict
        self.member: tuple[dict[str, Kleene], dict[str, tuple[Interval, bool]]] | None = None  # within an Exists' body
        self._instants: dict[str, tuple[datetime, ...]] = {}

    # ---- values

    def _visible(self, o: str, s: str, t: datetime) -> list[_Vintage]:
        return [v for v in self.inp.records.get((o, s), {}).get(t, ()) if v.published_at <= self.limit]

    def pick(self, o: str, s: str, t: datetime) -> _Pick:
        """One source's value of `o` for `t`, by the observable's revision policy."""
        vs = self._visible(o, s, t)
        if not vs:
            return _ABSENT
        obs = self.inp.observables[o]
        policy: str = obs.revision_policy
        if policy == "final":
            by = _utc(obs.final_deadline) if obs.final_deadline is not None else _utc(self.clock.determination)
            finals = [v for v in vs if v.final and v.published_at <= by]
            if finals:
                return _Pick("value", max(finals, key=lambda v: v.published_at).value)
            if self.tau < min(by, _utc(self.clock.cutoff)):
                return _Pick("unknown", why=f"not final yet (a final value counts if published by {by.isoformat()})")
            if obs.final_fallback is None:
                return _Pick("unknown", why=f"no final value by {by.isoformat()}, and {o} declares no fallback")
            policy = obs.final_fallback
        if policy == "first_print":
            return _Pick("value", min(vs, key=lambda v: v.published_at).value)
        in_time = [v for v in vs if v.published_at <= _utc(self.clock.expiration)]
        if not in_time:
            return _Pick("unknown", why=f"published only after expiration, {_iso(self.clock.expiration)}")
        return _Pick("value", max(in_time, key=lambda v: v.published_at).value)

    def value(self, o: str, t: datetime) -> _Pick:
        """The value of `o` for `t`, by the procedure's conflict policy."""
        agree = self.inp.agree.get(o)
        if agree:
            picks = [(s, self.pick(o, s, t)) for s in agree]
            known = {s: p.value for s, p in picks if p.kind == "value" and p.value is not None}
            if len({_value_key(v) for v in known.values()}) > 1:
                shown = ", ".join(f"{s} gives {_show_value(v)}" for s, v in known.items())
                return _Pick("disputed", why=f"the sources that must agree disagree: {shown}")
            if all(p.kind == "value" for _, p in picks):
                return picks[0][1]
            if all(p.kind == "absent" for _, p in picks):
                return _ABSENT
            waiting = [f"{s} ({p.why or 'none published'})" for s, p in picks if p.kind != "value"]
            return _Pick("unknown", why=f"agreement of {', '.join(agree)} not established: {'; '.join(waiting)}")
        order = self.inp.consulted[o]
        for k, s in enumerate(order):
            p = self.pick(o, s, t)
            if p.kind != "absent":
                if k and not self.closed:
                    # The highest-ranked source that has published is decided at the observation cutoff: until then a
                    # source ranked above this one may still publish, and would settle instead.
                    return _Pick("unknown", why=f"{s} has published, but {', '.join(order[:k])}, ranked above it, may "
                                                f"still publish until the observation cutoff, {_iso(self.clock.cutoff)}")  # fmt: skip
                return p
        return _ABSENT

    def instants(self, o: str) -> tuple[datetime, ...]:
        """The instants for which a consulted source has published a value of `o`, visible here."""
        if o not in self._instants:
            seen = {
                t
                for s in self.inp.consulted[o]
                for t, vs in self.inp.records.get((o, s), {}).items()
                if any(v.published_at <= self.limit for v in vs)
            }
            self._instants[o] = tuple(sorted(seen))
        return self._instants[o]

    def level_instants(self, e: Any) -> tuple[datetime, ...]:
        return tuple(sorted({t for o in _level(e) for t in self.instants(o)}))

    # ---- truth

    def truth(self, e: Any, at: datetime | None) -> _T:
        if isinstance(e, Cmp):
            assert at is not None  # _check_ir
            return self._cmp(e, at)
        if isinstance(e, And | Or):
            parts = [self.truth(a, at) for a in e.args]
            v = (conj if isinstance(e, And) else disj)(p.value for p in parts)
            return _T(v, _union(p.why for p in parts if p.value is Kleene.UNKNOWN) if v is Kleene.UNKNOWN else ())
        if isinstance(e, Not):
            inner = self.truth(e.arg, at)
            return _T(neg(inner.value), inner.why)
        if isinstance(e, Interpretive):
            if not self.interpretations:
                return _T(Kleene.UNKNOWN, (_Why(f"{e.id}: its interpretation is given only as of the evaluation"),))
            v = self.member[0][e.id] if self.member is not None else self.inp.interps[e.id]
            why = (_Why(f"{e.id} ({e.question_template}): not established or contested"),)
            return _T(v, why if v is Kleene.UNKNOWN else ())
        if isinstance(e, Exists):
            return self._exists(e, at)
        if isinstance(e, NumCmp):
            n = self.number(e.expr, at)
            v = n.interval.compare(e.op, Fraction(e.value))
            if v is not Kleene.UNKNOWN:
                return _T(v)
            return _T(v, (_Why(f"{e.id}: {e.expr.id} lies in {n.interval}, which {e.op} {e.value} does not decide"),
                          *n.why))  # fmt: skip
        if isinstance(e, At):
            return self.truth(e.expr, _utc(e.at))
        if isinstance(e, By):
            insts = [t for t in self.level_instants(e.expr) if _in(t, e.since, e.since_closed, e.deadline, e.closed)]
            return self._some(e.id, [self.truth(e.expr, t) for t in insts], f"by {_iso(e.deadline)}")
        if isinstance(e, Throughout):
            w = e.window
            insts = [t for t in self.level_instants(e.expr) if _in(t, w.start, w.start_closed, w.end, w.end_closed)]
            return self._every(e.id, [self.truth(e.expr, t) for t in insts])
        raise IRError(f"{e!r} is not a trigger condition")

    def _exists(self, e: Exists, at: datetime | None) -> _T:
        """Kleene over the members (core.Exists): TRUE where some member's body is TRUE; FALSE where the list is complete,
        the cutoff has passed and every member's body is FALSE; UNKNOWN otherwise, with why."""
        if not self.interpretations:
            return _T(Kleene.UNKNOWN, (_Why(f"{e.id}: its members are given only as of the evaluation"),))
        per, complete = self.inp.members[e.id]
        outer = self.member
        vals: list[tuple[str, _T]] = []
        try:
            for member in sorted(per):
                self.member = per[member]
                vals.append((member, self.truth(e.expr, at)))
        finally:
            self.member = outer
        if any(v.value is Kleene.TRUE for _, v in vals):
            return _T(Kleene.TRUE)
        unknown = _union(v.why for _, v in vals if v.value is Kleene.UNKNOWN)
        if unknown:
            return _T(Kleene.UNKNOWN, unknown)
        if complete is not Kleene.TRUE:
            named = ", ".join(src.name for src in sorted(e.sources, key=lambda src: src.rank))
            return _T(Kleene.UNKNOWN, (_Why(f"{e.id}: no documented {e.member} satisfies it, and {named} not all read "
                                            "through the observation cutoff, so the documented ones may not be all"),))  # fmt: skip
        if not self.closed:
            return _T(Kleene.UNKNOWN, (_Why(f"{e.id}: no {e.member} documented yet satisfies it; more may be documented "
                                            f"until {_iso(self.clock.cutoff)}"),))  # fmt: skip
        return _T(Kleene.FALSE)

    def _some(self, nid: str, vals: list[_T], where: str) -> _T:
        if any(v.value is Kleene.TRUE for v in vals):
            return _T(Kleene.TRUE)
        unknown = _union(v.why for v in vals if v.value is Kleene.UNKNOWN)
        if not self.closed:
            return _T(Kleene.UNKNOWN, (_Why(f"{nid}: no value published yet shows it {where}; more may be published "
                                            f"until the observation cutoff, {_iso(self.clock.cutoff)}"), *unknown))  # fmt: skip
        if not vals:
            return _T(Kleene.UNKNOWN, (_Why(f"{nid}: no value of its observables was published for any instant {where}"),))
        return _T(disj(v.value for v in vals), unknown)

    def _every(self, nid: str, vals: list[_T]) -> _T:
        if any(v.value is Kleene.FALSE for v in vals):
            return _T(Kleene.FALSE)
        unknown = _union(v.why for v in vals if v.value is Kleene.UNKNOWN)
        if not self.closed:
            return _T(Kleene.UNKNOWN, (_Why(f"{nid}: holds at every value published so far; more may be published "
                                            f"until the observation cutoff, {_iso(self.clock.cutoff)}"), *unknown))  # fmt: skip
        if not vals:
            return _T(Kleene.UNKNOWN, (_Why(f"{nid}: no value of its observables was published for any instant in its window"),))
        return _T(conj(v.value for v in vals), unknown)

    def _cmp(self, e: Cmp, at: datetime) -> _T:
        p = self.value(e.observable, at)
        if p.kind == "value":
            assert p.value is not None
            return _T(_compare(p.value, e.op, e.value))
        if p.kind == "absent":
            why = f"{e.observable}: no value published for {at.isoformat()} by {self.limit.isoformat()}"
        else:
            why = f"{e.observable} for {at.isoformat()}: {p.why}"
        return _T(Kleene.UNKNOWN, (_Why(why, disputed=p.kind == "disputed"),))

    # ---- numbers

    def number(self, e: Any, at: datetime | None) -> _N:
        if isinstance(e, ValueAt):
            p = self.value(e.observable, _utc(e.at))
            if p.kind == "value":
                assert isinstance(p.value, Number)
                return _N(Interval.point(Fraction(p.value.value)))
            why = p.why if p.kind != "absent" else f"no value published by {self.limit.isoformat()}"
            return _N(Interval.everything(), (_Why(f"{e.id}: {e.observable} for {_iso(e.at)}: {why}", p.kind == "disputed"),))
        if isinstance(e, ReadValue):
            if not self.interpretations:
                interval = _band(e, (e.bands[0][0], e.bands[-1][1]))[0]
                return _N(interval, (_Why(f"{e.id}: its reading is given only as of the evaluation"),))
            interval, whole = self.member[1][e.id] if self.member is not None else self.inp.readings[e.id]
            return _N(interval, (_Why(f"{e.id} ({e.question_template}): not established; its whole range"),) if whole else ())
        if isinstance(e, CountTrue):
            return self._count(e)
        if isinstance(e, DurationIn):
            return self._duration(e)
        if isinstance(e, Computed):
            return self._computed(e, at)
        raise IRError(f"{e!r} is not a number")

    def _count(self, e: CountTrue) -> _N:
        insts = self.level_instants(e.expr)
        values: list[Kleene] = []
        why: list[_Why] = []
        unknown_days: list[str] = []
        for d, ds, de in _days(e.days, self.restrict):
            day = [self.truth(e.expr, t) for t in insts if ds <= t < de]
            if any(v.value is Kleene.TRUE for v in day):
                values.append(Kleene.TRUE)
            elif not day:
                values.append(Kleene.UNKNOWN)
                unknown_days.append(f"{d.isoformat()} (no value published)")
            elif not self.closed:
                # Another value for the day may still be published, before the observation cutoff, and show it.
                values.append(Kleene.UNKNOWN)
                unknown_days.append(f"{d.isoformat()} (not shown so far; values may be published until the observation "
                                    f"cutoff, {_iso(self.clock.cutoff)})")  # fmt: skip
                why.extend(w for x in day if x.value is Kleene.UNKNOWN for w in x.why)
            else:
                v = disj(x.value for x in day)
                values.append(v)
                if v is Kleene.UNKNOWN:
                    unknown_days.append(d.isoformat())
                    why.extend(w for x in day if x.value is Kleene.UNKNOWN for w in x.why)
        bounds = count_true(plain(v) for v in values)
        interval = Interval(Fraction(bounds.low), Fraction(bounds.high))
        if not unknown_days:
            return _N(interval)
        head = _Why(f"{e.id}: {bounds.unknown} of {len(values)} days unknown, so the count is in [{bounds.low}, "
                    f"{bounds.high}]: {', '.join(unknown_days)}")  # fmt: skip
        return _N(interval, _union([(head,), tuple(why)]))

    def _duration(self, e: DurationIn) -> _N:
        w = e.window
        assert w.start is not None and w.end is not None
        start, end = _utc(w.start), _utc(w.end)
        if self.restrict is not None:
            r = self.restrict
            assert r.start is not None and r.end is not None
            start, end = max(start, _utc(r.start)), min(end, _utc(r.end))
        if end <= start:
            return _N(Interval.point(Fraction(0)))
        if not self.closed:
            # A value for any instant may still be published, before the observation cutoff, and split any stretch.
            total = _measure(start, end, e.unit, w.tz)
            return _N(Interval(Fraction(0), total), (_Why(f"{e.id}: values may be published until the observation "
                                                         f"cutoff, {_iso(self.clock.cutoff)}, for any instant of its "
                                                         "window, so any stretch may still change"),))  # fmt: skip
        insts = self.level_instants(e.expr)
        prior = [t for t in insts if t <= start]
        inside = [t for t in insts if start < t < end]
        head = self.truth(e.expr, prior[-1]) if prior else _T(
            Kleene.UNKNOWN, (_Why(f"{e.id}: no value at or before the window's start, {start.isoformat()}"),))
        points = [start, *inside, end]
        values = [head, *(self.truth(e.expr, t) for t in inside)]
        low = high = Fraction(0)
        why: list[_Why] = []
        for i, v in enumerate(values):
            a, b = points[i], points[i + 1]
            length = _measure(a, b, e.unit, w.tz)
            value = v.value
            if value is Kleene.TRUE:
                low += length
                high += length
            elif value is Kleene.UNKNOWN:
                high += length
                why.extend(v.why)
        return _N(Interval(low, high), _union([tuple(why)]) if low != high else ())

    def _computed(self, e: Computed, at: datetime | None) -> _N:
        node = registry.get(e.node)
        args = {name: self.number(sub, at) for name, sub in e.inputs.items()}
        why = _union(a.why for a in args.values())
        if all(a.interval.is_point for a in args.values()):
            out = _exact_number(node.fn(**{k: a.interval.lo for k, a in args.items()}), f"node {node.id}'s result")
            return _N(Interval.point(out))
        open_ = {k: a.interval for k, a in args.items() if not a.interval.is_point}
        not_monotone = sorted(k for k in open_ if k not in node.monotone)
        unbounded = sorted(k for k, iv in open_.items() if iv.lo is None or iv.hi is None)
        if not_monotone or unbounded:
            reason = (f"its node is not declared monotone in {not_monotone}" if not_monotone
                      else f"inputs {unbounded} are unbounded")  # fmt: skip
            return _N(Interval.everything(), (_Why(f"{e.id}: unknown, since {reason}"), *why))
        low, high = {}, {}
        for k, a in args.items():
            iv = a.interval
            if k in open_:
                up = node.monotone[k] == "increasing"
                low[k], high[k] = (iv.lo, iv.hi) if up else (iv.hi, iv.lo)
            else:
                low[k] = high[k] = iv.lo
        lo = _exact_number(node.fn(**low), f"node {node.id}'s result")
        hi = _exact_number(node.fn(**high), f"node {node.id}'s result")
        if lo > hi:
            raise ValueError(f"node {node.id} v{node.version} is not monotone as it declares: {lo} > {hi} at its corners")
        return _N(Interval(lo, hi), why)


def _compare(observed: CmpValue, op: str, threshold: CmpValue) -> Kleene:
    a, b = _value_key(observed)[1], _value_key(threshold)[1]
    if op == "==":
        return Kleene.TRUE if a == b else Kleene.FALSE
    if op == "!=":
        return Kleene.TRUE if a != b else Kleene.FALSE
    if isinstance(observed, Number) and isinstance(threshold, Number):
        result = _holds(Fraction(observed.value), op, Fraction(threshold.value))
    elif isinstance(observed, OnDay) and isinstance(threshold, OnDay):
        result = _holds(Fraction(observed.value.toordinal()), op, Fraction(threshold.value.toordinal()))
    else:
        raise IRError(f"{op} does not order {observed.kind} values")
    return Kleene.TRUE if result else Kleene.FALSE


def _days(w: TimeWindow, restrict: TimeWindow | None = None) -> list[tuple[date, datetime, datetime]]:
    """The calendar days of the window's zone wholly inside the window (and inside a payment period, where one
    restricts the count), each with its start and end instants."""
    windows = [w] + ([restrict] if restrict is not None else [])
    tz = ZoneInfo(w.tz)
    assert w.start is not None and w.end is not None
    out = []
    d = w.start.astimezone(tz).date()
    last = w.end.astimezone(tz).date()
    while d <= last:
        ds, de = _midnight(d, tz), _midnight(d + timedelta(days=1), tz)
        if all(_holds_day(ds, de, x) for x in windows):
            out.append((d, ds, de))
        d += timedelta(days=1)
    return out


def _in(t: datetime, lo: datetime | None, lo_closed: bool | None, hi: datetime | None, hi_closed: bool | None) -> bool:
    if lo is not None:
        lo = _utc(lo)
        if t < lo or (t == lo and not lo_closed):
            return False
    if hi is not None:
        hi = _utc(hi)
        if t > hi or (t == hi and not hi_closed):
            return False
    return True


def _midnight(d: date, tz: ZoneInfo) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=tz).astimezone(UTC)


def _holds_day(ds: datetime, de: datetime, w: TimeWindow) -> bool:
    """Whether the day [ds, de) lies wholly inside the window."""
    assert w.start is not None and w.end is not None
    s, e = _utc(w.start), _utc(w.end)
    return (ds > s or (ds == s and w.start_closed)) and de <= e


_MICROS = timedelta(microseconds=1)


def _measure(a: datetime, b: datetime, unit: str, tz: str) -> Fraction:
    """The length of [a, b) in the unit: hours, 24-hour periods, or calendar days of the zone (each day one unit,
    whatever its hours)."""
    if unit == "hour":
        return Fraction((b - a) // _MICROS, 3_600_000_000)
    if unit == "elapsed_day":
        return Fraction((b - a) // _MICROS, 86_400_000_000)
    zone = ZoneInfo(tz)
    total = Fraction(0)
    d = a.astimezone(zone).date()
    while True:
        ds, de = _midnight(d, zone), _midnight(d + timedelta(days=1), zone)
        if ds >= b:
            break
        overlap = min(b, de) - max(a, ds)
        if overlap > timedelta(0):
            total += Fraction(overlap // _MICROS, (de - ds) // _MICROS)
        d += timedelta(days=1)
    return total


# ---------------------------------------------------------------- the procedure


@dataclass
class _Outcome:
    truth: Kleene
    status: ProcedureStatus
    quantity: tuple[Decimal, Decimal] | None = None
    transfers: list[CashTransfer] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    policies: list[Ref] = field(default_factory=list)


@dataclass(frozen=True)
class _Early:
    lower: datetime  # the first instant at which the trigger is true if the interpretations held then
    upper: datetime  # the first instant at which it is true on what the inputs establish then
    view: _View

    def fired(self, expiration: datetime) -> bool:
        exp = _utc(expiration)
        return self.upper < exp or self.lower < exp


def evaluate(
    ir: ContractIR, observations: Observations, interpretations: Mapping[str, Interpretation], *, at: datetime
) -> ResolutionEvaluation:
    """What the contract settles to, on these observations and interpretations, at the instant `at`.

    observations: every published value the evaluation may read (all observed, or all synthetic: I4).
    interpretations: for each Interpretive node, the Kleene truth of its question's resolved judgment as of `at`
        (core.kleene.from_status); for each ReadValue node, its resolved band as (low, high), a run of bands, or the
        whole range where its question is not established. Every such node of the trigger and the payout is given.
    at: the instant of evaluation.
    """
    inp = _Inputs(ir, observations, interpretations)
    at = _utc(at)
    out = _procedure(inp, at)
    out.policies = list(dict.fromkeys([EVALUATOR, *_value_policies(ir), *out.policies]))
    return ResolutionEvaluation(
        contract=ir.ref,
        evidence_state_hash=inp.state_hash(at),
        trigger_truth=out.truth,
        procedure_status=out.status,
        quantity=out.quantity,
        transfers=out.transfers,
        unresolved=list(dict.fromkeys(out.unresolved)),
        policies=out.policies,
    )


def final_instant(ir: ContractIR) -> datetime:
    """The instant from which an evaluation's procedure status, on the same observations and interpretations, no longer
    changes with time: the later of the expiration and the observation cutoff (the unknown-at-expiry rule applies once
    both have passed), or, where the contract extends on an unknown trigger, the later of the extended pair."""
    t, proc = ir.timing, ir.resolution
    if proc.on_unknown_at_expiry == "extend":
        assert proc.extension_days is not None  # core.ResolutionProcedure
        return max(_shift(t.expiration, proc.extension_days, ir), _shift(t.observation_cutoff, proc.extension_days, ir))
    return max(_utc(t.expiration), _utc(t.observation_cutoff))


def _clause(ir: ContractIR, path: str) -> Ref:
    """A reference to the contract clause that states a policy (A05 §5.1 policy_refs)."""
    return Ref(id=f"{ir.id}/{path}", version=ir.version)


def _value_policies(ir: ContractIR) -> list[Ref]:
    roots = [ir.trigger] + ([ir.payout.quantity] if ir.payout.quantity is not None else [])
    read = dict.fromkeys(n.observable for r in roots for n in walk(r) if isinstance(n, Cmp | ValueAt))
    if not read:
        return []
    kinds = {o.id: o for o in ir.observables}
    refs = []
    for o in read:
        obs = kinds[o]
        refs.append(_clause(ir, f"observables/{o}/revision_policy/{obs.revision_policy}"))
        if obs.revision_policy == "final" and obs.final_fallback is not None:
            refs.append(_clause(ir, f"observables/{o}/final_fallback/{obs.final_fallback}"))
    refs.append(_clause(ir, f"resolution/conflict_policy/{ir.resolution.conflict_policy.procedure}"))
    return refs


def _procedure(inp: _Inputs, at: datetime) -> _Outcome:
    ir = inp.ir
    proc, timing = ir.resolution, ir.timing
    clock = _Clock(_utc(timing.expiration), _utc(timing.observation_cutoff))
    if proc.early_expiration == "when_true":
        early = _first_true(inp, clock, after=None, until=min(at, clock.expiration))
        if early is not None and early.fired(clock.expiration):
            return _early(inp, early)
    view = _View(inp, at, clock)
    t = view.truth(ir.trigger, None)
    if at < clock.expiration:
        return _pending(inp, view, t, "pending", f"the procedure resolves at expiration, {clock.expiration.isoformat()}")
    if t.value is not Kleene.UNKNOWN:
        return _resolved(inp, view, t, clock.expiration)
    if at < clock.determination:
        return _pending(inp, view, t, "pending", f"values the contract considers may be published until its "
                                                 f"observation cutoff, {clock.cutoff.isoformat()}")  # fmt: skip
    return _unknown_at_expiry(inp, clock, view, t, at)


def _first_true(inp: _Inputs, clock: _Clock, *, after: datetime | None, until: datetime) -> _Early | None:
    """The first instant, after `after` and by `until`, at which the trigger is true on the information then
    available. Information changes when a record is published, when the observation cutoff closes the instants, and when
    a "final" policy's deadline passes; the interpretations are given as of the evaluation, so before it the trigger is evaluated both without them (an upper
    bound on the first instant: true whatever they were then) and with them (a lower bound: true if they held then)."""
    cands = {p for p in inp.publications if p <= until and p <= clock.cutoff}
    cands |= {clock.cutoff} if clock.cutoff <= until else set()
    # A "final" policy's value changes, with no publication, when its deadline passes and its fallback applies.
    for o in inp.ir.observables:
        if o.revision_policy == "final":
            by = _utc(o.final_deadline) if o.final_deadline is not None else clock.determination
            if min(by, clock.cutoff) <= until:
                cands.add(min(by, clock.cutoff))
    cands.add(until)
    lower: datetime | None = None
    for tau in sorted(c for c in cands if after is None or c > after):
        given = _View(inp, tau, clock)
        v_given = given.truth(inp.ir.trigger, None).value
        if lower is None and v_given is Kleene.TRUE:
            lower = tau
        if tau == until or not inp.has_hidden:
            v_known = v_given
        else:
            v_known = _View(inp, tau, clock, interpretations=False).truth(inp.ir.trigger, None).value
        if v_known is Kleene.TRUE:
            assert lower is not None
            return _Early(lower, tau, given)
    return None


def _early(inp: _Inputs, early: _Early) -> _Outcome:
    ir, proc = inp.ir, inp.ir.resolution
    timing_kind = proc.early_expiration_at
    out = _Outcome(Kleene.TRUE, "resolved_yes", policies=[_clause(ir, f"resolution/early_expiration/when_true/{timing_kind}")])
    if timing_kind == "first_true" and early.lower == early.upper:
        _payout(inp, early.view, early.upper, "", out)
        return out
    if timing_kind == "per_exchange_rule":
        when = (f"the early expiration, whose date {proc.early_expiration_rule} sets (the trigger is true on the "
                f"information of {early.upper.isoformat()})")  # fmt: skip
    else:
        hidden = sorted(n.id for n in walk(ir.trigger) if isinstance(n, Interpretive | ReadValue | Exists))
        when = (f"the early expiration, at the instant the trigger was first true: between {early.lower.isoformat()} "
                f"and {early.upper.isoformat()}, since the instants at which {hidden} were established are not among "
                "the inputs")  # fmt: skip
    out.unresolved.append(f"expiration date: {when}")
    _payout(inp, early.view, None, when, out)
    return out


def _pending(inp: _Inputs, view: _View, t: _T, status: ProcedureStatus, reason: str) -> _Outcome:
    out = _Outcome(t.value, status, unresolved=[reason, *(w.text for w in t.why)])
    out.quantity = _quantity(inp, view)[0]
    return out


def _resolved(inp: _Inputs, view: _View, t: _T, resolution_at: datetime) -> _Outcome:
    if t.value is Kleene.TRUE:
        out = _Outcome(t.value, "resolved_yes")
        _payout(inp, view, resolution_at, "", out)
        return out
    out = _Outcome(t.value, "resolved_no")
    out.quantity = _quantity(inp, view)[0]
    return out


def _unknown_at_expiry(inp: _Inputs, clock: _Clock, view: _View, t: _T, at: datetime) -> _Outcome:
    ir, proc = inp.ir, inp.ir.resolution
    if any(w.disputed for w in t.why):
        out = _Outcome(t.value, "disputed", unresolved=[w.text for w in t.why])
        out.policies.append(_clause(ir, f"resolution/conflict_policy/{proc.conflict_policy.procedure}"))
        out.quantity = _quantity(inp, view)[0]
        return out
    rule = proc.on_unknown_at_expiry
    policy = _clause(ir, f"resolution/on_unknown_at_expiry/{rule}")
    if rule != "extend":
        out = _apply_rule(inp, view, t, rule, clock.expiration)
        out.policies.append(policy)
        return out
    assert proc.extension_days is not None and proc.after_extension is not None
    extended = _Clock(_shift(clock.expiration, proc.extension_days, ir), _shift(clock.cutoff, proc.extension_days, ir), True)
    policies = [policy, _clause(ir, f"resolution/extension_days/{proc.extension_days}")]
    if proc.early_expiration == "when_true":
        early = _first_true(inp, extended, after=clock.determination, until=min(at, extended.expiration))
        if early is not None and early.fired(extended.expiration):
            out = _early(inp, early)
            out.policies[:0] = policies
            return out
    view2 = _View(inp, at, extended)
    t2 = view2.truth(ir.trigger, None)
    if at < extended.expiration:
        out = _pending(inp, view2, t2, "extended", f"extended to {extended.expiration.isoformat()}, its rule for an "
                                                   "unknown trigger at expiry")  # fmt: skip
    elif t2.value is not Kleene.UNKNOWN:
        out = _resolved(inp, view2, t2, extended.expiration)
    elif at < extended.determination:
        out = _pending(inp, view2, t2, "extended", "values the contract considers may be published until its extended "
                                                   f"observation cutoff, {extended.cutoff.isoformat()}")  # fmt: skip
    elif any(w.disputed for w in t2.why):
        out = _Outcome(t2.value, "disputed", unresolved=[w.text for w in t2.why])
        out.quantity = _quantity(inp, view2)[0]
        policies.append(_clause(ir, f"resolution/conflict_policy/{proc.conflict_policy.procedure}"))
    else:
        out = _apply_rule(inp, view2, t2, proc.after_extension, extended.expiration)
        policies.append(_clause(ir, f"resolution/after_extension/{proc.after_extension}"))
    out.policies[:0] = policies
    return out


def _apply_rule(inp: _Inputs, view: _View, t: _T, rule: str, resolution_at: datetime) -> _Outcome:
    """An unknown trigger settles by the contract's rule; its truth stays unknown (A05 §5.2)."""
    ir = inp.ir
    if rule == "no":
        out = _Outcome(t.value, "resolved_no", unresolved=[w.text for w in t.why])
        out.quantity = _quantity(inp, view)[0]
        return out
    if rule == "yes":
        out = _Outcome(t.value, "resolved_yes", unresolved=[w.text for w in t.why])
        _payout(inp, view, resolution_at, "", out)
        return out
    out = _Outcome(t.value, "void", unresolved=[w.text for w in t.why])
    void = ir.resolution.void_transfer
    out.policies.append(_clause(ir, f"resolution/void_transfer/{void.returns}"))
    out.quantity = _quantity(inp, view)[0]
    paid = _pay_date(resolution_at, ir)
    if void.returns == "fixed_amount":
        assert void.amount is not None
        out.transfers.append(CashTransfer(at=paid, amount=void.amount, currency=ir.payout.currency, basis="void refund"))
        out.policies.append(_timing_policy(ir))
    elif void.returns == "premium":
        out.unresolved.append(f"void refund: the premium paid per unit is returned at {paid.isoformat()}; the premium "
                              "is the holder's acquisition price, which a contract evaluation is not given")  # fmt: skip
        out.policies.append(_timing_policy(ir))
    return out


# ---------------------------------------------------------------- payout and transfers


def _timing_policy(ir: ContractIR) -> Ref:
    t = ir.timing
    days = f"business/{t.holiday_calendar}" if t.lag_days == "business" else "calendar"
    return _clause(ir, f"timing/lags/{days}/{t.settlement_lag_days}+{t.payment_lag_days}/{t.tz}")


def _shift(t: datetime, days: int, ir: ContractIR) -> datetime:
    """`days` days after the instant in the contract's day count, at the same wall-clock time of its zone."""
    zone = ZoneInfo(ir.timing.tz)
    local = t.astimezone(zone)
    d = calendar.add_days(local.date(), days, ir.timing.lag_days, ir.timing.holiday_calendar)
    return datetime.combine(d, local.time(), tzinfo=zone).astimezone(UTC)


def _pay_date(resolution_at: datetime, ir: ContractIR) -> datetime:
    """Paid at the resolution instant plus the settlement lag, then the payment lag (A05 §5.3: the settlement
    determination and payment dates are distinct)."""
    determined = _shift(resolution_at, ir.timing.settlement_lag_days, ir)
    return _shift(determined, ir.timing.payment_lag_days, ir).astimezone(ZoneInfo(ir.timing.tz))


def _round(x: Fraction, quantum: Decimal, mode: str) -> Decimal:
    q = Fraction(quantum)
    n = x / q
    floor = n.numerator // n.denominator
    rest = n - floor
    if mode == "down":
        k = floor if n >= 0 else -((-n.numerator) // n.denominator)
    elif mode == "up":
        k = -((-n.numerator) // n.denominator) if n >= 0 else floor
    elif rest > Fraction(1, 2):
        k = floor + 1
    elif rest < Fraction(1, 2):
        k = floor
    elif mode == "half_even":
        k = floor if floor % 2 == 0 else floor + 1
    else:  # half_up: ties away from zero
        k = floor + 1 if n >= 0 else floor
    with localcontext(_EXACT):
        return Decimal(k) * quantum


def _amount(ir: ContractIR, q: Interval | None) -> tuple[Decimal, Decimal] | None:
    """The per-unit amount (rounded) over the quantity's interval, low and high; None where it is unbounded."""
    p = ir.payout
    u = Fraction(p.unit_value)
    if p.kind == "binary":
        a = _round(u, p.rounding_quantum, p.rounding_mode)
        return (a, a)
    assert q is not None
    if q.lo is None or q.hi is None:
        return None

    def f(x: Fraction) -> Decimal:
        v = u * x
        if p.kind == "capped_linear":
            assert p.cap is not None
            v = min(v, Fraction(p.cap))
        return _round(v, p.rounding_quantum, p.rounding_mode)

    ends = sorted([f(q.lo), f(q.hi)])
    return (ends[0], ends[1])


def _quantity(inp: _Inputs, view: _View) -> tuple[tuple[Decimal, Decimal] | None, _N | None]:
    """The payout quantity on this view, as the definition's (low, high), rounded outward where it does not terminate;
    None for a binary payout or an unbounded quantity."""
    ir = inp.ir
    if ir.payout.quantity is None:
        return None, None
    if ir.payout.payment == "per_period":
        parts = [_period(inp, view, p).number(ir.payout.quantity, None) for p in ir.payout.periods]
        lo = sum((n.interval.lo for n in parts if n.interval.lo is not None), Fraction(0))
        hi = sum((n.interval.hi for n in parts if n.interval.hi is not None), Fraction(0))
        bounded = all(n.interval.lo is not None and n.interval.hi is not None for n in parts)
        n = _N(Interval(lo, hi) if bounded else Interval.everything(), _union(x.why for x in parts))
    else:
        n = view.number(ir.payout.quantity, None)
    if n.interval.lo is None or n.interval.hi is None:
        return None, n
    return (_to_decimal(n.interval.lo, ROUND_FLOOR), _to_decimal(n.interval.hi, ROUND_CEILING)), n


def _payout(inp: _Inputs, view: _View, resolution_at: datetime | None, when: str, out: _Outcome) -> None:
    """The YES side's transfers: the payout by kind on its quantity, dated from the resolution instant (or each period's
    end), or what it is still unresolved on."""
    ir = inp.ir
    p = ir.payout
    out.quantity, n = _quantity(inp, view)
    rounding = _clause(ir, f"payout/rounding/{p.rounding_mode}/{p.rounding_quantum}")
    if p.payment == "per_period":
        for period in p.periods:
            assert period.end is not None and p.quantity is not None
            part = _period(inp, view, period).number(p.quantity, None)
            amount = _amount(ir, part.interval)
            label = f"{_iso(period.start) if period.start else '-inf'} to {_iso(period.end)}"
            if amount is not None and amount[0] == amount[1]:
                out.transfers.append(CashTransfer(at=_pay_date(_utc(period.end), ir), amount=amount[0],
                                                  currency=p.currency, basis="payout"))  # fmt: skip
            else:
                out.unresolved.append(_open_amount(ir, f"the period {label}", part, amount))
        out.policies += [rounding, _timing_policy(ir)]
        return
    amount = _amount(ir, n.interval if n is not None else None)
    if amount is None or amount[0] != amount[1]:
        assert n is not None
        out.unresolved.append(_open_amount(ir, "the payout", n, amount))
        return
    out.policies.append(rounding)
    if resolution_at is None:
        lag = f"{ir.timing.settlement_lag_days} + {ir.timing.payment_lag_days} {ir.timing.lag_days} days"
        out.unresolved.append(f"payout: {amount[0]} {p.currency} per unit, paid {lag} after {when}")
        return
    out.transfers.append(CashTransfer(at=_pay_date(resolution_at, ir), amount=amount[0], currency=p.currency,
                                      basis="payout"))  # fmt: skip
    out.policies.append(_timing_policy(ir))


def _period(inp: _Inputs, view: _View, period: TimeWindow) -> _View:
    """The view's information, with counts and durations restricted to one payment period."""
    out = _View(inp, view.tau, view.clock, interpretations=view.interpretations, restrict=period)
    out.member = view.member
    return out


def _open_amount(ir: ContractIR, what: str, n: _N, amount: tuple[Decimal, Decimal] | None) -> str:
    assert ir.payout.quantity is not None
    span = "unbounded" if amount is None else f"between {amount[0]} and {amount[1]} {ir.payout.currency} per unit"
    causes = "; ".join(w.text for w in n.why) or "its inputs are open"
    return (f"{what}: quantity {ir.payout.quantity.id} is {n.interval} at settlement, so the amount is {span}; the "
            f"contract states no missing-data rule for its quantity (its unknown-at-expiry rule is the trigger's): "
            f"{causes}")  # fmt: skip


# ---------------------------------------------------------------- published examples carry the evaluator's outcome


def example(
    ir: ContractIR,
    scenario_id: str,
    description: str,
    observations: Observations,
    interpretations: Mapping[str, Interpretation],
    *,
    at: datetime,
) -> ScenarioExample:
    """A published example: the scenario's outcome is this evaluator's, and the example names the evaluation that gave
    it (spec §8.5, I8)."""
    ev = evaluate(ir, observations, interpretations, at=at)
    return ScenarioExample(scenario_id=scenario_id, description=description, expected=ev.procedure_status,
                           evaluation=ev.content_hash())  # fmt: skip


def check_example(
    ir: ContractIR,
    published: ScenarioExample,
    observations: Observations,
    interpretations: Mapping[str, Interpretation],
    *,
    at: datetime,
) -> ResolutionEvaluation:
    """Re-evaluate a published example's scenario: its stated outcome must be the evaluator's (I8)."""
    ev = evaluate(ir, observations, interpretations, at=at)
    if published.expected != ev.procedure_status:
        raise ValueError(
            f"example {published.scenario_id} states {published.expected}; the evaluator gives {ev.procedure_status}"
        )
    if published.evaluation != ev.content_hash():
        raise ValueError(f"example {published.scenario_id} names an evaluation other than this scenario's")
    return ev


# ---------------------------------------------------------------- lowering for coverage (spec §8.6 step 1; A09)


@dataclass(frozen=True)
class Comparison:
    """One comparison of an atomic condition and its boundary points (A09 §10.3). kind: a numeric threshold at its
    declared precision, an instant cutoff at the comparison's resolution, or a count at k - 1, k, k + 1."""

    condition: str
    kind: Literal["threshold", "cutoff", "count"]
    detail: str
    points: tuple[cov.BoundaryPoint, ...]


@dataclass(frozen=True)
class Lowered:
    """The trigger as the coverage kernel's expression form. conditions: each atomic condition's IR node kind.
    constraints: what the IR itself makes impossible among its conditions (linked thresholds, repeated predicates, one
    observable's values at instants and over ranges), keyed by ids naming what each says; those over an observable's
    values mention other conditions of the kernel, "ir:<observable>:region<i>:class<k>", whose values the kernel's
    completions give. comparisons: every comparison's boundary points. layouts: for each observable whose values
    conditions compare at instants, the regions and classes those constraints are over, from which a completion is
    realized as observations (`Layout`)."""

    trigger: cov.Expr
    conditions: Mapping[str, str]
    constraints: Mapping[str, cov.Expr]
    comparisons: tuple[Comparison, ...]
    layouts: Mapping[str, Layout] = field(default_factory=dict)

    def rounds(self) -> tuple[tuple[cov.BoundaryPoint, ...], ...]:
        """The boundary points split so that each round gives a condition one comparison, as the coverage kernel takes
        them (a By's deadline and its threshold are two comparisons of one condition)."""
        rounds: list[dict[str, Comparison]] = []
        for c in self.comparisons:
            for r in rounds:
                if c.condition not in r:
                    r[c.condition] = c
                    break
            else:
                rounds.append({c.condition: c})
        return tuple(tuple(p for c in r.values() for p in c.points) for r in rounds)


def lower(ir: ContractIR, *, resolution: timedelta = timedelta(microseconds=1)) -> Lowered:
    """The trigger's boolean skeleton over its atomic conditions (Cmp, Interpretive, NumCmp, By, At, Throughout: every
    node other than And, Or and Not), each a variable named by its node id, with the boundary points of each comparison
    and the constraints the IR places between conditions over one quantity. `resolution` is the step of an instant
    cutoff's neighbours; its default is the smallest an instant represents (A09 §10.3: "neighboring representable
    times")."""
    _check_ir(ir)
    atoms: dict[str, Any] = {}

    def skeleton(e: Any) -> cov.Expr:
        if isinstance(e, And):
            return cov.And(*(skeleton(a) for a in e.args))
        if isinstance(e, Or):
            return cov.Or(*(skeleton(a) for a in e.args))
        if isinstance(e, Not):
            return cov.Not(skeleton(e.arg))
        atoms.setdefault(e.id, e)
        return cov.Var(e.id)

    trigger = skeleton(ir.trigger)
    kinds = {o.id: o for o in ir.observables}
    comparisons = [c for a in atoms.values() for c in _comparisons(a, kinds, resolution)]
    return Lowered(
        trigger=trigger,
        conditions={k: a.kind for k, a in atoms.items()},
        constraints=_implied(atoms, kinds),
        comparisons=tuple(comparisons),
        layouts={o: _layout(o, group, kinds[o]) for o, group in _reading_groups(atoms).items()},
    )


def counted_days(window: TimeWindow) -> list[tuple[date, datetime, datetime]]:
    """The calendar days a CountTrue over `window` counts, each with its start and end instants (as the evaluator
    counts them)."""
    _bounded(window, "a counted window")
    return _days(window)


def duration(a: datetime, b: datetime, unit: str, tz: str) -> Fraction:
    """The length of [a, b) in a DurationIn's unit, as the evaluator measures it."""
    return _measure(_utc(a), _utc(b), unit, tz)


def determination(ir: ContractIR) -> datetime:
    """The instant the contract's rule for an unknown at expiry can apply: its expiration, once its observation cutoff has
    passed too; evaluated then, a contract is never pending (`_Clock.determination`)."""
    return _Clock(_utc(ir.timing.expiration), _utc(ir.timing.observation_cutoff)).determination


def _single_cmp(e: Any) -> Cmp | None:
    return e if isinstance(e, Cmp) else None


def _stated_precision(v: Decimal) -> Decimal:
    exponent = v.as_tuple().exponent
    assert isinstance(exponent, int)
    return Decimal(1).scaleb(exponent)


def _comparisons(a: Any, kinds: Mapping[str, Observable], step: timedelta) -> list[Comparison]:
    out: list[Comparison] = []
    cid = a.id

    def threshold(op: str, value: CmpValue | Decimal, obs: Observable | None, detail: str,
                  domain: tuple[Any, Any] = (None, None)) -> None:  # fmt: skip
        if isinstance(value, OnDay):
            pts = cov.cutoff_points(cid, cast(cov.Op, op), value.value, timedelta(days=1))
            out.append(Comparison(cid, "cutoff", detail, pts))
            return
        if isinstance(value, Category | Flag):
            return
        v = value.value if isinstance(value, Number) else value
        precision = obs.precision if obs is not None and obs.precision is not None else _stated_precision(v)
        normal = _at_precision(op, Fraction(v), Fraction(precision))
        if normal is None:
            return  # no value at the precision meets the comparison's boundary: it is constant (see _implied)
        op2, t = normal
        pts = cov.threshold_points(cid, cast(cov.Op, op2), _to_decimal(t), precision, domain=domain)
        out.append(Comparison(cid, "threshold", detail, pts))

    def cutoff(op: str, instant: datetime, detail: str) -> None:
        out.append(Comparison(cid, "cutoff", detail, cov.cutoff_points(cid, cast(cov.Op, op), instant, step)))

    if isinstance(a, NumCmp):
        e = a.expr
        if isinstance(e, CountTrue):
            k = _at_precision(a.op, Fraction(a.value), Fraction(1))
            if k is not None:
                days = len(_days(e.days))
                pts = cov.count_points(cid, cast(cov.Op, k[0]), int(k[1]), days)
                out.append(Comparison(cid, "count", f"{e.id} {k[0]} {k[1]} over {days} days", pts))
        elif isinstance(e, ValueAt):
            threshold(a.op, Number(value=a.value), kinds[e.observable], f"{e.id} {a.op} {a.value}")
        else:
            domain: tuple[Any, Any] = (None, None)
            if isinstance(e, ReadValue):
                domain = (e.bands[0][0], e.bands[-1][1])
            elif isinstance(e, DurationIn):
                # A duration lies between zero and its window's whole length, in its unit (A09 §10.3: a boundary point
                # outside the values the quantity can take is infeasible, never a scenario).
                w = e.window
                assert w.start is not None and w.end is not None
                whole = _measure(_utc(w.start), _utc(w.end), e.unit, w.tz)
                domain = (Decimal(0), _to_decimal(whole, ROUND_FLOOR))
            threshold(a.op, a.value, None, f"{e.id} {a.op} {a.value}", domain)
    elif isinstance(a, At):
        c = _single_cmp(a.expr)
        if c is not None:
            threshold(c.op, c.value, kinds[c.observable], f"{c.observable} at {_iso(a.at)} {c.op} {_show_value(c.value)}")
    elif isinstance(a, By):
        cutoff("<=" if a.closed else "<", a.deadline, f"the instant it first holds, against the deadline {_iso(a.deadline)}")
        if a.since is not None:
            cutoff(">=" if a.since_closed else ">", a.since, f"the instant it holds, against its start {_iso(a.since)}")
        c = _single_cmp(a.expr)
        if c is not None:
            threshold(c.op, c.value, kinds[c.observable], f"{c.observable} {c.op} {_show_value(c.value)}, at some instant")
    elif isinstance(a, Throughout):
        w = a.window
        # The instant of the one value that breaks the condition, against the window's ends: the condition holds when
        # that value lies outside the window.
        if w.end is not None:
            cutoff(">" if w.end_closed else ">=", w.end, f"a breaking value's instant, against the window's end {_iso(w.end)}")
        if w.start is not None:
            cutoff("<" if w.start_closed else "<=", w.start,
                   f"a breaking value's instant, against the window's start {_iso(w.start)}")  # fmt: skip
        c = _single_cmp(a.expr)
        if c is not None:
            threshold(c.op, c.value, kinds[c.observable], f"{c.observable} {c.op} {_show_value(c.value)}, at every instant")
    return out


def _at_precision(op: str, v: Fraction, p: Fraction) -> tuple[str, Fraction] | None:
    """The comparison with its threshold on the grid of multiples of p, meaning the same for every value on the grid;
    None where it is constant there (== or != a value off the grid)."""
    if (v / p).denominator == 1:
        return (op, v)
    below = Fraction((v / p).numerator // (v / p).denominator) * p
    above = below + p
    if op in ("<", "<="):
        return ("<=", below)
    if op in (">", ">="):
        return (">=", above)
    return None


# ---- constraints the IR places between its conditions


def _implied(atoms: Mapping[str, Any], kinds: Mapping[str, Observable]) -> dict[str, cov.Expr]:
    """What the IR itself makes impossible among its conditions (A09 §10.2: linked thresholds and repeated predicates
    leave some states impossible, and no witness may be fabricated from one), as constraints for the coverage kernel.

    - Comparisons of one number (a count, a duration, a reading, a computed value): the truth assignments its values
      allow, where not every assignment is possible.
    - Comparisons of one observable's values at instants (At, By and Throughout over a single comparison, and a
      published value compared): `_readings`, exactly.
    - Repeated predicates: two conditions with one meaning are equal in every state."""
    lines: dict[str, list[tuple[str, str, Fraction]]] = {}
    domains: dict[str, tuple[str, Any]] = {}
    for cid, a in atoms.items():
        if isinstance(a, NumCmp) and not isinstance(a.expr, ValueAt):
            key = "num:" + _content(a.expr)
            lines.setdefault(key, []).append((cid, a.op, Fraction(a.value)))
            domains[key] = _num_domain(a.expr, kinds)
    out: dict[str, cov.Expr] = {}
    for key, conds in lines.items():
        dom = domains[key]
        if dom[0] == "bands":
            vectors = set()
            for band in dom[1]:
                vals = [band.compare(op, t) for _, op, t in conds]
                if all(v is not Kleene.UNKNOWN for v in vals):
                    vectors.add(tuple(v is Kleene.TRUE for v in vals))
        else:
            reps = _representatives([t for _, _, t in conds], dom)
            vectors = {tuple(_holds(x, op, t) for _, op, t in conds) for x in reps}
        _constrain(out, key, [cid for cid, _, _ in conds], vectors)
    for o, group in _reading_groups(atoms).items():
        out.update(_readings(_layout(o, group, kinds[o]), set(atoms)))
    # Repeated predicates: two atomic conditions with one meaning (different ids) are equal in every state.
    same: dict[str, list[str]] = {}
    for cid, a in atoms.items():
        same.setdefault(a.kind + ":" + _content(a), []).append(cid)
    for ids in same.values():
        for x, y in zip(ids, ids[1:], strict=False):
            out[f"same:{x}={y}"] = _iff(cov.Var(x), cov.Var(y))
    return out


def _iff(a: cov.Expr, b: cov.Expr) -> cov.Expr:
    return cov.And(cov.Or(cov.Not(a), b), cov.Or(a, cov.Not(b)))


Region = tuple[datetime | None, datetime | None]  # (t, t) an instant; (a, b) the open interval between, None unbounded
Bounds = tuple[datetime | None, bool, datetime | None, bool]  # low, low included, high, high included; None unbounded


def _reading_groups(atoms: Mapping[str, Any]) -> dict[str, list[tuple[str, Any]]]:
    """The atomic conditions that compare one observable's value at instants, by observable: At, By and Throughout over
    a single comparison, and a published value compared (a NumCmp over a ValueAt)."""
    groups: dict[str, list[tuple[str, Any]]] = {}
    for cid, a in atoms.items():
        if isinstance(a, NumCmp) and isinstance(a.expr, ValueAt):
            groups.setdefault(a.expr.observable, []).append((cid, a))
        elif isinstance(a, At | By | Throughout) and (c := _single_cmp(a.expr)) is not None:
            groups.setdefault(c.observable, []).append((cid, a))
    return groups


@dataclass(frozen=True)
class Layout:
    """How the conditions comparing one observable's values at instants are cut, for the exact constraint `_readings`
    states and for realizing any state it allows.

    Time is cut at every instant the conditions name (At's instant, a published value's instant, a By's bounds, a
    Throughout's window) into `regions`: those instants, and the open intervals between them. Instants in one interval
    belong to the same bounds, so a condition's truth depends only on which classes of value appear in each region, a
    class being a run of values every comparison treats alike, standing for the run by one of its values (`keys`, as the
    comparisons read it; `classes`, as an observation states it). The kernel variable `variable(i, k)` says that some
    instant of region i has class k's value; `used` are the regions within some condition's bounds, the only ones with
    variables, and a group of one condition has none. A state is realized by one observation per true variable: class
    k's value at `instant(i)`."""

    observable: str
    value_type: str
    conditions: tuple[str, ...]
    bounds: Mapping[str, Bounds]
    comparisons: Mapping[str, tuple[str, Any]]
    every: frozenset[str]  # the Throughout conditions: some instant of the window has a value, and none fails
    regions: tuple[Region, ...]
    used: tuple[int, ...]
    keys: tuple[Any, ...]
    classes: tuple[CmpValue, ...]

    def variable(self, i: int, k: int) -> str:
        return f"ir:{self.observable}:region{i}:class{k}"

    def within(self, i: int, condition: str) -> bool:
        """Whether region i lies within the condition's bounds."""
        lo, lo_closed, hi, hi_closed = self.bounds[condition]
        a, z = self.regions[i]
        if a is not None and a == z:
            return _in(a, lo, lo_closed, hi, hi_closed)
        return (lo is None or (a is not None and a >= lo)) and (hi is None or (z is not None and z <= hi))

    def satisfies(self, condition: str, k: int) -> bool:
        """Whether class k's value satisfies the condition's comparison."""
        op, t = self.comparisons[condition]
        x = self.keys[k]
        if self.value_type in ("number", "date"):
            return _holds(x, op, t)
        return (x == t) == (op == "==")

    def region_of(self, t: datetime) -> int:
        t = _utc(t)
        for i, (a, z) in enumerate(self.regions):
            if a is not None and a == z:
                if t == a:
                    return i
            elif (a is None or t > a) and (z is None or t < z):
                return i
        raise AssertionError("the regions partition time")

    def instant(self, i: int) -> datetime:
        """An instant of region i (the first of `instants`)."""
        return self.instants(i, 1)[0]

    def instants(self, i: int, m: int) -> tuple[datetime, ...]:
        """m distinct instants of region i, one for each class placed there: the instant itself (a region that is an
        instant holds one value), m points spread evenly across an interval (each at a whole second where that keeps
        it inside and distinct), or whole days beyond an unbounded end."""
        a, z = self.regions[i]
        if a is not None and a == z:
            if m > 1:
                raise IRError(f"region {i} of {self.observable} is an instant, which has one value, not {m}")
            return (a,)
        if a is None:
            assert z is not None  # the first region ends at the first instant
            return tuple(z - timedelta(days=j + 1) for j in range(m))
        if z is None:
            return tuple(a + timedelta(days=j + 1) for j in range(m))
        out: list[datetime] = []
        for j in range(m):
            t = a + (z - a) * (j + 1) / (m + 1)
            whole = t.replace(microsecond=0)
            out.append(whole if a < whole < z and whole not in out else t)
        if len(set(out)) != m or not all(a < t < z for t in out):
            raise IRError(f"region {i} of {self.observable} is too short for {m} distinct instants")
        return tuple(out)


def _layout(o: str, group: list[tuple[str, Any]], obs: Observable) -> Layout:
    def bounds(a: Any) -> Bounds:
        if isinstance(a, At):
            return (_utc(a.at), True, _utc(a.at), True)
        if isinstance(a, NumCmp):
            e = a.expr
            assert isinstance(e, ValueAt)  # _reading_groups groups only a compared published value here
            return (_utc(e.at), True, _utc(e.at), True)
        if isinstance(a, By):
            return (_utc(a.since) if a.since else None, bool(a.since_closed), _utc(a.deadline), a.closed)
        w = a.window
        return (_utc(w.start) if w.start else None, w.start_closed, _utc(w.end) if w.end else None, w.end_closed)

    def comparison(a: Any) -> tuple[str, Any]:
        if isinstance(a, NumCmp):
            return (a.op, Fraction(a.value))
        c = a.expr
        return (c.op, _ordinal(c.value) if isinstance(c.value, Number | OnDay) else c.value.value)

    thresholds = [comparison(a)[1] for _, a in group]
    keys: list[Any]
    classes: list[CmpValue]
    if obs.value_type == "number":
        keys = _representatives(thresholds, ("grid", _precision(obs)))
        assert obs.precision is not None
        classes = [Number(value=_to_decimal(x).quantize(obs.precision)) for x in keys]
    elif obs.value_type == "date":
        keys = _representatives(thresholds, ("integer", None))
        classes = [OnDay(value=date.fromordinal(int(x))) for x in keys]
    elif obs.value_type == "bool":
        keys = [True, False]
        classes = [Flag(value=True), Flag(value=False)]
    else:
        named = list(dict.fromkeys(thresholds))
        keys = [*named, None]  # each named value, and one none of them names
        other = "a value other than " + " or ".join(named)
        while other in named:
            other += "'"
        classes = [*(Category(value=x) for x in named), Category(value=other)]
    instants = sorted({t for _, a in group for t in bounds(a)[0::2] if t is not None})
    regions: list[Region] = [(None, instants[0])]
    for i, t in enumerate(instants):
        regions += [(t, t), (t, instants[i + 1] if i + 1 < len(instants) else None)]
    lay = Layout(
        observable=o,
        value_type=obs.value_type,
        conditions=tuple(cid for cid, _ in group),
        bounds={cid: bounds(a) for cid, a in group},
        comparisons={cid: comparison(a) for cid, a in group},
        every=frozenset(cid for cid, a in group if isinstance(a, Throughout)),
        regions=tuple(regions),
        used=(),
        keys=tuple(keys),
        classes=tuple(classes),
    )
    used = tuple(sorted({i for cid in lay.conditions for i in range(len(regions)) if lay.within(i, cid)}))
    return replace(lay, used=used)


def _readings(lay: Layout, taken: set[str]) -> dict[str, cov.Expr]:
    """Exactly which truth assignments the conditions reading one observable at instants allow, over its `Layout`.

    An instant has one value, so a region that is an instant has at most one class. Each condition is then a formula
    over the layout's variables: At (and a compared published value) is the class at its instant, which has a value; By,
    that some instant within its bounds has a satisfying value; Throughout, that some instant within its window has a
    value and none fails. The kernel states are two-valued, so each condition is established: a By or Throughout has
    some instant within its bounds. Any assignment satisfying these is realized by placing one instant per class in each
    region, and every set of observations gives one, so the constraint is exact. A group of one condition needs no other
    conditions: it is constrained only where its comparisons make it constant."""
    o = lay.observable
    if len(lay.conditions) == 1:
        cid = lay.conditions[0]
        values = {lay.satisfies(cid, k) for k in range(len(lay.keys))}
        if len(values) == 2:
            return {}
        return {f"ir:{o}:{cid}:constant": cov.Var(cid) if True in values else cov.Not(cov.Var(cid))}

    def var(i: int, k: int) -> cov.Var:
        name = lay.variable(i, k)
        if name in taken:
            raise IRError(f"an IR node id, {name!r}, is the name of a coverage variable the lowering needs")
        return cov.Var(name)

    out: dict[str, cov.Expr] = {}
    n = len(lay.keys)
    for i in lay.used:
        if lay.regions[i][0] is not None and lay.regions[i][0] == lay.regions[i][1]:
            for k in range(n):
                for m in range(k + 1, n):
                    out[f"ir:{o}:region{i}:one:{k}:{m}"] = cov.Or(cov.Not(var(i, k)), cov.Not(var(i, m)))
    for cid in lay.conditions:
        inside = [i for i in lay.used if lay.within(i, cid)]
        some = [var(i, k) for i in inside for k in range(n)]
        out[f"ir:{o}:{cid}:established"] = cov.Or(*some)
        if cid in lay.every:
            failing = [var(i, k) for i in inside for k in range(n) if not lay.satisfies(cid, k)]
            out[f"ir:{o}:{cid}"] = _iff(cov.Var(cid), cov.Not(cov.Or(*failing)))
        else:
            holding = [var(i, k) for i in inside for k in range(n) if lay.satisfies(cid, k)]
            out[f"ir:{o}:{cid}"] = _iff(cov.Var(cid), cov.Or(*holding))
    return out


def _precision(obs: Observable) -> Fraction:
    """A numeric observable's published precision, which the definition requires it to state."""
    if obs.precision is None:
        raise IRError(f"{obs.id} is compared as a number but states no precision")
    return Fraction(obs.precision)


def _ordinal(v: CmpValue) -> Fraction:
    if isinstance(v, Number):
        return Fraction(v.value)
    assert isinstance(v, OnDay)
    return Fraction(v.value.toordinal())


def _holds(x: Fraction, op: str, t: Fraction) -> bool:
    return {"<": x < t, "<=": x <= t, ">": x > t, ">=": x >= t, "==": x == t, "!=": x != t}[op]


def _num_domain(e: Any, kinds: Mapping[str, Observable]) -> tuple[str, Any]:
    if isinstance(e, CountTrue):
        return ("integer", (Fraction(0), Fraction(len(_days(e.days)))))
    if isinstance(e, DurationIn):
        w = e.window
        assert w.start is not None and w.end is not None
        return ("real", (Fraction(0), _measure(_utc(w.start), _utc(w.end), e.unit, w.tz)))
    if isinstance(e, ReadValue):
        return ("bands", tuple(_band(e, (lo, hi))[0] for lo, hi in e.bands))
    if isinstance(e, ValueAt):
        return ("grid", _precision(kinds[e.observable]))
    return ("real", (None, None))


def _representatives(thresholds: Sequence[Fraction], domain: tuple[str, Any]) -> list[Fraction]:
    """Values of the quantity, one in every region its thresholds cut its domain into."""
    kind, spec = domain
    ts = sorted(set(thresholds))
    if kind == "integer":
        bounds = spec if spec is not None else (None, None)
        cands = {Fraction(f) + d for t in ts for f in (t.numerator // t.denominator, -((-t.numerator) // t.denominator))
                 for d in (-1, 0, 1)}  # fmt: skip
        lo, hi = bounds
        cands |= {b for b in (lo, hi) if b is not None}
        return sorted(c for c in cands if (lo is None or c >= lo) and (hi is None or c <= hi))
    if kind == "grid":
        p = spec
        cands = set()
        for t in ts:
            k = t / p
            base = k.numerator // k.denominator
            cands |= {Fraction(base + d) * p for d in (-1, 0, 1, 2)}
        return sorted(cands)
    lo, hi = spec
    cands = set(ts)
    cands |= {(a + b) / 2 for a, b in zip(ts, ts[1:], strict=False)}
    if ts:
        cands |= {ts[0] - 1, ts[-1] + 1}
    cands |= {b for b in (lo, hi) if b is not None}
    return sorted(c for c in cands if (lo is None or c >= lo) and (hi is None or c <= hi))


def _constrain(out: dict[str, cov.Expr], key: str, ids: list[str], vectors: set[tuple[bool, ...]]) -> None:
    if len(vectors) == 2 ** len(ids):
        return
    if not vectors:
        raise IRError(f"conditions {ids} over one quantity cannot take any values together")

    def lit(cid: str, value: bool) -> cov.Expr:
        return cov.Var(cid) if value else cov.Not(cov.Var(cid))

    states = [cov.And(*(lit(c, v) for c, v in zip(ids, vec, strict=True))) for vec in sorted(vectors)]
    out[f"ir:{key}"] = states[0] if len(states) == 1 else cov.Or(*states)
