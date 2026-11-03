"""The contract IR and what evaluating it produces (spec §4 "contract IR"; amended by addendum A05 and A09).

Mechanical conditions are computed from observables; anything that requires reading law or prose is an Interpretive node
judged by decision models (spec §4). The trigger is evaluated in Kleene three-valued logic (spec §6.3). One evaluator
computes every outcome (I8); its result is a ResolutionEvaluation, with the trigger's truth separate from the procedure's
status and a dated per-unit transfer schedule (A05 §5.1).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from requisite.core.base import Day, Instant, Money, Record, Ref
from requisite.core.evidence import ArtifactId, SyntheticArtifactId
from requisite.core.judgment import TimeWindow, require_zone
from requisite.core.kleene import Kleene

# ---------------------------------------------------------------- observables


class Source(Record):
    """A source of an observable's values, ranked by authority (spec §4: rank 1 is highest)."""

    id: str
    name: str
    url: str | None = None
    rank: int
    role: str


class Observable(Record):
    """A quantity a contract settles on (spec §4).

    revision_policy: which vintage settles (spec §4). "final" is a finite rule only with a declared resolution deadline
    or a fallback (addendum A05 §5.4: "An unlimited `final` vintage is not a finite settlement rule unless 'final' has a
    declared resolution deadline or fallback"): `final_deadline` is the instant by which the value must be final, and
    `final_fallback` the vintage that settles if no final value is published by the settlement determination. unit and precision: what a numeric value is measured in and the precision at which it is published
    (A05 §5.4 unit conversions; A09 §10.3 thresholds at declared precision).
    """

    id: str
    description: str
    value_type: Literal["bool", "number", "date", "category", "text"]
    sources: Sequence[Source]
    schedule: str | None = None
    revision_policy: Literal["first_print", "as_of_expiry", "final"]
    final_deadline: Instant | None = None
    final_fallback: Literal["first_print", "as_of_expiry"] | None = None
    unit: str | None = None
    precision: Money | None = None

    @model_validator(mode="after")
    def _final(self) -> Observable:
        bounded = self.final_deadline is not None or self.final_fallback is not None
        if (self.revision_policy == "final") != bounded:
            raise ValueError("a 'final' revision policy declares a resolution deadline or a fallback, and only it does")
        if not self.sources:
            raise ValueError("an observable has at least one source")
        if self.value_type == "number" and (self.unit is None or self.precision is None):
            raise ValueError("a numeric observable states its unit and published precision")
        return self


# ---------------------------------------------------------------- the trigger expression (spec §4 Expr; §6.3)
# Every node has an id, unique within its contract, so that a clause of the prose links to the node it states (spec
# §13 WP16: "Each clause links to its IR node"), an operative exclusion names the node that carries it (addendum A05
# §5.4), and each atomic condition (Cmp, Interpretive) is a named condition of MC/DC coverage (spec §8.6 step 1).

Op = Literal["<", "<=", "==", "!=", ">=", ">"]


class Number(Record):
    kind: Literal["number"] = "number"
    value: Money


class Category(Record):
    """A category or text value, compared as given."""

    kind: Literal["text"] = "text"
    value: str


class Flag(Record):
    kind: Literal["bool"] = "bool"
    value: bool


class OnDay(Record):
    kind: Literal["date"] = "date"
    value: Day


# A comparison's value carries its type, so that it means the same after a round trip: an untagged "10" could be a
# number or a category (ours; spec §4 Observable.value_type names the kinds a comparison can be over).
CmpValue = Annotated[Number | Category | Flag | OnDay, Field(discriminator="kind")]
VALUE_KIND = {"number": "number", "category": "text", "text": "text", "bool": "bool", "date": "date"}


class Observation(Record):
    """One published value of an observable, from one source (spec §4 Observable; the evaluator's input, I8). `at` is the
    instant the value is for; `published_at` when the source published this value; `final` whether the source marks it
    final (for a "final" revision policy); `artifact` the observed evidence it was read from. The revision policy picks
    among a value's vintages: the first published (first_print), the latest published by expiration (as_of_expiry), or
    the final one (final)."""

    observable: str
    source: str
    at: Instant
    value: CmpValue
    published_at: Instant
    final: bool
    artifact: ArtifactId


class SyntheticObservation(Record):
    """A fuzzing scenario's or a simulation's observation, read from synthetic evidence; it can never stand where an
    observed one is required (spec I4: "enforce this at the type level")."""

    observable: str
    source: str
    at: Instant
    value: CmpValue
    published_at: Instant
    final: bool
    artifact: SyntheticArtifactId


class Cmp(Record):
    """observable op value, at the instant or day its enclosing temporal operator gives (spec §4). The value's type is
    the observable's (`VALUE_KIND`); a numeric value is compared at the observable's declared precision (A09 §10.3)."""

    kind: Literal["cmp"] = "cmp"
    id: str
    observable: str
    op: Op
    value: CmpValue


class And(Record):
    kind: Literal["and"] = "and"
    id: str
    args: Sequence[Expr]

    @model_validator(mode="after")
    def _arity(self) -> And:
        if len(self.args) < 2:
            raise ValueError("And takes at least two conditions")
        return self


class Or(Record):
    kind: Literal["or"] = "or"
    id: str
    args: Sequence[Expr]

    @model_validator(mode="after")
    def _arity(self) -> Or:
        if len(self.args) < 2:
            raise ValueError("Or takes at least two conditions")
        return self


class Not(Record):
    kind: Literal["not"] = "not"
    id: str
    arg: Expr


class By(Record):
    """The condition holds at some observation up to the deadline, which is included or not as `closed` says (spec §4
    "By(expr, t)"; A05 §5.4 endpoint inclusion). `since` is the earliest observation that counts, with its own
    inclusion, or None where every earlier observation counts: the sources give By no lower bound, so the contract
    states it (ours)."""

    kind: Literal["by"] = "by"
    id: str
    expr: Expr
    since: Instant | None
    since_closed: bool | None
    deadline: Instant
    closed: bool

    @model_validator(mode="after")
    def _since(self) -> By:
        if (self.since is None) != (self.since_closed is None):
            raise ValueError("a lower bound states whether it is included; no lower bound states nothing")
        if self.since is not None and not self.since <= self.deadline:
            raise ValueError("By's lower bound is not after its deadline")
        return self


class At(Record):
    """The condition holds at the instant `at` (spec §4)."""

    kind: Literal["at"] = "at"
    id: str
    expr: Expr
    at: Instant


class Throughout(Record):
    """The condition holds at every observation in the window (spec §4)."""

    kind: Literal["throughout"] = "throughout"
    id: str
    expr: Expr
    window: TimeWindow


class Interpretive(Record):
    """A condition a resolver must judge from text (spec §4): a question template and its bindings, answered by decision
    models and mapped to a truth by its answer's status (spec §6.3)."""

    kind: Literal["interpretive"] = "interpretive"
    id: str
    question_template: str
    bindings: Mapping[str, Any]


# The binding under which a judged condition of an Exists body is given the member it is asked about (core.Exists).
MEMBER = "member"


class Exists(Record):
    """Some member of an open class satisfies the condition: a criterion over every instrument, action or filing of a
    kind, each judged for itself (grammar findings, decisions: as in a filed contract's "a Qualifying Action has been
    taken", where a qualifying action is any instrument of the named kinds with one of the named effects, each
    instrument's own terms governing it).

    member: what one member is, in the contract's words. domain: the class `expr` ranges over, in the contract's words.
    sources: where the class's members are recorded, ranked as the contract ranks them (rank 1 highest), at least one:
    a member is a record one of them documents by the observation cutoff. on_conflict: which record governs where the
    sources' records of a member conflict, or a source retracts, corrects or removes information: the highest-ranked
    source's, or the official record of the body that took the action (as in "Where the records of the Source Agencies
    conflict, or where a Source Agency retracts, corrects, or removes information, the official records of the body that
    took the action govern"); the resolvers who list the members and judge each are bound by it. expr: the condition each member is
    tested on. Each judged
    condition of `expr` (an Interpretive or a ReadValue) is asked about one member at a time: its bindings are the
    contract's, and the member is given under the binding MEMBER, as its evidence identifies it (the record, its
    publisher and date), so the question names the member it is about; a judged node of `expr` binds no MEMBER itself.
    Every other node of `expr` is the contract's own, the same for each member. `expr` holds a judged condition (one
    with none says nothing of any member) and no further Exists (a member of a member has no meaning here).

    Evaluated over its members, an input of evaluation like the interpretations (requisite/contract/evaluate.py,
    Members): the members the sources document, each with its judged conditions resolved for it, and whether that list
    is complete. Which records are members is the sources' record, read; whether one qualifies is judged, per member;
    complete is TRUE where every one of `sources` was read through the observation cutoff, a fact of the retrieval,
    never a judgment of the text (with no source named, no list could be complete, so one is required) (decisions: each check is of the kind of thing it checks). Kleene (spec
    §6.3): TRUE where some member's condition is TRUE; FALSE where the list is complete, the observation cutoff has passed
    (a member may be documented until then) and every member's condition is FALSE, an empty complete list included;
    UNKNOWN otherwise (I6: a member not yet listed is not a member that fails)."""

    kind: Literal["exists"] = "exists"
    id: str
    member: str
    domain: str
    sources: Sequence[Source]
    on_conflict: Literal["highest_ranked", "acting_body_record"]
    expr: Expr

    @model_validator(mode="after")
    def _members(self) -> Exists:
        if not self.member.strip() or not self.domain.strip():
            raise ValueError(f"{self.id}: an Exists states what a member is and the class it ranges over")
        if not self.sources:
            raise ValueError(f"{self.id}: an Exists names where its members are recorded: with no source, no list of "
                             "members could be complete, and none could be read as absent (I6)")  # fmt: skip
        if len({s.id for s in self.sources}) != len(self.sources):
            raise ValueError(f"{self.id}: an Exists' source ids are distinct")
        nodes = list(walk(self.expr))
        if any(isinstance(n, Exists) for n in nodes):
            raise ValueError(f"{self.id}: an Exists holds no further Exists (a member of a member has no meaning here)")
        judged = [n for n in nodes if isinstance(n, Interpretive | ReadValue)]
        if not judged:
            raise ValueError(f"{self.id}: an Exists body judges something of each member; one with no judged condition "
                             "says nothing of any member")  # fmt: skip
        bound = [n.id for n in judged if MEMBER in n.bindings]
        if bound:
            raise ValueError(f"{self.id}: {bound} bind {MEMBER!r}, which is given per member, as each is asked about")
        return self


class CountTrue(Record):
    """The number of calendar days of `days` on which the condition holds, counted in the window's time zone (spec §4).
    Its value over three-valued days is an interval [k, k + u] (A05 §5.3)."""

    kind: Literal["count_true"] = "count_true"
    id: str
    expr: Expr
    days: TimeWindow


class DurationIn(Record):
    """The time within the window during which the condition holds, in `unit` (spec §4); an interval over unknowns, as
    CountTrue's is. calendar_day counts calendar days of the window's time zone; elapsed_day counts 24-hour periods, so
    a daylight-saving change does not change it (A05 §5.4: "Define ... day-count conventions")."""

    kind: Literal["duration_in"] = "duration_in"
    id: str
    expr: Expr
    window: TimeWindow
    unit: Literal["calendar_day", "elapsed_day", "hour"]


class ReadValue(Record):
    """A number a resolver reads from text, with its unit: a term an instrument sets (an alternative monthly wage, a
    modified overtime threshold). Decision models answer it as a closed set of bands (spec §4: a question answered by a
    distribution over a closed set), so its value is the interval of the band the resolved answer names, and unknown is
    the whole range (A05 §5.3: a number left open by unknowns is an interval, never a guess). Each band is [low, high),
    the last [low, high], so a value on a shared endpoint belongs to the band above it (A05 §5.4: "Define open/closed
    interval endpoints"). Ours: spec §9.3's specimen tests a cost computed from such terms, and spec §4's Interpretive
    yields only a truth."""

    kind: Literal["read_value"] = "read_value"
    id: str
    question_template: str
    bindings: Mapping[str, Any]
    unit: str
    bands: Sequence[tuple[Money, Money]]

    @model_validator(mode="after")
    def _bands(self) -> ReadValue:
        if not self.bands or any(lo >= hi for lo, hi in self.bands[:-1]) or self.bands[-1][0] > self.bands[-1][1]:
            raise ValueError("a reading's bands are nonempty intervals [low, high), the last [low, high]")
        if any(a[1] > b[0] for a, b in zip(self.bands, self.bands[1:], strict=False)):
            raise ValueError("a reading's bands are ordered and do not overlap")
        return self


class ValueAt(Record):
    """A numeric observable's value at an instant, selected by the contract's revision and conflict policies (spec §4
    Observable; ours: a computed test reads published rates, such as a minimum wage in effect on a day)."""

    kind: Literal["value_at"] = "value_at"
    id: str
    observable: str
    at: Instant


class Computed(Record):
    """A number computed by a registered consequence node from numeric sub-expressions (spec §5: "The contract IR
    compiles into the computation layer, so the settlement function and the hedge payoff are the same code"; I7: code
    owns consequences). The node is pure and registered (spec §4 @node), so the same arithmetic settles, prices and
    fuzzes; over interval inputs its value is the interval of its outputs."""

    kind: Literal["computed"] = "computed"
    id: str
    node: Ref
    inputs: Mapping[str, NumExpr]


NumExpr = Annotated[CountTrue | DurationIn | ReadValue | ValueAt | Computed, Field(discriminator="kind")]


class NumCmp(Record):
    """A count or duration compared with a threshold, e.g. CountTrue(...) >= 20 (ours: spec §4 puts CountTrue and
    DurationIn in the Expr union, and a trigger on them, such as spec §2.5's "how many days ... will station S record at
    least X inches", compares their value)."""

    kind: Literal["num_cmp"] = "num_cmp"
    id: str
    expr: NumExpr
    op: Op
    value: Money


Expr = Annotated[
    Cmp | And | Or | Not | By | At | Throughout | Interpretive | NumCmp | Exists,
    Field(discriminator="kind"),
]

for _m in (And, Or, Not, By, At, Throughout, CountTrue, DurationIn, ReadValue, ValueAt, Computed, NumCmp, Exists):
    _m.model_rebuild()


def walk(expr: Any) -> Iterator[Any]:
    """Every node of a trigger expression, the expression first."""
    yield expr
    if isinstance(expr, And | Or):
        for a in expr.args:
            yield from walk(a)
    elif isinstance(expr, Not):
        yield from walk(expr.arg)
    elif isinstance(expr, By | At | Throughout | CountTrue | DurationIn | Exists):
        yield from walk(expr.expr)
    elif isinstance(expr, NumCmp):
        yield from walk(expr.expr)
    elif isinstance(expr, Computed):
        for sub in expr.inputs.values():
            yield from walk(sub)


# ---------------------------------------------------------------- payout, timing, procedure


class Payout(Record):
    """Per unit (spec §4). currency, and rounding to `rounding_quantum` by `rounding_mode`, are explicit (A05 §5.1,
    §5.4: "Define ... rounding") wherever the amount is computed; a binary payout pays its unit value as stated, so it
    states a rounding only where its terms do. payment: paid once at settlement, or per
    period of `periods`, each period's amount paid after its end (A05 §5.3: "A count paid once at expiry and a strip paid
    periodically can have identical terminal amounts and very different liquidity consequences")."""

    kind: Literal["binary", "per_unit", "capped_linear"]
    unit_value: Money
    quantity: NumExpr | None = None
    cap: Money | None = None
    currency: str
    rounding_quantum: Money | None = Field(default=None, exclude_if=lambda v: v is None)
    rounding_mode: Literal["half_even", "half_up", "down", "up"] | None = Field(default=None, exclude_if=lambda v: v is None)
    payment: Literal["at_settlement", "per_period"]
    periods: Sequence[TimeWindow] = ()

    @model_validator(mode="after")
    def _shape(self) -> Payout:
        if self.kind == "binary" and self.quantity is not None:
            raise ValueError("a binary payout pays unit_value once; it has no quantity")
        if self.kind in ("per_unit", "capped_linear") and self.quantity is None:
            raise ValueError(f"a {self.kind} payout names the quantity it pays per unit of")
        if (self.kind == "capped_linear") != (self.cap is not None):
            raise ValueError("a cap belongs to a capped_linear payout, and it has one")
        if (self.payment == "per_period") != bool(self.periods):
            raise ValueError("a per-period payment lists its periods, and only it does")
        if (self.rounding_quantum is None) != (self.rounding_mode is None):
            raise ValueError("a rounding states its quantum and its mode together")
        if self.kind != "binary" and self.rounding_quantum is None:
            raise ValueError(f"a {self.kind} payout states how its amount is rounded (A05 §5.4)")
        return self


class Timing(Record):
    """The contract's dates (spec §4), with the cutoff for observations and the payment lag distinguished from the
    settlement determination (A05 §5.3: event, publication, observation cutoff, settlement determination and payment
    dates are different dates). Lags count calendar days or business days, in the time zone `tz` (A05 §5.4: "Define ...
    timezone, day-count conventions"); a business-day lag skips weekends and the holidays of `holiday_calendar`, whose
    one registered calendar is the United States federal holidays of 5 U.S.C. §6103 (ours: the calendar the contracts
    we build settle against; another is registered here before a contract can use it, A05 §5.4 "Mechanical policies
    must be executable"). settlement_lag: "exact" where the settlement determination falls settlement_lag_days after
    resolution; "at_most" where the contract sets a deadline instead ("no later than the day after the Expiration Date"):
    it falls between resolution and that day, so a payment's date is a range (CashTransfer.earliest) (ours, from the
    filed terms; "exact" is left out of the record, so a contract stated before this field hashes as it did)."""

    open: Instant
    close: Instant
    expiration: Instant
    observation_cutoff: Instant
    settlement_lag_days: int
    payment_lag_days: int
    lag_days: Literal["calendar", "business"]
    holiday_calendar: Literal["us_federal"] | None
    tz: str
    settlement_lag: Literal["exact", "at_most"] = Field(default="exact", exclude_if=lambda v: v == "exact")

    @model_validator(mode="after")
    def _order(self) -> Timing:
        if not self.open <= self.close <= self.expiration:
            raise ValueError("open <= close <= expiration")
        if self.settlement_lag_days < 0 or self.payment_lag_days < 0:
            raise ValueError("lags are non-negative")
        if (self.lag_days == "business") != (self.holiday_calendar is not None):
            raise ValueError("business-day lags name their holiday calendar, and only they do")
        require_zone(self.tz)
        return self


class ConflictPolicy(Record):
    """How disagreeing sources are reconciled: a registered procedure and its parameters (A05 §5.4: "Use a registered
    procedure ID plus parameters, or an explicit typed policy"). The two procedures registered so far are ours, the
    two readings of spec §4's precedence list that an evaluator can apply deterministically: precedence_first_available,
    the highest-ranked source in the procedure's precedence that has published settles; require_agreement, the sources
    named in `sources` must agree, or the result is disputed. A contract whose rule is neither is not supported exactly
    (A05 §5.5) until its procedure is registered here."""

    procedure: Literal["precedence_first_available", "require_agreement"]
    sources: Sequence[str] = ()

    @model_validator(mode="after")
    def _sources(self) -> ConflictPolicy:
        if self.procedure == "require_agreement" and len(set(self.sources)) < 2:
            raise ValueError("require_agreement names the two or more sources that must agree")
        if self.procedure == "precedence_first_available" and self.sources:
            raise ValueError("precedence_first_available follows the procedure's precedence; it names no sources")
        return self


class VoidTransfer(Record):
    """What a void returns per unit (A05 §5.2: "A void result does not imply a particular refund")."""

    returns: Literal["premium", "nothing", "fixed_amount"]
    amount: Money | None = None

    @model_validator(mode="after")
    def _amount(self) -> VoidTransfer:
        if (self.returns == "fixed_amount") != (self.amount is not None):
            raise ValueError("a fixed-amount void states its amount, and only it")
        return self


class ResolutionProcedure(Record):
    """How the contract resolves (spec §4, amended by A05 §5.2 and §5.4). on_unknown_at_expiry says what an unknown
    trigger means at settlement without changing the trigger's truth (A05 §5.2). early_expiration: "when_true" when the
    contract expires early once its trigger is true (spec §9.3: "early_close_condition"; the Western Grazers terms: "If
    an event described in the Payout Criterion occurs, expiration will be moved to an earlier date and time in accordance
    with Rule 7.2"). Its timing is what the contract states: "first_true" where the contract itself fixes the instant at
    which the trigger is first true on the observations then published; "per_exchange_rule" where it defers to a rule
    outside the IR's sources, named in early_expiration_rule, and then the outcome is determined once the trigger is
    true while the early date stays unresolved, named, until that rule is supplied (A05 §5.2: "Unsupported procedure
    details must be reported, not guessed"). An extension names its length and what happens if the trigger is still
    unknown after it. "per_rule": an unknown trigger at expiry is settled by a rule of the exchange outside the IR's
    sources, named in unknown_rule ("If an Expiration Value cannot be determined ... Kalshi has the right to determine
    payouts pursuant to Rule 7.1"): the outcome is that determination, which is not among the inputs, so the evaluation
    stays pending on it (A05 §5.2: "Unsupported procedure details must be reported, not guessed"). void_transfer: what a
    void returns (A05 §5.2: "The per-unit transfer policy must state whether acquisition premium, a fixed amount, fees,
    or another amount is returned"), required where the procedure can end void (on_unknown_at_expiry or after_extension "void"); a contract with no void path
    need not state one, and one stated there is never applied (contracts read before it was optional state one)."""

    precedence: Sequence[str]
    conflict_policy: ConflictPolicy
    on_unknown_at_expiry: Literal["no", "yes", "extend", "void", "per_rule"]
    unknown_rule: str | None = Field(default=None, exclude_if=lambda v: v is None)
    early_expiration: Literal["none", "when_true"]
    early_expiration_at: Literal["first_true", "per_exchange_rule"] | None = None
    early_expiration_rule: str | None = None
    extension_days: int | None = None
    after_extension: Literal["no", "yes", "void"] | None = None
    void_transfer: VoidTransfer | None = Field(default=None, exclude_if=lambda v: v is None)

    @model_validator(mode="after")
    def _extend(self) -> ResolutionProcedure:
        early = self.early_expiration == "when_true"
        if early != (self.early_expiration_at is not None):
            raise ValueError("an early expiration states its timing, and only it does")
        if (self.early_expiration_at == "per_exchange_rule") != (self.early_expiration_rule is not None):
            raise ValueError("a timing deferred to an exchange rule names that rule, and only it does")
        extending = self.on_unknown_at_expiry == "extend"
        if extending != (self.extension_days is not None and self.after_extension is not None):
            raise ValueError("an extension states its days and what follows it, and only an extension does")
        if (self.on_unknown_at_expiry == "per_rule") != (self.unknown_rule is not None):
            raise ValueError("an unknown left to an exchange rule names that rule, and only it does")
        if "void" in (self.on_unknown_at_expiry, self.after_extension) and self.void_transfer is None:
            raise ValueError("a procedure that can end void states what a void returns")
        return self


ProcedureStatus = Literal["resolved_yes", "resolved_no", "pending", "extended", "void", "disputed"]


class Exclusion(Record):
    """An exclusion clause. One that changes settlement is operative and names the IR or interpretive node that carries
    it; a disclosure note is marked non-operative (A05 §5.4)."""

    text: str
    operative: bool
    ir_node: str | None = None

    @model_validator(mode="after")
    def _mapped(self) -> Exclusion:
        if self.operative != (self.ir_node is not None):
            raise ValueError("an operative exclusion maps to an IR node, and a non-operative one does not")
        return self


class ScenarioExample(Record):
    """A published example: a scenario that passed fuzzing, with the evaluator's outcome (spec §8.5, I8). `evaluation`
    is the content hash of the ResolutionEvaluation that produced `expected`."""

    scenario_id: str
    description: str
    expected: ProcedureStatus
    evaluation: str


class ContractIR(Record):
    """A contract in the IR (spec §4). Every observable the trigger or payout references is declared."""

    id: str
    version: int
    observables: Sequence[Observable]
    trigger: Expr
    payout: Payout
    timing: Timing
    resolution: ResolutionProcedure
    examples: Sequence[ScenarioExample] = ()
    exclusions: Sequence[Exclusion] = ()

    @model_validator(mode="after")
    def _references(self) -> ContractIR:
        declared = {o.id for o in self.observables}
        if len(declared) != len(self.observables):
            raise ValueError("observable ids are distinct")
        used = [n.observable for e in (self.trigger, self.payout.quantity) if e is not None for n in walk(e)
                if isinstance(n, Cmp | ValueAt)]  # fmt: skip
        missing = sorted(set(used) - declared)
        if missing:
            raise ValueError(f"the IR references undeclared observables: {missing}")
        sources = {s.id for o in self.observables for s in o.sources}
        if set(self.resolution.precedence) - sources:
            raise ValueError("the precedence names only the observables' sources")
        if set(self.resolution.conflict_policy.sources) - sources:
            raise ValueError("a conflict policy names only the observables' sources")
        kinds = {o.id: VALUE_KIND[o.value_type] for o in self.observables}
        for e in (self.trigger, self.payout.quantity):
            for n in walk(e) if e is not None else ():
                if isinstance(n, Cmp) and n.value.kind != kinds[n.observable]:
                    raise ValueError(f"{n.id} compares {n.observable}, a {kinds[n.observable]}, with a {n.value.kind}")
                if isinstance(n, ValueAt) and kinds[n.observable] != "number":
                    raise ValueError(f"{n.id} reads {n.observable}'s value as a number; it is a {kinds[n.observable]}")
        nodes: dict[str, Any] = {}
        for e in (self.trigger, self.payout.quantity):
            for n in walk(e) if e is not None else ():
                if nodes.setdefault(n.id, n) != n:
                    raise ValueError(f"IR node id {n.id!r} names two different nodes; an id names one node")
        ids = set(nodes)
        dangling = [x.ir_node for x in self.exclusions if x.operative and x.ir_node not in ids]
        if dangling:
            raise ValueError(f"operative exclusions name IR nodes the contract does not have: {dangling}")
        return self

    @property
    def ref(self) -> Ref:
        return Ref(id=self.id, version=self.version)


# ---------------------------------------------------------------- evaluation (addendum A05 §5.1)


class CashTransfer(Record):
    """One dated per-unit transfer (A05 §5.1). basis: payout, contractual refund, fee, and so on. Where the contract
    sets a deadline rather than a date (Timing.settlement_lag "at_most"), `at` is the latest instant it can be paid and
    `earliest` the first; otherwise `at` is its date and earliest is not stated."""

    at: Instant
    amount: Money
    currency: str
    basis: str
    earliest: Instant | None = Field(default=None, exclude_if=lambda v: v is None)

    @model_validator(mode="after")
    def _range(self) -> CashTransfer:
        if self.earliest is not None and not self.earliest <= self.at:
            raise ValueError("a transfer's earliest date is not after its latest (`at`)")
        return self


class ResolutionEvaluation(Record):
    """The single evaluator's full output (I8; A05 §5.1). trigger_truth and procedure_status are separate: an unknown
    trigger can resolve NO under a stated policy without becoming false (A05 §5.2). quantity: the payout quantity as an
    interval where unknowns leave it open (A05 §5.3). unresolved: what the result still depends on."""

    contract: Ref
    evidence_state_hash: str
    trigger_truth: Kleene
    procedure_status: ProcedureStatus
    quantity: tuple[Money, Money] | None = None
    transfers: Sequence[CashTransfer]
    unresolved: Sequence[str] = ()
    policies: Sequence[Ref] = ()

    @model_validator(mode="after")
    def _consistent(self) -> ResolutionEvaluation:
        if self.procedure_status == "resolved_yes" and self.trigger_truth is Kleene.FALSE:
            raise ValueError("a false trigger never resolves YES")
        if self.procedure_status == "resolved_no" and self.trigger_truth is Kleene.TRUE:
            raise ValueError("a true trigger never resolves NO")
        if self.quantity is not None and not self.quantity[0] <= self.quantity[1]:
            raise ValueError("a quantity interval is [low, high]")
        return self


Support = Literal["supported_exact", "supported_with_explicit_assumptions", "unsupported"]
Equivalence = Literal[
    "counterexample_found",
    "tested_equivalent_on_suite",
    "exhaustive_equivalent_on_declared_finite_domain",
    "unsupported_or_unassessed",
]


class HarvestedContract(Record):
    """A filed contract read into the IR (spec §9.3, WP12; A05 §5.5: each harvested contract is supported exactly, with
    explicit assumptions, or unsupported, never silently simplified). `filing` names the committed harvest of the filing
    read (a FiledSeries key, for a listed series) and `filed_terms` the artifacts of the filed documents read; `ir` is set
    unless unsupported; `assumptions` and `unsupported_terms` say what the IR does not carry exactly; `equivalence` is A09
    §10.5's scope of the claim that the IR means what the filing says."""

    filing: str
    filed_terms: Sequence[str]
    support: Support
    ir: Ref | None
    assumptions: Sequence[str] = ()
    unsupported_terms: Sequence[str] = ()
    equivalence: Equivalence

    @model_validator(mode="after")
    def _support(self) -> HarvestedContract:
        if (self.support == "unsupported") != (self.ir is None):
            raise ValueError("an unsupported contract has no IR, and a supported one has one")
        if self.support == "supported_with_explicit_assumptions" and not self.assumptions:
            raise ValueError("support with assumptions lists them")
        if self.support == "supported_exact" and (self.assumptions or self.unsupported_terms):
            raise ValueError("exact support has no assumptions and no unsupported terms")
        if self.support == "unsupported" and not self.unsupported_terms:
            raise ValueError("an unsupported contract names the terms the IR cannot carry")
        return self


# ---------------------------------------------------------------- filed series (spec §9.3; F1)
# A listed series has one filed terms document: contract_terms_url is a field of the series, not of its markets, and the
# terms are written with placeholders that each market's own fields fill (the Western Grazers terms: "as in effect on
# the last day of <time period>"; fixtures/kalshi/KXCAGOAT). So a series is read as an instrument family: one market's
# IR, and bindings saying which IR value each per-market field fills (ours). That every market's operative terms are the
# representative's with its own values bound is the family's claim, not assumed: a market IR may differ from the
# representative only at its bound values (ContractFamily), each must re-derive from its market's record by the
# bindings (the contract proposal's schema gate), and each settled market is checked against how it settled (its tests
# gate).


class FiledSeries(Record):
    """A series of filed terms as F1 harvested it at the draw's instant (spec §9.3, §12 F1; decisions, F1's draw rule):
    its series record, its filed contract (contract_url, None where the series carries none) and current terms
    (contract_terms_url), and each harvested market's record by ticker. `as_of` is the draw's instant: a market's known
    outcome is the one its record shows settled by it."""

    series: str
    as_of: Instant
    why: str
    record: ArtifactId
    contract: ArtifactId | None
    terms: ArtifactId
    markets: Mapping[str, ArtifactId]

    @property
    def key(self) -> str:
        """The harvest's key: the series and the draw's instant, in UTC."""
        return f"{self.series}|{self.as_of.astimezone(UTC).isoformat()}"


ListingSlot = Literal[
    "threshold", "at", "since", "deadline", "window_start", "window_end", "open", "close", "expiration",
    "observation_cutoff",
]  # fmt: skip
TIMING_SLOTS = frozenset({"open", "close", "expiration", "observation_cutoff"})
# The node kinds that have each slot: a comparison's threshold, an instant read, a By's bounds, a window's ends.
SLOT_NODES: Mapping[str, tuple[type[Record], ...]] = {
    "threshold": (Cmp, NumCmp),
    "at": (At, ValueAt),
    "since": (By,),
    "deadline": (By,),
    "window_start": (Throughout, CountTrue, DurationIn),
    "window_end": (Throughout, CountTrue, DurationIn),
}


class ListingBinding(Record):
    """One per-market listing field of a series and the IR value it fills (ours; spec §9.3's per-market fields). `field`
    is the market record's field, a dotted path into its objects where the value is nested (custom_strike.Option).
    `node` and `slot` name the value it fills: a node's threshold (Cmp, NumCmp), the instant it reads (At, ValueAt), a
    By's bounds (since, deadline), or a window's ends (Throughout, CountTrue, DurationIn); or, with no node, one of the
    timing's dates (open, close, expiration, observation_cutoff). The field's value enters as it is, in the slot's type:
    a threshold takes the number, text or date its comparison's value is, an instant slot an instant."""

    field: str
    node: str | None
    slot: ListingSlot

    @model_validator(mode="after")
    def _target(self) -> ListingBinding:
        if not self.field.strip() or any(not part for part in self.field.split(".")):
            raise ValueError("a binding names a market field, its nested parts joined by dots")
        if (self.slot in TIMING_SLOTS) != (self.node is None):
            raise ValueError("a timing date is filled with no node, and a node's value with its node named")
        return self


class SettlementBinding(Record):
    """What a market record's settlement value reports (ours; spec §9.3: "settled markets supplying known outcomes"):
    the observable whose value the exchange settled on, and the market field (`at_field`) giving the instant that value
    is for."""

    observable: str
    at_field: str


def _binding_problems(ir: ContractIR, bindings: Sequence[ListingBinding], settlement: SettlementBinding | None
                      ) -> list[str]:  # fmt: skip
    """What makes bindings name values the IR does not have: each binding fills a slot its node has, no value is filled
    twice, and the settlement value reports an observable the IR declares."""
    nodes = {n.id: n for e in (ir.trigger, ir.payout.quantity) if e is not None for n in walk(e)}
    problems = []
    filled = [(b.node, b.slot) for b in bindings]
    twice = sorted({f"{n or 'timing'}.{s}" for n, s in filled if filled.count((n, s)) > 1})
    if twice:
        problems.append(f"values filled by more than one field: {twice}")
    for b in bindings:
        if b.node is None:
            continue
        node = nodes.get(b.node)
        if node is None:
            problems.append(f"{b.field} fills {b.slot} of {b.node}, which is not a node of the IR")
        elif not isinstance(node, SLOT_NODES[b.slot]):
            problems.append(f"{b.field} fills {b.slot} of {b.node}, a {node.kind} node, which has no {b.slot}")
        elif b.slot == "since" and isinstance(node, By) and node.since is None:
            problems.append(f"{b.field} fills the lower bound of {b.node}, which states none (nor whether it is included)")
    if settlement is not None and settlement.observable not in {o.id for o in ir.observables}:
        problems.append(f"the settlement value reports {settlement.observable}, which the IR does not declare")
    return problems


class ContractFamily(Record):
    """A filed series read into the IR as an instrument family (spec §9.3, WP12): the harvested contract's support
    classification (A05 §5.5), one representative market's IR, the bindings that give each market's IR from its record,
    what its settlement value reports, and each harvested market's IR so derived, so that each re-derives from what is
    committed. An unsupported series has no IR, bindings or market IRs."""

    contract: HarvestedContract
    representative: ContractIR | None
    bindings: Sequence[ListingBinding] = ()
    settlement: SettlementBinding | None = None
    markets: Sequence[ContractIR] = ()

    @model_validator(mode="after")
    def _family(self) -> ContractFamily:
        supported = self.contract.support != "unsupported"
        if supported != (self.representative is not None):
            raise ValueError("a supported series has its representative IR, and an unsupported one has none")
        if self.representative is None:
            if self.bindings or self.settlement is not None or self.markets:
                raise ValueError("an unsupported series has no bindings, settlement binding or market IRs")
            return self
        rep = self.representative
        if self.contract.ir != rep.ref:
            raise ValueError("the harvested contract names its representative IR")
        ids = [m.id for m in self.markets]
        if len(set(ids)) != len(ids):
            raise ValueError("each market has one IR")
        if any(m.version != rep.version for m in self.markets):
            raise ValueError("a family's market IRs carry its representative's version")
        problems = _binding_problems(rep, self.bindings, self.settlement)
        if problems:
            raise ValueError("; ".join(problems))
        shape = _unbound(rep, self.bindings)
        differ = [m.id for m in self.markets if _unbound(m, self.bindings) != shape]
        if differ:
            raise ValueError(f"market IRs {differ} differ from the representative beyond the values their bindings fill "
                             "(a source, policy, operator or rule no binding varies: A09 §10.4, A05 §5.5)")
        return self


_WINDOW_FIELD = {"throughout": "window", "count_true": "days", "duration_in": "window"}
_UNBOUND = "<bound>"


def _unbound(ir: ContractIR, bindings: Sequence[ListingBinding]) -> Any:
    """An IR's content with its id and every value its bindings fill blanked: two market IRs of one family have this
    equal, since only their bound values may differ."""
    dump = ir.model_dump(mode="json")
    dump["id"] = _UNBOUND
    for b in bindings:
        if b.node is None:
            dump["timing"][b.slot] = _UNBOUND
    by_node: dict[str, list[ListingBinding]] = {}
    for b in bindings:
        if b.node is not None:
            by_node.setdefault(b.node, []).append(b)

    def blank(node: dict[str, Any]) -> None:
        """Blank a node's bound values, then its child nodes', following only the IR's own structure (an And's or Or's
        args, a Not's arg, a temporal or count node's expr, a NumCmp's expr, a Computed's inputs), so a parameter of an
        Interpretive or ReadValue node (its bindings) is compared as it is."""
        for b in by_node.get(node["id"], []):
            if b.slot == "threshold":
                node["value"] = _UNBOUND
            elif b.slot in ("at", "since", "deadline"):
                node[b.slot] = _UNBOUND
            else:
                node[_WINDOW_FIELD[node["kind"]]]["start" if b.slot == "window_start" else "end"] = _UNBOUND
        children = [*node.get("args", ()), *([node["arg"]] if "arg" in node else []),
                    *([node["expr"]] if "expr" in node else []), *node.get("inputs", {}).values()]  # fmt: skip
        for child in children:
            blank(child)

    blank(dump["trigger"])
    if dump["payout"].get("quantity") is not None:
        blank(dump["payout"]["quantity"])
    return dump


# ---------------------------------------------------------------- rulebook prose and fuzz packets (spec §8.5-8.6; I1)

ProseSectionName = Literal[
    "underlying_and_definitions", "source_hierarchy_and_conflict_rule", "trigger", "examples", "timing",
    "unknown_at_expiry", "revisions",
]  # fmt: skip
# spec §8.5: "Prose is generated from the IR by a clause drafter into fixed sections: underlying and definitions, source
# hierarchy and conflict rule, trigger, qualifying and non-qualifying examples, timing, unknown-at-expiry and revisions."
SECTION_ORDER: tuple[ProseSectionName, ...] = (
    "underlying_and_definitions", "source_hierarchy_and_conflict_rule", "trigger", "examples", "timing",
    "unknown_at_expiry", "revisions",
)  # fmt: skip


class RulebookSection(Record):
    """One section of a rulebook: its text, and the ids of the IR nodes and observables it states (spec §8.7: "Every
    clause links to its IR node")."""

    name: ProseSectionName
    text: str
    ir_nodes: Sequence[str]


class Rulebook(Record):
    """A contract version's rulebook prose, generated from its IR by the clause drafter and proposed as a "clause" (spec
    I1; core.REQUIRED_GATES: schema and tests), so it enters state only through its gates. Spec §8.5: "The sections are
    underlying and definitions, source hierarchy and conflict rule, trigger, qualifying and non-qualifying examples,
    timing, unknown at expiry, and revisions", and "Examples are not hand-picked. They are the coverage scenarios from
    §8.6 that passed fuzzing, rendered with their IR outcomes." So a drafted rulebook holds every section but the
    examples, and an examples section, where present, is empty: the examples are rendered by code from a fuzz report's
    passed scenarios, never drafted. Its sections in §8.5's order, each once, is ours (a rulebook reads in one order).
    The tests gate holds every section to the contract's own nodes and observables, and requires every condition that
    decides settlement to be stated (ours, from the drafter's task in §8.5: it compiles the IR into prose)."""

    contract: Ref
    sections: Sequence[RulebookSection]

    @model_validator(mode="after")
    def _sections(self) -> Rulebook:
        names = [s.name for s in self.sections]
        if len(set(names)) != len(names) or names != [n for n in SECTION_ORDER if n in names]:
            raise ValueError(f"a rulebook's sections are spec §8.5's, each once, in its order: {SECTION_ORDER}")
        drafted = [n for n in SECTION_ORDER if n != "examples"]
        missing = [n for n in drafted if n not in names]
        if missing:
            raise ValueError(f"a rulebook holds every section of spec §8.5 but the examples; it lacks {missing}")
        examples = [s for s in self.sections if s.name == "examples"]
        if examples and (examples[0].text.strip() or examples[0].ir_nodes):
            raise ValueError("a drafted rulebook's examples section is empty: examples are the scenarios that passed "
                             "fuzzing, rendered by code (spec §8.5), never drafted")  # fmt: skip
        return self


class ScenarioPacket(Record):
    """One fuzz scenario's evidence packet, written by the scenario renderer and proposed as a "scenario" (spec I1;
    core.REQUIRED_GATES: schema and tests). Its documents are synthetic artifacts, committed only in the fuzz run's
    synthetic case (spec I4). The tests gate is the rendering check (spec §8.6 step 4: "Ask the atomic condition
    questions of the rendered packet. If judges don't recover the intended assignment, the rendering is at fault"),
    asked in the original wording. evidence_at is the instant the scenario's evidence stands at, and that every document
    is dated by it is ours: a document dated after it could not be among the evidence then."""

    contract: Ref
    scenario: str
    state_hash: str
    evidence_at: Instant
    documents: Sequence[SyntheticArtifactId]

    @model_validator(mode="after")
    def _documents(self) -> ScenarioPacket:
        if not self.documents:
            raise ValueError("a scenario's packet holds its documents")
        return self
