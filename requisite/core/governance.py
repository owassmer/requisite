"""Proposals, gates, approvals and the records of decisions (spec §4 "governance", §2.4, §3; amended by addendum §12)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, Literal

from pydantic import model_validator

from requisite.core.base import Day, Instant, Money, Probability, Rational, Record, Ref, exact_sum_is_one
from requisite.core.evidence import ArtifactId, ClaimId
from requisite.core.judgment import Assumption, Diagnosis, TimeWindow

# ---------------------------------------------------------------- proposals and gates (spec §3 I1, §4)

Gate = Literal["schema", "tests", "stability", "relevance", "human"]
ProposalKind = Literal[
    "question",
    "template",
    "proposition",
    "constraint",
    "node",
    "clause",
    "assumption",
    "claim",
    "paraphrase_set",
    "scenario",
    # Ours, each something an agent proposes that spec §4's list does not name:
    "decision_spec",  # the objective, horizon and loss an agent drafts for G1 (spec §2.4 G1; addendum §12 DecisionSpec)
    "frame",  # the wide flow's frame criteria, drawn from the approved objective (decisions, "Discovery finds ...")
    "contract",  # a whole filed contract read into the IR (spec §9.3, WP12)
    "objective",  # a wide run's objective an agent drafts for G1 (spec §2.4 G1; §7.2; decisions, "Discovery finds ...")
    "entity_model",  # a deep case's driver tree and liquidity profile an agent proposes for G1 (spec §7.1 step 1)
    "event_families",  # a wide run's event families an agent proposes (the approved objective: "defined by the run")
    "lead",  # a lead promoted into a deep case at G1 (spec §7.2 step 6: "Promote top leads into deep cases at gate G1")
    "observable",  # a candidate observable for construction (spec §8.1: "Each candidate becomes an Observable")
]
ProposalStatus = Literal["proposed", "validated", "active", "promoted", "rejected", "retired"]


class GateResult(Record):
    """One gate's verdict on a proposal (spec §4)."""

    gate: Gate
    passed: bool
    detail: str


class Proposal(Record):
    """The agents' only output (spec §3 I1, §4). `payload` is validated against its kind's record at the schema gate;
    `case` names the case it belongs to (ours: committed state is per case)."""

    id: str
    case: str
    kind: ProposalKind
    payload: Mapping[str, Any]
    author: str
    rationale: str
    status: ProposalStatus
    gates: Sequence[GateResult] = ()

    @model_validator(mode="after")
    def _status_follows_gates(self) -> Proposal:
        """A status says what the gates decided, so a proposal cannot claim one its gates did not give (spec I1: "A
        Proposal becomes active only after its required gates pass"). validated, active and promoted need every
        required gate run and passed, in order; rejected needs a failed gate; proposed has at most a pending human
        gate."""
        required = required_gates(self.kind, self.payload)
        ran = [g.gate for g in self.gates]
        expected = [*required, "human"] if self.status == "promoted" else list(required)
        if ran != expected[: len(ran)]:
            raise ValueError(f"gate results {ran} are not the required gates {list(required)} in order")
        passed = [g.passed for g in self.gates]
        if self.status in ("validated", "active"):
            if ran != list(required) or not all(passed):
                raise ValueError(f"a {self.status} proposal has passed every required gate: {list(required)}")
        elif self.status == "promoted":
            # spec §9: "promoted (in the library, after gate G4)": its required gates and then a human's G4 approval.
            if ran != [*required, "human"] or not all(passed) or "G4" not in self.gates[-1].detail:
                raise ValueError(f"a promoted proposal passed {list(required)} and then a G4 human approval")
        elif self.status == "rejected":
            if not self.gates or self.gates[-1].passed or not all(passed[:-1]):
                raise ValueError("a rejected proposal ends at the gate that failed")
        elif self.status == "proposed":
            if self.gates and (self.gates[-1].gate != "human" or self.gates[-1].passed or not all(passed[:-1])):
                raise ValueError("a proposed proposal has no gate results, or awaits its human gate")
        return self


# The gates a proposal must pass to become active, by kind (spec §3 I1: "schema, tests, stability, relevance and human,
# as applicable to its kind"). The spec gives no table; each entry is ours, with the passage it rests on:
# - every kind: schema (spec §9: "validated (schema and tests pass)").
# - tests where the kind is code or compiles to code: node (spec §9 ConsequenceFunction "code with tests"), clause and
#   contract (spec §5: the contract layer's validator is the IR type check and the three-valued evaluator), template
#   (spec §9 QuestionTemplate, ClauseTemplate with an example generator), claim (its text is its artifact's span: spec
#   §5 "Supports (with a span)"), proposition and constraint (spec §5: SAT consistency, acyclic conditioning), scenario
#   (feasible under the world constraints: addendum A09 §10.2), frame (its criteria run and freeze a frame), observable
#   (its event is the case's and construction's structure templates give an IR over it the evaluator accepts: task 004-1,
#   "Each candidate is a ContractIR the evaluator accepts"; spec §8.2 names the families).
# - relevance for a question: it reaches a decision (spec §2.2; spec §4 AtomicQuestion.reaches).
# - human where a gate G1-G4 is the human's: decision_spec, objective and entity_model (G1: "the objective, horizon and
#   loss definition"; spec §7.1 step 1: the entity model's parameters "a human confirms at gate G1").
#   G2 (contested diagnosis), G3 (contract acceptance) and G4 (template promotion) are approvals of committed state, not
#   of a proposal becoming active, and are recorded as Approvals.
# Stability is not a gate to become active: it is measured on active questions in the requisite loop (spec §2.3), and a
# split's sub-questions must be stable (spec §2.1 compound) through SPLIT_GATES.
REQUIRED_GATES: Mapping[str, tuple[Gate, ...]] = {
    "question": ("schema", "relevance"),
    "template": ("schema", "tests"),
    "proposition": ("schema", "tests"),
    "constraint": ("schema", "tests"),
    "node": ("schema", "tests"),
    "clause": ("schema", "tests"),
    "assumption": ("schema",),
    "claim": ("schema", "tests"),
    "paraphrase_set": ("schema",),
    "scenario": ("schema", "tests"),
    "decision_spec": ("schema", "human"),
    "frame": ("schema", "tests"),
    "contract": ("schema", "tests"),
    "objective": ("schema", "human"),
    "entity_model": ("schema", "human"),
    "event_families": ("schema", "tests"),
    "lead": ("schema", "human"),
    "observable": ("schema", "tests"),
}
# A question proposed to replace a parent (a split) passes stability (spec §2.1: "A proposed split yields stable
# sub-questions") and relevance (spec §2.2: "A split is admitted only if it raises relevance or is needed for
# stability").
SPLIT_GATES: tuple[Gate, ...] = ("schema", "stability", "relevance")


def required_gates(kind: str, payload: Mapping[str, Any]) -> tuple[Gate, ...]:
    if kind == "question" and payload.get("replaces"):
        return SPLIT_GATES
    return REQUIRED_GATES[kind]


# ---------------------------------------------------------------- human gates (spec §2.4)

HumanGate = Literal["G1", "G2", "G3", "G4"]

# Which human gate a kind's "human" gate is (spec §2.4). An approval names what it authorizes: one given at another gate
# authorizes something else, so it never passes this one (external evaluation, November 2: a G4 approval of the exact
# payload activated an objective that requires G1). G1 approves "the objective, horizon and loss definition" (spec
# §2.4): a wide run's objective, a deep case's decision spec, and its entity model ("a human confirms at gate G1", spec
# §7.1 step 1); a lead's promotion opens a deep case whose objective it sets, at G1 (spec §7.2 step 6).
HUMAN_GATE: Mapping[str, HumanGate] = {
    "decision_spec": "G1",
    "objective": "G1",
    "entity_model": "G1",
    "lead": "G1",
}
if set(HUMAN_GATE) != {k for k, gates in REQUIRED_GATES.items() if "human" in gates}:
    raise RuntimeError("every kind with a human gate names which one it is, and no other kind does")


class Approval(Record):
    """A human's decision at a gate (spec §2.4, §3 "Humans ... Approval"). It approves one version of its subject with
    one lineage (addendum A11 §11.2: "the versions approved by each human gate"; a change of lineage leaves the approval
    standing for the old lineage only)."""

    id: str
    gate: HumanGate
    subject: Ref
    lineage_hash: str
    approver: str
    at: Instant
    decision: Literal["approved", "approved_with_edits", "rejected"]
    reason: str
    scope: str


class Override(Record):
    """A human edit of committed state, stored as a correction on the specific record (spec §11 step 5: "An override
    without a reason is rejected")."""

    subject: Ref
    field: str
    old: Any
    new: Any
    approver: str
    at: Instant
    reason: str

    @model_validator(mode="after")
    def _reason(self) -> Override:
        if not self.reason.strip():
            raise ValueError("an override without a reason is rejected (spec §11)")
        return self


class AcceptedRisk(Record):
    """A scenario knowingly left unprotected, with its amount and horizon (addendum §12 AcceptanceRecord: "residual-risk
    amounts and horizons")."""

    scenario: str
    amount: Money
    horizon: TimeWindow
    description: str


class AcceptanceRecord(Record):
    """Immutable, written at decision time (spec §4, §11 step 1; amended by addendum §12 and A10 §11.1). The objective is
    version-pinned; the assessed domain, price policy and outstanding uncertainties bound what was accepted."""

    case: str
    decided_at: Instant
    objective: Ref
    decision: str
    chosen: str | None
    alternatives: Sequence[str]
    assessed_domain: str
    price_policy: Ref
    scenario_probabilities: Mapping[str, Probability]
    accepted_uncovered: Sequence[AcceptedRisk]
    outstanding_uncertainties: Sequence[str]
    budget: Money | None
    approvals: Sequence[str]
    approval_scope: str

    @model_validator(mode="after")
    def _probabilities(self) -> AcceptanceRecord:
        if self.scenario_probabilities:
            exact_sum_is_one(list(self.scenario_probabilities.values()), "scenario probabilities")
        return self


# ---------------------------------------------------------------- the decision (addendum §12 DecisionSpec)


class FloorPoint(Record):
    day: Day
    amount: Money


class DecisionSpec(Record):
    """The decision a case serves (addendum §12; spec §2.4 G1: "the objective, horizon and loss definition"). objective
    is the risk functional (spec §8.3 default and variants); tolerance is the economic tolerance within which actions
    count as equally good (addendum A02 §3.2, A03 §3.3); information lists what is observable before decision_time
    (A03 §3.6)."""

    id: str
    version: int
    entity: str
    decision_time: Instant
    horizon: TimeWindow
    loss_definition: str
    actions: Sequence[str]
    objective: Literal["liquidity_cvar", "expected_liquidity_shortfall", "terminal_residual"]
    alpha: Probability | None
    tolerance: Money
    liquidity_floor: Sequence[FloorPoint]
    currency: str
    information: Sequence[str]
    recourse: str

    @model_validator(mode="after")
    def _alpha(self) -> DecisionSpec:
        if (self.objective == "liquidity_cvar") != (self.alpha is not None):
            raise ValueError("a CVaR objective states its alpha, and only it does")
        if self.alpha is not None and not self.alpha < 1:
            raise ValueError("alpha lies in [0, 1)")
        # The acceptable actions are those within the tolerance of optimal (A02 §3.2; A03 §3.3): a negative one admits
        # none, not even the optimum, so it is no tolerance (ours; QuestionDiagnostic holds the same of its copy).
        if self.tolerance < 0:
            raise ValueError("the tolerance is not negative: actions within it of optimal count as equally good")
        return self


class RefinementWitness(Record):
    """Feasible states that share the current representation but call for divergent epsilon-optimal actions or show a
    material value discrepancy (addendum §12 RefinementWitness; A02 §3.2): a stable coarse question that must still be
    refined. value_discrepancy is the difference in decision value between the states under the current action, in the
    decision's currency. tolerance is the decision's declared economic tolerance (DecisionSpec.tolerance): the acceptable
    actions are those within it of optimal, and a discrepancy is material when it exceeds it (A02 §3.2)."""

    question: Ref
    decision: Ref
    tolerance: Money
    shared_answer: str
    states: Sequence[str]
    acceptable_actions: Sequence[Sequence[str]]
    value_discrepancy: Money | None
    nodes: Sequence[str]

    @model_validator(mode="after")
    def _witness(self) -> RefinementWitness:
        if len(set(self.states)) < 2 or len(self.states) != len(self.acceptable_actions):
            raise ValueError("a witness is two or more distinct feasible states, each with its acceptable actions")
        if len(set(self.states)) != len(self.states):
            raise ValueError("a witness's states are distinct: one state has one acceptable set")
        sets = [set(a) for a in self.acceptable_actions]
        divergent = not set.intersection(*sets)
        material = self.value_discrepancy is not None and self.value_discrepancy > self.tolerance
        if not (divergent or material):
            raise ValueError("a witness's states share no acceptable action, or differ materially in value")
        return self


# ---------------------------------------------------------------- wide mode (addendum A07; decisions, "two-phase")


class Stratum(Record):
    name: str
    frame_size: int
    sample_size: int

    @model_validator(mode="after")
    def _sizes(self) -> Stratum:
        if not 0 <= self.sample_size <= self.frame_size:
            raise ValueError("a stratum's sample is part of its frame")
        return self


class SubStratum(Record):
    """A phase-2 stratum nested in a phase-1 stratum, formed from phase-1 model labels (decisions, "two-phase"); its
    phase-1 count and allocation are realized when phase 1 is drawn."""

    name: str
    parent: str
    rule: str
    phase1_count: int
    sample_size: int

    @model_validator(mode="after")
    def _sizes(self) -> SubStratum:
        if not 0 <= self.sample_size <= self.phase1_count:
            raise ValueError("a phase-2 stratum's gold sample is part of its phase-1 items")
        return self


class SamplingDesign(Record):
    """The declared estimand and sampling design of a wide run (addendum A07 §8.1, §12). frame_hash and frame_size
    identify the frozen frame; unit and estimand say what is estimated (the unit is the item: decisions, "two-phase");
    phase1 and phase2 are the strata and sample sizes; allocation, seed and estimator make the draw and the inference
    reproducible; predictor pins the templates and models that produced f; gold_policy states adjudication and the
    unresolved-label category (A07 §8.1)."""

    id: str
    version: int
    frame: Ref
    frame_hash: str
    frame_size: int
    unit: Literal["item"]
    estimand: str
    filter_policy: str
    dedup_policy: str
    sample: Literal["census_then_gold", "two_phase_stratified_srswor", "independent_two_sample"]
    phase1: Sequence[Stratum]
    phase2: Sequence[SubStratum]
    allocation: Ref
    seed: int
    estimator: Ref
    predictor: Sequence[Ref]
    gold_policy: str

    @model_validator(mode="after")
    def _strata(self) -> SamplingDesign:
        if sum(s.frame_size for s in self.phase1) != self.frame_size:
            raise ValueError("phase-1 strata partition the frame")
        if self.sample == "independent_two_sample":
            # Two samples drawn independently from the frame (A07 §8.2: "an unlabeled sample and a separate labeled
            # correction sample"): the labelled one is not a subsample of the other, so no nesting applies; each of its
            # strata is a stratum of the frame.
            frame = {s.name: s for s in self.phase1}
            if any(s.parent not in frame for s in self.phase2):
                raise ValueError("an independent labelled sample's strata are strata of the frame")
            for stratum in self.phase1:
                labelled = [s for s in self.phase2 if s.parent == stratum.name]
                if stratum.frame_size and not any(s.sample_size for s in labelled):
                    raise ValueError(f"frame stratum {stratum.name} has items and no labelled sample: their inclusion "
                                     "is zero (A07 §8.2: pi_i > 0)")  # fmt: skip
                if any(s.phase1_count != stratum.frame_size for s in labelled):
                    raise ValueError("an independent labelled stratum is drawn from its whole frame stratum")
            return self
        parents = {s.name: s for s in self.phase1}
        if any(s.parent not in parents for s in self.phase2):
            raise ValueError("every phase-2 stratum is nested in a phase-1 stratum")
        for name, stratum in parents.items():
            children = [s for s in self.phase2 if s.parent == name]
            if children and sum(s.phase1_count for s in children) != stratum.sample_size:
                raise ValueError(f"the phase-2 strata of {name} partition its {stratum.sample_size} phase-1 items")
        if self.sample == "census_then_gold" and any(s.sample_size != s.frame_size for s in self.phase1):
            raise ValueError("a census reads every item of the frame")
        # Every unit has a positive chance of a gold label (addendum A07 §8.2: "inclusion probabilities pi_i > 0"):
        # every nonempty phase-1 stratum is sampled, its phase-1 sample is partitioned into phase-2 strata, and each
        # nonempty phase-2 stratum gets at least min(2, its items) gold labels (decisions: at least 2, or the whole
        # stratum when it is that small).
        for stratum in self.phase1:
            if stratum.frame_size and not stratum.sample_size:
                raise ValueError(f"phase-1 stratum {stratum.name} has items and no sample: their inclusion is zero")
            if stratum.sample_size and not [s for s in self.phase2 if s.parent == stratum.name]:
                raise ValueError(f"phase-1 stratum {stratum.name}'s sample has no phase-2 strata to draw gold from")
        for sub in self.phase2:
            if sub.sample_size < min(2, sub.phase1_count):
                raise ValueError(f"phase-2 stratum {sub.name} draws {sub.sample_size} of {sub.phase1_count}: at least "
                                 "min(2, its items)")  # fmt: skip
        return self


class DrawnItem(Record):
    """One item a sample drew, with its stratum and its inclusion probability, exact (A07 §8.2: pi_i > 0)."""

    item: str
    stratum: str
    inclusion_prob: Probability

    @model_validator(mode="after")
    def _positive(self) -> DrawnItem:
        if self.inclusion_prob == 0:
            raise ValueError("a drawn item has a positive inclusion probability")
        return self


class SampleDraw(Record):
    """A phase of a wide run's sample as drawn: which items, from which strata, with what chance, under which design and
    seed, so the draw is reproducible and the estimator reads the probabilities it assumes (addendum A07 §8.4: "Store
    the design, strata, cluster IDs and any required joint-inclusion or replicate-weight information"). Each item's
    inclusion_prob is its chance in this phase as drawn: in phase 1, its chance of selection from the frame; in phase 2
    of a two-phase design (decisions, "two-phase": standard survey sampling, Särndal, Swensson and Wretman 1992, ch. 9,
    ours and not the addendum's), its chance of a gold label given the realized phase-1 sample. The two-phase estimator
    uses the two separately; their product is not the item's marginal chance of a gold label when phase-2 strata depend
    on the phase-1 sample, and no record calls it that. In an independent two-sample design (A07 §8.2: "an unlabeled
    sample and a separate labeled correction sample"), phase 2's is the labelled sample's own chance of drawing the item
    from the frame."""

    design: Ref
    phase: Literal[1, 2]
    seed: int
    items: Sequence[DrawnItem]

    @model_validator(mode="after")
    def _items(self) -> SampleDraw:
        ids = [i.item for i in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("a draw lists each item once")
        return self


class GoldLabel(Record):
    """A label of one sampled item (spec §4; amended by addendum A07), by the labeller named: a human, or the
    coordinator (a model of another family than the judges whose answers the gold sample measures) at the human's
    direction and under his review (decisions, F4's gold, November 2), the labeller always named so that what the
    estimate rests on is stated. inclusion_prob is the item's phase-2 chance
    as its SampleDraw records it, exact (in a two-phase design, given the realized phase-1 sample); `outcome` keeps an explicit unresolved category, never dropped or read as zero (A07 §8.1).
    `blind`: the labeller did not see any model's answer (ours: a label that saw the model's answer measures agreement
    with it, not truth)."""

    item: str
    question: Ref
    design: Ref
    stratum: str
    inclusion_prob: Probability
    outcome: Literal["labeled", "unresolved"]
    label: str | None
    labeler: str
    labeled_at: Instant
    blind: bool

    @model_validator(mode="after")
    def _label(self) -> GoldLabel:
        if (self.outcome == "labeled") != (self.label is not None):
            raise ValueError("a labelled item has a label, and an unresolved one has none")
        if self.inclusion_prob == 0:
            raise ValueError("a sampled item has a positive inclusion probability")
        return self


# ---------------------------------------------------------------- recomputation lineage (addendum A11)


class InputReceipt(Record):
    name: str
    value_hash: str
    lineage_hash: str


class DerivationReceipt(Record):
    """How a node's value was derived (addendum A11 §11.2, §12): the value hash, used for early cutoff, kept apart from
    the lineage hash, which changes with any input's provenance, policy, gold label, provider version or dependency edge
    even when the value does not."""

    node: Ref
    value_hash: str
    lineage_hash: str
    inputs: Sequence[InputReceipt]
    computed_at: Instant


# ---------------------------------------------------------------- wide runs (spec §7.2-7.3; decisions, "Discovery finds ...")


class WideObjective(Record):
    """What a wide run looks for, approved at G1 (spec §2.4: "the objective, horizon and loss definition"). The flow
    derives its frame from this and from nothing else, so which corpus and which leads it reaches are its findings
    (decisions, "Discovery finds what it analyzes"). population: whose exposure; events: the class of pending external
    change; horizon: the window the exposure matters over; loss_definition: what counts as the loss; estimand: what the
    run's prevalences estimate, in its unit (A07 §8.1: "The target is not implicit")."""

    id: str
    version: int
    population: str
    events: str
    horizon: TimeWindow
    loss_definition: str
    estimand: str
    unit: Literal["item"]


DocumentType = Literal["Proposed Rule", "Rule", "Notice"]  # regulations.gov's document types that state a change
# Which dockets a frame reaches, by the state of their measure at as_of (requisite/evidence/frame.py states each test):
# "open", a comment period is running; "closed_pending_final", every comment period has closed and the measure is still
# pending as the objective's events define pending (proposed but not final, or final but not yet in effect: a final rule
# whose effective date, as the Federal Register states it, is after as_of); "either", one or the other. (Coordinator,
# November 2, from the external evaluation: the earlier test dropped every docket whose proposal a final rule followed,
# which narrowed the approved objective's events to "proposed but not final".)
CommentStatus = Literal["open", "closed_pending_final", "either"]


class FrameCriteria(Record):
    """How the flow reaches its frame in the public record, derived from an approved WideObjective. The corpus is the
    public record of affected parties, regulatory comment dockets first, through the regulations.gov API (spec §7.3);
    the frame is snapshot into immutable artifacts and split into items (spec §7.2 step 1). The criteria are ours, the
    regulations.gov search's own terms for what to reach: documents of the given types, posted in the window, matching
    any search term, with their dockets' public comments; comment_status narrows to open comment periods, closed ones
    whose rule is still pending, or either, since the objective's events are pending changes (spec §12: "F2 and F4 use
    forward, currently unresolved events. The factory queries regulations.gov for open comment periods"; the choice of
    docket is the flow's, decisions, "Discovery finds what it analyzes"); min_comments excludes dockets too small to
    estimate a prevalence over. Code runs the criteria at `as_of`, and the frame is whatever they return, frozen."""

    id: str
    version: int
    objective: Ref
    source: Literal["regulations.gov"]
    as_of: Instant
    search_terms: Sequence[str]
    document_types: Sequence[DocumentType]
    posted_from: Day
    posted_to: Day | None
    agencies: Sequence[str]
    comment_status: CommentStatus
    min_comments: int

    @model_validator(mode="after")
    def _criteria(self) -> FrameCriteria:
        if not self.search_terms or not self.document_types:
            raise ValueError("a frame names its search terms and the document types it reaches")
        if self.posted_to is not None and self.posted_to < self.posted_from:
            raise ValueError("a frame's posting window starts before it ends")
        if self.min_comments < 1:
            raise ValueError("a frame's dockets have at least one comment")
        return self


class FrameSearch(Record):
    """One search the frame ran: its request (the API path and parameters, never the key) and how many documents it
    returned. artifacts: the responses it returned, every page and every continuation of the listing, in order (spec I2:
    the frame's evidence is held as content-addressed artifacts, as an item's is; ours: a count recomputes from them)."""

    term: str
    request: str
    documents: int
    artifacts: Sequence[ArtifactId]


class FrameDocket(Record):
    """A docket the searches reached, kept in the frame or dropped, with why (its comment status, its comment count).
    artifacts: the listings of its documents and of its comments that the decision rests on, in order (spec I2; ours: the
    status and the count recompute from them).

    comments: the public comments posted by as_of, or None for a docket its comment status drops before its comments
    are listed. Its status is decided from its documents alone, and a dropped docket's count decides nothing, so its
    comments are not retrieved; the count is then not known and is not stated, never zero (spec I6). A kept docket
    always has its count (the frame's items are its comments). (Coordinator, November 1: listing every reached docket's
    comments costs more requests than the source's hourly limit allows in a run, most of them for dockets the status
    test drops.)"""

    docket: str
    kept: bool
    reason: str
    comments: int | None
    artifacts: Sequence[ArtifactId]

    @model_validator(mode="after")
    def _counted(self) -> FrameDocket:
        if self.kept and self.comments is None:
            raise ValueError("a kept docket's comments are listed and counted: they are the frame's items")
        return self


class FrameItem(Record):
    """One item of the frame. withdrawn: the source lists it as withdrawn; it stays in the frame, since it was public at
    as_of (task 002-1, the coordinator's reasoning: a withdrawn comment remains in the public record, marked)."""

    id: str
    docket: str
    posted_at: Instant
    withdrawn: bool


class ItemMeasure(Record):
    """The measure an item is asked about (coordinator, November 2, from the external evaluation: a docket is not one
    measure): the document the item addresses as its source states it (core.Item.addresses), or its docket's own
    measure where the source states none, said so in the statement. statement: the document's title, agency, type, id
    and posting date, and its status as of the frame's as_of (proposed, final not yet in effect, in effect, or its
    effective date not known), as the exposure question binds it. artifacts: the responses the statement is read from
    (the frame's listing of its docket's documents or the document's own record, and the Federal Register's statement of
    the deciding rule's effective date; spec I2). Committed once, when the item is read, and read thereafter."""

    item: str
    document: str
    statement: str
    artifacts: Sequence[ArtifactId]


class CostStatement(Record):
    """One statement a measure's own text makes of what it requires someone to pay, or of what it costs or saves those
    it binds (Owen, November 2: question 2 is asked over "the agency's own cost statements"). quote: the text's words,
    exactly as the rule's published text has them (provenance, checked as text against that text). The rest is the
    reader's reading of those words into a fixed form (an agent's reading, never a pattern matched by code): payer, who
    must pay or bears the cost, as the text names them; pays_for, what for; direction, whether it means paying more,
    less, the same, or the text does not say; amount, currency and per, the sum as an exact decimal in units of the
    currency and its basis ("per petition", "per year, annualized"), where the words state one; scope, whether that sum
    is what all who pay it pay together ("all_affected"), what each one pays ("per_entity"), or the words do not say
    ("not_stated", as for a statement with no sum); affected, how many pay it, with affected_quote, the text's words that
    state that number (checked as text, as the quote is), where the text states it (Owen, November 2: "imperative to
    scale actual impact": a total is never one business's cost, nor the reverse).

    The fixed form is ours (coordinator, November 2), shaped by what the exposure question asks of it: whether a measure
    would change what a business pays. So direction is closed to the four answers that question can use (more, less,
    unchanged, and not_stated, which keeps a statement that does not say from being read as either); and an amount is a
    sum of money only with its currency (an amount with no currency is not a sum anyone pays, and a currency with no
    amount states nothing), its currency written as an ISO 4217 code is (three capital letters; whether the letters
    name a currency in the standard's list is not checked here)."""

    quote: str
    payer: str
    pays_for: str
    direction: Literal["more", "less", "unchanged", "not_stated"]
    amount: Money | None = None
    currency: str | None = None
    per: str | None = None
    scope: Literal["all_affected", "per_entity", "not_stated"]
    affected: int | None = None
    affected_quote: str | None = None

    @model_validator(mode="after")
    def _read(self) -> CostStatement:
        if not self.quote.strip() or not self.payer.strip() or not self.pays_for.strip():
            raise ValueError("a cost statement quotes the text and says who pays for what")
        if (self.amount is None) != (self.currency is None):
            raise ValueError("an amount states its currency, and a currency belongs to an amount")
        code = self.currency
        if code is not None and not (len(code) == 3 and code.isalpha() and code.isupper()):
            raise ValueError("a currency is written as an ISO 4217 code: three capital letters")
        if self.amount is None and self.scope != "not_stated":
            raise ValueError("a scope is the scope of an amount: a statement with no amount has scope not_stated")
        if (self.affected is None) != (self.affected_quote is None):
            raise ValueError("a count of those who pay is given with the words that state it, and only with them")
        if self.affected is not None and self.affected < 0:  # a count (a text may state that none pay it): ours
            raise ValueError("a count of those who pay is not negative")
        return self


class MeasureText(Record):
    """What a measure is, as its own published text states it (Owen, November 2, after the gold session showed only a
    title: "would its direct legal effect change what a business pays" cannot be answered from a title). One per
    document and reader version, shared by every item that addresses the document. summary: the agency's own summary of the measure, verbatim (the
    Federal Register's abstract of the document, the SUMMARY its published text opens with), or None with summary_why
    where the source states none. statements: the measure's own statements of who pays what (CostStatement), read from
    its whole published text in consecutive parts, in the order the text makes them; none_why: where the text makes no
    such statement, the reader's account of what it does say of costs (a measure with no cost statement is read, never
    assumed). artifacts: the source's record of the document (its abstract) and its full text (spec I2). readings: the
    agent runs that read the text, one per part (requisite/agents). reader: the reader and its version that read the
    text (its prompt and answer type); a MeasureText is committed once per document and reader. fr_number: the Federal Register document it was read
    from, or None where it was read from regulations.gov (its record states no number, or one the Federal Register does
    not serve)."""

    document: str
    reader: str
    fr_number: str | None
    summary: str | None
    summary_why: str | None = None
    statements: Sequence[CostStatement]
    none_why: str | None = None
    artifacts: Sequence[ArtifactId]
    readings: Sequence[str]

    @model_validator(mode="after")
    def _stated(self) -> MeasureText:
        if (self.summary is None) == (self.summary_why is None):
            raise ValueError("a measure's summary is the source's, or why the source states none, and not both")
        if self.summary is not None and not self.summary.strip():
            raise ValueError("a summary has words; where the source states none, summary_why says so")
        if bool(self.statements) == (self.none_why is not None):
            raise ValueError("a measure's text states who pays what, or the reader says what it says of costs instead")
        if self.none_why is not None and not self.none_why.strip():
            raise ValueError("where a measure's text states no cost, the reader's account of what it says is not empty")
        if not self.artifacts:
            raise ValueError("a measure's text rests on the records it was read from (spec I2)")
        if not self.readings:
            raise ValueError("a measure's text was read: its readings name the agent runs")
        return self


class Frame(Record):
    """A wide run's frozen frame: what its criteria searched, which dockets they reached and kept, and every item of the
    kept dockets posted by as_of (spec §7.2 step 1: a snapshot of the corpus, split into items). Its content hash is the
    frame_hash a SamplingDesign names (addendum A07 §8.1), so the frame a sample was drawn from is the frame recorded.
    The API serves its record as it stands when asked, so a frame states what was known at as_of only when everything in
    it was retrieved by as_of: prospective when its retrieval ended at or before as_of (spec I3: observed claims come from
    artifacts observed by the evaluation time), and otherwise a retrospective reconstruction, whose fields that can
    change after posting (a comment period's end, a withdrawal) are as served after as_of (addendum A11 §11.3: "Do not
    classify a historical reconstruction as clean prospective evaluation")."""

    id: str
    criteria: Ref
    as_of: Instant
    retrieval_started: Instant
    retrieval_ended: Instant
    mode: Literal["prospective", "retrospective_reconstruction"]
    searches: Sequence[FrameSearch]
    dockets: Sequence[FrameDocket]
    items: Sequence[FrameItem]

    @model_validator(mode="after")
    def _items(self) -> Frame:
        if self.retrieval_started > self.retrieval_ended:
            raise ValueError("a frame's retrieval starts before it ends")
        prospective = self.retrieval_ended <= self.as_of
        if (self.mode == "prospective") != prospective:
            kind = "prospective" if prospective else "a retrospective reconstruction"
            raise ValueError(f"a frame whose retrieval ended {'by' if prospective else 'after'} its as_of is {kind}")
        kept = {d.docket for d in self.dockets if d.kept}
        if any(i.docket not in kept for i in self.items):
            raise ValueError("a frame's items come from the dockets it kept")
        if any(i.posted_at > self.as_of for i in self.items):
            raise ValueError("a frame holds the items posted by its as_of (spec I3)")
        ids = [i.id for i in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("a frame lists each item once")
        return self


# ---------------------------------------------------------------- a wide run's results (spec §7.2 steps 5-6)


class EventFamily(Record):
    """One event family of a wide run. The objective approved at G1 defines it: "An event family is a group of events
    that share a cost channel (labor or input) and a mechanism ... Families are defined by the run from what it finds."
    It holds the measures it groups, each the document a phase-1 item addresses (requisite/evidence/frame.measures); an
    item's family is its measure's (coordinator, November 2, from the external evaluation: a family per docket assumed a
    docket is one measure). The channel is free text here, since which channels a run's loss counts is its objective's
    (an objective over energy costs names energy); the agent is given them."""

    id: str
    name: str
    channel: str  # the cost channel, among those the run's objective defines its loss over (for F4's: labor, inputs)
    mechanism: str
    measures: Sequence[str]

    @model_validator(mode="after")
    def _measures(self) -> EventFamily:
        if not self.measures or len(set(self.measures)) != len(self.measures):
            raise ValueError("a family groups one or more measures, each once")
        if not self.channel.strip() or not self.mechanism.strip():
            raise ValueError("a family states its cost channel and its mechanism")
        return self


class EventFamilies(Record):
    """A wide run's event families over the measures its phase-1 items address, proposed by an agent (kind
    "event_families"; spec I1). Every such measure is in exactly one family, so the estimate can be reported for each
    family (the objective: "reported overall and separately for each event family"); the proposal's tests gate checks
    this against the read items."""

    frame: str
    families: Sequence[EventFamily]

    @model_validator(mode="after")
    def _partition(self) -> EventFamilies:
        ids = [f.id for f in self.families]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("a run's families have distinct ids, and there is at least one")
        measures = [m for f in self.families for m in f.measures]
        if len(set(measures)) != len(measures):
            raise ValueError("a measure is in one family only")
        return self


Pieces = Sequence[tuple[float, float]]  # a set of shares: closed intervals in increasing order, possibly none


Rays = Sequence[tuple[float | None, float | None]]  # a set of reals: closed intervals, None an unbounded end


class RatioRead(Record):
    """One completion's estimate of a share (core.WideEstimate): the point estimate (None where the denominator's
    estimate is 0), the score set to read, within [0, 1], and Fieller's set as it is over the real line, an unbounded
    end None (decisions, "F4's estimand is a ratio": "Fieller's set is reported raw (two rays or the whole line where
    theta_den is not distinguishable from zero)")."""

    estimate: float | None
    interval: Pieces
    interval_kind: str
    fieller: Rays
    fieller_kind: str

    @model_validator(mode="after")
    def _ordered(self) -> RatioRead:
        ends = [x for lo, hi in self.interval for x in (lo, hi)]
        if any(not 0.0 <= x <= 1.0 for x in ends) or ends != sorted(ends):
            raise ValueError("a share's score set is ordered closed intervals within [0, 1]")
        return self


class ScopeEstimate(Record):
    """The share for one scope ("overall", or a family's id) under the two completions that bound the unresolved gold
    labels (decisions, "F4 from the labels to the leads": never read as 0): `read` runs from the lowest completion's
    lower bound to the highest's upper bound, None where either set is empty, and then `reason` says why."""

    scope: str
    lowest: RatioRead
    highest: RatioRead
    read: tuple[float, float] | None
    reason: str

    @model_validator(mode="after")
    def _read(self) -> ScopeEstimate:
        empty = not self.lowest.interval or not self.highest.interval
        hull = None if empty else (self.lowest.interval[0][0], self.highest.interval[-1][1])
        if self.read != hull:
            raise ValueError("a scope's read runs from the lowest completion's lower bound to the highest completion's "
                             "upper bound, None exactly where either set is empty")  # fmt: skip
        return self


class WideEstimate(Record):
    """A wide run's estimate (spec §7.2 step 5; §7.4): the share the objective defines, overall and per family, under the
    run's two-phase design, from its committed phase-1 judgments and gold labels. The scope is the frame's items with
    readable text, and how many had none is stated (decisions, "F4 from the labels to the leads"). Every number
    recomputes from the records it names by their content hashes (the draws, the families' proposal, the gold labels,
    the phase-1 judgments) (addendum A11 §11.2: a versioned derivation)."""

    objective: Ref
    frame: str
    design: Ref
    draws: tuple[str, str]  # the content hashes of the phase-1 draw and the gold draw it reads
    families: str
    gold_labels: Sequence[str]  # the content hashes of the gold labels it reads
    judgments: Sequence[str]  # the content hashes of the phase-1 judgments its predictions are read from
    level: float
    interval_method: str
    phase1_items: int
    unreadable_items: int
    gold_items: int
    unresolved_gold_labels: int
    scopes: Sequence[ScopeEstimate]

    @model_validator(mode="after")
    def _scopes(self) -> WideEstimate:
        names = [x.scope for x in self.scopes]
        if not names or names[0] != "overall" or len(set(names)) != len(names):
            raise ValueError("an estimate reports the overall share first, then each family once")
        return self


class QuantifiedImpact(Record):
    """A cost impact a lead's filing states, extracted by an agent as a claim (spec §7.2: "a number with units,
    extracted by an agent as a Claim proposal"). claim: the filer's words that state it, a span of the filing's text
    (code finds the quote there, so the words are the filer's). amount, currency, period: the amount of money those
    words state, as the impact extractor read it from them (an agent's reading, shown beside the quote, never a pattern
    matched by code: decisions, F4 from the labels to the leads). annual: the amount per year where the period gives one
    (code: a month's x 12; a one-time or other amount has none)."""

    claim: ClaimId
    amount: Rational
    currency: str
    period: Literal["year", "month", "one_time", "other"]
    annual: Rational | None


class Lead(Record):
    """A business a wide run found (spec §7.2 step 5: "ranked leads"): a phase-1 item in the estimand's numerator, by its
    gold labels where it has them, otherwise by the phase-1 judge (P(in the population) x P(exposed) >= 0.5), with the
    item templates' answers (each the judge's most probable answer, not_established on a tie: ours), its phase-1
    prediction whichever the basis (the one the estimate used: a question the judge refused takes its stratum's mean,
    decisions, "F4 from the labels to the leads"), and its quantified impact, ranked by the rule
    fixed before any lead was seen (decisions, "F4 from the labels to the leads")."""

    rank: int
    tier: Literal[1, 2, 3]
    item: str
    docket: str
    measure: str  # the document the item addresses, which its questions were asked about (core.Item.addresses)
    family: str
    submitter: str | None
    basis: Literal["gold", "judged"]
    probability: float  # the phase-1 prediction P(in the population) x P(exposed) the estimate used: judged, not exact
    answers: Mapping[str, str]
    judgments: Sequence[str]
    impact: Sequence[QuantifiedImpact]

    @model_validator(mode="after")
    def _probability(self) -> Lead:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("a lead's judged probability lies in [0, 1]")
        if self.basis == "judged" and self.probability < 0.5:
            raise ValueError("a lead by the judge alone has probability at least 0.5 (decisions, F4 to the leads)")
        return self


class LeadList(Record):
    """A wide run's ranked leads, from its estimate's records."""

    estimate: str
    leads: Sequence[Lead]

    @model_validator(mode="after")
    def _ranked(self) -> LeadList:
        if [x.rank for x in self.leads] != list(range(1, len(self.leads) + 1)):
            raise ValueError("leads are ranked 1, 2, ... in order")
        return self


class LeadPromotion(Record):
    """A lead proposed for promotion into a deep case (kind "lead"), decided by a human at G1 (spec §7.2 step 6)."""

    leads: str
    lead: Lead
    case: str


# ---------------------------------------------------------------- agent runs (spec §13 WP15)


class AgentRun(Record):
    """One agent call, its trace (spec §13 WP15: "each run leaves a trace of Proposals and gate results"). `response` is
    the cassette key of the recorded call, so the run's input, prompt and answer are kept whole; `proposals` are the ids
    of the Proposals code made from its answer, whose gate results the store holds."""

    id: str
    role: str
    model: str
    effort: str
    input_hash: str
    output_hash: str
    response: str
    proposals: Sequence[str]
    at: Instant


# ---------------------------------------------------------------- deep discovery (spec §7.1)


class Parameter(Record):
    """A number in an entity model, sourced or assumed, never neither (spec §7.1 step 1: "Each parameter is either sourced
    (a claim) or an assumption"). value is exact money or a count; unit says which; claims cite the evidence for a
    sourced value, and an assumed value carries its Assumption (I5: it enters only as a labelled assumption)."""

    name: str
    value: Money
    unit: str
    claims: Sequence[ClaimId] = ()
    assumption: Assumption | None = None

    @model_validator(mode="after")
    def _basis(self) -> Parameter:
        if bool(self.claims) == (self.assumption is not None):
            raise ValueError("a parameter is sourced by claims or assumed, exactly one")
        return self


class Driver(Record):
    """One line of the P&L as quantity times price per period (spec §7.1 step 1: "revenue and cost lines decomposed into
    quantity x price drivers"). period is the period the quantity and price are stated per, from a closed set so that
    code converts every driver into the monthly cash flows materiality computes over the horizon (ours: spec §7.1 names
    no period)."""

    id: str
    line: Literal["revenue", "cost"]
    name: str
    quantity: Parameter
    price: Parameter
    period: Literal["month", "quarter", "year"]


class LiquidityProfile(Record):
    """The entity's cash position and its timing (spec §7.1 step 1: "cash, credit, payment terms, seasonality"), each a
    parameter, sourced or assumed. seasonality is twelve calendar-month factors on the drivers' monthly amounts,
    January first, averaging exactly one, so that seasonality moves cash between months without changing a driver's
    annual total as its quantity and price state it (ours: spec §7.1 names seasonality and no form for it)."""

    cash: Parameter
    credit_line: Parameter
    receivable_days: Parameter
    payable_days: Parameter
    seasonality: Sequence[Parameter]

    @model_validator(mode="after")
    def _seasonality(self) -> LiquidityProfile:
        if len(self.seasonality) != 12 or sum((Fraction(p.value) for p in self.seasonality), Fraction(0)) != 12:
            raise ValueError("seasonality is twelve monthly factors averaging exactly one")
        return self


class EntityModel(Record):
    """A deep case's model of the business (spec §7.1 step 1), proposed by an agent and confirmed by a human at G1:
    its driver tree, liquidity profile, planning horizon and the loss definition the client cares about."""

    id: str
    version: int
    entity: str
    drivers: Sequence[Driver]
    liquidity: LiquidityProfile
    horizon: TimeWindow
    loss_definition: str
    currency: str

    @model_validator(mode="after")
    def _drivers(self) -> EntityModel:
        ids = [d.id for d in self.drivers]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("an entity model has drivers with distinct ids")
        return self


class DatedImpact(Record):
    """One month's cash-flow delta of an exposure, an exact interval (decisions, "Numbers"; A04 §4.1: interval-valued
    consequences, never a band's midpoint)."""

    month: Day
    low: Rational
    high: Rational

    @model_validator(mode="after")
    def _ordered(self) -> DatedImpact:
        if self.low > self.high:
            raise ValueError("a dated impact is an interval [low, high]")
        return self


class QuestionDiagnostic(Record):
    """A question behind a register row, with its stability and relevance diagnostics (spec §7.5: "stability and
    relevance diagnostics"; spec §2.1-2.2): its resolved judgment, its scores and primary diagnosis, and its tie-robust
    regret for the named decision at its tolerance (addendum A03 §3.3), informational where that regret is within the
    tolerance, for that decision under the tested assumptions (the ids of the assumptions and model it was tested
    under) and that tolerance only (A02-A03 §3.4: "non-material to the named decision under the tested assumptions and
    tolerance, not globally irrelevant or removable from the evidence record")."""

    question: Ref
    resolved: str
    s_arg: float
    s_prob: float
    primary_diagnosis: Diagnosis
    decision: Ref
    tolerance: Money
    assumptions: Sequence[str]
    regret: Rational
    informational: bool

    @model_validator(mode="after")
    def _scoped(self) -> QuestionDiagnostic:
        if not (0 <= self.s_arg <= 1 and 0 <= self.s_prob <= 1):
            raise ValueError("stability scores lie in [0, 1] (A01 §2.1)")
        if self.regret < 0 or self.tolerance < 0:
            raise ValueError("min-max regret and the tolerance are non-negative (A03 §3.3: a regret is J - inf J)")
        if self.informational != (self.regret <= Fraction(self.tolerance)):
            raise ValueError("informational means regret within the decision's tolerance, and only that (A03 §3.3)")
        return self


class RegisterRow(Record):
    """One driver-event exposure in the register (spec §7.5: "Each row holds the entity, driver, event (a proposition
    id), mechanism, direction, magnitude band and dated cash impact. It adds the probability source, expected and tail
    impact, liquidity-breach flag, hedgeability screen and stability and relevance diagnostics, with links to every
    claim and judgment"; spec §13 WP11a: "Every register row traces to claims, judgments and nodes").

    event is a proposition id; direction, mechanism and magnitude_band are spec §7.1 step 3's typed answers, with
    not_established kept. cash_impact is computed by code from the magnitude band through the driver tree, month by
    month, as exact intervals (decisions, "Numbers"), never a band's midpoint (A04 §4.1: "Keep interval-valued
    consequences, or attach an approved distribution/scenario expansion"). baseline names the counterfactual the impacts
    are measured against, standalone or marginal to the named exposures, since such impacts are never summed into a
    portfolio loss without checking overlap and dependence (A06 §6.1). probability names the ProbabilityEstimate of the
    event. breach is three-valued (I6). The traces name the observed claims (a real case's, never synthetic: spec I4),
    judgments and computation nodes (vertex ids) every number rests on."""

    entity: str
    driver: str
    event: str
    direction: Literal["increase", "decrease", "none", "not_established"]
    mechanism: Literal["cost", "revenue", "timing", "capacity", "not_established"]
    magnitude_band: str
    cash_impact: Sequence[DatedImpact]
    baseline: str
    probability: str
    expected_impact: tuple[Rational, Rational]
    tail_impact: tuple[Rational, Rational]
    breach: Literal["true", "false", "unknown"]
    hedgeable: Literal["true", "false", "unknown"]
    diagnostics: Sequence[QuestionDiagnostic]
    claims: Sequence[ClaimId]
    judgments: Sequence[str]
    nodes: Sequence[str]

    @model_validator(mode="after")
    def _traced(self) -> RegisterRow:
        for name, (lo, hi) in (("expected", self.expected_impact), ("tail", self.tail_impact)):
            if lo > hi:
                raise ValueError(f"the {name} impact is an interval [low, high]")
        if not self.claims or not self.judgments or not self.nodes:
            raise ValueError("a register row traces to the claims, judgments and computation nodes its numbers rest on")
        if not self.diagnostics:
            raise ValueError("a register row carries the stability and relevance diagnostics of its questions")
        return self
