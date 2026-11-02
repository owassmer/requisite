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
#   (feasible under the world constraints: addendum A09 §10.2), frame (its criteria run and freeze a frame).
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
    """A human label of one sampled item (spec §4; amended by addendum A07). inclusion_prob is the item's phase-2 chance
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
    status and the count recompute from them)."""

    docket: str
    kept: bool
    reason: str
    comments: int
    artifacts: Sequence[ArtifactId]


class FrameItem(Record):
    """One item of the frame. withdrawn: the source lists it as withdrawn; it stays in the frame, since it was public at
    as_of (task 002-1, the coordinator's reasoning: a withdrawn comment remains in the public record, marked)."""

    id: str
    docket: str
    posted_at: Instant
    withdrawn: bool


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
