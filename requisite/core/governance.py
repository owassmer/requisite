"""Proposals, gates, approvals and the records of decisions (spec §4 "governance", §2.4, §3; amended by addendum §12)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import model_validator

from requisite.core.base import Day, Instant, Money, Probability, Record, Ref, exact_sum_is_one
from requisite.core.judgment import TimeWindow

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
# - human where a gate G1-G4 is the human's: decision_spec (G1: "the objective, horizon and loss definition").
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
        if len(self.states) < 2 or len(self.states) != len(self.acceptable_actions):
            raise ValueError("a witness is two or more feasible states, each with its acceptable actions")
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
        parents = {s.name: s for s in self.phase1}
        if any(s.parent not in parents for s in self.phase2):
            raise ValueError("every phase-2 stratum is nested in a phase-1 stratum")
        for name, stratum in parents.items():
            children = [s for s in self.phase2 if s.parent == name]
            if children and sum(s.phase1_count for s in children) != stratum.sample_size:
                raise ValueError(f"the phase-2 strata of {name} partition its {stratum.sample_size} phase-1 items")
        if self.sample == "census_then_gold" and any(s.sample_size != s.frame_size for s in self.phase1):
            raise ValueError("a census reads every item of the frame")
        return self


class GoldLabel(Record):
    """A human label of one sampled item (spec §4; amended by addendum A07). The design and stratum give its inclusion
    probability, exact; `outcome` keeps an explicit unresolved category, never dropped or read as zero (A07 §8.1).
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
