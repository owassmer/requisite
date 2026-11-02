"""Questions, the state a judge sees, judgments and their resolution (spec §4 "answers and questions", "state and
judgments"; spec §2.1, §6; amended by addendum A01, A04, A11)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import model_validator

from requisite.core.base import Day, Instant, Money, Record, Ref
from requisite.core.evidence import ClaimId, SyntheticClaimId
from requisite.core.kleene import AnswerStatus

# ---------------------------------------------------------------- answers and questions


class AnswerValue(Record):
    """One answer of a question's closed set (spec §4). `binds` gives the proposition values choosing it implies, which
    is how a resolved interpretive judgment binds a proposition (spec §5 cross-layer bindings)."""

    value: str
    meaning: str
    status: AnswerStatus
    binds: Mapping[str, Any] = {}


class TimeWindow(Record):
    """A window of time (spec §4), with each endpoint's inclusion stated (addendum A05 §5.4: "Define open/closed interval
    endpoints, timezone"). `tz` is the IANA zone in which calendar days of the window are counted."""

    start: Instant | None
    end: Instant | None
    start_closed: bool
    end_closed: bool
    tz: str = "UTC"

    @model_validator(mode="after")
    def _order(self) -> TimeWindow:
        if self.start is not None and self.end is not None and not self.start <= self.end:
            raise ValueError("a window's start is not after its end")
        return self


class Definition(Record):
    """A term's definition a question relies on (spec §4: AtomicQuestion.definitions and JudgmentState.definitions are
    Definition ids; spec §6.1 renders them under DEFINITIONS)."""

    id: str
    version: int
    term: str
    text: str
    source: str


Task = Literal["interpret", "classify", "score", "forecast", "resolve"]


class AtomicQuestion(Record):
    """A bounded, typed question over a closed answer set (spec §4).

    target: the subject, predicate and window stated explicitly. answers: the closed set; a task other than a forecast
    includes a not_established answer (spec §4). reaches: decision ids reachable from this question, computed by code
    (spec §4, §2.2). replaces: the questions this one splits or refines, where it is a sub-question (ours: spec §2.1
    "compound ... Replace with sub-questions plus a composition node" and addendum A02 §3.2's refinement need a
    sub-question to say what it replaces, and its gates follow from that: core.governance.SPLIT_GATES).
    """

    id: str
    version: int
    template_id: str | None = None
    bindings: Mapping[str, Any] = {}
    task: Task
    target: str
    window: TimeWindow | None = None
    answers: Sequence[AnswerValue]
    definitions: Sequence[str] = ()
    evidence_roles: Sequence[str] = ()
    paraphrase_set: str | None = None
    reaches: Sequence[str] = ()
    replaces: Sequence[Ref] = ()

    @model_validator(mode="after")
    def _answers(self) -> AtomicQuestion:
        values = [a.value for a in self.answers]
        if len(values) < 2:
            raise ValueError("a closed answer set has at least two answers")
        if len(set(values)) != len(values):
            raise ValueError("answer values are distinct")
        if self.task != "forecast" and not any(a.status == "not_established" for a in self.answers):
            raise ValueError("a non-forecast question includes a not_established answer (spec §4)")
        return self

    @property
    def ref(self) -> Ref:
        return Ref(id=self.id, version=self.version)


# ---------------------------------------------------------------- state


class Assumption(Record):
    """A hypothetical fixed for one question, never established by evidence (spec §4; I5: it enters a judgment only
    through assumed, rendered in its own labelled section)."""

    text: str
    binds: Mapping[str, Any]
    reason: str


EvaluationMode = Literal["prospective", "historical_replay", "retrospective_reconstruction"]


class JudgmentState(Record):
    """What a judge sees in a real case: observed claims as of an instant, labelled assumptions, definitions (spec §4,
    §6.1). Observed claims are observed-namespace ids only (I4). `mode` says whether as_of is now or a reconstruction of
    the past (addendum A11 §11.3: a historical reconstruction is not a clean prospective evaluation)."""

    as_of: Instant
    observed: Sequence[ClaimId]
    assumed: Sequence[Assumption] = ()
    definitions: Sequence[str] = ()
    mode: EvaluationMode = "prospective"


class SyntheticJudgmentState(Record):
    """The state of a synthetic case (fuzzing, simulation): its observed claims are synthetic, and it can never be the
    state of a real case (I4)."""

    as_of: Instant
    observed: Sequence[SyntheticClaimId]
    assumed: Sequence[Assumption] = ()
    definitions: Sequence[str] = ()
    scenario: str


# ---------------------------------------------------------------- judges and judgments

Provider = Literal["openai_decisions", "clef", "jev", "reasoning_model"]


class JudgeConfig(Record):
    """One configuration of the stability harness (spec §2.1 axes; spec §4). model_cutoff is the model's known training
    cutoff, or "unknown", recorded as such and never guessed (I10; addendum A11 §11.3: "Record unknown cutoffs as
    unknown")."""

    provider: Provider
    model: str
    model_cutoff: Day | Literal["unknown"]
    paraphrase_id: str
    render_id: str
    distractor_id: str | None = None


SUM_TOLERANCE = 1e-10  # the oracle's (spec/verify_math.py `distribution`); decisions, "Numbers"


class Judgment(Record):
    """One configuration's answer to one question over one state (spec §4).

    distribution: probabilities over the question's answer values, summing to one within 1e-10. A provider reports at a
    fixed step (`precision`); where its reported vector's sum is off one by no more than half a step per answer, the
    vector is a rounding, corrected proportionally, and `roundoff` (one minus the reported sum) records the correction
    (addendum A04 §4.2: "Numerical round-off corrections need a declared tolerance and log"). Beyond that it is no
    rounding, and the judgment has no distribution, never a normalized one. refusal: where the provider returned no usable distribution (a
    refusal, or probabilities that fail the check), what it returned instead; a refusal is an answer with no
    probabilities, never a zero (ours, from Slope's Decisions client). Exactly one of the two is set.
    response: the cassette key of the recorded response this judgment was read from (ours: every judgment shown is a
    recorded response of the named provider; posture, "Verification and done").
    cost_usd: Decimal (decisions: money is Decimal; spec §4 gives float).
    """

    question: Ref
    state_hash: str
    config: JudgeConfig
    distribution: Mapping[str, float] | None
    refusal: str | None = None
    roundoff: float
    precision: float
    response: str
    latency_ms: int
    cost_usd: Money
    created_at: Instant

    @model_validator(mode="after")
    def _one_of(self) -> Judgment:
        if (self.distribution is None) == (self.refusal is None):
            raise ValueError("a judgment has a distribution or a refusal, exactly one")
        if self.distribution is not None:
            check_distribution(self.distribution)
            if not 0 < self.precision <= 1 or abs(self.roundoff) > len(self.distribution) * self.precision / 2 + 1e-12:
                raise ValueError("a corrected round-off is within half a reporting step per answer")
        return self


def check_distribution(distribution: Mapping[str, float]) -> None:
    ps = list(distribution.values())
    if len(ps) < 2:
        raise ValueError("a distribution covers a closed set of at least two answers")
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in ps):
        raise ValueError("probabilities are finite and in [0, 1]")
    if abs(math.fsum(ps) - 1.0) > SUM_TOLERANCE:
        raise ValueError(f"probabilities sum to {math.fsum(ps)!r}, not 1 within {SUM_TOLERANCE}; never normalized")


# ---------------------------------------------------------------- resolution

Diagnosis = Literal[
    "stable", "imprecise_target", "contested", "evidence_gap", "compound", "incoherent_state", "unexplained"
]
Probe = Literal["paraphrase_provider", "evidence", "split", "state"]
EvidenceStatus = Literal["established", "contradicted", "not_established", "contested"]


class ProbeResult(Record):
    """One diagnostic probe's outcome and the evidence for it, retained whatever the primary diagnosis (addendum A01
    §2.4: "retain all probe evidence and competing explanations"). `fired` says whether its condition in the spec §2.1
    table held; `scores` holds the stability scores it measured."""

    probe: Probe
    fired: bool
    indicates: Diagnosis | None
    scores: Mapping[str, float]
    detail: str
    judgments: Sequence[str] = ()


class ResolvedJudgment(Record):
    """A question's judgments combined (spec §4, amended by addendum A01, A04 and §12).

    judgments: the content hashes of the Judgments combined. pooled: the descriptive pooled distribution under
    `pooling_policy` (A01 §2.3), with the per-configuration distributions kept in the judgments. s_arg, s_prob: the
    perturbation diagnostics (spec §2.1; A01 §2.1) under `tie_policy`. thresholds: the versioned stability thresholds
    applied (A01 §2.1: "configurable, versioned heuristics"). primary_diagnosis and probes: A01 §2.4. evidence_status and
    application_state under `application_policy` (A04 §4.1: a provider's probability does not by itself make its top
    answer an established fact; application_state is an answer value, or "review"). approvals: Approval ids (spec §2.4).
    """

    question: Ref
    state_hash: str
    judgments: Sequence[str]
    pooled: Mapping[str, float]
    pooling_policy: Ref
    s_arg: float
    s_prob: float
    tie_policy: Ref
    thresholds: Ref
    stable: bool
    primary_diagnosis: Diagnosis
    probes: Sequence[ProbeResult]
    evidence_status: EvidenceStatus
    application_policy: Ref
    application_state: str
    approvals: Sequence[str] = ()

    @model_validator(mode="after")
    def _scores(self) -> ResolvedJudgment:
        if not (0 <= self.s_arg <= 1 and 0 <= self.s_prob <= 1):
            raise ValueError("stability scores lie in [0, 1]")
        if len(self.judgments) < 2:
            raise ValueError("stability compares at least two configurations")
        check_distribution(self.pooled)
        if (self.primary_diagnosis == "stable") != self.stable:
            raise ValueError("the primary diagnosis is 'stable' exactly when the question is stable")
        if self.primary_diagnosis == "contested" and not self.approvals:
            # spec §2.1: contested goes to "Human review; hold as not_established until resolved" (G2).
            if self.evidence_status not in ("not_established", "contested") or self.application_state != "review":
                raise ValueError("a contested question is held as not established, in review, until a human resolves it")
        return self

