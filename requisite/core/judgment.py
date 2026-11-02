"""Questions, the state a judge sees, judgments and their resolution (spec §4 "answers and questions", "state and
judgments"; spec §2.1, §6; amended by addendum A01, A04, A11)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
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


def require_zone(tz: str) -> None:
    """A time zone is a real IANA zone, so a day count in it can be executed (A05 §5.4: "Mechanical policies must be
    executable")."""
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise ValueError(f"{tz!r} is not an IANA time zone") from e


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
        require_zone(self.tz)
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


class Paraphrase(Record):
    """One meaning-preserving rewording of a template's target and its answers' meanings (spec §2.1 "Paraphrase:
    meaning-preserving rewordings of the target and answer meanings"). The original wording is the paraphrase "p0"."""

    id: str
    target: str
    meanings: Mapping[str, str]


class QuestionTemplate(Record):
    """A reusable question with slots (spec §9 library objects: "QuestionTemplate (slots, answer semantics, frozen
    paraphrase set)"). `target` names each slot as {slot}; an AtomicQuestion instantiates it with bindings (spec §4
    AtomicQuestion.template_id and bindings), inheriting its answers and its paraphrase set."""

    id: str
    version: int
    task: Task
    target: str
    slots: Sequence[str]
    answers: Sequence[AnswerValue]
    paraphrase_set: str | None = None

    @model_validator(mode="after")
    def _slots(self) -> QuestionTemplate:
        import string

        named = {f for _, f, _, _ in string.Formatter().parse(self.target) if f}
        if named != set(self.slots):
            raise ValueError(f"the target's placeholders {sorted(named)} are the template's slots {sorted(self.slots)}")
        values = [a.value for a in self.answers]
        if len(values) < 2 or len(set(values)) != len(values):
            raise ValueError("a template's answers are a closed set of at least two distinct values")
        if self.task != "forecast" and not any(a.status == "not_established" for a in self.answers):
            raise ValueError("a non-forecast template includes a not_established answer (spec §4)")
        return self


class ParaphraseSet(Record):
    """A template's frozen set of paraphrases (spec §2.1: "from a frozen, reviewed set per template"; spec §3: agents
    propose paraphrase sets). Frozen: a changed set is a new version, so a stability score names the set it used. Every
    paraphrase words every answer of the template, and keeps the template's slots."""

    id: str
    version: int
    template: Ref
    slots: Sequence[str]
    answers: Sequence[str]
    paraphrases: Sequence[Paraphrase]

    @model_validator(mode="after")
    def _paraphrases(self) -> ParaphraseSet:
        import string

        ids = [p.id for p in self.paraphrases]
        if len(set(ids)) != len(ids) or "p0" not in ids:
            raise ValueError("a paraphrase set has distinct ids and includes the original wording, p0")
        for p in self.paraphrases:
            if set(p.meanings) != set(self.answers):
                raise ValueError(f"paraphrase {p.id} words answers {sorted(p.meanings)}, not {sorted(self.answers)}")
            named = {f for _, f, _, _ in string.Formatter().parse(p.target) if f}
            if named != set(self.slots):
                raise ValueError(f"paraphrase {p.id} keeps the template's slots {sorted(self.slots)}")
        return self


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
TIE_TOLERANCE = 1e-12  # the oracle's (spec/verify_math.py `stability`): answers this close to the maximum are tied


def top_answers(p: Mapping[str, float]) -> frozenset[str]:
    """The answers tied at a distribution's maximum, within TIE_TOLERANCE (addendum A01 §2.1: top-answer ties are
    handled explicitly); empty for an empty distribution."""
    if not p:
        return frozenset()
    top = max(p.values())
    return frozenset(a for a, x in p.items() if abs(x - top) <= TIE_TOLERANCE)


def correct_roundoff(reported: Sequence[float], step: float) -> list[float] | None:
    """The distribution closest to the reported values (in squared distance) among those the reported vector is a
    rounding of: each value within half a step of its reported value, and summing to one. That is an equal shift of every
    value, clipped to its interval [max(0, r - step/2), min(1, r + step/2)], with the shift that makes the sum one. None
    when no distribution rounds to the reported values: then they are not a rounding, and are refused, never normalized
    (addendum A04 §4.2). Computed exactly: each reported value and the step are read as the decimals they print as, the
    feasibility test is exact, and the shift is solved exactly on the piecewise-linear sum; only the result is a float.
    The definition of a round-off correction: the provider layer applies it, and every Judgment is checked against it."""
    r = [Fraction(repr(float(v))) for v in reported]
    half = Fraction(repr(float(step))) / 2
    lo = [max(Fraction(0), v - half) for v in r]
    hi = [min(Fraction(1), v + half) for v in r]
    if sum(lo) > 1 or sum(hi) < 1:
        return None

    def total(shift: Fraction) -> Fraction:
        return sum((min(h, max(lw, v + shift)) for v, lw, h in zip(r, lo, hi, strict=True)), Fraction(0))

    # The sum is piecewise linear and nondecreasing in the shift, with breakpoints where a value meets an end of its
    # interval: find the segment where it crosses one and solve on it.
    points = sorted({lw - v for v, lw in zip(r, lo, strict=True)} | {h - v for v, h in zip(r, hi, strict=True)})
    left = next(x for x in reversed(points) if total(x) <= 1)
    right = next((x for x in points if total(x) >= 1), left)
    if total(left) == 1 or right == left:
        shift = left
    else:
        shift = left + (1 - total(left)) * (right - left) / (total(right) - total(left))
    exact = [min(h, max(lw, v + shift)) for v, lw, h in zip(r, lo, hi, strict=True)]
    assert sum(exact) == 1
    return [float(x) for x in exact]


class Judgment(Record):
    """One configuration's answer to one question over one state (spec §4).

    reported: the probabilities as the provider reported them, at its reporting step (`precision`). distribution: a
    distribution over the question's answer values, summing to one within 1e-10, each value within half a step of its
    reported value: the reported vector is a rounding of it. A reported vector that is no rounding of any distribution
    has no distribution, never a normalized one; the correction from reported to distribution is the log of the
    round-off (addendum A04 §4.2: "Numerical round-off corrections need a declared tolerance and log"). refusal: where the provider returned no usable distribution (a
    refusal, or probabilities that fail the check), what it returned instead; a refusal is an answer with no
    probabilities, never a zero (ours, from Slope's Decisions client). Exactly one of the two is set.
    response: the cassette key of the recorded response this judgment was read from (ours: every judgment shown is a
    recorded response of the named provider; posture, "Verification and done").
    cost_usd: Decimal (decisions: money is Decimal; spec §4 gives float).
    """

    question: Ref
    state_hash: str
    config: JudgeConfig
    reported: Mapping[str, float] | None
    distribution: Mapping[str, float] | None
    refusal: str | None = None
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
            if self.reported is None or set(self.reported) != set(self.distribution):
                raise ValueError("a distribution is a correction of the reported probabilities, over the same answers")
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in self.reported.values()):
                raise ValueError("reported probabilities are finite and in [0, 1]")
            if not 0 < self.precision <= 1:
                raise ValueError("a reporting step lies in (0, 1]")
            answers = list(self.reported)
            corrected = correct_roundoff([self.reported[a] for a in answers], self.precision)
            if corrected is None:
                raise ValueError("no distribution rounds to the reported values: they are refused, never corrected")
            if any(self.distribution[a] != p for a, p in zip(answers, corrected, strict=True)):
                raise ValueError("the distribution is the round-off correction of the reported values (correct_roundoff)")
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

