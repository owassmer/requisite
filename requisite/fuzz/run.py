"""A fuzz run of one contract version: its IR against its prose, through rendered evidence and decision models (spec
§8.6; addendum A09). Code decides which scenarios matter (requisite.fuzz.scenarios), an agent makes each realistic,
decision models test whether the prose resolves it as the IR does, and code scores the result into a committed
core.FuzzReport.

The steps, after the plan (coverage, stress classes, each scenario's inputs, its outcome and transfers from the
evaluator):

- Split. Before anything is asked, each scenario falls in the development or the held-out set by a seeded rule over the
  contract version alone (A09 §10.4; A01 §2.5), so a scenario keeps its set while the prose is revised.
- Rendering (agent). The scenario renderer writes the scenario's packet; each document is committed in the run's
  synthetic case as a SyntheticArtifact with one SyntheticClaim quoting it whole (I4), its title, issuing source and
  publication date heading its text.
- Rendering check (decision models). Each atomic condition the scenario fixes is asked of the packet: an interpretive
  condition by its own library template where the library holds it, otherwise by condition_holds over the condition as
  code states it (requisite.fuzz.statements), with each answer meaning true, false or not established. A comparison of
  a computed number asks its inputs instead, the readings it reads (each by its own bands) and the values it reads
  (each as published), since the arithmetic is code's (spec I7). Every provider answers in the original wording; a
  question is recovered when the pooled answers' top is exactly the intended answer. A packet with a document dated
  after the evidence instant, or with any question not recovered, is rendered again with what was not recovered, up to
  RERENDERS times; one that still fails is recorded as rendering_passed false, its cause the rendering's (confirmed by
  the check), and is excluded from scoring (spec §8.6 step 4: a failed rendering is never counted against the contract).
- Resolution (decision models). K configurations, every provider in every paraphrase of the resolver template the
  procedure calls for (requisite/library/templates/fuzz.json: VOID, EXTENDED or DISPUTED among the answers where the
  procedure can end so), answer from the prose and the packet. Given only the original wording, K is the providers.
- Scoring (code). Over the scenarios whose rendering passed: fidelity, the share whose prose outcome (the top of the
  resolvers' pooled distribution, equal weight per provider: A01 §2.3; a tie at the top is no outcome, so ambiguous)
  equals the IR's; clarity, the mean S_arg over configurations; ambiguity, the mean pooled probability of AMBIGUOUS. The
  prose's transfers are the evaluator's for the prose outcome (`forced`), compared with the IR's dated transfers.
- Attribution. A scenario fails where prose and IR disagree, or its own ambiguity or clarity misses the thresholds. Its
  sections are those whose IR nodes it exercises (the conditions its outcome turns on, with the nodes around them) and
  those its stress class bears on; the suspected clauses are ranked over all scored scenarios by the Ochiai coefficient
  (Abreu, Zoeteweij and van Gemund 2007, spectrum-based fault localization; ours). Each live hypothesis is opened as an
  Attribution, every one open until a minimal counterexample or human gold checks it (A09 §10.1): a passing rendering
  check does not make the clause the cause (A.T18).
- The report: A09 §10.4's hard gates, and the equivalence the scenarios support (A09 §10.5).

Every judgment is a recorded provider call (requisite.judgment.providers) and every packet a recorded agent call
(requisite.agents.runner); both are committed in the synthetic case, with the scenarios' own records.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import model_validator

from requisite import core
from requisite.agents import runner
from requisite.agents.answers import PacketDocument, PacketDraft, ProseDraft
from requisite.agents.trace import made_from, record_run
from requisite.contract import evaluate
from requisite.contract.evaluate import Interpretation
from requisite.core.contract import Equivalence
from requisite.core.evidence import synthetic_artifact_id
from requisite.core.kleene import Kleene, from_status
from requisite.core.results import HARD_GATES, Cause, HardGate, JudgeOutcome, same_transfers
from requisite.fuzz import statements
from requisite.fuzz.scenarios import FORCED, STRESS, FuzzError, Plan, Scenario, plan, sources
from requisite.judgment import providers as judging
from requisite.judgment.render import TEXT, ask, render_state
from requisite.library import templates as library
from requisite.optimize import liquidity
from requisite.stability.scores import EQUAL_PER_PROVIDER, pool, scores, top_set
from requisite.store import Refused, Store, open_case

THRESHOLDS = core.Ref(id="fuzz_thresholds.spec_8_6", version=1)
FIDELITY, CLARITY, AMBIGUITY = 0.95, 0.9, 0.05  # spec §8.6's starting thresholds (A01 §2.1: versioned heuristics)
RERENDERS = 3
HELD_OUT = Fraction(3, 10)
SECTION_ORDER = ("underlying_and_definitions", "source_hierarchy_and_conflict_rule", "trigger", "examples", "timing",
                 "unknown_at_expiry", "revisions")  # fmt: skip  (spec §8.5)
SECTION_TITLES = {
    "underlying_and_definitions": "Underlying and definitions",
    "source_hierarchy_and_conflict_rule": "Source hierarchy and conflict rule",
    "trigger": "Trigger",
    "examples": "Qualifying and non-qualifying examples",
    "timing": "Timing",
    "unknown_at_expiry": "Unknown at expiry",
    "revisions": "Revisions",
}
# The sections a stress class bears on beside the conditions it exercises (spec §8.5's sections; ours).
STRESS_SECTIONS = {
    "conflicting_sources": ("source_hierarchy_and_conflict_rule",),
    "revised_value": ("revisions",),
    "evidence_missing_at_expiry": ("unknown_at_expiry",),
}
STRESS_WORDS = {  # the scenario renderer's own words for each class (requisite/agents/prompts/scenario_renderer.md)
    "conflicting_sources": "conflicting sources",
    "revised_value": "a revised value",
    "renamed_or_recodified_instrument": "a renamed or recodified instrument",
    "partial_scope": "partial scope",
    "stay_or_injunction": "a stay or injunction",
    "evidence_missing_at_expiry": "evidence missing at expiry",
}
ANSWER = {Kleene.TRUE: "holds", Kleene.FALSE: "does_not_hold", Kleene.UNKNOWN: "not_established"}

JudgeCall = Callable[[judging.Ask], core.Judgment]
AgentCall = Callable[..., runner.AgentResult[Any]]


# ---------------------------------------------------------------- inputs


def resolver_template(ir: core.ContractIR) -> str:
    """The resolution template whose answers are the outcomes the contract's procedure can reach at expiration besides
    YES, NO and the judges' AMBIGUOUS: void or extended by its rule for an unknown at expiry, disputed by its conflict
    rule (the evaluator, at its determination instant)."""
    proc = ir.resolution
    _, agree = sources(ir)
    parts = [{"void": "void", "extend": "extended"}.get(proc.on_unknown_at_expiry, "")]
    parts.append("disputed" if agree else "")
    parts = [p for p in parts if p]
    return "resolve_contract" + ("_" + "_".join(parts) if parts else "")


Prose = ProseDraft | core.Rulebook  # the drafter's answer, or the committed rulebook it became: sections alike


def prose_hash(prose: Prose) -> str:
    """The prose's content hash, over its sections."""
    return core.sha256(core.canonical_json({"sections": [s.model_dump(mode="json") for s in prose.sections]}))


def prose_text(prose: Prose) -> str:
    """The rulebook as a resolver reads it: each section with text, under its title, in spec §8.5's order."""
    by = {s.name: s for s in prose.sections}
    return "\n\n".join(f"{SECTION_TITLES[n]}\n{by[n].text.strip()}" for n in SECTION_ORDER if n in by and by[n].text.strip())


def prose_problems(ir: core.ContractIR, prose: Prose) -> list[str]:
    """What keeps prose from being this contract's rulebook (spec §8.5: the drafter adds nothing the IR lacks and leaves
    out nothing it has): a section citing what is neither a node nor an observable of the IR, or a condition that
    decides settlement stated by no section (a condition is stated where a section cites it, a node inside it, or an
    observable it reads)."""
    roots = [ir.trigger] + ([ir.payout.quantity] if ir.payout.quantity is not None else [])
    nodes = {n.id: n for r in roots for n in core.walk(r)}
    known = set(nodes) | {o.id for o in ir.observables}
    out = []
    cited: set[str] = set()
    for section in prose.sections:
        unknown = sorted(set(section.ir_nodes) - known)
        if unknown:
            out.append(f"section {section.name} cites {unknown}, which are neither node ids nor observable ids of this "
                       f"IR; it may cite only {sorted(known)}")  # fmt: skip
        cited |= set(section.ir_nodes)
    for cid in evaluate.lower(ir).conditions:
        inside = list(core.walk(nodes[cid]))
        ids = {n.id for n in inside} | {n.observable for n in inside if isinstance(n, core.Cmp | core.ValueAt)}
        if not ids & cited:
            out.append(f"condition {cid}, which decides settlement, is stated by no section")
    return out


def _check_prose(ir: core.ContractIR, prose: Prose) -> dict[str, frozenset[str]]:
    """Each section's IR nodes, an observable's id standing for the nodes that read it; refused where a section names
    what the contract does not have, since then no clause links to its node (spec §8.5, WP16)."""
    names = [s.name for s in prose.sections]
    if len(set(names)) != len(names):
        raise FuzzError(f"the prose repeats sections: {names}")
    roots = [ir.trigger] + ([ir.payout.quantity] if ir.payout.quantity is not None else [])
    nodes = {n.id: n for r in roots for n in core.walk(r)}
    observables = {o.id for o in ir.observables}
    out: dict[str, frozenset[str]] = {}
    for s in prose.sections:
        unknown = sorted(set(s.ir_nodes) - set(nodes) - observables)
        if unknown:
            raise FuzzError(f"section {s.name} names {unknown}, which are neither IR nodes nor observables of {ir.id}")
        ids = {i for i in s.ir_nodes if i in nodes}
        ids |= {n.id for n in nodes.values() if isinstance(n, core.Cmp | core.ValueAt) and n.observable in s.ir_nodes}
        out[s.name] = frozenset(ids)
    return out


# ---------------------------------------------------------------- one scenario through the models


@dataclass(frozen=True)
class Question:
    """A rendering-check question: what it asks, the answer that recovers the scenario, and its words for the
    renderer."""

    question: core.AtomicQuestion
    expected: str  # an answer value, or for a library interpretive template, a truth's status ("true"/"false"/"unknown")
    by_status: bool
    statement: str
    intended: str


@dataclass
class Packet:
    attempt: int
    agent: runner.AgentResult[Any]
    agent_input: dict[str, Any]
    documents: list[tuple[core.SyntheticArtifact, bytes, core.SyntheticClaim]]
    judgments: list[core.Judgment] = field(default_factory=list)
    missed: list[dict[str, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.missed


@dataclass
class Resolved:
    """A scenario after the models: its packets in order, and the resolution of the last one."""

    scenario: Scenario
    split: Literal["development", "held_out"]
    packets: list[Packet]
    judgments: list[core.Judgment] = field(default_factory=list)
    pooled: dict[str, float] = field(default_factory=dict)
    outcome: JudgeOutcome = "ambiguous"
    s_arg: float = 0.0

    @property
    def passed(self) -> bool:
        return self.packets[-1].passed


@dataclass(frozen=True)
class Run:
    """What a fuzz run did: its committed report, its plan, each scenario's packets and resolution, and its K."""

    report: core.FuzzReport
    plan: Plan
    resolved: tuple[Resolved, ...]
    resolver_template: str
    configurations: tuple[tuple[str, str], ...]  # (provider, paraphrase id)
    key: str


ASKED = ("check", "read", "value")  # what a rendering-check question asks of a node: a condition, a reading, a value


def question_prefix(contract: core.Ref) -> str:
    """The prefix of every question a fuzz run of this contract version asks. A rendering-check question is
    `<prefix>.<asked>.<node id>`, named by the IR node it asks about (ASKED: a condition, a reading or a published
    value); the resolution question is `<prefix>.resolve.<the prose hash's first 12 digits>`."""
    return f"fuzz.{contract.id}.v{contract.version}"


def asked_node(contract: core.Ref, question_id: str) -> str | None:
    """The IR node of this contract version a rendering-check question asks about, read from the id it was given
    (question_prefix); None for any other question."""
    for asked in ASKED:
        head = f"{question_prefix(contract)}.{asked}."
        if question_id.startswith(head):
            return question_id[len(head):]
    return None


def _questions(p: Plan, s: Scenario) -> list[Question]:
    ir = p.ir
    out: list[Question] = []
    prefix = question_prefix(ir.ref)
    holds = library.template("condition_holds")
    meanings = {a.value: a.meaning for a in holds.answers}

    def condition(qid: str, statement: str, truth: Kleene) -> Question:
        q = library.instantiate(holds, {"condition": statement}, question_id=qid)
        return Question(q, ANSWER[truth], False, statement, meanings[ANSWER[truth]])

    roots = [ir.trigger]
    nodes = {n.id: n for r in roots for n in core.walk(r)}
    for cid in p.lowered.conditions:
        node, truth = nodes[cid], s.truths[cid]
        if isinstance(node, core.Interpretive):
            templates = library.all_templates()
            if node.question_template in templates:
                t = templates[node.question_template]
                q = library.instantiate(t, dict(node.bindings), question_id=f"{prefix}.check.{cid}")
                intended = " or ".join(a.meaning for a in t.answers if from_status(a.status) is truth)
                out.append(Question(q, truth.value, True, q.target, intended))
            else:
                out.append(condition(f"{prefix}.check.{cid}", statements.condition(ir, node), truth))
        elif isinstance(node, core.NumCmp) and isinstance(node.expr, core.Computed):
            for name, sub in sorted(node.expr.inputs.items()):
                if isinstance(sub, core.ReadValue):
                    out.append(_reading(p, s, sub, f"{prefix}.read.{sub.id}"))
                elif isinstance(sub, core.ValueAt):
                    have = [f for f in s.facts if f.observable == sub.observable and f.at == sub.at]
                    if have:
                        stmt = statements.published_value(ir, sub.observable, sub.at, have[0].value, p.sources)
                        out.append(condition(f"{prefix}.value.{sub.id}", stmt, Kleene.TRUE))
                    else:
                        stmt = statements.number(ir, sub) + ". " + statements.policy(ir, sub.observable, p.sources)
                        out.append(condition(f"{prefix}.value.{sub.id}", f"A value is published for it: {stmt}",
                                             Kleene.FALSE))  # fmt: skip
                else:
                    raise FuzzError(f"{cid}: {node.expr.id}'s input {name} is a {sub.kind}, not asked of a packet")
        else:
            out.append(condition(f"{prefix}.check.{cid}", statements.checked(ir, node, p.sources), truth))
    return out


def _reading(p: Plan, s: Scenario, r: core.ReadValue, qid: str) -> Question:
    q = statements.reading_question(r, qid)
    lo, hi = cast(tuple[Decimal, Decimal], s.interpretations[r.id])
    bands = [(b[0], b[1]) for b in r.bands]
    expected = f"band{bands.index((lo, hi))}" if (lo, hi) in bands else "not_established"
    meaning = {a.value: a.meaning for a in q.answers}[expected]
    return Question(q, expected, False, q.target, meaning)


def _renderer_input(p: Plan, s: Scenario, questions: Sequence[Question]) -> dict[str, Any]:
    ir = p.ir
    obs = []
    for o in ir.observables:
        obs.append({"id": o.id, "description": o.description, "value_type": o.value_type, "unit": o.unit,
                    "precision": None if o.precision is None else str(o.precision), "schedule": o.schedule,
                    "sources": [{"id": x.id, "name": x.name, "rank": x.rank, "role": x.role, "url": x.url}
                                for x in o.sources]})  # fmt: skip
    return {
        "contract_observables": obs,
        "evidence_as_of": p.at.isoformat(),
        "every_document_published_by": p.at.isoformat(),
        "conditions": [{"condition": q.statement, "value": q.intended} for q in questions],
        "published_values": [f.shown() for f in s.facts],
        "stress_class": STRESS_WORDS[s.stress] if s.stress else None,
    }


def _documents(s: Scenario, agent: runner.AgentResult[Any], at: datetime
               ) -> list[tuple[core.SyntheticArtifact, bytes, core.SyntheticClaim]]:  # fmt: skip
    out = []
    draft = cast(PacketDraft, agent.output)
    for d in draft.documents:
        content = _document_text(d).encode()
        art = core.SyntheticArtifact(
            id=synthetic_artifact_id(content), scenario=s.id, generator=f"scenario_renderer {agent.run[:16]}",
            media_type="text/plain", size=len(content), published_at=d.published_at, created_at=at,
        )  # fmt: skip
        text = content.decode()
        claim = core.SyntheticClaim(id=f"syn_clm_{core.sha256(f'{art.id}|0:{len(text)}|{text}')}", artifact=art.id,
                                    span=(0, len(text)), text=text, authority=d.source)  # fmt: skip
        out.append((art, content, claim))
    return out


def _document_text(d: PacketDocument) -> str:
    return f"{d.title}\nIssued by: {d.source}\nPublished: {d.published_at.isoformat()}\n\n{d.text.strip()}"


def _state(p: Plan, s: Scenario, packet: Packet) -> core.SyntheticJudgmentState:
    return core.SyntheticJudgmentState(as_of=p.at, observed=[c.id for _, _, c in packet.documents], scenario=s.id)


def _rendered(p: Plan, s: Scenario, packet: Packet) -> tuple[core.SyntheticJudgmentState, Any]:
    state = _state(p, s, packet)
    claims = {c.id: c for _, _, c in packet.documents}
    artifacts = {a.id: a for a, _, _ in packet.documents}
    return state, render_state(state, TEXT, claims=claims, artifacts=artifacts, definitions={})


def _config(provider: str, paraphrase: str) -> core.JudgeConfig:
    spec = judging.PROVIDERS[provider]
    return core.JudgeConfig(provider=spec.name, model=spec.model, model_cutoff=spec.model_cutoff,
                            paraphrase_id=paraphrase, render_id=TEXT.id)  # fmt: skip


def _top(judgments: Sequence[core.Judgment]) -> tuple[frozenset[str], dict[str, float] | None]:
    dists = [(j.config.provider, dict(j.distribution)) for j in judgments if j.distribution is not None]
    if not dists:
        return frozenset(), None
    pooled = dict(pool(dists, EQUAL_PER_PROVIDER).distribution)
    return top_set(pooled), pooled


@dataclass(frozen=True)
class _Models:
    providers: tuple[str, ...]
    configurations: tuple[tuple[str, str], ...]
    resolver: core.QuestionTemplate
    paraphrases: core.ParaphraseSet | None
    judge: JudgeCall
    agent: AgentCall


def _check(p: Plan, s: Scenario, packet: Packet, questions: Sequence[Question], m: _Models) -> None:
    late = [d.title for d in cast(PacketDraft, packet.agent.output).documents if d.published_at > p.at]
    for title in late:
        packet.missed.append({"condition": f"the document {title!r}", "intended": f"published by {p.at.isoformat()}",
                              "judges_read": "it is dated after the evidence instant"})  # fmt: skip
    state, rendered = _rendered(p, s, packet)
    h = state.content_hash()
    for q in questions:
        asked = ask(q.question, library.original(q.question))
        js = [m.judge(judging.Ask(q.question, h, rendered, asked, _config(pv, "p0"))) for pv in m.providers]
        packet.judgments += js
        meaning = {a.value: a for a in q.question.answers}

        def recovered(top: frozenset[str], q: Question = q, meaning: Mapping[str, core.AnswerValue] = meaning) -> bool:
            if q.by_status:
                return bool(top) and all(from_status(meaning[a].status).value == q.expected for a in top)
            return top == {q.expected}

        # Each provider on its own, so "every provider recovered it" is what was checked (external evaluation, November
        # 2: refusals were dropped before pooling, and one provider's answer passed the check for all).
        misread = []
        for j in js:
            if j.distribution is None:
                misread.append(f"{j.config.provider} gave no answer (refused: {j.refusal})")
                continue
            top = top_set(dict(j.distribution))
            if not recovered(top):
                misread.append(f"{j.config.provider} read " + " or ".join(meaning[a].meaning for a in sorted(top)))
        if misread:
            packet.missed.append({"condition": q.statement, "intended": q.intended, "judges_read": "; ".join(misread)})


def _scenario(p: Plan, s: Scenario, split: Literal["development", "held_out"], rules: str, prose_key: str,
              m: _Models) -> Resolved:  # fmt: skip
    questions = _questions(p, s)
    base = _renderer_input(p, s, questions)
    packets: list[Packet] = []
    for attempt in range(1 + RERENDERS):
        agent_input = dict(base)
        if packets:
            last = packets[-1]
            agent_input["previous_packet"] = [d.model_dump(mode="json")
                                              for d in cast(PacketDraft, last.agent.output).documents]  # fmt: skip
            agent_input["not_recovered"] = last.missed
        result = m.agent("scenario_renderer", agent_input, PacketDraft, effort="high")
        packet = Packet(attempt, result, agent_input, _documents(s, result, p.at))
        _check(p, s, packet, questions, m)
        packets.append(packet)
        if packet.passed:
            break
    out = Resolved(s, split, packets)
    # Resolution of the last packet: a passing one is scored; a failing one is recorded, never scored.
    state, rendered = _rendered(p, s, packets[-1])
    q = library.instantiate(m.resolver, {"rules": rules}, question_id=f"{question_prefix(p.ir.ref)}.resolve.{prose_key[:12]}")
    for provider, pid in m.configurations:
        wording = _paraphrased(q, m.paraphrases, pid) if m.paraphrases is not None else library.original(q)
        out.judgments.append(m.judge(judging.Ask(q, state.content_hash(), rendered, ask(q, wording), _config(provider, pid))))
    dists = [dict(j.distribution) for j in out.judgments if j.distribution is not None]
    if len(dists) < 2:
        raise FuzzError(f"scenario {s.id}: {len(dists)} of {len(out.judgments)} resolver configurations gave a "
                        "distribution; clarity compares at least two")  # fmt: skip
    top, pooled = _top(out.judgments)
    assert pooled is not None
    out.pooled = pooled
    out.outcome = cast(JudgeOutcome, next(iter(top))) if len(top) == 1 else "ambiguous"
    out.s_arg = scores(dists).s_arg
    return out


TERM_ANSWERS = (  # core.TERM_READINGS, with their meanings
    core.AnswerValue(value="states", meaning="the rules state exactly this term", status="affirmative"),
    core.AnswerValue(value="states_otherwise", meaning="the rules state this term differently (another amount, date, "
                     "currency, period or refund)", status="negative"),  # fmt: skip
    core.AnswerValue(value="not_stated", meaning="the rules do not state this term", status="not_established"),
)
assert tuple(a.value for a in TERM_ANSWERS) == core.TERM_READINGS


def term_question(prefix: str, term: str, statement: str, rules: str) -> core.AtomicQuestion:
    return core.AtomicQuestion(
        id=f"{prefix}.term.{term}", version=1, task="classify",
        target=("Do the contract rules below state the following payment term, exactly as given? Compare what the rules "
                f"say with the term; do not compute anything the rules do not state.\nTerm: {statement}\n"
                f"Contract rules:\n{rules}"),
        answers=list(TERM_ANSWERS),
    )  # fmt: skip


def read_terms(p: Plan, rules: str, prose_key: str, m: _Models) -> tuple[list[core.TermReading], list[core.Judgment]]:
    """Each payment term of the IR asked of the prose, by every provider in the original wording, over no evidence: the
    term is the IR's, as code states it (statements.payment_terms), and the readers say whether the prose states it."""
    prefix = f"{question_prefix(p.ir.ref)}.{prose_key[:12]}"
    state = core.SyntheticJudgmentState(as_of=p.at, observed=[], scenario="payment_terms")
    rendered = render_state(state, TEXT, claims={}, artifacts={}, definitions={})
    out, judgments = [], []
    for term, statement in statements.payment_terms(p.ir).items():
        q = term_question(prefix, term, statement, rules)
        js = [m.judge(judging.Ask(q, state.content_hash(), rendered, ask(q, library.original(q)), _config(pv, "p0")))
              for pv in m.providers]  # fmt: skip
        top, pooled = _top(js)
        read = next(iter(top)) if len(top) == 1 else "ambiguous"
        out.append(core.TermReading(term=cast(Any, term), statement=statement, answers=pooled or {}, read=cast(Any, read),
                                    judgments=[j.content_hash() for j in js]))  # fmt: skip
        judgments += js
    return out, judgments


def _paraphrased(q: core.AtomicQuestion, pset: core.ParaphraseSet, pid: str) -> core.Paraphrase:
    """One paraphrase of the resolver template, its slot filled with the question's binding (the prose)."""
    x = next(x for x in pset.paraphrases if x.id == pid)
    return core.Paraphrase(id=x.id, target=library.fill(x.target, q.bindings), meanings=dict(x.meanings))


# ---------------------------------------------------------------- scoring and the report


def concrete_input(p: Plan, s: Scenario) -> str:
    """What the scenario puts before the renderer and so before the readers: the hash of its renderer request (the
    conditions and their intended values, the published values, the stress class, the evidence instant). Two scenarios
    with one concrete input are one test case, however they were reached."""
    return core.sha256(core.canonical_json(_renderer_input(p, s, _questions(p, s))))


def _split(p: Plan) -> dict[str, Literal["development", "held_out"]]:
    """The held-out share of the scenarios, drawn over their concrete inputs (concrete_input) in an order seeded by the
    contract version alone: the same test cases are held out however the prose changes (A09 §10.4; A01 §2.5), and
    scenarios with one concrete input fall in one set (external evaluation, November 2: two scenarios with the same state,
    renderer request and outcome had sat on both sides). Whole groups are held out, in their order, until the held-out
    scenarios reach HELD_OUT of the run's, skipping any group that would leave no development scenario; a run whose
    scenarios are one test case holds none out, and its held_out_reported_separately gate says so."""
    seed = core.sha256(p.ir.content_hash() + "|requisite.fuzz.split")
    groups: dict[str, list[Scenario]] = {}
    for s in p.scenarios:
        groups.setdefault(concrete_input(p, s), []).append(s)
    ordered = sorted(groups, key=lambda k: core.sha256(f"{seed}|{k}"))
    n = len(p.scenarios)
    want = math.ceil(n * HELD_OUT) if len(groups) >= 2 else 0
    held: set[str] = set()
    for k in ordered:
        if len(held) >= want:
            break
        if len(held) + len(groups[k]) < n:  # a group that would leave no development scenario is skipped
            held |= {s.id for s in groups[k]}
    return {s.id: ("held_out" if s.id in held else "development") for s in p.scenarios}


def _money(x: Fraction) -> Decimal:
    return evaluate._to_decimal(x)  # exact: sums of Decimal amounts terminate


# ---------------------------------------------------------------- forced outcomes (the prose's transfers; severity)


def forced(ir: core.ContractIR, truth: Kleene, observations: Sequence[core.SyntheticObservation],
           interpretations: Mapping[str, Interpretation], at: datetime) -> core.ResolutionEvaluation:  # fmt: skip
    """What the evaluator gives the contract when its trigger is the given truth at the determination instant: its
    payout, timing and procedure on the scenario's inputs (the payout quantity's included), with the trigger standing as
    one interpretive node. So the transfers of an outcome other than the IR's are the evaluator's, never computed here
    (I8). The resolvers judge the evidence available at expiration, so an outcome forced is an outcome at expiration:
    early expiration does not apply to it."""
    roots = [ir.trigger] + ([ir.payout.quantity] if ir.payout.quantity is not None else [])
    if any(n.id == FORCED for r in roots for n in core.walk(r)):
        raise FuzzError(f"the contract has a node named {FORCED!r}, the name a forced outcome takes")
    node = core.Interpretive(id=FORCED, question_template=FORCED, bindings={})
    proc = ir.resolution.model_copy(update={"early_expiration": "none", "early_expiration_at": None,
                                            "early_expiration_rule": None})  # fmt: skip
    sub = ir.model_copy(update={"trigger": node, "resolution": proc})
    payout = {n.id for n in core.walk(ir.payout.quantity) if isinstance(n, core.ReadValue)} if ir.payout.quantity else set()
    given: dict[str, Interpretation] = {k: v for k, v in interpretations.items() if k in payout}
    given[FORCED] = truth
    return evaluate.evaluate(sub, observations, given, at=at)


def determined(ev: core.ResolutionEvaluation, ir: core.ContractIR) -> bool:
    """Whether an evaluation's transfers are settled amounts: a NO, a void with its stated refund, or a YES with every
    payment made (a quantity still an interval, a refund of the premium or an early date left to a rule are not)."""
    if ev.procedure_status == "resolved_no":
        return True
    if ev.procedure_status == "void":
        returns = ir.resolution.void_transfer.returns
        return returns == "nothing" or (returns == "fixed_amount" and len(ev.transfers) == 1)
    if ev.procedure_status == "resolved_yes":
        expected = len(ir.payout.periods) if ir.payout.payment == "per_period" else 1
        return len([t for t in ev.transfers if t.basis == "payout"]) == expected
    return False


def total(ev: core.ResolutionEvaluation) -> Fraction:
    return sum((Fraction(t.amount) for t in ev.transfers), Fraction(0))


def _liquidity(ir: core.ContractIR, a: Sequence[core.CashTransfer], b: Sequence[core.CashTransfer]) -> Decimal:
    """The largest difference, over the dates either schedule pays on, between the holder's cash under the two
    schedules, per unit: the business cash engine's own accounting (optimize.liquidity) of a holder with no other flows
    (A09 §10.4: "a dated liquidity-impact discrepancy using the same business cash engine")."""
    from zoneinfo import ZoneInfo

    zone = ZoneInfo(ir.timing.tz)
    dates = sorted({t.at.astimezone(zone).date() for t in [*a, *b]})
    if not dates:
        return Decimal(0)

    def receipts(ts: Sequence[core.CashTransfer]) -> tuple[tuple[Decimal, ...], ...]:
        row = [Decimal(0)] * len(dates)
        for t in ts:
            row[dates.index(t.at.astimezone(zone).date())] += t.amount
        return (tuple(row),)

    zeros = tuple(Decimal(0) for _ in dates)
    problem = liquidity.LiquidityProblem(
        dates=tuple(dates), probabilities=(Fraction(1),), initial_cash=Decimal(0), operating=(zeros,), loss=(zeros,),
        reserve=zeros, alpha=Fraction(0), budget=Decimal(0),
        candidates=(liquidity.Candidate("prose", Decimal(0), dates[0], receipts(a), Decimal(1)),
                    liquidity.Candidate("ir", Decimal(0), dates[0], receipts(b), Decimal(1))),
    )  # fmt: skip
    prose_cash = liquidity.evaluate(problem, [1, 0]).cash[0]
    ir_cash = liquidity.evaluate(problem, [0, 1]).cash[0]
    return max(abs(x - y) for x, y in zip(prose_cash, ir_cash, strict=True))


def _prose_side(p: Plan, r: Resolved, terms: Sequence[core.TermReading]
                ) -> tuple[list[core.CashTransfer] | None, Decimal | None, Decimal | None]:  # fmt: skip
    """What the prose pays on the scenario: the evaluator's transfers for the prose outcome, where the readers read the
    prose to state every payment term that outcome rests on (`terms`; core.terms_used), paying nothing included;
    otherwise None, and no discrepancy computed, since what the prose pays is then not established. With the amount and
    liquidity discrepancies."""
    s, ir = r.scenario, p.ir
    ev = s.evaluation
    if r.outcome == ev.procedure_status:
        prose_ev: core.ResolutionEvaluation | None = ev
    elif r.outcome in ("resolved_yes", "resolved_no", "void", "extended"):
        truth = {"resolved_yes": Kleene.TRUE, "resolved_no": Kleene.FALSE}.get(r.outcome, Kleene.UNKNOWN)
        prose_ev = forced(ir, truth, s.observations, s.interpretations, p.at)
        if prose_ev.procedure_status != r.outcome:
            raise FuzzError(f"scenario {s.id}: the outcome {r.outcome} forced gives {prose_ev.procedure_status}")
    else:
        prose_ev = None  # ambiguous or disputed: the prose determines no payment
    if prose_ev is None:
        return None, None, None  # the outcome determines no payment (core.FuzzScenario.prose_transfers)
    transfers = list(prose_ev.transfers)
    read: dict[str, str] = {t.term: t.read for t in terms}
    if any(read.get(term) != "states" for term in core.terms_used(r.outcome, transfers)):
        return None, None, None  # what the prose pays is not established (core.FuzzScenario.prose_transfers)
    if not determined(prose_ev, ir):
        return None, None, None  # its amount is not yet determined: unknown, never zero (A05 §5.3)
    if not determined(ev, ir):
        return transfers, None, None
    amount = _money(abs(total(prose_ev) - total(ev)))
    return transfers, amount, _liquidity(ir, prose_ev.transfers, ev.transfers)


def _severity(p: Plan, s: Scenario, mandatory: bool) -> Literal["critical", "material", "minor"]:
    """What a wrong outcome could cost per unit (A09 §10.4: a small share of failures can hold every material case):
    the largest difference between the IR's transfers and those of any outcome the contract could have been resolved to
    instead (YES, NO, its unknown-at-expiry rule), an undetermined amount counting as one. Critical: a mandatory scenario
    with money at stake; material: another with money at stake; minor: none at stake."""
    ev = s.evaluation
    stake = False
    for truth in (Kleene.TRUE, Kleene.FALSE, Kleene.UNKNOWN):
        alt = forced(p.ir, truth, s.observations, s.interpretations, p.at)
        if not determined(alt, p.ir) or not determined(ev, p.ir) or total(alt) != total(ev):
            stake = True
    if not stake:
        return "minor"
    return "critical" if mandatory else "material"


def _ochiai(failed: set[str], tags: Mapping[str, frozenset[str]], sections: Sequence[str]) -> dict[str, float]:
    total_failed = len(failed)
    out = {}
    for sec in sections:
        ef = sum(1 for sid, t in tags.items() if sec in t and sid in failed)
        ep = sum(1 for sid, t in tags.items() if sec in t and sid not in failed)
        denom = math.sqrt(total_failed * (ef + ep))
        out[sec] = ef / denom if denom else 0.0
    return out


def _tags(p: Plan, s: Scenario, sections: Mapping[str, frozenset[str]]) -> frozenset[str]:
    tagged = {name for name, nodes in sections.items() if nodes & s.exercised}
    tagged |= set(STRESS_SECTIONS.get(s.stress or "", ()))
    if any(isinstance(pin.point.value, datetime) for pin in s.pins) and "timing" in sections:
        tagged.add("timing")
    return frozenset(t for t in tagged if t in sections)


def _metrics(rs: Sequence[Resolved]) -> tuple[float, float, float] | None:
    scored = [r for r in rs if r.passed]
    if not scored:
        return None
    fidelity = sum(1 for r in scored if r.outcome == r.scenario.evaluation.procedure_status) / len(scored)
    clarity = sum(r.s_arg for r in scored) / len(scored)
    ambiguity = sum(r.pooled.get("ambiguous", 0.0) for r in scored) / len(scored)
    return fidelity, clarity, ambiguity


def _shown_metrics(m: tuple[float, float, float] | None, n: int) -> str:
    if m is None:
        return f"{n} scenarios, none rendered"
    return f"{n} scenarios: fidelity {m[0]:.3f}, clarity {m[1]:.3f}, ambiguity {m[2]:.3f}"


def run(
    ir: core.ContractIR,
    rulebook: core.Proposal,
    store: Store,
    *,
    providers: Sequence[str] = tuple(judging.PROVIDERS),
    resolver_paraphrases: core.ParaphraseSet | None = None,
    reference: Sequence[core.Observation] = (),
    judge: JudgeCall | None = None,
    agent: AgentCall | None = None,
    workers: int = 8,
) -> Run:
    """Fuzz one contract version's prose against its IR, in the synthetic case `store`, and commit the report. The
    judges are the recorded provider layer and the renderer the recorded agent runner unless others are given."""
    judge = judge or judging.judge
    agent = agent or runner.run
    if not store.synthetic:
        raise Refused("a fuzz run's packets are synthetic evidence: run it in a synthetic case (spec I4)")
    if rulebook.kind != "clause" or rulebook.status != "active":
        raise Refused(f"{rulebook.id} is not an active clause proposal: the prose fuzzed is a rulebook that passed its "
                      f"gates ({rulebook.status})")  # fmt: skip
    prose = core.Rulebook.model_validate_json(core.canonical_json(dict(rulebook.payload)))
    if prose.contract != ir.ref:
        raise Refused(f"{rulebook.id} is the rulebook of {prose.contract.id} v{prose.contract.version}, not of {ir.id} "
                      f"v{ir.version}")  # fmt: skip
    sections = _check_prose(ir, prose)
    template_id = resolver_template(ir)
    resolver = library.template(template_id)
    if resolver_paraphrases is not None and resolver_paraphrases.template != core.Ref(id=resolver.id, version=resolver.version):
        raise FuzzError(f"the paraphrase set {resolver_paraphrases.id} is for {resolver_paraphrases.template.id}; this "
                        f"contract's procedure is asked by {resolver.id}")  # fmt: skip
    pids = [x.id for x in resolver_paraphrases.paraphrases] if resolver_paraphrases is not None else ["p0"]
    configurations = tuple((pv, pid) for pv in providers for pid in pids)
    if len(configurations) < 2:
        raise FuzzError(f"clarity compares two or more resolver configurations; {configurations} is one")
    models = _Models(tuple(providers), configurations, resolver, resolver_paraphrases, judge, agent)
    p = plan(ir, reference=reference)
    if not p.scenarios:
        raise FuzzError(f"{ir.id} has no scenario to test: its trigger's constraints admit no state ({p.whole_trigger})")
    split = _split(p)
    key = prose_hash(prose)
    rules = prose_text(prose)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(_scenario, p, s, split[s.id], rules, key, models) for s in p.scenarios]
        resolved = [f.result() for f in futures]
    terms, term_judgments = read_terms(p, rules, key, models)
    packets = _propose_packets(store, p, resolved, models)
    resolvers = [_config(pv, pid) for pv, pid in configurations]
    report = _report(p, resolved, sections, key, rulebook.id, resolvers, packets, terms)
    _commit(store, p, resolved, report, key, packets, term_judgments)
    return Run(report, p, tuple(resolved), template_id, configurations, f"{ir.id}.v{ir.version}.{key[:16]}")


def _packet_record(p: Plan, s: Scenario, packet: Packet) -> core.ScenarioPacket:
    return core.ScenarioPacket(contract=p.ir.ref, scenario=s.id, state_hash=s.state_hash, evidence_at=p.at,
                               documents=[a.id for a, _, _ in packet.documents])  # fmt: skip


def _propose_packets(store: Store, p: Plan, resolved: Sequence[Resolved], m: _Models) -> dict[str, core.Proposal]:
    """Each rendering proposed as a scenario (spec I1), in order, its documents committed first; its tests gate is the
    rendering check, run again over the same packet (the judges' answers replay from their recordings), and it must
    give what the check gave as the scenario was worked. Returns each scenario's last proposal."""
    pending: dict[str, tuple[Scenario, Packet]] = {}

    def gate(_: Store, proposal: core.Proposal) -> core.GateResult:
        scenario, packet = pending[proposal.id]
        again = Packet(packet.attempt, packet.agent, packet.agent_input, packet.documents)
        _check(p, scenario, again, _questions(p, scenario), m)
        if again.missed != packet.missed:
            raise FuzzError(f"scenario {scenario.id}'s rendering check gave another result when run again: the judges' "
                            "answers did not replay")  # fmt: skip
        if again.missed:
            missed = "; ".join(f"{x['condition']} (intended: {x['intended']}; judges read: {x['judges_read']})"
                               for x in again.missed)  # fmt: skip
            return core.GateResult(gate="tests", passed=False, detail=f"the rendering check missed: {missed}")
        return core.GateResult(gate="tests", passed=True, detail=(
            "the rendering check: every provider recovered each intended condition value from the packet, in the "
            "original wording, and every document is dated by the evidence instant"))  # fmt: skip

    store.register_gate("scenario", "tests", gate)
    out: dict[str, core.Proposal] = {}
    for r in resolved:
        for packet in r.packets:
            for art, content, claim in packet.documents:
                if store.get("synthetic_artifact", art.id, core.SyntheticArtifact) is None:
                    store.commit_synthetic_artifact(art.model_copy(update={"created_at": store.clock()}), content)
                store.commit_synthetic_claim(claim)
            payload = _packet_record(p, r.scenario, packet).model_dump(mode="json")
            pid = f"scenario-{r.scenario.id}-{core.sha256(core.canonical_json(payload))[:16]}"
            proposal = store.proposal(pid)
            if proposal is None:
                pending[pid] = (r.scenario, packet)
                proposal = store.submit_proposal(core.Proposal(
                    id=pid, case=store.case, kind="scenario", payload=payload, author=packet.agent.run[:16],
                    rationale=f"scenario {r.scenario.id}'s evidence packet, rendering {packet.attempt + 1}",
                    status="proposed",
                ))  # fmt: skip
            if (proposal.status == "active") != packet.passed:
                raise FuzzError(f"scenario {r.scenario.id}'s proposal {pid} is {proposal.status}, where its rendering "
                                f"check {'passed' if packet.passed else 'failed'}")  # fmt: skip
            if store.get("agent_run", packet.agent.run[:16], core.AgentRun) is None:
                record_run(store, packet.agent, packet.agent_input, [pid])
            out[r.scenario.id] = proposal
    return out


def _report(p: Plan, resolved: Sequence[Resolved], sections: Mapping[str, frozenset[str]], key: str, prose: str,
            resolvers: Sequence[core.JudgeConfig], packets: Mapping[str, core.Proposal],
            terms: Sequence[core.TermReading]) -> core.FuzzReport:  # fmt: skip
    ir = p.ir
    scenarios: list[core.FuzzScenario] = []
    for r in resolved:
        s = r.scenario
        transfers, amount, liquid = _prose_side(p, r, terms)
        mandatory = s.id in p.mandatory
        scenarios.append(core.FuzzScenario(
            id=s.id, state_hash=s.state_hash, packet=packets[s.id].id, split=r.split, mandatory=mandatory,
            severity=_severity(p, s, mandatory), ir_outcome=s.evaluation.procedure_status,
            ir_evaluation=s.evaluation.content_hash(), ir_transfers=list(s.evaluation.transfers),
            rendering_passed=r.passed, resolver_outcomes=r.pooled, prose_outcome=r.outcome,
            prose_transfers=transfers, amount_error=amount, liquidity_discrepancy=liquid,
        ))  # fmt: skip
    by_id = {x.id: x for x in scenarios}
    rendered = [r for r in resolved if r.passed]
    metrics = _metrics(resolved)
    if metrics is None:
        raise FuzzError(f"no packet of {ir.id}'s {len(resolved)} scenarios passed the rendering check, so nothing "
                        "measures its prose; the renderings' misses are in the run's packets")  # fmt: skip
    diverged = {r.scenario.id for r in rendered if r.outcome != r.scenario.evaluation.procedure_status
                or same_transfers(by_id[r.scenario.id]) is False}  # fmt: skip
    unread = {r.scenario.id for r in rendered if r.scenario.id not in diverged and by_id[r.scenario.id].prose_transfers is None}
    failed = diverged | {r.scenario.id for r in rendered if r.pooled.get("ambiguous", 0.0) > AMBIGUITY or r.s_arg < CLARITY}
    tags = {r.scenario.id: _tags(p, r.scenario, sections) for r in rendered}
    suspicion = _ochiai(failed, tags, list(sections))
    attributions = _attributions(p, resolved, diverged, failed, tags, suspicion)
    gates = _gates(p, resolved, scenarios, diverged, unread)
    if diverged:
        equivalence: Equivalence = "counterexample_found"
    elif len(rendered) == len(resolved) and not unread:
        equivalence = "tested_equivalent_on_suite"
    else:
        equivalence = "unsupported_or_unassessed"
    return core.FuzzReport(
        contract=ir.ref, prose=prose, prose_hash=key, resolvers=list(resolvers),
        pooling_policy=core.Ref(id=EQUAL_PER_PROVIDER.name, version=EQUAL_PER_PROVIDER.version),
        whole_trigger=p.whole_trigger, terms=list(terms), obligations=list(p.obligations), boundaries=list(p.boundaries),
        scenarios=scenarios, fidelity=metrics[0], clarity=metrics[1], ambiguity=metrics[2], hard_gates=gates,
        attributions=attributions, equivalence=equivalence,
    )  # fmt: skip


WORDING = ("prose_divergence", "contract_ambiguity")  # the causes that lie in the contract's wording


def _attributions(p: Plan, resolved: Sequence[Resolved], diverged: set[str], failed: set[str],
                  tags: Mapping[str, frozenset[str]], suspicion: Mapping[str, float]) -> list[core.Attribution]:  # fmt: skip
    out: list[core.Attribution] = []
    for r in resolved:
        s = r.scenario
        if not r.passed:
            missed = "; ".join(f"{m['condition']} (intended: {m['intended']}; judges read: {m['judges_read']})"
                               for m in r.packets[-1].missed)  # fmt: skip
            out.append(core.Attribution(scenario=s.id, cause="rendering", status="open", evidence=(
                f"the rendering check failed {len(r.packets)} renderings; the last missed: {missed}. The readers did not "
                "recover the intended state from the packet: the renderer, the check's question or the readers may be at "
                "fault, which the check alone does not tell apart (A09 §10.1)")))  # fmt: skip
            continue
        if s.id not in failed:
            continue
        ranked = sorted(tags[s.id], key=lambda sec: (-suspicion[sec], SECTION_ORDER.index(sec)))
        clauses = ", ".join(f"{sec} ({suspicion[sec]:.2f})" for sec in ranked) or "none states a condition it turns on"
        pooled = ", ".join(f"{k} {v:.2f}" for k, v in sorted(r.pooled.items(), key=lambda kv: -kv[1]))
        what = (f"IR {s.evaluation.procedure_status}, prose {r.outcome} (pooled: {pooled}; S_arg {r.s_arg:.2f})"
                + ("" if s.id in diverged else "; outcome agrees, but over the scenario thresholds"))  # fmt: skip
        decisive = sorted(c for c in p.lowered.conditions if c in s.exercised)
        evidence = (f"suspected clauses: {clauses}. {what}. Conditions its outcome turns on: {decisive or 'none alone'}"
                    + (f"; stress class {s.stress}" if s.stress else "") + ".")  # fmt: skip
        causes: list[str] = []
        if s.id in diverged and r.outcome != "ambiguous":
            causes.append("prose_divergence")
        if r.outcome == "ambiguous" or r.pooled.get("ambiguous", 0.0) > AMBIGUITY or r.s_arg < CLARITY:
            causes.append("contract_ambiguity")
        if s.stress in ("conflicting_sources", "revised_value"):
            causes.append("source_selection")
        causes += ["interpreter", "scenario_truth"]
        if p.lowered.constraints:
            causes.append("world_constraints")
        wording = [(cast(core.ProseSectionName, sec), suspicion[sec]) for sec in ranked]
        for cause in causes:
            out.append(core.Attribution(scenario=s.id, cause=cast(Cause, cause), status="open", evidence=evidence,
                                        suspected_clauses=wording if cause in WORDING else []))  # fmt: skip
    return out


def _gates(p: Plan, resolved: Sequence[Resolved], scenarios: Sequence[core.FuzzScenario],
           diverged: set[str], unread: set[str]) -> list[HardGate]:  # fmt: skip
    by = {x.id: x for x in scenarios}
    passed = {r.scenario.id for r in resolved if r.passed}
    critical = sorted(sid for sid in diverged if by[sid].severity == "critical" and by[sid].mandatory)
    # The gate says no critical scenario's outcome or transfers differ; where the prose's payment is not established,
    # that is not shown, so the gate does not pass.
    unshown = sorted(sid for sid in unread if by[sid].severity == "critical" and by[sid].mandatory)
    detail = "; ".join(x for x in (
        f"mismatches on critical mandatory scenarios, none reviewed: {critical}" if critical else "",
        f"critical mandatory scenarios whose prose payment is not established (its payment terms are not read as "
        f"stated): {unshown}" if unshown else "") if x) or (
        "no critical mandatory scenario's prose outcome or transfers differ from the IR's")  # fmt: skip
    gates = [HardGate(name=HARD_GATES[0], passed=not critical and not unshown, detail=detail)]
    policy = [r for r in resolved if r.scenario.stress in ("conflicting_sources", "revised_value", "evidence_missing_at_expiry")]
    # A substitution is another determinate outcome or payment than the IR's policy gives; an AMBIGUOUS answer is no
    # substitution but ambiguity, which the attributions and the ambiguity score carry.
    wrong = sorted(r.scenario.id for r in policy if r.scenario.id in diverged and r.outcome != "ambiguous")
    untested = sorted(r.scenario.id for r in policy if not r.passed)
    detail = (f"source-priority and missing-data scenarios: {sorted(r.scenario.id for r in policy)}; the prose resolves "
              f"to another determinate outcome or payment on {wrong or 'none'}; untested (rendering failed): "
              f"{untested or 'none'}")  # fmt: skip
    if not policy:
        detail = "no scenario of this contract exercises a source priority, a revision or a missing-data rule"
    gates.append(HardGate(name=HARD_GATES[1], passed=not wrong and not untested, detail=detail))
    unmet = [f"{'+'.join(o.conditions)}: {o.status}" for o in p.obligations if o.status == "uncovered"]
    unmet += [f"{'+'.join(o.conditions)}: witness {w} not rendered" for o in p.obligations if o.status == "met"
              for w in o.witnesses if w not in passed]  # fmt: skip
    unmet += [f"{b.condition} at {b.point}: no scenario" for b in p.boundaries if b.feasible and b.scenario is None]
    unmet += [f"{b.condition} at {b.point}: scenario {b.scenario} not rendered" for b in p.boundaries
              if b.feasible and b.scenario is not None and b.scenario not in passed]  # fmt: skip
    unmet += [f"{b.condition} at {b.point}: unconstructed ({'; '.join(b.excluded_by)})" for b in p.boundaries
              if b.status == "unconstructed"]  # fmt: skip
    unmet += list(p.notes)
    infeasible = sum(1 for b in p.boundaries if b.status == "infeasible")
    redundant = [o.conditions[0] for o in p.obligations if o.status == "redundant"]
    gates.append(HardGate(name=HARD_GATES[2], passed=not unmet, detail=(
        f"unmet: {unmet}" if unmet else
        f"every feasible obligation met by rendered scenarios ({len(p.obligations)} obligations; redundant under the "
        f"IR's constraints: {redundant or 'none'}; {infeasible} boundary points proved infeasible, with their "
        "proofs)")))  # fmt: skip
    gates.append(HardGate(name=HARD_GATES[3], passed=True, detail=(
        "this run accepts no ambiguity: accepting one is a human's decision at G3, recorded with its consequence and "
        "scope in an AcceptanceRecord (spec §8.6 step 7)")))  # fmt: skip
    dev = [r for r in resolved if r.split == "development"]
    held = [r for r in resolved if r.split == "held_out"]
    held_metrics = _metrics(held)
    gates.append(HardGate(name=HARD_GATES[4], passed=held_metrics is not None, detail=(
        f"development {_shown_metrics(_metrics(dev), len(dev))}; held out {_shown_metrics(held_metrics, len(held))}")))
    return gates


# ---------------------------------------------------------------- committed state


def _commit(store: Store, p: Plan, resolved: Sequence[Resolved], report: core.FuzzReport, key: str,
            packets: Mapping[str, core.Proposal], term_judgments: Sequence[core.Judgment] = ()) -> None:  # fmt: skip
    """The run's records, on the caller's thread, after its packets are proposed: each scenario's own definition, every
    judgment, and the report. Records already committed by an earlier run stand as they are."""
    for s in p.scenarios:
        if store.get("synthetic_artifact", s.artifact, core.SyntheticArtifact) is None:
            art = core.SyntheticArtifact(id=s.artifact, scenario=s.id, generator="requisite.fuzz.scenarios v1",
                                         media_type="application/json", size=len(s.content), published_at=None,
                                         created_at=store.clock())  # fmt: skip
            store.commit_synthetic_artifact(art, s.content)
    for r in resolved:
        for packet in r.packets:
            for j in packet.judgments:
                store.commit_judgment(j)
        for j in r.judgments:
            store.commit_judgment(j)
    for j in term_judgments:
        store.commit_judgment(j)
    store.commit("fuzz_report", f"{p.ir.id}.v{p.ir.version}.{key[:16]}", report)


def reports(store: Store) -> list[core.FuzzReport]:
    return store.all("fuzz_report", core.FuzzReport)


# ---------------------------------------------------------------- a case's contract, as the CLI runs it


def contract_in_case(store: Store, spec: str) -> core.ContractIR:
    """The contract a run names: a path to a ContractIR's JSON, or the id of a contract IR committed in the case (a
    market of a filed series whose reading passed its gates: requisite/contract/harvest.commit), its latest version
    unless "id@version" names one."""
    path = Path(spec)
    if path.suffix == ".json" and path.is_file():
        return core.ContractIR.model_validate_json(path.read_text())
    cid, _, version = spec.partition("@")
    found = [ir for ir in store.all("contract", core.ContractIR)
             if ir.id == cid and (not version or ir.version == int(version))]  # fmt: skip
    if not found:
        raise Refused(f"no contract {spec!r} is committed in case {store.case} (a reading of its series passes its gates "
                      "first: requisite contract read), and no contract IR file at that path")  # fmt: skip
    return max(found, key=lambda ir: ir.version)


def active_paraphrases(store: Store, template_id: str) -> core.ParaphraseSet | None:
    """The case's active paraphrase set for a template, or None where the case has none (then K is the providers
    in the original wording)."""
    for p in store.all("proposal", core.Proposal):
        current = store.proposal(p.id)
        if (current is not None and current.kind == "paraphrase_set" and current.status == "active"
                and dict(current.payload).get("template", {}).get("id") == template_id):  # fmt: skip
            return core.ParaphraseSet.model_validate_json(core.canonical_json(dict(current.payload)))
    return None


def _prose_of(ir: core.ContractIR) -> type[ProseDraft]:
    """ProseDraft held to this IR as the drafter answers (prose_problems), so the runner returns what keeps it from being
    the contract's rulebook to the drafter with the reason, as any refused answer is (spec §13)."""

    class ProseOfThisContract(ProseDraft):
        @model_validator(mode="after")
        def _is_the_contracts(self) -> ProseOfThisContract:
            core.Rulebook(contract=ir.ref, sections=[core.RulebookSection(name=x.name, text=x.text,
                                                                          ir_nodes=list(x.ir_nodes))
                                                     for x in self.sections])  # spec §8.5's sections  # fmt: skip
            problems = prose_problems(ir, self)
            if problems:
                raise ValueError("; ".join(problems))
            return self

    return ProseOfThisContract


def rulebook_gate(store: Store, proposal: core.Proposal) -> core.GateResult:
    """The tests gate of a clause proposal: its rulebook is its contract's (prose_problems), the contract being the IR
    committed in the case under the rulebook's reference."""
    rb = core.Rulebook.model_validate_json(core.canonical_json(dict(proposal.payload)))
    found = [ir for ir in store.all("contract", core.ContractIR) if ir.ref == rb.contract]
    if not found:
        return core.GateResult(gate="tests", passed=False,
                               detail=f"no contract {rb.contract.id} v{rb.contract.version} is committed in this case")  # fmt: skip
    problems = prose_problems(found[0], rb)
    if problems:
        return core.GateResult(gate="tests", passed=False, detail="; ".join(problems))
    return core.GateResult(gate="tests", passed=True, detail=(
        f"every section cites only {rb.contract.id}'s nodes and observables, and every condition that decides its "
        "settlement is stated"))  # fmt: skip


def clause_payload(ir: core.ContractIR, prose: Prose) -> dict[str, Any]:
    """A clause proposal's payload: the prose as the contract's rulebook, validated by the clause's schema gate, which
    says why where it is not one."""
    return {"contract": ir.ref.model_dump(mode="json"),
            "sections": [{"name": x.name, "text": x.text, "ir_nodes": list(x.ir_nodes)} for x in prose.sections]}  # fmt: skip


def clause_id(payload: Mapping[str, Any]) -> str:
    """A clause proposal's id, its payload's hash: the same prose is the same proposal."""
    return f"clause-{core.sha256(core.canonical_json(dict(payload)))[:16]}"


def propose_prose(case: Store, ir: core.ContractIR, prose: ProseDraft, author: str) -> core.Proposal:
    """The prose proposed as the contract's rulebook (kind "clause"; spec I1), through its gates. The same prose proposed
    again is the proposal it already is."""
    case.register_gate("clause", "tests", rulebook_gate)
    payload = clause_payload(ir, prose)
    pid = clause_id(payload)
    existing = case.proposal(pid)
    if existing is not None:
        return existing
    return case.submit_proposal(core.Proposal(
        id=pid, case=case.case, kind="clause", payload=payload, author=author, status="proposed",
        rationale=f"rulebook prose of {ir.id} v{ir.version}, generated from its IR (spec §8.5)",
    ))  # fmt: skip


def prose_proposal(case: Store, ir: core.ContractIR, agent: AgentCall | None = None) -> core.Proposal:
    """The clause drafter's rulebook prose of the IR (spec §8.5), proposed in the case, its run's trace beside it."""
    agent_input = {"ir": ir.model_dump(mode="json")}
    result = (agent or runner.run)("clause_drafter", agent_input, _prose_of(ir), effort="high")
    if done := made_from(case, result):
        return done[0]
    proposal = propose_prose(case, ir, cast(ProseDraft, result.output), result.run[:16])
    record_run(case, result, agent_input, [proposal.id])
    return proposal


def fuzz_store(case: Store, ir: core.ContractIR) -> Store:
    """The run's synthetic case, beside the real one: its packets never enter the real case (spec I4)."""
    return Store(case.root / "fuzz" / f"{ir.id}.v{ir.version}", case=f"{case.case}/fuzz/{ir.id}.v{ir.version}",
                 synthetic=True, clock=case.clock)  # fmt: skip


def from_case(case_dir: Path, contract: str, *, prose: Path | None = None, reference: Path | None = None,
              providers: Sequence[str] = tuple(judging.PROVIDERS), workers: int = 8,
              judge: JudgeCall | None = None, agent: AgentCall | None = None) -> tuple[Run, Store]:  # fmt: skip
    """`requisite fuzz <contract> --case <dir>`: a case's contract fuzzed against its prose (the file given, or the
    clause drafter's), with the case's active paraphrase set for the resolver template where it has one, and the
    reference observations given (a JSON list of core.Observation) for values the IR reads only through a computation."""
    x = prepare(case_dir, contract, prose=prose, reference=reference, agent=agent)
    return run(x.ir, x.rulebook, x.store, providers=providers, resolver_paraphrases=x.paraphrases, reference=x.reference,
               judge=judge, agent=agent, workers=workers), x.store  # fmt: skip


@dataclass(frozen=True)
class Prepared:
    """What a fuzz run of a case's contract reads: the case, its contract, the rulebook fuzzed, the case's active
    paraphrase set for the resolver template, the reference observations, and the run's synthetic case."""

    case: Store
    ir: core.ContractIR
    rulebook: core.Proposal
    paraphrases: core.ParaphraseSet | None
    reference: tuple[core.Observation, ...]
    store: Store


def prepare(case_dir: Path, contract: str, *, prose: Path | None = None, reference: Path | None = None,
            agent: AgentCall | None = None) -> Prepared:  # fmt: skip
    """A case's contract and its rulebook (the prose file given, or the clause drafter's), as `from_case` fuzzes them."""
    case = open_case(case_dir)
    ir = contract_in_case(case, contract)
    if prose is not None:
        text = ProseDraft.model_validate_json(prose.read_text())
        rulebook = propose_prose(case, ir, text, f"file:{core.sha256(prose.read_text())[:12]}")
    else:
        rulebook = prose_proposal(case, ir, agent)
    if rulebook.status != "active":
        detail = "; ".join(f"{g.gate}: {g.detail}" for g in rulebook.gates if not g.passed)
        raise FuzzError(f"the prose of {ir.id} v{ir.version} is not its rulebook ({rulebook.id}, {rulebook.status}): {detail}")
    pset = active_paraphrases(case, resolver_template(ir))
    observed = ([core.Observation.model_validate_json(core.canonical_json(o)) for o in json.loads(reference.read_text())]
                if reference is not None else [])  # fmt: skip
    return Prepared(case, ir, rulebook, pset, tuple(observed), fuzz_store(case, ir))


def summary(r: Run, store: Store) -> str:
    """The run as the CLI prints it."""
    rep, p = r.report, r.plan
    providers = list(dict.fromkeys(pv for pv, _ in r.configurations))
    pids = list(dict.fromkeys(pid for _, pid in r.configurations))
    k = (f"K = {len(r.configurations)}: {len(providers)} providers x {len(pids)} paraphrases ({', '.join(pids)})"
         if len(pids) > 1 else f"K = {len(r.configurations)}: the providers alone, in the original wording (no "
         f"paraphrase set for {r.resolver_template} is active in the case)")  # fmt: skip
    status = {st: sum(1 for o in rep.obligations if o.status == st) for st in ("met", "uncovered", "redundant", "vacuous")}
    feasible = [b for b in rep.boundaries if b.feasible]
    held = sum(1 for x in rep.scenarios if x.split == "held_out")
    passed = sum(1 for x in rep.scenarios if x.rendering_passed)
    lines = [
        f"fuzz {rep.contract.id} v{rep.contract.version}, prose {rep.prose_hash[:12]}, resolver template "
        f"{r.resolver_template}, {k}",
        f"coverage: trigger {rep.whole_trigger}; obligations {status}; boundary points {len(rep.boundaries)}, "
        f"{len(feasible)} feasible, {sum(1 for b in feasible if b.scenario)} with a scenario",
        f"scenarios: {len(rep.scenarios)} ({held} held out); renderings passed {passed} of {len(rep.scenarios)}",
        f"fidelity {rep.fidelity:.3f} (starting threshold >= {FIDELITY}: {'met' if rep.fidelity >= FIDELITY else 'missed'}); "
        f"clarity {rep.clarity:.3f} (>= {CLARITY}: {'met' if rep.clarity >= CLARITY else 'missed'}); "
        f"ambiguity {rep.ambiguity:.3f} (<= {AMBIGUITY}: {'met' if rep.ambiguity <= AMBIGUITY else 'missed'})",
    ]
    for g in rep.hard_gates:
        lines.append(f"  gate {g.name}: {'passed' if g.passed else 'NOT PASSED'}: {g.detail}")
    lines.append(f"equivalence: {rep.equivalence}")
    by = {x.scenario.id: x for x in r.resolved}
    for x in rep.scenarios:
        res = by[x.id]
        stress = f" [{res.scenario.stress}]" if res.scenario.stress else ""
        pins = f" at {', '.join(pin.shown() for pin in res.scenario.pins)}" if res.scenario.pins else ""
        lines.append(f"  {x.id} {res.scenario.role}{stress}{pins} ({x.split}, {x.severity}): IR {x.ir_outcome}, prose "
                     f"{x.prose_outcome}; rendering {'passed' if x.rendering_passed else 'FAILED'} after "
                     f"{len(res.packets)} render(s); P(ambiguous) {res.pooled.get('ambiguous', 0.0):.2f}, S_arg {res.s_arg:.2f}")  # fmt: skip
    for a in rep.attributions:
        lines.append(f"  attribution {a.scenario} {a.cause} ({a.status}): {a.evidence}")
    for b in rep.boundaries:
        if not b.feasible:
            lines.append(f"  boundary {b.condition} at {b.point}: {b.status} ({'; '.join(b.excluded_by)})")
    for note in p.notes:
        lines.append(f"  note: {note}")
    lines.append(f"committed: fuzz_report {r.key} in {store.root}")
    return "\n".join(lines)


__all__ = ["AMBIGUITY", "ASKED", "CLARITY", "asked_node", "question_prefix", "FIDELITY", "RERENDERS", "STRESS", "THRESHOLDS", "Run", "active_paraphrases", "determined",
           "forced", "total",
           "contract_in_case", "from_case", "propose_prose", "prose_problems", "prose_proposal", "rulebook_gate", "fuzz_store", "prose_hash", "prose_text", "reports",
           "resolver_template", "run", "summary"]  # fmt: skip
