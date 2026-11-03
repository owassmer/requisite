"""A clause that fails the wording test is revised, and the test runs again (spec §8.6 step 7: "The drafter revises it and
fuzzing reruns, until thresholds pass or a human accepts the remaining ambiguity"; addendum A09 §10.1, §10.4; A01 §2.5).

The base. The contract's rulebook is fuzzed (requisite.fuzz.run; its recorded answers replay), and its committed report
is the one revised from. A rulebook this flow made is never a base: its rounds are numbered from the drafted rulebook.

A round, numbered from 1:
- Suspicion. The sections the report's open wording attributions (prose divergence, contract ambiguity) name, on the
  development scenarios only, each revised whatever its rank, so the held-out set never steers what is revised. A round
  runs only while the development scenarios miss spec §8.6's thresholds: whether to revise is decided by the
  development set too.
- Revision. Each suspected section is revised in a revision of its own, so a pass can be laid to one section. The
  clause drafter (role clause_reviser) is given the IR, the rulebook as its readers read it, the section and the IR nodes
  it states, and the evidence of the failing development scenarios that suspect it: what each fixes, the documents the
  readers saw, how the IR resolves it, and what the readers answered and where they fell short. It never sees a held-out
  scenario. Its revised sections replace the base's, and the rulebook is proposed as a clause (spec I1), whose tests gate
  holds it to the contract (requisite.fuzz.run.rulebook_gate) and to differing from the base only in the section it
  revises: one that also changes another section is refused, and where that section is not suspected the report says the
  fix lies outside the suspected sections, so the attribution points at the wrong place. Where the drafter finds the
  readers were asked about a case the IR itself does not decide, that is reported as a finding for the contract, never
  settled in prose.
- Re-run. Each revision is fuzzed over the same plan and the same split, which is the contract version's
  (requisite.fuzz.run._split); that every scenario keeps its abstract state and its set is checked on the two reports,
  run by run. Only the resolvers' answers are new: the scenarios, their packets and the rendering check are the base's,
  replayed.
- Attribution by evidence. A suspicion is confirmed for a failing development scenario when revising its section alone
  turns that scenario into a passing one; otherwise the scenario is not resolved by that revision, and its attribution
  stays open. Where several sections are each confirmed, the rulebook combining their revisions is fuzzed once more, and
  that run is the one reported; where one is, its own run is.
- The report. The run reported beside its base, on the development and the held-out scenarios separately: fidelity,
  clarity and ambiguity, before and after. A pass counts only on the held-out scenarios, which no revision saw: a run
  that meets the thresholds on the development scenarios and misses them on the held-out ones is reported as exactly
  that.

Rounds go on from the run reported while the development scenarios still miss the thresholds and each round fails fewer
of them than its base. Every round after the first is flagged: by then a held-out result has been reported for a revised
rulebook, so the held-out set is no longer unseen (A01 §2.5), and a flagged round's pass is never presented as
held-out-validated.

Examples (spec §8.5: "the coverage scenarios from §8.6 that passed fuzzing, rendered with their IR outcomes"). A run
that passes (every hard gate, and the thresholds on the development and on the held-out scenarios) of a rulebook no
flagged round made publishes its passing coverage scenarios (MC/DC witnesses and boundary values, spec §8.6 step 1)
as the rulebook's examples: each a core.ScenarioExample, its description rendered by code from the scenario and its
outcome the evaluator's, committed in the case under the rulebook. A run that does not pass publishes none.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import model_validator

from requisite import core
from requisite.agents import runner
from requisite.agents.answers import PacketDraft, ProseSection, SectionsRevised
from requisite.agents.trace import record_run
from requisite.contract import evaluate
from requisite.core.kleene import Kleene
from requisite.core.results import HardGate
from requisite.fuzz import run as fuzz
from requisite.fuzz import statements
from requisite.fuzz.run import AMBIGUITY, CLARITY, FIDELITY, SECTION_ORDER, SECTION_TITLES, WORDING
from requisite.fuzz.scenarios import FuzzError, Plan, Scenario
from requisite.judgment import providers as judging
from requisite.store import GateCheck, Refused, Store, open_case

ROLE = "clause_reviser"  # the clause drafter revising a section (requisite/agents/prompts/clause_reviser.md)
COMBINER = "requisite.fuzz.revise"  # the author of a rulebook code combines from confirmed revisions
EXAMPLES = "rulebook_example"  # a rulebook's published examples, each a core.ScenarioExample keyed "<rulebook>|<scenario>"
COVERAGE_SET = ("coverage", "boundary")  # spec §8.6 step 1: MC/DC witnesses and boundary values
Split = Literal["development", "held_out"]
SPLIT_WORDS = {"development": "development", "held_out": "held out"}

# ---------------------------------------------------------------- where a run stands


@dataclass(frozen=True)
class SplitScore:
    """One split of a run: its scenarios, those whose rendering passed, those of them that fail, and spec §8.6's scores
    over the rendered ones (`standing`)."""

    split: Split
    scenarios: tuple[str, ...]
    rendered: tuple[str, ...]
    failing: tuple[str, ...]
    scores: tuple[float, float, float] | None

    @property
    def meets(self) -> bool:
        """Every scenario rendered and scored, and the scores within spec §8.6's starting thresholds."""
        if not self.scenarios or len(self.rendered) != len(self.scenarios) or self.scores is None:
            return False
        f, c, a = self.scores
        return f >= FIDELITY and c >= CLARITY and a <= AMBIGUITY

    def passing(self, sid: str) -> bool:
        return sid in self.rendered and sid not in self.failing

    def shown(self) -> str:
        n = f"{len(self.scenarios)} scenarios"
        if self.scores is None:
            return f"{n}, none rendered: no score"
        f, c, a = self.scores
        unrendered = [s for s in self.scenarios if s not in self.rendered]
        tail = f"; not rendered: {', '.join(unrendered)}" if unrendered else ""
        failing = f" (failing: {', '.join(self.failing)})" if self.failing else ""
        verdict = "meets the thresholds" if self.meets else "misses the thresholds"
        return f"{n}: fidelity {f:.3f}, clarity {c:.3f}, ambiguity {a:.3f}: {verdict}{failing}{tail}"


@dataclass(frozen=True)
class Standing:
    """A fuzz run of one rulebook, its development and held-out scenarios scored apart (A09 §10.4)."""

    run: fuzz.Run
    rulebook: core.Proposal
    development: SplitScore
    held_out: SplitScore

    @property
    def gates_failed(self) -> tuple[HardGate, ...]:
        return tuple(g for g in self.run.report.hard_gates if not g.passed)

    @property
    def passes(self) -> bool:
        """Every hard gate passed, and the thresholds met on the development and on the held-out scenarios."""
        return not self.gates_failed and self.development.meets and self.held_out.meets

    def split(self, name: Split) -> SplitScore:
        return self.development if name == "development" else self.held_out


def standing(r: fuzz.Run, rulebook: core.Proposal) -> Standing:
    """A run's standing, read from its committed report: each scenario's set, whether its rendering passed, and whether
    it fails, which is whether the report opened an attribution for it although its rendering passed (the report's own
    judgment of failure, whatever it measures). The scores of a set are the run's own (requisite.fuzz.run._metrics,
    which the report's held_out_reported_separately gate states), over that set's scenarios: clarity needs each
    scenario's S_arg, which the report does not record."""
    if r.report.prose != rulebook.id:
        raise FuzzError(f"fuzz report {r.key} is of rulebook {r.report.prose}, not {rulebook.id}")
    attributed = {a.scenario for a in r.report.attributions}

    def score(split: Split) -> SplitScore:
        xs = [x for x in r.report.scenarios if x.split == split]
        rendered = [x.id for x in xs if x.rendering_passed]
        ids = {x.id for x in xs}
        return SplitScore(split, tuple(x.id for x in xs), tuple(rendered),
                          tuple(sid for sid in rendered if sid in attributed),
                          fuzz._metrics([x for x in r.resolved if x.scenario.id in ids]))  # fmt: skip

    return Standing(r, rulebook, score("development"), score("held_out"))


def suspected(st: Standing) -> dict[str, tuple[str, ...]]:
    """Each section the report's open wording attributions suspect on development scenarios, in spec §8.5's order, with
    those scenarios. Every section an attribution names is revised, so its suspicion score is not read: which sections a
    scenario's attribution names is code's, from the IR nodes the scenario exercises, and whether it is attributed at all
    is its own result. Held-out attributions are never read here, so the held-out set cannot steer what is revised."""
    development = set(st.development.scenarios)
    out: dict[str, list[str]] = {}
    for a in st.run.report.attributions:
        if a.status != "open" or a.cause not in WORDING or a.scenario not in development:
            continue
        for name, _ in a.suspected_clauses:
            if a.scenario not in out.setdefault(name, []):
                out[name].append(a.scenario)
    return {n: tuple(out[n]) for n in SECTION_ORDER if n in out}


# ---------------------------------------------------------------- rulebooks and their revisions


def rulebook_of(p: core.Proposal) -> core.Rulebook:
    return core.Rulebook.model_validate_json(core.canonical_json(dict(p.payload)))


def compose(base: core.Rulebook, sections: Sequence[ProseSection | core.RulebookSection]) -> core.Rulebook:
    """The base rulebook with the given sections in place of its own."""
    by = {x.name: x for x in sections}
    unknown = sorted(set(by) - {s.name for s in base.sections})
    if unknown:
        raise ValueError(f"the rulebook has no section {unknown}")
    return core.Rulebook(contract=base.contract, sections=[
        core.RulebookSection(name=s.name, text=by[s.name].text, ir_nodes=list(by[s.name].ir_nodes)) if s.name in by
        else s for s in base.sections])  # fmt: skip


def changed(base: core.Rulebook, revised: core.Rulebook) -> tuple[str, ...]:
    """The sections in which a rulebook differs from its base, in order."""
    old = {s.name: s for s in base.sections}
    return tuple(s.name for s in revised.sections if old.get(s.name) != s)


def made_by_revision(case: Store, p: core.Proposal) -> bool:
    """Whether a rulebook is one this flow made: a revision the clause drafter proposed as clause_reviser, or one code
    combined from confirmed revisions."""
    if p.author == COMBINER:
        return True
    return any(t.role == ROLE and p.id in t.proposals for t in case.all("agent_run", core.AgentRun))


@dataclass(frozen=True)
class _Pending:
    """What a revision's tests gate holds it to: its base, the base's report, the sections it revises, and the sections
    the report suspects."""

    base: core.Proposal
    report: str
    revises: frozenset[str]
    suspected: frozenset[str]


def revision_gate(pending: Mapping[str, _Pending]) -> GateCheck:
    """The tests gate of a clause proposal: the rulebook gate, and for a revision, that it differs from its base only in
    the sections it revises (spec §8.6 step 7: each a suspected section, so a pass is laid to it)."""

    def check(store: Store, proposal: core.Proposal) -> core.GateResult:
        plain = fuzz.rulebook_gate(store, proposal)
        x = pending.get(proposal.id)
        if not plain.passed or x is None:
            return plain
        assert x.revises <= x.suspected  # code revises only what the report suspects
        diff = changed(rulebook_of(x.base), rulebook_of(proposal))
        outside = [n for n in diff if n not in x.revises]
        if outside:
            why = [f"{n}, which the report suspects too and which is revised in a revision of its own" if n in x.suspected
                   else f"{n}, which fuzz report {x.report} does not suspect from its development evidence"
                   for n in outside]  # fmt: skip
            return core.GateResult(gate="tests", passed=False, detail=(
                f"it differs from rulebook {x.base.id} outside {', '.join(sorted(x.revises, key=SECTION_ORDER.index))}: "
                f"in {'; '.join(why)}. A revision changes only the sections it revises, each one the report suspects, "
                "so that a pass can be laid to them"))  # fmt: skip
        if not diff:
            return core.GateResult(gate="tests", passed=False, detail=f"it is rulebook {x.base.id} unchanged")
        return core.GateResult(gate="tests", passed=True, detail=(
            f"{plain.detail}; it differs from rulebook {x.base.id} only in {', '.join(diff)}, which fuzz report "
            f"{x.report} suspects from its development evidence"))  # fmt: skip

    return check


def _answer_of(ir: core.ContractIR, base: core.Rulebook) -> type[SectionsRevised]:
    """SectionsRevised held to being a revision of this rulebook of this contract as the drafter answers, so the runner
    returns what keeps it from being one with the reason (as requisite.fuzz.run._prose_of does). That it revises only
    the section asked is the proposal's gate, not this: a revision that needs another section is the drafter's finding,
    refused and reported, never corrected away."""

    class RevisionOfThisRulebook(SectionsRevised):
        @model_validator(mode="after")
        def _revises_this_rulebook(self) -> RevisionOfThisRulebook:
            names = [x.name for x in self.sections]
            if len(set(names)) != len(names):
                raise ValueError(f"each revised section is returned once: {names}")
            if "examples" in names:
                raise ValueError("leave the examples section out: the examples are the scenarios that pass, written by code")
            problems = fuzz.prose_problems(ir, compose(base, self.sections))
            if problems:
                raise ValueError("; ".join(problems))
            return self

    return RevisionOfThisRulebook


# ---------------------------------------------------------------- what the drafter is given


def _transfers(ts: Sequence[core.CashTransfer]) -> list[dict[str, str]]:
    return [{"at": t.at.isoformat(), "amount": str(t.amount), "currency": t.currency, "basis": t.basis} for t in ts]


def shortfalls(r: fuzz.Resolved, x: core.FuzzScenario) -> list[str]:
    """Where a scored scenario's readers fell short, in the report's fields: an outcome other than the IR's, weight on
    AMBIGUOUS, or disagreement (S_arg, which the run holds and the report does not)."""
    out = []
    if x.prose_outcome != x.ir_outcome:
        out.append(f"the readers' pooled outcome is {x.prose_outcome}; the IR's is {x.ir_outcome}")
    amb = x.resolver_outcomes.get("ambiguous", 0.0)
    if amb > AMBIGUITY:
        out.append(f"the readers put {amb:.2f} on AMBIGUOUS, more than {AMBIGUITY}")
    if r.s_arg < CLARITY:
        out.append(f"the readers' top answers agree at S_arg {r.s_arg:.2f}, less than {CLARITY}")
    return out


def _evidence(r: fuzz.Resolved, x: core.FuzzScenario) -> dict[str, Any]:
    """One failing development scenario as the drafter is given it. Not the report's attribution text: its suspicion
    scores are computed over every scored scenario, the held-out ones included."""
    s = r.scenario
    first = r.packets[0].agent_input  # the renderer's input: what the scenario fixes
    documents = cast(PacketDraft, r.packets[-1].agent.output).documents  # the packet the readers read
    return {
        "scenario": s.id,
        "kind": s.role,
        "stress_class": first["stress_class"],
        "boundary_values": [pin.shown() for pin in s.pins],
        "conditions": first["conditions"],
        "published_values": first["published_values"],
        "documents": [d.model_dump(mode="json") for d in documents],
        "the_ir_resolves": x.ir_outcome,
        "the_ir_pays": _transfers(x.ir_transfers),
        "readers": {
            "pooled": dict(x.resolver_outcomes), "outcome": x.prose_outcome, "agreement_s_arg": r.s_arg,
            "by_reader": [{"provider": j.config.provider, "paraphrase": j.config.paraphrase_id,
                           "distribution": None if j.distribution is None else dict(j.distribution)}
                          for j in r.judgments],
        },  # fmt: skip
        "falls_short": shortfalls(r, x),
    }  # fmt: skip


def reviser_input(st: Standing, section: str, scenarios: Sequence[str]) -> dict[str, Any]:
    """What the drafter is given to revise one section: the IR, the rulebook as the readers read it, the section and the
    IR nodes and observables it states, and the evidence of the failing development scenarios that suspect it. A held-out
    scenario is refused here: the drafter never sees one."""
    ir = st.run.plan.ir
    report = {x.id: x for x in st.run.report.scenarios}
    held = [sid for sid in scenarios if report[sid].split != "development"]
    if held:
        raise FuzzError(f"scenarios {held} are held out: the drafter never sees a held-out scenario (A01 §2.5)")
    rb = rulebook_of(st.rulebook)
    roots = [ir.trigger] + ([ir.payout.quantity] if ir.payout.quantity is not None else [])
    nodes: dict[str, Any] = {n.id: n for root in roots for n in core.walk(root)}
    nodes.update({o.id: o for o in ir.observables if o.id not in nodes})
    sec = next(x for x in rb.sections if x.name == section)
    by = {r.scenario.id: r for r in st.run.resolved}
    return {
        "ir": ir.model_dump(mode="json"),
        "rulebook": [{"name": x.name, "title": SECTION_TITLES[x.name], "text": x.text, "ir_nodes": list(x.ir_nodes)}
                     for x in rb.sections],
        "revise": {"name": section, "title": SECTION_TITLES[section],
                   "states": {i: nodes[i].model_dump(mode="json") for i in sec.ir_nodes}},
        "thresholds": {"ambiguous_at_most": AMBIGUITY, "agreement_s_arg_at_least": CLARITY},
        "failing_scenarios": [_evidence(by[sid], report[sid]) for sid in scenarios],
    }  # fmt: skip


# ---------------------------------------------------------------- one round


@dataclass(frozen=True)
class Revision:
    """One revision and its re-run. sections: the sections it revises (one, or the confirmed ones combined); failing: the
    failing development scenarios it answers to; proposal: the revised rulebook as proposed (None where the drafter left
    the section as it was); changed: the sections the revised rulebook changes; unsuspected: of those, the ones the
    report does not suspect (the attribution points at the wrong place); standing: its re-run, where its proposal passed
    its gates; resolved and unresolved: its failing scenarios that pass in the re-run and those that do not; regressed:
    development scenarios that passed in the base and fail in the re-run. gate: the gate result that decides it."""

    sections: tuple[str, ...]
    failing: tuple[str, ...]
    answer: SectionsRevised | None
    proposal: core.Proposal | None = None
    gate: core.GateResult | None = None
    changed: tuple[str, ...] = ()
    unsuspected: tuple[str, ...] = ()
    standing: Standing | None = None
    resolved: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    regressed: tuple[str, ...] = ()

    @property
    def refused(self) -> str | None:
        """Why the revision was refused at its gates, or None where it passed them."""
        if self.proposal is None or self.gate is None:
            return None
        if self.gate.passed and self.proposal.status == "active":
            return None
        return self.gate.detail


@dataclass(frozen=True)
class Round:
    number: int
    base: Standing
    suspected: Mapping[str, tuple[str, ...]]
    revisions: tuple[Revision, ...]
    combined: Revision | None

    @property
    def flagged(self) -> bool:
        """Every round after the first: a held-out result of a revised rulebook has been reported before it."""
        return self.number > 1

    @property
    def confirmed(self) -> tuple[Revision, ...]:
        return tuple(r for r in self.revisions if r.resolved)

    @property
    def reported(self) -> Standing | None:
        """The run this round reports: the combination's where several sections are confirmed, the one confirmed
        revision's where one is, none where none is."""
        if self.combined is not None:
            return self.combined.standing
        return self.confirmed[0].standing if len(self.confirmed) == 1 else None


@dataclass
class _Context:
    case: Store
    store: Store
    ir: core.ContractIR
    providers: tuple[str, ...]
    paraphrases: core.ParaphraseSet | None
    reference: tuple[core.Observation, ...]
    judge: fuzz.JudgeCall | None
    agent: fuzz.AgentCall | None
    workers: int
    pending: dict[str, _Pending] = field(default_factory=dict)

    def fuzz(self, rulebook: core.Proposal) -> fuzz.Run:
        return fuzz.run(self.ir, rulebook, self.store, providers=self.providers, resolver_paraphrases=self.paraphrases,
                        reference=self.reference, judge=self.judge, agent=self.agent, workers=self.workers)  # fmt: skip


def same_split(base: fuzz.Run, again: fuzz.Run) -> None:
    """A re-run's scenarios are the base's, each with its abstract state in the same set, as the two reports record them:
    refused otherwise, since then the held-out scenarios would not be the ones no revision saw (A01 §2.5). The split is
    the contract version's (requisite.fuzz.run._split); this checks what it gave, however it is drawn."""
    before = {x.id: (x.state_hash, x.split) for x in base.report.scenarios}
    after = {x.id: (x.state_hash, x.split) for x in again.report.scenarios}
    if before != after:
        moved = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        raise FuzzError(f"fuzz report {again.key} does not keep fuzz report {base.key}'s scenarios in their sets: {moved}")


def _trace(case: Store, result: runner.AgentResult[Any], agent_input: Mapping[str, Any], proposals: Sequence[str]) -> None:
    """The drafter's run, traced once with the proposals it led to (requisite.agents.trace)."""
    traces = [t for t in case.all("agent_run", core.AgentRun) if t.id == result.run[:16]]
    if not any(list(t.proposals) == list(proposals) for t in traces):
        record_run(case, result, agent_input, proposals)


def _propose(ctx: _Context, base: Standing, revised: core.Rulebook, revises: Sequence[str],
             suspected: Sequence[str], author: str, rationale: str) -> tuple[core.Proposal, core.GateResult]:  # fmt: skip
    """A revised rulebook proposed as a clause (spec I1), its tests gate holding it to its base; with the gate result that
    decides it as this revision. A proposal of the same rulebook that stands (proposed or active) is that proposal, never
    gated again (its status is its own), and this revision's gate is judged on it as it is; one that was rejected is
    proposed again, and its gates judge it again (requisite.agents.trace)."""
    payload = fuzz.clause_payload(ctx.ir, revised)
    pid = fuzz.clause_id(payload)
    pending = _Pending(base.rulebook, base.run.key, frozenset(revises), frozenset(suspected))
    existing = ctx.case.proposal(pid)
    if existing is not None and existing.status != "rejected":
        return existing, revision_gate({pid: pending})(ctx.case, existing)
    ctx.pending[pid] = pending
    ctx.case.register_gate("clause", "tests", revision_gate(ctx.pending))
    proposal = ctx.case.submit_proposal(core.Proposal(id=pid, case=ctx.case.case, kind="clause", payload=payload,
                                                      author=author, rationale=rationale, status="proposed"))  # fmt: skip
    return proposal, proposal.gates[-1]


def _rerun(ctx: _Context, base: Standing, rev: Revision) -> Revision:
    """A proposed revision fuzzed again, where it passed its gates, and its failing scenarios scored."""
    if rev.proposal is None or rev.refused is not None:
        return replace(rev, unresolved=rev.failing)
    again = ctx.fuzz(rev.proposal)
    same_split(base.run, again)
    st = standing(again, rev.proposal)
    resolved = tuple(s for s in rev.failing if st.development.passing(s))  # a revision answers to development scenarios
    regressed = tuple(s for s in base.development.scenarios if base.development.passing(s) and not st.development.passing(s))
    return replace(rev, standing=st, resolved=resolved, unresolved=tuple(s for s in rev.failing if s not in resolved),
                   regressed=regressed)  # fmt: skip


def _single(ctx: _Context, number: int, base: Standing, section: str, sus: Mapping[str, tuple[str, ...]],
            result: runner.AgentResult[Any], agent_input: Mapping[str, Any]) -> Revision:  # fmt: skip
    answer = cast(SectionsRevised, result.output)
    rb = rulebook_of(base.rulebook)
    revised = compose(rb, answer.sections)
    diff = changed(rb, revised)
    rev = Revision((section,), sus[section], answer, changed=diff, unsuspected=tuple(n for n in diff if n not in sus))
    if not diff:
        _trace(ctx.case, result, agent_input, [])
        return replace(rev, unresolved=rev.failing)
    rationale = (f"The clause drafter's revision of the {SECTION_TITLES[section]} section of rulebook {base.rulebook.id} "
                 f"({ctx.ir.id} v{ctx.ir.version}), round {number}, drawn from the development attributions of fuzz "
                 f"report {base.run.key}, which suspect it for scenarios {', '.join(rev.failing)} (spec §8.6 step 7). "
                 f"The drafter's account: {answer.rationale}")  # fmt: skip
    proposal, gate = _propose(ctx, base, revised, (section,), list(sus), result.run[:16], rationale)
    _trace(ctx.case, result, agent_input, [proposal.id])
    return _rerun(ctx, base, replace(rev, proposal=proposal, gate=gate))


def _combined(ctx: _Context, number: int, base: Standing, confirmed: Sequence[Revision],
              sus: Mapping[str, tuple[str, ...]]) -> Revision:  # fmt: skip
    names = tuple(n for n in SECTION_ORDER if any(n in r.sections for r in confirmed))
    sections = [s for r in confirmed if r.proposal is not None for s in rulebook_of(r.proposal).sections
                if s.name in r.sections]  # fmt: skip
    rb = rulebook_of(base.rulebook)
    revised = compose(rb, sections)
    failing = tuple(dict.fromkeys(s for r in confirmed for s in r.failing))
    sources = "; ".join(f"{r.sections[0]} from rulebook {r.proposal.id}" for r in confirmed if r.proposal is not None)
    rationale = (f"Rulebook {base.rulebook.id} ({ctx.ir.id} v{ctx.ir.version}) with the confirmed revisions of round "
                 f"{number} combined: {sources}. Each was confirmed when revising its section alone turned failing "
                 f"development scenarios of fuzz report {base.run.key} into passing ones (spec §8.6 step 7).")  # fmt: skip
    proposal, gate = _propose(ctx, base, revised, names, list(sus), COMBINER, rationale)
    rev = Revision(names, failing, None, proposal=proposal, gate=gate, changed=changed(rb, revised))
    return _rerun(ctx, base, rev)


def _round(ctx: _Context, number: int, base: Standing) -> Round:
    sus = suspected(base)
    inputs = {s: reviser_input(base, s, sus[s]) for s in sus}
    answer = _answer_of(ctx.ir, rulebook_of(base.rulebook))
    call = ctx.agent or runner.run
    with ThreadPoolExecutor(max_workers=max(1, min(ctx.workers, len(inputs)))) as ex:
        futures = {s: ex.submit(call, ROLE, inputs[s], answer, effort="high") for s in sus}
        results = {s: f.result() for s, f in futures.items()}
    revisions = tuple(_single(ctx, number, base, s, sus, results[s], inputs[s]) for s in sus)
    confirmed = [r for r in revisions if r.resolved]
    combined = _combined(ctx, number, base, confirmed, sus) if len(confirmed) > 1 else None
    return Round(number, base, sus, revisions, combined)


# ---------------------------------------------------------------- the rulebook's examples (spec §8.5)

TRUTH_WORDS = {Kleene.TRUE: "holds", Kleene.FALSE: "does not hold", Kleene.UNKNOWN: "is not established"}
OUTCOME_WORDS = {"resolved_yes": "resolves YES", "resolved_no": "resolves NO", "void": "is void",
                 "extended": "is extended", "pending": "is pending", "disputed": "is disputed"}  # fmt: skip


def describe(p: Plan, s: Scenario, x: core.FuzzScenario) -> str:
    """A scenario in words, by code: the values it publishes, each reading's band, each atomic condition and its truth,
    and the evaluator's outcome and transfers."""
    ir = p.ir
    roots = [ir.trigger] + ([ir.payout.quantity] if ir.payout.quantity is not None else [])
    nodes = {n.id: n for root in roots for n in core.walk(root)}
    observables = {o.id: o for o in ir.observables}
    parts = []
    for pin in s.pins:
        parts.append(f"At the boundary {pin.shown()}.")
    for f in s.facts:
        o = observables[f.observable]
        source = next((src.name for src in o.sources if src.id == f.source), f.source)
        parts.append(f"{source} publishes, for {f.at.isoformat()}, {o.description} [{o.id}] = {statements.value(f.value, o)}"
                     f" (published {f.published_at.isoformat()}{', final' if f.final else ''}).")  # fmt: skip
    for rid, v in sorted(s.interpretations.items()):
        node = nodes.get(rid)
        if isinstance(node, core.ReadValue) and isinstance(v, tuple):
            lo, hi = cast(tuple[Decimal, Decimal], v)
            last = (lo, hi) == tuple(node.bands[-1])
            parts.append(f"{statements.number(ir, node)} is {statements.band(lo, hi, last, node.unit)}.")
    for c in p.lowered.conditions:
        parts.append(f"{statements.condition(ir, nodes[c]).rstrip('.')}: {TRUTH_WORDS[s.truths[c]]}.")
    pays = "; ".join(f"{t.amount} {t.currency} ({t.basis}) at {t.at.isoformat()}" for t in x.ir_transfers)
    parts.append(f"The contract {OUTCOME_WORDS[x.ir_outcome]}" + (f", paying {pays}." if pays else "; nothing is paid."))
    return " ".join(parts)


def examples(st: Standing) -> tuple[core.ScenarioExample, ...]:
    """The run's passing coverage scenarios (spec §8.6 step 1's set: MC/DC witnesses and boundary values), in the plan's
    order, each made by the settlement function from the scenario's own inputs at the plan's instant (I8:
    requisite.contract.evaluate.example), and checked against the outcome and evaluation the committed report records
    for it."""
    p = st.run.plan
    report = {x.id: x for x in st.run.report.scenarios}
    out = []
    for r in st.run.resolved:
        s = r.scenario
        if s.role not in COVERAGE_SET or not st.split(r.split).passing(s.id):
            continue
        x = report[s.id]
        made = evaluate.example(p.ir, s.id, describe(p, s, x), s.observations, s.interpretations, at=p.at)
        if (made.expected, made.evaluation) != (x.ir_outcome, x.ir_evaluation):
            raise FuzzError(f"scenario {s.id}: the evaluator gives {made.expected} ({made.evaluation[:12]}), and fuzz "
                            f"report {st.run.key} records {x.ir_outcome} ({x.ir_evaluation[:12]})")  # fmt: skip
        out.append(made)
    return tuple(out)


def examples_text(xs: Sequence[core.ScenarioExample]) -> str:
    """The examples section as a resolver reads it."""
    return "\n".join(f"{i}. ({x.scenario_id}) {x.description}" for i, x in enumerate(xs, 1))


def publish(case: Store, st: Standing) -> tuple[core.ScenarioExample, ...]:
    """A passing run's examples, committed under its rulebook. Only a run that passes publishes; the caller decides it is
    not a flagged round's."""
    if not st.passes:
        raise Refused(f"fuzz report {st.run.key} does not pass, so it publishes no examples")
    xs = examples(st)
    for x in xs:
        case.commit(EXAMPLES, f"{st.rulebook.id}|{x.scenario_id}", x)
    return xs


def published(case: Store, rulebook: str) -> list[core.ScenarioExample]:
    """The examples committed under a rulebook, in key order."""
    found = (case.get(EXAMPLES, k, core.ScenarioExample) for k in case.keys(EXAMPLES) if k.startswith(f"{rulebook}|"))
    return [x for x in found if x is not None]


# ---------------------------------------------------------------- the flow


@dataclass(frozen=True)
class Revised:
    """What revising did: the base run, each round, why the rounds ended, and the examples published."""

    base: Standing
    rounds: tuple[Round, ...]
    stopped: str
    examples: tuple[core.ScenarioExample, ...] = ()

    @property
    def last(self) -> tuple[int, Standing]:
        """The latest run reported and its round (0: the base)."""
        for rnd in reversed(self.rounds):
            if rnd.reported is not None:
                return rnd.number, rnd.reported
        return 0, self.base

    @property
    def validated(self) -> bool:
        """The latest run passes, and no flagged round made its rulebook: its pass is held-out-validated."""
        number, st = self.last
        return st.passes and number <= 1


def revise(case: Store, ir: core.ContractIR, rulebook: core.Proposal, store: Store, *,
           providers: Sequence[str] = tuple(judging.PROVIDERS), resolver_paraphrases: core.ParaphraseSet | None = None,
           reference: Sequence[core.Observation] = (), judge: fuzz.JudgeCall | None = None,
           agent: fuzz.AgentCall | None = None, workers: int = 8) -> Revised:  # fmt: skip
    """Fuzz the rulebook (its committed report the base), then revise and fuzz again by rounds (module docstring). The
    judges are the recorded provider layer and the renderer and the drafter the recorded agent runner unless others are
    given."""
    if made_by_revision(case, rulebook):
        raise Refused(f"rulebook {rulebook.id} is a revision this flow made; its rounds are numbered from the rulebook it "
                      "was revised from, so revise from that one")  # fmt: skip
    ctx = _Context(case, store, ir, tuple(providers), resolver_paraphrases, tuple(reference), judge, agent, workers)
    base = standing(ctx.fuzz(rulebook), rulebook)
    rounds: list[Round] = []
    current = base
    while True:
        if current.development.meets:
            stopped = "the development scenarios meet the thresholds"
            break
        sus = suspected(current)
        if not sus:
            stopped = ("the development scenarios miss the thresholds, and no failing one has an open wording attribution "
                       f"that suspects a section (development: {current.development.shown()})")  # fmt: skip
            break
        rnd = _round(ctx, len(rounds) + 1, current)
        rounds.append(rnd)
        nxt = rnd.reported
        if nxt is None:
            stopped = (f"in round {rnd.number}, no revision of a suspected section turned a failing development scenario "
                       "into a passing one, so no run is reported and the attributions stay open")  # fmt: skip
            break
        if len(nxt.development.failing) >= len(current.development.failing):
            stopped = (f"round {rnd.number}'s run fails {len(nxt.development.failing)} development scenarios, no fewer "
                       f"than the {len(current.development.failing)} of its base")  # fmt: skip
            break
        current = nxt
    out = Revised(base, tuple(rounds), stopped)
    if out.validated:
        out = replace(out, examples=publish(case, out.last[1]))
    return out


def from_case(case_dir: Path, contract: str, *, prose: Path | None = None, reference: Path | None = None,
              providers: Sequence[str] = tuple(judging.PROVIDERS), workers: int = 8,
              judge: fuzz.JudgeCall | None = None, agent: fuzz.AgentCall | None = None) -> Revised:  # fmt: skip
    """`requisite fuzz revise <contract> --case <dir>`: the contract's rulebook (as `requisite fuzz` takes it) fuzzed,
    revised and fuzzed again."""
    x = fuzz.prepare(case_dir, contract, prose=prose, reference=reference, agent=agent)
    return revise(x.case, x.ir, x.rulebook, x.store, providers=providers, resolver_paraphrases=x.paraphrases,
                  reference=x.reference, judge=judge, agent=agent, workers=workers)  # fmt: skip


def publish_run(case_dir: Path, r: fuzz.Run) -> tuple[tuple[core.ScenarioExample, ...], str]:
    """`requisite fuzz`: a passing run of a rulebook this flow did not make publishes its examples (a revision's are
    published by its round, which knows whether it is flagged); the examples, and the reason where there are none."""
    case = open_case(case_dir)
    try:
        rulebook = case.proposal(r.report.prose)
        if rulebook is None:
            raise Refused(f"rulebook {r.report.prose} is not proposed in case {case.case}")
        st = standing(r, rulebook)
        if made_by_revision(case, rulebook):
            return (), (f"rulebook {rulebook.id} is a revision this flow made: its examples are published by its round "
                        "(requisite fuzz revise), which knows whether a held-out result informed it")  # fmt: skip
        if not st.passes:
            return (), f"the run does not pass ({_why_not(st)}), so it publishes no examples"
        return publish(case, st), ""
    finally:
        case.close()


# ---------------------------------------------------------------- as the CLI prints it


def _why_not(st: Standing) -> str:
    parts = [f"{SPLIT_WORDS[s.split]} {'meets' if s.meets else 'misses'} the thresholds"
             for s in (st.development, st.held_out)]  # fmt: skip
    if st.gates_failed:
        parts.append(f"hard gates not passed: {', '.join(g.name for g in st.gates_failed)}")
    return "; ".join(parts)


def _arrow(before: SplitScore, after: SplitScore) -> str:
    if before.scores is None or after.scores is None:
        return f"{before.shown()} -> {after.shown()}"
    names = ("fidelity", "clarity", "ambiguity")
    moved = ", ".join(f"{n} {a:.3f} -> {b:.3f}" for n, a, b in zip(names, before.scores, after.scores, strict=True))
    verdict = "meets the thresholds" if after.meets else "misses the thresholds"
    failing = f" (failing: {', '.join(after.failing)})" if after.failing else ""
    return f"{len(after.scenarios)} scenarios: {moved}: {verdict}{failing}"


def _gates(st: Standing) -> str:
    return "all passed" if not st.gates_failed else "not passed: " + ", ".join(g.name for g in st.gates_failed)


MEASURE = ("fidelity: the share of rendered scenarios whose resolvers' pooled outcome is the IR's; clarity: their mean "
           "S_arg; ambiguity: their mean weight on AMBIGUOUS")  # what a split's scores are (spec §8.6 step 6)


def verdict(out: Revised) -> str:
    """The latest run reported, judged by its report's scores and hard gates and claiming no more than they measure."""
    number, st = out.last
    where = f"round {number}'s run" if number else "the run of the rulebook as drafted"
    scope = f"as fuzz report {st.run.key} measures them ({MEASURE})"
    if st.passes and number <= 1:
        return (f"{where} meets spec §8.6's thresholds on the development scenarios and on the held-out scenarios, which "
                f"no revision saw, and passes every hard gate, {scope}: held-out-validated to that measure")  # fmt: skip
    if st.passes:
        return (f"{where} meets the thresholds on the development and the held-out scenarios and passes every hard gate, "
                f"{scope}; but the round is flagged: a held-out result was reported before it, so its pass is not "
                "held-out-validated and it publishes no examples")  # fmt: skip
    if st.development.meets and not st.held_out.meets:
        return (f"{where} passes on the development scenarios and fails on the held-out scenarios ({st.held_out.shown()}),"
                f" {scope}: the wording does not carry to scenarios it was never tuned to")  # fmt: skip
    if st.development.meets and st.held_out.meets:
        return f"{where} meets the thresholds on both sets, but its hard gates are {_gates(st)}"
    return f"{where} fails on the development scenarios ({st.development.shown()}); held out: {st.held_out.shown()}"


def _revision_lines(rev: Revision) -> list[str]:
    head = "combined " + ", ".join(rev.sections) if rev.answer is None else rev.sections[0]
    lines = []
    if rev.proposal is None:
        lines.append(f"  {head}: the drafter left the section as it was; not resolved by this revision: "
                     f"{', '.join(rev.unresolved)}")  # fmt: skip
    elif rev.refused is not None:
        lines.append(f"  {head}: rulebook {rev.proposal.id} refused at its gates: {rev.refused}")
        if rev.unsuspected:
            lines.append(f"    the drafter's fix needs {', '.join(rev.unsuspected)}, which the report does not suspect: "
                         "the attribution points at the wrong place")  # fmt: skip
        lines.append(f"    not resolved by this revision: {', '.join(rev.unresolved)}; their attributions stay open")
    else:
        assert rev.standing is not None
        st = rev.standing
        lines.append(f"  {head}: rulebook {rev.proposal.id}, fuzz report {st.run.key}")
        for sid in rev.failing:
            lines.append(f"    {sid}: " + ("confirmed: revising " + ("these sections" if rev.answer is None else "this section alone")
                                          + " turned it from failing to passing" if sid in rev.resolved
                                          else "not resolved by this revision; its attribution stays open"))  # fmt: skip
        lines.append(f"    development: {st.development.shown()}")
        if rev.regressed:
            lines.append(f"    development scenarios that passed in the base and fail here: {', '.join(rev.regressed)}")
    if rev.answer is not None:
        lines.append(f"    the drafter: {rev.answer.rationale}")
        for case in rev.answer.not_decided_by_the_ir:
            lines.append(f"    not decided by the IR (a finding for the contract, not its wording): {case}")
    return lines


def summary(out: Revised) -> str:
    base = out.base
    ir = base.run.plan.ir
    lines = [
        f"revise {ir.id} v{ir.version}: rulebook {base.rulebook.id}, fuzz report {base.run.key}",
        "the development/held-out split is the contract version's: every run below keeps each scenario's abstract state "
        "in the set it has here, and the drafter is shown development scenarios only",
        f"base: development {base.development.shown()}; held out {base.held_out.shown()}; hard gates {_gates(base)}",
    ]
    for rnd in out.rounds:
        flag = ("FLAGGED: a held-out result of a revised rulebook was reported before this round, so its pass is not "
                "held-out-validated and publishes no examples" if rnd.flagged
                else "the first: no held-out result has informed a revision")  # fmt: skip
        lines.append(f"round {rnd.number} ({flag})")
        lines.append("  suspected from the development evidence alone: " + "; ".join(
            f"{name} ({', '.join(sids)})" for name, sids in rnd.suspected.items()))
        for rev in rnd.revisions:
            lines += _revision_lines(rev)
        if rnd.combined is not None:
            lines += _revision_lines(rnd.combined)
        reported = rnd.reported
        if reported is None:
            lines.append("  no run is reported: no section was confirmed" if rnd.combined is None else
                         "  no run is reported: the combination did not pass its gates")  # fmt: skip
            continue
        lines.append(f"  the run reported: fuzz report {reported.run.key} (rulebook {reported.rulebook.id}); before -> after:")
        lines.append(f"    development: {_arrow(rnd.base.development, reported.development)}")
        lines.append(f"    held out (never shown to the drafter): {_arrow(rnd.base.held_out, reported.held_out)}")
        lines.append(f"    hard gates: {_gates(rnd.base)} -> {_gates(reported)}")
        for sid in dict.fromkeys(s for sids in rnd.suspected.values() for s in sids):
            by = [r.sections[0] for r in rnd.revisions if sid in r.resolved]
            lines.append(f"  attribution {sid}: " + (f"confirmed by revising {', '.join(by)}" if by else
                                                     "open: not resolved by any revision of this round"))  # fmt: skip
    lines.append(f"stopped: {out.stopped}")
    lines.append(f"verdict: {verdict(out)}")
    number, st = out.last
    if out.examples:
        lines.append(f"examples published under rulebook {st.rulebook.id} (the passing coverage scenarios, rendered by "
                     "code with the evaluator's outcomes):")  # fmt: skip
        lines += [f"  {line}" for line in examples_text(out.examples).splitlines()]
    else:
        lines.append("no examples published: " + ("its coverage scenarios include none that passes" if out.validated
                                                  else "only a held-out-validated pass publishes them"))  # fmt: skip
    return "\n".join(lines)


def examples_line(xs: Sequence[core.ScenarioExample], reason: str) -> str:
    if not xs:
        return f"no examples published: {reason}"
    return ("examples published (the passing coverage scenarios, rendered by code with the evaluator's outcomes):\n"
            + "\n".join(f"  {line}" for line in examples_text(xs).splitlines()))  # fmt: skip


__all__ = ["COMBINER", "EXAMPLES", "ROLE", "Revised", "Revision", "Round", "SplitScore", "Standing", "changed", "compose",
           "describe", "examples", "examples_line", "examples_text", "from_case", "made_by_revision", "publish",
           "publish_run", "published", "revise", "reviser_input", "revision_gate", "rulebook_of", "same_split",
           "standing", "summary", "suspected", "verdict"]  # fmt: skip
