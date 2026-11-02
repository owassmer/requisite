"""Wide discovery, orchestrated (spec §7.2–7.4; outcome 002). Code runs every step; agents only draft, judges only
answer, and a human decides at G1. Which corpus and which leads the run reaches are its own findings (decisions,
"Discovery finds what it analyzes"): nothing here names a docket, an agency or a rule.

The steps, each a function a CLI command calls:
1. draft_objective: an agent drafts the run's objective from a human's brief; it awaits G1.
2. approve: the human approves the objective (or edits, or rejects) at G1.
3. draft_frame: an agent derives the frame criteria from the approved objective; the proposal's tests gate runs them
   and freezes the frame (requisite/evidence/frame.py).
4-8. The sample, its reading, its labels, the stability test and the gold draw, by the design decisions.md fixes
   ("F4's sample design"); below. Inference and leads follow the human's gold labels.
"""

from __future__ import annotations

import random
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from fractions import Fraction
from typing import Any

from pydantic import BaseModel

from requisite import core
from requisite.agents import runner
from requisite.agents.answers import FrameDraft, ObjectiveDraft, ParaphrasesDraft
from requisite.discovery import labels, sampling
from requisite.evidence.frame import docket_documents, entries, frame_gate, measure_document
from requisite.evidence.frame import frames as evidence_frames
from requisite.evidence.items import committed, read_items
from requisite.evidence.regulations_gov import RegulationsGov
from requisite.judgment import providers
from requisite.library import templates as library
from requisite.store import Refused, Store


def _now() -> datetime:
    return datetime.now(UTC)


def _proposal_id(kind: str, payload: Mapping[str, Any]) -> str:
    return f"{kind}-{core.sha256(core.canonical_json(dict(payload)))[:16]}"


def _record_run(store: Store, result: runner.AgentResult[Any], input: Mapping[str, Any], proposals: list[str]) -> None:
    output = result.output.model_dump(mode="json") if isinstance(result.output, BaseModel) else result.output
    run = core.AgentRun(
        id=result.run[:16],
        role=result.role,
        model=result.model,
        effort=result.effort,
        input_hash=core.sha256(core.canonical_json(dict(input))),
        output_hash=core.sha256(core.canonical_json(output)),
        response=result.run,
        proposals=proposals,
        at=store.clock(),
    )
    store.commit("agent_run", run.id, run)


def draft_objective(store: Store, brief: str, today: date) -> core.Proposal:
    """Step 1: the objective_drafter's draft of the run's objective, submitted as a proposal that awaits G1."""
    input = {"brief": brief, "today": today.isoformat()}
    result = runner.run("objective_drafter", input, ObjectiveDraft, effort="high")
    draft = result.output
    objective = core.WideObjective(
        id="objective",
        version=1 + len(_proposals(store, "objective")),
        population=draft.population,
        events=draft.events,
        horizon=draft.horizon,
        loss_definition=draft.loss_definition,
        estimand=draft.estimand,
        unit="item",
    )
    payload = objective.model_dump(mode="json")
    proposal = store.submit_proposal(
        core.Proposal(
            id=_proposal_id("objective", payload),
            case=store.case,
            kind="objective",
            payload=payload,
            author=result.run[:16],
            rationale="; ".join(draft.open_questions) or "no open questions",
            status="proposed",
        )
    )
    _record_run(store, result, input, [proposal.id])
    return proposal


def approve(
    store: Store, proposal_id: str, *, gate: str, approver: str, decision: str, reason: str, scope: str
) -> core.Proposal:
    """A human's decision at a gate on one proposal, approving exactly its payload as it stands (spec §2.4; an approval
    covers one lineage: addendum A11 §11.2)."""
    proposal = store.proposal(proposal_id)
    if proposal is None:
        raise Refused(f"no proposal {proposal_id}")
    approval = core.Approval(
        id=f"{gate}-{proposal_id}-{len(store.all('approval', core.Approval)) + 1}",
        gate=gate,  # type: ignore[arg-type]
        subject=core.Ref(id=proposal_id, version=1),
        lineage_hash=core.sha256(core.canonical_json(dict(proposal.payload))),
        approver=approver,
        at=store.clock(),
        decision=decision,  # type: ignore[arg-type]
        reason=reason,
        scope=scope,
    )
    store.commit_approval(approval)
    return store.resubmit_after_approval(proposal_id)


def _proposals(store: Store, kind: str) -> list[core.Proposal]:
    """This case's proposals of a kind, each as it now stands."""
    out = []
    for p in store.all("proposal", core.Proposal):
        current = store.proposal(p.id)
        if current is not None and current.kind == kind:
            out.append(current)
    return out


def approved_objective(store: Store) -> core.WideObjective:
    """The case's objective that passed G1: its one active objective proposal."""
    active = [p for p in _proposals(store, "objective") if p.status == "active"]
    if not active:
        raise Refused("no objective has passed G1 in this case: draft one and have it approved")
    if len(active) > 1:
        raise Refused(f"{len(active)} objectives have passed G1 in this case; a run has one")
    return core.WideObjective.model_validate_json(core.canonical_json(dict(active[0].payload)))


def draft_frame(
    store: Store, as_of: datetime, client: Callable[[], RegulationsGov] = RegulationsGov
) -> core.Proposal:
    """Step 3: the frame_drafter's criteria for the approved objective, submitted as a proposal; its tests gate is the
    criteria's run and freeze (the evidence adapter's frame_gate)."""
    store.register_gate("frame", "tests", frame_gate(client))
    objective = approved_objective(store)
    input = {"objective": objective.model_dump(mode="json"), "as_of": as_of.isoformat()}
    result = runner.run("frame_drafter", input, FrameDraft, effort="high")
    d = result.output
    criteria = core.FrameCriteria(
        id="frame",
        version=1 + len(_proposals(store, "frame")),
        objective=core.Ref(id=objective.id, version=objective.version),
        source="regulations.gov",
        as_of=as_of,
        search_terms=d.search_terms,
        document_types=d.document_types,
        posted_from=d.posted_from,
        posted_to=d.posted_to,
        agencies=d.agencies,
        comment_status=d.comment_status,
        min_comments=d.min_comments,
    )
    payload = criteria.model_dump(mode="json")
    proposal = store.submit_proposal(
        core.Proposal(
            id=_proposal_id("frame", payload),
            case=store.case,
            kind="frame",
            payload=payload,
            author=result.run[:16],
            rationale=f"{d.rationale}\nExpected misses: {d.expected_misses}",
            status="proposed",
        )
    )
    _record_run(store, result, input, [proposal.id])
    return proposal


def draft_paraphrases(store: Store, template_id: str) -> core.Proposal:
    """A template's frozen paraphrase set (spec §2.1: "from a frozen, reviewed set per template"): the paraphrase_writer's
    two rewordings with the original, p0, submitted as a paraphrase_set proposal. Its schema gate checks every
    paraphrase keeps the template's slots and words every answer (core.ParaphraseSet)."""
    t = library.template(template_id)
    input = {"template": t.model_dump(mode="json")}
    result = runner.run("paraphrase_writer", input, ParaphrasesDraft, effort="high")
    paraphrases = [library.original(t)] + [
        core.Paraphrase(id=p.id, target=p.target, meanings=p.meanings) for p in result.output.paraphrases
    ]
    pset = core.ParaphraseSet(
        id=f"ps.{t.id}",
        version=t.version,
        template=core.Ref(id=t.id, version=t.version),
        slots=list(t.slots),
        answers=[a.value for a in t.answers],
        paraphrases=paraphrases,
    )
    payload = pset.model_dump(mode="json")
    proposal = store.submit_proposal(
        core.Proposal(
            id=_proposal_id("paraphrase_set", payload),
            case=store.case,
            kind="paraphrase_set",
            payload=payload,
            author=result.run[:16],
            rationale=f"paraphrases of {t.id} v{t.version}",
            status="proposed",
        )
    )
    _record_run(store, result, input, [proposal.id])
    return proposal


def paraphrase_set(store: Store, template_id: str) -> core.ParaphraseSet:
    """The case's active paraphrase set for a template."""
    for p in _proposals(store, "paraphrase_set"):
        if p.status == "active" and p.payload.get("template", {}).get("id") == template_id:
            return core.ParaphraseSet.model_validate_json(core.canonical_json(dict(p.payload)))
    raise Refused(f"no active paraphrase set for template {template_id}")


# ---------------------------------------------------------------- the sample (decisions, "F4's sample design")
#
# 4. draw_sample: the phase-1 sample of the frozen frame, by the design's rule.
# 5. read_sample: the drawn items read whole and frozen (requisite/evidence/items.py).
# 6. label_phase1: both objective questions on every phase-1 item, by the phase-1 configuration.
# 7. test_stability: every stability configuration on a random subsample (spec §2.1, §7.2 step 3).
# 8. draw_gold: the gold sample, a stratified subsample of phase 1 by the judge's probabilities, for the human's labels.

PHASE1_SIZE = 1200
GOLD_SIZE = 130
STABILITY_SIZE = 60
PHASE2_CUT = 0.5
EXPOSED = ("labor", "inputs", "labor_and_inputs")  # exposure_to_measure's answers in the estimand's numerator
BANDS: tuple[tuple[int | None, str], ...] = (
    (100, "under_100"), (1_000, "100_to_999"), (10_000, "1000_to_9999"), (None, "10000_plus"),
)  # fmt: skip


def the_frame(store: Store) -> core.Frame:
    """The frame the case's active frame proposal froze."""
    active = [p for p in _proposals(store, "frame") if p.status == "active"]
    if len(active) != 1:
        raise Refused(f"{len(active)} frame proposals are active in this case; a run has one" if active
                      else "no frame proposal has passed its gates in this case: run `wide frame`")  # fmt: skip
    criteria = core.Ref(id=str(active[0].payload["id"]), version=int(active[0].payload["version"]))
    taken = [f for f in evidence_frames(store) if f.criteria == criteria]
    if len(taken) != 1:
        raise Refused(f"{len(taken)} frames are frozen for criteria {criteria.id} v{criteria.version}; a run has one")
    return taken[0]


def band(comments: int) -> str:
    """A docket's phase-1 stratum: its comment count's decade."""
    for upper, name in BANDS:
        if upper is None or comments < upper:
            return name
    raise AssertionError("the last band is unbounded")


def phase1_strata(frame: core.Frame) -> dict[str, list[core.FrameItem]]:
    """The frame's items by phase-1 stratum, in band order and, within a stratum, in the frame's order."""
    counts = {d.docket: d.comments for d in frame.dockets if d.kept}
    out: dict[str, list[core.FrameItem]] = {name: [] for _, name in BANDS}
    for item in frame.items:
        count = counts[item.docket]
        assert count is not None  # core.FrameDocket: a kept docket has its count
        out[band(count)].append(item)
    return {name: items for name, items in out.items() if items}


def allocate(sizes: Mapping[str, int], total: int) -> dict[str, int]:
    """Equal shares of `total` over the strata, in their order; a stratum with no more items than its share is taken
    whole and what it leaves is shared among the rest, until every share fits; the indivisible remainder goes one each
    to the strata in order."""
    out: dict[str, int] = {}
    rest = dict(sizes)
    budget = total
    while rest:
        share = budget // len(rest)
        whole = {k: n for k, n in rest.items() if n <= share}
        if not whole:
            break
        for k, n in whole.items():
            out[k] = n
            budget -= n
            del rest[k]
    if rest:
        share, extra = divmod(budget, len(rest))
        for i, k in enumerate(rest):
            out[k] = share + (1 if i < extra else 0)
    return {k: out[k] for k in sizes}


def seed(content_hash: str, salt: str = "") -> int:
    """A seed nobody chooses: the first 16 hex digits of a record's content hash, or of the hash of it and a salt."""
    return int((core.sha256(content_hash + salt) if salt else content_hash)[:16], 16)


def design(frame: core.Frame) -> core.Ref:
    return core.Ref(id=f"two_phase.{frame.id}", version=1)


def drawn(store: Store, phase: int) -> core.SampleDraw:
    """The case's committed draw of a phase."""
    frame = the_frame(store)
    d = store.get("sample_draw", f"{design(frame).id}.phase{phase}", core.SampleDraw)
    if d is None:
        raise Refused(f"no phase-{phase} sample is drawn in this case: run `wide {'sample' if phase == 1 else 'gold-draw'}`")
    return d


def draw_sample(store: Store) -> core.SampleDraw:
    """Step 4: the phase-1 sample, a stratified simple random sample of the frame without replacement (sampling.py)."""
    frame = the_frame(store)
    strata = phase1_strata(frame)
    sizes = {k: len(v) for k, v in strata.items()}
    n = allocate(sizes, PHASE1_SIZE)
    s = seed(frame.content_hash())
    picks = sampling.draw_phase1(phase1_frame_sizes=sizes, phase1_sample_sizes=n, seed=s)
    items = [core.DrawnItem(item=strata[k][int(i)].id, stratum=k, inclusion_prob=Fraction(n[k], sizes[k]))
             for k in strata for i in picks[k]]  # fmt: skip
    draw = core.SampleDraw(design=design(frame), phase=1, seed=s, items=items)
    store.commit("sample_draw", f"{draw.design.id}.phase1", draw)
    return draw


def read_sample(
    store: Store, client: Callable[[], RegulationsGov] = RegulationsGov, *, workers: int = 8
) -> list[core.Item]:
    """Step 5: every phase-1 item read whole and committed (items already read are not read again)."""
    ids = [d.item for d in drawn(store, 1).items]
    have = committed(store, ids)
    todo = [i for i in ids if i not in have]
    if todo:
        regulations_gov = client()
        try:
            read_items(regulations_gov, store, entries(store, todo), workers=workers)
        finally:
            regulations_gov.close()
        have = committed(store, ids)
    return [have[i] for i in ids if i in have]


def measures(store: Store) -> dict[str, str]:
    """Each kept docket's measure, bound into the exposure question: the document frame.measure_document names, stated
    by its title, agency, type, id and posting date, from the frame's committed documents listing."""
    frame = the_frame(store)
    out = {}
    for d in frame.dockets:
        if not d.kept:
            continue
        m = measure_document(docket_documents(store, frame, d), frame.as_of)
        if m is None:
            raise Refused(f"kept docket {d.docket} has neither a proposed rule nor a running comment period by as_of")
        a = m.attributes
        out[d.docket] = (f"{a.get('title')} ({a.get('agencyId')} {a.get('documentType')} {m.id}, posted "
                         f"{str(a.get('postedDate'))[:10]})")  # fmt: skip
    return out


@dataclass(frozen=True)
class Labelled:
    """What a labelling step did: the judgments it holds, and the items whose filing has no readable text, which no
    judge is asked about (each with the reason labels.item_state gives)."""

    judgments: list[core.Judgment]
    unreadable: dict[str, str]


def _states(store: Store, ids: Sequence[str]) -> tuple[list[labels.ItemState], dict[str, str]]:
    now = store.clock()
    states, unreadable = [], {}
    have = committed(store, ids)
    for item_id in ids:
        item = have.get(item_id)
        if item is None:
            raise Refused(f"sampled item {item_id} is not read: run `wide read`")
        try:
            states.append(labels.item_state(store, item, evaluated_at=now))
        except Refused as e:
            unreadable[item_id] = str(e)
    return states, unreadable


def _judge(
    store: Store, ids: Sequence[str], configs: Sequence[tuple[str, str, Any]], workers: int,
    on_done: Callable[[int, int], None] | None,
) -> Labelled:  # fmt: skip
    objective, ms = approved_objective(store), measures(store)
    psets = {t: paraphrase_set(store, t) for t in labels.OBJECTIVE_TEMPLATES}
    states, unreadable = _states(store, ids)
    asks = [a for st in states for a in labels.asks(st, labels.questions(objective, ms, st.item), psets, configs)]
    done = 0
    lock = threading.Lock()

    def tick(_: int, j: core.Judgment) -> None:
        nonlocal done
        with lock:
            done += 1
            if on_done is not None:
                on_done(done, len(asks))

    # Each response is in its cassette as it arrives, so an interrupted run resumes by replaying it; the store is
    # written on this thread, after every ask is answered.
    judgments = providers.judge_all(asks, workers=workers, on_done=tick)
    for j in judgments:
        store.commit_judgment(j)
    return Labelled(judgments, unreadable)


def label_phase1(store: Store, *, workers: int = 16, on_done: Callable[[int, int], None] | None = None) -> Labelled:
    """Step 6: both objective questions on every phase-1 item, by labels.PHASE1_CONFIG."""
    return _judge(store, [d.item for d in drawn(store, 1).items], [labels.PHASE1_CONFIG], workers, on_done)


def stability_sample(store: Store) -> list[str]:
    """A simple random subsample of the phase-1 items, STABILITY_SIZE of them, seeded by the phase-1 draw."""
    draw = drawn(store, 1)
    rng = random.Random(seed(draw.content_hash(), "stability"))  # noqa: S311 (a sample, not a secret)
    return rng.sample(sorted(d.item for d in draw.items), min(STABILITY_SIZE, len(draw.items)))


def test_stability(
    store: Store, *, workers: int = 16, on_done: Callable[[int, int], None] | None = None
) -> tuple[dict[str, labels.TemplateStability], dict[str, str]]:
    """Step 7: every configuration of labels.stability_configs on the stability subsample; each template's
    stability over the items that returned two or more distributions, and the unreadable items."""
    done = _judge(store, stability_sample(store), labels.stability_configs(), workers, on_done)
    by_template: dict[str, list[core.Judgment]] = {t: [] for t in labels.OBJECTIVE_TEMPLATES}
    for j in done.judgments:
        by_template[j.question.id.split(".")[0]].append(j)
    return {t: labels.template_stability(t, js) for t, js in by_template.items()}, done.unreadable


def predictions(store: Store) -> dict[str, tuple[float, float]]:
    """Each phase-1 item's judged P(in_population) and P(exposure in EXPOSED), from its phase-1 judgments. An item
    with no readable text, or a question the judge refused, has no prediction and is not in the result."""
    objective, ms = approved_objective(store), measures(store)
    states, _ = _states(store, [d.item for d in drawn(store, 1).items])
    spec = providers.PROVIDERS[labels.PHASE1_CONFIG[0]]
    phase1 = core.JudgeConfig(provider=spec.name, model=spec.model, model_cutoff=spec.model_cutoff,
                              paraphrase_id=labels.PHASE1_CONFIG[1], render_id=labels.PHASE1_CONFIG[2].id)  # fmt: skip
    found: dict[tuple[str, str], core.Judgment] = {
        (j.question.id, j.state_hash): j for j in store.all("judgment", core.Judgment) if j.config == phase1
    }
    out = {}
    for st in states:
        qs = labels.questions(objective, ms, st.item)
        h = st.state.content_hash()
        voice = found.get((qs["voice_in_population"].id, h))
        exposure = found.get((qs["exposure_to_measure"].id, h))
        if voice is None or exposure is None or voice.distribution is None or exposure.distribution is None:
            continue
        out[st.item.id] = (float(voice.distribution["in_population"]),
                           float(sum(exposure.distribution.get(a, 0) for a in EXPOSED)))  # fmt: skip
    return out


def gold_allocation(strata: Sequence[str]) -> dict[tuple[str, int], int]:
    """GOLD_SIZE over the nonempty phase-1 strata equally (the remainder one each in band order), each split 80/20
    between the judge's below-cut and above-cut strata, each at least two (decisions, "Phase-2 allocation")."""
    per, extra = divmod(GOLD_SIZE, len(strata))
    out = {}
    for i, h in enumerate(strata):
        g = per + (1 if i < extra else 0)
        above = max(2, int(Fraction(g, 5) + Fraction(1, 2)))
        out[(h, 0)], out[(h, 1)] = max(2, g - above), above
    return out


def draw_gold(store: Store) -> core.SampleDraw:
    """Step 8: the gold sample (sampling.draw_phase2), stratified within each phase-1 stratum by the judge's probability
    that the item is in the estimand's numerator, cut at PHASE2_CUT. Every phase-1 item needs that probability."""
    draw = drawn(store, 1)
    f = predictions(store)
    missing = [d.item for d in draw.items if d.item not in f]
    if missing:
        raise Refused(f"{len(missing)} phase-1 item(s) have no phase-1 prediction (unread, unreadable or refused), so "
                      f"the phase-2 strata are undefined for them: {', '.join(missing[:10])}")  # fmt: skip
    strata = list(dict.fromkeys(d.stratum for d in draw.items))
    allocation = gold_allocation(strata)
    stratifier = [f[d.item][0] * f[d.item][1] for d in draw.items]
    s = seed(draw.content_hash(), "phase2")
    mask = sampling.draw_phase2(phase1_stratum=[d.stratum for d in draw.items], stratifier=stratifier,
                                phase2_cuts=[PHASE2_CUT], phase2_allocation=allocation, seed=s)  # fmt: skip
    cell = [(d.stratum, int(v >= PHASE2_CUT)) for d, v in zip(draw.items, stratifier, strict=True)]
    held = {c: cell.count(c) for c in set(cell)}
    items = [core.DrawnItem(item=d.item, stratum=f"{c[0]}.bin{c[1]}",
                            inclusion_prob=Fraction(min(allocation[c], held[c]), held[c]))
             for d, c, chosen in zip(draw.items, cell, mask, strict=True) if chosen]  # fmt: skip
    gold = core.SampleDraw(design=draw.design, phase=2, seed=s, items=items)
    store.commit("sample_draw", f"{gold.design.id}.phase2", gold)
    return gold
