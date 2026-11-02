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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from fractions import Fraction
from typing import Any

from numpy import random as np_random

from requisite import cassettes, core
from requisite.agents import runner
from requisite.agents.answers import FrameDraft, ObjectiveDraft, ParaphrasesDraft
from requisite.agents.trace import made_from, record_run
from requisite.discovery import labels, sampling
from requisite.evidence import extract
from requisite.evidence import frame as framing
from requisite.evidence.federal_register import FederalRegister
from requisite.evidence.fetched import free_budget
from requisite.evidence.frame import entries, frame_gate
from requisite.evidence.frame import frames as evidence_frames
from requisite.evidence.items import committed, read_items, reextracted, stale
from requisite.evidence.mirrulations import Mirrulations
from requisite.evidence.regulations_gov import RegulationsGov
from requisite.judgment import providers
from requisite.library import templates as library
from requisite.store import Refused, Store


def _now() -> datetime:
    return datetime.now(UTC)


def _proposal_id(kind: str, payload: Mapping[str, Any]) -> str:
    return f"{kind}-{core.sha256(core.canonical_json(dict(payload)))[:16]}"


def draft_objective(store: Store, brief: str, today: date) -> core.Proposal:
    """Step 1: the objective_drafter's draft of the run's objective, submitted as a proposal that awaits G1."""
    input = {"brief": brief, "today": today.isoformat()}
    result = runner.run("objective_drafter", input, ObjectiveDraft, effort="high")
    if done := made_from(store, result):
        return done[0]  # this run's answer is already proposed (agents/trace.py)
    draft = result.output
    objective = core.WideObjective(
        id="objective",
        version=1 + len(proposals(store, "objective")),
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
    record_run(store, result, input, [proposal.id])
    return proposal


def approve(
    store: Store, proposal_id: str, *, gate: str, approver: str, decision: str, reason: str, scope: str
) -> core.Proposal:
    """A human's decision at a gate on one proposal, approving exactly its payload as it stands (spec §2.4; an approval
    covers one lineage: addendum A11 §11.2). The decision is recorded through the cassettes as the human's answer to
    that gate on that payload (provider "human"), as a gold label is: a run replaying the case from a clean clone
    replays the decision and the instant it was made, and refuses one given differently from the recording."""
    proposal = store.proposal(proposal_id)
    if proposal is None:
        raise Refused(f"no proposal {proposal_id}")
    lineage = core.sha256(core.canonical_json(dict(proposal.payload)))
    human_decision(store, gate, "proposal", proposal_id, 1, lineage, approver=approver, decision=decision, reason=reason,
                   scope=scope)  # fmt: skip
    return store.resubmit_after_approval(proposal_id)


def human_decision(
    store: Store, gate: str, kind: str, subject: str, version: int, lineage: str, *, approver: str, decision: str,
    reason: str, scope: str,
) -> core.Approval:
    """A human's decision at a gate on one subject at one lineage, recorded through the cassettes as the human's answer
    (provider "human"), as a gold label is: a run replaying the case from a clean clone replays the decision and the
    instant it was made, and refuses one given differently from the recording. `kind` names the subject in the
    recorded request ("proposal", "question")."""
    given = {"decision": decision, "reason": reason, "scope": scope}
    body = {"gate": gate, kind: subject, "lineage_hash": lineage, "approver": approver}
    recorded = cassettes.call(cassettes.HUMAN, "approve", body, lambda: ({**given, "at": store.clock().isoformat()}, 0.0),
                              budget=free_budget(), estimate_usd=0.0)  # fmt: skip
    answer = dict(recorded.response)
    differs = {k: (answer[k], v) for k, v in given.items() if answer[k] != v}
    if differs:
        raise Refused(f"{approver}'s recorded decision at {gate} on {subject} differs from the one given: "
                      + "; ".join(f"{k}: recorded {r!r}, given {g!r}" for k, (r, g) in differs.items()))  # fmt: skip
    approval = core.Approval(
        id=f"{gate}-{subject}-{len(store.all('approval', core.Approval)) + 1}",
        gate=gate,  # type: ignore[arg-type]
        subject=core.Ref(id=subject, version=version),
        lineage_hash=lineage,
        approver=approver,
        at=datetime.fromisoformat(str(answer["at"])),
        decision=decision,  # type: ignore[arg-type]
        reason=reason,
        scope=scope,
    )
    store.commit_approval(approval)
    return approval


def proposals(store: Store, kind: str) -> list[core.Proposal]:
    """This case's proposals of a kind, each as it now stands."""
    out = []
    for p in store.all("proposal", core.Proposal):
        current = store.proposal(p.id)
        if current is not None and current.kind == kind:
            out.append(current)
    return out


def approved_objective(store: Store) -> core.WideObjective:
    """The case's objective that passed G1: its one active objective proposal."""
    active = [p for p in proposals(store, "objective") if p.status == "active"]
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
    if done := made_from(store, result):
        return done[0]  # this run's answer is already proposed (agents/trace.py)
    d = result.output
    criteria = core.FrameCriteria(
        id="frame",
        version=1 + len(proposals(store, "frame")),
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
    record_run(store, result, input, [proposal.id])
    return proposal


def draft_paraphrases(store: Store, template_id: str) -> core.Proposal:
    """A template's frozen paraphrase set (spec §2.1: "from a frozen, reviewed set per template"): the paraphrase_writer's
    two rewordings with the original, p0, submitted as a paraphrase_set proposal. Its schema gate checks every
    paraphrase keeps the template's slots and words every answer (core.ParaphraseSet)."""
    t = library.template(template_id)
    input = {"template": t.model_dump(mode="json")}
    result = runner.run("paraphrase_writer", input, ParaphrasesDraft, effort="high")
    if done := made_from(store, result):
        return done[0]  # this run's answer is already proposed (agents/trace.py)
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
    record_run(store, result, input, [proposal.id])
    return proposal


def library_ref(template_id: str) -> core.Ref:
    """A library template as its current version."""
    return core.Ref(id=template_id, version=library.template(template_id).version)


def paraphrase_set(store: Store, template: core.Ref) -> core.ParaphraseSet:
    """The case's active paraphrase set for exactly this template version: a set words one version of one question
    template (a library template at the library's version, a case's refined question at its own), and a set of another
    version words another question."""
    for p in proposals(store, "paraphrase_set"):
        t = p.payload.get("template", {})
        if p.status == "active" and t.get("id") == template.id and t.get("version") == template.version:
            return core.ParaphraseSet.model_validate_json(core.canonical_json(dict(p.payload)))
    raise Refused(f"no active paraphrase set for template {template.id} v{template.version}: run `requisite paraphrases`")


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
    active = [p for p in proposals(store, "frame") if p.status == "active"]
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
    store: Store, client: Callable[[], RegulationsGov] = RegulationsGov, *, workers: int = 8,
    federal_register: Callable[[], FederalRegister] = FederalRegister,
    mirror: Callable[[], Mirrulations] | None = None,
) -> list[core.Item]:
    """Step 5: every phase-1 item read whole and committed, with the measure it addresses (core.ItemMeasure; items and
    measures already committed are not read again). With a mirror, each comment's record is the mirror's copy where it
    is current against the frame's official listing, and the API's otherwise (requisite/evidence/mirrulations.py)."""
    ids = [d.item for d in drawn(store, 1).items]
    have = committed(store, ids)
    todo = [i for i in ids if i not in have]
    if todo:
        regulations_gov = client()
        copies = mirror() if mirror is not None else None
        sample = entries(store, todo)
        try:
            read_items(regulations_gov, store, sample, workers=workers, mirror=copies,
                       official=framing.listed_comments(store, the_frame(store), sample) if copies else None)  # fmt: skip
        finally:
            regulations_gov.close()
            if copies is not None:
                copies.close()
    # Each sample item read again from its own bytes where a newer reader reads a file of it differently (committed):
    # the stale parts found and the item committed in this thread, the files read in parallel.
    held = list(committed(store, ids).values())
    jobs = [(it, part) for it in held for part in stale(store, it)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        newer = list(pool.map(lambda j: extract.extract(j[1].reader, j[1].artifact, j[1].content), jobs))
    by_item: dict[str, dict[int, extract.Extraction]] = {}
    for (it, part), x in zip(jobs, newer, strict=True):
        by_item.setdefault(it.id, {})[part.part] = x
    for it in held:
        if it.id in by_item:
            reextracted(store, it, by_item[it.id])
    have = committed(store, ids)
    read = [have[i] for i in ids if i in have]
    unmeasured = [i for i in read if store.get("item_measure", i.id, core.ItemMeasure) is None]
    if unmeasured:
        found = framing.measures(store, the_frame(store), unmeasured, regulations_gov=client,
                                 federal_register=federal_register)  # fmt: skip
        for m in found.values():
            store.commit("item_measure", m.item, m)
    return read


def measures(store: Store, ids: Sequence[str]) -> dict[str, core.ItemMeasure]:
    """Each item's measure, as committed when it was read (core.ItemMeasure)."""
    out = {i: store.get("item_measure", i, core.ItemMeasure) for i in ids}
    missing = [i for i, m in out.items() if m is None]
    if missing:
        raise Refused(f"{len(missing)} item(s) have no measure committed: run `wide read` ({', '.join(missing[:5])})")
    return {i: m for i, m in out.items() if m is not None}


@dataclass(frozen=True)
class Labelled:
    """What a labelling step did: the judgments it holds, and the items whose filing has no readable part, which no
    judge is asked about (each with what its parts record)."""

    judgments: list[core.Judgment]
    unreadable: dict[str, str]


def _states(store: Store, ids: Sequence[str]) -> tuple[list[labels.ItemState], dict[str, str]]:
    """The readable items' judgment states, and the items with no readable part, each with what its parts record. Which
    items are unreadable is read from each item's parts (core.ItemPart), as the reader recorded them; a state that
    cannot be built for a readable item is an error that stops the step, never a reason to set the item aside (external
    evaluation, November 2: integrity failures had been counted as unreadable and taken out of the gold's scope)."""
    now = store.clock()
    states, unreadable = [], {}
    have = committed(store, ids)
    ours = {i: have[i].unread_by_us for i in ids if i in have and have[i].unread_by_us}
    if ours:
        raise Refused(f"{len(ours)} item(s) hold a file published in a format no reader here reads; read it before any "
                      "item is judged (core.ItemPart.by): " + "; ".join(f"{i}: {u}" for i, u in list(ours.items())[:10]))  # fmt: skip
    for item_id in ids:
        item = have.get(item_id)
        if item is None:
            raise Refused(f"sampled item {item_id} is not read: run `wide read`")
        if not item.readable:
            unreadable[item_id] = "; ".join(item.unreadable)
            continue
        states.append(labels.item_state(store, item, evaluated_at=now))
    return states, unreadable


def _judge(
    store: Store, ids: Sequence[str], configs: Sequence[tuple[str, str, Any]], workers: int,
    on_done: Callable[[int, int], None] | None,
) -> Labelled:  # fmt: skip
    objective = approved_objective(store)
    psets = {t: paraphrase_set(store, library_ref(t)) for t in labels.OBJECTIVE_TEMPLATES}
    states, unreadable = _states(store, ids)
    ms = measures(store, [st.item.id for st in states])
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


def unreadable(store: Store) -> dict[str, str]:
    """The phase-1 items with no readable text, each with why: outside the run's readable scope (decisions, "F4 from the
    labels to the leads")."""
    return _states(store, [d.item for d in drawn(store, 1).items])[1]


def _phase1(store: Store) -> dict[tuple[str, str], core.Judgment]:
    """The phase-1 judgment of each question over each state. Two responses to one question over one state under one
    configuration exist only where the request changed (a response is a function of its request: cassettes); an answer
    stands over a refusal, since a refusal of one form of the request is not one of the question (providers.AS_TEXT),
    and two different answers are refused, neither standing for the other."""
    spec = providers.PROVIDERS[labels.PHASE1_CONFIG[0]]
    phase1 = core.JudgeConfig(provider=spec.name, model=spec.model, model_cutoff=spec.model_cutoff,
                              paraphrase_id=labels.PHASE1_CONFIG[1], render_id=labels.PHASE1_CONFIG[2].id)  # fmt: skip
    found: dict[tuple[str, str], core.Judgment] = {}
    for j in store.all("judgment", core.Judgment):
        if j.config != phase1:
            continue
        key = (j.question.id, j.state_hash)
        have = found.get(key)
        if have is None or (have.distribution is None and j.distribution is not None):
            found[key] = j
        elif j.distribution is not None and j.distribution != have.distribution:
            raise Refused(f"{key[0]} over state {key[1][:12]} has two answers under the phase-1 configuration")
    return found


def phase1_judgments(store: Store) -> list[str]:
    """The content hashes of the phase-1 judgments the predictions read, in a stable order."""
    asked = {q for q in labels.OBJECTIVE_TEMPLATES}
    return sorted(j.content_hash() for (q, _), j in _phase1(store).items() if q.split(".")[0] in asked)


def predictions(store: Store) -> dict[str, tuple[float, float]]:
    """Each readable phase-1 item's judged P(in_population) and P(exposure in EXPOSED), from its phase-1 judgments
    (decisions, "F4 from the labels to the leads"): a question the judge refused takes the mean of that question's
    probability over the answered items of its phase-1 stratum. An item with no readable text is not in the result
    (unreadable); a question not yet asked is an error, since the labels are a step of their own."""
    objective = approved_objective(store)
    draw = drawn(store, 1)
    stratum = {d.item: d.stratum for d in draw.items}
    states, _ = _states(store, [d.item for d in draw.items])
    ms = measures(store, [st.item.id for st in states])
    found = _phase1(store)
    answered: dict[str, list[float | None]] = {}
    unasked = []
    for st in states:
        qs = labels.questions(objective, ms, st.item)
        h = st.state.content_hash()
        pair: list[float | None] = []
        for template, answers in (("voice_in_population", ("in_population",)), ("exposure_to_measure", EXPOSED)):
            j = found.get((qs[template].id, h))
            if j is None:
                unasked.append(st.item.id)
                break
            pair.append(None if j.distribution is None else float(sum(j.distribution.get(a, 0) for a in answers)))
        else:
            answered[st.item.id] = pair
    if unasked:
        raise Refused(f"{len(unasked)} phase-1 item(s) are not labelled yet: run `wide label` ({', '.join(unasked[:5])})")
    means: dict[tuple[str, int], float] = {}
    for q in (0, 1):
        for h in set(stratum.values()):
            vals = [x for i, v in answered.items() if stratum[i] == h and (x := v[q]) is not None]
            if vals:
                means[(h, q)] = sum(vals) / len(vals)
    out = {}
    for item, pair in answered.items():
        filled = []
        for q in (0, 1):
            v = pair[q]
            if v is None:
                if (stratum[item], q) not in means:
                    raise Refused(f"the judge refused every item of stratum {stratum[item]} on one question")
                v = means[(stratum[item], q)]
            filled.append(v)
        out[item] = (filled[0], filled[1])
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


COVERAGE_DRAWS = 1000


def gold_coverage(store: Store) -> sampling.RatioCoverageReport:
    """The gold design's measured coverage of the approved estimand, a ratio (items exposed among items in the
    population's voice; decisions, "Wide inference as built"), before the gold is drawn (decisions, "Phase-2
    allocation"). No label is known before the gold, so labels are simulated from the judge's own probabilities, seeded
    from the phase-1 draw: an item is in the denominator with P(in_population), and in the numerator with that times
    P(exposed); the phase-1 sample is the simulated frame, taken whole, so what is measured is the gold design's own:
    each phase-2 stratum's expected gold, denominator and numerator items, and the ratio interval's coverage if the
    judge were calibrated. It is a statement about the design under that label model, not about the frame's truth."""
    draw = drawn(store, 1)
    f = predictions(store)  # readable items: an unreadable one is known outside the scope, not drawn for gold
    rng = np_random.default_rng(seed(draw.content_hash(), "coverage"))
    cols: dict[str, tuple[list[int], list[float], list[int], list[float], list[float]]] = {}
    for d in (d for d in draw.items if d.item in f):
        voice, exposed = f[d.item]
        yd, fd, yn, fn, st = cols.setdefault(d.stratum, ([], [], [], [], []))
        in_voice = int(rng.random() < voice)
        yd.append(in_voice)
        fd.append(voice)
        yn.append(in_voice * int(rng.random() < exposed))
        fn.append(voice * exposed)
        st.append(voice * exposed)
    strata = list(cols)
    frame = {h: (yd, fd, {"exposed": (yn, fn)}, st) for h, (yd, fd, yn, fn, st) in cols.items()}
    return sampling.measure_ratio_coverage(
        frame=frame, phase1_sample_sizes={h: len(cols[h][0]) for h in strata}, phase2_cuts=[PHASE2_CUT],
        phase2_allocation=gold_allocation(strata), draws=COVERAGE_DRAWS, seed=seed(draw.content_hash(), "coverage"),
    )  # fmt: skip


def draw_gold(store: Store) -> core.SampleDraw:
    """Step 8: the gold sample (sampling.draw_phase2) of the readable phase-1 items, stratified within each phase-1
    stratum by the judge's probability that the item is in the estimand's numerator, cut at PHASE2_CUT."""
    draw = drawn(store, 1)
    f = predictions(store)
    readable = [d for d in draw.items if d.item in f]  # an unreadable item's labels are known: nothing to draw
    strata = list(dict.fromkeys(d.stratum for d in readable))
    allocation = gold_allocation(strata)
    stratifier = [f[d.item][0] * f[d.item][1] for d in readable]
    s = seed(draw.content_hash(), "phase2")
    mask = sampling.draw_phase2(phase1_stratum=[d.stratum for d in readable], stratifier=stratifier,
                                phase2_cuts=[PHASE2_CUT], phase2_allocation=allocation, seed=s)  # fmt: skip
    cell = [(d.stratum, int(v >= PHASE2_CUT)) for d, v in zip(readable, stratifier, strict=True)]
    held = {c: cell.count(c) for c in set(cell)}
    items = [core.DrawnItem(item=d.item, stratum=f"{c[0]}.bin{c[1]}",
                            inclusion_prob=Fraction(min(allocation[c], held[c]), held[c]))
             for d, c, chosen in zip(readable, cell, mask, strict=True) if chosen]  # fmt: skip
    gold = core.SampleDraw(design=draw.design, phase=2, seed=s, items=items)
    store.commit("sample_draw", f"{gold.design.id}.phase2", gold)
    unread = [d for d in draw.items if d.item not in f]
    store.commit("sampling_design", draw.design.id, sampling_design(store, draw, held, allocation, unread))
    return gold


def sampling_design(
    store: Store, draw: core.SampleDraw, held: Mapping[tuple[str, int], int], allocation: Mapping[tuple[str, int], int],
    unread: Sequence[core.DrawnItem],
) -> core.SamplingDesign:  # fmt: skip
    """The run's design as drawn, the record its draws name (addendum A07 §8.1; external evaluation, November 2: the
    draws named a design no record held): the frame, its phase-1 strata and sizes, the phase-2 cells the judge's
    probabilities formed in each, with the items each holds and the gold each drew, and the unreadable items' cell,
    taken whole since their indicators are known (decisions, "F4 from the labels to the leads"). Committed with the gold
    draw, when every part of it is known; the rules it states were fixed in decisions.md before the frame was seen."""
    frame, objective = the_frame(store), approved_objective(store)
    strata = phase1_strata(frame)
    drawn_in = {h: sum(1 for d in draw.items if d.stratum == h) for h in strata}
    phase2 = []
    for h in strata:
        n_unread = sum(1 for d in unread if d.stratum == h)
        if n_unread:
            phase2.append(core.SubStratum(name=f"{h}.unreadable", parent=h, phase1_count=n_unread, sample_size=n_unread,
                                          rule="no readable part (core.Item.readable): both indicators known 0, taken whole"))  # fmt: skip
        for b in (0, 1):
            n = held.get((h, b), 0)
            if n:
                rule = f"P(in_population) x P(exposed) {'<' if b == 0 else '>='} {PHASE2_CUT}, by the phase-1 judge"
                phase2.append(core.SubStratum(name=f"{h}.bin{b}", parent=h, phase1_count=n,
                                              sample_size=min(allocation[(h, b)], n), rule=rule))  # fmt: skip
    spec = providers.PROVIDERS[labels.PHASE1_CONFIG[0]]
    return core.SamplingDesign(
        id=draw.design.id, version=draw.design.version, frame=core.Ref(id=frame.id, version=1),
        frame_hash=frame.content_hash(), frame_size=len(frame.items), unit="item", estimand=objective.estimand,
        filter_policy="every item of the frame is in the frame; an item with no readable part is in the unreadable cell, "
                      "outside the readable scope; a part no reader here reads stops the run until it is read",
        dedup_policy="none: each filing is its own item, identical texts included",
        sample="two_phase_stratified_srswor",
        phase1=[core.Stratum(name=h, frame_size=len(items), sample_size=drawn_in[h]) for h, items in strata.items()],
        phase2=phase2,
        allocation=core.Ref(id="phase1_equal_shares.gold_equal_per_stratum_80_20", version=1),
        seed=draw.seed,
        estimator=core.Ref(id="two_phase_ratio", version=1),
        predictor=[*(core.Ref(id=t, version=library.template(t).version) for t in labels.OBJECTIVE_TEMPLATES),
                   core.Ref(id=f"{spec.name}:{spec.model}:{labels.PHASE1_CONFIG[1]}:{labels.PHASE1_CONFIG[2].id}",
                            version=1)],
        gold_policy="one labeller, blind to every model's answer; 'unsure' and not_established are unresolved, never 0; "
                    "the estimate is bounded over unresolved labels by its two completions",
    )  # fmt: skip
