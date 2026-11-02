"""Wide discovery, orchestrated (spec §7.2–7.4; outcome 002). Code runs every step; agents only draft, judges only
answer, and a human decides at G1. Which corpus and which leads the run reaches are its own findings (decisions,
"Discovery finds what it analyzes"): nothing here names a docket, an agency or a rule.

The steps, each a function a CLI command calls:
1. draft_objective: an agent drafts the run's objective from a human's brief; it awaits G1.
2. approve: the human approves the objective (or edits, or rejects) at G1.
3. draft_frame: an agent derives the frame criteria from the approved objective.
The frame's freezing, the phase-1 sample, labelling, stability, the gold draw, inference and leads follow (this module
grows with the evidence adapter, task 002-1, and the inference kernel, task 000-2).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any

from pydantic import BaseModel

from requisite import core
from requisite.agents import runner
from requisite.agents.answers import FrameDraft, ObjectiveDraft, ParaphrasesDraft
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


def draft_frame(store: Store, as_of: datetime) -> core.Proposal:
    """Step 3: the frame_drafter's criteria for the approved objective, submitted as a proposal; its tests gate is the
    criteria's run and freeze (registered by the evidence adapter)."""
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
