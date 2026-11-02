"""From the lead a human promoted at G1 to an open deep case on that business (spec §7.2 step 6: "Promote top leads into
deep cases at gate G1"; §7.1 step 1; outcome 003). Code opens the case from committed records and the business's own
words; an agent proposes the entity model; every choice of the decision is a human's, given on the command line and
confirmed at G1 with the model.

`open_lead` (the CLI's `requisite deep open`):
1. The lead: a `lead` proposal of the wide case, active by its G1 approval (core.HUMAN_GATE: an approval at any other
   gate authorizes something else and opens nothing).
2. The deep case, its own store at <wide case dir>/deep/<LeadPromotion.case>. Its evidence is the lead's own filing, as
   the wide case committed it: the item, its artifacts with their bytes, its extractions, and its claims (each read
   part's text, asserted by the filer, as labels.item_state states it, and the lead's quoted impacts), each under its
   own content-addressed id and checked equal to the wide case's record; and the ItemMeasure of the document it
   addresses with that statement's artifacts. Nothing else is gathered (spec §7.1 names no gathering step: the filing is
   the business's evidence the flow found).
3. The event: the measure the filing addresses taking effect, its window the approved objective's horizon, proposed as
   a proposition (deep.propose_event). The objective's loss counts what a measure costs "taking effect as proposed or
   enacted", so that is the event its exposures are about.
4. The deep templates' paraphrase sets (requisite/library/templates/deep.json and forecast.json, which deep.py asks),
   drafted by the paraphrase writer where the case holds none active (wide.draft_paraphrases).
5. The entity model: the decomposer's answer (agents.answers.EntityModelDraft) over the filing's claims, the objective's
   event class, horizon and loss, and the event. Code builds core.EntityModel from it, refusing a number that cites a
   claim the case does not hold or quotes words its claim does not contain; the answer type refuses one with neither a
   claim nor an assumption and its reason. Its entity is the lead's, its horizon and loss definition the approved
   objective's (the case serves the objective's decision, approved at G1). Proposed as `entity_model`, it awaits G1.
6. The decision spec (addendum §12): its horizon and loss definition the approved objective's, its entity the lead's,
   its decision time the case's evidence's as_of, and its objective, alpha, tolerance, liquidity floor, actions,
   information, recourse and the currency its amounts are in a human's, as given (`Choices`: never a default). Proposed
   as `decision_spec`, it awaits G1.

The decision time is the instant from which everything the case holds is observable (spec I3; core.observable_at): the
latest of the frame's as_of and each held artifact's publication and availability. The filing was read after the frame
was taken, and a fetched text is evidenced only from its retrieval (addendum A11 §11.3), so a decision time at the
frame's as_of would hold none of the business's words; this is the as_of the wide flow judged the same filing at
(labels.item_state), and never earlier than the run's pinned as_of.

Each step is idempotent: a step run again over the same records and the same recorded answers returns what it
proposed (agents/trace.made_from), so the command can be run again after any refusal is resolved.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from requisite import core
from requisite.agents import runner
from requisite.agents.answers import EntityModelDraft, NumberDraft
from requisite.agents.trace import made_from, record_run
from requisite.discovery import deep, labels, wide
from requisite.evidence.items import committed
from requisite.library import templates as library
from requisite.library.nodes.materiality import horizon_days
from requisite.store import Refused, Store
from requisite.store import open_case as open_store

AgentCall = Callable[..., runner.AgentResult[Any]]

DECOMPOSER = "decomposer"
MODEL_ID = "entity_model"
DECISION_ID = "decision"
# deep.py's module docstring: "The deep templates' paraphrase sets (requisite/library/templates/deep.json,
# forecast.json) are active in the case."
TEMPLATE_FILES = ("deep.json", "forecast.json")
# The event a measure's exposures are about: the approved objective's loss counts what it costs "taking effect as
# proposed or enacted" (the objective of the wide run; spec §7.1 step 2's event families are measures of this kind).
EVENT = "The measure below takes legal effect, as proposed or as made final. The measure: {statement}"


@dataclass(frozen=True)
class Choices:
    """The decision's parameters a human gives (addendum §12 DecisionSpec; spec §2.4 G1), each recorded as given and
    never defaulted. alpha: an exact probability, None only where the human stated there is none (an objective that is
    not a CVaR). floor: (day, amount) points, empty only where the human stated there is no floor. currency: the
    currency the tolerance and the floor are stated in, which is the entity model's."""

    objective: str
    alpha: str | None
    tolerance: str
    floor: tuple[tuple[str, str], ...]
    actions: tuple[str, ...]
    information: tuple[str, ...]
    recourse: str
    currency: str


@dataclass(frozen=True)
class Opened:
    """What `open_lead` committed or found committed, for the command to show."""

    root: Path
    store: Store
    lead: core.Proposal
    promotion: core.LeadPromotion
    approval: core.Approval
    item: core.Item
    measure: core.ItemMeasure
    claims: tuple[core.Claim, ...]
    artifacts: tuple[core.Artifact, ...]
    event: core.Proposal
    paraphrases: tuple[core.Proposal, ...]
    model: core.Proposal
    draft: EntityModelDraft
    decision: core.Proposal


# ---------------------------------------------------------------- 1. the lead


def approved_lead(store: Store, lead_id: str) -> tuple[core.Proposal, core.LeadPromotion, core.Approval]:
    """The lead proposal, active by its G1 approval, with that approval; refused otherwise, saying which."""
    p = store.proposal(lead_id)
    if p is None:
        raise Refused(f"case {store.case} holds no proposal {lead_id}")
    if p.kind != "lead":
        raise Refused(f"{lead_id} is a {p.kind} proposal, not a lead")
    gate = core.HUMAN_GATE["lead"]
    approvals = [a for a in store.all("approval", core.Approval) if a.subject.id == lead_id]
    if p.status != "active":
        last = p.gates[-1] if p.gates else None
        why = f"lead {lead_id} is {p.status}" + (f" ({last.gate} gate: {last.detail})" if last else "")
        elsewhere = sorted({a.gate for a in approvals if a.gate != gate})
        if elsewhere:
            why += (f"; its approval at {', '.join(elsewhere)} authorizes something else: a lead's promotion into a deep "
                    f"case is decided at {gate} (spec §7.2 step 6)")  # fmt: skip
        if p.status == "proposed":
            why += f". A deep case opens only from a lead approved at {gate}: requisite approve {lead_id} --gate {gate}"
        raise Refused(why)
    lineage = core.sha256(core.canonical_json(dict(p.payload)))
    covering = [a for a in approvals if a.gate == gate and a.lineage_hash == lineage and a.decision != "rejected"]
    if not covering:  # an active lead passed its human gate on such an approval (Store._human)
        raise Refused(f"lead {lead_id} is active but no approval at {gate} covers its payload")
    promotion = core.LeadPromotion.model_validate_json(core.canonical_json(dict(p.payload)))
    return p, promotion, covering[-1]


def deep_root(case_dir: Path, promotion: core.LeadPromotion) -> Path:
    return Path(case_dir) / "deep" / promotion.case


# ---------------------------------------------------------------- 2. the evidence, copied as committed


def _copy_artifact(src: Store, dst: Store, art_id: str) -> core.Artifact:
    """An artifact and its bytes as the source case committed them, its predecessors first (spec I2: a changed source is
    a new version linked to the one before; the store holds the chain)."""
    a = src.get("artifact", art_id, core.Artifact)
    if a is None:
        raise Refused(f"artifact {art_id} is not committed in case {src.case}")
    if a.predecessor is not None:
        _copy_artifact(src, dst, a.predecessor)
    dst.commit_artifact(a, src.content(a.id))  # refused unless the bytes are the ones its id names
    return a


def _copy_filing(src: Store, dst: Store, item: core.Item, lead: core.Lead) -> tuple[list[core.Claim], list[core.Artifact]]:
    """The lead's filing in the deep case: the item, its artifacts and extractions, and its claims, each checked equal to
    the wide case's committed record."""
    dst.commit("item", f"{item.id}|{item.content_hash()}", item)
    artifacts = [_copy_artifact(src, dst, a) for a in item.artifacts]
    for part in item.parts:
        if part.extraction is None:
            continue
        assert part.artifact is not None  # core.ItemPart: text is read from retrieved content
        text = src.extraction(part.artifact, part.extraction)
        if text is None:
            raise Refused(f"item {item.id}: {part.label}'s extraction {part.extraction} is not committed in case {src.case}")
        dst.commit_extraction(part.artifact, part.extraction, text)
    try:
        filed = labels.item_state(dst, item, evaluated_at=dst.clock())
    except ValueError as e:  # an item with no readable part has no claims to copy
        raise Refused(f"lead item {item.id}: {e}") from None
    claims = list(filed.claims.values())
    for imp in lead.impact:
        c = src.get("claim", imp.claim, core.Claim)
        if c is None or c.artifact not in item.artifacts:
            raise Refused(f"the lead's quoted impact {imp.claim} is not a committed claim of item {item.id}'s filing")
        dst.commit_claim(c)
        claims.append(c)
    for c in claims:
        if src.get("claim", c.id, core.Claim) != c:
            raise Refused(f"claim {c.id} of item {item.id}'s filing is not committed in case {src.case} as the deep case "
                          "holds it: the deep case holds only the wide case's records")  # fmt: skip
    return list({c.id: c for c in claims}.values()), artifacts


def _decision_time(frame: core.Frame, artifacts: Sequence[core.Artifact]) -> datetime:
    """The first instant at which every held artifact is observable (core.observable_at: published by then and its text
    evidenced by then), never before the frame's as_of."""
    instants = [frame.as_of]
    for a in artifacts:
        instants += [a.available_from, a.published_at if a.published_at is not None else a.available_from]
    return max(instants)


# ---------------------------------------------------------------- 3-4. the event and the paraphrase sets


def _proposed(store: Store, kind: str, payload: Mapping[str, Any]) -> core.Proposal | None:
    """A proposal of this exact payload the case already holds and has not rejected."""
    want = core.canonical_json(dict(payload))
    for p in wide.proposals(store, kind):
        if p.status != "rejected" and core.canonical_json(dict(p.payload)) == want:
            return p
    return None


def _propose_event(store: Store, event: core.Proposition, lead_id: str, measure: core.ItemMeasure) -> core.Proposal:
    found = _proposed(store, "proposition", event.model_dump(mode="json"))
    if found is not None:
        return found
    p = deep.propose_event(store, event, author="requisite", rationale=(
        f"the measure lead {lead_id}'s filing addresses ({measure.document}), as its committed ItemMeasure states it, "
        "taking effect within the approved objective's horizon"))  # fmt: skip
    if p.status != "active":
        raise Refused(f"the event {event.id} was not admitted: {p.gates[-1].detail}")
    return p


def deep_templates() -> list[str]:
    """The templates deep.py asks, as the library's deep and forecast files define them."""
    out = []
    for name in TEMPLATE_FILES:
        out += [str(t["id"]) for t in json.loads((library.TEMPLATES / name).read_text())["templates"]]
    return out


def _paraphrases(store: Store) -> list[core.Proposal]:
    """Each deep template's active paraphrase set: the one the case holds, or the paraphrase writer's."""
    out = []
    for t in deep_templates():
        ref = wide.library_ref(t)
        held = [p for p in wide.proposals(store, "paraphrase_set") if p.status == "active"
                and p.payload.get("template", {}).get("id") == ref.id
                and p.payload.get("template", {}).get("version") == ref.version]  # fmt: skip
        p = held[0] if held else wide.draft_paraphrases(store, t)
        if p.status != "active":
            raise Refused(f"the paraphrase set of {t} was not admitted ({p.id}): {p.gates[-1].detail}")
        out.append(p)
    return out


# ---------------------------------------------------------------- 5. the entity model


def entity_of(lead: core.Lead) -> str:
    return lead.submitter or f"the filer of {lead.item} (name not published)"


def _source(item: core.Item, c: core.Claim) -> str:
    part = next((p for p in item.parts if p.artifact == c.artifact and p.extraction == c.extraction), None)
    where = part.label if part is not None else "a part"
    return f"{where} of filing {item.id} in {item.container}, retrieved from {part.locator if part else c.artifact}"


def decomposer_input(item: core.Item, lead: core.Lead, claims: Sequence[core.Claim], artifacts: Mapping[str, core.Artifact],
                     objective: core.WideObjective, event: core.Proposition) -> dict[str, Any]:  # fmt: skip
    """What the decomposer reads (its prompt: "the business's public statements ..., each as a numbered claim with its
    source, and the objective of the case (the event class and the loss the client cares about)"), in the filing's
    order."""
    out = []
    for n, c in enumerate(claims, 1):
        a = artifacts[c.artifact]
        out.append({"number": n, "id": c.id, "text": c.text, "said_by": c.authority, "source": _source(item, c),
                    "published": a.published_at.isoformat() if a.published_at else "not stated"})  # fmt: skip
    return {
        "business": entity_of(lead),
        "claims": out,
        "objective": {"event_class": objective.events, "loss_definition": objective.loss_definition,
                      "horizon": objective.horizon.model_dump(mode="json")},
        "event": event.text,
    }  # fmt: skip


def check_sources(draft: EntityModelDraft, held: Mapping[str, core.Claim]) -> None:
    """Every sourced number cites claims the case holds, each containing the words quoted from it
    (EntityModelDraft.check_quotes, which the runner applies before it accepts an answer); refused otherwise."""
    try:
        draft.check_quotes({cid: c.text for cid, c in held.items()})
    except ValueError as e:
        raise Refused(f"the decomposer's {e}") from None


def basis(n: NumberDraft) -> str:
    """How a number is known, for the human at G1: the words each claim states it in, or the assumption and why."""
    if n.assumption is not None:
        return f"assumed: {n.assumption.text} (no claim states it: {n.assumption.reason})"
    return "; ".join(f"quoted from claim {c.id}: \"{c.words}\"" for c in n.claims)


def _rationale(draft: EntityModelDraft) -> str:
    lines = [draft.rationale, "", "Each number, as the decomposer gave it and code checked it against the case's claims:"]
    lines += [f"- {where}: {n.name} = {n.value} {n.unit}; {basis(n)}" for where, n in draft.numbers()]
    return "\n".join(lines)


def _next_version(store: Store, kind: str) -> int:
    return 1 + max((int(p.payload.get("version", 0)) for p in wide.proposals(store, kind)), default=0)


def _propose_model(store: Store, item: core.Item, lead: core.Lead, claims: Sequence[core.Claim],
                   artifacts: Mapping[str, core.Artifact], objective: core.WideObjective, event: core.Proposition,
                   agent: AgentCall) -> tuple[core.Proposal, EntityModelDraft]:  # fmt: skip
    agent_input = decomposer_input(item, lead, claims, artifacts, objective, event)
    held = {c.id: c.text for c in store.all("claim", core.Claim)}
    result = agent(DECOMPOSER, agent_input, EntityModelDraft, effort="high", context={"claims": held})
    draft = cast(EntityModelDraft, result.output)
    if done := made_from(store, result):
        return done[0], draft  # this run's answer is already proposed (agents/trace.py)
    check_sources(draft, {c.id: c for c in store.all("claim", core.Claim)})
    model = draft.model(id=MODEL_ID, version=_next_version(store, "entity_model"), entity=entity_of(lead),
                        horizon=objective.horizon, loss_definition=objective.loss_definition)  # fmt: skip
    payload = model.model_dump(mode="json")
    proposal = store.submit_proposal(core.Proposal(
        id=wide._proposal_id("entity_model", payload), case=store.case, kind="entity_model", payload=payload,
        author=result.run[:16], rationale=_rationale(draft), status="proposed"))  # fmt: skip
    record_run(store, result, agent_input, [proposal.id])
    return proposal, draft


# ---------------------------------------------------------------- 6. the decision spec


def decision_spec(choices: Choices, *, entity: str, decision_time: datetime, objective: core.WideObjective,
                  version: int = 1) -> core.DecisionSpec:  # fmt: skip
    """The decision spec from the human's choices, the lead's entity, the evidence's as_of and the approved objective's
    horizon and loss; refused, with the definition's reasons, where the choices do not make one."""
    try:
        return core.DecisionSpec.model_validate({
            "id": DECISION_ID, "version": version, "entity": entity, "decision_time": decision_time,
            "horizon": objective.horizon, "loss_definition": objective.loss_definition, "actions": list(choices.actions),
            "objective": choices.objective, "alpha": choices.alpha, "tolerance": choices.tolerance,
            "liquidity_floor": [{"day": d, "amount": a} for d, a in choices.floor], "currency": choices.currency,
            "information": list(choices.information), "recourse": choices.recourse,
        })  # fmt: skip
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(str(x) for x in err['loc']) or 'decision'}: {err['msg']}" for err in e.errors())
        raise Refused(f"the decision's parameters as given do not make a decision spec (addendum §12): {problems}") from None


def _propose_decision(store: Store, spec: core.DecisionSpec, model: core.EntityModel, lead_id: str) -> core.Proposal:
    if spec.currency != model.currency:
        raise Refused(f"the decision's amounts are given in {spec.currency}, and the entity model's in {model.currency}: "
                      "a case has one currency; give the tolerance and the floor in the entity model's")  # fmt: skip
    for p in wide.proposals(store, "decision_spec"):
        same = {**dict(p.payload), "version": spec.version}
        if p.status != "rejected" and core.canonical_json(same) == spec.canonical():
            return p
    spec = spec.model_copy(update={"version": _next_version(store, "decision_spec")})
    payload = spec.model_dump(mode="json")
    return store.submit_proposal(core.Proposal(
        id=wide._proposal_id("decision_spec", payload), case=store.case, kind="decision_spec", payload=payload,
        author="requisite", status="proposed", rationale=(
            f"the decision the deep case on lead {lead_id} serves: its horizon and loss definition the approved "
            "objective's, its entity the lead's, its decision time the evidence's as_of, and its objective, alpha, "
            "tolerance, liquidity floor, actions, information, recourse and currency as the human gave them")))  # fmt: skip


# ---------------------------------------------------------------- the step


def open_lead(case_dir: Path, case: str, lead_id: str, choices: Choices, *, agent: AgentCall | None = None) -> Opened:
    """Open the deep case of a lead approved at G1 (this module's steps 1-6)."""
    wide_store = open_store(Path(case_dir))
    if wide_store.case != case:
        raise Refused(f"{case_dir} holds case {wide_store.case}, not {case}")
    lead_p, promotion, approval = approved_lead(wide_store, lead_id)
    lead = promotion.lead
    objective, frame = wide.approved_objective(wide_store), wide.the_frame(wide_store)
    try:
        horizon_days(objective.horizon)
    except ValueError as e:
        raise Refused(f"the approved objective's horizon cannot be a deep case's: {e} (deep.horizon_days: cash flows run "
                      "month by month over a bounded horizon)") from None  # fmt: skip
    item = committed(wide_store, [lead.item]).get(lead.item)
    if item is None:
        raise Refused(f"the lead's item {lead.item} is not read in case {case}: run `wide read`")
    measure = wide.measures(wide_store, [item.id])[item.id]
    held = [wide_store.get("artifact", a, core.Artifact) for a in [*item.artifacts, *measure.artifacts]]
    missing = [a for a, h in zip([*item.artifacts, *measure.artifacts], held, strict=True) if h is None]
    if missing:
        raise Refused(f"the lead's filing and its measure name artifacts case {case} does not hold: {missing}")
    decision_time = _decision_time(frame, [h for h in held if h is not None])
    spec = decision_spec(choices, entity=entity_of(lead), decision_time=decision_time, objective=objective)

    root = deep_root(Path(case_dir), promotion)
    store = Store(root, case=promotion.case)
    claims, artifacts = _copy_filing(wide_store, store, item, lead)
    artifacts += [_copy_artifact(wide_store, store, a) for a in measure.artifacts]
    store.commit("item_measure", item.id, measure)
    event = core.Proposition(id=measure.document, text=EVENT.format(statement=measure.statement),
                             window=objective.horizon, domain="bool")  # fmt: skip
    event_p = _propose_event(store, event, lead_id, measure)
    paraphrases = _paraphrases(store)
    model_p, draft = _propose_model(store, item, lead, claims, {a.id: a for a in artifacts}, objective, event,
                                    agent or runner.run)  # fmt: skip
    model = core.EntityModel.model_validate_json(core.canonical_json(dict(model_p.payload)))
    decision_p = _propose_decision(store, spec, model, lead_id)
    return Opened(root, store, lead_p, promotion, approval, item, measure, tuple(claims), tuple(dict.fromkeys(artifacts)),
                  event_p, tuple(paraphrases), model_p, draft, decision_p)  # fmt: skip


# ---------------------------------------------------------------- what the command prints


def _window(w: core.TimeWindow) -> str:
    start = w.start.isoformat() if w.start else "no start"
    end = w.end.isoformat() if w.end else "no end"
    return f"[{start}, {end}{']' if w.end_closed else ')'} {w.tz}"


def _status(p: core.Proposal) -> str:
    from requisite.dossier.proposals import awaiting, human_gate

    if awaiting(p):
        gate, asks = human_gate(p.kind)
        return f"awaits {gate}: a human decides on {asks}"
    return p.status


def render(o: Opened) -> str:
    lead, model = o.promotion.lead, core.EntityModel.model_validate_json(core.canonical_json(dict(o.model.payload)))
    spec = core.DecisionSpec.model_validate_json(core.canonical_json(dict(o.decision.payload)))
    deep_case = ["--case-dir", str(o.root), "--case", o.store.case]
    out = [
        f"deep case {o.store.case}: {o.root}",
        f"from lead {o.lead.id} (rank {lead.rank}: {entity_of(lead)}, item {lead.item} in {lead.docket}, family "
        f"{lead.family}), approved at {o.approval.gate} by {o.approval.approver} ({o.approval.id})",
        f"evidence, the lead's filing as the wide case committed it: item {o.item.id}, {len(o.artifacts)} artifact(s), "
        f"{len(o.claims)} claim(s); the measure it addresses, {o.measure.document}",
    ]  # fmt: skip
    for c in o.claims:
        text = c.text if len(c.text) <= 160 else c.text[:157] + "..."
        out.append(f"  {c.id}: {text!r} ({c.authority})")
    event = core.Proposition.model_validate_json(core.canonical_json(dict(o.event.payload)))
    out.append(f"event {event.id} ({o.event.id}, {o.event.status}), window {_window(event.window) if event.window else '-'}:")
    out.append(f"  {event.text}")
    out.append("paraphrase sets: " + ", ".join(f"{p.payload['template']['id']} ({p.status})" for p in o.paraphrases))
    out.append("")
    out.append(f"entity model {o.model.id} v{model.version}: {_status(o.model)}")
    out.append(f"  {model.entity}, in {model.currency}, over {_window(model.horizon)}")
    out.append(f"  loss: {model.loss_definition}")
    numbers = dict(o.draft.numbers())
    for d in model.drivers:
        out.append(f"  {d.id}: {d.name}, a {d.line} line: quantity x price per {d.period}")
        for part in ("quantity", "price"):
            p = getattr(d, part)
            out.append(f"    {part} {p.value} {p.unit} ({p.name}): {basis(numbers[f'driver {d.id} {part}'])}")
    liq = model.liquidity
    for where, p in (("cash", liq.cash), ("credit line", liq.credit_line), ("receivable days", liq.receivable_days),
                     ("payable days", liq.payable_days)):  # fmt: skip
        out.append(f"  {where} {p.value} {p.unit} ({p.name}): {basis(numbers[where])}")
    seasons = [numbers[f"seasonality month {m}"] for m in range(1, 13)]
    out.append(f"  seasonality, January to December: {', '.join(str(s.value) for s in seasons)}")
    for b in dict.fromkeys(basis(s) for s in seasons):
        out.append(f"    {b}")
    out.append(f"  rationale: {o.draft.rationale}")
    out.append("")
    alpha = "none" if spec.alpha is None else f"{spec.alpha.numerator}/{spec.alpha.denominator}"
    floor = ", ".join(f"{f.day.isoformat()}: {f.amount} {spec.currency}" for f in spec.liquidity_floor) or "none"
    out += [
        f"decision spec {o.decision.id} v{spec.version}: {_status(o.decision)}",
        f"  {spec.entity}; decision time {spec.decision_time.isoformat()} (the evidence's as_of); horizon "
        f"{_window(spec.horizon)}; loss: {spec.loss_definition}",
        f"  as the human gave them: objective {spec.objective}, alpha {alpha}, tolerance {spec.tolerance} "
        f"{spec.currency}, liquidity floor {floor}; actions {', '.join(spec.actions)}; information "
        f"{'; '.join(spec.information) or 'none'}; recourse {spec.recourse}",
        "",
    ]  # fmt: skip
    waiting = [p for p in (o.model, o.decision) if p.status == "proposed"]
    if waiting:
        out.append("awaiting G1 (requisite approve <proposal> --gate G1 " + " ".join(deep_case) + " --approver <you> "
                   "--reason <why> --scope <scope>):")  # fmt: skip
        out += [f"  {p.id}: {_status(p)}" for p in waiting]
    out.append(f"then: requisite deep sensitivities --case {o.root}")
    return "\n".join(out)
