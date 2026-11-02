"""A wide run's results after its gold labels (spec §7.2 steps 5-6; decisions, "F4 from the labels to the leads"):

9. families: an agent proposes the event families over the kept dockets' measures (the objective: "Families are
   defined by the run from what it finds"); its tests gate holds every kept docket in exactly one family.
10. estimate: the objective's share, overall and per family, from the committed phase-1 judgments and gold labels
   under the run's two-phase design, bounded over the unresolved gold labels by its two completions.
11. leads: the phase-1 items in the numerator, asked the spec's item templates and their stated impacts extracted, ranked
   by the rule fixed before any lead was seen.
12. promote: a lead proposed for a deep case, decided by a human at G1.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Literal, cast

import numpy as np

from requisite import core
from requisite.agents import runner
from requisite.agents.answers import FamiliesDraft, ImpactsDraft
from requisite.agents.trace import made_from, record_run
from requisite.discovery import inference, labels, wide
from requisite.evidence.items import committed
from requisite.judgment import providers
from requisite.judgment.render import ask as ask_question
from requisite.judgment.render import render_state
from requisite.library import templates as library
from requisite.store import Refused, Store

LEAD_TEMPLATES = ("cost_driver", "direction", "seeks_certainty")  # spec §7.2's item templates asked of each lead
LEAD_CUT = 0.5  # decisions, "F4 from the labels to the leads": P(in_population) x P(exposed) >= 0.5


# ---------------------------------------------------------------- 9. families


def _kept(store: Store) -> list[str]:
    return sorted(d.docket for d in wide.the_frame(store).dockets if d.kept)


def families_gate(store: Store, proposal: core.Proposal) -> core.GateResult:
    """The tests gate of an event_families proposal: its frame is the case's, and every kept docket of it is in exactly
    one family (core.EventFamilies holds the 'at most one'; this holds the 'every')."""
    fams = core.EventFamilies.model_validate_json(core.canonical_json(dict(proposal.payload)))
    frame = wide.the_frame(store)
    if fams.frame != frame.id:
        return core.GateResult(gate="tests", passed=False, detail=f"the case's frame is {frame.id}, not {fams.frame}")
    kept = set(_kept(store))
    placed = {d for f in fams.families for d in f.dockets}
    missing, foreign = sorted(kept - placed), sorted(placed - kept)
    if missing or foreign:
        detail = "; ".join(x for x in (f"kept dockets in no family: {missing}" if missing else "",
                                        f"dockets the frame did not keep: {foreign}" if foreign else "") if x)  # fmt: skip
        return core.GateResult(gate="tests", passed=False, detail=detail)
    return core.GateResult(gate="tests", passed=True,
                           detail=f"all {len(kept)} kept dockets in exactly one of {len(fams.families)} families")  # fmt: skip


def draft_families(store: Store, agent: Callable[..., Any] | None = None) -> core.Proposal:
    """Step 9: the family_drafter's families over the kept dockets' measures, proposed through their gates."""
    store.register_gate("event_families", "tests", families_gate)
    objective, frame, ms = wide.approved_objective(store), wide.the_frame(store), wide.measures(store)
    dockets = [{"docket": d.docket, "measure": ms[d.docket], "comments": d.comments} for d in frame.dockets if d.kept]
    agent_input = {"events": objective.events, "loss_definition": objective.loss_definition,
                   "dockets": sorted(dockets, key=lambda x: str(x["docket"]))}  # fmt: skip
    result = (agent or runner.run)("family_drafter", agent_input, FamiliesDraft, effort="high")
    if done := made_from(store, result):
        return done[0]
    d = cast(FamiliesDraft, result.output)
    payload = {"frame": frame.id, "families": [dict(f.model_dump()) for f in d.families]}
    proposal = store.submit_proposal(core.Proposal(
        id=wide._proposal_id("event_families", payload), case=store.case, kind="event_families", payload=payload,
        author=result.run[:16], rationale=d.rationale, status="proposed",
    ))  # fmt: skip
    record_run(store, result, agent_input, [proposal.id])
    return proposal


def the_families(store: Store) -> tuple[core.Proposal, core.EventFamilies]:
    active = [p for p in wide._proposals(store, "event_families") if p.status == "active"]
    if len(active) != 1:
        raise Refused(f"{len(active)} event_families proposals are active; run `wide families`" if not active
                      else f"{len(active)} event_families proposals are active; a run has one")  # fmt: skip
    return active[0], core.EventFamilies.model_validate_json(core.canonical_json(dict(active[0].payload)))


# ---------------------------------------------------------------- 10. estimate

Label = Literal[0, 1, "U"]


def _gold(store: Store, gold: core.SampleDraw) -> dict[str, tuple[Label, Label]]:
    """Each gold item's (in the population's voice, exposed) as labelled: 1, 0, or "U" (unsure, not_established or not
    yet labelled: unresolved, never 0)."""
    labelled: dict[tuple[str, str], core.GoldLabel] = {}
    for g in store.all("gold_label", core.GoldLabel):
        labelled[(g.item, g.question.id.split(".")[0])] = g
    out: dict[str, tuple[Label, Label]] = {}
    for d in gold.items:
        voice, exposure = labelled.get((d.item, "voice_in_population")), labelled.get((d.item, "exposure_to_measure"))

        def value(g: core.GoldLabel | None, yes: tuple[str, ...], no: tuple[str, ...]) -> Label:
            if g is None or g.outcome != "labeled" or g.label is None:
                return "U"
            return 1 if g.label in yes else 0 if g.label in no else "U"

        out[d.item] = (value(voice, ("in_population",), ("other_author",)),
                       value(exposure, wide.EXPOSED, ("no_loss",)))  # fmt: skip
    return out


def _complete(d: Label, e: Label, lowest: bool) -> tuple[int, int]:
    """(denominator, numerator) under the completion (decisions, "F4 from the labels to the leads")."""
    if lowest:
        e2 = 0 if e == "U" else e
        d2 = (0 if e2 == 1 else 1) if d == "U" else d
    else:
        e2 = 1 if e == "U" else e
        d2 = (1 if e2 == 1 else 0) if d == "U" else d
    return int(d2), int(d2 * e2)


def _pieces(s: inference.RatioSet) -> list[tuple[float, float]]:
    return [(float(lo), float(hi)) for lo, hi in s.pieces]


def _rays(s: inference.RatioSet) -> list[tuple[float | None, float | None]]:
    return [(None if np.isinf(lo) else float(lo), None if np.isinf(hi) else float(hi)) for lo, hi in s.pieces]


def estimate(store: Store, *, level: float = 0.95) -> core.WideEstimate:
    """Step 10: the share, overall and per family, committed (spec §7.2 step 5; §7.4)."""
    objective = wide.approved_objective(store)
    frame, draw, gold = wide.the_frame(store), wide.drawn(store, 1), wide.drawn(store, 2)
    fam_proposal, fams = the_families(store)
    family_of = {d: f.id for f in fams.families for d in f.dockets}
    f = wide.predictions(store)
    unread = wide.unreadable(store)
    labels_ = _gold(store, gold)
    strata = wide.phase1_strata(frame)
    items = list(draw.items)
    docket = {i.id: i.docket for i in frame.items}
    readable_strata = list(dict.fromkeys(d.stratum for d in items if d.item in f))
    allocation = wide.gold_allocation(readable_strata)
    cuts: list[float]
    if unread:  # an unreadable item's indicators are known 0: its own cell, below every prediction, taken whole
        cuts = [0.0, wide.PHASE2_CUT]
        held = {h: sum(1 for d in items if d.stratum == h and d.item in unread) for h in strata}
        alloc = {(h, 0): held.get(h, 0) for h in strata} | {(h, b + 1): n for (h, b), n in allocation.items()}
    else:
        cuts, alloc = [wide.PHASE2_CUT], dict(allocation)
    stratifier = [-1.0 if d.item in unread else f[d.item][0] * f[d.item][1] for d in items]
    observed = [d.item in unread or d.item in labels_ for d in items]
    f_den = [0.0 if d.item in unread else f[d.item][0] for d in items]
    f_num = [0.0 if d.item in unread else f[d.item][0] * f[d.item][1] for d in items]
    scopes = ["overall", *(x.id for x in fams.families)]

    def run(lowest: bool) -> dict[str, inference.FrameRatio]:
        y = [(0, 0) if d.item in unread or d.item not in labels_ else _complete(*labels_[d.item], lowest) for d in items]
        nums = {}
        for scope in scopes:
            inside = [scope == "overall" or family_of.get(docket[d.item]) == scope for d in items]
            nums[scope] = ([fn if k else 0.0 for fn, k in zip(f_num, inside, strict=True)],
                           [yy[1] if k else 0 for yy, k in zip(y, inside, strict=True)])  # fmt: skip
        return inference.two_phase_ratio(
            frame_size=len(frame.items), unit="item", phase1_frame_sizes={h: len(v) for h, v in strata.items()},
            phase1_sample_sizes={h: sum(1 for d in items if d.stratum == h) for h in strata}, phase2_cuts=cuts,
            phase2_allocation=alloc, phase1_stratum=[d.stratum for d in items], stratifier=np.array(stratifier),
            gold=np.array(observed), f_den=np.array(f_den), y_den=np.array([yy[0] for yy in y]),
            numerators={k: (np.array(v[0]), np.array(v[1])) for k, v in nums.items()}, level=level,
        )  # fmt: skip

    low, high = run(True), run(False)

    def read(r: inference.FrameRatio) -> core.RatioRead:
        return core.RatioRead(estimate=r.estimate, interval=_pieces(r.interval), interval_kind=r.interval.kind,
                              fieller=_rays(r.fieller), fieller_kind=r.fieller.kind)  # fmt: skip

    out = []
    for scope in scopes:
        lo, hi = read(low[scope]), read(high[scope])
        if lo.interval and hi.interval:
            rd: tuple[float, float] | None = (lo.interval[0][0], hi.interval[-1][1])
            reason = "from the lowest completion's lower bound to the highest completion's upper bound"
        else:
            rd = None
            reason = (low[scope] if not lo.interval else high[scope]).interval.reason
        out.append(core.ScopeEstimate(scope=scope, lowest=lo, highest=hi, read=rd, reason=reason))
    unresolved = sum(1 for d, e in labels_.values() for x in (d, e) if x == "U")
    est = core.WideEstimate(
        objective=core.Ref(id=objective.id, version=objective.version), frame=frame.id, design=draw.design,
        draws=(draw.content_hash(), gold.content_hash()), families=fam_proposal.id,
        gold_labels=sorted(g.content_hash() for g in store.all("gold_label", core.GoldLabel)
                           if g.design == draw.design),  # fmt: skip
        judgments=wide.phase1_judgments(store),
        level=level, interval_method=inference.RATIO_INTERVAL_METHOD,
        phase1_items=len(items), unreadable_items=len(unread), gold_items=len(gold.items),
        unresolved_gold_labels=unresolved, scopes=out,
    )  # fmt: skip
    store.commit("wide_estimate", f"{draw.design.id}.estimate", est)
    return est


# ---------------------------------------------------------------- 11. leads


def _lead_asks(st: labels.ItemState) -> list[providers.Ask]:
    """The lead templates asked of one item in the phase-1 configuration, in their original wording."""
    provider, _, render = labels.PHASE1_CONFIG
    spec = providers.PROVIDERS[provider]
    rendered = render_state(st.state, render, claims=st.claims, artifacts=st.artifacts, definitions={})
    out = []
    for tid in LEAD_TEMPLATES:
        t = library.template(tid)
        q = library.instantiate(t, {}, question_id=tid, paraphrase_set=f"ps.{tid}")
        config = core.JudgeConfig(provider=spec.name, model=spec.model, model_cutoff=spec.model_cutoff,
                                  paraphrase_id="p0", render_id=render.id)  # fmt: skip
        out.append(providers.Ask(q, st.state.content_hash(), rendered, ask_question(q, library.original(t)), config))
    return out


def _extract(st: labels.ItemState, measure: str, agent: Callable[..., Any]) -> tuple[Any, dict[str, Any]]:
    """The impact_extractor's answer for one filing (an agent call, safe on a worker thread: it touches no store)."""
    agent_input = {"comment": "\n\n".join(c.text for c in st.claims.values()), "measure": measure}
    return agent("impact_extractor", agent_input, ImpactsDraft, effort="medium"), agent_input


def _impacts(store: Store, st: labels.ItemState, result: Any, agent_input: Mapping[str, Any]) -> list[core.QuantifiedImpact]:
    """The impact_extractor's quotes of the cost impacts the filing states, each checked by code (the quote is in the
    filing's text and states the amount) and committed as a claim; a quote that fails is not a claim."""
    texts = {cid: c.text for cid, c in st.claims.items()}
    record_run(store, result, agent_input, [])
    out = []
    for imp in cast(ImpactsDraft, result.output).impacts:
        try:
            amount = Decimal(imp.amount)
        except InvalidOperation:
            continue
        digits = re.sub(r"[^0-9.]", "", imp.quote.replace(",", ""))
        if not imp.quote or str(amount.normalize()).rstrip("0").rstrip(".") not in digits.rstrip("0").rstrip(".") + digits:
            continue
        for cid, text in texts.items():
            at = text.find(imp.quote)
            if at < 0:
                continue
            base = st.claims[cid]
            span = (base.span[0] + at, base.span[0] + at + len(imp.quote))
            claim = core.Claim(id=core.claim_id(base.artifact, base.extraction, span, imp.quote), artifact=base.artifact,
                               extraction=base.extraction, span=span, text=imp.quote, authority=base.authority)  # fmt: skip
            store.commit_claim(claim)
            annual = Fraction(amount) * 12 if imp.period == "month" else Fraction(amount) if imp.period == "year" else None
            out.append(core.QuantifiedImpact(claim=claim.id, amount=Fraction(amount), currency=imp.currency.upper(),
                                             period=imp.period, annual=annual))  # fmt: skip
            break
    return out


def find_leads(store: Store, *, workers: int = 16, agent: Callable[..., Any] | None = None) -> core.LeadList:
    """Step 11: the leads, asked, extracted and ranked (decisions, "F4 from the labels to the leads")."""
    est_key = f"{wide.drawn(store, 1).design.id}.estimate"
    if store.get("wide_estimate", est_key, core.WideEstimate) is None:
        raise Refused("the estimate is not committed: run `wide estimate`")
    _, fams = the_families(store)
    family_of = {d: f.id for f in fams.families for d in f.dockets}
    f = wide.predictions(store)
    gold = _gold(store, wide.drawn(store, 2))
    ms = wide.measures(store)
    candidates: dict[str, tuple[Literal["gold", "judged"], float]] = {}
    for item, (voice, exposed) in f.items():
        g = gold.get(item)
        if g is not None and "U" not in g:
            if g == (1, 1):
                candidates[item] = ("gold", voice * exposed)
        elif voice * exposed >= LEAD_CUT:
            candidates[item] = ("judged", voice * exposed)
    states, _ = wide._states(store, sorted(candidates))
    by_item = {st.item.id: st for st in states}
    asks = [a for st in states for a in _lead_asks(st)]
    judgments = providers.judge_all(asks, workers=workers)
    for j in judgments:
        store.commit_judgment(j)
    answers: dict[str, dict[str, str]] = {}
    jids: dict[str, list[str]] = {}
    for a, j in zip(asks, judgments, strict=True):
        item = next(i for i, st in by_item.items() if st.state.content_hash() == a.state_hash)
        top = "not_established"
        if j.distribution is not None:
            ranked = sorted(j.distribution.items(), key=lambda kv: (-float(kv[1]), kv[0]))
            if len(ranked) == 1 or float(ranked[0][1]) > float(ranked[1][1]):
                top = ranked[0][0]
        answers.setdefault(item, {})[a.question.id] = top
        jids.setdefault(item, []).append(j.content_hash())
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {i: ex.submit(_extract, st, ms[st.item.container], agent or runner.run) for i, st in by_item.items()}
        extracted = {i: fu.result() for i, fu in futures.items()}
    impacts = {i: _impacts(store, by_item[i], *extracted[i]) for i in sorted(extracted)}
    items = committed(store, list(candidates))

    def key(i: str) -> tuple[Any, ...]:
        asks_ = answers[i].get("seeks_certainty") == "asks"
        usd = [x.annual for x in impacts[i] if x.annual is not None and x.currency == "USD"]
        tier = 3 - int(asks_) - int(bool(impacts[i]))
        return (tier, candidates[i][0] != "gold", -candidates[i][1], -(max(usd) if usd else Fraction(-1)), i)

    ranked_items = sorted(candidates, key=key)
    leads = [core.Lead(rank=n, tier=cast(Literal[1, 2, 3], key(i)[0]), item=i, docket=items[i].container,
                       family=family_of[items[i].container], submitter=items[i].submitter, basis=candidates[i][0],
                       probability=candidates[i][1], answers=answers[i], judgments=jids[i], impact=impacts[i])
             for n, i in enumerate(ranked_items, 1)]  # fmt: skip
    out = core.LeadList(estimate=est_key, leads=leads)
    store.commit("lead_list", f"{wide.drawn(store, 1).design.id}.leads", out)
    return out


# ---------------------------------------------------------------- 12. promote


def propose_lead(store: Store, rank: int) -> core.Proposal:
    """Step 12: a lead proposed for a deep case; a human decides at G1 (`requisite approve <id> --gate G1`)."""
    design = wide.drawn(store, 1).design.id
    leads = store.get("lead_list", f"{design}.leads", core.LeadList)
    if leads is None:
        raise Refused("no leads are committed: run `wide leads`")
    lead = next((x for x in leads.leads if x.rank == rank), None)
    if lead is None:
        raise Refused(f"no lead ranked {rank}: there are {len(leads.leads)}")
    payload = core.LeadPromotion(leads=f"{design}.leads", lead=lead, case=f"deep-{lead.item}").model_dump(mode="json")
    existing = store.proposal(wide._proposal_id("lead", payload))
    if existing is not None:
        return existing
    return store.submit_proposal(core.Proposal(
        id=wide._proposal_id("lead", payload), case=store.case, kind="lead", payload=payload, author="requisite",
        rationale=f"lead {rank}: {lead.submitter or lead.item} on {lead.docket} ({lead.family})", status="proposed",
    ))  # fmt: skip


def summary(est: core.WideEstimate) -> Sequence[Mapping[str, Any]]:
    return [{"scope": s.scope, "read": s.read, "lowest": s.lowest.estimate, "highest": s.highest.estimate,
             "reason": s.reason} for s in est.scopes]  # fmt: skip
