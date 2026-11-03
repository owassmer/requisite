"""Labelling a wide run's items with its templates (spec §7.2 steps 2–3): each item's text as the observed state, each
template instantiated with the objective's terms, asked of decision models through the recorded provider layer.

The state of an item is its own filing: one observed claim per artifact, quoting the artifact's whole extracted text,
as of when the item was read (the frame's as_of fixes which items are in the frame; an item's text is evidenced from
its retrieval, so its state is as of then: addendum A11 §11.3, core.observable_at). The objective's terms bind the
templates' slots, and a docket's measure binds the exposure template, so the questions asked are exactly the ones the
objective's estimand counts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from requisite import core
from requisite.judgment import providers
from requisite.judgment.render import RENDERS, Render, ask, render_state
from requisite.library import templates as library
from requisite.stability.diagnose import S_ARG, S_PROB
from requisite.stability.scores import scores
from requisite.store import Refused, Store

OBJECTIVE_TEMPLATES = ("voice_in_population", "exposure_to_measure")


@dataclass(frozen=True)
class ItemState:
    item: core.Item
    state: core.JudgmentState
    claims: Mapping[str, core.Claim]
    artifacts: Mapping[str, core.Artifact]


def item_state(store: Store, item: core.Item, *, evaluated_at: datetime) -> ItemState:
    """The item's filing as a judgment state: each read part's text as one observed claim (spec §5: Supports, with a
    span) asserted by its submitter, committed through the store, every artifact observable at the state's as_of (spec
    I3). Only a readable item has a state (core.Item.readable); anything missing that the item says was read is an
    integrity failure and stops the step, never a reason to set the item aside."""
    if not item.readable:
        raise ValueError(f"item {item.id} has no readable part, so it has no state to judge: {item.unreadable}")
    claims: dict[str, core.Claim] = {}
    artifacts: dict[str, core.Artifact] = {}
    authority = item.submitter or "the filer (name not published)"
    for art_id in item.artifacts:
        artifact = store.get("artifact", art_id, core.Artifact)
        if artifact is None:
            raise Refused(f"item {item.id} names artifact {art_id}, which is not committed")
        artifacts[art_id] = artifact
    for part in item.parts:
        if part.extraction is None:
            continue
        assert part.artifact is not None  # core.ItemPart: text is read from retrieved content
        text = store.extraction(part.artifact, part.extraction)
        if text is None:
            raise Refused(f"item {item.id}: {part.label}'s extraction {part.extraction} is not committed")
        claim = core.Claim(
            id=core.claim_id(part.artifact, part.extraction, (0, len(text)), text, authority),
            artifact=part.artifact,
            extraction=part.extraction,
            span=(0, len(text)),
            text=text,
            authority=authority,
        )
        store.commit_claim(claim)
        claims[claim.id] = claim
    as_of = max(a.available_from for a in artifacts.values())
    for a in artifacts.values():
        if not core.observable_at(a, as_of, evaluated_at):
            raise Refused(f"artifact {a.id} of item {item.id} is not observable as of {as_of}")
    state = core.JudgmentState(as_of=as_of, observed=list(claims))
    return ItemState(item, state, claims, artifacts)


def horizon(window: core.TimeWindow) -> str:
    """The objective's horizon as the question states it."""
    start = window.start.isoformat() if window.start is not None else "no start"
    end = window.end.isoformat() if window.end is not None else "no end"
    return (f"from {start} ({'included' if window.start_closed else 'excluded'}) to {end} "
            f"({'included' if window.end_closed else 'excluded'}), {window.tz}")  # fmt: skip


def measure_said(mt: core.MeasureText) -> str:
    """What a measure's own text states, as the exposure question binds it and the labeller reads it: the source's
    summary, and each statement of who pays what, quoted, with the reader's reading after it (core.MeasureText)."""
    src = "the Federal Register" if mt.fr_number else "regulations.gov"
    lines = [f"Summary (the agency's own, as {src} publishes it): {mt.summary}" if mt.summary is not None
             else f"Summary: none ({mt.summary_why})."]  # fmt: skip
    if mt.statements:
        lines.append("What its text says of who pays what (each statement quoted from the text; after the dash, an AI "
                     "reader's reading of it):")  # fmt: skip
        for n, c in enumerate(mt.statements, 1):
            amount = f"; {c.amount} {c.currency}{f' {c.per}' if c.per else ''}" if c.amount is not None else ""
            lines.append(f'{n}. "{c.quote}" - {c.payer}; {c.pays_for}; pays {c.direction.replace("_", " ")}{amount}')
    else:
        lines.append(f"What its text says of who pays what: nothing; {mt.none_why}")
    return "\n".join(lines)


def questions(
    objective: core.WideObjective, measures: Mapping[str, core.ItemMeasure], texts: Mapping[str, core.MeasureText],
    item: core.Item,
) -> dict[str, core.AtomicQuestion]:
    """The objective's two questions for one item: the population; and the exposure question with the measure the item
    addresses (core.ItemMeasure), what that measure's own text states (core.MeasureText), and the objective's event
    class, horizon and loss definition bound, each as the approved objective states it, so the question asked is the one
    the estimand counts."""
    if item.id not in measures:
        raise Refused(f"no measure is committed for item {item.id}")
    measure = measures[item.id]
    if measure.document not in texts:
        raise Refused(f"the text of measure {measure.document} is not read: run `requisite wide measure-text`")
    voice = library.instantiate(
        library.template("voice_in_population"),
        {"population": objective.population},
        question_id="voice_in_population",
        paraphrase_set="ps.voice_in_population",
    )
    exposure = library.instantiate(
        library.template("exposure_to_measure"),
        {"measure": measure.statement, "measure_text": measure_said(texts[measure.document]), "events": objective.events,
         "horizon": horizon(objective.horizon), "loss_definition": objective.loss_definition},
        question_id=f"exposure_to_measure.{measure.document}",
        paraphrase_set="ps.exposure_to_measure",
    )  # fmt: skip
    return {"voice_in_population": voice, "exposure_to_measure": exposure}


def paraphrased(q: core.AtomicQuestion, pset: core.ParaphraseSet, paraphrase_id: str) -> core.Paraphrase:
    """One paraphrase of a question's template, its slots filled with the question's bindings."""
    p = next(x for x in pset.paraphrases if x.id == paraphrase_id)
    return core.Paraphrase(id=p.id, target=library.fill(p.target, q.bindings), meanings=dict(p.meanings))


def asks(
    st: ItemState,
    qs: Mapping[str, core.AtomicQuestion],
    psets: Mapping[str, core.ParaphraseSet],
    configs: Sequence[tuple[str, str, Render]],
) -> list[providers.Ask]:
    """Every (question, configuration) ask for one item; a configuration is (provider, paraphrase id, render)."""
    out = []
    state_hash = st.state.content_hash()
    for template_id, q in qs.items():
        for provider, paraphrase_id, render in configs:
            spec = providers.PROVIDERS[provider]
            rendered = render_state(st.state, render, claims=st.claims, artifacts=st.artifacts, definitions={})
            asked = ask(q, paraphrased(q, psets[template_id], paraphrase_id))
            config = core.JudgeConfig(
                provider=spec.name, model=spec.model, model_cutoff=spec.model_cutoff, paraphrase_id=paraphrase_id,
                render_id=render.id,
            )  # fmt: skip
            out.append(providers.Ask(q, state_hash, rendered, asked, config))
    return out


# Phase 1 labels every item with one configuration (spec §7.2 step 2: run every template over the corpus); stability
# varies every axis on a sample (step 3). The phase-1 configuration is the cheapest provider in the original wording and
# the plain render (coordinator, November 1).
PHASE1_CONFIG = ("openai_decisions", "p0", RENDERS[0])


def stability_configs() -> list[tuple[str, str, Render]]:
    """spec §2.1's axes at their default counts: every provider, three paraphrases, two renders."""
    return [(p, ph, r) for p in providers.PROVIDERS for ph in ("p0", "p1", "p2") for r in RENDERS]


@dataclass(frozen=True)
class TemplateStability:
    """A template's stability over a sample of items (spec §2.1: wide mode tests each template on a random sample of at
    least 50 items): each item's scores over the configurations, and the share of items that meet the thresholds."""

    template: str
    items: int
    stable_share: float
    mean_s_arg: float
    mean_s_prob: float
    per_item: Mapping[str, tuple[float, float]]
    refusals: int




def template_stability(template_id: str, judgments: Sequence[core.Judgment]) -> TemplateStability:
    """Scores per item (by state) over the configurations that returned a distribution."""
    by_state: dict[str, list[core.Judgment]] = {}
    for j in judgments:
        by_state.setdefault(j.state_hash, []).append(j)
    per_item: dict[str, tuple[float, float]] = {}
    refusals = 0
    for state_hash, js in by_state.items():
        dists = [dict(j.distribution) for j in js if j.distribution is not None]
        refusals += sum(1 for j in js if j.distribution is None)
        if len(dists) >= 2:
            sc = scores(dists)
            per_item[state_hash] = (sc.s_arg, sc.s_prob)
    n = len(per_item)
    stable = sum(1 for a, b in per_item.values() if a >= S_ARG and b >= S_PROB)
    return TemplateStability(
        template=template_id,
        items=n,
        stable_share=stable / n if n else 0.0,
        mean_s_arg=sum(a for a, _ in per_item.values()) / n if n else 0.0,
        mean_s_prob=sum(b for _, b in per_item.values()) / n if n else 0.0,
        per_item=per_item,
        refusals=refusals,
    )
