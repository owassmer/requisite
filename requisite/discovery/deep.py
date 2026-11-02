"""Deep discovery, once the case's entity model has passed G1 (spec §7.1 steps 3-5, the requisite loop over them §2.3;
WP11a; outcome 003). Code runs every step; decision models answer the typed questions, agents only propose (a refined
question, a split, a sharper band, candidate listed markets), and humans decide at the gates.

What a case holds before these steps (each committed by the steps that come before, at G1): an active entity model
(core.EntityModel: the driver tree, liquidity profile, horizon and loss definition), an active decision spec
(core.DecisionSpec: its decision time is the evidence's as_of, its tolerance is the economic tolerance relevance is
judged at, its alpha the tail level, its floor the liquidity floor; its horizon is the entity model's), the pending
events as active propositions (domain bool, each with its window), and the observed claims the judges read. The deep
templates' paraphrase sets (requisite/library/templates/deep.json, forecast.json) are active in the case.

The steps, each a function the CLI calls (`requisite deep <step> --case <dir>`):

1. `sensitivities`: for each driver and each pending event, spec §7.1 step 3's questions (direction, mechanism,
   magnitude band, lag band, conditionality), and for each event its forecast (requisite/library/templates/
   forecast.json), over the case's evidence state; every question in the phase-1 configuration, and every question
   that reaches the decision (its answer feeds the materiality graph) in spec §2.1's stability configurations,
   resolved into a core.ResolvedJudgment (requisite/stability/diagnose.py), the forecast's into the event's
   core.ProbabilityEstimate. Mechanism reaches no decision: it is asked once, and shown as that reading.
2. `materiality`: the case's computation graph (requisite/compute/runtime.py) over registered library nodes
   (requisite/library/nodes/materiality.py, compose.py), from committed records only, recomputed incrementally; a
   parameter change recomputes only what depends on it, and the diff shows it (`diff`).
3. `hedge`: spec §7.1 step 5's screen: for each exposure, whether an authoritative observable tracks the loss and
   whether the business could influence it, for the event's own record and for each listed market an agent finds close
   in Kalshi's read-only listing (requisite/market/kalshi.py). These reach no top-k decision and are asked once.
4. `register`: one core.RegisterRow per driver-event pair whose numbers the evidence bounds, every number a vertex of
   the graph, every row naming its claims, judgments and vertex ids; an exposure whose loss has no bound (an open band,
   an unestablished magnitude) is listed apart with the questions that leave it so.
5. `loop`: the requisite loop (spec §2.3) with the decision "enters the top k of the register". Each question's
   tie-robust relevance is its min-max regret over its feasible answers at the decision spec's tolerance (addendum A03
   §3.3); within the tolerance it is informational for that decision, out of the review queue and still inspectable
   (A02 §3.4). An unstable relevant question gets its diagnosis's action: an agent refines it (imprecise_target), or
   proposes a split whose sub-questions and composition are admitted only through core.SPLIT_GATES (compound), or it
   goes to a human (contested: G2; unexplained). A stable answer that hides a decision-relevant distinction yields a
   core.RefinementWitness and an agent's sharper question (A02 §3.2). The loop stops when every relevant question is
   stable and no witness remains (requisite), when only a human can act (awaiting_human), or when its budget is spent
   (blocked: a reproducible draft, never an approved conclusion, addendum A02 §3.7, A.T22); each run is committed as a
   RequisiteRun.

The top-k decision. The register's exposures are ranked by their materiality: the tail impact where the decision's
objective is a CVaR, the expected impact otherwise. Each is an interval, so the decision is a choice of k exposures
under interval uncertainty: an action is a set S of k exposures, and its cost under a state is its worst-case regret,
the most materiality a set of k could hold beyond what S holds, over every value the intervals allow, J(S) = max over
j of (the j largest upper ends outside S less the j smallest lower ends inside S) (a convex function of the values, so
its maximum over the box is at a corner). An exposure open above is in every set with bounded cost, one open below in
none. A question's relevance compares J across its answers by min-max regret (A03 §3.3); a forecast's, across the
probabilities its configurations gave (A03 §3.6: the event's outcome is not observable before the decision, so its
relevance is whether the judges' range of probabilities moves the decision).
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from typing import Any, Literal

from pydantic import ValidationError

import requisite.library.nodes  # noqa: F401  registers the library's consequence nodes the graph names
from requisite import core
from requisite.agents import runner
from requisite.agents.answers import CategoriesDraft, MarketsDraft, QuestionsDraft
from requisite.agents.trace import record_run
from requisite.compute import registry
from requisite.compute.runtime import Diff, Graph, Instance, Runtime, Source
from requisite.core.base import FrozenDict, freeze
from requisite.core.judgment import Diagnosis
from requisite.core.kleene import Kleene, conj, disj, from_status, neg
from requisite.discovery import labels, wide
from requisite.judgment import providers
from requisite.judgment.render import RENDERS, TEXT, Render, RenderedState, ask, render_state
from requisite.library import templates as library
from requisite.library.nodes.compose import binds_key
from requisite.relevance import queue as review
from requisite.relevance.regret import Regret, minmax_regret
from requisite.stability import diagnose
from requisite.store import Refused, Store

JudgeCall = Callable[[providers.Ask], core.Judgment]
AgentCall = Callable[..., runner.AgentResult[Any]]
SeriesListing = Callable[[datetime], Sequence[Mapping[str, Any]]]

SENSITIVITY = {
    "sensitivity_direction": "direction",
    "sensitivity_mechanism": "mechanism",
    "sensitivity_magnitude": "magnitude",
    "sensitivity_lag": "lag",
    "sensitivity_conditionality": "conditional",
}
FORECAST = "event_occurs"
# The slots whose answers feed the materiality graph, so reach the register's decision (spec §4 AtomicQuestion.reaches;
# spec §6.6: "Deep mode runs stability checks only on questions that reach a decision").
FEEDS = ("direction", "magnitude", "lag", "conditional", "forecast")
BINDS = {"direction": "direction", "magnitude": "magnitude", "lag": "lag_months", "conditional": "conditional"}
MAX_ATTEMPTS = 2  # an agent's attempts at one action on one question before it goes to a human (ours)
DEFAULT_BUDGET = 2000  # judge and agent calls the loop may make beyond the base run (spec §6.6: a cap on judge calls)
BASELINE = ("standalone: this exposure alone, if its event occurs, against the business's cash and undrawn credit at the "
            "horizon's start, with no other exposure and none of the business's other operating cash flows")  # fmt: skip


class BudgetSpent(RuntimeError):
    """The loop's budget of calls is spent: the run freezes as a blocked draft (addendum A02 §3.7)."""


# ---------------------------------------------------------------- the case


@dataclass(frozen=True)
class Case:
    """A deep case as its committed records give it, past G1."""

    store: Store
    model: core.EntityModel
    model_proposal: str
    decision: core.DecisionSpec
    decision_proposal: str
    events: tuple[core.Proposition, ...]
    state: core.JudgmentState
    claims: Mapping[str, core.Claim]
    artifacts: Mapping[str, core.Artifact]

    @property
    def exposures(self) -> list[tuple[core.Driver, core.Proposition]]:
        return [(d, e) for d in self.model.drivers for e in self.events]

    @property
    def state_hash(self) -> str:
        return self.state.content_hash()

    def rendered(self, render: Render) -> RenderedState:
        return render_state(self.state, render, claims=self.claims, artifacts=self.artifacts, definitions={})


def _active(store: Store, kind: str) -> list[core.Proposal]:
    return [p for p in wide.proposals(store, kind) if p.status == "active"]


def _payload[R: core.Record](p: core.Proposal, cls: type[R]) -> R:
    return cls.model_validate_json(core.canonical_json(dict(p.payload)))


def open_deep(store: Store) -> Case:
    """The case's entity model and decision spec past G1, its pending events and its evidence state."""
    models = sorted(((_payload(p, core.EntityModel), p.id) for p in _active(store, "entity_model")),
                    key=lambda m: m[0].version)  # fmt: skip
    if not models:
        raise Refused("no entity model has passed G1 in this case: a deep case starts from an approved entity model")
    model, model_id = models[-1]
    if len(models) > 1 and models[-2][0].version == model.version:
        raise Refused(f"two entity models of version {model.version} have passed G1; a case has one at a time")
    specs = sorted(((_payload(p, core.DecisionSpec), p.id) for p in _active(store, "decision_spec")),
                   key=lambda s: s[0].version)  # fmt: skip
    if not specs:
        raise Refused("no decision spec has passed G1 in this case: its tolerance, tail level and floor are the "
                      "decision's (spec §2.4 G1)")  # fmt: skip
    spec, spec_id = specs[-1]
    if spec.horizon != model.horizon:
        raise Refused(f"the decision spec's horizon {spec.horizon} is not the entity model's {model.horizon}: a case "
                      "has one planning horizon")  # fmt: skip
    if spec.currency != model.currency:
        raise Refused(f"the decision spec is in {spec.currency} and the entity model in {model.currency}")
    events = tuple(sorted((e for e in (_payload(p, core.Proposition) for p in _active(store, "proposition"))
                           if e.domain == "bool"), key=lambda e: e.id))  # fmt: skip
    if not events:
        raise Refused("no pending event is an active proposition in this case")
    state, claims, artifacts = _evidence(store, spec.decision_time)
    return Case(store, model, model_id, spec, spec_id, events, state, claims, artifacts)


def _evidence(store: Store, as_of: datetime) -> tuple[core.JudgmentState, dict[str, core.Claim], dict[str, core.Artifact]]:
    """The case's evidence state as of the decision time: every observed claim the case holds whose artifact is
    observable then (spec I3, core.observable_at)."""
    now = store.clock()
    claims: dict[str, core.Claim] = {}
    artifacts: dict[str, core.Artifact] = {}
    for c in store.all("claim", core.Claim):
        a = store.get("artifact", c.artifact, core.Artifact)
        if a is not None and core.observable_at(a, as_of, now):
            claims[c.id], artifacts[a.id] = c, a
    if not claims:
        raise Refused(f"no observed claim of this case is observable as of {as_of.isoformat()} (spec I3)")
    return core.JudgmentState(as_of=as_of, observed=sorted(claims)), claims, artifacts


# ---------------------------------------------------------------- questions


@dataclass(frozen=True)
class Asked:
    """A question of the case and where it sits: its slot (what its answer is about), the exposure or event it is
    about, and for a sub-question, the question it replaces and its attempt and position."""

    question: core.AtomicQuestion
    slot: str
    event: str
    driver: str | None = None
    observable: str | None = None
    parent: str | None = None
    attempt: str | None = None
    position: int = 0

    @property
    def id(self) -> str:
        return self.question.id

    @property
    def exposure(self) -> str | None:
        return None if self.driver is None else f"{self.driver}.{self.event}"

    @property
    def feeds(self) -> bool:
        """Whether its answers enter a vertex of the register decision's computation (spec §4 AtomicQuestion.reaches)."""
        return self.slot in FEEDS


def _money(x: Decimal) -> str:
    return f"{x:,}"


def driver_text(d: core.Driver) -> str:
    return (f"{d.name}, a {d.line} line of the business: {d.quantity.name} ({_money(d.quantity.value)} "
            f"{d.quantity.unit}) times {d.price.name} ({_money(d.price.value)} {d.price.unit}), per {d.period}")  # fmt: skip


def _window(w: core.TimeWindow | None) -> str:
    if w is None:
        return "no stated window"
    start = w.start.isoformat() if w.start else "any time"
    end = w.end.isoformat() if w.end else "no end"
    return f"from {start} to {end} ({w.tz})"


def event_text(e: core.Proposition) -> str:
    return e.text if e.window is None else f"{e.text} (window: {_window(e.window)})"


def decision_ref(case: Case) -> str:
    return f"register.top_k.{case.decision.id}"


def _q(t: str, bindings: Mapping[str, Any], qid: str, case: Case, slot: str, window: core.TimeWindow | None = None
       ) -> core.AtomicQuestion:  # fmt: skip
    q = library.instantiate(library.template(t), bindings, question_id=qid, paraphrase_set=f"ps.{t}")
    reaches = [decision_ref(case)] if slot in FEEDS else []
    return q.model_copy(update={"window": window, "reaches": tuple(reaches)})


def base_questions(case: Case) -> list[Asked]:
    """Spec §7.1 step 3's questions for every driver-event pair, and each event's forecast."""
    out: list[Asked] = []
    horizon = _window(case.model.horizon)
    for e in case.events:
        out.append(Asked(_q(FORECAST, {"event": e.text}, f"deep.{FORECAST}.{e.id}", case, "forecast", e.window),
                         "forecast", e.id))  # fmt: skip
        for d in case.model.drivers:
            for t, slot in SENSITIVITY.items():
                b = {"driver": driver_text(d), "event": event_text(e)}
                if "horizon" in library.template(t).slots:
                    b["horizon"] = horizon
                out.append(Asked(_q(t, b, f"deep.{t}.{d.id}.{e.id}", case, slot), slot, e.id, d.id))
    return out


REFINER = "question_refiner"


def _number(tag: str) -> int:
    return int("".join(ch for ch in tag if ch.isdigit()) or "0")


def _replacements(store: Store, known: Mapping[str, Asked]) -> tuple[list[Asked], dict[str, list[Asked]]]:
    """Every sub-question an agent run proposed in the case (any status), and for each replaced question the
    sub-questions of its latest complete attempt, in the order the run proposed them. An attempt is the question
    proposals one question_refiner run led to, as its trace lists them (core.AgentRun); it is complete when every one
    passed its gates, so a split cut short (its trace never written) or partly rejected replaces nothing."""
    found: dict[tuple[str, str], list[tuple[core.AtomicQuestion, str]]] = {}
    for t in store.all("agent_run", core.AgentRun):
        if t.role != REFINER:
            continue
        props = [p for p in (store.proposal(pid) for pid in t.proposals) if p is not None and p.kind == "question"]
        members = [(_payload(p, core.AtomicQuestion), str(p.status)) for p in props]
        parents = {r.id for q, _ in members for r in q.replaces}
        if len(parents) != 1:
            continue
        tag = members[0][0].id.rsplit("/", 1)[1].rsplit(".", 1)[0]
        found.setdefault((parents.pop(), tag), members)
    subs: list[Asked] = []
    complete: dict[str, tuple[int, list[Asked]]] = {}
    pending = dict(known)
    # Parents first: a sub-question's own sub-questions name it, so attempts are taken in order of depth.
    for (parent, tag), members in sorted(found.items(), key=lambda kv: (kv[0][0].count("/"), kv[0][0], _number(kv[0][1]))):
        if parent not in pending:
            continue
        p = pending[parent]
        asked = [Asked(q, p.slot, p.event, p.driver, p.observable, parent, tag, i) for i, (q, _) in enumerate(members, 1)]
        subs += asked
        pending.update({a.id: a for a in asked})
        if all(status == "active" for _, status in members) and _number(tag) >= complete.get(parent, (0, []))[0]:
            complete[parent] = (_number(tag), asked)
    return subs, {parent: asked for parent, (_, asked) in complete.items()}


def _pset(store: Store, q: Asked) -> core.ParaphraseSet:
    template = q.question.template_id or q.id
    try:
        return wide.paraphrase_set(store, wide.library_ref(template) if q.question.template_id else q.question.ref)
    except Refused:
        raise Refused(f"no active paraphrase set for {template}: run `requisite paraphrases --template {template}` "
                      "before the stability configurations are asked") from None  # fmt: skip


# ---------------------------------------------------------------- judging


@dataclass
class Spend:
    """The calls a run makes, judge and agent alike, against its budget (spec §6.6: "Each run has caps on judge
    calls"); no limit until one is set."""

    limit: int | None = None
    made: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def take(self) -> None:
        with self.lock:
            if self.limit is not None and self.made >= self.limit:
                raise BudgetSpent(f"the run's budget is spent at {self.made} calls")
            self.made += 1


@dataclass
class Judge:
    """The judge calls of a run: each distinct ask made once (a replayed cassette or a test's double), counted."""

    call: JudgeCall
    spend: Spend = field(default_factory=Spend)
    cache: dict[str, core.Judgment] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, a: providers.Ask) -> core.Judgment:
        key = core.sha256(core.canonical_json([a.question.id, a.question.version, a.state_hash,
                                               a.config.model_dump(mode="json"), a.asked.instructions,
                                               dict(a.asked.answers), dict(a.rendered.state)]))  # fmt: skip
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        self.spend.take()
        j = self.call(a)
        with self.lock:
            self.cache[key] = j
        return j


@dataclass
class Agents:
    """The agent calls of a run, counted against the same budget."""

    call: AgentCall
    spend: Spend

    def __call__(self, *args: Any, **kwargs: Any) -> runner.AgentResult[Any]:
        self.spend.take()
        return self.call(*args, **kwargs)


def _config(provider: str, paraphrase: str, render: Render) -> core.JudgeConfig:
    spec = providers.PROVIDERS[provider]
    return core.JudgeConfig(provider=spec.name, model=spec.model, model_cutoff=spec.model_cutoff,
                            paraphrase_id=paraphrase, render_id=render.id)  # fmt: skip


def _asks(case: Case, q: Asked, pset: core.ParaphraseSet | None, configs: Sequence[tuple[str, str, Render]],
          rendered: Mapping[str, RenderedState]) -> list[providers.Ask]:  # fmt: skip
    out = []
    for provider, pid, render in configs:
        wording = labels.paraphrased(q.question, pset, pid) if pset is not None else library.original(q.question)
        out.append(providers.Ask(q.question, case.state_hash, rendered[render.id], ask(q.question, wording),
                                 _config(provider, pid, render)))  # fmt: skip
    return out


def judge_questions(case: Case, qs: Sequence[Asked], judge: Judge, *, workers: int = 8) -> dict[str, list[core.Judgment]]:
    """Every question in spec §2.1's stability configurations, which include the phase-1 configuration, each judgment
    committed. A question whose answers enter no vertex of the decision's computation (the mechanism, the hedgeability
    screen's) is asked in them too: the register applies its answer, and an answer is applied only from a resolution
    under the application policy (A04 §4.1), never from one configuration's reading."""
    rendered = {r.id: case.rendered(r) for r in RENDERS}
    plan: list[tuple[str, providers.Ask]] = []
    for q in qs:
        pset = _pset(case.store, q)
        plan += [(q.id, a) for a in _asks(case, q, pset, labels.stability_configs(), rendered)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        judgments = list(pool.map(judge, [a for _, a in plan]))
    out: dict[str, list[core.Judgment]] = {q.id: [] for q in qs}
    for (qid, _), j in zip(plan, judgments, strict=True):
        case.store.commit_judgment(j)
        out[qid].append(j)
    return out


def applied_truth(q: Asked, r: core.ResolvedJudgment | None) -> Kleene:
    """A question's applied answer as a truth (spec §6.3): its status where the application policy applies an answer,
    unknown where it applies none (I6: never read as false)."""
    if r is None or r.application_state == "review":
        return Kleene.UNKNOWN
    status = next(a.status for a in q.question.answers if a.value == r.application_state)
    return from_status(status)


# ---------------------------------------------------------------- resolution and probability


def _exact_occurs(q: core.AtomicQuestion, j: core.Judgment) -> Fraction:
    occurs = next(a.value for a in q.answers if dict(a.binds).get("occurs") is True)
    assert j.distribution is not None
    return Fraction(repr(float(j.distribution[occurs])))


def estimate(case: Case, e: core.Proposition, q: core.AtomicQuestion, resolved: core.ResolvedJudgment,
             judgments: Sequence[core.Judgment]) -> core.ProbabilityEstimate:  # fmt: skip
    """The event's probability from its forecast (spec §5: "a forecast judgment supplies a branch probability"): the
    pooled P(occurs) under the resolved judgment's pooling policy, computed exactly from each configuration's
    distribution read as the decimals it prints as (decisions, "Numbers"), with the configurations' range as its
    uncertainty (A04 §4.1). An ensemble estimate, not a posterior from independent witnesses (A01 §2.3)."""
    js = diagnose.usable(judgments)
    per_provider: dict[str, int] = {}
    for j in js:
        per_provider[j.config.provider] = per_provider.get(j.config.provider, 0) + 1
    values = [_exact_occurs(q, j) for j in js]
    weights = [Fraction(1, len(per_provider) * per_provider[j.config.provider]) for j in js]
    p = sum((w * v for w, v in zip(weights, values, strict=True)), Fraction(0))
    return core.ProbabilityEstimate(
        id=f"pe.{e.id}.{resolved.content_hash()[:16]}", proposition=e.id, branch_value="true", conditioning=[],
        meaning="physical", source="judgment", source_ref=resolved.content_hash(), value=p,
        interval=(min(values), max(values)), evidence_hash=case.state_hash, as_of=case.state.as_of,
    )  # fmt: skip


@dataclass
class Resolutions:
    """Every question's judgments and, for each tested one with two or more distributions, its resolution."""

    asked: dict[str, Asked]
    base: list[Asked]
    replaced_by: dict[str, list[Asked]]
    judgments: dict[str, list[core.Judgment]]
    resolved: dict[str, core.ResolvedJudgment]
    estimates: dict[str, core.ProbabilityEstimate]
    probes: dict[str, list[core.ProbeResult]]

    def effective(self, qid: str) -> list[Asked]:
        """The questions in play for one question, in order: itself, or its replacement's, recursively."""
        if qid not in self.replaced_by:
            return [self.asked[qid]]
        return [x for s in self.replaced_by[qid] for x in self.effective(s.id)]

    def in_play(self) -> list[Asked]:
        return [x for b in self.base for x in self.effective(b.id)]

    def forecast_leaf(self, event: str) -> Asked:
        base = next(b for b in self.base if b.slot == "forecast" and b.event == event)
        (leaf,) = self.effective(base.id)  # a forecast is refined, never split (diagnose.composition_problems)
        return leaf


def resolve_case(case: Case, judge: Judge, *, extra: Sequence[Asked] = ()) -> Resolutions:
    """Step 1: every question of the case's structure judged and resolved, with each event's probability."""
    base = list(base_questions(case)) + list(extra)
    known = {q.id: q for q in base}
    subs, replaced_by = _replacements(case.store, known)
    asked = {**known, **{s.id: s for s in subs}}
    judgments = judge_questions(case, list(asked.values()), judge)
    attempts: dict[str, dict[str, list[Asked]]] = {}
    for s in subs:
        if s.parent is not None and s.attempt is not None and s.attempt.startswith("split"):
            attempts.setdefault(s.parent, {}).setdefault(s.attempt, []).append(s)
    probes: dict[str, list[core.ProbeResult]] = {}
    for parent, by_attempt in attempts.items():
        p = asked[parent]
        chosen = replaced_by.get(parent)
        # The split probe's evidence is the split in play where there is one, otherwise the latest attempt's.
        key = chosen[0].attempt if chosen and chosen[0].attempt in by_attempt else sorted(by_attempt)[-1]
        members = sorted(by_attempt[key], key=lambda a: a.position)
        props = [_proposal_of(case.store, m.id) for m in members]
        probes[parent] = [diagnose.split_probe(p.question, [diagnose.SubQuestion(m.question, judgments[m.id])
                                                            for m in members], [x for x in props if x])]  # fmt: skip
    resolved: dict[str, core.ResolvedJudgment] = {}
    approvals = [a for a in case.store.all("approval", core.Approval) if a.gate == "G2" and a.decision != "rejected"]
    for qid, q in asked.items():
        if len(diagnose.usable(judgments[qid])) >= 2:
            r = diagnose.resolve(q.question, case.state_hash, judgments[qid], probes=probes.get(qid, ()))
            # A G2 approval names the contested resolution it decided on (its lineage); the answer stays held as not
            # established (spec §2.1), the approval recorded beside it.
            g2 = [a.id for a in approvals if a.subject.id == qid and a.lineage_hash == contested_lineage(r)]
            if g2:
                r = r.model_copy(update={"approvals": tuple(g2)})
            case.store.commit("resolved_judgment", r.content_hash(), r)
            resolved[qid] = r
    res = Resolutions(asked, base, replaced_by, judgments, resolved, {}, probes)
    for e in case.events:
        leaf = res.forecast_leaf(e.id)
        if leaf.id in resolved:
            pe = estimate(case, e, leaf.question, resolved[leaf.id], judgments[leaf.id])
            case.store.commit("probability_estimate", pe.id, pe)
            res.estimates[e.id] = pe
    return res


def _proposal_of(store: Store, qid: str) -> str | None:
    for p in wide.proposals(store, "question"):
        if p.payload.get("id") == qid:
            return p.id
    return None


def feasible(res: Resolutions, q: Asked) -> tuple[FrozenDict, ...]:
    """The binds of the answers a question leaves feasible: its applied answer, or every substantive answer
    where none is applied (A03 §3.3; I6: an unresolved question is never read as one answer)."""
    r = res.resolved.get(q.id)
    applied = r.application_state if r is not None else "review"
    values = frozenset({applied}) if applied in {a.value for a in q.question.answers} else frozenset({"not_established"})
    if not any(a.status == "not_established" for a in q.question.answers):  # a forecast: every branch
        values = frozenset(a.value for a in q.question.answers) if applied == "review" else values
    return diagnose.feasible_binds(q.question, values)


# ---------------------------------------------------------------- the computation graph


def _src(id: str, layer: str, value: Any, *records: core.Record) -> Source:
    return Source(id, layer, value, tuple(records))  # type: ignore[arg-type]


def _inst(id: str, node: str, **inputs: str) -> Instance:
    return Instance(id=id, node=core.Ref(id=node, version=1), inputs=inputs, layer="computation")


def _leaf(q: Asked, res: Resolutions) -> Source:
    r = res.resolved.get(q.id)
    records: tuple[core.Record, ...] = (q.question, r) if r is not None else (q.question, *res.judgments[q.id])
    return _src(f"v:q.{q.id}", "proposition", feasible(res, q), *records)


def _slot(q: Asked, res: Resolutions, out: list[Source | Instance]) -> str:
    """The vertex whose value is a question's feasible binds as the structure answers it: its own source, its
    refinement's, or its split's composition (requisite/library/nodes/compose.py)."""
    subs = res.replaced_by.get(q.id)
    if not subs:
        out.append(_leaf(q, res))
        return f"v:q.{q.id}"
    if len(subs) == 1:
        return _slot(subs[0], res, out)
    heads = [_slot(s, res, out) for s in subs]
    tail = f"v:split.{q.id}.{len(subs)}"
    out.append(_inst(tail, "compose.last", step=heads[-1]))
    for i in range(len(subs) - 1, 0, -1):
        vid = f"v:split.{q.id}.{i}"
        out.append(_inst(vid, "compose.then", head=heads[i - 1], tail=tail))
        tail = vid
    return tail


def exposure_vertices(x: str) -> dict[str, str]:
    return {k: f"v:{x}.{k}" for k in ("effect", "start", "cash", "loss", "expected", "tail", "breach")}


def build_graph(case: Case, res: Resolutions) -> Graph:
    """The case's materiality graph from committed records only (spec §5: every source names its records)."""
    m, d_spec, liq = case.model, case.decision, case.model.liquidity
    vs: list[Source | Instance] = [
        _src("v:horizon", "computation", m.horizon, m.horizon),
        _inst("v:days", "deep.horizon_days", horizon="v:horizon"),
        _src("v:as_of", "computation", d_spec.decision_time, d_spec),
        _src("v:alpha", "computation", d_spec.alpha, d_spec),
        _src("v:floor", "computation", tuple((f.day, f.amount) for f in d_spec.liquidity_floor), d_spec),
        _src("v:cash", "computation", liq.cash.value, liq.cash),
        _src("v:credit", "computation", liq.credit_line.value, liq.credit_line),
        _src("v:payable_days", "computation", liq.payable_days.value, liq.payable_days),
        _src("v:receivable_days", "computation", liq.receivable_days.value, liq.receivable_days),
        _src("v:seasonality", "computation", tuple(p.value for p in liq.seasonality), *liq.seasonality),
    ]
    for d in m.drivers:
        vs += [
            _src(f"v:{d.id}.quantity", "computation", d.quantity.value, d.quantity),
            _src(f"v:{d.id}.price", "computation", d.price.value, d.price),
            _src(f"v:{d.id}.period", "computation", d.period, d),
            _src(f"v:{d.id}.line", "computation", d.line, d),
            _inst(f"v:{d.id}.amounts", "deep.monthly_amounts", quantity=f"v:{d.id}.quantity", price=f"v:{d.id}.price",
                  period=f"v:{d.id}.period", seasonality="v:seasonality", days="v:days"),
        ]  # fmt: skip
    for e in case.events:
        vs.append(_src(f"v:{e.id}.window", "proposition", e.window, e))
        pe = res.estimates.get(e.id)
        if pe is not None:
            vs.append(_src(f"v:{e.id}.p", "proposition", pe.value, pe))
    by_slot = {(b.exposure, b.slot): b for b in res.base if b.exposure is not None}
    for d, e in case.exposures:
        x = f"{d.id}.{e.id}"
        slots = {s: _slot(by_slot[(x, s)], res, vs) for s in ("direction", "magnitude", "lag", "conditional")}
        v = exposure_vertices(x)
        terms = "v:payable_days" if d.line == "cost" else "v:receivable_days"
        vs += [
            _inst(v["effect"], "deep.effect", direction=slots["direction"], magnitude=slots["magnitude"],
                  conditional=slots["conditional"]),
            _inst(v["start"], "deep.cash_start", as_of="v:as_of", window=f"v:{e.id}.window", lag=slots["lag"],
                  terms_days=terms, horizon="v:horizon"),
            _inst(v["cash"], "deep.cash_impact", amounts=f"v:{d.id}.amounts", days="v:days", line=f"v:{d.id}.line",
                  effect=v["effect"], start=v["start"]),
            _inst(v["loss"], "deep.horizon_loss", cash=v["cash"]),
        ]  # fmt: skip
        if e.id in res.estimates:
            vs += [
                _inst(v["expected"], "deep.expected_impact", probability=f"v:{e.id}.p", loss=v["loss"]),
                _inst(v["tail"], "deep.tail_impact", probability=f"v:{e.id}.p", alpha="v:alpha", loss=v["loss"]),
                _inst(v["breach"], "deep.breach", cash=v["cash"], days="v:days", probability=f"v:{e.id}.p",
                      cash_on_hand="v:cash", credit_line="v:credit", floor="v:floor"),
            ]  # fmt: skip
    return Graph(vs)


def evaluate(graph: Graph) -> dict[str, Any]:
    """Every vertex's value by the registered nodes, in topological order, committing nothing: the same functions the
    runtime runs, for the hypothetical states relevance compares (A03 §3.3)."""
    out: dict[str, Any] = {}
    for v in graph:
        if isinstance(v, Source):
            out[v.id] = v.value
        else:
            node = registry.get(v.node)
            out[v.id] = freeze(node.fn(**{name: out[src] for name, src in v.inputs.items()}))
    return out


def materiality(case: Case, values: Mapping[str, Any]) -> dict[str, tuple[Fraction | None, Fraction | None]]:
    """Each exposure's materiality for the register's ranking: its tail impact where the decision's objective is a
    CVaR, its expected impact otherwise (an interval; None: unbounded). Exposures whose event has no probability have
    none."""
    key = "tail" if case.decision.objective == "liquidity_cvar" else "expected"
    out = {}
    for d, e in case.exposures:
        vid = exposure_vertices(f"{d.id}.{e.id}")[key]
        if vid in values:
            out[f"{d.id}.{e.id}"] = values[vid]
    return out


# ---------------------------------------------------------------- the top-k decision and relevance

Box = Mapping[str, tuple[Fraction | None, Fraction | None]]


def worst_regret(chosen: frozenset[str], boxes: Box) -> Fraction | None:
    """J(S): the most materiality k exposures could hold beyond what S holds, over every value the intervals allow
    (None: unbounded): the best of j swaps of S's smallest lower ends for the largest upper ends outside it."""
    inside = [boxes[x][0] for x in chosen]
    outside = [boxes[x][1] for x in boxes if x not in chosen]
    swaps = min(len(inside), len(outside))
    if swaps == 0:
        return Fraction(0)
    if any(v is None for v in outside) or any(v is None for v in inside):
        return None
    hi = sorted((v for v in outside if v is not None), reverse=True)
    lo = sorted(v for v in inside if v is not None)
    best = acc = Fraction(0)
    for j in range(swaps):
        acc += hi[j] - lo[j]
        best = max(best, acc)
    return best


def actions(boxes: Box, k: int) -> list[frozenset[str]]:
    names = sorted(boxes)
    return [frozenset(c) for c in itertools.combinations(names, min(k, len(names)))]


def action_name(s: frozenset[str]) -> str:
    return " + ".join(sorted(s)) or "(none)"


def costs(states: Mapping[str, Box], k: int) -> dict[str, dict[str, Fraction | None]]:
    out = {}
    for label, boxes in states.items():
        out[label] = {action_name(s): worst_regret(s, boxes) for s in actions(boxes, k)}
    return out


@dataclass(frozen=True)
class Relevance:
    """A question's relevance to the top-k decision: its min-max regret over its feasible answers (None: unbounded),
    the spread of the decision's optimal objective across them, and whether it is informational at the tolerance."""

    regret: Fraction | None
    spread: Fraction | None
    informational: bool
    states: tuple[str, ...]
    detail: str


def _spread(c: Mapping[str, Mapping[str, Fraction | None]]) -> Fraction | None:
    best = []
    for row in c.values():
        bounded = [v for v in row.values() if v is not None]
        if not bounded:
            return None
        best.append(min(bounded))
    return max(best) - min(best)


def relevance_of(states: Mapping[str, Box], k: int, tolerance: Fraction, *, enters: bool = True
                 ) -> tuple[Relevance, Regret]:  # fmt: skip
    """The min-max regret of the top-k decision over a question's feasible states, and whether it is within the
    tolerance (A03 §3.3). `enters`: whether the question's answers enter a vertex of the decision's computation; where
    they do not, its states are all the current one and its regret is computed as zero."""
    c = costs(states, k)
    r = minmax_regret(c, epsilon=tolerance)
    informational = r.regret is not None and r.regret <= tolerance
    how = "" if enters else "its answers enter no vertex of the decision's computation, so "
    detail = (f"the top-{k} decision: {how}min-max regret {_show(r.regret)} over {len(states)} feasible state(s), "
              f"{'within' if informational else 'beyond'} the tolerance {tolerance}")  # fmt: skip
    return Relevance(r.regret, _spread(c), informational, tuple(states), detail), r


def _show(x: Fraction | None) -> str:
    if x is None:
        return "unbounded"
    return f"{x.numerator}" if x.denominator == 1 else f"{float(x):,.2f} ({x.numerator}/{x.denominator})"


def _with(graph: Graph, vid: str, value: Any) -> Graph:
    v = graph[vid]
    assert isinstance(v, Source)
    return graph.replace(Source(v.id, v.layer, value, v.records))


def _boxes(case: Case, graph: Graph) -> Box:
    return materiality(case, evaluate(graph))


def question_states(case: Case, res: Resolutions, graph: Graph, q: Asked) -> dict[str, Box]:
    """The decision's states under each feasible answer of a question in play: each substantive answer's binds in place
    of its feasible set (A03 §3.3: not_established is not a state of the world), or for a forecast, each end of its
    configurations' range of probabilities. A question whose answers enter no vertex of the computation leaves every
    state as it is under each answer, so its regret is computed as zero, not assumed."""
    pe = res.estimates.get(q.event)
    if pe is None or pe.interval is None:
        return {}  # with no probability for its event, its exposures have no materiality to compare
    if q.slot == "forecast":
        lo, hi = pe.interval
        pv = f"v:{q.event}.p"
        return {f"P = {_show(p)}": _boxes(case, _with(graph, pv, p)) for p in dict.fromkeys((lo, hi))}
    vid = f"v:q.{q.id}"
    substantive = [a for a in q.question.answers if a.status in ("affirmative", "negative")]
    if vid not in graph:
        current = _boxes(case, graph)
        return {a.value: current for a in substantive}
    return {a.value: _boxes(case, _with(graph, vid, (freeze(dict(a.binds)),))) for a in substantive}


# ---------------------------------------------------------------- decision sufficiency (addendum A02 §3.2)

CandidateSource = Literal["answer_meaning", "definition", "loss_mechanism", "library_template"]


class Candidate(core.Record):
    """One candidate refinement the witness search considered for a question (A02 §3.2: "The probe draws candidates from
    the current definitions, loss mechanisms, known edge cases and template library"): its source, whether it was tried
    (its feasible states pinned and their acceptable top-k sets compared at the tolerance), the states compared, and the
    witness it yielded (a core.RefinementWitness's content hash), or None. Not tried: why it could not be."""

    source: CandidateSource
    tried: bool
    description: str
    states: Sequence[str]
    witness: str | None


class Searched(core.Record):
    """The refinement set searched for one relevant, perturbation-stable question under its shared answer (A02 §3.7:
    "no material decision-sufficiency witness remains in the searched refinement set")."""

    question: core.Ref
    shared_answer: str
    candidates: Sequence[Candidate]


@dataclass(frozen=True)
class Witnessed:
    """A witness (core.RefinementWitness) found by one candidate, and where between its states the acceptable top-k
    sets stop sharing a set with those of the first: between `below` and `above` of the bound value `key`, found by
    bisection, so a sharper question can put its boundary there. `agent`: whether a sharper question can take it (an
    answer's band), or only a human (the event's timing, a proposition's window)."""

    witness: core.RefinementWitness
    key: str
    below: Fraction
    above: Fraction
    agent: bool


BISECTIONS = 24


def _acceptable(case: Case, graph: Graph, k: int, tolerance: Fraction,
                states: Mapping[str, Graph]) -> dict[str, frozenset[str]]:  # fmt: skip
    """Each state's acceptable top-k sets, A_eps (A02 §3.2), at the tolerance."""
    return dict(minmax_regret(costs({n: _boxes(case, g) for n, g in states.items()}, k), epsilon=tolerance).optimal_sets)


def _pinned(case: Case, graph: Graph, k: int, tolerance: Fraction, q: Asked, decision: core.Ref, shared: str,
            points: Sequence[Fraction], pin: Callable[[Fraction], Graph], key: str, show: Callable[[Fraction], str],
            agent: bool) -> tuple[list[str], Witnessed | None]:  # fmt: skip
    """The states at `points` compared; a witness where two share no acceptable set, with its bisected boundary."""
    names = [f"{q.id} with {key} = {show(v)}, every other question at its current state" for v in points]
    sets = _acceptable(case, graph, k, tolerance, {n: pin(v) for n, v in zip(names, points, strict=True)})
    pair = next(((i, j) for i in range(len(points)) for j in range(i + 1, len(points))
                 if sets[names[i]] and sets[names[j]] and not sets[names[i]] & sets[names[j]]), None)  # fmt: skip
    if pair is None:
        return names, None
    i, j = pair
    first = sets[names[i]]
    below, above = points[i], points[j]
    for _ in range(BISECTIONS):
        mid = (below + above) / 2
        here = _acceptable(case, graph, k, tolerance, {"mid": pin(mid)})["mid"]
        below, above = (mid, above) if here & first else (below, mid)
    x = q.exposure or q.event
    w = core.RefinementWitness(
        question=q.question.ref, decision=decision, tolerance=case.decision.tolerance, shared_answer=shared,
        states=[names[i], names[j]], acceptable_actions=[sorted(sets[names[i]]), sorted(sets[names[j]])],
        value_discrepancy=None, nodes=list(exposure_vertices(x).values()) if q.exposure else
        [v for d in case.model.drivers for v in exposure_vertices(f"{d.id}.{q.event}").values()],
    )  # fmt: skip
    return names, Witnessed(w, key, below, above, agent)


def search(case: Case, res: Resolutions, graph: Graph, q: Asked, k: int, tolerance: Fraction,
           decision: core.Ref) -> tuple[Searched, Witnessed | None]:  # fmt: skip
    """A02 §3.2's search for one question: candidate refinements from its shared answer's meaning (the ends of the band
    it binds), its definitions, the loss mechanisms (for a forecast, when within its window the event happens) and the
    library's templates (finer bands of the same value), each tried where it can be, and the first witness found.
    Perturbation stability is never read as sufficiency (A02 §3.1): a stable answer is searched like any other."""
    r = res.resolved[q.id]
    out: list[Candidate] = []
    found: Witnessed | None = None

    def add(source: CandidateSource, description: str, tried: Callable[[], tuple[list[str], Witnessed | None]] | None) -> None:
        nonlocal found
        if tried is None:
            out.append(Candidate(source=source, tried=False, description=description, states=[], witness=None))
            return
        states, w = tried()
        if w is not None:
            case.store.commit("refinement_witness", w.witness.content_hash(), w.witness)
            found = found or w
        out.append(Candidate(source=source, tried=True, description=description, states=states,
                             witness=None if w is None else w.witness.content_hash()))  # fmt: skip

    defs = [*q.question.definitions, *case.state.definitions]
    add("definition", f"definitions {', '.join(defs)}: their alternative readings are proposed by an agent, not "
        "enumerated by code" if defs else "the question and its state use no definitions", None)  # fmt: skip
    if q.slot == "forecast":
        shared = "occurs within its window"
        e = next(ev for ev in case.events if ev.id == q.event)
        w = e.window
        first = max(case.state.as_of, w.start) if w is not None and w.start is not None else None
        if w is None or first is None or w.end is None or w.end <= first:
            add("loss_mechanism", f"when {e.id} happens: its window has no two distinct ends after as_of to compare", None)
        else:
            days = (w.end - first).days
            span = (Fraction(0), Fraction(days))

            def pin_day(d: Fraction) -> Graph:
                at = first + timedelta(days=int(d))
                return _with(graph, f"v:{e.id}.window", w.model_copy(update={"start": at, "end": at}))

            add("loss_mechanism", f"when within its window {e.id} happens, the timing every exposure's loss starts from",
                lambda: _pinned(case, graph, k, tolerance, q, decision, shared, list(span), pin_day, "event day",
                                lambda d: (first + timedelta(days=int(d))).date().isoformat(), False))  # fmt: skip
        return Searched(question=q.question.ref, shared_answer=shared, candidates=out), found
    answer = next(a for a in q.question.answers if a.value == r.application_state)
    key = BINDS.get(q.slot)
    band = dict(answer.binds).get(key) if key else None
    add("loss_mechanism", "the exposure's other loss mechanisms are not finer states of this answer: its band and its "
        "lag are questions of their own (searched where relevant), the event's timing is searched under the event's "
        "forecast, and its payment terms and amounts are G1-approved parameters, not searched", None)  # fmt: skip
    if key is None or not isinstance(band, list | tuple) or len(band) != 2:
        add("answer_meaning", f"{answer.value} binds one value ({dict(answer.binds)}): its meaning holds no finer states", None)
        add("library_template", f"no band to divide: {answer.value} binds one value", None)
        return Searched(question=q.question.ref, shared_answer=answer.value, candidates=out), found
    lo, hi = Fraction(str(band[0])), None if band[1] is None else Fraction(str(band[1]))
    vid = f"v:q.{q.id}"

    def pin(v: Fraction) -> Graph:
        return _with(graph, vid, (freeze({key: (str(v), str(v))}),))

    if hi is None:
        add("answer_meaning", f"{answer.value} is open above: it has no upper end to pin, and its exposure's loss has no "
            "bound (listed apart from the register)", None)  # fmt: skip
        add("library_template", f"{answer.value} is open above: no finer band bounds it", None)
        return Searched(question=q.question.ref, shared_answer=answer.value, candidates=out), found
    if lo < hi:
        add("answer_meaning", f"the ends of {answer.value}: {key} = {_decimal(lo)} and {_decimal(hi)}",
            lambda: _pinned(case, graph, k, tolerance, q, decision, answer.value, [lo, hi], pin, key, _decimal, True))
    else:
        add("answer_meaning", f"{answer.value} binds one point ({key} = {_decimal(lo)})", None)
    cuts = sorted({Fraction(str(x)) for t in library.all_templates().values() for a in t.answers
                   for b in [dict(a.binds).get(key)] if isinstance(b, list | tuple) and len(b) == 2
                   for x in b if x is not None and lo < Fraction(str(x)) < hi})  # fmt: skip
    if cuts:
        add("library_template", f"the library's bands of {key} inside {answer.value}: boundaries at "
            f"{', '.join(_decimal(c) for c in cuts)}",
            lambda: _pinned(case, graph, k, tolerance, q, decision, answer.value, [lo, *cuts, hi], pin, key, _decimal,
                            True))  # fmt: skip
    else:
        add("library_template", f"no library template's band of {key} has a boundary inside {answer.value}", None)
    return Searched(question=q.question.ref, shared_answer=answer.value, candidates=out), found


# ---------------------------------------------------------------- the hedgeability screen (spec §7.1 step 5)


@dataclass(frozen=True)
class Observable:
    key: str
    text: str
    market: str | None  # the listed series' ticker, for a listed market


def event_observable(e: core.Proposition) -> Observable:
    return Observable("event", f"the official record of whether, and when, this happens: {event_text(e)}", None)


def _kalshi_listing(as_of: datetime) -> Sequence[Mapping[str, Any]]:
    from requisite.market.kalshi import Kalshi

    client = Kalshi(snapshot=as_of)
    try:
        return [entry for entry, _ in client.series_list().entries]
    finally:
        client.close()


@dataclass(frozen=True)
class Scouted:
    event: str
    markets: tuple[Observable, ...]
    categories: tuple[str, ...]
    not_listed: tuple[str, ...]
    runs: tuple[str, ...]


def scout(case: Case, e: core.Proposition, listing: Sequence[Mapping[str, Any]], agent: AgentCall) -> Scouted:
    """An agent's search of the read-only listing for markets close to the event (two passes: categories, then series
    within them); every series it names is checked against the listing, and one not listed is stated, never asked."""
    by_category: dict[str, list[Mapping[str, Any]]] = {}
    for s in listing:
        by_category.setdefault(str(s.get("category") or "(none)"), []).append(s)
    for v in by_category.values():
        v.sort(key=lambda s: str(s.get("ticker")))
    event = {"statement": e.text, "window": _window(e.window)}
    first_input = {"event": event, "categories": [
        {"category": c, "series": len(v), "titles": [str(s.get("title")) for s in v[:5]]}
        for c, v in sorted(by_category.items())]}  # fmt: skip
    first = agent("market_scout", first_input, CategoriesDraft, effort="high")
    record_run(case.store, first, first_input, [])
    chosen = tuple(c for c in first.output.categories if c in by_category)
    series = [s for c in chosen for s in by_category[c]]
    runs = [first.run]
    tickers: list[str] = []
    if series:
        second_input = {"event": event, "series": [{"ticker": str(s.get("ticker")), "title": str(s.get("title")),
                                                     "category": str(s.get("category"))} for s in series]}  # fmt: skip
        second = agent("market_scout", second_input, MarketsDraft, effort="high")
        record_run(case.store, second, second_input, [])
        runs.append(second.run)
        tickers = list(dict.fromkeys(second.output.tickers))
    listed = {str(s.get("ticker")): s for s in listing}
    markets = []
    for t in tickers:
        if t in listed:
            s = listed[t]
            sources = ", ".join(str(x.get("name")) for x in s.get("settlement_sources") or []) or "sources not stated"
            markets.append(Observable(f"kalshi.{t}", f"the settlement of the contracts listed on Kalshi as series {t}, "
                                                     f"\"{s.get('title')}\", settled from {sources}", t))  # fmt: skip
    return Scouted(e.id, tuple(markets), chosen, tuple(t for t in tickers if t not in listed), tuple(runs))


@dataclass(frozen=True)
class Screen:
    """The hedgeability screen of each exposure, from the screen's questions as resolved under the application policy
    (I6: an answer not applied is unknown): `hedgeable`, some observable asked about tracks the loss and the business
    cannot influence it; `found_market_tracks`, some listed market the scout found tracks the loss, false where none it
    found does, and so where it found none, which says nothing of markets outside the categories it searched (named in
    `scouted`)."""

    hedgeable: Mapping[str, Kleene]
    found_market_tracks: Mapping[str, Kleene]
    questions: tuple[Asked, ...]
    judgments: Mapping[str, Sequence[core.Judgment]]
    resolved: Mapping[str, core.ResolvedJudgment]
    scouted: Mapping[str, Scouted]


def loss_text(case: Case, d: core.Driver, e: core.Proposition) -> str:
    return (f"the change in {d.name} (a {d.line} line) if this happens: {event_text(e)}; the loss the business cares "
            f"about is {case.model.loss_definition}")  # fmt: skip


def screen(case: Case, judge: Judge, agent: AgentCall, listing: SeriesListing) -> Screen:
    """Step 3: the observable questions for each exposure over the event's own record and the listed markets found."""
    entries = listing(case.state.as_of)
    scouted = {e.id: scout(case, e, entries, agent) for e in case.events}
    qs: list[Asked] = []
    for e in case.events:
        for o in (event_observable(e), *scouted[e.id].markets):
            qs.append(Asked(_q("client_can_influence", {"observable": o.text},
                               f"deep.client_can_influence.{e.id}.{o.key}", case, "influence"), "influence", e.id,
                            observable=o.key))  # fmt: skip
            for d in case.model.drivers:
                qs.append(Asked(_q("observable_tracks_loss", {"loss": loss_text(case, d, e), "observable": o.text},
                                   f"deep.observable_tracks_loss.{d.id}.{e.id}.{o.key}", case, "tracks"), "tracks",
                                e.id, d.id, o.key))  # fmt: skip
    js = judge_questions(case, qs, judge)
    resolved: dict[str, core.ResolvedJudgment] = {}
    for q in qs:
        if len(diagnose.usable(js[q.id])) >= 2:
            r = diagnose.resolve(q.question, case.state_hash, js[q.id])
            case.store.commit("resolved_judgment", r.content_hash(), r)
            resolved[q.id] = r
    by_id = {q.id: q for q in qs}

    def truth(qid: str) -> Kleene:
        return applied_truth(by_id[qid], resolved.get(qid))

    hedgeable: dict[str, Kleene] = {}
    found: dict[str, Kleene] = {}
    for d, e in case.exposures:
        x = f"{d.id}.{e.id}"
        per: list[Kleene] = []
        markets: list[Kleene] = []
        for o in (event_observable(e), *scouted[e.id].markets):
            tracks = truth(f"deep.observable_tracks_loss.{x}.{o.key}")
            per.append(conj([tracks, neg(truth(f"deep.client_can_influence.{e.id}.{o.key}"))]))
            if o.market is not None:
                markets.append(tracks)
        hedgeable[x], found[x] = disj(per), disj(markets)
    return Screen(hedgeable, found, tuple(qs), js, resolved, scouted)


# ---------------------------------------------------------------- the analysis


@dataclass
class Standing:
    """A question's standing for the top-k decision: its resolution, its relevance (None where its event has no
    probability, so its exposures have no materiality to compare), its next action and why."""

    asked: Asked
    resolved: core.ResolvedJudgment | None
    relevance: Relevance | None
    action: str
    reason: str

    @property
    def relevant(self) -> bool:
        """Material to the decision: its regret exceeds the tolerance or is unbounded, or could not be assessed."""
        return self.relevance is None or not self.relevance.informational


@dataclass
class Analysis:
    case: Case
    k: int
    decision: core.Ref
    tolerance: Fraction
    res: Resolutions
    graph: Graph
    values: dict[str, Any]
    standings: dict[str, Standing]
    searched: dict[str, Searched]
    witnesses: dict[str, Witnessed]
    queue: list[review.Entry]
    top_k: tuple[frozenset[str], ...]
    blocked: dict[str, str]
    mechanism: dict[str, Asked]


def register_decision(case: Case, k: int) -> core.Ref:
    """The decision "enters the top k of the register" (spec §7.1), named with its k and its decision spec."""
    if k < 1:
        raise ValueError("the register's top k holds at least one exposure")
    return core.Ref(id=f"register.top_{k}.{case.decision.id}", version=case.decision.version)


def _unbounded_by(res: Resolutions, x: str) -> str:
    """Why an exposure's materiality has no bound: the questions in play for it whose feasible answers leave it open."""
    why = []
    for q in res.in_play():
        if q.exposure == x and q.slot in ("direction", "magnitude"):
            r = res.resolved.get(q.id)
            state = r.application_state if r is not None else "not resolved"
            why.append(f"{q.id} ({state})")
    return ("its loss has no bound: the feasible answers of " + ", ".join(why) + " leave a band open (A04 §4.3: "
            "fetch the evidence or approve an assumption)")  # fmt: skip


def analyse(case: Case, k: int, judge: Judge) -> Analysis:
    """Steps 1-2 and the requisite loop's measurements: resolutions, the graph's values, every question's relevance,
    the refinement search of each relevant perturbation-stable question, the witnesses and the review queue."""
    res = resolve_case(case, judge)
    graph = build_graph(case, res)
    values = evaluate(graph)
    decision = register_decision(case, k)
    tol = Fraction(case.decision.tolerance)
    boxes = materiality(case, values)
    current = costs({"current": boxes}, k)["current"]
    bounded = [v for v in current.values() if v is not None]
    top = tuple(s for s in actions(boxes, k) if bounded and current[action_name(s)] == min(bounded))
    blocked = {x: _unbounded_by(res, x) for x, (lo, hi) in boxes.items() if lo is None or hi is None}
    for d, e in case.exposures:
        if f"{d.id}.{e.id}" not in boxes:
            blocked[f"{d.id}.{e.id}"] = (f"its event {e.id} has no probability: its forecast returned fewer than two "
                                         "distributions")  # fmt: skip
    standings: dict[str, Standing] = {}
    searched: dict[str, Searched] = {}
    witnesses: dict[str, Witnessed] = {}
    mechanism: dict[str, Asked] = {}
    entries: list[review.Entry] = []
    for q in res.in_play():
        if q.slot == "mechanism" and q.exposure is not None:
            mechanism[q.exposure] = q
        r = res.resolved.get(q.id)
        states = question_states(case, res, graph, q)
        rel = relevance_of(states, k, tol, enters=f"v:q.{q.id}" in graph or q.slot == "forecast")[0] if states else None
        action, reason = _next(q, r, rel, res)
        if r is not None and r.primary_diagnosis == "unexplained" and action == "human":
            # spec §2.1: unexplained goes to "Human review; save as a fixture": the question as asked, keyed by the
            # resolution that holds its judgments and every probe's evidence (A01 §2.4).
            case.store.commit("question_fixture", f"{q.id}|{r.content_hash()}", q.question)
            reason += f" (saved as question_fixture {q.id}|{r.content_hash()[:12]})"
        standing = standings[q.id] = Standing(q, r, rel, action, reason)
        # The event's timing within its window is a distinction of the event itself, which its forecast's probability
        # range says nothing about: it is searched for every event with a probability, whatever the forecast's own
        # relevance and stability. An exposure's question is searched where it is relevant and perturbation-stable with
        # an applied answer; an informational one is not refined further (spec §2.2).
        timing = q.slot == "forecast" and q.event in res.estimates
        if not standing.relevant and not timing:
            continue
        if timing or (r is not None and r.stable and r.application_state != "review"):
            searched[q.id], found = search(case, res, graph, q, k, tol, decision)
            if found is not None:
                witnesses[q.id] = found
                why = "perturbation-stable, but its shared answer covers states the decision tells apart (A02 §3.2)"
                if not found.agent:
                    standing.action, standing.reason = "human", f"{why}: when within its window the event happens; a human refines the event"
                elif _attempts(res, q.id, "sharpen") < MAX_ATTEMPTS:
                    standing.action, standing.reason = "sharpen", why
                else:
                    standing.action, standing.reason = "human", (
                        f"{why}; {MAX_ATTEMPTS} sharper questions were proposed and none was admitted: a human decides")
        if standing.relevant and (r is None or not r.stable or _unknown(q, r)):
            spread = rel.spread if rel is not None else None
            entries.append(review.Entry(q.question.ref, standing.reason, r.s_arg if r else 0.0, r.s_prob if r else 0.0,
                                        spread, standing.action))  # fmt: skip
    return Analysis(case, k, decision, tol, res, graph, values, standings, searched, witnesses, review.order(entries),
                    top, blocked, mechanism)  # fmt: skip


def _unknown(q: Asked, r: core.ResolvedJudgment) -> bool:
    """A perturbation-stable answer that establishes nothing: the judges agree the evidence does not settle it, which is
    no resolution for business use (A02 §3.1). A forecast applies no answer by policy, and is not one."""
    return r.stable and r.evidence_status == "not_established" and q.slot != "forecast"


def _attempts(res: Resolutions, qid: str, kind: str) -> int:
    return len({s.attempt for s in res.asked.values() if s.parent == qid and s.attempt and s.attempt.startswith(kind)})


def _next(q: Asked, r: core.ResolvedJudgment | None, rel: Relevance | None, res: Resolutions) -> tuple[str, str]:
    """The prescribed next action for a question in play (spec §2.1's table, a primary routing, not a proven cause:
    A01 §2.4; spec §2.2; A02 §3.1, §3.7)."""
    if rel is None:
        return "human", "its relevance could not be assessed: its event has no probability"
    if rel.informational:
        return "none", f"informational for {rel.detail} (A02 §3.4: non-material to this decision under these answers only)"
    if r is None:
        return "human", "too few configurations returned a distribution to resolve it"
    if r.stable:
        if _unknown(q, r):
            return "human", ("perturbation-stable and relevant, but the judges agree the evidence does not establish it: "
                             "fetch the evidence or approve an assumption (A02 §3.1; A04 §4.3)")  # fmt: skip
        return "none", f"perturbation-stable (the configurations agree; not a check that the answer is right); relevant: {rel.detail}"
    dx = r.primary_diagnosis
    if dx == "imprecise_target":
        if _attempts(res, q.id, "refine") < MAX_ATTEMPTS:
            return "refine", f"routed as imprecise_target: {diagnose.ACTIONS[dx][1]}"
        return "human", f"imprecise_target after {MAX_ATTEMPTS} refinements that were not admitted: human review"
    if dx == "contested":
        return "human", f"routed as contested: {diagnose.ACTIONS[dx][1]}"
    if dx == "compound":
        return "none", "compound: replaced by its sub-questions and their composition"
    if _attempts(res, q.id, "split") == 0 and q.slot != "forecast":
        return "split", "unstable along no single axis of the base run: the split probe follows (spec §2.1)"
    return "human", f"routed as unexplained: {diagnose.ACTIONS['unexplained'][1]}"


# ---------------------------------------------------------------- acting: refinements and splits through the gates


@dataclass
class _Pending:
    parent: Asked
    kind: str
    witness: Witnessed | None
    analysis: Analysis


class Gates:
    """The question gates a split or refinement passes (core.SPLIT_GATES), registered on the case's store."""

    def __init__(self, store: Store, judge: Judge):
        self.judge = judge
        self.pending: dict[str, _Pending] = {}
        self.measured: dict[str, list[core.Judgment]] = {}
        store.register_gate("question", "stability", self.stability)
        store.register_gate("question", "relevance", self.relevance)

    def stability(self, store: Store, proposal: core.Proposal) -> core.GateResult:
        p = self.pending.get(proposal.id)
        if p is None:
            return core.GateResult(gate="stability", passed=False, detail="no requisite loop proposed this question")
        q = _payload(proposal, core.AtomicQuestion)
        asked = Asked(q, p.parent.slot, p.parent.event, p.parent.driver, p.parent.observable, p.parent.id)
        js = judge_questions(p.analysis.case, [asked], self.judge)[q.id]
        self.measured[q.id] = js
        usable = diagnose.usable(js)
        if len(usable) < 2:
            return core.GateResult(gate="stability", passed=False, detail=f"{len(usable)} distribution(s): too few")
        sc = diagnose.scores([dict(j.distribution) for j in usable if j.distribution is not None])
        ok = diagnose.is_stable(sc)
        return core.GateResult(gate="stability", passed=ok, detail=f"S_arg {sc.s_arg:.3f}, S_prob {sc.s_prob:.3f} over "
                               f"{len(usable)} configurations ({'stable' if ok else 'not stable'} at spec §2.1's "
                               "thresholds)")  # fmt: skip

    def relevance(self, store: Store, proposal: core.Proposal) -> core.GateResult:
        p = self.pending[proposal.id]
        q = _payload(proposal, core.AtomicQuestion)
        r = p.analysis.res.resolved.get(p.parent.id)
        if p.kind in ("refine", "split"):
            unstable = r is not None and not r.stable
            return core.GateResult(gate="relevance", passed=unstable, detail=(
                f"needed for stability: {p.parent.id} is unstable ({r.primary_diagnosis}) and relevant to "
                f"{p.analysis.decision.id}" if unstable and r else f"{p.parent.id} is stable: a {p.kind} is not needed"))  # fmt: skip
        assert p.witness is not None
        return self._sharpens(p, q)

    def _sharpens(self, p: _Pending, q: core.AtomicQuestion) -> core.GateResult:
        """A sharper question is admitted where it tells the witness's states apart, each answer within the parent's
        applied answer, together covering it, and where its own answers move the decision beyond the tolerance (spec
        §2.2: "A split is admitted only if it raises relevance")."""
        an = p.analysis
        key = BINDS[p.parent.slot]
        r = an.res.resolved[p.parent.id]
        parent_band = next(dict(a.binds)[key] for a in p.parent.question.answers if a.value == r.application_state)
        plo, phi = Fraction(str(parent_band[0])), Fraction(str(parent_band[1]))
        bands = []
        for a in q.answers:
            if a.status in ("affirmative", "negative"):
                lo, hi = dict(a.binds)[key]
                bands.append((Fraction(str(lo)), None if hi is None else Fraction(str(hi))))
        problems = []
        if any(lo < plo or hi is None or hi > phi for lo, hi in bands):
            problems.append(f"an answer lies outside {r.application_state} [{plo}, {phi}]")
        covered = sorted(b for b in bands if b[1] is not None)
        reach = plo
        for lo, hi in covered:
            if lo > reach:
                break
            reach = max(reach, hi) if hi is not None else reach
        if reach < phi:
            problems.append(f"its answers leave part of [{plo}, {phi}] uncovered")
        if any(lo <= plo and hi is not None and hi >= phi for lo, hi in bands):
            problems.append("one answer covers both of the witness's states")
        states = {a.value: _boxes(an.case, _with(an.graph, f"v:q.{p.parent.id}", (freeze(dict(a.binds)),)))
                  for a in q.answers if a.status in ("affirmative", "negative")}  # fmt: skip
        rel, _ = relevance_of(states, an.k, an.tolerance)
        if rel.informational:
            problems.append(f"its answers do not move the decision ({rel.detail})")
        return core.GateResult(gate="relevance", passed=not problems, detail="; ".join(problems) or (
            f"tells the witness's states apart and raises relevance: {rel.detail}"))  # fmt: skip


def _draft_input(an: Analysis, q: Asked, kind: str, witness: Witnessed | None) -> dict[str, Any]:
    r = an.res.resolved.get(q.id)
    earlier = [{"question": s.question.target, "answers": [a.value for a in s.question.answers]}
               for s in an.res.asked.values() if s.parent == q.id]  # fmt: skip
    return {
        "task": kind,
        "question": {"target": q.question.target, "task": q.question.task, "answers": [
            {"value": a.value, "meaning": a.meaning, "status": a.status, "binds": _plain(a.binds)}
            for a in q.question.answers]},
        "evidence": an.case.rendered(TEXT).text,
        "why": an.standings[q.id].reason,
        "measurements": [{"probe": p.probe, "detail": p.detail, "scores": dict(p.scores)} for p in r.probes] if r else [],
        "witness": None if witness is None else {
            "shared_answer": witness.witness.shared_answer, "states": list(witness.witness.states),
            "acceptable_top_k": [list(s) for s in witness.witness.acceptable_actions],
            "decision_changes_between": {witness.key: [_decimal(witness.below), _decimal(witness.above)]}},
        "earlier_attempts": earlier,
    }  # fmt: skip


def _decimal(x: Fraction, places: int = 6) -> str:
    """A bound as a decimal string to `places` places, for an agent to read (the exact value is the witness's)."""
    return f"{float(x):.{places}f}".rstrip("0").rstrip(".")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def _drafted(parent: Asked, kind: str, number: int, draft: QuestionsDraft) -> tuple[list[core.AtomicQuestion],
                                                                                  list[core.ParaphraseSet], list[str]]:  # fmt: skip
    """The drafted questions as the case's records, or the reasons they are not (an answer the definition refuses is
    refused here, with its reasons)."""
    problems: list[str] = []
    n = len(draft.questions)
    if kind in ("refine", "sharpen") and n != 1:
        problems.append(f"a {kind} is one question, not {n}")
    if kind == "split" and n < 2:
        problems.append(f"a split is two or more sub-questions, not {n}")
    if parent.question.task == "forecast" and kind == "split":
        problems.append("a forecast is not split into a decision list: its composition is an event tree (spec §5)")
    qs, psets = [], []
    for i, d in enumerate(draft.questions, 1):
        qid = f"{parent.id}/{kind}{number}.{i}"
        try:
            answers = [core.AnswerValue(value=a.value, meaning=a.meaning, status=a.status, binds=dict(a.binds))
                       for a in d.answers]  # fmt: skip
            q = core.AtomicQuestion(
                id=qid, version=1, task=parent.question.task, target=d.target, window=parent.question.window,
                answers=answers, paraphrase_set=f"ps.{qid}", reaches=list(parent.question.reaches),
                replaces=[parent.question.ref],
            )  # fmt: skip
            pset = core.ParaphraseSet(
                id=f"ps.{qid}", version=1, template=q.ref, slots=[], answers=[a.value for a in answers],
                paraphrases=[library.original(q), *(core.Paraphrase(id=p.id, target=p.target, meanings=p.meanings)
                                                    for p in d.paraphrases)],
            )  # fmt: skip
        except (ValidationError, ValueError) as e:
            problems.append(f"question {i}: {e}")
            continue
        qs.append(q)
        psets.append(pset)
    if not problems:
        if kind == "refine":
            same = [(a.value, binds_key(a.binds)) for a in parent.question.answers]
            if [(a.value, binds_key(a.binds)) for a in qs[0].answers] != same:
                problems.append("a refinement keeps the question's answer values and what each binds")
        elif kind == "split":
            problems += diagnose.composition_problems(parent.question, qs)
    return qs, psets, problems


def act(an: Analysis, q: Asked, kind: str, gates: Gates, agent: AgentCall) -> list[core.Proposal]:
    """One agent action on one question: the question_refiner's draft, submitted as proposals through the store's
    gates (spec I1), a paraphrase set before each question it words."""
    witness = an.witnesses.get(q.id) if kind == "sharpen" else None
    number = 1 + len({s.attempt for s in an.res.asked.values() if s.parent == q.id})
    input = _draft_input(an, q, kind, witness)
    result = agent("question_refiner", input, QuestionsDraft, effort="high")
    qs, psets, problems = _drafted(q, kind, number, result.output)
    submitted: list[core.Proposal] = []
    if problems:
        record_run(an.case.store, result, input, [])
        an.standings[q.id].reason += f"; the refiner's {kind} was refused: {'; '.join(problems)}"
        return submitted
    for sub, pset in zip(qs, psets, strict=True):
        ps_payload = pset.model_dump(mode="json")
        submitted.append(an.case.store.submit_proposal(core.Proposal(
            id=f"paraphrase_set-{core.sha256(core.canonical_json(ps_payload))[:16]}", case=an.case.store.case,
            kind="paraphrase_set", payload=ps_payload, author=result.run[:16],
            rationale=f"the frozen paraphrases of {sub.id}", status="proposed")))  # fmt: skip
        payload = sub.model_dump(mode="json")
        pid = f"question-{core.sha256(core.canonical_json(payload))[:16]}"
        gates.pending[pid] = _Pending(q, kind, witness, an)
        submitted.append(an.case.store.submit_proposal(core.Proposal(
            id=pid, case=an.case.store.case, kind="question", payload=payload, author=result.run[:16],
            rationale=result.output.rationale, status="proposed")))  # fmt: skip
    record_run(an.case.store, result, input, [p.id for p in submitted])
    return submitted


# ---------------------------------------------------------------- the register (spec §7.5)


def _direction(fs: Sequence[Mapping[str, Any]]) -> str:
    values = {b.get("direction") for b in fs}
    return str(values.pop()) if len(values) == 1 else "not_established"


def _band(res: Resolutions, base: Asked, fs: Sequence[Mapping[str, Any]]) -> str:
    """The magnitude band as the structure answers it: the applied answer's value where one question answers it, and
    otherwise the interval its feasible answers bind, said to be not established where a question in play applies no
    answer."""
    leaves = res.effective(base.id)
    applied = [res.resolved[q.id].application_state if q.id in res.resolved else "review" for q in leaves]
    if len(leaves) == 1 and applied[0] != "review":
        if leaves[0].question.template_id is not None:
            return applied[0]
        answer = next(a for a in leaves[0].question.answers if a.value == applied[0])
        lo, hi = dict(answer.binds)["magnitude"]
        upper = "no upper bound" if hi is None else _decimal(Fraction(str(hi)))
        return f"{applied[0]}: between {_decimal(Fraction(str(lo)))} and {upper} of the driver's amount"
    lows = [Fraction(str(b["magnitude"][0])) for b in fs]
    highs = [b["magnitude"][1] for b in fs]
    hi = "no upper bound" if any(h is None for h in highs) else _decimal(max(Fraction(str(h)) for h in highs))
    text = f"between {_decimal(min(lows))} and {hi} of the driver's amount, as {len(leaves)} questions compose it"
    return text if "review" not in applied else f"not established: {text}"


def _mechanism(q: Asked | None, res: Resolutions) -> str:
    """The mechanism the application policy applies (A04 §4.1), not_established where it applies none."""
    r = None if q is None else res.resolved.get(q.id)
    if q is None or r is None or r.application_state == "review":
        return "not_established"
    answer = next(a for a in q.question.answers if a.value == r.application_state)
    return str(dict(answer.binds).get("mechanism", "not_established"))


def diagnostic(an: Analysis, s: Standing) -> core.QuestionDiagnostic | None:
    """A tested question's stability and relevance for the register (core.QuestionDiagnostic), where its regret is
    bounded; an unbounded regret is in the run's standings."""
    if s.resolved is None or s.relevance is None or s.relevance.regret is None:
        return None
    c = an.case
    return core.QuestionDiagnostic(
        question=s.asked.question.ref, resolved=s.resolved.content_hash(), s_arg=s.resolved.s_arg,
        s_prob=s.resolved.s_prob, primary_diagnosis=s.resolved.primary_diagnosis, decision=an.decision,
        tolerance=c.decision.tolerance,
        assumptions=[f"{c.model.id}@v{c.model.version}", f"{c.decision.id}@v{c.decision.version}",
                     *sorted(f"{pe.id}" for pe in an.res.estimates.values()),
                     "every other question at its resolved state"],
        regret=s.relevance.regret, informational=s.relevance.informational,
    )  # fmt: skip


@dataclass(frozen=True)
class Register:
    rows: tuple[core.RegisterRow, ...]
    blocked: Mapping[str, str]
    top_k: tuple[frozenset[str], ...]
    screen: Screen
    runtime: Runtime
    diff: Diff


def _interval(v: tuple[Fraction | None, Fraction | None]) -> tuple[Fraction, Fraction] | None:
    lo, hi = v
    return None if lo is None or hi is None else (lo, hi)


def register(an: Analysis, scr: Screen, runtime: Runtime | None = None) -> Register:
    """Step 4: the register's rows, every number read from the runtime's recompute of the case's graph, every row naming
    the claims, judgments and vertex ids its numbers rest on; each instance vertex's receipt is committed under its id."""
    case, res = an.case, an.res
    rt = runtime or Runtime(case.store)
    diff = rt.recompute(an.graph)
    for vertex in an.graph:
        if isinstance(vertex, Instance):
            receipt = rt.state(vertex.id).receipt
            assert receipt is not None
            committed = VertexReceipt(vertex=vertex.id, layer=vertex.layer, receipt=receipt)
            case.store.commit("vertex", f"{vertex.id}|{receipt.lineage_hash}", committed)
    by_slot = {(b.exposure, b.slot): b for b in res.base if b.exposure is not None}
    rows: list[core.RegisterRow] = []
    blocked = dict(an.blocked)
    for d, e in case.exposures:
        x = f"{d.id}.{e.id}"
        v = exposure_vertices(x)
        if x in blocked:
            continue
        cash = rt.value(v["cash"])
        exp, tail = _interval(rt.value(v["expected"])), _interval(rt.value(v["tail"]))
        if exp is None or tail is None or any(lo is None or hi is None for _, lo, hi in cash):
            blocked[x] = _unbounded_by(res, x)
            continue
        leaves = [q for s in ("direction", "magnitude", "lag", "conditional") for q in res.effective(by_slot[(x, s)].id)]
        leaves.append(res.forecast_leaf(e.id))
        diagnostics = [dg for q in leaves if (dg := diagnostic(an, an.standings[q.id])) is not None]
        if not diagnostics:
            blocked[x] = "none of its questions has a bounded regret for this decision"
            continue
        asked = [*leaves, *(q for q in scr.questions if q.exposure == x or (q.slot == "influence" and q.event == e.id))]
        if (mq := an.mechanism.get(x)) is not None:
            asked.append(mq)
        judgments = sorted({j.content_hash() for q in asked
                            for j in [*res.judgments.get(q.id, ()), *scr.judgments.get(q.id, ())]})  # fmt: skip
        params = [r for vid in (f"v:{d.id}.quantity", f"v:{d.id}.price", "v:cash", "v:credit", "v:seasonality",
                                "v:payable_days" if d.line == "cost" else "v:receivable_days")
                  for r in _vertex_records(an.graph, vid)]  # fmt: skip
        # Every judgment read the case's whole evidence state, and each sourced parameter cites its own claims.
        claims = sorted(set(case.state.observed) | {c for p in params if isinstance(p, core.Parameter) for c in p.claims})
        direction = rt.value(_effect_input(an.graph, v["effect"], "direction"))
        magnitude = rt.value(_effect_input(an.graph, v["effect"], "magnitude"))
        rows.append(core.RegisterRow(
            entity=case.model.entity, driver=d.id, event=e.id, direction=_direction(direction),  # type: ignore[arg-type]
            mechanism=_mechanism(an.mechanism.get(x), res),  # type: ignore[arg-type]
            magnitude_band=_band(res, by_slot[(x, "magnitude")], magnitude),
            cash_impact=[core.DatedImpact(month=m, low=lo, high=hi) for m, lo, hi in cash],
            baseline=BASELINE, probability=res.estimates[e.id].id, expected_impact=exp, tail_impact=tail,
            breach=rt.value(v["breach"]), hedgeable=scr.hedgeable[x].value, diagnostics=diagnostics,
            claims=claims, judgments=judgments,
            nodes=[*v.values(), f"v:{d.id}.amounts", "v:days", *_compositions(an.graph, v["effect"], v["start"])],
        ))  # fmt: skip
    for row in rows:
        case.store.commit("register_row", f"{row.driver}.{row.event}|{row.content_hash()}", row)
    return Register(tuple(rows), blocked, an.top_k, scr, rt, diff)


def _effect_input(graph: Graph, effect: str, name: str) -> str:
    v = graph[effect]
    assert isinstance(v, Instance)
    return v.inputs[name]


def _compositions(graph: Graph, *vids: str) -> list[str]:
    """The composition vertices a split adds under these instances."""
    out: list[str] = []
    seen: set[str] = set()
    stack = list(vids)
    while stack:
        vid = stack.pop()
        v = graph[vid]
        if isinstance(v, Instance):
            for src in v.inputs.values():
                if src.startswith("v:split.") and src not in seen:
                    seen.add(src)
                    out.append(src)
                stack.append(src)
    return sorted(out)


def _vertex_records(graph: Graph, vid: str) -> tuple[core.Record, ...]:
    v = graph[vid] if vid in graph else None
    return v.records if isinstance(v, Source) else ()


# ---------------------------------------------------------------- the loop's record


class VertexReceipt(core.Record):
    """A computed vertex of a case's graph as committed: its id (decisions, "Computation": vertex ids are "v:<name>"),
    its layer and its derivation receipt, so a register row's vertex ids name records the store holds (WP11a)."""

    vertex: str
    layer: Literal["evidence", "proposition", "computation", "contract"]
    receipt: core.DerivationReceipt


class QuestionStanding(core.Record):
    """One question's standing at the end of a requisite run. s_arg, s_prob and primary_diagnosis are its resolution's:
    perturbation diagnostics and a routing, not a check that its answer is right (A01 §2.2, §2.4). regret is its
    computed min-max regret for the run's decision (None with regret_unbounded where it has no bound, None alone where
    it could not be computed); informational says only that regret is within the tolerance (A02 §3.4). feeds says
    whether its answers enter a vertex of the decision's computation."""

    question: core.Ref
    slot: str
    exposure: str | None
    event: str
    feeds: bool
    resolved: str | None
    s_arg: float | None
    s_prob: float | None
    primary_diagnosis: Diagnosis | None
    application_state: str | None
    regret: core.Rational | None
    regret_unbounded: bool
    informational: bool
    action: str
    reason: str
    replaced_by: Sequence[core.Ref] = ()


class GateStanding(core.Record):
    """Whether a human gate's approval covers what it must, as it now stands (addendum A11 §11.2: an approval approves
    one lineage). G1: the entity model's and the decision spec's proposals, their lineage the hash of their payload. G2:
    a contested relevant question (spec §2.4), its lineage the content hash of its resolution before approval."""

    gate: Literal["G1", "G2"]
    subject: str
    lineage: str
    approval: str | None
    covers: bool


StopName = Literal["perturbation_failures", "blocking_unknowns", "decision_sufficiency", "approvals", "budget_spent"]


class StopCondition(core.Record):
    """One of A02 §3.7's stopping conditions and whether it held when the run stopped, with the questions it names.
    perturbation_failures: every relevant question whose configurations disagree is resolved (replaced by an admitted
    refinement or split) or escalated to a human. blocking_unknowns: every relevant question whose answer is not applied
    is identified as such (held by construction; the questions are listed). decision_sufficiency: no witness found in
    the searched refinement set is left without an admitted refinement or an escalation. approvals: every gate standing
    covers its current lineage. budget_spent: the run stopped because its budget was spent."""

    condition: StopName
    held: bool
    detail: str
    questions: Sequence[core.Ref] = ()


class RequisiteRun(core.Record):
    """A requisite run's frozen result (spec §2.3; addendum A02 §3.7): the decision it ran for, at what tolerance, which
    stopping conditions held, the refinement set it searched, the gates, and the questions. local_requisite: the
    perturbation failures are resolved or escalated, the blocking unknowns are listed, no witness in the searched set is
    left unaddressed, and every approval covers its lineage, a designation bounded by these records (A02 §3.7: "A
    local_requisite designation is bounded by these records"). awaiting_approval: all but the approvals hold. blocked:
    the budget was spent first, a reproducible draft and never an approved conclusion."""

    case: str
    decision: core.Ref
    k: int
    tolerance: core.Money
    entity_model: core.Ref
    decision_spec: core.Ref
    status: Literal["local_requisite", "awaiting_approval", "blocked"]
    approved: Literal[False] = False
    conditions: Sequence[StopCondition]
    iterations: int
    budget: int
    spent: int
    questions: Sequence[QuestionStanding]
    queue: Sequence[core.Ref]
    escalated: Sequence[core.Ref]
    unknowns: Sequence[core.Ref]
    searched: Sequence[Searched]
    gates: Sequence[GateStanding]
    outstanding_gates: Sequence[str]
    witnesses: Sequence[str]
    proposals: Sequence[str]
    top_k: Sequence[Sequence[str]]
    rows: Sequence[str]
    blocked_exposures: Mapping[str, str]
    domain: str


def _standing(an: Analysis, s: Standing) -> QuestionStanding:
    r, rel = s.resolved, s.relevance
    return QuestionStanding(
        question=s.asked.question.ref, slot=s.asked.slot, exposure=s.asked.exposure, event=s.asked.event,
        feeds=s.asked.feeds or s.asked.slot == "forecast", resolved=r.content_hash() if r else None,
        s_arg=r.s_arg if r else None, s_prob=r.s_prob if r else None,
        primary_diagnosis=r.primary_diagnosis if r else None, application_state=r.application_state if r else None,
        regret=rel.regret if rel else None, regret_unbounded=rel is not None and rel.regret is None,
        informational=rel is not None and rel.informational, action=s.action, reason=s.reason,
        replaced_by=[a.question.ref for a in an.res.replaced_by.get(s.asked.id, [])],
    )  # fmt: skip


def _replaced(an: Analysis) -> list[QuestionStanding]:
    """The questions the structure replaced, still inspectable with their own resolution."""
    out = []
    for qid, subs in an.res.replaced_by.items():
        q = an.res.asked[qid]
        r = an.res.resolved.get(qid)
        out.append(QuestionStanding(
            question=q.question.ref, slot=q.slot, exposure=q.exposure, event=q.event, feeds=q.feeds,
            resolved=r.content_hash() if r else None, s_arg=r.s_arg if r else None, s_prob=r.s_prob if r else None,
            primary_diagnosis=r.primary_diagnosis if r else None, application_state=r.application_state if r else None,
            regret=None, regret_unbounded=False, informational=False, action="replaced",
            reason=f"replaced by {', '.join(s.id for s in subs)}", replaced_by=[s.question.ref for s in subs],
        ))  # fmt: skip
    return out


def gate_standings(an: Analysis) -> list[GateStanding]:
    """G1 on the case's entity model and decision spec, and G2 on every contested relevant question in play, each as
    its approvals now stand."""
    store, case = an.case.store, an.case
    approvals = store.all("approval", core.Approval)
    out = []
    for pid in (case.model_proposal, case.decision_proposal):
        p = store.proposal(pid)
        assert p is not None
        lineage = core.sha256(core.canonical_json(dict(p.payload)))
        out.append(_gate(approvals, "G1", pid, lineage))
    for s in an.standings.values():
        if s.relevant and s.resolved is not None and s.resolved.primary_diagnosis == "contested":
            out.append(_gate(approvals, "G2", s.asked.id, contested_lineage(s.resolved)))
    return out


def _gate(approvals: Sequence[core.Approval], gate: Literal["G1", "G2"], subject: str, lineage: str) -> GateStanding:
    mine = [a for a in approvals if a.gate == gate and a.subject.id == subject]
    covering = [a for a in mine if a.lineage_hash == lineage and a.decision != "rejected"]
    shown = covering[-1] if covering else (mine[-1] if mine else None)
    return GateStanding(gate=gate, subject=subject, lineage=lineage, approval=shown.id if shown else None,
                        covers=bool(covering))  # fmt: skip


def contested_lineage(r: core.ResolvedJudgment) -> str:
    """What a G2 approval of a contested question approves: its resolution as it stands, without approvals."""
    return r.model_copy(update={"approvals": ()}).content_hash()


@dataclass(frozen=True)
class Outcome:
    run: RequisiteRun
    analysis: Analysis
    register: Register


def loop(store: Store, k: int, *, budget: int = DEFAULT_BUDGET, judge: JudgeCall | None = None,
         agent: AgentCall | None = None, listing: SeriesListing | None = None) -> Outcome:  # fmt: skip
    """Step 5: the requisite loop over steps 1-3 (spec §2.3), stopping by addendum A02 §3.7, committed as a
    RequisiteRun. The base run (every question of the structure the case holds, judged and resolved, and the screen) is
    the case's; `budget` caps the calls the loop makes beyond it, judge and agent alike (spec §6.6). A run picks up the
    structure the case holds, so a run that stopped blocked is continued by running again with a larger budget."""
    case = open_deep(store)
    spend = Spend()
    calls, agents = Judge(judge or providers.judge, spend), Agents(agent or runner.run, spend)
    an = analyse(case, k, calls)
    scr = screen(case, calls, agents, listing or _kalshi_listing)
    start = spend.made
    spend.limit = start + budget
    gates = Gates(store, calls)
    proposals: list[str] = []
    searched: dict[str, Searched] = dict(an.searched)
    iterations, spent = 1, False
    tried: set[tuple[str, str, int]] = set()
    while True:
        todo = [(s.asked, s.action) for s in an.standings.values() if s.action in AGENT_ACTIONS]
        todo = [(q, a) for q, a in todo if (q.id, a, _attempts(an.res, q.id, a)) not in tried]
        if not todo:
            break
        try:
            for q, a in todo:
                tried.add((q.id, a, _attempts(an.res, q.id, a)))
                proposals += [p.id for p in act(an, q, a, gates, agents)]
            iterations += 1
            an = analyse(case, k, calls)
            searched.update(an.searched)
        except BudgetSpent:
            spent = True
            spend.limit = spend.made  # nothing more is asked: what is judged so far is replayed from the run's cache
            an = analyse(case, k, calls)
            searched.update(an.searched)
            break
    reg = register(an, scr)
    if not spent:
        for s in an.standings.values():
            if s.action in AGENT_ACTIONS:  # an agent's attempt was refused and is not retried in this run
                s.action, s.reason = "human", f"{s.reason}; the agent's {s.action} was not admitted, so a human decides"
    run = _record(an, reg, budget, spend.made - start, iterations, spent, proposals, searched)
    store.commit("requisite_run", f"{run.decision.id}|{run.content_hash()}", run)
    return Outcome(run, an, reg)


AGENT_ACTIONS = ("refine", "split", "sharpen")


def _record(an: Analysis, reg: Register, budget: int, used: int, iterations: int, spent: bool, proposals: Sequence[str],
            searched: Mapping[str, Searched]) -> RequisiteRun:  # fmt: skip
    """The run's record: A02 §3.7's conditions and which held, from the analysis it stopped at."""
    case = an.case
    standings = list(an.standings.values())
    relevant = [s for s in standings if s.relevant]
    failures = [s for s in relevant if s.resolved is None or not s.resolved.stable]
    pending = [s for s in standings if s.action in AGENT_ACTIONS]
    escalated = [s for s in standings if s.action == "human"]
    unknowns = [s for s in relevant if s.asked.slot != "forecast" and (s.resolved is None or s.resolved.application_state
                                                                       == "review" or _unknown(s.asked, s.resolved))]  # fmt: skip
    found = {qid: w for qid, w in an.witnesses.items()}
    open_witnesses = [an.standings[qid] for qid in found if an.standings[qid].action in AGENT_ACTIONS]
    gate_list = gate_standings(an)
    c1 = not [s for s in failures if s.action in AGENT_ACTIONS]
    c3 = not open_witnesses
    c4 = all(g.covers for g in gate_list)
    tried = sum(1 for x in searched.values() for c in x.candidates if c.tried)
    conditions = [
        core_cond("perturbation_failures", c1, f"{len(failures)} relevant question(s) whose configurations disagree: "
                  f"{len([s for s in failures if s.action == 'human'])} escalated, "
                  f"{len([s for s in failures if s.action in AGENT_ACTIONS])} with an agent's action still to take; "
                  f"{len([q for q in an.res.replaced_by if q in an.res.resolved and not an.res.resolved[q].stable])} "
                  "other(s) replaced by an admitted refinement or split", failures),
        core_cond("blocking_unknowns", True, f"{len(unknowns)} relevant question(s) apply no established answer, each "
                  "listed and never read as agreement (A02 §3.1)", unknowns),
        core_cond("decision_sufficiency", c3, f"{len(searched)} question(s) searched, {tried} candidate refinement(s) "
                  f"tried, {len(found)} witness(es) found in the final structure, {len(open_witnesses)} left without an "
                  "admitted refinement or an escalation", open_witnesses),
        core_cond("approvals", c4, "; ".join(f"{g.gate} on {g.subject}: {'covers' if g.covers else 'missing'} lineage "
                                             f"{g.lineage[:12]}" for g in gate_list), ()),
        core_cond("budget_spent", spent, f"{used} of {budget} calls beyond the base run", pending if spent else ()),
    ]  # fmt: skip
    status = "blocked" if spent and not (c1 and c3) else ("local_requisite" if c1 and c3 and c4 else "awaiting_approval")
    outstanding = [f"{g.gate} on {g.subject}: no approval covers its current lineage" for g in gate_list if not g.covers]
    outstanding += [f"human review of {s.asked.id}: {s.reason}" for s in escalated
                    if not (s.resolved is not None and s.resolved.primary_diagnosis == "contested")]  # fmt: skip
    return RequisiteRun(
        case=case.store.case, decision=an.decision, k=an.k, tolerance=case.decision.tolerance,
        entity_model=core.Ref(id=case.model.id, version=case.model.version),
        decision_spec=core.Ref(id=case.decision.id, version=case.decision.version), status=status,  # type: ignore[arg-type]
        conditions=conditions, iterations=iterations, budget=budget, spent=used,
        questions=[_standing(an, s) for s in standings] + _replaced(an), queue=[e.question for e in an.queue],
        escalated=[s.asked.question.ref for s in escalated], unknowns=[s.asked.question.ref for s in unknowns],
        searched=sorted(searched.values(), key=lambda x: x.question.id), gates=gate_list,
        outstanding_gates=outstanding, witnesses=sorted(w.witness.content_hash() for w in found.values()),
        proposals=list(proposals), top_k=[sorted(s) for s in an.top_k], rows=[r.content_hash() for r in reg.rows],
        blocked_exposures=dict(reg.blocked),
        domain=(f"{len(an.res.asked)} questions over {len(case.exposures)} driver-event pair(s) and {len(case.events)} "
                f"event(s); relevance over each question's substantive answers (a forecast's over its configurations' "
                f"probabilities) at tolerance {case.decision.tolerance}; refinement candidates from answer meanings, "
                f"definitions, loss mechanisms and the library's templates for each relevant perturbation-stable "
                f"question; refinements and splits proposed by the question_refiner agent, at most {MAX_ATTEMPTS} per "
                "question and action"),
    )  # fmt: skip


def core_cond(name: StopName, held: bool, detail: str, standings: Sequence[Standing]) -> StopCondition:
    return StopCondition(condition=name, held=held, detail=detail, questions=[s.asked.question.ref for s in standings])


# ---------------------------------------------------------------- events


def proposition_tests(store: Store, proposal: core.Proposal) -> core.GateResult:
    """A proposition's tests gate (spec §5: the proposition layer's validator is SAT consistency and acyclic
    conditioning): its parents are propositions the case holds, its conditioning is acyclic, and it names no
    constraint, since the case's constraints and their consistency check are not part of a deep case's single-stage
    events (WP5's event trees are)."""
    p = _payload(proposal, core.Proposition)
    held = {x.id: x for x in (_payload(q, core.Proposition) for q in _active(store, "proposition"))}
    held[p.id] = p
    problems = [f"parent {x} is not a proposition of this case" for x in p.parents if x not in held]
    if p.constraints:
        problems.append(f"names constraints {list(p.constraints)}, which a deep case's events do not take")
    seen: set[str] = set()
    stack = list(p.parents)
    while stack and not problems:
        x = stack.pop()
        if x == p.id:
            problems.append("its conditioning is cyclic")
        elif x not in seen and x in held:
            seen.add(x)
            stack += list(held[x].parents)
    return core.GateResult(gate="tests", passed=not problems,
                           detail="; ".join(problems) or "its parents are held and its conditioning is acyclic")  # fmt: skip


def approve_question(store: Store, question_id: str, *, approver: str, decision: str, reason: str, scope: str,
                     judge: JudgeCall | None = None) -> core.Approval:  # fmt: skip
    """G2 (spec §2.4: "any contested diagnosis on a relevant question"): a human's decision on a contested question,
    approving its resolution as it now stands (its lineage), recorded as the human's answer (wide.human_decision). The
    question's answer stays held as not established (spec §2.1); the approval is recorded beside its resolution."""
    case = open_deep(store)
    res = resolve_case(case, Judge(judge or providers.judge))
    r = res.resolved.get(question_id)
    if r is None or r.primary_diagnosis != "contested":
        state = "not resolved" if r is None else f"routed {r.primary_diagnosis}"
        raise Refused(f"G2 decides a contested question; {question_id} is {state}")
    return wide.human_decision(store, "G2", "question", question_id, res.asked[question_id].question.version,
                               contested_lineage(r), approver=approver, decision=decision, reason=reason, scope=scope)  # fmt: skip


def propose_event(store: Store, event: core.Proposition, *, author: str, rationale: str) -> core.Proposal:
    """A pending event proposed into the case as a proposition, through its gates (spec I1)."""
    store.register_gate("proposition", "tests", proposition_tests)
    payload = event.model_dump(mode="json")
    return store.submit_proposal(core.Proposal(
        id=f"proposition-{core.sha256(core.canonical_json(payload))[:16]}", case=store.case, kind="proposition",
        payload=payload, author=author, rationale=rationale, status="proposed"))  # fmt: skip


# ---------------------------------------------------------------- the CLI's steps


def sensitivities(store: Store, *, judge: JudgeCall | None = None) -> tuple[Case, Resolutions]:
    case = open_deep(store)
    return case, resolve_case(case, Judge(judge or providers.judge))


def recompute(store: Store, *, judge: JudgeCall | None = None, runtime: Runtime | None = None
              ) -> tuple[Case, Resolutions, Runtime, Diff]:  # fmt: skip
    """Step 2: the case's graph recomputed (on `runtime`, where one carries an earlier recompute: only what changed is
    recomputed, and the diff says what)."""
    case, res = sensitivities(store, judge=judge)
    rt = runtime or Runtime(store)
    diff = rt.recompute(build_graph(case, res))
    return case, res, rt, diff


def compare(store: Store, before: int, after: int, *, judge: JudgeCall | None = None) -> Diff:
    """The diff between the materiality of two G1-approved versions of the entity model, each with the G1-approved
    decision spec of its horizon: the earlier computed first, then the later on the same runtime, so only what the
    change reaches is recomputed (spec §5; WP11a)."""
    models = {m.version: m for m in (_payload(p, core.EntityModel) for p in _active(store, "entity_model"))}
    specs = sorted(((_payload(p, core.DecisionSpec), p.id) for p in _active(store, "decision_spec")),
                   key=lambda s: s[0].version)  # fmt: skip
    case = open_deep(store)
    rt = Runtime(store)
    calls = Judge(judge or providers.judge)
    diff: Diff | None = None
    for v in (before, after):
        if v not in models:
            raise Refused(f"no G1-approved entity model of version {v} in this case")
        found = next(((sp, sid) for sp, sid in reversed(specs) if sp.horizon == models[v].horizon), None)
        if found is None:
            raise Refused(f"no G1-approved decision spec has entity model v{v}'s horizon")
        spec, spec_id = found
        state, claims, artifacts = _evidence(store, spec.decision_time)
        c = Case(store, models[v], case.model_proposal, spec, spec_id, case.events, state, claims, artifacts)
        diff = rt.recompute(build_graph(c, resolve_case(c, calls)))
    assert diff is not None
    return diff


def run_register(store: Store, k: int, *, judge: JudgeCall | None = None, agent: AgentCall | None = None,
                 listing: SeriesListing | None = None) -> tuple[Analysis, Register]:  # fmt: skip
    case = open_deep(store)
    calls = Judge(judge or providers.judge)
    an = analyse(case, k, calls)
    scr = screen(case, calls, agent or runner.run, listing or _kalshi_listing)
    return an, register(an, scr)


def run_screen(store: Store, *, judge: JudgeCall | None = None, agent: AgentCall | None = None,
               listing: SeriesListing | None = None) -> tuple[Case, Screen]:  # fmt: skip
    case = open_deep(store)
    return case, screen(case, Judge(judge or providers.judge), agent or runner.run, listing or _kalshi_listing)


def latest_run(store: Store, k: int | None = None) -> RequisiteRun:
    runs = [r for r in store.all("requisite_run", RequisiteRun) if k is None or r.k == k]
    if not runs:
        raise Refused("no requisite run is committed in this case: run `requisite deep loop`")
    return runs[-1]




# ---------------------------------------------------------------- what the CLI prints


def _iv(v: tuple[Fraction | None, Fraction | None]) -> str:
    lo, hi = v
    return f"[{'-inf' if lo is None else f'{float(lo):,.2f}'}, {'+inf' if hi is None else f'{float(hi):,.2f}'}]"


def render_sensitivities(case: Case, res: Resolutions) -> str:
    lines = [f"case {case.store.case}: {case.model.entity}; entity model {case.model.id} v{case.model.version}, "
             f"decision spec {case.decision.id} v{case.decision.version}; evidence as of {case.state.as_of.isoformat()} "
             f"({len(case.state.observed)} claim(s), state {case.state_hash[:12]})"]  # fmt: skip
    for q in res.in_play():
        r = res.resolved.get(q.id)
        n = len(res.judgments[q.id])
        if r is None:
            lines.append(f"  {q.id}: {len(diagnose.usable(res.judgments[q.id]))} of {n} configurations returned a "
                         "distribution, too few to resolve it")  # fmt: skip
            continue
        lines.append(f"  {q.id}: {len(r.judgments)} of {n} configurations, S_arg {r.s_arg:.3f}, S_prob {r.s_prob:.3f}, "
                     f"{'perturbation-stable' if r.stable else f'routed {r.primary_diagnosis}'}; applies "
                     f"{r.application_state} ({r.evidence_status})")  # fmt: skip
    for e in case.events:
        pe = res.estimates.get(e.id)
        if pe is None:
            lines.append(f"  event {e.id}: no probability (its forecast returned fewer than two distributions)")
        else:
            lo, hi = pe.interval or (pe.value, pe.value)
            lines.append(f"  event {e.id}: P(occurs) = {float(pe.value):.4f} ({pe.id}; configurations from "
                         f"{float(lo):.4f} to {float(hi):.4f})")  # fmt: skip
    return "\n".join(lines)


def render_materiality(case: Case, rt: Runtime) -> str:
    lines = [f"materiality over the horizon {_window(case.model.horizon)}, in {case.model.currency} (cash deltas; "
             "negative is cash out)"]  # fmt: skip
    for d, e in case.exposures:
        v = exposure_vertices(f"{d.id}.{e.id}")
        lines.append(f"  {d.id} x {e.id}: loss if the event occurs {_iv(rt.value(v['loss']))}")
        if v["expected"] in rt.graph:
            lines.append(f"    expected {_iv(rt.value(v['expected']))}, tail {_iv(rt.value(v['tail']))}, breach "
                         f"{rt.value(v['breach'])}")  # fmt: skip
        for month, lo, hi in rt.value(v["cash"]):
            lines.append(f"    {month.isoformat()[:7]} {_iv((lo, hi))}")
    return "\n".join(lines)


def render_screen(case: Case, scr: Screen) -> str:
    lines = []
    for e in case.events:
        s = scr.scouted[e.id]
        lines.append(f"event {e.id}: the market scout searched {', '.join(s.categories) or 'no category'}; listed "
                     f"markets close: {', '.join(m.market or '' for m in s.markets) or 'none found'}")  # fmt: skip
        if s.not_listed:
            lines.append(f"  named but not in the listing, so not asked about: {', '.join(s.not_listed)}")
    for d, e in case.exposures:
        x = f"{d.id}.{e.id}"
        lines.append(f"  {x}: hedgeable {scr.hedgeable[x].value}; a listed market the scout found tracks the loss "
                     f"{scr.found_market_tracks[x].value}")  # fmt: skip
    return "\n".join(lines)


def render_register(an: Analysis, reg: Register) -> str:
    lines = [f"register for {an.decision.id} v{an.decision.version} (k = {an.k}, tolerance {an.tolerance}); the "
             f"minimax top k: {' | '.join(action_name(s) for s in reg.top_k) or '(none)'}"]  # fmt: skip
    for r in reg.rows:
        lines.append(f"  {r.driver} x {r.event}: {r.direction}, {r.mechanism}, {r.magnitude_band}; expected "
                     f"{_iv(r.expected_impact)}, tail {_iv(r.tail_impact)}, breach {r.breach}, hedgeable {r.hedgeable}; "
                     f"probability {r.probability}")  # fmt: skip
        for dg in r.diagnostics:
            dx = "perturbation-stable" if dg.primary_diagnosis == "stable" else f"routed {dg.primary_diagnosis}"
            lines.append(f"    {dg.question.id}: S_arg {dg.s_arg:.3f}, S_prob {dg.s_prob:.3f}, {dx}, "
                         f"regret {_show(dg.regret)}{' (informational)' if dg.informational else ''}")  # fmt: skip
        lines.append(f"    traces {len(r.claims)} claim(s), {len(r.judgments)} judgment(s), vertices {', '.join(r.nodes)}")
    for x, why in reg.blocked.items():
        lines.append(f"  {x}: not in the register: {why}")
    return "\n".join(lines)


def render_run(run: RequisiteRun) -> str:
    note = {"blocked": " (a blocked draft, not an approved conclusion)",
            "local_requisite": " (bounded by the records below: A02 §3.7)", "awaiting_approval": ""}[run.status]  # fmt: skip
    lines = [f"requisite run for {run.decision.id} v{run.decision.version} (k = {run.k}, tolerance {run.tolerance}): "
             f"{run.status.upper()}{note}; {run.iterations} iteration(s), {run.spent} of {run.budget} calls spent"]  # fmt: skip
    lines.append("  stopping conditions (A02 §3.7):")
    lines += [f"    [{'held' if c.held else 'NOT HELD'}] {c.condition}: {c.detail}" for c in run.conditions]
    lines.append(f"  top k: {' | '.join(' + '.join(s) for s in run.top_k) or '(none)'}")
    by_id = {q.question.id: q for q in run.questions}
    lines.append("  review queue (priority: relevance x (1 - min(S_arg, S_prob)), ties by the objective's spread):")
    lines += [f"    {i}. {r.id}: {by_id[r.id].action}: {by_id[r.id].reason}" for i, r in enumerate(run.queue, 1)] or ["    (empty)"]
    lines.append("  informational for this decision (regret within the tolerance; out of the queue, inspectable):")
    lines += [f"    {q.question.id}: {q.reason}" for q in run.questions if q.informational] or ["    (none)"]
    lines.append("  relevant and settled:")
    lines += [f"    {q.question.id}: {q.reason}" for q in run.questions
              if not q.informational and q.action == "none"] or ["    (none)"]  # fmt: skip
    lines.append("  unknowns (relevant, no established answer applied):")
    lines += [f"    {r.id}" for r in run.unknowns] or ["    (none)"]
    lines.append("  replaced:")
    lines += [f"    {q.question.id} ({'perturbation-stable' if q.primary_diagnosis == 'stable' else f'routed {q.primary_diagnosis}'}"
              f"): {q.reason}" for q in run.questions if q.action == "replaced"] or ["    (none)"]  # fmt: skip
    lines.append("  searched refinement set (A02 §3.2):")
    for x in run.searched:
        lines.append(f"    {x.question.id} under {x.shared_answer}:")
        lines += [f"      {c.source}: {'tried' if c.tried else 'not tried'}: {c.description}"
                  f"{f'; witness {c.witness[:12]}' if c.witness else ''}" for c in x.candidates]  # fmt: skip
    if not run.searched:
        lines.append("    (no relevant perturbation-stable question)")
    lines.append(f"  proposals: {len(run.proposals)}")
    lines.append("  outstanding:")
    lines += [f"    {g}" for g in run.outstanding_gates] or ["    (none)"]
    for exposure, why in run.blocked_exposures.items():
        lines.append(f"  {exposure}: not in the register: {why}")
    lines.append(f"  domain: {run.domain}")
    return "\n".join(lines)
