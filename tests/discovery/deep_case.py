"""A seeded deep case and test doubles of its providers, agent and market listing (the tests' own input, never a
cassette). The business, its drivers, its evidence and every judge's answer are this file's: a test states how the
judges answer by a rule over the question, the configuration and the wording, so each diagnosis it checks is seeded."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal as D
from fractions import Fraction as F
from pathlib import Path
from typing import Any

from requisite import core
from requisite.agents.runner import AgentResult
from requisite.core.evidence import artifact_id, claim_id
from requisite.core.governance import FloorPoint
from requisite.discovery import deep
from requisite.judgment import providers
from requisite.library import templates as library
from requisite.store import Store

NOW = datetime(2026, 11, 2, 12, tzinfo=UTC)
AS_OF = datetime(2026, 11, 1, tzinfo=UTC)
HORIZON = core.TimeWindow(start=AS_OF, end=datetime(2027, 11, 1, tzinfo=UTC), start_closed=True, end_closed=False)
WINDOW = core.TimeWindow(start=AS_OF, end=datetime(2027, 3, 31, tzinfo=UTC), start_closed=True, end_closed=True)
EVIDENCE = (
    "We run twelve crews of landscapers year round and pay them about $4,000 a month each. The proposed overtime rule "
    "would add roughly ten percent to our labor cost within a couple of months of taking effect. Fuel is a small cost "
    "for us and the rule does not touch it."
)
TEMPLATES = (*deep.SENSITIVITY, deep.FORECAST, "observable_tracks_loss", "client_can_influence")


def claimed(store: Store, text: str, locator: str) -> core.Claim:
    content = text.encode()
    art = core.Artifact(id=artifact_id(content), locator=locator, source="regulations.gov", media_type="text/plain",
                        size=len(content), published_at=datetime(2026, 10, 1, tzinfo=UTC),
                        published_at_basis="the comment's posted date", observed_at=datetime(2026, 10, 15, tzinfo=UTC),
                        availability="retrieved")  # fmt: skip
    store.commit_artifact(art, content)
    store.commit_extraction(art.id, "plain-v1", text)
    c = core.Claim(id=claim_id(art.id, "plain-v1", (0, len(text)), text, "a landscaping company"), artifact=art.id,
                   extraction="plain-v1",
                   span=(0, len(text)), text=text, authority="a landscaping company")  # fmt: skip
    store.commit_claim(c)
    return c


def approve(store: Store, p: core.Proposal) -> core.Proposal:
    lineage = core.sha256(core.canonical_json(dict(p.payload)))
    store.commit_approval(core.Approval(id=f"G1-{p.id}", gate="G1", subject=core.Ref(id=p.id, version=1),
                                        lineage_hash=lineage, approver="a test", at=NOW, decision="approved",
                                        reason="the fixture's", scope="this case"))  # fmt: skip
    return store.resubmit_after_approval(p.id)


def submit(store: Store, kind: str, record: core.Record) -> core.Proposal:
    payload = record.model_dump(mode="json")
    return store.submit_proposal(core.Proposal(
        id=f"{kind}-{core.sha256(core.canonical_json(payload))[:16]}", case=store.case, kind=kind, payload=payload,
        author="a test", rationale="the fixture's", status="proposed"))  # fmt: skip


def paraphrases(store: Store, template_id: str) -> None:
    t = library.template(template_id)
    ps = [library.original(t)] + [
        core.Paraphrase(id=pid, target=f"{word}: {t.target}", meanings={a.value: f"{word}, {a.meaning}" for a in t.answers})
        for pid, word in (("p1", "Put another way"), ("p2", "In other words"))
    ]  # fmt: skip
    submit(store, "paraphrase_set", core.ParaphraseSet(id=f"ps.{t.id}", version=t.version,
                                                       template=core.Ref(id=t.id, version=t.version), slots=list(t.slots),
                                                       answers=[a.value for a in t.answers], paraphrases=ps))  # fmt: skip


def param(name: str, value: str, unit: str, claim: core.Claim | None) -> core.Parameter:
    if claim is not None:
        return core.Parameter(name=name, value=D(value), unit=unit, claims=[claim.id])
    return core.Parameter(name=name, value=D(value), unit=unit,
                          assumption=core.Assumption(text=f"{name} is {value} {unit}", binds={},
                                                     reason="the filing does not state it"))  # fmt: skip


@dataclass(frozen=True)
class DriverSpec:
    id: str
    name: str
    quantity: str
    price: str
    line: str = "cost"


DRIVERS = (DriverSpec("labor", "crew wages", "12", "4000"), DriverSpec("fuel", "fuel", "1000", "4"))
# Two drivers whose exposures compete for the top one: labor's 5-15% rise and materials' 15-40% rise overlap, so which
# enters the top k depends on labor's band, lag, direction and conditionality and on materials' band.
COMPETING = (DriverSpec("labor", "crew wages", "12", "4000"), DriverSpec("materials", "materials", "1", "15000"))


def entity_model(claim: core.Claim, drivers: Sequence[DriverSpec] = DRIVERS, version: int = 1,
                 cash: str = "200000", credit: str = "50000") -> core.EntityModel:  # fmt: skip
    one = param("seasonality", "1", "factor", None)
    liquidity = core.LiquidityProfile(cash=param("cash", cash, "USD", None), credit_line=param("credit", credit, "USD", None),
                                      receivable_days=param("receivable days", "30", "days", None),
                                      payable_days=param("payable days", "30", "days", None), seasonality=[one] * 12)  # fmt: skip
    return core.EntityModel(
        id="em", version=version, entity="A landscaping company", currency="USD", horizon=HORIZON,
        loss_definition="incremental operating cash cost",
        drivers=[core.Driver(id=d.id, line=d.line, name=d.name,  # type: ignore[arg-type]
                             quantity=param(f"{d.name} quantity", d.quantity, "units/month", claim),
                             price=param(f"{d.name} price", d.price, "USD/unit", claim), period="month")
                 for d in drivers], liquidity=liquidity,
    )  # fmt: skip


def decision_spec(tolerance: str = "1000") -> core.DecisionSpec:
    return core.DecisionSpec(
        id="decision", version=1, entity="A landscaping company", decision_time=AS_OF, horizon=HORIZON,
        loss_definition="incremental operating cash cost", actions=["protect", "accept"], objective="liquidity_cvar",
        alpha=F(9, 10), tolerance=D(tolerance), liquidity_floor=[FloorPoint(day=date(2026, 11, 1), amount=D("0"))],
        currency="USD", information=[], recourse="none",
    )  # fmt: skip


EVENT = core.Proposition(id="E1", text="The overtime rule takes effect", window=WINDOW, domain="bool")


def make_case(tmp_path: Path, *, drivers: Sequence[DriverSpec] = DRIVERS, tolerance: str = "1000",
              cash: str = "200000", credit: str = "50000", evidence: str = EVIDENCE,
              events: Sequence[core.Proposition] = (EVENT,), templates: Sequence[str] = ()) -> Store:  # fmt: skip
    """A deep case past G1: the entity model and decision spec approved, the pending events proposed, and the paraphrase
    sets of deep's templates (and of `templates`) active."""
    store = Store(tmp_path / "case", case="deep", clock=lambda: NOW)
    claim = claimed(store, evidence, "https://example.org/comment/1")
    for t in (*TEMPLATES, *templates):
        paraphrases(store, t)
    approve(store, submit(store, "entity_model", entity_model(claim, drivers, cash=cash, credit=credit)))
    approve(store, submit(store, "decision_spec", decision_spec(tolerance)))
    for event in events:
        assert deep.propose_event(store, event, author="a test", rationale="the fixture's").status == "active"
    return store


# ---------------------------------------------------------------- the judges


def spread(values: Sequence[str], probs: Mapping[str, str]) -> dict[str, float]:
    """A distribution over every answer: the stated probabilities, the rest shared equally (all exact decimals)."""
    rest = [v for v in values if v not in probs]
    left = 1 - sum(F(p) for p in probs.values())
    each = left / len(rest) if rest else F(0)
    out = {v: F(probs[v]) if v in probs else each for v in values}
    assert sum(out.values()) == 1
    return {v: float(p) for v, p in out.items()}


def leaning(values: Sequence[str], top: str, p: str = "0.9") -> dict[str, float]:
    assert top in values, f"{top} is not among {values}"
    rest = [v for v in values if v != top]
    share = (1 - F(p)) / len(rest)
    return {v: float(F(p) if v == top else share) for v in values}


Rule = Callable[[providers.Ask], Mapping[str, float]]

# Each base question's answer in the seeded world: labor rises 5-15% within one to three months, unconditionally; the
# rule does nothing to fuel; materials rise 15-40% within a month; the event is likelier than not.
ANSWERS: dict[str, str] = {
    "sensitivity_direction.labor": "increase", "sensitivity_mechanism.labor": "cost",
    "sensitivity_magnitude.labor": "5_to_15pct", "sensitivity_lag.labor": "1_to_3_months",
    "sensitivity_conditionality.labor": "unconditional",
    "sensitivity_direction.fuel": "none", "sensitivity_mechanism.fuel": "cost",
    "sensitivity_magnitude.fuel": "under_1pct", "sensitivity_lag.fuel": "under_1_month",
    "sensitivity_conditionality.fuel": "unconditional",
    "sensitivity_direction.materials": "increase", "sensitivity_mechanism.materials": "cost",
    "sensitivity_magnitude.materials": "15_to_40pct", "sensitivity_lag.materials": "under_1_month",
    "sensitivity_conditionality.materials": "unconditional",
    "observable_tracks_loss": "tracks", "client_can_influence": "cannot_influence",
}  # fmt: skip


def template_of(a: providers.Ask) -> str:
    return a.question.template_id or a.question.id


def base_rule(a: providers.Ask) -> Mapping[str, float]:
    values = list(a.asked.answers)
    t = a.question.template_id
    if t == deep.FORECAST:
        return {"occurs": 0.6, "does_not_occur": 0.4}
    if t in ("observable_tracks_loss", "client_can_influence"):
        return leaning(values, ANSWERS[t])
    if t is not None:
        driver = a.question.id.split(".")[2]
        return leaning(values, ANSWERS[f"{t}.{driver}"])
    # an agent's question: its first substantive answer, firmly
    return leaning(values, next(v for v in values if v != "not_established"))


@dataclass
class Judges:
    """A test double of the providers: answers by `rule`, as the provider layer's Judgment, counting every call."""

    rule: Rule = base_rule
    calls: list[providers.Ask] = field(default_factory=list)

    def __call__(self, a: providers.Ask) -> core.Judgment:
        self.calls.append(a)
        dist = dict(self.rule(a))
        step = 0.001 if any(round(v, 2) != v for v in dist.values()) else 0.01
        return core.Judgment(question=core.Ref(id=a.question.id, version=a.question.version), state_hash=a.state_hash,
                             config=a.config, reported=dist, distribution=dist, precision=step,
                             response=f"double-{core.sha256(core.canonical_json([a.asked.instructions, a.config.model_dump(mode='json')]))[:16]}",
                             latency_ms=0, cost_usd=D("0"), created_at=NOW)  # fmt: skip


# ---------------------------------------------------------------- the agent and the listing


Drafter = Callable[[str, Mapping[str, Any]], Any]


@dataclass
class Agent:
    """A test double of the agent runner: `draft(role, input)` gives the answer, recorded like a run."""

    draft: Drafter
    calls: list[tuple[str, Mapping[str, Any]]] = field(default_factory=list)

    def __call__(self, role: str, input: Mapping[str, Any], output: type, *, effort: str) -> AgentResult[Any]:
        self.calls.append((role, input))
        value = self.draft(role, input)
        assert isinstance(value, output), f"{role} answered a {type(value).__name__}, not a {output.__name__}"
        run = "run-" + core.sha256(core.canonical_json([role, dict(input)]))
        return AgentResult(role, value, run, "test-double", effort, True)


def no_markets(role: str, input: Mapping[str, Any]) -> Any:
    from requisite.agents.answers import CategoriesDraft, MarketsDraft

    if role == "market_scout":
        if "categories" in input:
            return CategoriesDraft(categories=["Politics"], rationale="a rule would be listed under politics")
        return MarketsDraft(tickers=[], rationale="no listed series settles on this rule")
    raise AssertionError(f"no draft for {role}")


LISTING = [
    {"ticker": "KXEXAMPLE", "title": "Will an unrelated bill pass?", "category": "Politics",
     "settlement_sources": [{"name": "Congress.gov"}]},
    {"ticker": "KXSPORT", "title": "Who wins the cup?", "category": "Sports", "settlement_sources": []},
]  # fmt: skip


def listing(as_of: datetime) -> list[Mapping[str, Any]]:
    return list(LISTING)


def para(target: str, answers: Sequence[Any]) -> list[Any]:
    from requisite.agents.answers import ParaphraseDraft

    return [ParaphraseDraft(id=p, target=f"{word}: {target}", meanings={a.value: f"{word}, {a.meaning}" for a in answers})
            for p, word in (("p1", "Put another way"), ("p2", "In other words"))]  # fmt: skip


def refiner(role: str, input: Mapping[str, Any]) -> Any:
    """The question_refiner's double: a refinement keeps the answers and states the window; a sharper question puts its
    boundary where the witness says the decision changes; a split asks first whether the rule covers the driver."""
    from requisite.agents.answers import DraftAnswer, DraftQuestion, QuestionsDraft

    if role == "market_scout":
        return no_markets(role, input)
    q, task = input["question"], input["task"]
    if task == "refine":
        answers = [DraftAnswer(**a) for a in q["answers"]]
        t = "Counting from the day the event takes effect: " + q["target"]
        return QuestionsDraft(questions=[DraftQuestion(target=t, answers=answers, paraphrases=para(t, answers))],
                              rationale="the window was read two ways")  # fmt: skip
    if task == "sharpen":
        shared = input["witness"]["shared_answer"]
        a = next(x for x in q["answers"] if x["value"] == shared)
        ((key, (lo, hi)),) = a["binds"].items()
        cut = str(F(input["witness"]["decision_changes_between"][key][1]))
        answers = [
            DraftAnswer(value="lower", meaning=f"from {lo} to {cut}", status="affirmative", binds={key: [lo, cut]}),
            DraftAnswer(value="upper", meaning=f"from {cut} to {hi}", status="affirmative", binds={key: [cut, hi]}),
            DraftAnswer(value="not_established", meaning="the evidence does not establish which", status="not_established",
                        binds={}),
        ]  # fmt: skip
        t = f"Given that it is {shared}: {q['target']}"
        return QuestionsDraft(questions=[DraftQuestion(target=t, answers=answers, paraphrases=para(t, answers))],
                              rationale="the decision changes inside the band")  # fmt: skip
    if task == "split":
        gate = [DraftAnswer(value="covered", meaning="the rule covers it", status="affirmative", binds={}),
                DraftAnswer(value="exempt", meaning="the rule exempts it", status="negative",
                            binds=dict(next(x for x in q["answers"] if x["value"] in ("under_1pct", "none"))["binds"])),
                DraftAnswer(value="not_established", meaning="the evidence does not establish whether",
                            status="not_established", binds={})]  # fmt: skip
        rest = [DraftAnswer(**x) for x in q["answers"]]
        t1, t2 = "Does the rule cover this driver at all?", f"Where the rule covers this driver: {q['target']}"
        return QuestionsDraft(questions=[DraftQuestion(target=t1, answers=gate, paraphrases=para(t1, gate)),
                                         DraftQuestion(target=t2, answers=rest, paraphrases=para(t2, rest))],
                              rationale="it asked whether the rule applies and how much at once")  # fmt: skip
    raise AssertionError(f"no draft for {task}")
