"""A deep case opened from the lead a human promoted at G1 (requisite/discovery/lead.py, `requisite deep open`), driven
through the CLI as a user runs it, over a wide case built by the wide flow's own functions on their test doubles
(tests/dossier/cases.py): the run's families, estimate and leads, the promotion, G1, the deep case opened, G1 on its
entity model and decision spec, then every deep step on that business.

The public record is the fixture docket with one more filing: a small operator stating its costs (the world's own text,
as every judge's answer and agent's draft here is the test's own). The flow ranks it first, since it asks for
predictability and states a quantified impact; the entity model quotes the numbers it states and assumes the rest, each
with its reason, as a comment letter leaves it."""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import deep_case as dc
import pytest
from cases import recording, wide_case
from fake_api import DOCKET, FakeAPI, comment_entry, fixture_world, load
from pydantic import ValidationError
from test_wide_flow import OBJECTIVE, agent_double, stand_in

from requisite import cli, core
from requisite.agents import runner
from requisite.agents.answers import (
    Assumed,
    DriverDraft,
    EntityModelDraft,
    ImpactDraft,
    ImpactsDraft,
    LiquidityDraft,
    NumberDraft,
    ParaphraseDraft,
    ParaphrasesDraft,
    StatedBy,
)
from requisite.agents.runner import AgentResult
from requisite.compute.runtime import Graph, Runtime, Source
from requisite.discovery import deep, lead, wide
from requisite.evidence.items import committed
from requisite.judgment import providers
from requisite.store import open_case

OPERATOR = "Skyward Charter LLC"
FILED = f"{DOCKET}-0007"
POSTED, MODIFIED = "2026-05-26T04:00:00Z", "2026-05-26T15:12:40Z"
FILING = (
    "Skyward Charter LLC is a small business with 38 employees that flies two Boeing 757-300 airplanes on charter "
    "contracts. The inspections this proposal mandates would take each airplane out of service for about 120 hours of "
    "mechanic time, $11,400 per airplane at the $95 an hour we pay our contract mechanics. We spend about $310,000 a "
    "month on jet fuel. We ask the FAA to allow the inspections at each airplane's next scheduled heavy check, so that "
    "we can plan for the cost."
)
IMPACT = "$11,400 per airplane"
DAY = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
# A deep case's cash flows run month by month over a bounded horizon: the objective approved at G1 states one.
BOUNDED = OBJECTIVE.model_copy(update={"horizon": core.TimeWindow(start=DAY, end=DAY + timedelta(days=365),
                                                                  start_closed=True, end_closed=False)})  # fmt: skip
HUMAN = ["--approver", "owen", "--reason", "as proposed", "--scope", "this case"]
DECISION = ["--objective", "liquidity_cvar", "--alpha", "9/10", "--tolerance", "1000", "--floor", f"{DAY.date()}=0",
            "--actions", "protect", "accept", "--information", "--recourse", "none", "--currency", "USD"]  # fmt: skip


def world(api: FakeAPI) -> FakeAPI:
    """The fixture docket, and the operator's comment on its proposed rule, in the recorded shape (Aviation Partners
    Boeing's record, which has no attachment, with the operator's own attributes)."""
    fixture_world(api)
    body = json.loads(json.dumps(load("comment_1.json")["body"]).replace(f"{DOCKET}-0004", FILED))
    body["data"]["attributes"].update(comment=FILING, organization=OPERATOR, title=f"Comment from {OPERATOR}",
                                      postedDate=POSTED, receiveDate=POSTED, modifyDate=MODIFIED)  # fmt: skip
    api.add_comment(DOCKET, comment_entry(FILED, POSTED, MODIFIED), body)
    return api


# ---------------------------------------------------------------- the agents' doubles


def rewordings(template: Mapping[str, Any]) -> ParaphrasesDraft:
    return ParaphrasesDraft(paraphrases=[
        ParaphraseDraft(id=p, target=f"{word}: {template['target']}",
                        meanings={a["value"]: f"{word}, {a['meaning']}" for a in template["answers"]})
        for p, word in (("p1", "Put another way"), ("p2", "In other words"))])  # fmt: skip


def quoted(agent_input: Mapping[str, Any], name: str, value: str, unit: str, words: str) -> NumberDraft:
    claim = next(c for c in agent_input["claims"] if words in c["text"])
    return NumberDraft(name=name, value=value, unit=unit, claims=[StatedBy(id=claim["id"], words=words)], assumption=None)


def assumed(name: str, value: str, unit: str, text: str, reason: str) -> NumberDraft:
    return NumberDraft(name=name, value=value, unit=unit, claims=[], assumption=Assumed(text=text, reason=reason))


def operator_model(agent_input: Mapping[str, Any], *, rate: NumberDraft | None = None) -> EntityModelDraft:
    """The decomposer's double: the two prices the filing states, quoted from the claims it is given; every other number
    assumed with its reason. `rate`: the mechanic rate as a test gives it instead."""
    labor = DriverDraft(id="labor", line="cost", name="contract mechanic labor", period="month",
                        quantity=assumed("mechanic hours", "400", "hours/month", "400 hours of routine mechanic time a "
                                         "month for two airplanes", "the filing states the inspection's hours, not the "
                                         "routine workload"),
                        price=rate or quoted(agent_input, "mechanic rate", "95", "USD/hour", "$95 an hour"))  # fmt: skip
    fuel = DriverDraft(id="fuel", line="cost", name="jet fuel", period="month",
                       quantity=assumed("fuel bills", "1", "bill/month", "the fuel spending is one bill a month",
                                        "the filing states its fuel spending as a monthly total"),
                       price=quoted(agent_input, "monthly fuel spending", "310000", "USD/bill",
                                    "$310,000 a month on jet fuel"))  # fmt: skip
    one = assumed("seasonality", "1", "factor", "no seasonal pattern", "the filing states no seasonality")
    liquidity = LiquidityDraft(
        cash=assumed("cash on hand", "600000", "USD", "two months of the stated fuel and mechanic spending",
                     "the filing states no cash position"),
        credit_line=assumed("credit line", "0", "USD", "no credit line", "the filing states none"),
        receivable_days=assumed("receivable days", "30", "days", "charter customers pay in 30 days",
                                "the filing states no payment terms"),
        payable_days=assumed("payable days", "30", "days", "mechanics and fuel are paid in 30 days",
                             "the filing states no payment terms"),
        seasonality=[one] * 12,
    )  # fmt: skip
    return EntityModelDraft(drivers=[labor, fuel], liquidity=liquidity, currency="USD",
                            rationale="mechanic labor is the line the measure moves; fuel carries most of the cash")  # fmt: skip


@dataclass
class Agents:
    """The agent runner's double for every role the flows call, each answer the test's own, recorded like a run (its id
    is its request's hash, so a step run again replays the same run)."""

    decomposer: Callable[[Mapping[str, Any]], EntityModelDraft] = operator_model
    calls: list[str] = field(default_factory=list)

    def __call__(self, role: str, agent_input: Mapping[str, Any], output: type, *, effort: str, root: Any = None,
                 context: Mapping[str, Any] | None = None) -> AgentResult[Any]:  # fmt: skip
        self.calls.append(role)
        if role == "family_drafter":
            return agent_double(role, agent_input, output, effort=effort, context=context)
        if role == "impact_extractor":
            stated = [ImpactDraft(quote=IMPACT, amount="11400", currency="USD", period="one_time")]
            value: Any = ImpactsDraft(impacts=stated if IMPACT in agent_input["comment"] else [])
        elif role == "paraphrase_writer":
            value = rewordings(agent_input["template"])
        elif role == "decomposer":
            value = self.decomposer(agent_input)
        else:
            value = dc.refiner(role, agent_input)
        assert isinstance(value, output), f"{role} answered a {type(value).__name__}, not a {output.__name__}"
        return AgentResult(role, value, "run-" + core.sha256(core.canonical_json([role, dict(agent_input)])),
                           "test-double", effort, True)  # fmt: skip


# ---------------------------------------------------------------- the wide case


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The wide case through its gold labels, built once (tests/dossier/cases.py) over the world above."""
    tmp = tmp_path_factory.mktemp("lead")
    with pytest.MonkeyPatch.context() as mp:
        wide_case(mp, tmp, objective=BOUNDED, world=world)
    return tmp


@pytest.fixture
def wide_dir(built: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A copy of the built case and its recordings, in a process that records as the build did (the doubles answer)."""
    shutil.copytree(built, tmp_path / "w")
    recording(monkeypatch, tmp_path / "w")
    monkeypatch.setattr(providers, "_send", stand_in)
    return tmp_path / "w" / "wide"


def where(wide_dir: Path) -> list[str]:
    return ["--case-dir", str(wide_dir), "--case", "wide"]


def promote_first(wide_dir: Path) -> str:
    """The run's families, estimate and leads, and its first lead proposed for promotion, through the CLI."""
    for step in ("families", "estimate", "leads"):
        assert cli.main(["wide", step, *where(wide_dir)]) == 0
    assert cli.main(["wide", "promote", *where(wide_dir), "--rank", "1"]) == 0
    (p,) = wide.proposals(open_case(wide_dir), "lead")
    return p.id


def sources_behind(graph: Graph, vids: Iterable[str]) -> list[Source]:
    """Every source vertex a set of vertices is computed from."""
    seen: set[str] = set()
    out: list[Source] = []
    stack = list(vids)
    while stack:
        vid = stack.pop()
        if vid in seen:
            continue
        seen.add(vid)
        v = graph[vid]
        if isinstance(v, Source):
            out.append(v)
        else:
            stack += list(v.inputs.values())
    return out


# ---------------------------------------------------------------- end to end


def test_a_lead_approved_at_g1_opens_a_deep_case_on_its_business_and_every_deep_step_runs_on_it(
    wide_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    agents = Agents()
    monkeypatch.setattr(runner, "run", agents)
    lead_id = promote_first(wide_dir)
    wide_store = open_case(wide_dir)
    first = wide_store.all("lead_list", core.LeadList)[-1].leads[0]
    # The run ranks the operator first: it asks for predictability and states an impact, kept as a claim of its filing.
    assert (first.submitter, first.tier, [i.amount for i in first.impact]) == (OPERATOR, 1, [11400])
    assert cli.main(["approve", lead_id, "--gate", "G1", *where(wide_dir), *HUMAN]) == 0
    capsys.readouterr()

    assert cli.main(["deep", "open", *where(wide_dir), "--lead", lead_id, *DECISION]) == 0
    shown = capsys.readouterr().out
    promotion = core.LeadPromotion.model_validate_json(core.canonical_json(dict(wide_store.proposal(lead_id).payload)))
    root = wide_dir / "deep" / promotion.case
    assert shown.startswith(f"deep case {promotion.case}: {root}\n")
    store = open_case(root)
    assert store.case == promotion.case

    # The evidence is the lead's filing exactly as the wide case committed it, and nothing else.
    item = committed(wide_store, [first.item])[first.item]
    measure = wide_store.get("item_measure", item.id, core.ItemMeasure)
    assert measure is not None and store.all("item", core.Item) == [item]
    assert store.get("item_measure", item.id, core.ItemMeasure) == measure
    arts = store.all("artifact", core.Artifact)
    assert {a.id for a in arts} == {*item.artifacts, *measure.artifacts}
    assert all(a == wide_store.get("artifact", a.id, core.Artifact) and store.content(a.id) == wide_store.content(a.id)
               for a in arts)  # fmt: skip
    claims = {c.id: c for c in store.all("claim", core.Claim)}
    assert claims and all(c == wide_store.get("claim", c.id, core.Claim) and c.artifact in item.artifacts
                          for c in claims.values())  # fmt: skip
    assert {i.claim for i in first.impact} < set(claims)  # the impact quoted, beside the filing's text
    assert any(FILING in c.text for c in claims.values())

    # The event is the measure the filing addresses, over the approved objective's horizon.
    (event_p,) = wide.proposals(store, "proposition")
    event = core.Proposition.model_validate_json(core.canonical_json(dict(event_p.payload)))
    assert event_p.status == "active" and event.id == measure.document == first.measure
    assert measure.statement in event.text and event.window == BOUNDED.horizon and event.domain == "bool"
    active = {p.payload["template"]["id"] for p in wide.proposals(store, "paraphrase_set") if p.status == "active"}
    assert active == set(lead.deep_templates()) >= {*deep.SENSITIVITY, deep.FORECAST, "observable_tracks_loss",
                                                    "client_can_influence"}  # fmt: skip

    # The entity model and the decision spec await G1; every number of the model is quoted from a claim of the filing
    # the case holds, or assumed with its reason.
    (model_p,) = wide.proposals(store, "entity_model")
    (spec_p,) = wide.proposals(store, "decision_spec")
    assert (model_p.status, spec_p.status) == ("proposed", "proposed")
    model = core.EntityModel.model_validate_json(core.canonical_json(dict(model_p.payload)))
    assert (model.entity, model.horizon, model.loss_definition, model.currency) == (
        OPERATOR, BOUNDED.horizon, BOUNDED.loss_definition, "USD")
    rate = next(d for d in model.drivers if d.id == "labor").price
    assert rate.value == 95 and len(rate.claims) == 1 and "$95 an hour" in claims[rate.claims[0]].text
    spec = core.DecisionSpec.model_validate_json(core.canonical_json(dict(spec_p.payload)))
    frame = wide.the_frame(wide_store)
    # Its decision time is the first instant everything the case holds is observable (I3), never before the frame's.
    held = [x for a in arts for x in (a.available_from, a.published_at or a.available_from)]
    assert spec.decision_time == max([frame.as_of, *held]) > frame.as_of
    assert all(core.observable_at(a, spec.decision_time, store.clock()) for a in arts)
    assert (spec.entity, spec.horizon, spec.loss_definition) == (OPERATOR, BOUNDED.horizon, BOUNDED.loss_definition)
    assert (spec.objective, spec.alpha, spec.tolerance, spec.currency, spec.recourse) == (
        "liquidity_cvar", Fraction(9, 10), Decimal("1000"), "USD", "none")
    assert [(f.day, f.amount) for f in spec.liquidity_floor] == [(DAY.date(), 0)]
    assert (list(spec.actions), list(spec.information)) == (["protect", "accept"], [])
    for gate_line in (f"  {model_p.id}: awaits G1", f"  {spec_p.id}: awaits G1"):
        assert gate_line in shown
    assert f'quoted from claim {rate.claims[0]}: "$95 an hour"' in shown
    assert "assumed: no credit line (no claim states it: the filing states none)" in shown

    # Run again, it opens the same case: the same records, nothing proposed twice.
    kinds = {k: store.keys(k) for k in store.kinds()}
    assert set(kinds) == {"case", "item", "artifact", "extraction", "claim", "item_measure", "proposal", "agent_run"}
    assert cli.main(["deep", "open", *where(wide_dir), "--lead", lead_id, *DECISION]) == 0
    assert capsys.readouterr().out == shown
    assert {k: store.keys(k) for k in store.kinds()} == kinds
    assert [len(store.proposal_history(p.id)) for p in (event_p, model_p, spec_p)] == [1, 1, 1]

    # G1 on both, then every deep step on that business, unchanged.
    deep_case = ["--case-dir", str(root), "--case", promotion.case]
    for p in (model_p, spec_p):
        assert cli.main(["approve", p.id, "--gate", "G1", *deep_case, *HUMAN]) == 0
    judges = dc.Judges()
    monkeypatch.setattr(providers, "judge", judges)
    monkeypatch.setattr(deep, "_kalshi_listing", dc.listing)
    for step in ("sensitivities", "materiality", "hedge"):
        assert cli.main(["deep", step, "--case", str(root)]) == 0
    for step in ("register", "loop"):
        assert cli.main(["deep", step, "--case", str(root), "--k", "1"]) == 0
    out = capsys.readouterr().out
    assert f"case {promotion.case}: {OPERATOR}" in out and f"labor x {event.id}:" in out

    # Every number in the register is computed from the approved model's parameters, each quoted from a claim the
    # case copied from the lead's filing or assumed with its reason, and from judgments over that filing.
    rows = store.all("register_row", core.RegisterRow)
    assert {(r.entity, r.driver, r.event) for r in rows} == {(OPERATOR, "labor", event.id), (OPERATOR, "fuel", event.id)}
    case = deep.open_deep(store)
    assert case.model == model and case.decision == spec and set(case.claims) == set(claims)
    # The judges read each of the filing's words once: the lead's quoted impact stays a claim of the case, citable by
    # the model, and lies within its part's claim, so the judges' evidence state does not show it again.
    assert set(case.state.observed) == set(claims) - {i.claim for i in first.impact}
    graph = deep.build_graph(case, deep.resolve_case(case, deep.Judge(judges)))
    rt = Runtime(store)
    rt.recompute(graph)
    sourced = set()
    for row in rows:
        v = deep.exposure_vertices(f"{row.driver}.{row.event}")
        assert (row.expected_impact, row.tail_impact) == (rt.value(v["expected"]), rt.value(v["tail"]))
        assert set(row.claims) <= set(claims)
        params = [r for s in sources_behind(graph, row.nodes) for r in s.records if isinstance(r, core.Parameter)]
        driver = next(d for d in model.drivers if d.id == row.driver)
        assert all(x in params for x in (driver.quantity, driver.price, model.liquidity.cash, model.liquidity.credit_line))
        for p in params:
            assert set(p.claims) <= set(claims) if p.claims else p.assumption is not None and p.assumption.reason.strip()
            sourced |= set(p.claims)
    assert sourced and sourced <= set(claims)

    # The dossier renders the opened case: its register, and the model and decision approved at G1.
    capsys.readouterr()
    assert cli.main(["dossier", *deep_case]) == 0
    pages = "".join(f.read_text() for f in (root / "dossier").rglob("*.html"))
    assert OPERATOR in pages and model_p.id in pages and spec_p.id in pages and event.id in pages


# ---------------------------------------------------------------- refusals


def test_a_lead_not_approved_at_g1_opens_nothing_and_an_approval_at_another_gate_does_not_open_it(
    wide_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(runner, "run", Agents())
    lead_id = promote_first(wide_dir)
    capsys.readouterr()
    promotion = core.LeadPromotion.model_validate_json(core.canonical_json(dict(open_case(wide_dir).proposal(lead_id).payload)))
    root = wide_dir / "deep" / promotion.case
    assert cli.main(["deep", "open", *where(wide_dir), "--lead", lead_id, *DECISION]) == 1
    err = capsys.readouterr().err
    assert f"lead {lead_id} is proposed (human gate: awaiting a human approval at G1" in err and not root.exists()
    # G4 approves a template's promotion: given to a lead, it authorizes nothing here.
    assert cli.main(["approve", lead_id, "--gate", "G4", *where(wide_dir), *HUMAN]) == 0
    capsys.readouterr()
    assert cli.main(["deep", "open", *where(wide_dir), "--lead", lead_id, *DECISION]) == 1
    err = capsys.readouterr().err
    assert f"lead {lead_id} is proposed" in err and "its approval at G4 authorizes something else" in err
    assert not root.exists()
    assert cli.main(["approve", lead_id, "--gate", "G1", *where(wide_dir), *HUMAN]) == 0
    assert cli.main(["deep", "open", *where(wide_dir), "--lead", lead_id, *DECISION]) == 0
    assert root.exists()


def test_a_lead_whose_item_has_no_measure_committed_is_refused_and_named(
    wide_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(runner, "run", Agents())
    lead_id = promote_first(wide_dir)
    assert cli.main(["approve", lead_id, "--gate", "G1", *where(wide_dir), *HUMAN]) == 0
    item = open_case(wide_dir).all("lead_list", core.LeadList)[-1].leads[0].item
    # The case as a reader that committed no measure left it (core.ItemMeasure was added on November 2).
    db = sqlite3.connect(wide_dir / "requisite.db")
    with db:
        db.execute("delete from records where kind = 'item_measure' and key = ?", (item,))
    db.close()
    capsys.readouterr()
    assert cli.main(["deep", "open", *where(wide_dir), "--lead", lead_id, *DECISION]) == 1
    assert f"no measure committed: run `wide read` ({item})" in capsys.readouterr().err
    assert not (wide_dir / "deep").exists()


def test_a_decomposer_answer_citing_a_claim_the_case_does_not_hold_or_words_it_does_not_contain_is_refused(
    wide_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(runner, "run", Agents())
    lead_id = promote_first(wide_dir)
    assert cli.main(["approve", lead_id, "--gate", "G1", *where(wide_dir), *HUMAN]) == 0
    wide_store = open_case(wide_dir)
    # Another business's filing in the same docket: a claim the wide case holds, and the deep case does not.
    other = next(c for c in wide_store.all("claim", core.Claim) if c.authority == "Boeing Commercial Airplanes")
    elsewhere = NumberDraft(name="mechanic rate", value="95", unit="USD/hour",
                            claims=[StatedBy(id=other.id, words="Concur")], assumption=None)  # fmt: skip
    monkeypatch.setattr(runner, "run", Agents(lambda i: operator_model(i, rate=elsewhere)))
    capsys.readouterr()
    assert cli.main(["deep", "open", *where(wide_dir), "--lead", lead_id, *DECISION]) == 1
    err = capsys.readouterr().err
    assert f"driver labor price (mechanic rate = 95 USD/hour) cites claim {other.id}, which this case does not hold" in err
    promotion = core.LeadPromotion.model_validate_json(core.canonical_json(dict(wide_store.proposal(lead_id).payload)))
    store = open_case(wide_dir / "deep" / promotion.case)
    assert not wide.proposals(store, "entity_model") and not wide.proposals(store, "decision_spec")
    assert other.id not in store.keys("claim")

    def misquoted(agent_input: Mapping[str, Any]) -> EntityModelDraft:
        right = quoted(agent_input, "mechanic rate", "95", "USD/hour", "$95 an hour").claims[0].id
        wrong = NumberDraft(name="mechanic rate", value="95", unit="USD/hour",
                            claims=[StatedBy(id=right, words="$59 an hour")], assumption=None)  # fmt: skip
        return operator_model(agent_input, rate=wrong)

    monkeypatch.setattr(runner, "run", Agents(misquoted))
    assert cli.main(["deep", "open", *where(wide_dir), "--lead", lead_id, *DECISION]) == 1
    assert "quotes '$59 an hour' from claim" in capsys.readouterr().err
    assert not wide.proposals(store, "entity_model")


def test_an_answers_quotes_are_checked_as_text_against_what_the_agent_read_and_its_amounts_are_its_reading() -> None:
    # As the runner validates an answer (runner.run: the caller's context, never sent to the agent). Provenance is checked
    # as text: a quote is copied from the words the agent was given. What the words state is the agent's reading, in the
    # answer's structured form (an exact decimal, a currency code, a period): no figure is matched against the words.
    texts = ["We expect the rule to cost us $2.5 million a year in added payroll.",
             "Our hangar lease alone runs eleven thousand four hundred dollars a month."]  # fmt: skip
    said = {"quote": "$2.5 million a year", "amount": "2500000", "currency": "USD", "period": "year"}
    spelled = {"quote": "eleven thousand four hundred dollars a month", "amount": "11400", "currency": "USD",
               "period": "month"}  # fmt: skip
    read = ImpactsDraft.model_validate_json(json.dumps({"impacts": [said, spelled]}), context={"texts": texts})
    assert [i.amount for i in read.impacts] == [Decimal(2500000), Decimal(11400)]
    with pytest.raises(ValidationError, match="not found"):
        ImpactsDraft.model_validate_json(json.dumps({"impacts": [{**said, "quote": "$2.5M a year"}]}),
                                         context={"texts": texts})  # fmt: skip
    with pytest.raises(ValidationError, match="not found"):  # one part's words: a quote across two parts is no claim
        ImpactsDraft.model_validate_json(json.dumps({"impacts": [{**said, "quote": "payroll. Our hangar"}]}),
                                         context={"texts": texts})  # fmt: skip
    for amount in ("2.5 million", "2,500,000", 2500000.0):  # the structured form is an exact decimal
        with pytest.raises(ValidationError):
            ImpactsDraft.model_validate_json(json.dumps({"impacts": [{**said, "amount": amount}]}), context={"texts": texts})

    claim = "clm_" + "0" * 64
    agent_input = {"claims": [{"id": claim, "text": "We pay contract mechanics $95 an hour and spend "
                                                    "$310,000 a month on jet fuel."}]}  # fmt: skip
    draft = operator_model(agent_input).model_dump(mode="json")
    held = {claim: agent_input["claims"][0]["text"]}
    assert EntityModelDraft.model_validate_json(json.dumps(draft), context={"claims": held}).numbers()
    with pytest.raises(ValidationError, match="whose text does not contain those words"):
        EntityModelDraft.model_validate_json(json.dumps(draft).replace("$95 an hour", "$59 an hour"),
                                             context={"claims": held})  # fmt: skip
    with pytest.raises(ValidationError, match="which this case does not hold"):
        EntityModelDraft.model_validate_json(json.dumps(draft), context={"claims": {}})


def test_the_decision_is_the_humans_as_given_and_refused_where_it_does_not_make_one(
    wide_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(runner, "run", Agents())
    lead_id = promote_first(wide_dir)
    assert cli.main(["approve", lead_id, "--gate", "G1", *where(wide_dir), *HUMAN]) == 0
    capsys.readouterr()

    def given(**changed: str) -> list[str]:
        out = list(DECISION)
        for flag, value in changed.items():
            out[out.index(f"--{flag}") + 1] = value
        return out

    # A CVaR objective states its tail level; no deep case is opened on a decision that is not one.
    assert cli.main(["deep", "open", *where(wide_dir), "--lead", lead_id, *given(alpha="none")]) == 1
    assert "a CVaR objective states its alpha" in capsys.readouterr().err
    assert not (wide_dir / "deep").exists()
    # The tolerance and the floor are amounts in the entity model's currency.
    assert cli.main(["deep", "open", *where(wide_dir), "--lead", lead_id, *given(currency="EUR")]) == 1
    assert "the decision's amounts are given in EUR, and the entity model's in USD" in capsys.readouterr().err
    # Without a flag, nothing is opened: none has a default.
    with pytest.raises(SystemExit):
        cli.main(["deep", "open", *where(wide_dir), "--lead", lead_id, *DECISION[:-2]])
    assert "the following arguments are required: --currency" in capsys.readouterr().err


def test_an_objective_whose_horizon_has_no_end_cannot_be_a_deep_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    case_dir = wide_case(monkeypatch, tmp_path, objective=OBJECTIVE, world=world)
    assert OBJECTIVE.horizon.end is None
    monkeypatch.setattr(runner, "run", Agents())
    lead_id = promote_first(case_dir)
    assert cli.main(["approve", lead_id, "--gate", "G1", *where(case_dir), *HUMAN]) == 0
    capsys.readouterr()
    assert cli.main(["deep", "open", *where(case_dir), "--lead", lead_id, *DECISION]) == 1
    assert "the approved objective's horizon cannot be a deep case's: a planning horizon has a start and an end" in (
        capsys.readouterr().err)
    assert not (case_dir / "deep").exists()


def test_the_decomposers_answer_type_refuses_a_number_neither_quoted_nor_assumed_with_its_reason() -> None:
    with pytest.raises(ValidationError, match="a parameter is sourced by claims or assumed, exactly one"):
        NumberDraft(name="cash", value="1", unit="USD", claims=[], assumption=None)
    with pytest.raises(ValidationError, match="the reason no claim states it"):
        assumed("cash", "1", "USD", "a month of costs", " ")
    with pytest.raises(ValidationError, match="String should match pattern"):  # the definition's ClaimId
        NumberDraft(name="cash", value="1", unit="USD", claims=[StatedBy(id="claim 1", words="$1")], assumption=None)
    with pytest.raises(ValidationError, match="each claim cited gives the words"):
        NumberDraft(name="cash", value="1", unit="USD", claims=[StatedBy(id=f"clm_{'0' * 64}", words="")], assumption=None)
    with pytest.raises(ValidationError, match="averaging exactly one"):
        LiquidityDraft(cash=assumed("c", "1", "USD", "t", "r"), credit_line=assumed("c", "1", "USD", "t", "r"),
                       receivable_days=assumed("d", "1", "days", "t", "r"),
                       payable_days=assumed("d", "1", "days", "t", "r"),
                       seasonality=[assumed("s", "1.1", "factor", "t", "r")] * 12)  # fmt: skip
