"""The wide flow's steps after G1, end to end over the fixture world (decisions, "F4's sample design"): the frame taken
by its proposal's gate, the phase-1 sample drawn and read, both objective questions labelled by the phase-1 judge, and
the gold sample drawn from the judge's probabilities, with the docket's measure bound into the exposure question. The
judges are stand-ins recorded into a fresh cassette directory, as the provider layer's live run records."""

from datetime import UTC, datetime
from fractions import Fraction

import pytest
from fake_api import DOCKET, PROPOSAL_ABSTRACT, PROPOSAL_COST, FakeAPI, cost_reader, fixture_world
from test_frame import criteria, register

from requisite import cassettes, core
from requisite.agents import runner
from requisite.agents.runner import AgentResult
from requisite.discovery import labels, wide
from requisite.evidence.frame import frame_gate
from requisite.judgment import providers
from requisite.library import templates as library
from requisite.store import Refused, Store

T = datetime(2026, 11, 1, 20, 0, tzinfo=UTC)
OBJECTIVE = core.WideObjective(
    id="objective", version=1, population="Small businesses under 500 employees.", events="pending cost measures",
    horizon=core.TimeWindow(start=T, end=None, start_closed=True, end_closed=False), loss_definition="higher costs",
    estimand="share of items", unit="item",
)  # fmt: skip


def stand_in(spec, body):
    """A judge that answers by the filing's submitter: an airline is in the population and exposed through labor."""
    text = str(body)
    name = (body["questions"][0]["name"] if spec.name == "openai_decisions" else next(iter(body["questions"])))
    if "population definition" in text:
        p = {"in_population": 0.8, "other_author": 0.15, "not_established": 0.05}
    elif "part of the author's economics" in text:
        p = {"labor": 0.9, "inputs": 0.02, "energy": 0.02, "financing": 0.02, "revenue": 0.01, "compliance": 0.01,
             "other": 0.01, "not_established": 0.01}  # fmt: skip
    elif "would do to its author" in text:
        p = {"cost_increase": 0.9, "cost_decrease": 0.03, "revenue_loss": 0.03, "not_established": 0.04}
    elif "relief, protection or predictability" in text:
        p = {"asks": 0.85, "does_not": 0.1, "not_established": 0.05}
    else:
        p = {"labor": 0.7, "inputs": 0.05, "labor_and_inputs": 0.05, "no_loss": 0.1, "not_in_class": 0.05,
             "not_established": 0.05}  # fmt: skip
    if spec.name == "openai_decisions":
        answers = [{"type": "choice", "name": name, "probabilities": [{"value": v, "probability": x} for v, x in p.items()]}]
        return {"model": spec.model, "answers": answers}, 0.0
    return {"model": spec.model, "answers": {name: {"type": "choice", "probabilities": p}}}, 0.0


def approve(store: Store, kind: str, payload: dict) -> None:
    p = store.submit_proposal(core.Proposal(id=f"{kind}-{core.sha256(core.canonical_json(payload))[:12]}", case=store.case, kind=kind, payload=payload, author="fixture",
                                            rationale="fixture", status="proposed"))  # fmt: skip
    if p.status != "active":
        wide.approve(store, p.id, gate="G1", approver="fixture", decision="approved", reason="fixture", scope="fixture")


@pytest.fixture
def case(live, make_client, make_fr, scan, monkeypatch):
    monkeypatch.setattr(cassettes, "CASSETTES", live / "cassettes")
    monkeypatch.setattr(providers, "_send", stand_in)
    monkeypatch.setattr(runner, "run", cost_reader)
    api = fixture_world(FakeAPI(), scanned_0003=scan)
    store = Store(live / "case", case="wide")
    approve(store, "objective", OBJECTIVE.model_dump(mode="json"))
    for t in labels.OBJECTIVE_TEMPLATES:
        tpl = library.template(t)
        o = library.original(tpl)
        pset = core.ParaphraseSet(
            id=f"ps.{t}", version=tpl.version, template=core.Ref(id=t, version=tpl.version), slots=list(tpl.slots),
            answers=[a.value for a in tpl.answers],
            paraphrases=[o, *(core.Paraphrase(id=i, target=o.target + suffix, meanings=dict(o.meanings))
                              for i, suffix in (("p1", " Answer from the filing."), ("p2", " Read the whole filing.")))],
        )  # fmt: skip
        approve(store, "paraphrase_set", pset.model_dump(mode="json"))
    fr = register()
    store.register_gate("frame", "tests", frame_gate(lambda: make_client(api, live / "cassettes", live),
                                                     lambda: make_fr(fr, live / "cassettes", live)))  # fmt: skip
    approve(store, "frame", criteria(datetime(2026, 6, 20, 12, tzinfo=UTC), "closed_pending_final").model_dump(mode="json"))
    return store, (lambda: make_client(api, live / "cassettes", live)), (lambda: make_fr(fr, live / "cassettes", live))


def test_the_flow_draws_reads_labels_and_draws_its_gold_by_the_design(case):
    store, client, fr = case
    frame = wide.the_frame(store)
    assert [d.docket for d in frame.dockets if d.kept] == [DOCKET]
    draw = wide.draw_sample(store)
    assert draw.phase == 1 and draw.seed == wide.seed(frame.content_hash())
    n = len(frame.items)
    assert n >= 3 and [(d.stratum, d.inclusion_prob) for d in draw.items] == [("under_100", Fraction(1))] * n  # a census
    items = wide.read_sample(store, client, federal_register=fr)
    assert sorted(i.id for i in items) == sorted(d.item for d in draw.items)
    assert wide.read_sample(store, client, federal_register=fr) == items  # read once; a second run reads nothing again

    # Each item is asked about the document it addresses, with that document's status as of the frame's as_of: the
    # proposal, its final rule (posted October 1) not yet posted by June 20.
    ms = wide.measures(store, [i.id for i in items])
    served = [i for i in items if i.addresses is not None]
    assert served and all(ms[i.id].document == i.addresses == f"{DOCKET}-0001" for i in served)
    statement = ms[served[0].id].statement
    assert f"Proposed Rule {DOCKET}-0001, posted" in statement and "proposed, not final" in statement

    # What the measure does, as its own published text states it: the agency's summary and its statement of who pays
    # what, quoted from the text (found there), with the reader's reading; bound into the exposure question.
    texts = wide.read_measure_texts(store, federal_register=fr)
    mt = texts[f"{DOCKET}-0001"]
    assert mt.summary == PROPOSAL_ABSTRACT and mt.fr_number == "2026-07001" and len(mt.artifacts) == 2
    assert [(c.quote, c.payer, c.direction, str(c.amount), c.scope, c.affected) for c in mt.statements] == [
        (PROPOSAL_COST, "air carriers", "more", "12000000", "all_affected", 64)]
    assert store.get("measure_text", f"{DOCKET}-0001|cost_statements/2", core.MeasureText) == mt
    q = labels.questions(wide.approved_objective(store), ms, texts, served[0])["exposure_to_measure"]
    assert PROPOSAL_ABSTRACT in q.target and PROPOSAL_COST in q.target and q.version == 4
    assert "(all who pay it together)" in q.target and "paid by 64" in q.target  # a total is never one carrier's cost

    done = wide.label_phase1(store)
    readable = len(items) - len(done.unreadable)
    assert len(done.judgments) == 2 * readable
    assert {j.question.id for j in done.judgments} == {"voice_in_population", f"exposure_to_measure.{DOCKET}-0001"}
    assert len(store.all("judgment", core.Judgment)) == len(done.judgments)

    f = wide.predictions(store)
    assert sorted(f) == sorted(i.id for i in items if i.id not in done.unreadable), f
    assert all(abs(v[0] - 0.8) < 1e-9 and abs(v[1] - 0.8) < 1e-9 for v in f.values()), f
    assert not done.unreadable, done.unreadable  # every fixture filing has text (0003 by OCR)
    report = wide.gold_coverage(store)
    # The simulated labels are drawn by a seed of the draw, whose frame is named by its retrieval's time: where they put
    # an item in the denominator, the ratio's coverage is measured; where none (0.2 ** 4 of seeds here), it says so.
    assert report.draws == wide.COVERAGE_DRAWS
    if report.denominator_truth > 0:
        assert [r.numerator for r in report.ratios] == ["exposed"] and report.not_measured is None
    else:
        assert report.ratios == () and report.not_measured
    assert [(g.stratum, g.bin) for g in report.gold if g.expected_gold > 0] == [("under_100", 1)]
    gold = wide.draw_gold(store)
    # Every item's probability, 0.64, is above the cut: the above-cut stratum of 26 holds them all and takes them whole.
    assert gold.phase == 2 and [(d.stratum, d.inclusion_prob) for d in gold.items] == [("under_100.bin1", Fraction(1))] * n
    assert gold.seed == wide.seed(draw.content_hash(), "phase2")
    # The design both draws name is committed whole: the frame it drew from, its strata, the cells and their gold.
    design = store.get("sampling_design", f"{draw.design.id}.{wide.phase1_questions()}", core.SamplingDesign)
    assert design is not None and design.frame_hash == frame.content_hash() and design.seed == draw.seed
    assert [(s.name, s.phase1_count, s.sample_size) for s in design.phase2] == [("under_100.bin1", n, n)]


def agent_double(role, agent_input, output, *, effort, context=None):
    """The family_drafter and impact_extractor answering as a test's own input (never a cassette), validated as the
    runner validates an answer, against the context the caller gives."""
    if role == "family_drafter":
        measures = [m["document"] for m in agent_input["measures"]]
        draft = {"families": [{"id": "airline_labor", "name": "Airline labor", "channel": "labor",
                               "mechanism": "the measure raises the hours or pay airlines must provide",
                               "measures": measures}], "rationale": "one measure, one family"}  # fmt: skip
    else:
        draft = {"impacts": []}
    return AgentResult(role, output.model_validate(draft, context=context), "run-" + core.sha256(core.canonical_json(agent_input)),
                       "test-double", effort, True)


def test_the_flow_goes_from_gold_labels_to_an_estimate_ranked_leads_and_a_lead_proposed_at_g1(case):
    from requisite.discovery import results

    store, client, fr = case
    wide.draw_sample(store)
    wide.read_sample(store, client, federal_register=fr)
    wide.read_measure_texts(store, federal_register=fr)
    wide.label_phase1(store)
    gold = wide.draw_gold(store)
    fams = results.draft_families(store, agent_double)
    assert fams.status == "active", fams.gates
    # Owen's labels: the first gold item unsure on exposure, the rest in the population and exposed.
    for n, d in enumerate(gold.items):
        for tid, label in (("voice_in_population", "in_population"),
                           ("exposure_to_measure", None if n == 0 else "labor")):  # fmt: skip
            store.commit_gold_label(core.GoldLabel(
                item=d.item, question=core.Ref(id=tid, version=library.template(tid).version), design=gold.design,
                stratum=d.stratum,
                inclusion_prob=d.inclusion_prob, outcome="labeled" if label else "unresolved", label=label,
                labeler="owen", labeled_at=T, blind=True,
            ))  # fmt: skip
    est = results.estimate(store)
    assert [s.scope for s in est.scopes] == ["overall", "airline_labor"]
    assert est.unresolved_gold_labels == 1 and est.gold_items == len(gold.items)
    overall = est.scopes[0]
    n = len(gold.items)
    # A census: the lowest completion leaves the unsure item out of the numerator, the highest puts it in.
    assert overall.lowest.estimate == pytest.approx((n - 1) / n) and overall.highest.estimate == pytest.approx(1.0)
    assert overall.read is not None and overall.read[1] == pytest.approx(1.0)
    leads = results.find_leads(store, agent=agent_double)
    # The gold-confirmed items lead (tier 2: they ask for relief, state no number), then the unsure one by the judge.
    assert [x.basis for x in leads.leads] == ["gold"] * (n - 1) + ["judged"]
    assert all(x.tier == 2 and x.answers["seeks_certainty"] == "asks" for x in leads.leads)
    promoted = results.propose_lead(store, 1)
    assert promoted.kind == "lead" and promoted.status != "active"  # it waits for a human at G1


def test_a_phase1_answer_stands_over_a_refusal_of_the_same_question_and_state_and_two_answers_are_refused(tmp_path):
    # Two responses to one question over one state under the phase-1 configuration exist only where the request changed
    # (a state declined as JSON for its size, then read as text: providers.AS_TEXT); the answer stands over the refusal.
    spec = providers.PROVIDERS[labels.PHASE1_CONFIG[0]]
    config = core.JudgeConfig(provider=spec.name, model=spec.model, model_cutoff=spec.model_cutoff,
                              paraphrase_id=labels.PHASE1_CONFIG[1], render_id=labels.PHASE1_CONFIG[2].id)  # fmt: skip
    base = dict(question=core.Ref(id="voice_in_population.x", version=1), state_hash="ab" * 32, config=config,
                precision=0.01, latency_ms=1, cost_usd=0, created_at=datetime(2026, 11, 2, tzinfo=UTC))  # fmt: skip
    refused = core.Judgment(**base, reported=None, distribution=None, refusal="declined as JSON for its size",
                            response="1" * 64)  # fmt: skip
    answered = core.Judgment(**base, reported={"in_population": 0.8, "not_in_population": 0.2},
                             distribution={"in_population": 0.8, "not_in_population": 0.2}, refusal=None,
                             response="2" * 64)  # fmt: skip
    for order in ((refused, answered), (answered, refused)):
        store = Store(tmp_path / str(len(list(tmp_path.iterdir()))), case="w")
        for j in order:
            store.commit_judgment(j)
        assert wide._phase1(store) == {("voice_in_population.x", 1, "ab" * 32): answered}
        assert wide.phase1_judgments(store) == [answered.content_hash()]
    other = answered.model_copy(update={"distribution": {"in_population": 0.1, "not_in_population": 0.9},
                                        "reported": {"in_population": 0.1, "not_in_population": 0.9},
                                        "response": "3" * 64})  # fmt: skip
    store.commit_judgment(other)
    with pytest.raises(Refused, match="two answers"):
        wide._phase1(store)
