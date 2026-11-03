"""Deep discovery past G1 (requisite/discovery/deep.py): the register traced and recomputed (spec §13 WP11a), relevance
and the review queue (WP7; addendum A01 §2.2, A02-A03), diagnosis (spec §2.1; WP4) and the loop's stopping (A02 §3.7).

Every case is seeded (tests/discovery/deep_case.py): the judges answer by a rule over the question and the
configuration, so each instability is the one the test states, and the agent's drafts are the test's own.
"""

from collections import Counter
from decimal import Decimal as D
from fractions import Fraction as F

import deep_case as dc
import pytest

from requisite import core
from requisite.discovery import deep
from requisite.judgment import providers
from requisite.stability import diagnose

PROVIDERS = {"openai_decisions": 0, "clef": 1, "jev": 2}


def answering(overrides):
    """A rule: the seeded world's answers, except where `overrides` (question id -> rule) says otherwise."""

    def rule(a: providers.Ask):
        for qid, r in overrides.items():
            if a.question.id == qid:
                return r(a)
        return dc.base_rule(a)

    return rule


def by_paraphrase(answers):
    """Each paraphrase read one way, by every judge in every render: a vague question."""
    return lambda a: dc.leaning(list(a.asked.answers), answers[a.config.paraphrase_id])


def by_provider(answers):
    return lambda a: dc.leaning(list(a.asked.answers), answers[a.config.provider])


def checkerboard(first, second):
    """Disagreement along no single axis: a judge's answer flips with provider and paraphrase together."""
    return lambda a: dc.leaning(list(a.asked.answers),
                                first if (PROVIDERS[a.config.provider] + int(a.config.paraphrase_id[1])) % 2 == 0
                                else second)  # fmt: skip


def same_top_by_render(top, runner_up):
    """Every configuration's top answer is the same; the text render is near certain of it and the JSON render barely
    prefers it (addendum A01 §2.2's fixture: six at (0.999, 0.001) and six at (0.501, 0.499))."""

    def rule(a: providers.Ask):
        values = list(a.asked.answers)
        hi, lo = ("0.999", "0.001") if a.config.render_id == "text-as-listed" else ("0.501", "0.499")
        return dc.spread(values, {top: hi, runner_up: lo, **{v: "0" for v in values if v not in (top, runner_up)}})

    return rule


# ---------------------------------------------------------------- WP11a


@pytest.mark.covers("WP11a.trace")
def test_every_register_row_traces_to_claims_judgments_and_nodes_held_in_the_store(tmp_path):
    store = dc.make_case(tmp_path, drivers=dc.COMPETING, tolerance="100")
    an, reg = deep.run_register(store, 1, judge=dc.Judges(), agent=dc.Agent(dc.refiner), listing=dc.listing)
    assert {(r.driver, r.event) for r in reg.rows} == {("labor", "E1"), ("materials", "E1")} and not reg.blocked
    vertices = {v.vertex: v for v in store.all("vertex", deep.VertexReceipt)}
    for row in reg.rows:
        assert row in store.all("register_row", core.RegisterRow)
        assert all(store.get("claim", c, core.Claim) is not None for c in row.claims)
        assert all(store.get("judgment", h, core.Judgment) is not None for h in row.judgments)
        assert all(store.get("resolved_judgment", dg.resolved, core.ResolvedJudgment) is not None for dg in row.diagnostics)
        assert store.get("probability_estimate", row.probability, core.ProbabilityEstimate) is not None
        for vid in row.nodes:
            committed = vertices[vid].receipt
            assert store.get("derivation", committed.lineage_hash, core.DerivationReceipt) == committed
        # Every resolution a diagnostic names combines judgments the row lists, so each judgment it rests on is named.
        for dg in row.diagnostics:
            resolved = store.get("resolved_judgment", dg.resolved, core.ResolvedJudgment)
            assert resolved is not None and set(resolved.judgments) <= set(row.judgments)
        # Each number recomputes from committed state by an independent computation: the horizon loss is minus the sum
        # of the dated impacts, the expected impact is the event's committed probability times it, and with that
        # probability above 1 - alpha the CVaR tail is the loss itself.
        pe = store.get("probability_estimate", row.probability, core.ProbabilityEstimate)
        assert pe is not None and pe.value == F(3, 5) and pe.value > 1 - F(9, 10)
        loss = (-sum(m.high for m in row.cash_impact), -sum(m.low for m in row.cash_impact))
        assert row.expected_impact == (pe.value * loss[0], pe.value * loss[1])
        assert row.tail_impact == loss
    labor = next(r for r in reg.rows if r.driver == "labor")
    # Labor: $48,000 a month, 5-15% higher, from between Dec 31 (Nov 1 + one month + 30 days' payable terms) and Jul 30
    # (Mar 31 + three months + 30 days) to the horizon's end on Oct 31: 1 day of December at most, all of Jan-Oct.
    assert labor.cash_impact[1] == core.DatedImpact(month=dc.date(2026, 12, 1), low=F(-7200, 31), high=F(0))
    assert -sum(m.low for m in labor.cash_impact) == F(7200, 31) + 10 * 7200
    assert labor.direction == "increase" and labor.magnitude_band == "5_to_15pct" and labor.mechanism == "cost"
    assert labor.breach == "false" and labor.hedgeable == "true"


@pytest.mark.covers("WP11a.recompute")
def test_a_parameter_change_recomputes_only_what_depends_on_it_and_the_diff_shows_it(tmp_path):
    store = dc.make_case(tmp_path)
    claim = store.all("claim", core.Claim)[0]
    v2 = dc.entity_model(claim, (dc.DriverSpec("labor", "crew wages", "12", "4400"), dc.DRIVERS[1]), version=2)
    dc.approve(store, dc.submit(store, "entity_model", v2))
    judges = dc.Judges()
    diff = deep.compare(store, 1, 2, judge=judges)
    changed = {c.id for c in diff.value_changes()}
    assert changed == {"v:labor.price", "v:labor.amounts", "v:labor.E1.cash", "v:labor.E1.loss", "v:labor.E1.expected",
                       "v:labor.E1.tail"}  # fmt: skip
    # Only labor's computations ran again; fuel's, the horizon's and the event's did not.
    assert set(diff.recomputed) <= {v for v in changed} | {"v:labor.E1.effect", "v:labor.E1.start", "v:labor.E1.breach"}
    assert not any("fuel" in v or v == "v:days" for v in diff.recomputed)
    # A price 10% higher scales labor's loss by exactly 11/10, shown old and new.
    loss = next(c for c in diff.value_changes() if c.id == "v:labor.E1.loss")
    assert loss.new == (loss.old[0] * F(11, 10), loss.old[1] * F(11, 10))
    # Labor's questions state its amounts, so they were asked again: their values held, their justification changed,
    # and the diff shows that apart from the values.
    assert {c.id for c in diff.lineage_changes()} >= {"v:q.deep.sensitivity_magnitude.labor.E1", "v:labor.E1.effect"}
    assert not any("fuel" in c.id for c in diff.lineage_changes())


# ---------------------------------------------------------------- WP7 and addendum A01-A03


@pytest.mark.covers("WP7.informational")
def test_a_question_whose_answers_change_no_top_k_choice_is_informational_out_of_the_queue_and_inspectable(tmp_path):
    # Fuel does not move (its direction is stably "none"), so however its lag is read the top one is labor: fuel's lag
    # is vague, and unstable, and informational for this decision all the same.
    store = dc.make_case(tmp_path, tolerance="100")
    lag = "deep.sensitivity_lag.fuel.E1"
    judges = dc.Judges(answering({lag: by_paraphrase({"p0": "under_1_month", "p1": "over_12_months", "p2": "under_1_month"})}))
    agent = dc.Agent(dc.refiner)
    out = deep.loop(store, 1, judge=judges, agent=agent, listing=dc.listing)
    standing = next(q for q in out.run.questions if q.question.id == lag)
    assert standing.primary_diagnosis == "imprecise_target" and standing.s_arg < diagnose.S_ARG
    assert standing.informational and standing.regret == 0 and standing.action == "none"
    assert "informational for the top-1 decision" in standing.reason and "tolerance 100" in standing.reason
    assert lag not in {r.id for r in out.run.queue}
    assert [role for role, _ in agent.calls] == ["market_scout", "market_scout"]  # nobody was asked to refine it
    assert not [p for p in store.all("proposal", core.Proposal) if p.kind == "question"]
    # Still inspectable: its resolution and every judgment behind it are committed, and the run names the decision it
    # is informational for.
    resolved = store.get("resolved_judgment", standing.resolved, core.ResolvedJudgment)
    assert resolved is not None and len(resolved.judgments) == 18
    assert out.run.decision == core.Ref(id="register.top_1.decision", version=1)
    fuel = next(r for r in out.register.rows if r.driver == "fuel")
    diag = next(d for d in fuel.diagnostics if d.question.id == lag)
    assert diag.informational and diag.decision == out.run.decision and diag.tolerance == D("100")


@pytest.mark.covers("WP7.queue")
def test_the_review_queue_orders_by_relevance_and_instability_with_s_prob_and_ties_by_spread(tmp_path):
    def queue(root, seeded):
        store = dc.make_case(tmp_path / root, drivers=dc.COMPETING, tolerance="100")
        return deep.analyse(deep.open_deep(store), 1, deep.Judge(dc.Judges(answering(seeded)))).queue

    # The providers disagree on labor's lag (S_arg 12/18); every configuration gives labor's direction the same top
    # answer (S_arg = 1) while their probabilities differ materially (S_prob 0.838): it still ranks, below the lag.
    lag, direction = queue("a", {
        "deep.sensitivity_lag.labor.E1": by_provider({"openai_decisions": "1_to_3_months", "clef": "3_to_12_months",
                                                      "jev": "3_to_12_months"}),
        "deep.sensitivity_direction.labor.E1": same_top_by_render("increase", "decrease"),
        "deep.sensitivity_lag.fuel.E1": by_paraphrase({"p0": "under_1_month", "p1": "over_12_months", "p2": "under_1_month"}),
    })  # fmt: skip
    assert (lag.question.id, direction.question.id) == ("deep.sensitivity_lag.labor.E1", "deep.sensitivity_direction.labor.E1")
    assert lag.s_arg == pytest.approx(2 / 3) and lag.priority == pytest.approx(1 - min(lag.s_arg, lag.s_prob))
    assert direction.s_arg == 1 and direction.s_prob < diagnose.S_PROB
    assert lag.priority > direction.priority > 0 and direction.priority == pytest.approx(1 - direction.s_prob)
    # Labor's direction and materials' conditionality are unstable alike: tied in priority, they go by the spread of the
    # decision's objective across their answers, the larger first.
    first, second = queue("b", {
        "deep.sensitivity_direction.labor.E1": same_top_by_render("increase", "decrease"),
        "deep.sensitivity_conditionality.materials.E1": same_top_by_render("unconditional", "conditional"),
    })  # fmt: skip
    assert first.priority == second.priority > 0
    assert (first.question.id, second.question.id) == ("deep.sensitivity_direction.labor.E1",
                                                       "deep.sensitivity_conditionality.materials.E1")  # fmt: skip
    assert first.spread is not None and second.spread is not None and first.spread > second.spread


@pytest.mark.covers("A.T02")
def test_same_top_answer_with_materially_different_probabilities_fails_s_prob_and_keeps_review_priority(tmp_path):
    store = dc.make_case(tmp_path, drivers=dc.COMPETING, tolerance="100")
    qid = "deep.sensitivity_conditionality.labor.E1"
    an = deep.analyse(deep.open_deep(store), 1,
                      deep.Judge(dc.Judges(answering({qid: same_top_by_render("unconditional", "conditional")}))))
    r = an.res.resolved[qid]
    assert r.s_arg == 1 and r.s_prob == pytest.approx(0.838, abs=1e-3) and not r.stable
    assert r.application_state == "review" and r.evidence_status == "not_established"
    entry = next(e for e in an.queue if e.question.id == qid)
    assert entry.priority == pytest.approx(1 - r.s_prob) and entry.priority > 0  # not zero because S_arg = 1


@pytest.mark.covers("A.T03")
def test_a_stable_coarse_band_hiding_a_decision_relevant_distinction_yields_a_witness_and_a_sharper_question(tmp_path):
    # Labor's "5-15%" is stable, and at its two ends a different exposure belongs in the top one: the coarse answer
    # hides a distinction the decision needs.
    store = dc.make_case(tmp_path, drivers=dc.COMPETING, tolerance="100")
    magnitude = "deep.sensitivity_magnitude.labor.E1"
    agent = dc.Agent(dc.refiner)
    out = deep.loop(store, 1, judge=dc.Judges(), agent=agent, listing=dc.listing)
    w = next(w for w in store.all("refinement_witness", core.RefinementWitness) if w.question.id == magnitude)
    assert w.shared_answer == "5_to_15pct" and w.tolerance == D("100")
    assert [list(s) for s in w.acceptable_actions] == [["materials.E1"], ["labor.E1"]]
    # The sharper question the agent proposed passed its gates: stable, and its answers tell the two states apart and
    # move the decision beyond the tolerance (a new material distinction).
    sharper = [p for p in (store.proposal(i) for i in out.run.proposals) if p and p.kind == "question"
               and p.payload["id"].startswith(f"{magnitude}/sharpen")]  # fmt: skip
    admitted = [p for p in sharper if p.status == "active"]
    assert admitted and [g.gate for g in admitted[0].gates] == list(core.governance.SPLIT_GATES)
    assert "tells the witness's states apart" in admitted[0].gates[-1].detail
    replaced = next(q for q in out.run.questions if q.question.id == magnitude)
    assert replaced.action == "replaced" and replaced.replaced_by[0].id == admitted[0].payload["id"]


@pytest.mark.covers("A.T22")
def test_a_spent_budget_freezes_a_reproducible_blocked_draft_never_an_approved_conclusion(tmp_path):
    vague = {"deep.sensitivity_lag.labor.E1": by_paraphrase({"p0": "1_to_3_months", "p1": "3_to_12_months",
                                                             "p2": "1_to_3_months"})}  # fmt: skip

    def run(root):
        store = dc.make_case(tmp_path / root, drivers=dc.COMPETING, tolerance="100")
        return store, deep.loop(store, 1, budget=3, judge=dc.Judges(answering(vague)), agent=dc.Agent(dc.refiner),
                                listing=dc.listing).run  # fmt: skip

    store, first = run("a")
    assert first.status == "blocked" and first.approved is False and first.spent == first.budget == 3
    held = {c.condition: c for c in first.conditions}
    lag = core.Ref(id="deep.sensitivity_lag.labor.E1", version=1)
    assert held["budget_spent"].held and not held["perturbation_failures"].held
    assert lag in held["perturbation_failures"].questions and lag in first.unknowns
    assert first in store.all("requisite_run", deep.RequisiteRun)
    assert "blocked draft, not an approved conclusion" in deep.render_run(first)
    # The searched domain, the tolerance, the unresolved questions and the gates are the draft's own record, and the
    # same case run again with the same budget freezes the same draft.
    assert first.tolerance == D("100") and first.domain and first.searched and first.gates
    _, again = run("b")
    assert again == first


# ---------------------------------------------------------------- diagnosis (spec §2.1; WP4)


@pytest.mark.covers("DX.base")
def test_each_unstable_question_gets_one_primary_diagnosis_and_its_action_with_every_probe_kept(tmp_path):
    store = dc.make_case(tmp_path, drivers=dc.COMPETING, tolerance="100")
    vague, contested = "deep.sensitivity_lag.labor.E1", "deep.sensitivity_direction.materials.E1"
    compound, unexplained = "deep.sensitivity_magnitude.labor.E1", "deep.sensitivity_magnitude.materials.E1"
    seeded = {
        vague: by_paraphrase({"p0": "1_to_3_months", "p1": "3_to_12_months", "p2": "1_to_3_months"}),
        contested: by_provider({"openai_decisions": "increase", "clef": "none", "jev": "none"}),
        compound: checkerboard("5_to_15pct", "15_to_40pct"),
        unexplained: checkerboard("15_to_40pct", "5_to_15pct"),
    }

    def subs(a):  # the agent's sub-questions: labor's split holds, materials' does not
        values = list(a.asked.answers)
        if a.question.id.startswith(f"{unexplained}/split"):
            return checkerboard(values[0], values[1])(a)
        return dc.base_rule(a)

    rule = answering(seeded)
    judges = dc.Judges(lambda a: subs(a) if "/split" in a.question.id else rule(a))
    out = deep.loop(store, 1, judge=judges, agent=dc.Agent(dc.refiner), listing=dc.listing)
    res = out.analysis.res
    dx = {q: res.resolved[q].primary_diagnosis for q in (vague, contested, compound, unexplained)}
    assert dx == {vague: "imprecise_target", contested: "contested", compound: "compound", unexplained: "unexplained"}
    standing = {q.question.id: q for q in out.run.questions}
    # Each diagnosis took its prescribed action (spec §2.1's table).
    assert standing[vague].action == "replaced" and "/refine1.1" in standing[vague].replaced_by[0].id
    assert standing[contested].action == "human" and "G2" in standing[contested].reason
    assert f"G2 on {contested}: no approval covers its current lineage" in out.run.outstanding_gates
    r = res.resolved[contested]
    assert (r.evidence_status, r.application_state) == ("contested", "review")
    assert standing[compound].action == "replaced" and len(standing[compound].replaced_by) == 2
    assert standing[unexplained].action == "human" and "saved as question_fixture" in standing[unexplained].reason
    assert store.all("question_fixture", core.AtomicQuestion)[0].id == unexplained
    assert {k: diagnose.ACTIONS[k][0] for k in ("imprecise_target", "contested", "compound", "unexplained")} == {
        "imprecise_target": "refine", "contested": "human", "compound": "split", "unexplained": "human"}
    # Every probe's evidence is kept beside the primary diagnosis (A01 §2.4): the base run's split by paraphrase and
    # by provider for each, and the split probe's sub-question scores where a split was tried.
    for q in dx:
        base = res.resolved[q].probes[0]
        assert base.probe == "paraphrase_provider" and len(base.judgments) == 18
        assert {k.split(".")[0] for k in base.scores} >= {"paraphrase:p0", "provider:clef", "s_arg"}
    for q, fired in ((compound, True), (unexplained, False)):
        (split,) = [p for p in res.resolved[q].probes if p.probe == "split"]
        assert split.fired is fired and "composed.s_arg" in split.scores


@pytest.mark.covers("WP4.imprecise_target")
def test_a_vague_question_yields_imprecise_target(tmp_path):
    store = dc.make_case(tmp_path, drivers=dc.COMPETING, tolerance="100")
    qid = "deep.sensitivity_lag.labor.E1"
    rule = answering({qid: by_paraphrase({"p0": "1_to_3_months", "p1": "3_to_12_months", "p2": "1_to_3_months"})})
    case = deep.open_deep(store)
    res = deep.resolve_case(case, deep.Judge(dc.Judges(rule)))
    r = res.resolved[qid]
    assert not r.stable and r.primary_diagnosis == "imprecise_target"
    (base,) = r.probes
    assert base.fired and base.indicates == "imprecise_target"
    assert all(base.scores[f"paraphrase:{p}.s_arg"] == 1 for p in ("p0", "p1", "p2"))  # each wording agrees with itself
    assert all(base.scores[f"provider:{p}.s_arg"] < diagnose.S_ARG for p in PROVIDERS)  # no judge agrees across them
    assert r.application_state == "review" and r.evidence_status == "not_established"


@pytest.mark.covers("WP4.compound_split")
def test_splitting_a_compound_question_raises_stability(tmp_path):
    store = dc.make_case(tmp_path, drivers=dc.COMPETING, tolerance="100")
    qid = "deep.sensitivity_magnitude.labor.E1"
    judges = dc.Judges(answering({qid: checkerboard("5_to_15pct", "15_to_40pct")}))
    out = deep.loop(store, 1, judge=judges, agent=dc.Agent(dc.refiner), listing=dc.listing)
    res = out.analysis.res
    parent = res.resolved[qid]
    subs = res.replaced_by[qid]
    assert parent.primary_diagnosis == "compound" and len(subs) == 2
    (split,) = [p for p in parent.probes if p.probe == "split"]
    assert parent.s_arg < diagnose.S_ARG and split.scores["composed.s_arg"] == 1 > parent.s_arg
    assert all(res.resolved[s.id].stable for s in subs)
    # The split was admitted through the split gates, and the composition is in the graph the register reads.
    for s in subs:
        p = store.proposal(deep._proposal_of(store, s.id) or "")
        assert p is not None and p.status == "active" and [g.gate for g in p.gates] == list(core.governance.SPLIT_GATES)
    labor = next(r for r in out.register.rows if r.driver == "labor")
    assert any(n.startswith(f"v:split.{qid}") for n in labor.nodes)
    assert Counter(type(v).__name__ for v in out.analysis.graph if v.id.startswith(f"v:split.{qid}")) == {"Instance": 2}


# ---------------------------------------------------------------- what the register does not claim


def test_an_exposure_whose_loss_the_evidence_does_not_bound_is_listed_apart_and_goes_to_a_human(tmp_path):
    # The judges agree, stably, that the evidence does not establish labor's magnitude: its band is any band, open
    # above, so its loss has no bound. It is never given numbers; it is forced into the top one (an exposure that may
    # be the largest cannot be left out), and the question goes to a human as an application-blocking unknown.
    store = dc.make_case(tmp_path, drivers=dc.COMPETING, tolerance="100")
    qid = "deep.sensitivity_magnitude.labor.E1"
    judges = dc.Judges(answering({qid: lambda a: dc.leaning(list(a.asked.answers), "not_established")}))
    out = deep.loop(store, 1, judge=judges, agent=dc.Agent(dc.refiner), listing=dc.listing)
    assert [r.driver for r in out.register.rows] == ["materials"]
    assert "labor.E1" in out.run.blocked_exposures and qid in out.run.blocked_exposures["labor.E1"]
    assert out.run.top_k == (("labor.E1",),)
    standing = next(q for q in out.run.questions if q.question.id == qid)
    assert standing.primary_diagnosis == "stable" and standing.application_state == "not_established"
    assert standing.action == "human" and "does not establish it" in standing.reason
    assert qid in [r.id for r in out.run.queue] and core.Ref(id=qid, version=1) in out.run.unknowns
    # A02 §3.7: the run may stop with the unknown identified, never read as agreement; the designation is bounded by it.
    unknowns = next(c for c in out.run.conditions if c.condition == "blocking_unknowns")
    assert unknowns.held and core.Ref(id=qid, version=1) in unknowns.questions
    assert out.run.status == "local_requisite"


def test_the_market_screen_asks_about_listed_markets_the_scout_finds_and_states_one_not_listed(tmp_path):
    from requisite.agents.answers import CategoriesDraft, MarketsDraft

    def scout(role, input):
        if "categories" in input:
            assert {c["category"] for c in input["categories"]} == {"Politics", "Sports"}
            return CategoriesDraft(categories=["Politics", "Weather"], rationale="r")
        assert [s["ticker"] for s in input["series"]] == ["KXEXAMPLE"]  # only the categories that exist
        return MarketsDraft(tickers=["KXEXAMPLE", "KXMADEUP"], rationale="r")

    def rule(a):
        if a.question.template_id == "observable_tracks_loss" and a.question.id.endswith(".kalshi.KXEXAMPLE"):
            return dc.leaning(list(a.asked.answers), "does_not_track")
        return dc.base_rule(a)

    store = dc.make_case(tmp_path)
    case, scr = deep.run_screen(store, judge=dc.Judges(rule), agent=dc.Agent(scout), listing=dc.listing)
    s = scr.scouted["E1"]
    assert [o.market for o in s.markets] == ["KXEXAMPLE"] and s.not_listed == ("KXMADEUP",) and s.categories == ("Politics",)
    asked = {q.id for q in scr.questions}
    assert "deep.observable_tracks_loss.labor.E1.kalshi.KXEXAMPLE" in asked and not any("KXMADEUP" in q for q in asked)
    # The event's own record tracks the loss and the business cannot move it; the listed market does not track it.
    assert scr.hedgeable["labor.E1"].value == "true" and scr.found_market_tracks["labor.E1"].value == "false"
    assert "named but not in the listing, so not asked about: KXMADEUP" in deep.render_screen(case, scr)


# ---------------------------------------------------------------- the stopping contract (addendum A02 §3.7)


def test_the_run_records_which_stopping_conditions_held_and_the_refinement_set_it_searched(tmp_path):
    store = dc.make_case(tmp_path, drivers=dc.COMPETING, tolerance="100")
    out = deep.loop(store, 1, judge=dc.Judges(), agent=dc.Agent(dc.refiner), listing=dc.listing)
    run = out.run
    assert [c.condition for c in run.conditions] == ["perturbation_failures", "blocking_unknowns",
                                                     "decision_sufficiency", "approvals", "budget_spent"]  # fmt: skip
    held = {c.condition: c.held for c in run.conditions}
    assert held == {"perturbation_failures": True, "blocking_unknowns": True, "decision_sufficiency": True,
                    "approvals": True, "budget_spent": False} and run.status == "local_requisite"  # fmt: skip
    # The searched set: for labor's 5-15% band, the band's ends were tried and yielded the witness the sharper question
    # answered; the library holds no finer band and the question no definitions, each said, not tried.
    first = next(x for x in run.searched if x.question.id == "deep.sensitivity_magnitude.labor.E1")
    by_source = {c.source: c for c in first.candidates}
    assert by_source["answer_meaning"].tried and by_source["answer_meaning"].witness in run.witnesses + tuple(
        w.content_hash() for w in store.all("refinement_witness", core.RefinementWitness))
    assert not by_source["library_template"].tried and not by_source["definition"].tried
    # The event's forecast is searched for when within its window the event happens (a loss mechanism).
    forecast = next(x for x in run.searched if x.question.id == "deep.event_occurs.E1")
    assert [c.source for c in forecast.candidates if c.tried] == ["loss_mechanism"]
    # G1 covers the entity model's and the decision spec's payloads as they stand.
    assert [(g.gate, g.covers) for g in run.gates] == [("G1", True), ("G1", True)]


def test_a_contested_relevant_question_awaits_g2_and_its_approval_is_recorded_beside_its_resolution(tmp_path, monkeypatch):
    from requisite import cassettes, secrets

    monkeypatch.delenv("REQUISITE_REPLAY_ONLY", raising=False)
    monkeypatch.setattr(secrets, "known_values", lambda: [])
    monkeypatch.setattr(cassettes, "CASSETTES", tmp_path / "cassettes")
    monkeypatch.setattr(cassettes, "SPEND_LOG", tmp_path / "spend.jsonl")
    store = dc.make_case(tmp_path, drivers=dc.COMPETING, tolerance="100")
    qid = "deep.sensitivity_lag.labor.E1"
    judges = dc.Judges(answering({qid: by_provider({"openai_decisions": "1_to_3_months", "clef": "3_to_12_months",
                                                    "jev": "3_to_12_months"})}))  # fmt: skip
    before = deep.loop(store, 1, judge=judges, agent=dc.Agent(dc.refiner), listing=dc.listing).run
    assert before.status == "awaiting_approval"
    (g2,) = [g for g in before.gates if g.gate == "G2"]
    assert g2.subject == qid and not g2.covers and g2.approval is None
    with pytest.raises(deep.Refused, match="G2 decides a contested question"):
        deep.approve_question(store, "deep.sensitivity_lag.materials.E1", approver="owen", decision="approved",
                              reason="r", scope="s", judge=judges)  # fmt: skip
    a = deep.approve_question(store, qid, approver="owen", decision="approved", scope="this case",
                              reason="the rule's effective date is unsettled; hold the lag open", judge=judges)
    assert a.gate == "G2" and a.lineage_hash == g2.lineage
    after = deep.loop(store, 1, judge=judges, agent=dc.Agent(dc.refiner), listing=dc.listing)
    (g2,) = [g for g in after.run.gates if g.gate == "G2"]
    assert g2.covers and g2.approval == a.id and after.run.status == "local_requisite"
    r = after.analysis.res.resolved[qid]
    assert r.approvals == (a.id,) and (r.evidence_status, r.application_state) == ("contested", "review")
    assert core.Ref(id=qid, version=1) in after.run.unknowns  # still held as not established
