"""One valid instance of every canonical record (requisite/core), used to show each round-trips. A record added to core
without an instance here fails tests/core/test_roundtrip.py, so none is left unexercised."""

from datetime import UTC, date, datetime
from decimal import Decimal as D
from fractions import Fraction as F

from requisite import core as c
from requisite.core.evidence import artifact_id, claim_id
from requisite.core.governance import FloorPoint, InputReceipt
from requisite.core.results import HARD_GATES, HardGate
from requisite.core.world import ConditioningStep

T0 = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
T1 = datetime(2026, 12, 31, 23, 59, tzinfo=UTC)
H = "0" * 64

ART = artifact_id(b"the docket comment text")
TEXT = "the docket comment text"
CLM = claim_id(ART, "plain-v1", (4, 10), TEXT[4:10], "a commenter")
SYN_ART = "syn_art_" + "1" * 64
REF = c.Ref(id="q.cover", version=1)
WINDOW = c.TimeWindow(start=T0, end=T1, start_closed=True, end_closed=False, tz="America/New_York")

ANSWERS = [
    c.AnswerValue(value="yes", meaning="the provision covers it", status="affirmative", binds={"P": True}),
    c.AnswerValue(value="no", meaning="it does not", status="negative", binds={"P": False}),
    c.AnswerValue(value="ne", meaning="the evidence does not establish it", status="not_established"),
]
QUESTION = c.AtomicQuestion(id="q.cover", version=1, task="interpret", target="Does X cover Y in 2027?", answers=ANSWERS)
CONFIG = c.JudgeConfig(
    provider="clef", model="@cf/cloudflare/clef", model_cutoff="unknown", paraphrase_id="p1", render_id="text"
)
JUDGMENT = c.Judgment(
    question=REF,
    state_hash=H,
    config=CONFIG,
    reported={"yes": 0.7, "no": 0.2, "ne": 0.1},
    distribution={"yes": 0.7, "no": 0.2, "ne": 0.1},
    precision=0.0001,
    response=H,
    latency_ms=120,
    cost_usd=D("0.0004"),
    created_at=T0,
)
SOURCE = c.Source(id="noaa", name="NOAA GHCN-Daily", rank=1, role="statistical_release")
RAIN = c.Observable(
    id="rain",
    description="daily precipitation at station S",
    value_type="number",
    sources=[SOURCE],
    revision_policy="first_print",
    unit="inch",
    precision=D("0.01"),
)
WET = c.Cmp(id="wet", observable="rain", op=">=", value=c.Number(value=D("0.10")))
COUNT = c.CountTrue(id="wet_days", expr=WET, days=WINDOW)
TRIGGER = c.NumCmp(id="trigger", expr=COUNT, op=">=", value=D("20"))
PROCEDURE = c.ResolutionProcedure(
    precedence=["noaa"],
    conflict_policy=c.ConflictPolicy(procedure="precedence_first_available"),
    on_unknown_at_expiry="extend",
    early_expiration="none",
    extension_days=10,
    after_extension="no",
    void_transfer=c.VoidTransfer(returns="premium"),
)
TIMING = c.Timing(
    open=T0, close=T1, expiration=T1, observation_cutoff=T1, settlement_lag_days=3, payment_lag_days=2,
    lag_days="calendar", holiday_calendar=None, tz="America/New_York",
)
PAYOUT = c.Payout(
    kind="per_unit",
    unit_value=D("100"),
    quantity=COUNT,
    currency="USD",
    rounding_quantum=D("0.01"),
    rounding_mode="half_even",
    payment="at_settlement",
)
CONTRACT = c.ContractIR(
    id="wet-spring",
    version=1,
    observables=[RAIN],
    trigger=TRIGGER,
    payout=PAYOUT,
    timing=TIMING,
    resolution=PROCEDURE,
    exclusions=[c.Exclusion(text="Not advice.", operative=False)],
)
BINDING = c.ListingBinding(field="floor_strike", node="trigger", slot="threshold")
SETTLEMENT = c.SettlementBinding(observable="rain", at_field="close_time")
SUMMARY = c.CashSummary(
    quantities={"y": D("0.1818181818")},
    premium=D("1.818181818"),
    cash=[[D("98.18"), D("98.18")], [D("98.18"), D("0")]],
    deficit=[D("0"), D("0")],
    breach_tolerance=D("1E-7"),
    breach_dates=[[], []],
    loss=[D("0"), D("120")],
    receipts=[D("0"), D("21.8")],
    residual=[D("1.8"), D("100")],
    cvar=F(0),
    expected_liquidity_shortfall=F(0),
    breach_probability=F(0),
)


PARAM = c.Parameter(name="crew hours", value=D("2000"), unit="hours/month", claims=[CLM])
PRICE = c.Parameter(name="wage", value=D("22.50"), unit="USD/hour",
                    assumption=c.Assumption(text="the posted wage", binds={}, reason="not in the filing"))  # fmt: skip
FLAT = c.Parameter(name="seasonality", value=D("1"), unit="factor",
                   assumption=c.Assumption(text="no seasonality", binds={}, reason="the filing gives none"))  # fmt: skip
LIQ = c.LiquidityProfile(cash=PARAM, credit_line=PARAM, receivable_days=PARAM, payable_days=PARAM,
                         seasonality=[FLAT] * 12)  # fmt: skip


IMPACT = c.DatedImpact(month=date(2027, 1, 1), low=F(100), high=F(400))
DIAG = c.QuestionDiagnostic(question=REF, resolved=H, s_arg=1.0, s_prob=0.93, primary_diagnosis="stable",
                            decision=c.Ref(id="decision", version=1), tolerance=D("10"), assumptions=["em v1"],
                            regret=F(50), informational=False)  # fmt: skip


FAMILY = c.EventFamily(id="overtime", name="Overtime pay", channel="labor", mechanism="a higher overtime threshold",
                       measures=["DOCKET-1-0001"])  # fmt: skip
READ = c.RatioRead(estimate=0.4, interval=[(0.2, 0.6)], interval_kind="interval", fieller=[(None, -3.0), (0.18, None)],
                   fieller_kind="two_rays")  # fmt: skip
IMPACT_Q = c.QuantifiedImpact(claim=CLM, amount=F(80000), currency="USD", period="year", annual=F(80000))
LEAD = c.Lead(rank=1, tier=1, item="COMMENT-1", docket="DOCKET-1", measure="DOCKET-1-0001", family="overtime", submitter="A landscaping company",
              basis="judged", probability=0.72, answers={"seeks_certainty": "asks"}, judgments=[H], impact=[IMPACT_Q])  # fmt: skip


def examples() -> list[c.Record]:
    return [
        c.Ref(id="x", version=1),
        WINDOW,
        *ANSWERS,
        c.Definition(id="d.cover", version=1, term="covered", text="...", source="statute §2"),
        QUESTION,
        c.Assumption(text="Relief is granted.", binds={"relief": True}, reason="conditioning for P(B | A)"),
        c.JudgmentState(as_of=T0, observed=[CLM]),
        c.SyntheticJudgmentState(as_of=T0, observed=["syn_clm_" + "2" * 64], scenario="s1"),
        CONFIG,
        JUDGMENT,
        c.QuestionTemplate(id="t.cover", version=1, task="interpret", target="Does {provision} cover {party} in 2027?",
                           slots=["provision", "party"], answers=ANSWERS, paraphrase_set="ps.cover"),  # fmt: skip
        c.Paraphrase(id="p0", target="Does {provision} cover {party} in 2027?",
                     meanings={"yes": "covered", "no": "not covered", "ne": "unsettled"}),  # fmt: skip
        c.ParaphraseSet(
            id="ps.cover", version=1, template=c.Ref(id="t.cover", version=1), slots=["provision", "party"],
            answers=["yes", "no", "ne"],
            paraphrases=[c.Paraphrase(id="p0", target="Does {provision} cover {party} in 2027?",
                                      meanings={"yes": "covered", "no": "not covered", "ne": "unsettled"})],
        ),  # fmt: skip
        c.ProbeResult(probe="paraphrase_provider", fired=False, indicates=None, scores={"s_arg": 1.0}, detail="-"),
        c.ResolvedJudgment(
            question=REF,
            state_hash=H,
            judgments=[H, "1" * 64],
            pooled={"yes": 0.7, "no": 0.2, "ne": 0.1},
            pooling_policy=c.Ref(id="equal_per_provider_split_among_configurations", version=1),
            s_arg=1.0,
            s_prob=0.97,
            tie_policy=c.Ref(id="top_set_equality", version=1),
            thresholds=c.Ref(id="spec-2.1-start", version=1),
            stable=True,
            primary_diagnosis="stable",
            probes=[],
            evidence_status="established",
            application_policy=c.Ref(id="approval-required", version=1),
            application_state="review",
        ),
        c.Artifact(
            id=ART,
            locator="https://api.regulations.gov/v4/comments/X",
            source="regulations.gov",
            media_type="text/plain",
            size=len(TEXT),
            published_at=T0,
            published_at_basis="postedDate",
            observed_at=T0,
            availability="retrieved",
        ),
        c.Item(id="COMMENT-1", source="regulations.gov", container="DOCKET-1", addresses="DOCKET-1-0001", posted_at=T0,
               submitter="A Company", text=TEXT, parts=[
                   c.ItemPart(label="the comment", locator="https://api.regulations.gov/v4/comments/COMMENT-1",
                              artifact=ART, extraction="rg-comment-text-v1", unreadable=None, by=None),
                   c.ItemPart(label="attachment 1", locator="https://downloads.regulations.gov/x.wpd", artifact=ART,
                              extraction=None, unreadable=".wpd: no reader here, and no PDF rendition is offered",
                              by="reader")]),  # fmt: skip
        c.ItemPart(label="attachment 2", locator=None, artifact=None, extraction=None,
                   unreadable="restricted by the agency; no file is published", by="source"),
        c.Claim(id=CLM, artifact=ART, extraction="plain-v1", span=(4, 10), text=TEXT[4:10], authority="a commenter"),
        c.SyntheticArtifact(id=SYN_ART, scenario="s1", generator="run-1", media_type="text/plain", size=5,
                            published_at=T0, created_at=T0),  # fmt: skip
        c.RulebookSection(name="trigger", text="Pays if rainfall reaches 0.10 inches.", ir_nodes=["rain_ge_0.10"]),
        c.Rulebook(contract=c.Ref(id="wet-spring", version=1), sections=[
            c.RulebookSection(name="underlying_and_definitions", text="Rainfall at the station.", ir_nodes=["rain"]),
            c.RulebookSection(name="source_hierarchy_and_conflict_rule", text="NOAA, then the station.", ir_nodes=[]),
            c.RulebookSection(name="trigger", text="Pays if rainfall reaches 0.10 inches.", ir_nodes=["rain_ge_0.10"]),
            c.RulebookSection(name="examples", text="", ir_nodes=[]),
            c.RulebookSection(name="timing", text="Expires June 1.", ir_nodes=[]),
            c.RulebookSection(name="unknown_at_expiry", text="Resolves No.", ir_nodes=[]),
            c.RulebookSection(name="revisions", text="The first print settles.", ir_nodes=[]),
        ]),  # fmt: skip
        c.ScenarioPacket(contract=c.Ref(id="wet-spring", version=1), scenario="s1", state_hash=H, evidence_at=T0,
                         documents=[SYN_ART]),  # fmt: skip
        c.SyntheticClaim(
            id="syn_clm_" + c.sha256(f"{SYN_ART}|0:5|hello"), artifact=SYN_ART, span=(0, 5), text="hello", authority="x"
        ),
        c.Proposition(id="P", text="X covers Y", domain="bool"),
        c.Constraint(id="k", kind="implies", members=["A", "B"]),
        ConditioningStep(proposition="A", value="true"),
        c.ProbabilityEstimate(
            id="pe1",
            proposition="relief",
            branch_value="granted",
            conditioning=[ConditioningStep(proposition="A", value="true")],
            meaning="physical",
            source="judgment",
            source_ref=H,
            value=F(1, 3),
            interval=(F(1, 4), F(1, 2)),
            evidence_hash=H,
            as_of=T0,
        ),
        c.Branch(proposition="relief", value="granted", probability="pe1"),
        c.BranchPoint(
            id="bp",
            branches=[
                c.Branch(proposition="relief", value="granted", probability="pe1"),
                c.Branch(proposition="relief", value="denied", probability="pe2"),
            ],
        ),
        c.EventTree(
            id="tree",
            version=1,
            root=c.BranchPoint(
                id="root",
                branches=[
                    c.Branch(proposition="relief", value="granted", probability="pe1"),
                    c.Branch(proposition="relief", value="denied", probability="pe2"),
                ],
            ),
        ),
        SOURCE,
        RAIN,
        c.Observation(observable="rain", source="noaa", at=T0, value=c.Number(value=D("0.12")), published_at=T0,
                      final=False, artifact=ART),  # fmt: skip
        c.SyntheticObservation(observable="rain", source="noaa", at=T0, value=c.Number(value=D("0.12")),
                               published_at=T0, final=False, artifact=SYN_ART),  # fmt: skip
        WET,
        c.And(id="a", args=[WET, c.Not(id="n", arg=WET)]),
        c.Or(id="o", args=[WET, WET]),
        c.Not(id="n2", arg=WET),
        c.By(id="by", expr=WET, since=None, since_closed=None, deadline=T1, closed=True),
        c.At(id="at", expr=WET, at=T1),
        c.Throughout(id="th", expr=WET, window=WINDOW),
        c.Interpretive(id="relief", question_template="qt.relief", bindings={"rule": "X"}),
        COUNT,
        c.Number(value=D("1.5")),
        c.Category(value="granted"),
        c.Flag(value=True),
        c.OnDay(value=date(2027, 4, 1)),
        c.DurationIn(id="dur", expr=WET, window=WINDOW, unit="calendar_day"),
        c.ReadValue(id="wage", question_template="qt.monthly_wage", bindings={"instrument": "X"}, unit="USD/month",
                    bands=[(D("0"), D("3000")), (D("3000"), D("6000"))]),  # fmt: skip
        c.ValueAt(id="minwage", observable="rain", at=T1),
        c.Computed(id="net_cost", node=c.Ref(id="net_monthly_cost", version=1),
                   inputs={"wage": c.ValueAt(id="minwage", observable="rain", at=T1)}),  # fmt: skip
        TRIGGER,
        PAYOUT,
        TIMING,
        c.ConflictPolicy(procedure="require_agreement", sources=["noaa", "station-log"]),
        c.VoidTransfer(returns="fixed_amount", amount=D("5")),
        PROCEDURE,
        c.Exclusion(text="Excludes declared emergencies.", operative=True, ir_node="wet"),
        c.ScenarioExample(scenario_id="s1", description="20 wet days", expected="resolved_yes", evaluation=H),
        CONTRACT,
        c.CashTransfer(at=T1, amount=D("100"), currency="USD", basis="payout"),
        c.ResolutionEvaluation(
            contract=c.Ref(id="wet-spring", version=1),
            evidence_state_hash=H,
            trigger_truth=c.Kleene.UNKNOWN,
            procedure_status="resolved_no",
            quantity=(D("18"), D("23")),
            transfers=[],
            policies=[c.Ref(id="on_unknown_no", version=1)],
        ),
        c.HarvestedContract(
            filing="filing-1",
            filed_terms=[ART],
            support="supported_with_explicit_assumptions",
            ir=c.Ref(id="wet-spring", version=1),
            assumptions=["first-print values"],
            equivalence="tested_equivalent_on_suite",
        ),
        c.FiledSeries(series="KXWET", as_of=T0, why="drawn for greater", record=ART, contract=None, terms=ART,
                      markets={"KXWET-26DEC31-T20": ART}),  # fmt: skip
        BINDING,
        SETTLEMENT,
        c.ContractFamily(
            contract=c.HarvestedContract(filing="KXWET|2026-11-01T12:00:00+00:00", filed_terms=[ART],
                                         support="supported_exact", ir=CONTRACT.ref,
                                         equivalence="tested_equivalent_on_suite"),
            representative=CONTRACT,
            bindings=[BINDING, c.ListingBinding(field="close_time", node="wet_days", slot="window_end"),
                      c.ListingBinding(field="expiration_time", node=None, slot="expiration")],
            settlement=SETTLEMENT,
            markets=[CONTRACT],
        ),  # fmt: skip
        c.GateResult(gate="schema", passed=True, detail="ok"),
        c.Proposal(id="pr1", case="case1", kind="question", payload={"id": "q"}, author="run-1", rationale="r",
                   status="proposed"),  # fmt: skip
        c.Approval(
            id="ap1",
            gate="G1",
            subject=c.Ref(id="decision", version=1),
            lineage_hash=H,
            approver="owen",
            at=T0,
            decision="approved",
            reason="as drafted",
            scope="case1",
        ),
        c.Override(subject=REF, field="application_state", old="review", new="yes", approver="owen", at=T0,
                   reason="read the order"),  # fmt: skip
        c.AcceptedRisk(scenario="s2", amount=D("50"), horizon=WINDOW, description="a temporary cost"),
        c.AcceptanceRecord(
            case="case1",
            decided_at=T0,
            objective=c.Ref(id="decision", version=1),
            decision="buy",
            chosen="y",
            alternatives=["none"],
            assessed_domain="the scenario set of case1 v3",
            price_policy=c.Ref(id="indicative", version=1),
            scenario_probabilities={"s1": F(1, 2), "s2": F(1, 2)},
            accepted_uncovered=[],
            outstanding_uncertainties=[],
            budget=D("20"),
            approvals=["ap1"],
            approval_scope="case1",
        ),
        FloorPoint(day=date(2027, 1, 1), amount=D("0")),
        c.DecisionSpec(
            id="decision",
            version=1,
            entity="a company",
            decision_time=T0,
            horizon=WINDOW,
            loss_definition="incremental cash cost",
            actions=["buy", "none"],
            objective="liquidity_cvar",
            alpha=F(9, 10),
            tolerance=D("100"),
            liquidity_floor=[FloorPoint(day=date(2027, 1, 1), amount=D("0"))],
            currency="USD",
            information=[],
            recourse="none",
        ),
        c.RefinementWitness(
            question=REF, decision=c.Ref(id="decision", version=1), tolerance=D("10"), shared_answer="no", states=["none covered", "half covered"],
            acceptable_actions=[["none"], ["buy"]], value_discrepancy=D("90"), nodes=["n1"],
        ),  # fmt: skip
        c.Stratum(name="all", frame_size=10, sample_size=4),
        c.SubStratum(name="all.pos", parent="all", rule="f >= 0.5", phase1_count=2, sample_size=2),
        c.SamplingDesign(
            id="design1",
            version=1,
            frame=c.Ref(id="frame1", version=1),
            frame_hash=H,
            frame_size=10,
            unit="item",
            estimand="share of items in a business's voice reporting labor-cost exposure",
            filter_policy="none",
            dedup_policy="none",
            sample="two_phase_stratified_srswor",
            phase1=[c.Stratum(name="all", frame_size=10, sample_size=4)],
            phase2=[
                c.SubStratum(name="all.pos", parent="all", rule="f >= 0.5", phase1_count=2, sample_size=2),
                c.SubStratum(name="all.neg", parent="all", rule="f < 0.5", phase1_count=2, sample_size=2),
            ],
            allocation=c.Ref(id="min2", version=1),
            seed=7,
            estimator=c.Ref(id="two_phase_difference", version=1),
            predictor=[c.Ref(id="t.exposure", version=1)],
            gold_policy="single labeller; unresolved kept as its own category",
        ),
        c.DrawnItem(item="i1", stratum="all", inclusion_prob=F(2, 5)),
        c.SampleDraw(design=c.Ref(id="design1", version=1), phase=2, seed=7,
                     items=[c.DrawnItem(item="i1", stratum="all.pos", inclusion_prob=F(1))]),  # fmt: skip
        c.GoldLabel(
            item="i1",
            question=REF,
            design=c.Ref(id="design1", version=1),
            stratum="all.pos",
            inclusion_prob=F(2, 5),
            outcome="unresolved",
            label=None,
            labeler="owen",
            labeled_at=T0,
            blind=True,
        ),
        c.WideObjective(
            id="obj", version=1, population="U.S. businesses", events="pending federal rules changing labor or input costs",
            horizon=WINDOW, loss_definition="incremental operating cash cost",
            estimand="share of comments in a business's voice describing exposure, by event family", unit="item",
        ),  # fmt: skip
        c.FrameCriteria(
            id="frame", version=1, objective=c.Ref(id="obj", version=1), source="regulations.gov", as_of=T0,
            search_terms=["wage"], document_types=["Proposed Rule"], posted_from=date(2026, 1, 1), posted_to=None,
            agencies=[], comment_status="either", min_comments=50,
        ),  # fmt: skip
        c.FrameSearch(term="wage", request="documents?filter[searchTerm]=wage", documents=3, artifacts=[ART]),
        c.FrameDocket(docket="DOCKET-1", kept=True, reason="open; 120 comments", comments=120, artifacts=[ART]),
        c.FrameItem(id="COMMENT-1", docket="DOCKET-1", posted_at=T0, withdrawn=False),
        c.ItemMeasure(item="COMMENT-1", document="DOCKET-1-0001",
                      statement='"Overtime threshold" (WHD Proposed Rule DOCKET-1-0001, posted 2026-04-21). Its status as '
                                "of 2026-11-01: proposed, not final: no final rule follows it in its docket by as_of.",
                      artifacts=[ART]),  # fmt: skip
        c.Frame(id="frame1", criteria=c.Ref(id="frame", version=1), as_of=T0, retrieval_started=T0, retrieval_ended=T0,
                mode="prospective",
                searches=[c.FrameSearch(term="wage", request="documents?filter[searchTerm]=wage", documents=3, artifacts=[ART])],
                dockets=[c.FrameDocket(docket="DOCKET-1", kept=True, reason="open; 120 comments", comments=120, artifacts=[ART])],
                items=[c.FrameItem(id="COMMENT-1", docket="DOCKET-1", posted_at=T0, withdrawn=False)]),  # fmt: skip
        FAMILY,
        c.EventFamilies(frame="frame1", families=[FAMILY]),
        READ,
        c.ScopeEstimate(scope="overall", lowest=READ, highest=READ, read=(0.2, 0.6), reason="the two completions' hull"),
        c.WideEstimate(objective=c.Ref(id="objective", version=1), frame="frame1", design=c.Ref(id="two_phase", version=1),
                       draws=(H, H), families="event_families-1", gold_labels=[H], judgments=[H], level=0.95, interval_method="score", phase1_items=1200,
                       unreadable_items=3, gold_items=130, unresolved_gold_labels=2,
                       scopes=[c.ScopeEstimate(scope="overall", lowest=READ, highest=READ, read=(0.2, 0.6),
                                               reason="the two completions' hull")]),  # fmt: skip
        IMPACT_Q,
        LEAD,
        c.LeadList(estimate="estimate-1", leads=[LEAD]),
        c.LeadPromotion(leads="leads-1", lead=LEAD, case="deep-1"),
        c.AgentRun(id="run1", role="objective_drafter", model="claude-opus-5-5", effort="high", input_hash=H,
                   output_hash=H, response=H, proposals=["pr1"], at=T0),  # fmt: skip
        PARAM,
        c.Driver(id="labor", line="cost", name="crew wages", quantity=PARAM, price=PRICE, period="month"),
        LIQ,
        c.EntityModel(id="em", version=1, entity="A landscaping company", currency="USD", horizon=WINDOW,
                      loss_definition="incremental cash cost",
                      drivers=[c.Driver(id="labor", line="cost", name="crew wages", quantity=PARAM, price=PRICE,
                                        period="month")], liquidity=LIQ),  # fmt: skip
        IMPACT,
        DIAG,
        c.RegisterRow(entity="A landscaping company", driver="labor", event="P.overtime_rule", direction="increase",
                      mechanism="cost", magnitude_band="5_to_15pct", cash_impact=[IMPACT], baseline="standalone",
                      probability="pe1", expected_impact=(F(1000), F(5000)), tail_impact=(F(4000), F(9000)),
                      breach="unknown", hedgeable="true", diagnostics=[DIAG], claims=[CLM], judgments=[H],
                      nodes=["v:exposure.labor"]),  # fmt: skip
        InputReceipt(name="x", value_hash=H, lineage_hash=H),
        c.DerivationReceipt(node=c.Ref(id="n", version=1), value_hash=H, lineage_hash=H,
                            inputs=[InputReceipt(name="x", value_hash=H, lineage_hash=H)], computed_at=T0),  # fmt: skip
        c.PricedCandidate(ref="y", ir=c.Ref(id="wet-spring", version=1), price=D("10"), pricing="indicative",
                          price_source="judged probability x payout x 1.2", cap=D("1")),  # fmt: skip
        SUMMARY,
        c.ReducedCost(candidate="y", value=D("-110"), measured_improvement=F(20)),
        c.OptimizationResult(
            decision=c.Ref(id="decision", version=1),
            objective="liquidity_cvar",
            scenarios_hash=H,
            probabilities_hash=H,
            candidates=[
                c.PricedCandidate(ref="y", ir=c.Ref(id="wet-spring", version=1), price=D("10"), pricing="indicative",
                                  price_source="model", cap=D("1"))  # fmt: skip
            ],
            dates=[date(2027, 1, 1), date(2027, 2, 1)],
            solver="HiGHS",
            epsilon=D("1E-9"),
            scale=D("100"),
            primary_status="optimal",
            secondary_status="optimal",
            primary_objective=D("0"),
            secondary_objective=D("1.818181818"),
            recomputed_objective=F(0),
            primary_residuals={"cash": 0.0, "tail": 0.0, "budget": 0.0, "bounds": 0.0},
            secondary_residuals={"cash": 0.0, "tail": 0.0, "budget": 0.0, "bounds": 0.0, "primary_bound": 0.0},
            hedged=SUMMARY,
            unhedged=SUMMARY,
            cash_duals=[[D("0"), D("0")], [D("0"), D("0")]],
            budget_dual=D("0"),
            reduced_costs=[c.ReducedCost(candidate="y", value=D("-110"), measured_improvement=F(20))],
        ),
        c.CoverageObligation(conditions=["A"], status="met", witnesses=["s1", "s2"], proof=None),
        c.BoundaryPoint(condition="rain_ge_0.10", point="0.09", status="feasible", scenario="s3"),
        c.FuzzScenario(id="s1", state_hash=H, packet="scenario-s1-1", split="held_out", mandatory=True, severity="critical", ir_outcome="resolved_yes",
                       ir_evaluation=H, ir_transfers=[c.CashTransfer(at=T1, amount=D("100"), currency="USD",
                       basis="payout")], rendering_passed=True, resolver_outcomes={"resolved_yes": 1.0},
                       prose_outcome="resolved_yes", prose_transfers=[c.CashTransfer(at=T1, amount=D("100"),
                       currency="USD", basis="payout")], amount_error=D("0"), liquidity_discrepancy=D("0")),  # fmt: skip
        c.TermReading(term="payout_yes", statement="when it resolves YES, one contract pays 100 USD",
                      answers={"states": 0.9, "not_stated": 0.1}, read="states", judgments=[H]),
        c.Attribution(scenario="s1", cause="contract_ambiguity", status="open", evidence="minimal pair pending",
                      suspected_clauses=[("trigger", 0.93)]),  # fmt: skip
        HardGate(name="no unreviewed critical mismatch", passed=True, detail="-"),
        c.FuzzReport(
            contract=c.Ref(id="wet-spring", version=1),
            prose="clause-1",
            prose_hash=H,
            resolvers=[CONFIG],
            pooling_policy=c.Ref(id="equal_per_provider_split_among_configurations", version=1),
            whole_trigger="ok",
            terms=[],
            obligations=[c.CoverageObligation(conditions=["B"], status="redundant", witnesses=[], proof="UNSAT"),
                         c.CoverageObligation(conditions=["A", "B"], status="uncovered", witnesses=[], proof=None)],
            boundaries=[],
            scenarios=[],
            fidelity=1.0,
            clarity=1.0,
            ambiguity=0.0,
            hard_gates=[HardGate(name=n, passed=True, detail="-") for n in HARD_GATES],
            attributions=[],
            equivalence="unsupported_or_unassessed",
        ),
    ]
