"""A fuzz run end to end (requisite/fuzz/run.py; spec §8.6; addendum A09), on test doubles of the scenario renderer and
the decision models (tests/fuzz/doubles.py): each test's own input, never a cassette.

The contract: a merger contract that resolves YES when the board approved the agreement and the regulator cleared it,
each an interpretive condition asked by the library's condition_holds template, NO when either is not established by
expiration. Its prose names, in each section, the IR nodes the section states."""

from typing import Any

import irkit
import pytest
from doubles import FAILS, HOLDS, UNSET, Judges, Renderer, truth
from irkit import contract

from requisite import core as c
from requisite.agents.answers import ProseDraft, ProseSection
from requisite.core.results import same_transfers
from requisite.fuzz import run as fuzz
from requisite.fuzz import statements
from requisite.store import Store

APPROVED = c.Interpretive(id="approved", question_template="condition_holds",
                          bindings={"condition": "The company's board approved the merger agreement."})  # fmt: skip
CLEARED = c.Interpretive(id="cleared", question_template="condition_holds",
                         bindings={"condition": "The competition regulator cleared the merger."})  # fmt: skip
IR = contract(c.And(id="merger", args=[APPROVED, CLEARED]))
TERMS = statements.payment_terms(IR)
CLEARANCE = "Clearance means a determination by the competition regulator that the merger may proceed."
UNDEFINED = "Clearance means a Glorp-certified determination by the competition regulator."


def prose(definitions: str = CLEARANCE) -> ProseDraft:
    return ProseDraft(sections=[
        ProseSection(name="underlying_and_definitions", text=definitions, ir_nodes=["cleared"]),
        ProseSection(name="source_hierarchy_and_conflict_rule", text="The company's filings and the regulator's "
                     "published decisions are the sources; where they conflict, the regulator's decision governs.",
                     ir_nodes=[]),
        ProseSection(name="trigger", text="The contract resolves YES if, by expiration, the company's board approved the "
                     "merger agreement and the competition regulator cleared the merger.",
                     ir_nodes=["merger", "approved", "cleared"]),
        ProseSection(name="timing", text="The contract expires on June 1, 2027 at 14:00 UTC. "
                     + "".join(f"{t[0].upper()}{t[1:]}. " for t in TERMS.values()).strip(),
                     ir_nodes=[]),  # the IR's payment terms, as code states them
        ProseSection(name="unknown_at_expiry", text="If either condition is not established by expiration, the contract "
                     "resolves NO.", ir_nodes=[]),
        ProseSection(name="revisions", text="A decision later revised is read as first published.", ir_nodes=[]),
    ])  # fmt: skip


def careful(rules: str, found: dict[str, str], ask: Any) -> dict[str, float]:
    """Resolvers who apply the rules as written: YES only where both conditions are established."""
    both = truth(found, "board approved") == "true" and truth(found, "regulator cleared") == "true"
    return {"resolved_yes": 0.95} if both else {"resolved_no": 0.95}


def senses(conditions: list[dict[str, str]]) -> dict[str, str]:
    return {("approved" if "board approved" in x["condition"] else "cleared"): x["value"] for x in conditions}


def rulebook(tmp_path, text: ProseDraft | None = None) -> c.Proposal:
    """The prose proposed as the contract's rulebook in a real case holding the contract (spec I1)."""
    case = Store(tmp_path / "case", case="merger")
    if case.get("contract", f"{IR.id}|{IR.version}", c.ContractIR) is None:
        case.commit("contract", f"{IR.id}|{IR.version}", IR)
    proposal = fuzz.propose_prose(case, IR, text or prose(), author="test")
    assert proposal.status == "active", proposal.gates
    return proposal


def fuzzed(tmp_path, renderer: Renderer, judges: Judges, text: ProseDraft | None = None) -> fuzz.Run:
    store = Store(tmp_path / "fuzz", case="merger.fuzz", synthetic=True)
    return fuzz.run(IR, rulebook(tmp_path, text), store, judge=judges, agent=renderer, workers=1)


def by_state(r: fuzz.Run, approved: str, cleared: str, stress: str | None = None) -> fuzz.Resolved:
    want = {"approved": approved, "cleared": cleared}
    found = [x for x in r.resolved if {k: v.value for k, v in x.scenario.truths.items()} == want
             and x.scenario.stress == stress]  # fmt: skip
    assert found, (want, stress)
    return found[0]


@pytest.mark.covers("WP9.rendering_check")
def test_a_packet_that_does_not_encode_its_scenario_is_caught_rendered_again_and_never_counted(tmp_path):
    # Seeded: the first packet of the scenario "approved, not cleared" states that the board did not approve; every
    # packet of "approved and cleared" states that the regulator did not clear (a renderer that never encodes it).
    def corrupt(input: dict, attempt: int) -> list[dict[str, str]] | None:
        if input["stress_class"] is not None:
            return None
        want = senses(input["conditions"])
        flip = None
        if want == {"approved": HOLDS, "cleared": FAILS} and attempt == 0:
            flip = "board approved"
        if want == {"approved": HOLDS, "cleared": HOLDS}:
            flip = "regulator cleared"
        if flip is None:
            return None
        return [{**x, "value": FAILS} if flip in x["condition"] else x for x in input["conditions"]]

    renderer = Renderer(corrupt)
    r = fuzzed(tmp_path, renderer, Judges(careful))
    report = {s.id: s for s in r.report.scenarios}

    # Caught: the check asked each condition of the packet, and the judges read "did not approve".
    caught = by_state(r, "true", "false")
    first, second = caught.packets[0], caught.packets[1]
    assert len(caught.packets) == 2 and not first.passed and second.passed
    assert [m["intended"] for m in first.missed] == [HOLDS] and "board approved" in first.missed[0]["condition"]
    # Each provider's own reading, so the check claims what each recovered (external evaluation, November 2).
    from requisite.judgment.providers import PROVIDERS
    assert first.missed[0]["judges_read"] == "; ".join(f"{pv} read {FAILS}" for pv in PROVIDERS)
    # Rendered again with what was not recovered, and that packet passed: the scenario is scored on it alone.
    rerender = next(x for x in renderer.calls if x.get("not_recovered") == first.missed)
    assert rerender["previous_packet"][0]["text"] != ""
    assert report[caught.scenario.id].rendering_passed
    final_state = fuzz._state(r.plan, caught.scenario, second).content_hash()
    assert {j.state_hash for j in caught.judgments} == {final_state}

    # Never encoded: four renderings (the first and RERENDERS more), recorded as failed, the rendering suspected (open:
    # the check shows the readers did not recover the state, not which of renderer, question or readers is at fault),
    # and excluded from scoring. Its resolvers read NO where the IR says YES, and that is not
    # counted against the contract: fidelity is over the scenarios whose rendering passed, and no counterexample is
    # claimed; nor is an equivalence, since one scenario was not tested.
    never = by_state(r, "true", "true")
    assert len(never.packets) == 1 + fuzz.RERENDERS and not never.passed
    assert never.outcome == "resolved_no" and never.scenario.evaluation.procedure_status == "resolved_yes"
    assert not report[never.scenario.id].rendering_passed
    assert [(a.cause, a.status) for a in r.report.attributions if a.scenario == never.scenario.id] == [("rendering", "open")]
    assert r.report.fidelity == 1.0
    assert r.report.equivalence == "unsupported_or_unassessed"
    gate = next(g for g in r.report.hard_gates if g.name == "mandatory_coverage_met_or_accepted")
    assert not gate.passed and f"witness {never.scenario.id} not rendered" in gate.detail

    # Committed in the synthetic case: every packet's documents and claims, every judgment, the report.
    store = Store(tmp_path / "fuzz", case="merger.fuzz", synthetic=True)
    artifacts = {a.id for a in store.all("synthetic_artifact", c.SyntheticArtifact)}
    assert {art.id for x in r.resolved for p in x.packets for art, _, _ in p.documents} <= artifacts
    # Every judgment is committed: the rendering checks', the resolutions' and the payment terms' (FuzzReport.terms).
    committed = {j.content_hash() for j in store.all("judgment", c.Judgment)}
    asked = {j.content_hash() for x in r.resolved for j in [*x.judgments, *(j for p in x.packets for j in p.judgments)]}
    assert committed == asked | {h for t in r.report.terms for h in t.judgments}
    assert {t.term for t in r.report.terms} == set(statements.payment_terms(IR))
    assert fuzz.reports(store) == [r.report]
    # Each rendering is a scenario proposal (spec I1) whose tests gate is the rendering check: the failed ones rejected,
    # the passing one active, and the report names the packet each scenario was read from.
    proposals = {p.id: store.proposal(p.id) for p in store.all("proposal", c.Proposal)}
    caught_ids = [pid for pid, x in proposals.items() if x is not None and x.kind == "scenario"
                  and x.payload["scenario"] == caught.scenario.id]  # fmt: skip
    assert sorted(proposals[i].status for i in caught_ids) == ["active", "rejected"]  # type: ignore[union-attr]
    assert proposals[report[caught.scenario.id].packet].status == "active"  # type: ignore[union-attr]
    never_final = proposals[report[never.scenario.id].packet]
    assert never_final is not None and never_final.status == "rejected"
    assert "the rendering check missed" in next(g.detail for g in never_final.gates if g.gate == "tests")


@pytest.mark.covers("A.T18")
def test_a_passing_rendering_check_with_resolvers_against_the_ir_leaves_the_cause_open(tmp_path):
    # Resolvers who read an unestablished regulator clearance as no obstacle: where the board approved and clearance is
    # not established, they say YES, and the IR (its rule for an unknown at expiry: NO) says NO. The packet encodes the
    # scenario exactly, so the rendering check passes.
    def lenient(rules: str, found: dict[str, str], ask: Any) -> dict[str, float]:
        a, b = truth(found, "board approved"), truth(found, "regulator cleared")
        return {"resolved_yes": 0.95} if a == "true" and b in ("true", "unknown") else {"resolved_no": 0.95}

    r = fuzzed(tmp_path, Renderer(), Judges(lenient))
    x = by_state(r, "true", "unknown", "evidence_missing_at_expiry")
    assert x.passed and x.packets[0].missed == [] and len(x.packets) == 1
    assert x.scenario.evaluation.procedure_status == "resolved_no" and x.outcome == "resolved_yes"
    assert r.report.equivalence == "counterexample_found"

    opened = [a for a in r.report.attributions if a.scenario == x.scenario.id]
    # The cause stays open: every hypothesis open, none confirmed, the clause not singled out as the cause.
    assert opened and {a.status for a in opened} == {"open"}
    causes = {a.cause for a in opened}
    assert {"prose_divergence", "interpreter", "scenario_truth"} <= causes and "rendering" not in causes
    # A missing-data rule the resolvers did not apply is a silent substitution (A09 §10.4).
    gate = next(g for g in r.report.hard_gates if g.name == "no_silent_source_or_missing_data_substitution")
    assert not gate.passed and x.scenario.id in gate.detail


@pytest.mark.covers("WP9.ambiguity_attributed")
def test_an_undefined_term_in_one_section_raises_ambiguity_where_it_decides_and_is_attributed_to_that_section(tmp_path):
    # Resolvers who cannot tell what "Glorp-certified" means: where the outcome turns on clearance (the board approved),
    # they cannot resolve it; elsewhere they resolve as written.
    def puzzled(rules: str, found: dict[str, str], ask: Any) -> dict[str, float]:
        if "Glorp" in rules and truth(found, "board approved") == "true":
            yes = truth(found, "regulator cleared") == "true"
            return {"ambiguous": 0.6, "resolved_yes" if yes else "resolved_no": 0.3}
        return careful(rules, found, ask)

    r = fuzzed(tmp_path / "undefined", Renderer(), Judges(puzzled), prose(UNDEFINED))
    assert r.report.ambiguity > fuzz.AMBIGUITY
    attributed = {a.scenario: a for a in r.report.attributions if a.cause == "contract_ambiguity"}
    for x in r.resolved:
        approved = x.scenario.truths["approved"].value == "true"
        if approved:  # the outcome turns on clearance: ambiguity over the threshold, attributed to the definitions
            assert "cleared" in x.scenario.exercised and x.pooled["ambiguous"] > fuzz.AMBIGUITY
            a = attributed[x.scenario.id]
            assert a.status == "open" and a.evidence.startswith("suspected clauses: underlying_and_definitions (")
        else:
            assert x.pooled["ambiguous"] <= fuzz.AMBIGUITY and x.scenario.id not in attributed

    # The same contract with the term defined: no ambiguity to attribute.
    clear = fuzzed(tmp_path / "defined", Renderer(), Judges(puzzled), prose(CLEARANCE))
    assert clear.report.ambiguity <= fuzz.AMBIGUITY
    assert not [a for a in clear.report.attributions if a.cause == "contract_ambiguity"]
    assert clear.report.equivalence == "tested_equivalent_on_suite"
    assert all(g.passed for g in clear.report.hard_gates)
    # The split is the contract version's: the same scenarios are held out whatever the prose.
    assert [(s.id, s.split) for s in clear.report.scenarios] == [(s.id, s.split) for s in r.report.scenarios]


def test_an_unestablished_condition_is_asked_and_expected_as_not_established(tmp_path):
    r = fuzzed(tmp_path, Renderer(), Judges(careful))
    x = by_state(r, "true", "unknown", "evidence_missing_at_expiry")
    stated = senses(r.resolved[0].packets[0].agent_input["conditions"])
    assert set(stated) == {"approved", "cleared"}
    assert senses(x.packets[0].agent_input["conditions"]) == {"approved": HOLDS, "cleared": UNSET}


def test_the_cli_fuzzes_a_cases_contract_against_the_clause_drafters_prose(tmp_path, monkeypatch, capsys):
    from requisite import cli
    from requisite.agents import runner
    from requisite.judgment import providers

    case = Store(tmp_path / "case", case="merger")
    # The case's contract, committed as a reading whose gates passed commits it (requisite/contract/harvest.commit).
    payload = IR.model_dump(mode="json")
    case.commit("contract", f"{IR.id}|{IR.version}", IR)
    case.close()
    renderer = Renderer()

    def agent(role, input, output, effort="high"):
        if role == "clause_drafter":
            assert input == {"ir": payload}
            return runner.AgentResult(role, prose(), c.sha256("drafter|" + prose().model_dump_json()), "test-double", effort, True)
        return renderer(role, input, output, effort)

    monkeypatch.setattr(runner, "run", agent)
    monkeypatch.setattr(providers, "judge", Judges(careful))
    assert cli.main(["fuzz", "k", "--case", str(tmp_path / "case"), "--workers", "1"]) == 0
    out = capsys.readouterr().out
    assert "fuzz k v1" in out and "K = 3: the providers alone, in the original wording" in out
    assert "equivalence: tested_equivalent_on_suite" in out and "fidelity 1.000" in out
    # The report is committed in the run's synthetic case, beside the real one; the drafter's run in the real case.
    synthetic = Store(tmp_path / "case" / "fuzz" / "k.v1", case="merger/fuzz/k.v1", synthetic=True)
    (report,) = fuzz.reports(synthetic)
    assert report.contract == IR.ref and report.prose_hash == fuzz.prose_hash(prose())
    real = Store(tmp_path / "case", case="merger")
    assert [r.role for r in real.all("agent_run", c.AgentRun)] == ["clause_drafter"]
    assert not real.all("synthetic_artifact", c.SyntheticArtifact)


def test_the_drafters_prose_may_cite_only_the_contracts_own_nodes_and_observables():
    # A citation the contract does not have is refused as the drafter answers, so the runner returns it with the reason.
    from pydantic import ValidationError

    from requisite.fuzz.run import _prose_of

    held = _prose_of(IR)
    good = prose().model_dump(mode="json")
    assert held.model_validate(good)
    bad = {**good, "sections": [{**good["sections"][0], "ir_nodes": [*good["sections"][0]["ir_nodes"], "payout"]},
                                *good["sections"][1:]]}  # fmt: skip
    with pytest.raises(ValidationError, match=r"cites \['payout'\], which are neither node ids nor observable ids"):
        held.model_validate(bad)


def test_the_prose_fuzzed_is_a_rulebook_that_passed_its_gates(tmp_path):
    # The clause drafter's prose enters state as a clause proposal (spec I1); its tests gate holds it to the contract's
    # own nodes and observables and to stating every condition that decides settlement.
    case = Store(tmp_path / "case", case="merger")
    case.commit("contract", f"{IR.id}|{IR.version}", IR)
    good = prose()
    held = fuzz.propose_prose(case, IR, good, author="test")
    assert held.kind == "clause" and held.status == "active"
    assert fuzz.propose_prose(case, IR, good, author="test").id == held.id  # the same prose is the same proposal
    silent = good.model_copy(update={"sections": [x.model_copy(update={"ir_nodes": [n for n in x.ir_nodes if n != "approved"]})
                                                  for x in good.sections]})  # fmt: skip
    rejected = fuzz.propose_prose(case, IR, silent, author="test")
    assert rejected.status == "rejected"
    assert "condition approved, which decides settlement, is stated by no section" in next(
        g.detail for g in rejected.gates if g.gate == "tests")
    lacking = good.model_copy(update={"sections": [x for x in good.sections if x.name != "revisions"]})
    refused = fuzz.propose_prose(case, IR, lacking, author="test")  # the schema gate: spec §8.5's sections
    assert refused.status == "rejected" and "lacks ['revisions']" in next(g.detail for g in refused.gates if g.gate == "schema")
    store = Store(tmp_path / "fuzz", case="merger.fuzz", synthetic=True)
    with pytest.raises(fuzz.Refused, match="not an active clause proposal"):
        fuzz.run(IR, rejected, store, judge=Judges(careful), agent=Renderer(), workers=1)
    # The report names its rulebook and the K configurations that read it.
    r = fuzzed(tmp_path, Renderer(), Judges(careful))
    assert r.report.prose == held.id and len(r.report.resolvers) == len(r.configurations) >= 2


@pytest.mark.parametrize("ir", [IR, contract(irkit.at("x", ">", 10, irkit.EXPIRY))], ids=["merger", "x_above_10"])
def test_the_split_keeps_one_concrete_input_in_one_set_and_holds_out_its_share(ir):
    # The split is over what the readers are shown (fuzz.concrete_input): scenarios with one renderer request are one
    # test case and fall in one set (external evaluation, November 2: two missing-evidence stress scenarios of an
    # "x > 10" contract, one state and one request, had sat on both sides). Whole groups are held out until the held-out
    # scenarios reach HELD_OUT of the run's, never every scenario; derived here from the plan, for any contract.
    import math

    p = fuzz.plan(ir)
    split = fuzz._split(p)
    assert set(split) == {s.id for s in p.scenarios}
    groups: dict[str, set[str]] = {}
    for s in p.scenarios:
        groups.setdefault(fuzz.concrete_input(p, s), set()).add(s.id)
    for ids in groups.values():
        assert len({split[i] for i in ids}) == 1, ids
    held = {i for i, x in split.items() if x == "held_out"}
    if len(groups) >= 2:
        assert held and len(held) < len(split)
        want = math.ceil(len(split) * fuzz.HELD_OUT)
        largest = max(len(ids) for ids in groups.values())
        assert len(held) < want + largest
        rest = [ids for ids in groups.values() if not ids & held]
        assert len(held) >= want or all(len(held) + len(ids) >= len(split) for ids in rest)
    else:
        assert not held


def test_prose_that_pays_another_amount_on_another_date_is_not_read_as_the_irs_payment(tmp_path):
    # The IR pays one contract's YES amount at its payment date; the prose states a hundred times the YES amount, a month
    # later, and what NO pays as the IR does, with the trigger unchanged. The readers resolve exactly as the IR does, so
    # the outcomes agree everywhere; what decides is which payment terms each scenario's outcome rests on (external
    # evaluation, November 2: such a prose scored fidelity 1.0 and tested_equivalent_on_suite).
    unit = IR.payout.unit_value
    other = TERMS["payout_yes"].replace(f"{unit.normalize():f} {IR.payout.currency}",
                                        f"{(unit * 100).normalize():f} {IR.payout.currency}")
    later = TERMS["payment_date"].replace(f"payment is made {IR.timing.payment_lag_days} ",
                                          f"payment is made {IR.timing.payment_lag_days + 30} ")
    assert other != TERMS["payout_yes"] and later != TERMS["payment_date"]  # this test's premise: both terms changed
    stated = {**TERMS, "payout_yes": other, "payment_date": later}
    timing = "The contract expires on June 1, 2027 at 14:00 UTC. " + "".join(f"{t[0].upper()}{t[1:]}. " for t in stated.values())
    sections = [s.model_copy(update={"text": timing.strip()}) if s.name == "timing" else s for s in prose().sections]
    r = fuzzed(tmp_path, Renderer(), Judges(careful), ProseDraft(sections=sections))
    reads = {t.term: t.read for t in r.report.terms}
    assert reads == {"payout_yes": "states_otherwise", "payout_no": "states", "payment_date": "states_otherwise"}
    rendered = [x for x in r.report.scenarios if x.rendering_passed]
    assert all(x.prose_outcome == x.ir_outcome for x in rendered)
    yes = [x for x in rendered if x.prose_outcome == "resolved_yes"]
    no = [x for x in rendered if x.prose_outcome == "resolved_no"]
    assert yes and no  # this test's premise
    # A YES scenario rests on what YES pays and on the payment date, both read as stated otherwise: its payment is not
    # established (None, never an empty list that says "pays nothing"), and neither is any difference, which a reading
    # of the whole rulebook cannot show for one scenario.
    assert all(x.prose_transfers is None and x.amount_error is None for x in yes)
    # A NO scenario rests on what NO pays, which the prose states as the IR does: both pay nothing, and they agree.
    assert all(x.prose_transfers is not None and not x.prose_transfers and same_transfers(x) for x in no)
    # No equivalence is claimed and no counterexample: what the prose pays on YES is not established.
    assert r.report.equivalence == "unsupported_or_unassessed"
    critical_yes = [x for x in yes if x.severity == "critical" and x.mandatory]
    gate = r.report.hard_gates[0]  # A09 §10.4: no unreviewed critical mismatch
    assert gate.passed == (not critical_yes)
    if critical_yes:
        assert "prose payment is not established" in gate.detail
    # The same rulebook stating the IR's terms is read as stating them, and the IR's transfers stand for the prose's.
    same = fuzzed(tmp_path / "same", Renderer(), Judges(careful))
    assert {t.read for t in same.report.terms} == {"states"}
    assert all(x.prose_transfers is not None and list(x.prose_transfers) == list(x.ir_transfers)
               for x in same.report.scenarios if x.prose_outcome == x.ir_outcome)  # fmt: skip


def test_a_rendering_check_passes_only_where_every_provider_answers_and_recovers_the_state(tmp_path):
    # One provider reads the packets correctly; the others refuse every rendering-check question, as valid refusal
    # records. Before, refusals were dropped before pooling and the one answer passed the check for all (external
    # evaluation, November 2). Now no packet passes, every miss names the providers that gave no answer, and the run
    # refuses to report a measure of the prose, since no packet was checked by every provider.
    from requisite.fuzz.scenarios import FuzzError
    from requisite.judgment.providers import PROVIDERS

    reader = next(iter(PROVIDERS))
    careful_judges = Judges(careful)

    def partial(a: Any) -> c.Judgment:
        j = careful_judges(a)
        if a.question.task == "resolve" or ".term." in a.question.id or a.config.provider == reader:
            return j
        return j.model_copy(update={"distribution": None, "reported": None, "precision": None,
                                    "refusal": "the provider declined to answer"})  # fmt: skip

    with pytest.raises(FuzzError, match="passed the rendering check, so nothing measures its prose"):
        fuzzed(tmp_path, Renderer(), partial)  # type: ignore[arg-type]
    checks = [a for a in careful_judges.asked if ".check." in a.question.id]
    assert {a.config.provider for a in checks} == set(PROVIDERS)  # every provider was asked; the others' refusals count
