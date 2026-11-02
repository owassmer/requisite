"""A fuzz run end to end (requisite/fuzz/run.py; spec §8.6; addendum A09), on test doubles of the scenario renderer and
the decision models (tests/fuzz/doubles.py): each test's own input, never a cassette.

The contract: a merger contract that resolves YES when the board approved the agreement and the regulator cleared it,
each an interpretive condition asked by the library's condition_holds template, NO when either is not established by
expiration. Its prose names, in each section, the IR nodes the section states."""

from typing import Any

import pytest
from doubles import FAILS, HOLDS, UNSET, Judges, Renderer, truth
from irkit import contract

from requisite import core as c
from requisite.agents.answers import ProseDraft, ProseSection
from requisite.fuzz import run as fuzz
from requisite.store import Store

APPROVED = c.Interpretive(id="approved", question_template="condition_holds",
                          bindings={"condition": "The company's board approved the merger agreement."})  # fmt: skip
CLEARED = c.Interpretive(id="cleared", question_template="condition_holds",
                         bindings={"condition": "The competition regulator cleared the merger."})  # fmt: skip
IR = contract(c.And(id="merger", args=[APPROVED, CLEARED]))
CLEARANCE = "Clearance means a determination by the competition regulator that the merger may proceed."
UNDEFINED = "Clearance means a Glorp-certified determination by the competition regulator."


def prose(definitions: str = CLEARANCE) -> ProseDraft:
    return ProseDraft(sections=[
        ProseSection(name="underlying_and_definitions", text=definitions, ir_nodes=["cleared"]),
        ProseSection(name="trigger", text="The contract resolves YES if, by expiration, the company's board approved the "
                     "merger agreement and the competition regulator cleared the merger.",
                     ir_nodes=["merger", "approved", "cleared"]),
        ProseSection(name="timing", text="The contract expires on June 1, 2027 at 14:00 UTC.", ir_nodes=[]),
        ProseSection(name="unknown_at_expiry", text="If either condition is not established by expiration, the contract "
                     "resolves NO.", ir_nodes=[]),
    ])  # fmt: skip


def careful(rules: str, found: dict[str, str], ask: Any) -> dict[str, float]:
    """Resolvers who apply the rules as written: YES only where both conditions are established."""
    both = truth(found, "board approved") == "true" and truth(found, "regulator cleared") == "true"
    return {"resolved_yes": 0.95} if both else {"resolved_no": 0.95}


def senses(conditions: list[dict[str, str]]) -> dict[str, str]:
    return {("approved" if "board approved" in x["condition"] else "cleared"): x["value"] for x in conditions}


def fuzzed(tmp_path, renderer: Renderer, judges: Judges, text: ProseDraft | None = None) -> fuzz.Run:
    store = Store(tmp_path / "fuzz", case="merger.fuzz", synthetic=True)
    return fuzz.run(IR, text or prose(), store, judge=judges, agent=renderer, workers=1)


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
    assert first.missed[0]["judges_read"] == FAILS
    # Rendered again with what was not recovered, and that packet passed: the scenario is scored on it alone.
    rerender = next(x for x in renderer.calls if x.get("not_recovered") == first.missed)
    assert rerender["previous_packet"][0]["text"] != ""
    assert report[caught.scenario.id].rendering_passed
    final_state = fuzz._state(r.plan, caught.scenario, second).content_hash()
    assert {j.state_hash for j in caught.judgments} == {final_state}

    # Never encoded: four renderings (the first and RERENDERS more), recorded as failed, its cause the rendering's,
    # confirmed by the check, and excluded from scoring. Its resolvers read NO where the IR says YES, and that is not
    # counted against the contract: fidelity is over the scenarios whose rendering passed, and no counterexample is
    # claimed; nor is an equivalence, since one scenario was not tested.
    never = by_state(r, "true", "true")
    assert len(never.packets) == 1 + fuzz.RERENDERS and not never.passed
    assert never.outcome == "resolved_no" and never.scenario.evaluation.procedure_status == "resolved_yes"
    assert not report[never.scenario.id].rendering_passed
    assert [(a.cause, a.status) for a in r.report.attributions if a.scenario == never.scenario.id] == [("rendering", "confirmed")]
    assert r.report.fidelity == 1.0
    assert r.report.equivalence == "unsupported_or_unassessed"
    gate = next(g for g in r.report.hard_gates if g.name == "mandatory_coverage_met_or_accepted")
    assert not gate.passed and f"witness {never.scenario.id} not rendered" in gate.detail

    # Committed in the synthetic case: every packet's documents and claims, every judgment, the report.
    store = Store(tmp_path / "fuzz", case="merger.fuzz", synthetic=True)
    artifacts = {a.id for a in store.all("synthetic_artifact", c.SyntheticArtifact)}
    assert {art.id for x in r.resolved for p in x.packets for art, _, _ in p.documents} <= artifacts
    assert len(store.all("judgment", c.Judgment)) == len({j.content_hash() for x in r.resolved
                                                          for j in [*x.judgments, *(j for p in x.packets for j in p.judgments)]})  # fmt: skip
    assert fuzz.reports(store) == [r.report]


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
            return runner.AgentResult(role, prose(), "double-drafter-" + "0" * 40, "test-double", effort, True)
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
