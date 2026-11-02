"""Revising a clause that fails the wording test, and fuzzing again (requisite/fuzz/revise.py; spec §8.6 step 7; addendum
A09 §10.4; A01 §2.5), on test doubles of the clause drafter, the scenario renderer and the decision models
(tests/fuzz/doubles.py): each test's own input, never a cassette.

The contract is test_run's merger contract: YES when the board approved the agreement and the regulator cleared it, NO
when either is not established by expiration. Its split is the contract version's, as the fuzz run draws it
(requisite.fuzz.run._split, whose own properties test_run checks); each test derives what it expects from that split and
from each scenario's (approved, cleared) truths, never from scenario ids one split happened to give. The resolver doubles
hesitate over words a test seeds into one section (a term no reader can apply), on the scenarios whose outcome turns on
what that section states; the drafter double removes them, or not."""

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import pytest
from doubles import Judges, Renderer, truth
from irkit import contract

from requisite import core as c
from requisite.agents import runner
from requisite.agents.answers import ProseDraft, ProseSection
from requisite.fuzz import revise, statements
from requisite.fuzz import run as fuzz
from requisite.fuzz.scenarios import FuzzError
from requisite.store import Refused, Store

APPROVED = c.Interpretive(id="approved", question_template="condition_holds",
                          bindings={"condition": "The company's board approved the merger agreement."})  # fmt: skip
CLEARED = c.Interpretive(id="cleared", question_template="condition_holds",
                         bindings={"condition": "The competition regulator cleared the merger."})  # fmt: skip
IR = contract(c.And(id="merger", args=[APPROVED, CLEARED]))
TERMS = statements.payment_terms(IR)
CLEARANCE = "Clearance means a determination by the competition regulator that the merger may proceed."
GLORP = "Clearance means a Glorp-certified determination by the competition regulator."
TRIGGER = ("The contract resolves YES if, by expiration, the company's board approved the merger agreement and the "
           "competition regulator cleared the merger.")  # fmt: skip
UNKNOWN = "If either condition is not established by expiration, the contract resolves NO."
PLAN = fuzz.plan(IR)
SPLIT = fuzz._split(PLAN)  # the contract version's split, as every run of IR draws it
STATES = {s.id: (s.truths["approved"].value, s.truths["cleared"].value) for s in PLAN.scenarios}
HELD_OUT = {sid for sid, x in SPLIT.items() if x == "held_out"}


def where(split: str, matters: Callable[[str, str], bool]) -> set[str]:
    """The scenarios of a set whose (approved, cleared) truths make a seeded puzzle matter."""
    return {sid for sid, x in SPLIT.items() if x == split and matters(*STATES[sid])}


def only_in(split: str) -> set[tuple[str, str]]:
    """The (approved, cleared) states some scenario of this set has and no scenario of the other set has."""
    other = {STATES[sid] for sid, x in SPLIT.items() if x != split}
    return {STATES[sid] for sid, x in SPLIT.items() if x == split} - other


def prose(definitions: str = GLORP, trigger: str = TRIGGER, unknown: str = UNKNOWN) -> ProseDraft:
    return ProseDraft(sections=[
        ProseSection(name="underlying_and_definitions", text=definitions, ir_nodes=["cleared"]),
        ProseSection(name="source_hierarchy_and_conflict_rule", text="The company's filings and the regulator's "
                     "published decisions are the sources; where they conflict, the regulator's decision governs.",
                     ir_nodes=[]),
        ProseSection(name="trigger", text=trigger, ir_nodes=["merger", "approved", "cleared"]),
        ProseSection(name="timing", text="The contract expires on June 1, 2027 at 14:00 UTC. "
                     + "".join(f"{t[0].upper()}{t[1:]}. " for t in TERMS.values()).strip(),
                     ir_nodes=[]),  # the IR's payment terms, as code states them
        ProseSection(name="unknown_at_expiry", text=unknown, ir_nodes=[]),
        ProseSection(name="revisions", text="A decision later revised is read as first published.", ir_nodes=[]),
    ])  # fmt: skip


def careful(rules: str, found: Mapping[str, str]) -> dict[str, float]:
    both = truth(found, "board approved") == "true" and truth(found, "regulator cleared") == "true"
    return {"resolved_yes": 0.95} if both else {"resolved_no": 0.95}


def readers(*puzzles: tuple[str, Callable[[str, str], bool]]) -> Judges:
    """Resolvers who cannot apply a seeded word: where the rules carry it and the scenario's (approved, cleared) truths
    make it matter, they put 0.6 on AMBIGUOUS; elsewhere they resolve as written. A puzzle with no word is a
    hesitation over the scenario itself, whatever the prose."""

    def resolve(rules: str, found: Mapping[str, str], ask: Any) -> dict[str, float]:
        a, b = truth(found, "board approved"), truth(found, "regulator cleared")
        for word, matters in puzzles:
            if (not word or word in rules) and matters(a, b):
                yes = a == "true" and b == "true"
                return {"ambiguous": 0.6, "resolved_yes" if yes else "resolved_no": 0.3}
        return careful(rules, found)

    return Judges(resolve)


def glorp(a: str, b: str) -> bool:  # the definition of clearance decides where the board approved
    return a == "true" and b != "unknown"


class Drafter:
    """The clause drafter's double as clause_reviser: `revise(section, rulebook text by section)` returns the sections it
    revises, and the cases it says the IR does not decide; every input it was given is kept."""

    def __init__(self, revise_: Callable[[str, Mapping[str, str]], tuple[list[ProseSection], list[str]]]) -> None:
        self.revise = revise_
        self.inputs: list[dict[str, Any]] = []
        self.renderer = Renderer()

    def __call__(self, role: str, input: Mapping[str, Any], output: type, effort: str = "high") -> runner.AgentResult[Any]:
        if role == "scenario_renderer":
            return self.renderer(role, input, output, effort)
        assert role == revise.ROLE
        self.inputs.append(dict(input))
        texts = {s["name"]: s["text"] for s in input["rulebook"]}
        sections, undecided = self.revise(input["revise"]["name"], texts)
        answer = output.model_validate({"sections": [s.model_dump(mode="json") for s in sections],
                                        "not_decided_by_the_ir": undecided,
                                        "rationale": f"revised {input['revise']['name']}"})  # fmt: skip
        return runner.AgentResult(role, answer, c.sha256("reviser|" + c.canonical_json(dict(input))), "test-double",
                                  effort, True)  # fmt: skip


def section(name: str, text: str) -> ProseSection:
    return next(s for s in prose().sections if s.name == name).model_copy(update={"text": text})


def revised(tmp_path, judges: Judges, drafter: Drafter, base: ProseDraft | None = None) -> revise.Revised:
    case = Store(tmp_path / "case", case="merger")
    if case.get("contract", f"{IR.id}|{IR.version}", c.ContractIR) is None:
        case.commit("contract", f"{IR.id}|{IR.version}", IR)
    rulebook = fuzz.propose_prose(case, IR, base or prose(), author="test")
    store = fuzz.fuzz_store(case, IR)
    return revise.revise(case, IR, rulebook, store, judge=judges, agent=drafter, workers=1)


def held_out_seen(drafter: Drafter, held: set[str]) -> list[str]:
    """Held-out scenario ids anywhere in what the drafter was given."""
    return sorted(h for x in drafter.inputs for h in held if re.search(rf"\b{h}\b", json.dumps(x)))


def test_a_seeded_ambiguous_clause_is_revised_and_passes_on_the_held_out_scenarios(tmp_path):
    # The definitions section carries a term no reader can apply. The drafter revises it to the plain definition; and
    # it rewords the trigger, which leaves the term where it was, saying the IR does not decide one case it was asked
    # about.
    def fix(name: str, texts: Mapping[str, str]) -> tuple[list[ProseSection], list[str]]:
        if name == "underlying_and_definitions":
            return [section(name, CLEARANCE)], []
        return ([section(name, TRIGGER.replace("by expiration", "on or before expiration"))],
                ["whether a clearance given after expiration but dated before it counts"])

    drafter = Drafter(fix)
    out = revised(tmp_path, readers(("Glorp", glorp)), drafter)
    base = out.base
    failing_dev, failing_held = where("development", glorp), where("held_out", glorp)
    assert failing_dev and failing_held  # this test's premise: the seeded term matters on both sides of the split
    assert set(base.held_out.scenarios) == HELD_OUT  # the report keeps the contract version's split
    assert not base.development.meets and not base.held_out.meets
    assert set(base.held_out.failing) == failing_held

    (rnd,) = out.rounds
    assert rnd.number == 1 and not rnd.flagged
    # Suspected from the development scenarios only: those where the definition decides.
    assert list(rnd.suspected) == ["underlying_and_definitions", "trigger"]
    assert set(rnd.suspected["underlying_and_definitions"]) == failing_dev
    by = {r.sections[0]: r for r in rnd.revisions}
    definitions, trigger = by["underlying_and_definitions"], by["trigger"]
    # Each revision is its own, differs from the base only in its section, and passed its gates.
    assert definitions.changed == ("underlying_and_definitions",) and trigger.changed == ("trigger",)
    assert definitions.proposal is not None and definitions.proposal.status == "active"
    assert "only in underlying_and_definitions" in definitions.proposal.gates[-1].detail
    # Confirmed by evidence: revising the definitions alone turned every failing development scenario it was suspected
    # for into a passing one; rewording the trigger turned none.
    assert set(definitions.resolved) == set(rnd.suspected["underlying_and_definitions"]) and not definitions.unresolved
    assert trigger.resolved == () and set(trigger.unresolved) == set(rnd.suspected["trigger"])
    assert rnd.combined is None and rnd.reported is definitions.standing
    # The run reported passes, on the development scenarios and on the held-out ones, which no revision saw.
    after = rnd.reported
    assert after is not None and after.development.meets and after.held_out.meets and after.passes
    assert out.validated and out.stopped == "the development scenarios meet the thresholds"
    # The drafter saw development scenarios only; the re-runs kept every scenario's state in its set.
    assert held_out_seen(drafter, HELD_OUT) == []
    assert {s["scenario"] for x in drafter.inputs for s in x["failing_scenarios"]} == failing_dev
    for r in [definitions, trigger]:
        assert r.standing is not None
        assert [(x.id, x.state_hash, x.split) for x in r.standing.run.report.scenarios] == [
            (x.id, x.state_hash, x.split) for x in base.run.report.scenarios]
    text = revise.summary(out)
    assert "held-out-validated to that measure" in text
    one = min(failing_dev)
    assert f"{one}: not resolved by this revision; its attribution stays open" in text
    assert f"attribution {one}: confirmed by revising underlying_and_definitions" in text
    assert ("not decided by the IR (a finding for the contract, not its wording): whether a clearance given after "
            "expiration but dated before it counts") in text  # fmt: skip
    assert re.search(rf"held out \(never shown to the drafter\): {len(HELD_OUT)} scenarios: fidelity [0-9.]+ -> 1\.000", text)
    # The revision is a clause proposal in the case, its trace the drafter's run as clause_reviser; a rulebook this flow
    # made is never revised from, since its rounds count from the rulebook it came from.
    case = Store(tmp_path / "case", case="merger")
    assert revise.made_by_revision(case, definitions.proposal)
    with pytest.raises(Refused, match="is a revision this flow made"):
        revise.revise(case, IR, definitions.proposal, fuzz.fuzz_store(case, IR), judge=readers(), agent=drafter)


def test_a_revision_that_touches_an_unsuspected_section_is_refused(tmp_path):
    # The drafter's fix of the definitions also rewrites the revisions section, which no attribution suspects (no
    # scenario of this contract exercises it). The proposal is refused at its tests gate, and the report says the fix
    # lies outside the suspected sections. Its fix of the trigger also rewrites the definitions, suspected too but
    # revised in a revision of its own: refused as well.
    def overreach(name: str, texts: Mapping[str, str]) -> tuple[list[ProseSection], list[str]]:
        if name == "underlying_and_definitions":
            return [section(name, CLEARANCE), section("revisions", "A revised decision is read as last published.")], []
        if name == "trigger":
            return [section(name, TRIGGER), section("underlying_and_definitions", CLEARANCE)], []
        return [], []

    out = revised(tmp_path, readers(("Glorp", glorp)), Drafter(overreach))
    (rnd,) = out.rounds
    by = {r.sections[0]: r for r in rnd.revisions}
    definitions, trigger = by["underlying_and_definitions"], by["trigger"]
    assert definitions.proposal is not None and definitions.proposal.status == "rejected"
    assert definitions.unsuspected == ("revisions",) and definitions.standing is None
    assert definitions.refused is not None and (
        "in revisions, which fuzz report k.v1." in definitions.refused
        and "does not suspect from its development evidence" in definitions.refused)
    assert trigger.proposal is not None and trigger.proposal.status == "rejected" and trigger.unsuspected == ()
    assert trigger.refused is not None and "revised in a revision of its own" in trigger.refused
    assert rnd.confirmed == () and rnd.reported is None and not out.validated and out.examples == ()
    text = revise.summary(out)
    assert "the drafter's fix needs revisions, which the report does not suspect: the attribution points at the wrong place" in text
    assert "no revision of a suspected section turned a failing development scenario into a passing one" in out.stopped


def test_a_development_pass_with_a_held_out_fail_is_reported_as_such(tmp_path):
    # Beside the seeded term, the readers hesitate, whatever the prose, on a state only held-out scenarios have.
    # Revising the definitions passes the development scenarios; the held-out ones still fail.
    held_only = only_in("held_out")
    assert held_only  # this test's premise: some state of the contract is held out alone
    def fix(name: str, texts: Mapping[str, str]) -> tuple[list[ProseSection], list[str]]:
        return ([section(name, CLEARANCE)], []) if name == "underlying_and_definitions" else ([], [])

    out = revised(tmp_path, readers(("Glorp", glorp), ("", lambda a, b: (a, b) in held_only)), Drafter(fix))
    hesitant = where("held_out", lambda a, b: (a, b) in held_only)
    (rnd,) = out.rounds
    after = rnd.reported
    assert after is not None and after.development.meets and not after.held_out.meets
    assert set(after.held_out.failing) == hesitant and not after.passes and not out.validated
    assert out.stopped == "the development scenarios meet the thresholds" and out.examples == ()
    verdict = revise.verdict(out)
    assert verdict.startswith("round 1's run passes on the development scenarios and fails on the held-out scenarios")
    assert all(sid in verdict for sid in hesitant)
    # Their own attributions were never the drafter's to see.
    assert not hesitant & {sid for sids in rnd.suspected.values() for sid in sids}


def test_the_examples_are_exactly_the_passing_coverage_scenarios(tmp_path, monkeypatch, capsys):
    # Readers who resolve as written, but hesitate where the board did not approve and the regulator cleared: 0.08 on
    # AMBIGUOUS there, under the threshold on average. The run passes; those scenarios do not, so they are no example,
    # and the stress scenarios are not coverage scenarios.
    from requisite import cli
    from requisite.judgment import providers

    def resolve(rules: str, found: Mapping[str, str], ask: Any) -> dict[str, float]:
        if truth(found, "board approved") == "false" and truth(found, "regulator cleared") == "true":
            return {"resolved_no": 0.9, "ambiguous": 0.08}
        return careful(rules, found)

    case = Store(tmp_path / "case", case="merger")
    case.commit("contract", f"{IR.id}|{IR.version}", IR)
    case.close()
    drafted = prose(CLEARANCE)
    renderer = Renderer()

    def agent(role, input, output, effort="high"):
        if role == "clause_drafter":
            run = c.sha256("drafter|" + drafted.model_dump_json())  # a run id is a hash: the trace reads its first 16
            return runner.AgentResult(role, drafted, run, "test-double", effort, True)
        return renderer(role, input, output, effort)

    monkeypatch.setattr(runner, "run", agent)
    monkeypatch.setattr(providers, "judge", Judges(resolve))
    assert cli.main(["fuzz", "k", "--case", str(tmp_path / "case"), "--workers", "1"]) == 0
    out = capsys.readouterr().out
    synthetic = Store(tmp_path / "case" / "fuzz" / "k.v1", case="merger/fuzz/k.v1", synthetic=True)
    (report,) = fuzz.reports(synthetic)
    scenarios = {x.id: x for x in report.scenarios}
    real = Store(tmp_path / "case", case="merger")
    rulebook = report.prose
    published = revise.published(real, rulebook)
    # Exactly the coverage scenarios (no stress variant) that passed, in the plan's order: not the hesitant ones.
    hesitant = {sid for sid, st in STATES.items() if st == ("false", "true")}
    expected = [s.id for s in PLAN.scenarios if s.role in revise.COVERAGE_SET and s.id not in hesitant]
    assert hesitant and expected  # this test's premise
    assert [x.scenario_id for x in published] == expected
    for x in published:
        assert x.expected == scenarios[x.scenario_id].ir_outcome and x.evaluation == scenarios[x.scenario_id].ir_evaluation
    by_state = {STATES[x.scenario_id]: x for x in published}
    assert "board approved the merger agreement: holds." in by_state[("true", "true")].description
    assert "The contract resolves YES, paying" in by_state[("true", "true")].description
    assert by_state[("true", "false")].description.endswith("The contract resolves NO; nothing is paid.")
    assert "examples published (the passing coverage scenarios" in out and f"1. ({expected[0]})" in out
    assert not any(f"({sid})" in out for sid in hesitant)

    # A run that does not pass publishes none.
    drafted = prose(GLORP)
    monkeypatch.setattr(providers, "judge", readers(("Glorp", glorp)))
    assert cli.main(["fuzz", "k", "--case", str(tmp_path / "case"), "--workers", "1"]) == 0
    assert "no examples published: the run does not pass" in capsys.readouterr().out
    glorp_rulebook = fuzz.clause_id(fuzz.clause_payload(IR, prose(GLORP)))
    assert revise.published(Store(tmp_path / "case", case="merger"), glorp_rulebook) == []


def test_a_round_after_the_first_is_flagged_and_its_pass_publishes_no_examples(tmp_path):
    # Two seeded terms: Glorp in the definitions (where the board approved) and Zorp in the trigger (on a state only
    # development scenarios have). In round 1 the drafter fixes the definitions and swaps Zorp for another unreadable
    # term; round 1 reports the definitions' run, which still fails those scenarios. Round 2 revises the trigger from that
    # run and passes: flagged.
    dev_only = only_in("development")
    assert dev_only  # this test's premise: some state of the contract is in development alone
    zorped = where("development", lambda a, b: (a, b) in dev_only)
    zorp = TRIGGER.replace("cleared the merger.", "cleared the merger (Zorp rules apply).")

    def fix(name: str, texts: Mapping[str, str]) -> tuple[list[ProseSection], list[str]]:
        if name == "underlying_and_definitions":
            return [section(name, CLEARANCE)], []
        if name == "trigger":
            return [section(name, zorp.replace("Zorp", "Blorp") if "Glorp" in texts["underlying_and_definitions"]
                            else TRIGGER)], []  # fmt: skip
        return [], []

    puzzled = readers(("Glorp", glorp), ("Zorp", lambda a, b: (a, b) in dev_only), ("Blorp", lambda a, b: (a, b) in dev_only))
    out = revised(tmp_path, puzzled, Drafter(fix), prose(GLORP, zorp))
    first, second = out.rounds
    assert not first.flagged and second.flagged and second.number == 2
    assert first.reported is not None and set(first.reported.development.failing) == zorped
    assert second.base is first.reported and set(second.suspected["trigger"]) == zorped
    number, last = out.last
    assert number == 2 and last.passes and not out.validated and out.examples == ()
    assert "the round is flagged" in revise.verdict(out)
    assert "round 2 (FLAGGED" in revise.summary(out)
    assert revise.published(Store(tmp_path / "case", case="merger"), last.rulebook.id) == []


def test_several_confirmed_sections_are_combined_and_that_run_is_the_one_reported(tmp_path):
    # Glorp in the definitions decides where the board approved and clearance is known; Snarf in the unknown-at-expiry
    # rule decides where a condition is not established. Each revision alone resolves
    # its own scenarios; the rulebook combining both is fuzzed once more and reported.
    snarf = UNKNOWN + " Snarf provisions apply."

    def fix(name: str, texts: Mapping[str, str]) -> tuple[list[ProseSection], list[str]]:
        if name == "underlying_and_definitions":
            return [section(name, CLEARANCE)], []
        if name == "unknown_at_expiry":
            return [section(name, UNKNOWN)], []
        return [], []

    puzzled = readers(("Glorp", glorp), ("Snarf", lambda a, b: "unknown" in (a, b)))
    out = revised(tmp_path, puzzled, Drafter(fix), prose(GLORP, TRIGGER, snarf))
    (rnd,) = out.rounds
    assert [r.sections[0] for r in rnd.confirmed] == ["underlying_and_definitions", "unknown_at_expiry"]
    combined = rnd.combined
    assert combined is not None and combined.proposal is not None and combined.proposal.author == revise.COMBINER
    assert combined.changed == ("underlying_and_definitions", "unknown_at_expiry") and combined.proposal.status == "active"
    assert rnd.reported is combined.standing and combined.standing is not None and combined.standing.passes
    coverage = [s.id for s in PLAN.scenarios if s.role in revise.COVERAGE_SET]
    assert out.validated and [x.scenario_id for x in out.examples] == coverage
    snarfed = where("development", lambda a, b: "unknown" in (a, b))
    assert snarfed  # this test's premise
    assert f"attribution {min(snarfed)}: confirmed by revising unknown_at_expiry" in revise.summary(out)


def test_a_rulebook_revision_keeps_every_scenario_in_its_set(tmp_path):
    # The split is the contract version's: two rulebooks of one contract hold out the same scenarios, and a re-run whose
    # report moved one is refused.
    a = revised(tmp_path / "a", readers(), Drafter(lambda n, t: ([], [])), prose(CLEARANCE)).base.run
    b = revised(tmp_path / "b", readers(), Drafter(lambda n, t: ([], [])), prose(GLORP)).base.run
    assert a.report.prose != b.report.prose
    assert [(x.id, x.state_hash, x.split) for x in a.report.scenarios] == [
        (x.id, x.state_hash, x.split) for x in b.report.scenarios]
    revise.same_split(a, b)
    first = a.report.scenarios[0]
    moved = [x.model_copy(update={"split": "held_out" if first.split == "development" else "development"})
             if x.id == first.id else x for x in a.report.scenarios]  # fmt: skip
    with pytest.raises(FuzzError, match=rf"does not keep .*scenarios in their sets: \['{first.id}'\]"):
        revise.same_split(a, replace(b, report=b.report.model_copy(update={"scenarios": moved})))


def test_the_drafter_is_never_given_a_held_out_scenario(tmp_path):
    out = revised(tmp_path, readers(("Glorp", glorp)), Drafter(lambda n, t: ([], [])))
    with pytest.raises(FuzzError, match="held out: the drafter never sees a held-out scenario"):
        revise.reviser_input(out.base, "underlying_and_definitions", sorted(HELD_OUT)[:1])


def test_the_cli_revises_a_cases_contract(tmp_path, monkeypatch, capsys):
    from requisite import cli
    from requisite.judgment import providers

    case = Store(tmp_path / "case", case="merger")
    case.commit("contract", f"{IR.id}|{IR.version}", IR)
    case.close()

    def fix(name: str, texts: Mapping[str, str]) -> tuple[list[ProseSection], list[str]]:
        return ([section(name, CLEARANCE)], []) if name == "underlying_and_definitions" else ([], [])

    drafter = Drafter(fix)

    def agent(role, input, output, effort="high"):
        if role == "clause_drafter":
            return runner.AgentResult(role, prose(), c.sha256("drafter|" + prose().model_dump_json()), "test-double", effort, True)
        return drafter(role, input, output, effort)

    monkeypatch.setattr(runner, "run", agent)
    monkeypatch.setattr(providers, "judge", readers(("Glorp", glorp)))
    assert cli.main(["fuzz", "revise", "k", "--case", str(tmp_path / "case"), "--workers", "1"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("revise k v1: rulebook clause-")
    assert "round 1 (the first: no held-out result has informed a revision)" in out
    assert "verdict: round 1's run meets spec §8.6's thresholds on the development scenarios and on the held-out" in out
    assert "examples published under rulebook" in out
    # Run again, it replays: the same rulebooks, reports and examples, nothing proposed or traced twice.
    real = Store(tmp_path / "case", case="merger")
    before = (len(real.all("agent_run", c.AgentRun)), len(real.all("proposal", c.Proposal)))
    assert cli.main(["fuzz", "revise", "k", "--case", str(tmp_path / "case"), "--workers", "1"]) == 0
    assert capsys.readouterr().out == out
    assert (len(real.all("agent_run", c.AgentRun)), len(real.all("proposal", c.Proposal))) == before
    # The dossier shows the examples in the examples section of the rulebook that published them, and in no other.
    import html

    from requisite import dossier

    dossier.render(tmp_path / "case", "merger")
    (key,) = {k.split("|")[0] for k in real.keys(revise.EXAMPLES)}
    shown = [p for p in (tmp_path / "case" / "dossier").rglob("*.html") if "Published by code from a run" in p.read_text()]
    assert len(shown) == 1 and html.escape(key) in shown[0].read_text()
    assert html.escape(revise.examples_text(revise.published(real, key))) in shown[0].read_text()
