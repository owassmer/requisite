"""The cases the dossier tests render, each built by the product's own functions: nothing here writes a record the flow
would not write.

- `construction`: a real case holding a filed series (the contract tests' synthetic strike ladder, read by their
  contract_reader double twice: once as the exchange settled it, active, and once off by an inclusive bound, rejected by
  the settled-market check), the merger contract of the fuzz tests with its rulebook (active) and a rulebook citing a
  node the IR lacks (rejected at its tests gate), an objective awaiting G1, and a fuzz run of the merger contract on the
  fuzz tests' renderer and judge doubles, in its synthetic case beside it (requisite.fuzz.run.fuzz_store).
- `wide`: the wide flow over the regulations.gov fixtures, as tests/evidence/test_wide_flow.py runs it, through its gold
  labels: objective approved at G1, frame taken by its proposal's gate, phase-1 sample drawn, read and labelled, the
  stability configurations, the gold sample drawn and labelled.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import test_harvest as th
from doubles import Judges, Renderer
from fake_api import KEY, FakeAPI, FakeTime, fixture_world
from test_frame import criteria
from test_run import IR, careful, prose
from test_wide_flow import OBJECTIVE, stand_in

from requisite import cassettes, core, secrets
from requisite.agents.answers import ProseDraft, ProseSection
from requisite.contract import harvest
from requisite.discovery import gold, labels, wide
from requisite.evidence.frame import frame_gate
from requisite.evidence.regulations_gov import Pacer, RegulationsGov
from requisite.fuzz import run as fuzz
from requisite.judgment import providers
from requisite.library import templates as library
from requisite.store import Store


def recording(monkeypatch: pytest.MonkeyPatch, tmp: Path) -> Path:
    """Live calls allowed into the test's own cassette directory, as a recording run outside the sandbox: only the test
    doubles and the fake API answer them."""
    monkeypatch.delenv("REQUISITE_REPLAY_ONLY", raising=False)
    monkeypatch.setattr(secrets, "known_values", lambda: [KEY])
    monkeypatch.setattr(cassettes, "CASSETTES", tmp / "cassettes")
    monkeypatch.setattr(cassettes, "SPEND_LOG", tmp / "spend.jsonl")
    return tmp / "cassettes"


def construction(monkeypatch: pytest.MonkeyPatch, tmp: Path) -> Path:
    root = recording(monkeypatch, tmp)
    case_dir = tmp / "construction"
    case = Store(case_dir, case="construction")
    # F1: a filed series and two readings of it.
    th.ladder(case)
    exact = harvest.read(case, th.SERIES, agent=th.Reader(th.draft()), root=root)
    inclusive = harvest.read(case, th.SERIES, agent=th.Reader(th.draft(th.reading(op=">="))), root=root)
    assert (exact.status, inclusive.status) == ("active", "rejected"), (exact.gates, inclusive.gates)
    # A contract, its rulebook, and a rulebook that is not its.
    case.commit("contract", f"{IR.id}|{IR.version}", IR)
    rulebook = fuzz.propose_prose(case, IR, prose(), author="test")
    wrong = ProseDraft(sections=[*(s for s in prose().sections if s.name != "trigger"),
                                 ProseSection(name="trigger", text="Resolves YES when the merger closes.",
                                              ir_nodes=["merger", "closing"])])  # fmt: skip
    wrong.sections.sort(key=lambda s: core.SECTION_ORDER.index(s.name))
    refused = fuzz.propose_prose(case, IR, wrong, author="test")
    assert (rulebook.status, refused.status) == ("active", "rejected"), refused.gates
    # A wide objective awaiting its human gate.
    payload = OBJECTIVE.model_dump(mode="json")
    pending = case.submit_proposal(core.Proposal(id=f"objective-{core.sha256(core.canonical_json(payload))[:12]}",
                                                 case=case.case, kind="objective", payload=payload, author="test",
                                                 rationale="drafted for the test", status="proposed"))  # fmt: skip
    assert pending.status == "proposed"
    # The merger contract fuzzed against its rulebook, in its synthetic case.
    synthetic = fuzz.fuzz_store(case, IR)
    fuzz.run(IR, rulebook, synthetic, judge=Judges(careful), agent=Renderer(), workers=1)
    synthetic.close()
    case.close()
    return case_dir


def _approve(store: Store, kind: str, payload: dict) -> core.Proposal:
    p = store.submit_proposal(core.Proposal(id=f"{kind}-{core.sha256(core.canonical_json(payload))[:12]}",
                                            case=store.case, kind=kind, payload=payload, author="fixture",
                                            rationale="fixture", status="proposed"))  # fmt: skip
    if p.status == "proposed":
        p = wide.approve(store, p.id, gate="G1", approver="tester", decision="approved", reason="as drafted",
                         scope="this run")  # fmt: skip
    return p


def wide_case(monkeypatch: pytest.MonkeyPatch, tmp: Path) -> Path:
    root = recording(monkeypatch, tmp)
    monkeypatch.setattr(providers, "_send", stand_in)
    api = fixture_world(FakeAPI())
    time = FakeTime()

    def client() -> RegulationsGov:
        return RegulationsGov(transport=api.transport(), key=lambda: KEY, root=root,
                                   pacer=Pacer(time.clock, time.sleep),
                                   budget=cassettes.Budget(run_usd=0.0, run_calls=10**9, day_usd=float("inf"),
                                                           log=tmp / "spend.jsonl"))  # fmt: skip

    case_dir = tmp / "wide"
    store = Store(case_dir, case="wide")
    _approve(store, "objective", OBJECTIVE.model_dump(mode="json"))
    for t in labels.OBJECTIVE_TEMPLATES:
        tpl = library.template(t)
        o = library.original(tpl)
        pset = core.ParaphraseSet(
            id=f"ps.{t}", version=tpl.version, template=core.Ref(id=t, version=tpl.version), slots=list(tpl.slots),
            answers=[a.value for a in tpl.answers],
            paraphrases=[o, *(core.Paraphrase(id=i, target=o.target + suffix, meanings=dict(o.meanings))
                              for i, suffix in (("p1", " Answer from the filing."), ("p2", " Read the whole filing.")))],
        )  # fmt: skip
        _approve(store, "paraphrase_set", pset.model_dump(mode="json"))
    store.register_gate("frame", "tests", frame_gate(client))
    frame = _approve(store, "frame", criteria(datetime(2026, 6, 20, 12, tzinfo=UTC), "closed_pending_final")
                     .model_dump(mode="json"))  # fmt: skip
    assert frame.status == "active", frame.gates
    wide.draw_sample(store)
    wide.read_sample(store, client, workers=1)
    wide.label_phase1(store, workers=1)
    wide.test_stability(store, workers=1)
    gold_draw = wide.draw_gold(store)
    gold.label_session(store, gold_draw, wide.approved_objective(store), wide.measures(store), labeler="tester",
                       read=lambda _: "1", write=lambda _: None)  # fmt: skip
    store.close()
    return case_dir
