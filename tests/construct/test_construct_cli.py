"""Construction as a user runs it (`requisite construct <step> --case-dir <dir> --case <name>`), in process over a deep
case built in the test, the agent and the judges doubled; and the planning horizon moved, its diff showing exposure and
optimizer nodes changing and no settlement node (spec §15 beats 3 and 5)."""

from __future__ import annotations

from datetime import UTC, datetime
from fractions import Fraction

import construct_case as cc
import deep_case as dc

from requisite import cli, core
from requisite.agents import runner
from requisite.compute.runtime import Instance, Runtime
from requisite.construct import graph as cgraph
from requisite.construct import steps
from requisite.discovery import deep
from requisite.judgment import providers
from requisite.store import Store


def deep_case(tmp_path):
    store = dc.make_case(tmp_path, drivers=cc.DRIVERS, tolerance="100", cash="40000", credit="0",
                         templates=cc.HEDGEABILITY)  # fmt: skip
    judges = cc.Judges(cc.Rule())
    agent = dc.Agent(cc.Scout())
    deep.loop(store, 1, judge=judges, agent=agent, listing=dc.listing)
    return store, judges, agent


def test_construct_runs_end_to_end_through_the_cli(tmp_path, monkeypatch, capsys):
    store, judges, agent = deep_case(tmp_path)
    store.close()
    monkeypatch.setattr(providers, "judge", judges)
    monkeypatch.setattr(runner, "run", agent)
    where = ["--case-dir", str(tmp_path / "case"), "--case", "deep"]
    assert cli.main(["construct", "observables", *where, "--k", "1"]) == 0
    out = capsys.readouterr().out
    assert "observables for the top 1" in out and "E1.in_effect (observable-" in out and "passes true" in out
    assert [role for role, _ in agent.draft.calls if role == steps.SCOUT] == [steps.SCOUT]
    assert cli.main(["construct", "candidates", *where, "--k", "1"]) == 0
    out = capsys.readouterr().out
    assert "8 candidate contract(s), each accepted by the evaluator" in out and "strip12m.E1.in_effect" in out
    assert cli.main(["construct", "solve", *where, "--k", "1", "--budget", "30000", "--loading", "0.2",
                     "--loading-range", "0", "1/2", "--discount-rate", "0.05", "--eligibility", "bilateral"]) == 0  # fmt: skip
    out = capsys.readouterr().out
    for words in ("construction for construct.decision v1", "adverse model (the chosen structure): buys",
                  "favourable model (beside it)", "timing gap in E1:occurs", "repriced over the loading range",
                  "is optimal for liquidity_cvar at alpha 9/10 within the solver's tolerance"):  # fmt: skip
        assert words in out
    assert cli.main(["construct", "report", *where]) == 0
    assert capsys.readouterr().out == out
    again = Store(tmp_path / "case", case="deep")
    (record,) = again.all(steps.RECORD, core.ConstructionRecord)
    assert record.chosen == ("digital.E1.in_effect", "strip6m.E1.in_effect")
    assert (record.parameters.loading, record.parameters.loading_range) == (Fraction(1, 5), (0, Fraction(1, 2)))


def test_moving_the_horizon_changes_exposure_and_optimizer_nodes_and_no_settlement_node(tmp_path, monkeypatch, capsys):
    b = cc.build(tmp_path)
    cc.solve(b)
    claim = b.store.all("claim", core.Claim)[0]
    longer = core.TimeWindow(start=dc.AS_OF, end=datetime(2028, 5, 1, tzinfo=UTC), start_closed=True, end_closed=False)
    model = dc.entity_model(claim, cc.DRIVERS, version=2, cash="40000", credit="0").model_copy(update={"horizon": longer})
    dc.approve(b.store, dc.submit(b.store, "entity_model", model))
    spec = dc.decision_spec("100").model_copy(update={"version": 2, "horizon": longer})
    dc.approve(b.store, dc.submit(b.store, "decision_spec", spec))
    rt = Runtime(b.store)
    diff = deep.compare(b.store, 1, 2, judge=b.judges, runtime=rt)
    changed = {c.id for c in diff.value_changes()}
    # Exposure nodes and optimizer nodes change...
    assert {"v:days", "v:labor.E1.cash", "v:labor.E1.expected", "v:labor.daily", "v:construct.business",
            "v:construct.receipts.adverse", "v:construct.solve.adverse", "v:construct.solve.favourable",
            "v:construct.repricing"} <= changed  # fmt: skip
    # ...and settlement does not: no contract-layer value changes; the candidates and their settlement keep their
    # values, their lineage new only because the decision spec they read the decision time from is a new version.
    assert not diff.values["contract"]
    same = {c.id for c in diff.lineage_changes()}
    assert {"v:construct.candidates", "v:construct.settle.adverse", "v:construct.settle.favourable"} <= same
    assert not [vid for vid in changed if vid.startswith("v:construct.settle") or vid == "v:construct.candidates"]
    # A from-scratch recompute of the later graph equals the incremental one, node for node.
    after = deep.version_case(b.store, 2)
    graph = cgraph.full_graph(after, deep.resolve_case(after, deep.Judge(b.judges)))
    fresh = Runtime(b.store)
    fresh.recompute(graph)
    assert set(rt.graph.vertices) == set(graph.vertices)
    for v in graph:
        assert (rt.state(v.id).value_hash, rt.state(v.id).lineage_hash) == (
            fresh.state(v.id).value_hash, fresh.state(v.id).lineage_hash), v.id  # fmt: skip
    assert any(isinstance(v, Instance) and v.node.id == "construct.solve" for v in graph)
    # The user's command shows the same.
    monkeypatch.setattr(providers, "judge", b.judges)
    assert cli.main(["deep", "diff", "--case", str(tmp_path / "case"), "--from", "1", "--to", "2"]) == 0
    out = capsys.readouterr().out
    assert "computation v:construct.solve.adverse:" in out and "v:construct.settle.adverse: " in out
    assert not [line for line in out.splitlines() if line.strip().startswith("contract") and "->" in line]
