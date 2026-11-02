"""The incremental computation runtime (requisite/compute/runtime.py): recompute only descendants, early cutoff, the diff
by layer (spec §5, §13 WP6), and value against lineage (addendum A11 §11.2, §13 row 20).

The model is small and spans spec §5's four layers: spec §4's own node example (labor.monthly_cost_delta) feeds an
exposure over the planning horizon and a hedge quantity, while a settlement node reads observations against a strike.
Moving the horizon is spec §15's fifth beat: exposure and optimizer nodes change, settlement does not.
"""

from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal as D
from pathlib import Path
from typing import Any

import pytest

from requisite import core as c
from requisite.compute.registry import Node
from requisite.compute.runtime import Graph, Instance, Runtime, Source, receipt_lineage, value_hash
from requisite.core.evidence import artifact_id, claim_id
from requisite.store import Conflict, Store

NOW = datetime(2026, 11, 1, 18, 0, tzinfo=UTC)


class Nodes:
    """The test's own node lookup (the runtime's `nodes` parameter), counting every call of every function."""

    def __init__(self) -> None:
        self.table: dict[tuple[str, int], Node] = {}
        self.calls: Counter[str] = Counter()

    def add(self, id: str, version: int, inputs: list[str], fn: Callable[..., Any]) -> None:
        def counted(**kw: Any) -> Any:
            self.calls[id] += 1
            return fn(**kw)

        self.table[(id, version)] = Node(id, version, tuple(inputs), {}, counted)

    def __call__(self, ref: c.Ref) -> Node:
        return self.table[(ref.id, ref.version)]


def claim(text: str) -> c.Claim:
    art = artifact_id(text.encode())
    return c.Claim(id=claim_id(art, "plain-v1", (0, len(text)), text), artifact=art, extraction="plain-v1",
                   span=(0, len(text)), text=text, authority="a filer")  # fmt: skip


def count_at_or_above(readings: tuple[D, ...], strike: D, unit: D) -> D:
    return unit if sum(1 for r in readings if r >= strike) >= 3 else D(0)


def nodes() -> Nodes:
    n = Nodes()
    n.add("labor.monthly_cost_delta", 1, ["headcount", "regime_cost_adverse", "regime_cost_base"],
          lambda headcount, regime_cost_adverse, regime_cost_base: headcount * (regime_cost_adverse - regime_cost_base))
    n.add("bind.applies", 1, ["answer"], lambda answer: answer == "yes")
    n.add("exposure", 1, ["delta", "months", "applies"], lambda delta, months, applies: delta * months if applies else D(0))
    n.add("hedge.quantity", 1, ["exposure", "unit"], lambda exposure, unit: exposure / unit)
    n.add("settlement.per_unit", 1, ["readings", "strike", "unit"], count_at_or_above)
    n.add("hedge.payoff", 1, ["quantity", "per_unit"], lambda quantity, per_unit: quantity * per_unit)
    return n  # fmt: skip


def inst(id: str, node: str, layer: str, **inputs: str) -> Instance:
    return Instance(id=id, node=c.Ref(id=node, version=1), inputs=inputs, layer=layer)  # type: ignore[arg-type]


READINGS = (D("0.12"), D("0.31"), D("0.05"), D("0.18"))

GRAPH = Graph([
    Source("headcount", "computation", 40, (claim("We employ 40 seasonal workers."),)),
    Source("cost.adverse", "computation", D("5200"), (c.Assumption(text="adverse regime cost per worker-month",
                                                                   binds={"cost.adverse": "5200"}, reason="scenario"),)),
    Source("cost.base", "computation", D("4100"), (claim("Current cost is $4,100 per worker-month."),)),
    Source("horizon.months", "computation", 6, (c.Ref(id="decision", version=1),)),
    Source("answer.applies", "proposition", "yes", (c.Ref(id="resolved.q.applies", version=1),)),
    Source("unit.payout", "contract", D("1000"), (c.Ref(id="contract", version=1),)),
    Source("strike", "contract", D("0.10"), (c.Ref(id="contract", version=1),)),
    Source("readings", "evidence", READINGS, (claim("Daily totals: 0.12, 0.31, 0.05, 0.18"),)),
    inst("monthly_cost_delta", "labor.monthly_cost_delta", "computation", headcount="headcount",
         regime_cost_adverse="cost.adverse", regime_cost_base="cost.base"),
    inst("applies", "bind.applies", "proposition", answer="answer.applies"),
    inst("exposure", "exposure", "computation", delta="monthly_cost_delta", months="horizon.months", applies="applies"),
    inst("hedge.quantity", "hedge.quantity", "computation", exposure="exposure", unit="unit.payout"),
    inst("settlement.per_unit", "settlement.per_unit", "contract", readings="readings", strike="strike",
         unit="unit.payout"),
    inst("hedge.payoff", "hedge.payoff", "computation", quantity="hedge.quantity", per_unit="settlement.per_unit"),
])  # fmt: skip


def source(graph: Graph, id: str, **changes: Any) -> Source:
    s = graph[id]
    assert isinstance(s, Source)
    return Source(id, changes.get("layer", s.layer), changes.get("value", s.value), changes.get("records", s.records))


@pytest.fixture
def store(tmp_path: Path):
    s = Store(tmp_path / "case", case="case", clock=lambda: NOW)
    yield s
    s.close()


def built(store: Store) -> tuple[Runtime, Nodes]:
    n = nodes()
    rt = Runtime(store, nodes=n)
    first = rt.recompute(GRAPH)
    assert len(first.recomputed) == 6 and all(v == 1 for v in n.calls.values())  # each node once
    n.calls.clear()
    return rt, n


def by_layer(diff, kind: str = "values") -> dict[str, list[tuple[str, Any, Any]]]:
    groups = diff.values if kind == "values" else diff.lineage
    return {layer: [(ch.id, ch.old, ch.new) for ch in changes] for layer, changes in groups.items() if changes}


def test_the_model_computes_what_its_nodes_say(store):
    rt, _ = built(store)
    assert rt.value("monthly_cost_delta") == D("44000")
    assert rt.value("exposure") == D("264000")
    assert rt.value("hedge.quantity") == D("264")
    assert rt.value("settlement.per_unit") == D("1000")
    assert rt.value("hedge.payoff") == D("264000")


@pytest.mark.covers("WP6.descendants")
def test_one_changed_input_recomputes_only_its_descendants(store):
    rt, n = built(store)
    # Move the planning horizon: its descendants are exposure, the hedge quantity and the payoff, and only their
    # functions run, once each; the cost delta, the binding and the settlement do not run at all.
    rt.recompute(rt.graph.replace(source(rt.graph, "horizon.months", value=9)))
    assert n.calls == Counter({"exposure": 1, "hedge.quantity": 1, "hedge.payoff": 1})
    n.calls.clear()
    # A cost input reaches the cost delta too, and nothing on the settlement side.
    rt.recompute(rt.graph.replace(source(rt.graph, "cost.adverse", value=D("5300"))))
    assert n.calls == Counter({"labor.monthly_cost_delta": 1, "exposure": 1, "hedge.quantity": 1, "hedge.payoff": 1})
    n.calls.clear()
    # The same graph again changes nothing and runs nothing.
    again = rt.recompute(rt.graph)
    assert n.calls == Counter() and again.value_changes() == [] and again.lineage_changes() == []


@pytest.mark.covers("WP6.early_cutoff")
def test_an_unchanged_value_hash_stops_propagation(store):
    rt, n = built(store)
    # A new reading below the strike: the settlement is recomputed, its value hash is unchanged, so the payoff that
    # depends on it is not.
    diff = rt.recompute(rt.graph.replace(source(rt.graph, "readings", value=(*READINGS, D("0.02")))))
    assert n.calls == Counter({"settlement.per_unit": 1})
    assert diff.recomputed == ("settlement.per_unit",)
    assert by_layer(diff) == {"evidence": [("readings", READINGS, (*READINGS, D("0.02")))]}
    n.calls.clear()
    # Both regime costs rise by the same amount: the delta is recomputed once and is unchanged, and nothing below it runs.
    g = rt.graph.replace(source(rt.graph, "cost.adverse", value=D("5300")), source(rt.graph, "cost.base", value=D("4200")))
    diff = rt.recompute(g)
    assert n.calls == Counter({"labor.monthly_cost_delta": 1})
    assert [ch.id for ch in diff.value_changes()] == ["cost.adverse", "cost.base"]


@pytest.mark.covers("WP6.diff")
def test_the_diff_lists_exactly_the_changed_nodes_with_old_and_new_values_by_layer(store):
    rt, _ = built(store)
    # spec §15 beat 5: move the planning horizon; exposure and optimizer nodes change, settlement logic does not.
    diff = rt.recompute(rt.graph.replace(source(rt.graph, "horizon.months", value=9)))
    assert by_layer(diff) == {"computation": [
        ("horizon.months", 6, 9),
        ("exposure", D("264000"), D("396000")),
        ("hedge.quantity", D("264"), D("396")),
        ("hedge.payoff", D("264000"), D("396000")),
    ]}  # fmt: skip
    assert diff.lineage_changes() == [] and diff.stale_approvals == ()
    # A change across layers: the resolved answer turns, and a reading below the strike arrives. The settlement is
    # recomputed and unchanged, so it is not in the diff; everything listed changed, and nothing that changed is missing.
    g = rt.graph.replace(source(rt.graph, "answer.applies", value="no"),
                         source(rt.graph, "readings", value=(*READINGS, D("0.01"))))  # fmt: skip
    diff = rt.recompute(g)
    assert by_layer(diff) == {
        "evidence": [("readings", READINGS, (*READINGS, D("0.01")))],
        "proposition": [("answer.applies", "yes", "no"), ("applies", True, False)],
        "computation": [("exposure", D("396000"), D("0")), ("hedge.quantity", D("396"), D("0")),
                        ("hedge.payoff", D("396000"), D("0"))],
    }  # fmt: skip
    assert "settlement.per_unit" in diff.recomputed
    assert [ch.source for ch in diff.value_changes()] == [True, True, False, False, False, False]
    # Removing a vertex is a change too: its old value, and no new one.
    diff = rt.recompute(rt.graph.without("hedge.payoff"))
    assert by_layer(diff) == {"computation": [("hedge.payoff", D("0"), None)]}
    assert diff.values["computation"][0].new_value_hash is None


@pytest.mark.covers("A.T20")
def test_evidence_changes_and_the_number_does_not(store):
    rt, n = built(store)
    quantity = rt.state("hedge.quantity")
    approval = c.Approval(id="a1", gate="G3", subject=c.Ref(id="hedge.quantity", version=1),
                          lineage_hash=quantity.lineage_hash, approver="owen", at=NOW, decision="approved",
                          reason="sized against the filed headcount", scope="case")  # fmt: skip
    store.commit_approval(approval)
    assert rt.explain("hedge.quantity").approvals == ((approval, True),)
    # A new source supports the same headcount.
    new_claim = claim("Our H-2A petition lists 40 workers.")
    diff = rt.recompute(rt.graph.replace(source(rt.graph, "headcount", records=(new_claim,))))
    # Numeric early cutoff: no function runs and every number stands.
    assert n.calls == Counter() and diff.recomputed == () and diff.value_changes() == []
    assert rt.value("hedge.quantity") == D("264") and rt.state("hedge.quantity").value_hash == quantity.value_hash
    # The provenance change is visible: every value that rests on the headcount has a new lineage.
    assert by_layer(diff, "lineage") == {"computation": [
        (v, rt.value(v), rt.value(v)) for v in ("headcount", "monthly_cost_delta", "exposure", "hedge.quantity",
                                                 "hedge.payoff")
    ]}  # fmt: skip
    assert all(ch.old_lineage != ch.new_lineage for ch in diff.lineage_changes())
    # And so is the approval's: it covers the old lineage, not the current one.
    new_lineage = rt.state("hedge.quantity").lineage_hash
    assert [(s.approval.id, s.lineage) for s in diff.stale_approvals] == [("a1", new_lineage)]
    explained = rt.explain("hedge.quantity")
    assert explained.approvals == ((approval, False),)
    assert any("a1" in line and "not this one" in line for line in explained.lines())
    assert rt.explain("headcount").records == (f"Claim {new_claim.id}",)
    # Both derivations are on record: the old receipt stands, and the new lineage has its own.
    for lineage in (quantity.lineage_hash, new_lineage):
        assert store.get("derivation", lineage, c.DerivationReceipt).value_hash == quantity.value_hash
    # The approval stopped covering once, and a later change does not list it again; a runtime that starts afresh lists
    # it at its first build, since it misses the lineage it finds.
    later = rt.recompute(rt.graph.replace(source(rt.graph, "headcount", records=(claim("Payroll shows 40 workers."),))))
    assert later.lineage_changes() and later.stale_approvals == ()
    fresh = Runtime(store, nodes=nodes()).recompute(rt.graph)
    assert [s.approval.id for s in fresh.stale_approvals] == ["a1"]


def test_a_judgment_input_adds_its_question_state_and_configuration_to_the_key(store):
    # spec §5: "judgment nodes add question version, state hash and judge config". The provider's model version changes
    # and its answer does not: the binding the judgment feeds runs again, its value is unchanged, and the cutoff holds.
    def judged(model: str) -> c.Judgment:
        config = c.JudgeConfig(provider="clef", model=model, model_cutoff="unknown", paraphrase_id="p0", render_id="text")
        return c.Judgment(question=c.Ref(id="q.applies", version=1), state_hash="0" * 64, config=config,
                          reported={"yes": 0.9, "no": 0.05, "ne": 0.05}, distribution={"yes": 0.9, "no": 0.05, "ne": 0.05},
                          precision=0.01, response="k", latency_ms=1, cost_usd=D("0"), created_at=NOW)  # fmt: skip

    rt, n = built(store)
    g = rt.graph.replace(source(rt.graph, "answer.applies", records=(judged("@cf/clef-2026-10"),)))
    rt.recompute(g)
    assert n.calls == Counter({"bind.applies": 1})  # the key took in the judgment
    n.calls.clear()
    diff = rt.recompute(rt.graph.replace(source(rt.graph, "answer.applies", records=(judged("@cf/clef-2026-11"),))))
    assert n.calls == Counter({"bind.applies": 1}) and diff.value_changes() == []
    assert [ch.id for ch in diff.lineage_changes()] == ["answer.applies", "applies", "exposure", "hedge.quantity",
                                                       "hedge.payoff"]  # fmt: skip


def test_every_computed_value_has_its_receipt_committed(store):
    rt, _ = built(store)
    rt.recompute(rt.graph.replace(source(rt.graph, "horizon.months", value=12)))
    for v in rt.graph:
        if isinstance(v, Source):
            continue
        state = rt.state(v.id)
        receipt = store.get("derivation", state.lineage_hash, c.DerivationReceipt)
        assert receipt == state.receipt
        # The receipt is checkable from its own fields and from the values it names.
        assert receipt.lineage_hash == receipt_lineage(receipt.node, receipt.inputs)
        assert receipt.value_hash == value_hash(rt.value(v.id))
        assert {(i.name, i.value_hash, i.lineage_hash) for i in receipt.inputs} == {
            (name, rt.state(src).value_hash, rt.state(src).lineage_hash) for name, src in v.inputs.items()
        }


def test_a_node_changed_without_a_new_version_is_refused(store):
    built(store)
    # The same node id and version computing something else: its lineage is already committed with another value.
    changed = nodes()
    changed.add("hedge.quantity", 1, ["exposure", "unit"], lambda exposure, unit: exposure / unit / 2)
    other = Runtime(store, nodes=changed)
    with pytest.raises(Conflict, match="without a new version"):
        other.recompute(GRAPH)
    assert other.graph.order == ()  # nothing kept from a failed recompute
    # As a new version it is a new lineage, and is admitted.
    changed.add("hedge.quantity", 2, ["exposure", "unit"], lambda exposure, unit: exposure / unit / 2)
    g = GRAPH.replace(Instance("hedge.quantity", c.Ref(id="hedge.quantity", version=2), {"exposure": "exposure",
                                                                                          "unit": "unit.payout"},
                               "computation"))  # fmt: skip
    assert other.recompute(g).values["computation"][-2].id == "hedge.quantity" and other.value("hedge.quantity") == 132


def test_the_graph_admits_only_acyclic_well_bound_vertices_over_data(store):
    with pytest.raises(ValueError, match="cycle"):
        Graph([inst("a", "x", "computation", v="b"), inst("b", "x", "computation", v="a")])
    with pytest.raises(ValueError, match="not in the graph"):
        Graph([inst("a", "x", "computation", v="missing")])
    with pytest.raises(ValueError, match="committed records"):
        Source("constant", "computation", 7, ())
    rt = Runtime(store, nodes=nodes())
    with pytest.raises(ValueError, match="binds"):
        rt.recompute(GRAPH.replace(inst("applies", "bind.applies", "proposition", text="answer.applies")))
    # A value with no canonical hash (a function, a mutable set) cannot enter the graph.
    with pytest.raises(TypeError):
        rt.recompute(GRAPH.replace(source(GRAPH, "readings", value=print)))
    assert value_hash(1) != value_hash(D(1)) != value_hash("1") and value_hash([1, 2]) == value_hash((1, 2))
