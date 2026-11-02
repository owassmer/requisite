"""Properties of the runtime over random DAGs of pure nodes (requisite/compute/runtime.py; spec §5; addendum A11 §11.2).

After any change (a source's value, its records, a node's version or function, a rewired input, an added, removed or
moved vertex), an incremental recompute equals a computation from scratch node for node, in value hash and lineage hash
both. Its diff is exactly what two full computations differ in: the value changes are the vertices whose value hash
differs (outcome 006's own check), the lineage changes those whose lineage hash differs while their value hash does
not, and the stale approvals those whose vertex's lineage moved. Its functions run exactly once for each key the change
made new, and never otherwise (Mokhov, Mitchell and Peyton Jones 2018, Definition 2.1: minimality).
"""

import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from requisite import core as c
from requisite.compute.registry import Node
from requisite.compute.runtime import LAYERS, Graph, Instance, Runtime, Source
from requisite.store import Store

NOW = datetime(2026, 11, 1, 18, 0, tzinfo=UTC)
ARGS = ("a", "b", "c")
# Few distinct outputs, so equal values (and so early cutoff) are common.
OPS: dict[str, Callable[[list[int]], int]] = {
    "sum": sum,
    "mod3": lambda xs: sum(xs) % 3,
    "sign": lambda xs: (sum(xs) > 0) - (sum(xs) < 0),
    "max": max,
    "neg": lambda xs: -sum(xs),
}


class Lookup:
    """Nodes op1..op3 over one to three inputs, each at versions 1 and 2. Version 2 computes what version 1 does: a new
    version need not change a number, and its lineage still changes."""

    def __init__(self) -> None:
        self.calls = 0
        self.table: dict[tuple[str, int], Node] = {}
        for op, f in OPS.items():
            for k in (1, 2, 3):
                for version in (1, 2):
                    self.table[(f"{op}{k}", version)] = Node(f"{op}{k}", version, ARGS[:k], {}, self._counted(f))

    def _counted(self, f: Callable[[list[int]], int]) -> Callable[..., int]:
        def fn(**kw: int) -> int:
            self.calls += 1
            return f([kw[a] for a in sorted(kw)])

        return fn

    def __call__(self, ref: c.Ref) -> Node:
        return self.table[(ref.id, ref.version)]


VALUES = st.integers(-3, 3)
TOKENS = st.integers(0, 3)


def record(token: int) -> c.Ref:
    return c.Ref(id=f"record-{token}", version=1)


def instance(data: st.DataObject, id: str, before: list[str]) -> Instance:
    k = data.draw(st.integers(1, 3))
    op = data.draw(st.sampled_from(sorted(OPS)))
    inputs = {ARGS[i]: data.draw(st.sampled_from(before)) for i in range(k)}
    return Instance(id, c.Ref(id=f"{op}{k}", version=data.draw(st.integers(1, 2))), inputs, data.draw(st.sampled_from(LAYERS)))


def draw_graph(data: st.DataObject) -> list[Source | Instance]:
    vertices: list[Source | Instance] = []
    for i in range(data.draw(st.integers(1, 4))):
        vertices.append(Source(f"s{i}", data.draw(st.sampled_from(LAYERS)), data.draw(VALUES), (record(data.draw(TOKENS)),)))
    for j in range(data.draw(st.integers(1, 8))):
        vertices.append(instance(data, f"n{j}", [v.id for v in vertices]))
    return vertices


def mutate(data: st.DataObject, vertices: list[Source | Instance]) -> list[Source | Instance]:
    """One change, keeping the list in an order where every input comes earlier (so the graph stays acyclic)."""
    vs = list(vertices)
    sources = [i for i, v in enumerate(vs) if isinstance(v, Source)]
    instances = [i for i, v in enumerate(vs) if isinstance(v, Instance)]
    used = {src for v in vs if isinstance(v, Instance) for src in v.inputs.values()}
    kinds = ["add"] + (["layer"] if vs else []) + (["value", "records", "both"] if sources else [])
    kinds += ["version", "op", "rewire"] if instances else []
    kinds += ["remove"] if any(v.id not in used for v in vs) else []
    kind = data.draw(st.sampled_from(kinds))
    if kind in ("value", "records", "both"):
        i = data.draw(st.sampled_from(sources))
        s = vs[i]
        assert isinstance(s, Source)
        value = data.draw(VALUES) if kind != "records" else s.value
        records = (record(data.draw(TOKENS)),) if kind != "value" else s.records
        vs[i] = Source(s.id, s.layer, value, records)
    elif kind in ("version", "op", "rewire"):
        i = data.draw(st.sampled_from(instances))
        n = vs[i]
        assert isinstance(n, Instance)
        k = len(n.inputs)
        if kind == "version":
            vs[i] = Instance(n.id, c.Ref(id=n.node.id, version=3 - n.node.version), n.inputs, n.layer)
        elif kind == "op":
            vs[i] = Instance(n.id, c.Ref(id=f"{data.draw(st.sampled_from(sorted(OPS)))}{k}", version=n.node.version),
                             n.inputs, n.layer)  # fmt: skip
        else:
            slot = data.draw(st.sampled_from(sorted(n.inputs)))
            inputs = {**n.inputs, slot: data.draw(st.sampled_from([v.id for v in vs[:i]]))}
            vs[i] = Instance(n.id, n.node, inputs, n.layer)
    elif kind == "layer":
        i = data.draw(st.integers(0, len(vs) - 1))
        v, layer = vs[i], data.draw(st.sampled_from(LAYERS))
        vs[i] = Source(v.id, layer, v.value, v.records) if isinstance(v, Source) else Instance(v.id, v.node, v.inputs, layer)
    elif kind == "remove":
        vs.pop(data.draw(st.sampled_from([i for i, v in enumerate(vs) if v.id not in used])))
    else:
        fresh = f"x{max((int(v.id[1:]) for v in vs if v.id.startswith('x')), default=-1) + 1}"
        vs.append(instance(data, fresh, [v.id for v in vs]) if vs else Source(fresh, "evidence", data.draw(VALUES),
                                                                            (record(0),)))  # fmt: skip
    return vs


def hashes(rt: Runtime) -> dict[str, tuple[str, str]]:
    return {vid: (rt.state(vid).value_hash, rt.state(vid).lineage_hash) for vid in rt.graph.order}


def keys(rt: Runtime) -> set[str]:
    return {k for vid in rt.graph.order if (k := rt.state(vid).output_key) is not None}


@settings(max_examples=300, deadline=None)
@given(st.data())
def test_an_incremental_recompute_equals_a_computation_from_scratch(data: Any) -> None:
    before = draw_graph(data)
    after = before
    for _ in range(data.draw(st.integers(1, 3))):
        after = mutate(data, after)
    g0, g1 = Graph(before), Graph(after)
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp), case="p", clock=lambda: NOW)
        lookup = Lookup()
        scratch0 = Runtime(store, nodes=lookup)
        scratch0.recompute(g0)
        h0 = hashes(scratch0)
        approved = data.draw(st.lists(st.sampled_from(list(g0.order)), unique=True))
        for vid in approved:
            store.commit_approval(c.Approval(id=f"ap-{vid}", gate="G3", subject=c.Ref(id=vid, version=1),
                                             lineage_hash=h0[vid][1], approver="owen", at=NOW, decision="approved",
                                             reason="r", scope="p"))  # fmt: skip

        incremental = Runtime(store, nodes=lookup)
        incremental.recompute(g0)
        lookup.calls = 0
        diff = incremental.recompute(g1)
        calls = lookup.calls

        scratch1 = Runtime(store, nodes=lookup)
        scratch1.recompute(g1)
        h1 = hashes(scratch1)

        # Node for node, the incremental state is the state computed from scratch, in value and in lineage.
        assert hashes(incremental) == h1
        assert {v: incremental.state(v).output_key for v in g1.order} == {v: scratch1.state(v).output_key for v in g1.order}

        # The diff's value changes are exactly the vertices whose value hash differs between the two computations (a
        # vertex present in one only differs), with the old and new values, under the vertex's layer.
        differ = {v for v in set(h0) | set(h1) if h0.get(v, (None,))[0] != h1.get(v, (None,))[0]}
        assert {ch.id for ch in diff.value_changes()} == differ
        for ch in diff.value_changes():
            assert (ch.old_value_hash, ch.new_value_hash) == (h0.get(ch.id, (None,))[0], h1.get(ch.id, (None,))[0])
            assert ch.layer == (g1[ch.id] if ch.id in g1 else g0[ch.id]).layer
            assert ch.id not in g0 or scratch0.value(ch.id) == ch.old
            assert ch.id not in g1 or scratch1.value(ch.id) == ch.new
        # Its lineage changes are exactly those whose lineage differs while their value does not.
        moved = {v for v in set(h0) & set(h1) if h0[v][1] != h1[v][1] and h0[v][0] == h1[v][0]}
        assert {ch.id for ch in diff.lineage_changes()} == moved
        assert all(ch.layer == g1[ch.id].layer for ch in diff.lineage_changes())
        # A11 §11.2: no change of a source's records, a node's version or an input's lineage (a policy, a provider
        # version, a dependency edge) disappears from lineage because a value happens to match.
        for vid in set(h0) & set(h1):
            v0, v1 = g0[vid], g1[vid]
            if isinstance(v0, Source) and isinstance(v1, Source):
                rest = v0.records != v1.records
            elif isinstance(v0, Instance) and isinstance(v1, Instance):
                rest = v0.node != v1.node or set(v0.inputs) != set(v1.inputs) or any(
                    h0[v0.inputs[a]][1] != h1[v1.inputs[a]][1] for a in v0.inputs if a in v1.inputs)
            else:
                rest = True
            assert not rest or h0[vid][1] != h1[vid][1]
        # The approvals that no longer cover their vertex are exactly those of vertices whose lineage moved or left.
        stale = {f"ap-{v}" for v in approved if v not in h1 or h1[v][1] != h0[v][1]}
        assert {s.approval.id for s in diff.stale_approvals} == stale

        # Minimality: a function runs once for each key the change made new, and for nothing else.
        new_keys = keys(scratch1) - keys(scratch0)
        assert calls == len(new_keys) == len(diff.recomputed)

        # Every current value's derivation is committed.
        for vid in g1.order:
            s = incremental.state(vid)
            if s.receipt is not None:
                assert store.get("derivation", s.lineage_hash, c.DerivationReceipt) == s.receipt
        store.close()
