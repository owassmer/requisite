"""The incremental computation runtime (spec §5 "Incremental recomputation", §13 WP6; addendum A11 §11.2).

One graph serves spec §5's four. Every vertex carries its layer (evidence, proposition, computation, contract), which is
where its value lives, and an edge between layers is one of §5's cross-layer bindings. A vertex is a source or an
instance. A source is a value read from committed records (a claim, a judgment, an assumption, a parameter of the
decision, a policy, a gold label) and names the records it rests on. An instance applies a registered node (id and
version) to named inputs, each another vertex.

Each vertex has two hashes, kept apart (A11 §11.2: "Do not conflate 'same number' with 'same justification'"):
- its value hash, the canonical hash of its value, which decides early cutoff;
- its lineage hash, which covers how the value is justified. For a source, that is its id, its value hash and the
  identities of its records. For an instance, it is its node's id and version and each input's name, value hash and
  lineage hash, exactly the fields of its DerivationReceipt. So it changes with provenance, a policy, a gold label, a
  provider version or a dependency edge even where the value does not (A11 §11.2: "No policy change, gold-label update,
  provider-version change, or dependency-edge change may disappear merely because an intermediate value happens to
  match"). That holds for what feeds the graph, so whatever bears on a value enters as a source, and a node reads
  nothing else (I7; requisite/compute/purity.py).
An instance's output is keyed by its node id and version and its inputs' value hashes; an input derived from a judgment
adds its question version, state hash and judge configuration (spec §5).

A recompute takes the graph as it now stands and recomputes, in topological order, only the instances that depend on
what changed. Where a recomputed value hash is unchanged, its dependents' keys are unchanged and their functions do not
run (early cutoff: Mokhov, Mitchell and Peyton Jones, "Build Systems à la Carte", ICFP 2018, §2.3, kept in
research/sources/mokhov-mitchell-and-peyton-jones-build-systems-la-carte-icfp.md); where its lineage
changed, theirs is updated and a new receipt committed while their cached value stands. Every recompute returns a Diff:
the vertices whose value changed, old and new, by layer; separately those whose lineage changed with an unchanged value;
and the approvals that do not cover the lineage now current (an Approval approves one lineage hash). What explains a
value to a person is rendered from the current receipts (`explain`), never stored beside the value, so it cannot go
stale; an approval is the one record that holds a past lineage.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from typing import Any, Literal, get_args

from pydantic import BaseModel

from requisite.compute import registry
from requisite.compute.registry import Node
from requisite.core import Approval, DerivationReceipt, Judgment, Record, Ref, ResolvedJudgment, canonical_json, sha256
from requisite.core.base import FrozenDict, freeze
from requisite.core.governance import InputReceipt
from requisite.store import Store

Layer = Literal["evidence", "proposition", "computation", "contract"]
LAYERS: tuple[Layer, ...] = get_args(Layer)  # spec §5's four, in its order
NodeLookup = Callable[[Ref], Node]


# ---------------------------------------------------------------- hashes


def _type_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _encode(value: Any) -> Any:
    """The value as JSON, each type JSON would confuse tagged: an int, a Decimal, a Fraction and a string that print alike
    are different values, and a computation over one is not a computation over another. Only immutable data has a value
    hash, so a callable, a client or an open file cannot enter the graph as a value (I7: what a static walk cannot see
    coming in as an argument is refused here)."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"a value hash needs a finite float, not {value!r}")
        return value
    if isinstance(value, Decimal):
        return {"$decimal": str(value)}
    if isinstance(value, Fraction):
        return {"$fraction": f"{value.numerator}/{value.denominator}"}
    if isinstance(value, datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, date):
        return {"$date": value.isoformat()}
    if isinstance(value, timedelta):
        return {"$timedelta": [value.days, value.seconds, value.microseconds]}
    if isinstance(value, Enum):
        return {"$enum": _type_name(type(value)), "value": _encode(value.value)}
    if isinstance(value, BaseModel):
        return {"$record": _type_name(type(value)), "fields": value.model_dump(mode="json")}
    if isinstance(value, Mapping):
        pairs = [[_encode(k), _encode(v)] for k, v in value.items()]
        return {"$map": sorted(pairs, key=lambda p: canonical_json(p[0]))}
    if isinstance(value, list | tuple):  # the runtime freezes a list into a tuple; both are the same sequence
        return [_encode(v) for v in value]
    if isinstance(value, frozenset):
        return {"$set": sorted((_encode(v) for v in value), key=canonical_json)}
    if isinstance(value, bytes):
        return {"$bytes": value.hex()}
    raise TypeError(f"a {type(value).__name__} has no canonical value hash; a node's inputs and outputs are immutable data "
                    "(a set is held as a frozenset)")  # fmt: skip


def _hash(*parts: Any) -> str:
    return sha256(canonical_json(list(parts)))


def value_hash(value: Any) -> str:
    """The canonical hash of a value (spec §5: outputs are keyed by input value hashes)."""
    return _hash("value", _encode(value))


def receipt_lineage(node: Ref, inputs: Sequence[InputReceipt]) -> str:
    """The lineage hash a receipt's own fields determine: the node's id and version and each input's name, value hash and
    lineage (A11 §11.2). Any receipt can be checked against it."""
    ins = sorted([i.name, i.value_hash, i.lineage_hash] for i in inputs)
    return _hash("lineage", node.id, node.version, ins)


def _record_identity(record: Record) -> list[str]:
    return [_type_name(type(record)), record.content_hash()]


def _judgment_key(record: Judgment | ResolvedJudgment) -> list[Any]:
    """What a judgment adds to the key of an output it feeds (spec §5: "judgment nodes add question version, state hash
    and judge config"). A resolved judgment's configurations are those of the judgments it pooled, which it names by
    hash."""
    if isinstance(record, Judgment):
        q = record.question
        return ["judgment", q.id, q.version, record.state_hash, record.config.content_hash()]
    q = record.question
    return ["resolved", q.id, q.version, record.state_hash, sorted(record.judgments)]


# ---------------------------------------------------------------- the graph


@dataclass(frozen=True)
class Source:
    """A value read from committed records, and the records it rests on (their identities enter its lineage). A value
    with no record behind it would be a constant written into code, which the graph does not admit."""

    id: str
    layer: Layer
    value: Any
    records: tuple[Record, ...]

    def __post_init__(self) -> None:
        if self.layer not in LAYERS:
            raise ValueError(f"source {self.id}: a layer is one of {LAYERS}, not {self.layer!r}")
        records = tuple(self.records)
        if not records or not all(isinstance(r, Record) for r in records):
            raise ValueError(f"source {self.id} names the committed records its value is read from")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "value", freeze(self.value))


@dataclass(frozen=True)
class Instance:
    """A registered node (its id and version) applied to named inputs, each the id of a source or another instance."""

    id: str
    node: Ref
    inputs: Mapping[str, str]
    layer: Layer

    def __post_init__(self) -> None:
        if self.layer not in LAYERS:
            raise ValueError(f"instance {self.id}: a layer is one of {LAYERS}, not {self.layer!r}")
        object.__setattr__(self, "inputs", FrozenDict(dict(self.inputs)))


Vertex = Source | Instance


class Graph:
    """Sources and instances, acyclic, each id once. Immutable: a change is a new graph (`replace`, `without`)."""

    def __init__(self, vertices: Iterable[Vertex]):
        self.vertices: dict[str, Vertex] = {}
        for v in vertices:
            if v.id in self.vertices:
                raise ValueError(f"vertex {v.id} appears twice")
            self.vertices[v.id] = v
        self.dependents: dict[str, list[str]] = {vid: [] for vid in self.vertices}
        for v in self.vertices.values():
            if isinstance(v, Instance):
                for name, src in v.inputs.items():
                    if src not in self.vertices:
                        raise ValueError(f"instance {v.id} input {name} names {src}, which is not in the graph")
                    self.dependents[src].append(v.id)
        self.order: tuple[str, ...] = self._topological()

    def _topological(self) -> tuple[str, ...]:
        waiting = {vid: len(set(v.inputs.values())) if isinstance(v, Instance) else 0 for vid, v in self.vertices.items()}
        ready = deque(vid for vid, n in waiting.items() if n == 0)
        order: list[str] = []
        while ready:
            vid = ready.popleft()
            order.append(vid)
            for d in dict.fromkeys(self.dependents[vid]):
                waiting[d] -= 1
                if waiting[d] == 0:
                    ready.append(d)
        if len(order) != len(self.vertices):
            raise ValueError(f"the graph has a cycle through {sorted(set(self.vertices) - set(order))}")
        return tuple(order)

    def __getitem__(self, vid: str) -> Vertex:
        return self.vertices[vid]

    def __contains__(self, vid: object) -> bool:
        return vid in self.vertices

    def __iter__(self) -> Iterator[Vertex]:
        return (self.vertices[vid] for vid in self.order)

    def replace(self, *vertices: Vertex) -> Graph:
        """This graph with these vertices replacing those of the same id, or added."""
        new = {v.id: v for v in vertices}
        kept = [new.pop(v.id, v) for v in self.vertices.values()]
        return Graph([*kept, *new.values()])

    def without(self, *ids: str) -> Graph:
        return Graph(v for v in self.vertices.values() if v.id not in ids)


# ---------------------------------------------------------------- state and diff


@dataclass(frozen=True)
class State:
    """A vertex's value and hashes. `key` is what it adds to its consumers' keys: its value hash, or for a source read
    from judgments, that and the judgments' question versions, state hashes and configurations. `output_key` is an
    instance's own key (spec §5), `receipt` its committed derivation."""

    value: Any
    value_hash: str
    lineage_hash: str
    key: str
    output_key: str | None
    receipt: DerivationReceipt | None


@dataclass(frozen=True)
class Change:
    """One vertex's change. A vertex the recompute added has no old value (old_value_hash None), one it removed no new."""

    id: str
    layer: Layer
    source: bool
    old: Any
    new: Any
    old_value_hash: str | None
    new_value_hash: str | None
    old_lineage: str | None
    new_lineage: str | None


@dataclass(frozen=True)
class StaleApproval:
    """An approval of a vertex that does not cover its current lineage; `lineage` is None when the vertex left the graph."""

    approval: Approval
    lineage: str | None


@dataclass(frozen=True)
class Diff:
    """What one recompute changed (spec §5: "changed nodes with old and new values, grouped by layer"; A11 §11.2).
    values: the vertices whose value hash changed, by layer. lineage: those whose lineage changed and value did not.
    stale_approvals: the approvals this recompute leaves not covering their vertex: those that covered its old lineage,
    and for a vertex new to the runtime, those that miss its lineage (an explanation shows every approval's standing at
    any time). recomputed: the instances whose function ran, in the order it ran."""

    values: Mapping[Layer, tuple[Change, ...]]
    lineage: Mapping[Layer, tuple[Change, ...]]
    stale_approvals: tuple[StaleApproval, ...]
    recomputed: tuple[str, ...]

    def value_changes(self) -> list[Change]:
        return [c for layer in LAYERS for c in self.values[layer]]

    def lineage_changes(self) -> list[Change]:
        return [c for layer in LAYERS for c in self.lineage[layer]]

    def lines(self) -> list[str]:
        """The diff as a person reads it, by layer."""
        out = []
        for layer in LAYERS:
            for c in self.values[layer]:
                old = "(absent)" if c.old_value_hash is None else show(c.old)
                new = "(absent)" if c.new_value_hash is None else show(c.new)
                out.append(f"{layer:11} {c.id}: {old} -> {new}")
            for c in self.lineage[layer]:
                out.append(f"{layer:11} {c.id}: {show(c.new)}, same value, new lineage {c.new_lineage and c.new_lineage[:12]}")
        for s in self.stale_approvals:
            a = s.approval
            out.append(f"approval {a.id} ({a.gate}, {a.approver}) of {a.subject.id} covers lineage {a.lineage_hash[:12]}, "
                       f"not the current {s.lineage[:12] if s.lineage else '(removed)'}")  # fmt: skip
        return out


def show(value: Any) -> str:
    """A value as a line of an explanation or a diff shows it."""
    if isinstance(value, Record):
        name = getattr(value, "id", None)
        return f"{type(value).__name__} {name if isinstance(name, str) else value.content_hash()[:12]}"
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


@dataclass(frozen=True)
class ExplainedInput:
    name: str
    vertex: str
    value: Any
    value_hash: str
    lineage_hash: str


@dataclass(frozen=True)
class Explanation:
    """Why a vertex has its value, rendered from its current receipt (or, for a source, its records), with each approval
    of it and whether that approval covers the current lineage."""

    id: str
    layer: Layer
    value: Any
    value_hash: str
    lineage_hash: str
    node: Ref | None
    inputs: tuple[ExplainedInput, ...]
    records: tuple[str, ...]
    approvals: tuple[tuple[Approval, bool], ...]

    def lines(self) -> list[str]:
        how = f"node {self.node.id} v{self.node.version}" if self.node else "source"
        out = [f"{self.id} = {show(self.value)}  ({self.layer}; {how}; lineage {self.lineage_hash[:12]})"]
        out += [f"  {i.name} = {show(i.value)}  from {i.vertex} (lineage {i.lineage_hash[:12]})" for i in self.inputs]
        out += [f"  rests on {r}" for r in self.records]
        for a, covers in self.approvals:
            scope = "covers this lineage" if covers else f"covers lineage {a.lineage_hash[:12]}, not this one"
            out.append(f"  approval {a.id} ({a.gate}, {a.approver}, {a.decision}, scope {a.scope!r}): {scope}")
        return out


# ---------------------------------------------------------------- the runtime


class Runtime:
    """Computes a graph's values from its sources by registered nodes, incrementally, committing each computed value's
    receipt through the store. `nodes` looks a node up by (id, version): the product registry by default."""

    def __init__(self, store: Store, nodes: NodeLookup = registry.get):
        self.store = store
        self.nodes = nodes
        self.graph = Graph(())
        self._state: dict[str, State] = {}
        self._memo: dict[str, tuple[Any, str]] = {}  # output key -> (value, value hash)

    def state(self, vid: str) -> State:
        return self._state[vid]

    def value(self, vid: str) -> Any:
        return self._state[vid].value

    def recompute(self, graph: Graph) -> Diff:
        """Bring every value up to date with `graph`, recomputing only what depends on a change, and return the diff.
        The first recompute computes every vertex. Nothing is kept unless the whole recompute succeeds."""
        nodes = self._resolve(graph)
        old_graph, old_state = self.graph, self._state
        new_state: dict[str, State] = {}
        touched: list[str] = []  # vertices whose value or lineage changed, or that are new
        moved: set[str] = set()
        recomputed: list[str] = []
        receipts: list[DerivationReceipt] = []
        for vid in graph.order:
            v, before, old = graph[vid], old_graph.vertices.get(vid), old_state.get(vid)
            if isinstance(v, Source):
                if old is not None and before is v:
                    new_state[vid] = old
                    continue
                s = _source_state(v)
            else:
                same = isinstance(before, Instance) and before.node == v.node and before.inputs == v.inputs
                if old is not None and same and not moved.intersection(v.inputs.values()):
                    new_state[vid] = old
                    continue
                s, ran = self._instance_state(v, nodes[vid], new_state, old)
                if ran:
                    recomputed.append(vid)
                if s.receipt is not None and (old is None or old.lineage_hash != s.lineage_hash):
                    receipts.append(s.receipt)
            new_state[vid] = s
            if old is None or (old.value_hash, old.lineage_hash) != (s.value_hash, s.lineage_hash):
                touched.append(vid)
                moved.add(vid)
        committed = {r.lineage_hash: r for r in self.store.commit_derivations(receipts)}
        for vid in touched:
            s = new_state[vid]
            if s.receipt is not None and s.receipt.lineage_hash in committed:
                new_state[vid] = State(s.value, s.value_hash, s.lineage_hash, s.key, s.output_key,
                                       committed[s.receipt.lineage_hash])  # fmt: skip
        removed = [vid for vid in old_graph.order if vid not in graph]
        diff = self._diff(graph, new_state, touched, removed, recomputed)
        self.graph, self._state = graph, new_state
        return diff

    def _resolve(self, graph: Graph) -> dict[str, Node]:
        nodes = {}
        for v in graph:
            if isinstance(v, Instance):
                node = self.nodes(v.node)
                if (node.id, node.version) != (v.node.id, v.node.version):
                    raise ValueError(f"looking up {v.node.id} v{v.node.version} gave {node.id} v{node.version}")
                if set(v.inputs) != set(node.inputs):
                    raise ValueError(f"instance {v.id} binds {sorted(v.inputs)}; node {node.id} v{node.version} takes "
                                     f"{sorted(node.inputs)}")  # fmt: skip
                nodes[v.id] = node
        return nodes

    def _instance_state(
        self, inst: Instance, node: Node, states: Mapping[str, State], old: State | None
    ) -> tuple[State, bool]:
        ins = {name: states[vid] for name, vid in inst.inputs.items()}
        key = _hash("key", node.id, node.version, sorted([name, s.key] for name, s in ins.items()))
        ran = key not in self._memo
        if ran:
            out = freeze(node.fn(**{name: s.value for name, s in ins.items()}))
            self._memo[key] = (out, value_hash(out))
        value, vh = self._memo[key]
        receipts = tuple(InputReceipt(name=n, value_hash=s.value_hash, lineage_hash=s.lineage_hash)
                         for n, s in sorted(ins.items()))  # fmt: skip
        lineage = receipt_lineage(node.ref, receipts)
        if old is not None and old.lineage_hash == lineage and old.receipt is not None:
            receipt = old.receipt
        else:
            receipt = DerivationReceipt(node=node.ref, value_hash=vh, lineage_hash=lineage, inputs=receipts,
                                        computed_at=self.store.clock())  # fmt: skip
        return State(value, vh, lineage, vh, key, receipt), ran

    def _diff(
        self, graph: Graph, new: Mapping[str, State], touched: list[str], removed: list[str], recomputed: list[str]
    ) -> Diff:
        values: dict[Layer, list[Change]] = {layer: [] for layer in LAYERS}
        lineage: dict[Layer, list[Change]] = {layer: [] for layer in LAYERS}
        by_subject: dict[str, list[Approval]] = {}
        for a in self.store.all("approval", Approval):
            by_subject.setdefault(a.subject.id, []).append(a)
        stale: list[StaleApproval] = []
        for vid in [*touched, *removed]:
            o, n = self._state.get(vid), new.get(vid)
            v = graph[vid] if n is not None else self.graph[vid]
            change = Change(
                id=vid, layer=v.layer, source=isinstance(v, Source),
                old=o.value if o else None, new=n.value if n else None,
                old_value_hash=o.value_hash if o else None, new_value_hash=n.value_hash if n else None,
                old_lineage=o.lineage_hash if o else None, new_lineage=n.lineage_hash if n else None,
            )  # fmt: skip
            (values if change.old_value_hash != change.new_value_hash else lineage)[v.layer].append(change)
            # An approval stops covering here if it covered the old lineage; a vertex new to the runtime has no old
            # lineage, so every approval of it that misses the current one is listed.
            stale += [StaleApproval(a, change.new_lineage) for a in by_subject.get(vid, [])
                      if (o is None and a.lineage_hash != change.new_lineage)
                      or (o is not None and a.lineage_hash == change.old_lineage)]  # fmt: skip
        return Diff(
            values={layer: tuple(cs) for layer, cs in values.items()},
            lineage={layer: tuple(cs) for layer, cs in lineage.items()},
            stale_approvals=tuple(stale),
            recomputed=tuple(recomputed),
        )

    def explain(self, vid: str) -> Explanation:
        """Why `vid` has its value, from its current receipt (a source: from its records), and its approvals."""
        v, s = self.graph[vid], self._state[vid]
        approvals = tuple((a, a.lineage_hash == s.lineage_hash) for a in self.store.all("approval", Approval)
                          if a.subject.id == vid)  # fmt: skip
        if isinstance(v, Source):
            records = tuple(show(r) for r in v.records)
            return Explanation(vid, v.layer, s.value, s.value_hash, s.lineage_hash, None, (), records, approvals)
        assert s.receipt is not None  # an instance's state always carries its receipt
        inputs = tuple(
            ExplainedInput(i.name, v.inputs[i.name], self._state[v.inputs[i.name]].value, i.value_hash, i.lineage_hash)
            for i in s.receipt.inputs
        )
        return Explanation(vid, v.layer, s.value, s.value_hash, s.lineage_hash, s.receipt.node, inputs, (), approvals)


def _source_state(source: Source) -> State:
    vh = value_hash(source.value)
    judged = sorted(_judgment_key(r) for r in source.records if isinstance(r, Judgment | ResolvedJudgment))
    key = _hash("judged", vh, judged) if judged else vh
    lineage = _hash("source", source.id, vh, sorted(_record_identity(r) for r in source.records))
    return State(source.value, vh, lineage, key, None, None)
