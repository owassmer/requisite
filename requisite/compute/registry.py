"""Registered consequence nodes: pure functions that compute every consequence (spec §4 "@node", §5 computation layer;
invariant I7: dates, thresholds, payoffs, cash flows, aggregation and optimization are computed by registered nodes,
never by model prose).

A node is registered once under (id, version) with its named inputs. Changing what a node computes is a new version,
since outputs are keyed by node id and version (spec §5 incremental recomputation). Each input declares how the output
moves with it (`monotone`): where every input with an open interval is declared monotone, the output's interval is the
node at the box's corners, exactly; where one is not, the output over that interval is unknown (A05 §5.3: an interval
or an unknown, never a midpoint).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from requisite.core import Ref

Direction = Literal["increasing", "decreasing"]


@dataclass(frozen=True)
class Node:
    id: str
    version: int
    inputs: tuple[str, ...]
    monotone: Mapping[str, Direction]
    fn: Callable[..., Any]

    @property
    def ref(self) -> Ref:
        return Ref(id=self.id, version=self.version)


_REGISTRY: dict[tuple[str, int], Node] = {}


class UnknownNode(KeyError):
    pass


def node(
    id: str, version: int, inputs: list[str], monotone: Mapping[str, Direction] | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a pure function as the node (id, version) over the named inputs (spec §4)."""
    mono = dict(monotone or {})
    unknown = set(mono) - set(inputs)
    if unknown:
        raise ValueError(f"node {id} v{version} declares monotonicity for inputs it does not have: {sorted(unknown)}")

    def register(fn: Callable[..., Any]) -> Callable[..., Any]:
        key = (id, version)
        existing = _REGISTRY.get(key)
        if existing is not None and existing.fn is not fn:
            raise ValueError(f"node {id} v{version} is registered twice; a changed node is a new version (spec §5)")
        _REGISTRY[key] = Node(id, version, tuple(inputs), mono, fn)
        return fn

    return register


def get(ref: Ref) -> Node:
    try:
        return _REGISTRY[(ref.id, ref.version)]
    except KeyError:
        raise UnknownNode(f"no node {ref.id} v{ref.version} is registered") from None


def registered() -> list[Node]:
    return sorted(_REGISTRY.values(), key=lambda n: (n.id, n.version))
