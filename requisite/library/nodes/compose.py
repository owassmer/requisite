"""How sub-questions compose into the question they replace (spec §2.1 compound: "Replace with sub-questions plus a
composition node"; spec §4 AnswerValue.binds: the proposition values an answer implies).

A split is a decision list. Its sub-questions are asked in order; each substantive answer either binds the parent's
values (it is terminal: its binds are the parent's answer) or binds nothing (the next sub-question decides). The last
sub-question's substantive answers are all terminal. A sub-question whose answer is not established is read as any of
its substantive answers, since not_established is an evidence status and not a state of the world (addendum A03
§3.3), so the parent's feasible values are every terminal binding reachable that way. A refinement of one question is
a decision list of one step. The composition is code over the answers' binds, the same for every split; the agent that
proposes a split states only the sub-questions and what each answer binds.

A step is the sub-question's feasible answers, each as its binds (an empty mapping: continue). The result is the parent's
feasible answers as their binds, each once, in a canonical order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from requisite.compute.registry import node
from requisite.core import canonical_json

Binds = Mapping[str, Any]


def binds_key(b: Binds) -> str:
    """A binding's canonical text: equal bindings have equal keys, whatever their mapping or sequence types."""
    return canonical_json(_plain(b))


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def decision_list(steps: Sequence[Sequence[Binds]]) -> tuple[Binds, ...]:
    """The parent's feasible bindings from its sub-questions' feasible bindings, in order (see the module's account)."""
    if not steps:
        raise ValueError("a composition has at least one sub-question")
    out: dict[str, Binds] = {}
    for i, step in enumerate(steps):
        if not step:
            raise ValueError(f"sub-question {i + 1} has no feasible answer")
        last = i == len(steps) - 1
        go_on = False
        for b in step:
            if b:
                out.setdefault(binds_key(b), b)
            elif last:
                raise ValueError("the last sub-question's answers each bind the parent's values: a decision list ends")
            else:
                go_on = True
        if not go_on:
            break
    return tuple(out[k] for k in sorted(out))


# The decision list as registered nodes over a fixed set of inputs: the last sub-question alone, and one sub-question
# before the composition of those after it (whose bindings are all terminal), so a split of any length is a chain.


@node(id="compose.last", version=1, inputs=["step"])
def compose_last(step: Sequence[Binds]) -> tuple[Binds, ...]:
    """The last sub-question's feasible bindings, each terminal."""
    return decision_list([step])


@node(id="compose.then", version=1, inputs=["head", "tail"])
def compose_then(head: Sequence[Binds], tail: Sequence[Binds]) -> tuple[Binds, ...]:
    """One sub-question, then the composition of the sub-questions after it."""
    return decision_list([head, tail])
