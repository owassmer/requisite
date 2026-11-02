"""Test doubles of the scenario renderer and the decision models, for the fuzz run's tests: each test's own input, never
a cassette (the task's rule: a test double stands where a test needs a response no provider gave).

The renderer writes a scenario as one plain record: each published value it fixes, and one line per condition the
rendering check will ask, stating the condition and the answer meaning the scenario intends ("FINDING [statement] =>
meaning"). A test that needs a packet that does not encode its scenario rewrites those lines (`corrupt`).

The judges read such a record as a careful reader would. Asked a rendering-check question, every judge finds the line
for the condition the question states and answers what it says, 0.9 on it; a question no line states is not
established. Asked to resolve, each judge applies the test's own reading of the rules to the findings (`resolve`): the
test's model of a resolver, which is where a test seeds a divergence or an ambiguity.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from requisite import core
from requisite.agents import runner
from requisite.agents.answers import PacketDocument, PacketDraft
from requisite.judgment.providers import Ask

AT = datetime(2027, 1, 1, tzinfo=UTC)
FINDING = re.compile(r"FINDING \[(?P<statement>.*?)\] => (?P<meaning>.*)")
HOLDS = "the documents establish that the condition holds"
FAILS = "the documents establish that the condition does not hold"
UNSET = "the documents do not establish whether the condition holds"


class Renderer:
    """The scenario renderer's double. `corrupt(input, attempt)` may return other condition lines for one rendering."""

    def __init__(self, corrupt: Callable[[Mapping[str, Any], int], list[dict[str, str]] | None] | None = None,
                 published: datetime | None = None) -> None:  # fmt: skip
        self.corrupt = corrupt
        self.published = published
        self.calls: list[dict[str, Any]] = []
        self.attempts: dict[str, int] = {}
        self.lock = threading.Lock()

    def __call__(self, role: str, input: Mapping[str, Any], output: type, effort: str = "high") -> runner.AgentResult[Any]:
        assert role == "scenario_renderer" and output is PacketDraft
        base = core.canonical_json({k: v for k, v in input.items() if k not in ("previous_packet", "not_recovered")})
        with self.lock:
            self.calls.append(dict(input))
            attempt = self.attempts.get(base, 0)
            self.attempts[base] = attempt + 1
        conditions = list(input["conditions"])
        if self.corrupt is not None:
            conditions = self.corrupt(input, attempt) or conditions
        lines = [f"Value of {f['observable']} for {f['for']}: {f['value']} (source {f['source']}, published "
                 f"{f['published_at']})" for f in input["published_values"]]  # fmt: skip
        lines += [f"FINDING [{' '.join(c['condition'].split())}] => {c['value']}" for c in conditions]
        when = self.published or datetime.fromisoformat(input["evidence_as_of"])
        doc = PacketDocument(title="Record of the scenario", source="the record keeper", published_at=when,
                             text="\n".join(lines))  # fmt: skip
        run = "double-" + core.sha256(core.canonical_json(dict(input)))
        return runner.AgentResult(role, PacketDraft(documents=[doc]), run, "test-double", effort, True)


def findings(text: str) -> dict[str, str]:
    """The FINDING lines of a rendered state: statement -> the meaning it states. A claim's last line carries the
    rendered line's own suffix (" — authority, published ..."), which is not the finding's."""
    out = {}
    for line in text.splitlines():
        m = FINDING.search(line)
        if m:
            out[m.group("statement")] = m.group("meaning").split(" — ")[0]
    return out


def distribution(answers: list[str], weights: Mapping[str, float]) -> dict[str, float]:
    """A distribution in cents: the given answers' weights, the rest of the mass split over the other answers."""
    cents = {a: round(weights[a] * 100) for a in answers if a in weights}
    others = [a for a in answers if a not in weights]
    left = 100 - sum(cents.values())
    for i, a in enumerate(others):
        cents[a] = left // len(others) + (1 if i < left % len(others) else 0)
    return {a: cents[a] / 100 for a in answers}


def judgment(a: Ask, dist: Mapping[str, float]) -> core.Judgment:
    return core.Judgment(question=core.Ref(id=a.question.id, version=a.question.version), state_hash=a.state_hash,
                         config=a.config, reported=dict(dist), distribution=dict(dist), precision=0.01,
                         response="test-double", latency_ms=0, cost_usd=Decimal(0), created_at=AT)  # fmt: skip


Resolve = Callable[[str, Mapping[str, str], Ask], Mapping[str, float]]


class Judges:
    """The decision models' double: a careful reader of the renderer double's records, resolving by `resolve(rules,
    findings, ask)`, which returns weights on the resolver template's answers."""

    def __init__(self, resolve: Resolve) -> None:
        self.resolve = resolve
        self.asked: list[Ask] = []
        self.lock = threading.Lock()

    def __call__(self, a: Ask) -> core.Judgment:
        with self.lock:
            self.asked.append(a)
        answers = list(a.asked.answers)
        found = findings(a.rendered.text)
        if a.question.task == "resolve":
            return judgment(a, distribution(answers, self.resolve(str(a.question.bindings["rules"]), found, a)))
        target = " ".join(a.question.target.split())
        meaning = next((m for statement, m in found.items() if statement in target), None)
        by_meaning = {m: v for v, m in a.asked.answers.items()}
        top = by_meaning.get(meaning or "", "not_established")
        return judgment(a, distribution(answers, {top: 0.9}))


def truth(found: Mapping[str, str], key: str) -> str:
    """The sense of the finding whose statement contains `key`: "true", "false" or "unknown"."""
    meaning = next(m for s, m in found.items() if key in s)
    return {HOLDS: "true", FAILS: "false"}.get(meaning, "unknown")
