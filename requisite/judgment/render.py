"""The canonical state a judge reads (spec §6.1), and the question as it is asked.

Every provider sees the same state, split into what is observed, what is assumed and what is defined, rendered by code.
The render axis of the stability harness varies only presentation, never content: the format (text or JSON) and the
order of the observed claims (spec §2.1 "Render: order and format ... of the same evidence"). Assumptions appear only
under their own labelled heading (I5). A conditional forecast P(B | A) is an ordinary forecast question whose state
lists A under ASSUMED (spec §6.1).

The question goes to the provider as a question, never inside the state: its target with its window, and its closed
answer set with each answer's meaning. A paraphrase replaces the target's wording and the answers' meanings with a
frozen, reviewed rewording (spec §2.1 "Paraphrase"; core.Paraphrase, from the template's core.ParaphraseSet); the answer
values stay the same, so distributions compare across configurations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from requisite.core import (
    Artifact,
    Assumption,
    AtomicQuestion,
    Claim,
    Definition,
    JudgmentState,
    Paraphrase,
    SyntheticClaim,
    SyntheticJudgmentState,
    TimeWindow,
)

Format = Literal["text", "json"]
Order = Literal["as_listed", "reversed"]


@dataclass(frozen=True)
class Render:
    """One point of the render axis: a format and an order of the observed claims."""

    id: str
    format: Format
    order: Order


# The default render axis: two presentations (spec §2.1: render, default count 2).
TEXT = Render("text-as-listed", "text", "as_listed")
JSON = Render("json-reversed", "json", "reversed")
RENDERS = (TEXT, JSON)


@dataclass(frozen=True)
class ObservedLine:
    """An observed claim with what the reader needs to weigh it: who says it, when it was published, where it is."""

    ref: str
    text: str
    authority: str
    published: str
    artifact: str


@dataclass(frozen=True)
class RenderedState:
    """What a provider receives as the state (a JSON object, as the System One request takes it), with the render that
    produced it. `text` is the canonical text form, kept for the dossier and for reading."""

    render: Render
    state: Mapping[str, Any]
    text: str


def _when(at: datetime | None) -> str:
    return "publication date not stated" if at is None else at.date().isoformat()


def observed_lines(
    state: JudgmentState | SyntheticJudgmentState,
    claims: Mapping[str, Claim | SyntheticClaim],
    artifacts: Mapping[str, Any],
) -> list[ObservedLine]:
    """The state's observed claims, labelled [C1], [C2] ... in the order the state lists them. A synthetic case's
    document carries its own publication date in its text, as a fuzzing packet's documents do (requisite/fuzz/run.py),
    so its line says the date is as the document states it: the judge reads a packet as the evidence of the world it
    renders, and the quarantine is the type's (I4), not a label put before the judge."""
    lines = []
    for i, cid in enumerate(state.observed, 1):
        claim = claims[cid]
        artifact = artifacts.get(claim.artifact)
        published = _when(artifact.published_at) if isinstance(artifact, Artifact) else "as stated in the document"
        lines.append(ObservedLine(f"C{i}", claim.text, claim.authority, published, claim.artifact[-12:]))
    return lines


def render_state(
    state: JudgmentState | SyntheticJudgmentState,
    render: Render,
    *,
    claims: Mapping[str, Claim | SyntheticClaim],
    artifacts: Mapping[str, Any],
    definitions: Mapping[str, Definition],
    distractor: ObservedLine | None = None,
) -> RenderedState:
    """The canonical state in one presentation (spec §6.1). A distractor, when the configuration has one, is one more
    observed line: a plausible but irrelevant sourced claim (spec §2.1 "Distractor")."""
    observed = observed_lines(state, claims, artifacts)
    if distractor is not None:
        observed.append(distractor)
    if render.order == "reversed":
        observed = list(reversed(observed))
    assumed = [(f"A{i}", a) for i, a in enumerate(state.assumed, 1)]
    defined = [(f"D{i}", definitions[d]) for i, d in enumerate(state.definitions, 1)]
    text = _text(state.as_of, observed, assumed, defined)
    if render.format == "text":
        payload: dict[str, Any] = {"state": text}
    else:
        payload = {
            "as_of": state.as_of.isoformat(),
            "observed": [
                {"ref": o.ref, "claim": o.text, "authority": o.authority, "published": o.published,
                 "artifact": o.artifact}
                for o in observed
            ],  # fmt: skip
            "assumed_for_this_question_not_established": [{"ref": r, "assumption": a.text} for r, a in assumed],
            "definitions": [{"ref": r, "term": d.term, "definition": d.text} for r, d in defined],
        }
    return RenderedState(render, payload, text)


def _text(
    as_of: datetime,
    observed: Sequence[ObservedLine],
    assumed: Sequence[tuple[str, Assumption]],
    defined: Sequence[tuple[str, Definition]],
) -> str:
    out = [f"AS OF: {as_of.isoformat()}", "OBSERVED (sourced; may be relied on):"]
    out += [f"  [{o.ref}] {o.text} — {o.authority}, published {o.published}, artifact {o.artifact}" for o in observed]
    if not observed:
        out.append("  (none)")
    out.append("ASSUMED FOR THIS QUESTION (hypothetical; not established):")
    out += [f"  [{r}] {a.text}" for r, a in assumed] or ["  (none)"]
    out.append("DEFINITIONS:")
    out += [f"  [{r}] {d.term}: {d.text}" for r, d in defined] or ["  (none)"]
    return "\n".join(out)


def window_text(window: TimeWindow | None) -> str:
    if window is None:
        return ""
    lo = "[" if window.start_closed else "("
    hi = "]" if window.end_closed else ")"
    start = window.start.isoformat() if window.start else "-inf"
    end = window.end.isoformat() if window.end else "+inf"
    return f" Time window: {lo}{start}, {end}{hi}, days counted in {window.tz}."


@dataclass(frozen=True)
class AskedQuestion:
    """A question as asked: the instruction a provider reads and the closed answer set with each meaning."""

    name: str
    instructions: str
    answers: Mapping[str, str]


def ask(question: AtomicQuestion, paraphrase: Paraphrase) -> AskedQuestion:
    """The question in one paraphrase (spec §2.1). Its answer values are the question's own, whatever the wording."""
    values = [a.value for a in question.answers]
    if set(paraphrase.meanings) != set(values):
        raise ValueError(f"paraphrase {paraphrase.id} words answers {sorted(paraphrase.meanings)}, not {sorted(values)}")
    instructions = (
        f"{paraphrase.target.strip()}{window_text(question.window)}\n"
        "Answer from the state alone. Rely on what is under OBSERVED; treat what is under ASSUMED as hypothetical, "
        "given only for this question. If the state does not establish the answer, choose the answer that says so."
        if question.task != "forecast"
        else f"{paraphrase.target.strip()}{window_text(question.window)}\n"
        "Give your probability for each answer as of the state's AS OF instant, taking what is under ASSUMED as given."
    )
    return AskedQuestion(f"{question.id}.v{question.version}", instructions, {v: paraphrase.meanings[v] for v in values})
