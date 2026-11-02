"""The library's question templates (spec §9: QuestionTemplate), read from requisite/library/templates/*.json, and their
instantiation into atomic questions (spec §4: AtomicQuestion.template_id and bindings). Templates are data, written
over kinds, never a case (design/posture.md); a case supplies their bindings."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

from requisite import core

TEMPLATES = Path(__file__).resolve().parent / "templates"


@cache
def all_templates() -> dict[str, core.QuestionTemplate]:
    out: dict[str, core.QuestionTemplate] = {}
    for path in sorted(TEMPLATES.glob("*.json")):
        for raw in json.loads(path.read_text())["templates"]:
            t = core.QuestionTemplate(**raw)
            if t.id in out:
                raise ValueError(f"template {t.id} is defined twice in the library")
            out[t.id] = t
    return out


def template(template_id: str) -> core.QuestionTemplate:
    try:
        return all_templates()[template_id]
    except KeyError:
        raise KeyError(f"no library template {template_id!r}") from None


def fill(text: str, bindings: Mapping[str, Any]) -> str:
    """A template's text with its slots filled; every slot must be bound, and nothing else."""
    return text.format_map(dict(bindings))


def instantiate(
    t: core.QuestionTemplate, bindings: Mapping[str, Any], *, question_id: str, paraphrase_set: str | None = None
) -> core.AtomicQuestion:
    """An atomic question from a template and its bindings (spec §4), its answers the template's."""
    if set(bindings) != set(t.slots):
        raise ValueError(f"template {t.id} binds {sorted(t.slots)}, not {sorted(bindings)}")
    return core.AtomicQuestion(
        id=question_id,
        version=t.version,
        template_id=t.id,
        bindings=dict(bindings),
        task=t.task,
        target=fill(t.target, bindings),
        answers=list(t.answers),
        paraphrase_set=paraphrase_set,
    )


def original(t: core.QuestionTemplate) -> core.Paraphrase:
    return core.Paraphrase(id="p0", target=t.target, meanings={a.value: a.meaning for a in t.answers})
