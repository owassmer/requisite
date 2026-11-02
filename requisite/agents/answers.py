"""What each agent role answers (its --json-schema): a draft of a definition record plus what the human reviewing it needs.
These are answer shapes, never committed: code turns an answer into a Proposal of the definition's own record, which
the store's gates judge (spec I1). Each answer embeds the record's fields by using the record itself where it can, so
no field is defined twice."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from pydantic import BaseModel, ConfigDict

from requisite.core import TimeWindow
from requisite.core.governance import CommentStatus, DocumentType


class _Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObjectiveDraft(_Answer):
    """objective_drafter: a wide run's objective for G1 (core.WideObjective's fields; its id and version are code's)."""

    population: str
    events: str
    horizon: TimeWindow
    loss_definition: str
    estimand: str
    open_questions: Sequence[str]


class FrameDraft(_Answer):
    """frame_drafter: the frame criteria an approved objective implies (core.FrameCriteria's fields; its id, objective,
    source and as_of are code's)."""

    search_terms: Sequence[str]
    document_types: Sequence[DocumentType]
    posted_from: date
    posted_to: date | None
    agencies: Sequence[str]
    comment_status: CommentStatus
    min_comments: int
    rationale: str
    expected_misses: str


class ParaphraseDraft(_Answer):
    id: str
    target: str
    meanings: dict[str, str]


class ParaphrasesDraft(_Answer):
    """paraphrase_writer: two rewordings of a template (core.ParaphraseSet holds them with the original, p0)."""

    paraphrases: Sequence[ParaphraseDraft]
