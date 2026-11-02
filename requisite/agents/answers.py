"""What each agent role answers (its --json-schema): a draft of a definition record plus what the human reviewing it needs.
These are answer shapes, never committed: code turns an answer into a Proposal of the definition's own record, which
the store's gates judge (spec I1). Each answer embeds the record's fields by using the record itself where it can, so
no field is defined twice."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from requisite.core import ContractIR, TimeWindow
from requisite.core.contract import Support
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


class HarvestDraft(_Answer):
    """contract_reader: a filed contract read into the IR (core.HarvestedContract's support classification, and the IR
    itself as a core.ContractIR, validated as one)."""

    support: Support
    assumptions: Sequence[str]
    unsupported_terms: Sequence[str]
    ir: ContractIR | None


class ProseSection(_Answer):
    name: Literal[
        "underlying_and_definitions", "source_hierarchy_and_conflict_rule", "trigger", "examples", "timing",
        "unknown_at_expiry", "revisions",
    ]  # fmt: skip
    text: str
    ir_nodes: Sequence[str]


class ProseDraft(_Answer):
    """clause_drafter: the rulebook prose of an IR, its sections in spec §8.5's order, each naming the IR nodes it states."""

    sections: Sequence[ProseSection]


class PacketDocument(_Answer):
    title: str
    source: str
    published_at: datetime
    text: str


class PacketDraft(_Answer):
    """scenario_renderer: one scenario's synthetic evidence packet (quarantined: spec I4)."""

    documents: Sequence[PacketDocument]
