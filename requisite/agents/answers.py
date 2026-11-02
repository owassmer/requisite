"""What each agent role answers (its --json-schema): a draft of a definition record plus what the human reviewing it needs.
These are answer shapes, never committed: code turns an answer into a Proposal of the definition's own record, which
the store's gates judge (spec I1). Each answer embeds the record's fields by using the record itself where it can, so
no field is defined twice."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from requisite.core import ContractFamily, ContractIR, HarvestedContract, ListingBinding, SettlementBinding, TimeWindow
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
    """contract_reader: a filed series read into the IR as an instrument family (core.ContractFamily's fields that are
    the reader's): the support classification with its assumptions and unsupported terms (core.HarvestedContract), the
    representative market's IR (its id and version are code's), the bindings that say which IR value each per-market
    listing field fills (core.ListingBinding), and what a market record's settlement value reports
    (core.SettlementBinding). Validated as the core family it drafts, so an answer the definition refuses is refused
    here, with the definition's reasons."""

    support: Support
    assumptions: Sequence[str]
    unsupported_terms: Sequence[str]
    ir: ContractIR | None
    bindings: Sequence[ListingBinding]
    settlement: SettlementBinding | None

    @model_validator(mode="after")
    def _family(self) -> HarvestDraft:
        contract = HarvestedContract(
            filing="draft", filed_terms=(), support=self.support, ir=self.ir.ref if self.ir is not None else None,
            assumptions=self.assumptions, unsupported_terms=self.unsupported_terms,
            equivalence="unsupported_or_unassessed",
        )  # fmt: skip
        ContractFamily(contract=contract, representative=self.ir, bindings=self.bindings, settlement=self.settlement)
        return self


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


class FamilyDraft(_Answer):
    id: str
    name: str
    channel: str
    mechanism: str
    measures: Sequence[str]


class FamiliesDraft(_Answer):
    """family_drafter: a wide run's event families over the measures its items address (core.EventFamilies; its frame
    is code's). Every measure given is in exactly one family."""

    families: Sequence[FamilyDraft]
    rationale: str


class ImpactDraft(_Answer):
    quote: str
    amount: str
    currency: str
    period: Literal["year", "month", "one_time", "other"]


class ImpactsDraft(_Answer):
    """impact_extractor: the cost impacts a filing states of the measure it comments on, each quoted exactly, with its
    amount, currency and period; none where the filing states none."""

    impacts: Sequence[ImpactDraft]


class DraftAnswer(_Answer):
    """One answer of a drafted question (core.AnswerValue's fields): its value, what choosing it asserts, its status, and
    what it binds (the parent question's values it implies, or nothing where the next sub-question decides)."""

    value: str
    meaning: str
    status: Literal["affirmative", "negative", "not_established"]
    binds: dict[str, Any]


class DraftQuestion(_Answer):
    """A drafted question: its target (subject, predicate and window stated explicitly), its closed answer set, and two
    meaning-preserving rewordings of the target and the answers' meanings (ids p1 and p2) for its frozen paraphrase set."""

    target: str
    answers: Sequence[DraftAnswer]
    paraphrases: Sequence[ParaphraseDraft]


class QuestionsDraft(_Answer):
    """question_refiner: the questions that replace one question of a case (core.AtomicQuestion with `replaces`): one
    refined question, or a split's sub-questions in the order they are asked (requisite/library/nodes/compose.py)."""

    questions: Sequence[DraftQuestion]
    rationale: str


class CategoriesDraft(_Answer):
    """market_scout, first pass: the listing categories in which a contract settling on the event could be listed."""

    categories: Sequence[str]
    rationale: str


class MarketsDraft(_Answer):
    """market_scout, second pass: the listed series (by ticker) whose settlement could track the event, or none."""

    tickers: Sequence[str]
    rationale: str
