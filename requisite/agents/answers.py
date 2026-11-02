"""What each agent role answers (its --json-schema): a draft of a definition record plus what the human reviewing it needs.
These are answer shapes, never committed: code turns an answer into a Proposal of the definition's own record, which
the store's gates judge (spec I1). Each answer embeds the record's fields by using the record itself where it can, so
no field is defined twice."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationInfo, model_validator

from requisite.core import (
    Assumption,
    ContractFamily,
    ContractIR,
    Driver,
    EntityModel,
    HarvestedContract,
    LiquidityProfile,
    ListingBinding,
    Parameter,
    SettlementBinding,
    TimeWindow,
)
from requisite.core.base import Money
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


class SectionsRevised(_Answer):
    """clause_reviser: the clause drafter's revision of one rulebook section a fuzz run suspects (spec §8.6 step 7). sections:
    each section it revised, whole (the one it was asked to revise; another only where the failing scenarios cannot be
    fixed without it, which code refuses and reports). not_decided_by_the_ir: each case the readers were asked about that
    the IR itself does not decide, so no wording can settle it without adding to the contract. rationale: what the readers
    hesitated on, and how the revision settles it."""

    sections: Sequence[ProseSection]
    not_decided_by_the_ir: Sequence[str]
    rationale: str


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
    """One cost impact a filing states (core.QuantifiedImpact): the filer's words that state it, and the amount of money
    they state, read from those words as an exact decimal in units of the currency ("$2.5 million a year" is 2500000,
    per year)."""

    quote: str
    amount: Money
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
    period: Literal["year", "month", "one_time", "other"]


class ImpactsDraft(_Answer):
    """impact_extractor: the cost impacts a filing states of the measure it comments on, each quoted exactly, with its
    amount, currency and period; none where the filing states none."""

    impacts: Sequence[ImpactDraft]

    @model_validator(mode="after")
    def _quoted(self, info: ValidationInfo) -> ImpactsDraft:
        texts = (info.context or {}).get("texts")
        if texts is not None:
            self.check_quotes(texts)
        return self

    def check_quotes(self, texts: Sequence[str]) -> None:
        """Each quote is the filer's words: copied exactly from one part of the comment's text the extractor was given
        (provenance, checked as text; the amount is the extractor's reading of those words, never matched by code). The
        runner gives the texts as context, so a quote that is not there goes back to the extractor with this reason."""
        missing = [i.quote for i in self.impacts if not (i.quote and any(i.quote in t for t in texts))]
        if missing:
            raise ValueError(f"each quote is copied character for character from one part of the comment; not found: "
                             f"{missing}")  # fmt: skip


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


class StatedBy(_Answer):
    """A claim that states a number: its id, as given, and the words of its text that state the number, exactly as they
    appear there. The id is checked as the definition's ClaimId when the number is built (NumberDraft.parameter), not by a
    schema pattern: structured outputs refuse complex quantifiers such as a 64-digit hash's (research/sources/
    anthropic-claude-platform-docs-structured-outputs-json-schem.md)."""

    id: str
    words: str


class Assumed(_Answer):
    """An assumed number's assumption: what is assumed, and the reason no claim states it."""

    text: str
    reason: str


class NumberDraft(_Answer):
    """One number of the entity model (core.Parameter): its name, its value as an exact decimal and its unit, and either
    the claims that state it (each with the words that state it) or an assumption with its reason; never both, never
    neither."""

    name: str
    value: Money
    unit: str
    claims: Sequence[StatedBy]
    assumption: Assumed | None

    @model_validator(mode="after")
    def _basis(self) -> NumberDraft:
        self.parameter()  # the definition's rule: sourced by claims or assumed, exactly one (core.Parameter)
        if self.assumption is not None and not (self.assumption.text.strip() and self.assumption.reason.strip()):
            raise ValueError(f"{self.name}: an assumption states what it assumes and the reason no claim states it")
        if any(not c.words.strip() for c in self.claims):
            raise ValueError(f"{self.name}: each claim cited gives the words of its text that state the number")
        return self

    def parameter(self) -> Parameter:
        """The number as the definition's record. An entity model's assumption fixes a number, not a proposition, so it
        binds none (core.Assumption.binds)."""
        a = self.assumption
        return Parameter(name=self.name, value=self.value, unit=self.unit, claims=[c.id for c in self.claims],
                         assumption=None if a is None else Assumption(text=a.text, binds={}, reason=a.reason))  # fmt: skip


class DriverDraft(_Answer):
    """One revenue or cost line (core.Driver): quantity times price, each per `period`. id: a short name of the line,
    lowercase letters, digits and underscores, distinct among the drivers (it names the line in every question and
    computation of the case)."""

    id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]
    line: Literal["revenue", "cost"]  # core.Driver.line
    name: str
    quantity: NumberDraft
    price: NumberDraft
    period: Literal["month", "quarter", "year"]  # core.Driver.period

    def driver(self) -> Driver:
        return Driver(id=self.id, line=self.line, name=self.name, quantity=self.quantity.parameter(),
                      price=self.price.parameter(), period=self.period)  # fmt: skip


class LiquidityDraft(_Answer):
    """The business's liquidity (core.LiquidityProfile): cash on hand, its credit line, receivable and payable days, and
    twelve calendar-month seasonality factors, January first, averaging exactly one."""

    cash: NumberDraft
    credit_line: NumberDraft
    receivable_days: NumberDraft
    payable_days: NumberDraft
    seasonality: Sequence[NumberDraft]

    @model_validator(mode="after")
    def _profile(self) -> LiquidityDraft:
        self.profile()
        return self

    def profile(self) -> LiquidityProfile:
        return LiquidityProfile(cash=self.cash.parameter(), credit_line=self.credit_line.parameter(),
                                receivable_days=self.receivable_days.parameter(),
                                payable_days=self.payable_days.parameter(),
                                seasonality=[s.parameter() for s in self.seasonality])  # fmt: skip


class EntityModelDraft(_Answer):
    """decomposer: a business's driver tree and liquidity profile (core.EntityModel's fields that are the agent's; its id,
    version and entity are code's, and its horizon and loss definition are the case's objective's, approved at G1).
    currency: the ISO 4217 code of the currency the business's amounts are in. rationale: why these lines and no others,
    for the human who confirms the model at G1. Validated as the model it drafts, so an answer the definition refuses is
    refused here, with the definition's reasons."""

    drivers: Sequence[DriverDraft]
    liquidity: LiquidityDraft
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
    rationale: str

    @model_validator(mode="after")
    def _model(self, info: ValidationInfo) -> EntityModelDraft:
        unbounded = TimeWindow(start=None, end=None, start_closed=True, end_closed=False)
        self.model(id="draft", version=1, entity="draft", horizon=unbounded, loss_definition="draft")
        claims = (info.context or {}).get("claims")
        if claims is not None:
            self.check_quotes(claims)
        return self

    def numbers(self) -> list[tuple[str, NumberDraft]]:
        """Every number of the draft with where it sits."""
        out: list[tuple[str, NumberDraft]] = []
        for d in self.drivers:
            out += [(f"driver {d.id} quantity", d.quantity), (f"driver {d.id} price", d.price)]
        liq = self.liquidity
        out += [("cash", liq.cash), ("credit line", liq.credit_line), ("receivable days", liq.receivable_days),
                ("payable days", liq.payable_days)]  # fmt: skip
        out += [(f"seasonality month {m}", s) for m, s in enumerate(liq.seasonality, 1)]
        return out

    def check_quotes(self, claims: Mapping[str, str]) -> None:
        """Every sourced number cites claims of the case (id to text), each containing, exactly, the words quoted from it
        (provenance, checked as text; the number is the decomposer's reading of those words, never matched by code).
        The runner gives the claims as context, so a citation that fails goes back to the decomposer with this reason."""
        for where, n in self.numbers():
            for c in n.claims:
                if c.id not in claims:
                    raise ValueError(f"{where} ({n.name} = {n.value} {n.unit}) cites claim {c.id}, which this case does "
                                     "not hold: its claims are the lead's filing")  # fmt: skip
                if c.words not in claims[c.id]:
                    raise ValueError(f"{where} ({n.name} = {n.value} {n.unit}) quotes {c.words!r} from claim {c.id}, "
                                     "whose text does not contain those words")  # fmt: skip

    def model(self, *, id: str, version: int, entity: str, horizon: TimeWindow, loss_definition: str) -> EntityModel:
        return EntityModel(id=id, version=version, entity=entity, drivers=[d.driver() for d in self.drivers],
                           liquidity=self.liquidity.profile(), horizon=horizon, loss_definition=loss_definition,
                           currency=self.currency)  # fmt: skip
