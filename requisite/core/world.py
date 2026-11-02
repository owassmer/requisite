"""The proposition layer: propositions, constraints, probability estimates and event trees (spec §4 "world model", §5;
amended by addendum A04)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import model_validator

from requisite.core.base import Instant, Probability, Record
from requisite.core.judgment import TimeWindow


class Proposition(Record):
    """A precisely defined proposition, classification, quantity, date or event (spec §4; addendum §1.1 X_k).

    parents: the conditioning set; an event tree in v1 (spec §4, §5). probability: the id of the ProbabilityEstimate that
    supplies its probability, where it has one. This replaces spec §4's `prob_source`, whose `market_price` route
    contradicted I9 (addendum A04 §4.4).
    """

    id: str
    text: str
    window: TimeWindow | None = None
    domain: Literal["bool", "category", "number", "date"]
    parents: Sequence[str] = ()
    constraints: Sequence[str] = ()
    probability: str | None = None


class Constraint(Record):
    """A logical constraint between propositions (spec §4 Proposition.constraints: "implies / excludes / partition";
    spec §5: the proposition layer's validator is SAT consistency). implies: members[0] implies members[1]. excludes: at
    most one of the members holds. partition: exactly one holds."""

    id: str
    kind: Literal["implies", "excludes", "partition"]
    members: Sequence[str]

    @model_validator(mode="after")
    def _arity(self) -> Constraint:
        if self.kind == "implies" and len(self.members) != 2:
            raise ValueError("implies relates exactly two propositions")
        if self.kind != "implies" and len(self.members) < 2:
            raise ValueError(f"{self.kind} relates at least two propositions")
        if len(set(self.members)) != len(self.members):
            raise ValueError("a constraint's members are distinct")
        return self


class ConditioningStep(Record):
    proposition: str
    value: str


class ProbabilityEstimate(Record):
    """A probability in a declared model (addendum A04 §4.1 and §12).

    meaning: physical (a real-world probability), pricing (a risk-neutral or market-implied quantity) or hypothetical (an
    assumption). source: where it comes from; a price reaches a probability only through a price_to_probability model
    node with an explicit risk-premium assumption (I9; A04 §4.4), named in `source_ref`. value is exact (decisions,
    "Numbers"); `interval`, where set, is the uncertainty representation (A04 §4.1: a range is not a point).
    """

    id: str
    proposition: str
    branch_value: str
    conditioning: Sequence[ConditioningStep]
    meaning: Literal["physical", "pricing", "hypothetical"]
    source: Literal["judgment", "observed_frequency", "base_rate", "assumed", "price_to_probability_model"]
    source_ref: str
    value: Probability
    interval: tuple[Probability, Probability] | None = None
    evidence_hash: str
    as_of: Instant

    @model_validator(mode="after")
    def _interval(self) -> ProbabilityEstimate:
        if self.interval is not None and not self.interval[0] <= self.value <= self.interval[1]:
            raise ValueError("an estimate's value lies in its interval")
        if self.meaning == "physical" and self.source == "price_to_probability_model" and not self.source_ref:
            raise ValueError("a probability from prices names its conversion model (I9)")
        return self


class Branch(Record):
    """One outgoing event of a branch point: the proposition taking a value, its conditional probability given the path
    (a ProbabilityEstimate id), and what follows it."""

    proposition: str
    value: str
    probability: str
    then: BranchPoint | None = None


class BranchPoint(Record):
    """A node of an event tree. Its branches partition the parent state: mutually exclusive and exhaustive within the
    declared model (addendum A04 §4.2). Unknown is never a branch (A04 §4.3): an unresolved parameter is resolved by
    evidence, an approved assumption, a range of feasible values, or blocked optimization, never a third outcome."""

    id: str
    branches: Sequence[Branch]

    @model_validator(mode="after")
    def _branches(self) -> BranchPoint:
        if len(self.branches) < 2:
            raise ValueError("a branch point has at least two outgoing events")
        if any(b.value.lower() in {"unknown", "not_established"} for b in self.branches):
            raise ValueError("unknown is an information status, never a branch (addendum A04 §4.3)")
        keys = [(b.proposition, b.value) for b in self.branches]
        if len(set(keys)) != len(keys):
            raise ValueError("a branch point's events are distinct")
        return self


class EventTree(Record):
    """Sequential stage gates with alternative routes as branches; each branch probability is conditional on its path,
    never a product of independent stage probabilities (spec §5 "Probability structure in v1"; addendum A04 §4.2)."""

    id: str
    version: int
    root: BranchPoint


Branch.model_rebuild()
