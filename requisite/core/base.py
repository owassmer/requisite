"""What every canonical record shares: immutability, a closed set of fields, canonical hashing, exact probabilities.

Sources are cited beside each rule: `spec §n` (the build spec v3), `addendum Ann` or `addendum §n` (the mathematical
addendum, accepted by Owen on November 1), `decisions` (design/decisions.md), or `ours` with the reason.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from fractions import Fraction
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer, WithJsonSchema, model_validator


class Record(BaseModel):
    """A canonical record. Frozen, since committed state changes only by a new record (spec §3 I1, I2). Closed to fields
    it does not declare, since a field one package invents and another ignores is a second schema (addendum §12: "Do not
    create a second schema namespace that competes with §4"). Strict, so that a value is never coerced into a type it is
    not (ours: a "1" that becomes True is the kind of quiet coercion I6 forbids for unknowns)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, arbitrary_types_allowed=False)

    @model_validator(mode="after")
    def _freeze(self) -> Record:
        """Immutable all the way down: every sequence becomes a tuple and every mapping a FrozenDict, so no field of a
        committed record can be changed in place, a list of observed claims included (spec I4: "enforce this at the type
        level"; spec §4: AcceptanceRecord is "immutable"). `frozen=True` alone guards only the attributes."""
        for name in type(self).model_fields:
            object.__setattr__(self, name, freeze(getattr(self, name)))
        return self

    def canonical(self) -> str:
        """The record as canonical JSON: sorted keys, no whitespace, every value in its JSON form."""
        return canonical_json(self.model_dump(mode="json"))

    def content_hash(self) -> str:
        """sha256 of the canonical JSON (spec §4 JudgmentState.content_hash; spec §5: outputs keyed by content hashes)."""
        return sha256(self.canonical())


class FrozenDict(dict[Any, Any]):
    """A mapping that refuses every change after it is built. A dict subclass, so it serializes as a dict."""

    def _immutable(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("a committed record is immutable: build a new record instead")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable


def freeze(value: Any) -> Any:
    """A value with every list made a tuple and every dict a FrozenDict, recursively. Records are frozen by their own
    validation."""
    if isinstance(value, BaseModel):
        return value
    if isinstance(value, dict):
        return value if isinstance(value, FrozenDict) else FrozenDict({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(freeze(v) for v in value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256(text: str | bytes) -> str:
    data = text.encode() if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- exact probabilities (decisions, "Numbers")


def _exact_probability(x: object) -> Fraction:
    """An exact probability in [0, 1]. Accepted: Fraction, int, Decimal, or the string forms "n/d" and "0.25" (as JSON
    carries them). Refused: a float, whose exact value is not what was meant, and a bool (decisions: scenario and branch
    probabilities are exact, never normalized; addendum §4.2)."""
    if isinstance(x, bool) or isinstance(x, float):
        raise ValueError(f"a probability is exact (Fraction, Decimal, int or 'n/d'), not {type(x).__name__}")
    if isinstance(x, str):
        try:
            x = Fraction(x)
        except (ValueError, ZeroDivisionError) as e:
            raise ValueError(f"not an exact probability: {x!r}") from e
    if isinstance(x, Decimal):
        if not x.is_finite():
            raise ValueError(f"a probability is finite, not {x}")
        x = Fraction(x)
    if isinstance(x, int):
        x = Fraction(x)
    if not isinstance(x, Fraction):
        raise ValueError(f"a probability is exact, not {type(x).__name__}")
    if not 0 <= x <= 1:
        raise ValueError(f"a probability lies in [0, 1], not {x}")
    return x


Probability = Annotated[
    Fraction,
    BeforeValidator(_exact_probability),
    PlainSerializer(lambda f: f"{f.numerator}/{f.denominator}", return_type=str),
    WithJsonSchema({"type": "string", "pattern": r"^\d+/\d+$"}),
]


def _exact_rational(x: object) -> Fraction:
    """An exact rational of any size: a CVaR, an expectation (decisions, "Numbers"). A float is refused."""
    if isinstance(x, bool) or isinstance(x, float):
        raise ValueError(f"an exact quantity is a Fraction, Decimal, int or 'n/d', not {type(x).__name__}")
    if isinstance(x, str | int):
        try:
            return Fraction(x)
        except (ValueError, ZeroDivisionError) as e:
            raise ValueError(f"not an exact quantity: {x!r}") from e
    if isinstance(x, Decimal):
        if not x.is_finite():
            raise ValueError(f"an exact quantity is finite, not {x}")
        return Fraction(x)
    if isinstance(x, Fraction):
        return x
    raise ValueError(f"an exact quantity is a Fraction, Decimal, int or 'n/d', not {type(x).__name__}")


Rational = Annotated[
    Fraction,
    BeforeValidator(_exact_rational),
    PlainSerializer(lambda f: f"{f.numerator}/{f.denominator}", return_type=str),
    WithJsonSchema({"type": "string", "pattern": r"^-?\d+/\d+$"}),
]


def exact_sum_is_one(values: list[Fraction], what: str) -> None:
    """A partition's probabilities sum to exactly one; they are never normalized (addendum §4.2)."""
    total = sum(values, Fraction(0))
    if total != 1:
        raise ValueError(f"{what} sum to {total}, not 1; incoherent probabilities are rejected, never normalized")


# ---------------------------------------------------------------- money and time


def _money(x: object) -> Decimal:
    """Money is Decimal (decisions, "Numbers"; spec §4 uses Decimal). A float is refused; an int or a decimal string is
    read exactly."""
    if isinstance(x, bool) or isinstance(x, float):
        raise ValueError(f"money is Decimal, not {type(x).__name__}")
    if isinstance(x, int | str):
        try:
            x = Decimal(x)
        except ArithmeticError as e:  # decimal.InvalidOperation
            raise ValueError(f"not a decimal amount: {x!r}") from e
    if not isinstance(x, Decimal) or not x.is_finite():
        raise ValueError(f"money is a finite Decimal, not {x!r}")
    return x


Money = Annotated[
    Decimal,
    BeforeValidator(_money),
    PlainSerializer(lambda d: str(d), return_type=str),
    WithJsonSchema({"type": "string"}),
]


def _aware(x: object) -> datetime:
    """A datetime with its time zone (spec §14: explicit time zones; addendum A05 §5.4). A naive datetime is refused,
    since its instant is not defined."""
    if isinstance(x, str):
        x = datetime.fromisoformat(x)
    if not isinstance(x, datetime) or x.tzinfo is None or x.utcoffset() is None:
        raise ValueError(f"an instant needs its time zone: {x!r}")
    return x


Instant = Annotated[datetime, BeforeValidator(_aware)]


def _day(x: object) -> date:
    if isinstance(x, str):
        x = date.fromisoformat(x)
    if isinstance(x, datetime) or not isinstance(x, date):
        raise ValueError(f"a calendar day is a date, not {x!r}")
    return x


Day = Annotated[date, BeforeValidator(_day)]


class Ref(Record):
    """A versioned reference to another record or policy (addendum §5.1 VersionedRef)."""

    id: str
    version: int
