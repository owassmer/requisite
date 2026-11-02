"""Kleene's strong three-valued logic (spec §6.3; invariant I6: unknown stays unknown).

`Kleene` cannot be used as a Python truth value: `if x:` on it raises. An unknown therefore cannot be coerced to false
by an `if`, an `or`, `all()` or `any()` anywhere in the product (ours: I6 enforced by the type, not by care).
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Literal


class Kleene(Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    def __bool__(self) -> bool:
        raise TypeError(
            "a three-valued truth has no Python truth value (spec I6: unknown is never coerced to false); "
            "compare it with Kleene.TRUE, or use is_true / is_false"
        )

    def __and__(self, other: Kleene) -> Kleene:
        return conj((self, other))

    def __or__(self, other: Kleene) -> Kleene:
        return disj((self, other))

    def __invert__(self) -> Kleene:
        return neg(self)

    @property
    def is_true(self) -> bool:
        return self is Kleene.TRUE

    @property
    def is_false(self) -> bool:
        return self is Kleene.FALSE

    @property
    def is_unknown(self) -> bool:
        return self is Kleene.UNKNOWN


def conj(values: Iterable[Kleene]) -> Kleene:
    """AND: false if any is false; otherwise unknown if any is unknown; otherwise true (spec §6.3 table). The empty
    conjunction is true."""
    vals = tuple(values)
    _check(vals)
    if any(v is Kleene.FALSE for v in vals):
        return Kleene.FALSE
    if any(v is Kleene.UNKNOWN for v in vals):
        return Kleene.UNKNOWN
    return Kleene.TRUE


def disj(values: Iterable[Kleene]) -> Kleene:
    """OR: true if any is true; otherwise unknown if any is unknown; otherwise false (spec §6.3 table). The empty
    disjunction is false."""
    vals = tuple(values)
    _check(vals)
    if any(v is Kleene.TRUE for v in vals):
        return Kleene.TRUE
    if any(v is Kleene.UNKNOWN for v in vals):
        return Kleene.UNKNOWN
    return Kleene.FALSE


def neg(value: Kleene) -> Kleene:
    """NOT: swaps true and false; unknown stays unknown."""
    _check((value,))
    return {Kleene.TRUE: Kleene.FALSE, Kleene.FALSE: Kleene.TRUE, Kleene.UNKNOWN: Kleene.UNKNOWN}[value]


def of(value: bool | None) -> Kleene:
    """From the kernels' plain form (True, False, None for unknown)."""
    if value is None:
        return Kleene.UNKNOWN
    if value is True:
        return Kleene.TRUE
    if value is False:
        return Kleene.FALSE
    raise TypeError(f"a three-valued truth is True, False or None, not {value!r}")


def plain(value: Kleene) -> bool | None:
    """To the kernels' plain form."""
    _check((value,))
    return {Kleene.TRUE: True, Kleene.FALSE: False, Kleene.UNKNOWN: None}[value]


AnswerStatus = Literal["affirmative", "negative", "not_established", "contested"]


def from_status(status: AnswerStatus) -> Kleene:
    """An interpretive or resolve answer's status as a truth (spec §6.3): affirmative is true, negative is false, and
    not_established or contested is unknown."""
    table: dict[str, Kleene] = {
        "affirmative": Kleene.TRUE,
        "negative": Kleene.FALSE,
        "not_established": Kleene.UNKNOWN,
        "contested": Kleene.UNKNOWN,
    }
    if status not in table:
        raise ValueError(f"not an answer status: {status!r}")
    return table[status]


def _check(values: tuple[object, ...]) -> None:
    for v in values:
        if not isinstance(v, Kleene):
            raise TypeError(f"three-valued logic takes Kleene values, not {type(v).__name__} ({v!r})")
