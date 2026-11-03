"""CountTrue over three-valued daily values: an interval, never a count of the known days alone (addendum A05 §5.3).

A daily predicate is three-valued (spec §6.3): ``True``, ``False``, or ``None``, the Kleene unknown that not_established
and contested map to. With k days known true and u days unknown, the count of true days lies in [k, k + u]; returning
k would coerce every unknown day to false (I6). The interval stays until the contract's missing-data procedure
resolves it. Anything other than ``True``, ``False`` and ``None`` is refused, so a 0, 1 or empty string cannot stand in
for a truth value.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class CountBounds:
    """The number of true days, as the interval [low, high] the known and unknown days allow."""

    low: int  # days known true: k
    high: int  # days known true or unknown: k + u
    true: int
    false: int
    unknown: int

    @property
    def interval(self) -> tuple[int, int]:
        return (self.low, self.high)


def count_true(values: Iterable[bool | None]) -> CountBounds:
    """CountTrue over three-valued daily values: [k, k + u]."""
    true = false = unknown = 0
    for day, v in enumerate(values):
        if v is True:
            true += 1
        elif v is False:
            false += 1
        elif v is None:
            unknown += 1
        else:
            raise TypeError(f"day {day} is {v!r}: a three-valued day is True, False or None (unknown)")
    return CountBounds(low=true, high=true + unknown, true=true, false=false, unknown=unknown)
