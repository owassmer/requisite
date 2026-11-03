"""The business's dated cash, day by day over the planning horizon, from deep discovery's model of it
(requisite/library/nodes/materiality.py): construction's liquidity program reads the business on a dated grid (addendum
A06 §6.1), and that grid is the day (decisions, "The exposure register's dated cash impact is monthly": construction's
optimizer takes the business's cash on its own dated grid, where a payout late within a month shows as a breach).

Deep's model accrues a driver's monthly amount evenly over the month's days and starts an event's change on a day
(deep.cash_start: the event's day, its lag and the line's payment terms), so each day's flow is exact under it:
- A driver's baseline flow on a day is its calendar month's amount (quantity x price per month x the month's seasonality
  factor: deep.monthly_amounts) over the month's days, received on a revenue line and paid on a cost line. The business
  is in operation, so its receivables and payables keep that flow running from the horizon's first day; payment terms
  delay only when a change first reaches cash, as deep's model has it.
- An exposure's change on a day is its share of the driver's flow that day, from the day the change starts, at its two
  ends: the least (the end that lowers cash: the most negative share starting earliest, or the least positive share
  starting latest) and the greatest. Each end is one corner of deep.cash_impact's box of share and start, the same corner
  every day, so each end's path is one the model allows, and a month's sum of an end is deep.cash_impact's end for that
  month exactly. An end with no bound is None from the day the change can start.
- Where several events occur, their changes add: each event's change to a driver is a share of the driver's current
  amount, as the sensitivity questions ask it (requisite/library/templates/deep.json: "as a share of the driver's current
  amount"). That two events which would move one line by one underlying change are not counted twice is an assumption a
  construction record states where it applies (addendum A06 §6.1: register rows are standalone and "never sum[med] into
  portfolio loss without checking overlap and dependence"; the changes are composed here through the one driver model,
  never by summing register rows).

Every value is exact (Fractions); rounding to money happens where construction reads a path into its program.
"""

from __future__ import annotations

import calendar as cal
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from typing import Any
from zoneinfo import ZoneInfo

from requisite.compute.registry import node

Bound = Fraction | None
Flow = tuple[date, Fraction]  # a day and its cash flow, positive in
Change = tuple[date, Bound, Bound]  # a day and its change's least and greatest end


def _q(x: Any, what: str) -> Fraction:
    if isinstance(x, bool) or isinstance(x, float) or not isinstance(x, Fraction | Decimal | int | str):
        raise TypeError(f"{what} is an exact number, not {type(x).__name__}")
    return Fraction(str(x)) if isinstance(x, str) else Fraction(x)


def _bound(x: Any, what: str) -> Bound:
    return None if x is None else _q(x, what)


def horizon_days(days: tuple[date, date]) -> list[date]:
    first, last = days
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def _rates(amounts: Sequence[tuple[date, Any]]) -> dict[date, Fraction]:
    """Each calendar month's amount per day of the month."""
    return {m: _q(a, "a monthly amount") / cal.monthrange(m.year, m.month)[1] for m, a in amounts}


def _rate(rates: Mapping[date, Fraction], d: date) -> Fraction:
    m = date(d.year, d.month, 1)
    if m not in rates:
        raise ValueError(f"no amount for the month of {d}: the amounts cover the horizon's months")
    return rates[m]


def _line(line: str) -> int:
    if line not in ("cost", "revenue"):
        raise ValueError(f"a driver's line is cost or revenue, not {line!r}")
    return 1 if line == "revenue" else -1


@node(id="cash.daily_flows", version=1, inputs=["amounts", "days", "line"],
      monotone={"amounts": "increasing"})  # fmt: skip
def daily_flows(amounts: Sequence[tuple[date, Any]], days: tuple[date, date], line: str) -> tuple[Flow, ...]:
    """A driver's baseline cash flow on each day of the horizon: received on a revenue line, paid (negative) on a cost
    line."""
    sign, rates = _line(line), _rates(amounts)
    return tuple((d, sign * _rate(rates, d)) for d in horizon_days(days))


@node(id="cash.add_flows", version=1, inputs=["a", "b"])
def add_flows(a: Sequence[Flow], b: Sequence[Flow]) -> tuple[Flow, ...]:
    """Two flows over the same days, added day by day."""
    if [d for d, _ in a] != [d for d, _ in b]:
        raise ValueError("flows are added over the same days")
    return tuple((d, x + y) for (d, x), (_, y) in zip(a, b, strict=True))


@node(id="cash.daily_impact", version=1, inputs=["amounts", "days", "line", "effect", "start"])
def daily_impact(amounts: Sequence[tuple[date, Any]], days: tuple[date, date], line: str,
                 effect: tuple[Any, Any], start: tuple[date, date | None]) -> tuple[Change, ...]:  # fmt: skip
    """An exposure's cash change on each day of the horizon at its least and greatest ends (deep.effect's share of the
    driver's amount, from deep.cash_start's earliest or latest day)."""
    sign, rates = _line(line), _rates(amounts)
    lo, hi = _bound(effect[0], "an effect"), _bound(effect[1], "an effect")
    if lo is not None and hi is not None and lo > hi:
        raise ValueError("an effect is an interval [low, high]")
    # The change as a share of the day's cash flow (positive: more cash), its ends in order.
    k_lo, k_hi = (lo, hi) if sign > 0 else ((None if hi is None else -hi), (None if lo is None else -lo))
    earliest, latest = start

    def end(k: Bound, lowest: bool, d: date, rate: Fraction) -> Bound:
        if d < earliest:
            return Fraction(0)  # no corner has started
        if k is None:
            return None
        # The least end takes a cash-lowering share as early as it can start, a cash-raising one as late; the greatest
        # end the reverse.
        early = (k < 0) if lowest else (k > 0)
        begun = d >= earliest if early else (latest is not None and d >= latest)
        return k * rate if begun else Fraction(0)

    out: list[Change] = []
    for d in horizon_days(days):
        rate = _rate(rates, d)
        out.append((d, end(k_lo, True, d, rate), end(k_hi, False, d, rate)))
    return tuple(out)


@node(id="cash.no_change", version=1, inputs=["days"])
def no_change(days: tuple[date, date]) -> tuple[Change, ...]:
    """No change on any day: a scenario in which no event occurs."""
    return tuple((d, Fraction(0), Fraction(0)) for d in horizon_days(days))


@node(id="cash.add_change", version=1, inputs=["change", "impact"])
def add_change(change: Sequence[Change], impact: Sequence[Change]) -> tuple[Change, ...]:
    """A scenario's change with one more exposure's: the least ends added and the greatest ends added, day by day (an
    unbounded end stays unbounded)."""
    if [d for d, _, _ in change] != [d for d, _, _ in impact]:
        raise ValueError("changes are added over the same days")

    def plus(x: Bound, y: Bound) -> Bound:
        return None if x is None or y is None else x + y

    return tuple((d, plus(a, c), plus(b, e)) for (d, a, b), (_, c, e) in zip(change, impact, strict=True))


@node(id="cash.liquidity", version=1, inputs=["cash", "credit"], monotone={"cash": "increasing", "credit": "increasing"})
def liquidity(cash: Any, credit: Any) -> Fraction:
    """The liquidity the business holds at the horizon's start: cash on hand and the undrawn credit line (as deep's breach
    flag counts it: deep.breach)."""
    c, k = _q(cash, "cash"), _q(credit, "a credit line")
    if c < 0 or k < 0:
        raise ValueError("cash on hand and a credit line are not negative")
    return c + k


@node(id="cash.reserve", version=1, inputs=["floor", "days"])
def reserve(floor: Sequence[tuple[date, Any]], days: tuple[date, date]) -> tuple[Flow, ...]:
    """The liquidity floor in effect on each day: the latest point of the decision spec's floor on or before it, and none
    before its first (as deep.breach reads it)."""
    points = sorted((d, _q(a, "a floor amount")) for d, a in floor)
    out = []
    for d in horizon_days(days):
        before = [a for p, a in points if p <= d]
        out.append((d, before[-1] if before else Fraction(0)))
    return tuple(out)


def local_day(at: datetime, tz: str) -> date:
    return at.astimezone(ZoneInfo(tz)).date()


def _midnight(at: datetime, tz: str) -> bool:
    local = at.astimezone(ZoneInfo(tz))
    return (local.hour, local.minute, local.second, local.microsecond) == (0, 0, 0, 0)


def event_day(as_of: datetime, window: Any) -> tuple[str, date, date | None]:
    """When a pending event can happen, as days of its window's zone: the earliest, its window's first day or as_of's
    day, whichever is later (it is pending), and the latest, its window's last day (None with no end). A window that
    excludes its end at a midnight ends the day before; one that excludes its start at a midnight starts the day after."""
    tz = "UTC" if window is None else window.tz
    pending = local_day(as_of, tz)
    first = pending
    if window is not None and window.start is not None:
        start = local_day(window.start, tz)
        if not window.start_closed and _midnight(window.start, tz):
            start += timedelta(days=1)
        first = max(pending, start)
    last = None
    if window is not None and window.end is not None:
        last = local_day(window.end, tz)
        if not window.end_closed and _midnight(window.end, tz):
            last -= timedelta(days=1)
        if last < first:
            raise ValueError("the event's window ends before as_of: it is not pending")
    return tz, first, last
