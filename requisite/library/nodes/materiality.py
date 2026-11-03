"""Deep discovery's materiality: how an event's effect on one driver of a business becomes dated cash-flow deltas over the
planning horizon, and those deltas with the event's probability an expected impact, a tail impact and a liquidity-breach
flag (spec §7.1 step 4: "Code maps magnitude bands through the driver tree into dated cash-flow deltas. Combined with
probability sources, these yield expected impact, tail impact and liquidity-breach flags over the horizon"; I7).

Every quantity is an interval of exact rationals, never a band's midpoint (addendum A04 §4.1: "Keep interval-valued
consequences"), and an end with no bound is None: a band "40% or more" bounds a change from below only. The bands come
from the answers the case's questions leave feasible, each as its binds (requisite/library/templates/deep.json), so a
question not yet answered contributes every answer it could take (A03 §3.3).

The model, per driver-event pair (an exposure):
- The driver's amount for each calendar month of the horizon is quantity x price, per month (a quarter's or a year's
  amount spread evenly over its months), times that calendar month's seasonality factor (core.LiquidityProfile).
- The event changes the driver by a share delta of its amount: + for an increase, - for a decrease, 0 for none, within
  the magnitude band; a decrease is at most 100% (a driver's amount is not negative); where the effect is conditional, it
  may not follow at all, so the band's lower end is zero.
- The change starts when the event happens (within the event proposition's window, and after the state's as_of, since
  the event is pending), plus the lag band, plus the payment terms of the driver's line (payable days for a cost,
  receivable days for revenue: a changed amount moves cash that much later), and lasts to the horizon's end. Dates are
  calendar days in the horizon's time zone; a lag in months is added on the calendar (a day past a month's end is its
  last day), its lower end rounded down and its upper end up.
- A month's cash delta is the change times the month's amount times the share of the month's days that are in the
  horizon and on or after the start: at least the share after the latest start, at most the share after the earliest.
  A cost's change is paid (cash out); revenue's is received.
Within one exposure every month's lower ends come from one corner of the band and the start, and every upper end from
another, so sums and running sums of the monthly ends are the exact bounds of the totals and the paths.

The horizon loss is minus the sum of the cash deltas. Given the event's probability p, the loss is L with probability p
and 0 otherwise: the expected impact is p x L and the tail impact CVaR_alpha of that two-point loss (addendum A06 §6.3,
by requisite/optimize/cvar.py), each monotone in L, so each interval is its value at L's ends. A decision with no alpha
(an objective that is not a CVaR) takes the tail as the loss if the event occurs, CVaR's limit as alpha approaches one.
The breach flag says whether, if the event occurs, the exposure's running loss alone exceeds the cash and undrawn credit
at the horizon's start less the decision's liquidity floor in effect (core.DecisionSpec.liquidity_floor; no floor before
its first point) in some month: true where its least loss does, false where its greatest never does, otherwise unknown
(I6). The business's other operating cash flows are not part of it: the driver tree need not be the whole P&L, so the
flag measures the exposure against the liquidity the business holds, standalone.
"""

from __future__ import annotations

import calendar as cal
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from typing import Any
from zoneinfo import ZoneInfo

from requisite.compute.registry import node
from requisite.core import TimeWindow
from requisite.optimize.cvar import discrete_cvar

Bound = Fraction | None
Interval = tuple[Bound, Bound]
Month = tuple[date, Bound, Bound]  # (first day, low, high)


def _q(x: Any, what: str) -> Fraction:
    if isinstance(x, bool) or isinstance(x, float) or not isinstance(x, Fraction | Decimal | int | str):
        raise TypeError(f"{what} is an exact number, not {type(x).__name__}")
    return Fraction(str(x)) if isinstance(x, str) else Fraction(x)


def _bound(x: Any, what: str) -> Bound:
    return None if x is None else _q(x, what)


def _union(binds: Sequence[Mapping[str, Any]], key: str) -> tuple[Fraction, Bound]:
    """The union of the intervals the feasible answers bind under `key` (lows' minimum, highs' maximum, None
    unbounded)."""
    if not binds:
        raise ValueError(f"no feasible answer binds {key}")
    lows, highs = [], []
    for b in binds:
        if key not in b:
            raise ValueError(f"a feasible answer binds {sorted(b)}, not {key}")
        lo, hi = b[key]
        lows.append(_q(lo, key))
        highs.append(_bound(hi, key))
    return min(lows), (None if any(h is None for h in highs) else max(h for h in highs if h is not None))


def _local_day(at: datetime, tz: str) -> date:
    return at.astimezone(ZoneInfo(tz)).date()


@node(id="deep.horizon_days", version=1, inputs=["horizon"])
def horizon_days(horizon: TimeWindow) -> tuple[date, date]:
    """The horizon's first and last calendar day in its time zone. An open end at a local midnight excludes that day."""
    if horizon.start is None or horizon.end is None:
        raise ValueError("a planning horizon has a start and an end")
    first, last = _local_day(horizon.start, horizon.tz), _local_day(horizon.end, horizon.tz)
    end = horizon.end.astimezone(ZoneInfo(horizon.tz))
    if not horizon.end_closed and (end.hour, end.minute, end.second, end.microsecond) == (0, 0, 0, 0):
        last -= timedelta(days=1)
    if last < first:
        raise ValueError("the horizon holds no whole day")
    return first, last


def _months(first: date, last: date) -> list[date]:
    out, m = [], date(first.year, first.month, 1)
    while m <= last:
        out.append(m)
        m = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
    return out


PER_MONTH = {"month": Fraction(1), "quarter": Fraction(1, 3), "year": Fraction(1, 12)}


@node(id="deep.monthly_amounts", version=1, inputs=["quantity", "price", "period", "seasonality", "days"],
      monotone={"quantity": "increasing", "price": "increasing"})  # fmt: skip
def monthly_amounts(quantity: Any, price: Any, period: str, seasonality: Sequence[Any], days: tuple[date, date]
                    ) -> tuple[tuple[date, Fraction], ...]:  # fmt: skip
    """The driver's amount in each calendar month the horizon touches: quantity x price per month, times the month's
    seasonality factor."""
    q, p = _q(quantity, "quantity"), _q(price, "price")
    if q < 0 or p < 0:
        raise ValueError("a driver's quantity and price are not negative")
    if period not in PER_MONTH:
        raise ValueError(f"a driver's period is one of {sorted(PER_MONTH)}, not {period!r}")
    if len(seasonality) != 12:
        raise ValueError("seasonality is twelve monthly factors")
    factors = [_q(s, "a seasonality factor") for s in seasonality]
    return tuple((m, q * p * PER_MONTH[period] * factors[m.month - 1]) for m in _months(*days))


@node(id="deep.effect", version=1, inputs=["direction", "magnitude", "conditional"])
def effect(direction: Sequence[Mapping[str, Any]], magnitude: Sequence[Mapping[str, Any]],
           conditional: Sequence[Mapping[str, Any]]) -> Interval:  # fmt: skip
    """The change in the driver, as a share of its amount, over the feasible direction, magnitude and conditionality."""
    signs = set()
    for b in direction:
        d = b.get("direction")
        if d not in ("increase", "decrease", "none"):
            raise ValueError(f"a direction binds increase, decrease or none, not {d!r}")
        signs.add(d)
    if not signs:
        raise ValueError("no feasible direction")
    m_lo, m_hi = _union(magnitude, "magnitude")
    if m_lo < 0 or (m_hi is not None and m_hi < m_lo):
        raise ValueError("a magnitude band is a nonnegative interval")
    flags = set()
    for b in conditional:
        c = b.get("conditional")
        if not isinstance(c, bool):
            raise ValueError(f"conditionality binds true or false, not {c!r}")
        flags.add(c)
    if not flags:
        raise ValueError("no feasible conditionality")
    if True in flags:
        m_lo = Fraction(0)  # the further condition may not hold, and then the event does not move the driver
    lows: list[Fraction] = []
    highs: list[Bound] = []
    if "increase" in signs:
        lows.append(m_lo)
        highs.append(m_hi)
    if "decrease" in signs:
        dec_hi = Fraction(1) if m_hi is None else min(m_hi, Fraction(1))
        lows.append(-dec_hi)
        highs.append(-min(m_lo, Fraction(1)))
    if "none" in signs:
        lows.append(Fraction(0))
        highs.append(Fraction(0))
    hi = None if any(h is None for h in highs) else max(h for h in highs if h is not None)
    return min(lows), hi


def _add_months(day: date, months: int) -> date:
    y, m = divmod(day.month - 1 + months, 12)
    year, month = day.year + y, m + 1
    return date(year, month, min(day.day, cal.monthrange(year, month)[1]))


@node(id="deep.cash_start", version=1, inputs=["as_of", "window", "lag", "terms_days", "horizon"])
def cash_start(as_of: datetime, window: TimeWindow | None, lag: Sequence[Mapping[str, Any]], terms_days: Any,
               horizon: TimeWindow) -> tuple[date, date | None]:  # fmt: skip
    """The earliest and latest day the changed driver first moves cash (None: no latest day is bounded)."""
    tz = horizon.tz
    pending = _local_day(as_of, tz)
    start = pending if window is None or window.start is None else max(pending, _local_day(window.start, tz))
    end = None if window is None or window.end is None else _local_day(window.end, tz)
    if end is not None and end < pending:
        raise ValueError("the event's window ends before the state's as_of: it is not pending")
    lo, hi = _union(lag, "lag_months")
    terms = _q(terms_days, "payment terms in days")
    if lo < 0 or terms < 0 or (hi is not None and hi < lo):
        raise ValueError("a lag band and payment terms are not negative")
    earliest = _add_months(start, math.floor(lo)) + timedelta(days=math.floor(terms))
    if end is None or hi is None:
        return earliest, None
    return earliest, _add_months(end, math.ceil(hi)) + timedelta(days=math.ceil(terms))


def _share(month: date, days: tuple[date, date], start: date | None) -> Fraction:
    """The share of the month's days in the horizon and on or after `start` (None: none)."""
    if start is None:
        return Fraction(0)
    n = cal.monthrange(month.year, month.month)[1]
    first = max(days[0], month, start)
    last = min(days[1], date(month.year, month.month, n))
    return Fraction(max(0, (last - first).days + 1), n)


def _times(lo: Bound, hi: Bound, f_lo: Fraction, f_hi: Fraction) -> Interval:
    """[lo, hi] x [f_lo, f_hi] with 0 <= f_lo <= f_hi; a None end is unbounded."""
    low: Bound
    if lo is None:
        low = None if f_hi > 0 else Fraction(0)
    else:
        low = lo * (f_hi if lo < 0 else f_lo)
    high: Bound
    if hi is None:
        high = None if f_hi > 0 else Fraction(0)
    else:
        high = hi * (f_hi if hi > 0 else f_lo)
    return low, high


def _neg(x: Interval) -> Interval:
    lo, hi = x
    return (None if hi is None else -hi), (None if lo is None else -lo)


@node(id="deep.cash_impact", version=1, inputs=["amounts", "days", "line", "effect", "start"])
def cash_impact(amounts: Sequence[tuple[date, Any]], days: tuple[date, date], line: str, effect: Interval,
                start: tuple[date, date | None]) -> tuple[Month, ...]:  # fmt: skip
    """Each month's cash-flow delta of the exposure, [low, high], negative where cash goes out."""
    if line not in ("cost", "revenue"):
        raise ValueError(f"a driver's line is cost or revenue, not {line!r}")
    earliest, latest = start
    out: list[Month] = []
    for month, amount in amounts:
        a = _q(amount, "a monthly amount")
        change = _times(effect[0], effect[1], _share(month, days, latest), _share(month, days, earliest))
        scaled = (None if change[0] is None else change[0] * a, None if change[1] is None else change[1] * a)
        lo, hi = _neg(scaled) if line == "cost" else scaled
        out.append((month, lo, hi))
    return tuple(out)


def _sum(xs: Sequence[Bound]) -> Bound:
    return None if any(x is None for x in xs) else sum((x for x in xs if x is not None), Fraction(0))


@node(id="deep.horizon_loss", version=1, inputs=["cash"])
def horizon_loss(cash: Sequence[Month]) -> Interval:
    """The exposure's loss over the horizon if the event occurs: minus the sum of its cash deltas."""
    return _neg((_sum([lo for _, lo, _ in cash]), _sum([hi for _, _, hi in cash])))


def _p(probability: Any) -> Fraction:
    p = _q(probability, "a probability")
    if not 0 <= p <= 1:
        raise ValueError(f"a probability lies in [0, 1], not {p}")
    return p


@node(id="deep.expected_impact", version=1, inputs=["probability", "loss"])
def expected_impact(probability: Any, loss: Interval) -> Interval:
    """p x L at L's ends: the event's loss weighted by its probability (and nothing where it does not occur)."""
    p = _p(probability)

    def at(x: Bound) -> Bound:
        return Fraction(0) if p == 0 else (None if x is None else p * x)

    return at(loss[0]), at(loss[1])


@node(id="deep.tail_impact", version=1, inputs=["probability", "alpha", "loss"])
def tail_impact(probability: Any, alpha: Any, loss: Interval) -> Interval:
    """CVaR_alpha of the loss that is L with probability p and 0 otherwise, at L's ends; with no alpha, the loss if
    the event occurs (and nothing where it cannot)."""
    p = _p(probability)

    def at(x: Bound) -> Bound:
        if p == 0:
            return Fraction(0)
        if x is None:
            return None
        if alpha is None:
            return max(x, Fraction(0)) if p < 1 else x
        return discrete_cvar([Fraction(0), x], [1 - p, p], _q(alpha, "alpha"))

    return at(loss[0]), at(loss[1])


def _floor(points: Sequence[tuple[date, Any]], day: date) -> Fraction:
    before = [(d, _q(a, "a floor amount")) for d, a in points if d <= day]
    return max(before)[1] if before else Fraction(0)


@node(id="deep.breach", version=1, inputs=["cash", "days", "probability", "cash_on_hand", "credit_line", "floor"])
def breach(cash: Sequence[Month], days: tuple[date, date], probability: Any, cash_on_hand: Any, credit_line: Any,
           floor: Sequence[tuple[date, Any]]) -> str:  # fmt: skip
    """Whether the exposure's running loss, if the event occurs, exceeds the liquidity held less the floor in some
    month: "true", "false" or "unknown" (I6)."""
    if _p(probability) == 0:
        return "false"
    held = _q(cash_on_hand, "cash") + _q(credit_line, "credit line")
    least: Bound = Fraction(0)
    most: Bound = Fraction(0)
    certain, possible = False, False
    for month, lo, hi in cash:
        last = min(days[1], date(month.year, month.month, cal.monthrange(month.year, month.month)[1]))
        room = held - _floor(floor, last)
        least = None if least is None or hi is None else least - hi  # the running loss's lower end
        most = None if most is None or lo is None else most - lo  # its upper end
        if least is not None and least > room:
            certain = True
        if most is None or most > room:
            possible = True
    return "true" if certain else ("unknown" if possible else "false")
