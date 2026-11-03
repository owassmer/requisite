"""Business days for a contract's lags: weekends and the named holiday calendar's holidays are skipped (addendum A05
§5.4: "Define ... day-count conventions"; core.contract.Timing).

The one registered calendar is ``us_federal``: the legal public holidays of 5 U.S.C. §6103(a), computed by rule rather
than listed, each on the day a Monday-to-Friday week observes it. A holiday on a Saturday is observed the Friday before
(§6103(b)(1)); one on a Sunday the Monday after (Executive Order 11582 §3(a), which OPM applies beside §6103(b)). Sources:
research/sources/5-u-s-c-6103-holidays-lii-cornell.md (the eleven holidays and the Saturday rule),
research/sources/executive-order-11582-observance-of-holidays-1971-as-printed.md (the Sunday rule),
research/sources/u-s-office-of-personnel-management-federal-holidays-holiday-.md (OPM's statement of both).
Inauguration Day (§6103(c)) is a holiday only in and near Washington, so it is not in this calendar
(research/sources/5-u-s-c-6103-c-inauguration-day-lii-cornell.md).

The holidays have changed over time, and each year is computed under the law in force for it: the Monday holidays
from 1971 (Pub. L. 90-363), Veterans Day on the fourth Monday of October from 1971 to 1977 and on November 11 from 1978
(Pub. L. 94-97), the Birthday of Martin Luther King, Jr. from 1986 (Pub. L. 98-144, effective "the first January 1 that
occurs after the two-year period following" November 2, 1983), Juneteenth from 2021 (Pub. L. 117-17;
research/sources/5-u-s-c-6103-amendment-note-lii-cornell.md). Earlier years are refused rather than guessed.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import cache
from typing import Literal

Calendar = Literal["us_federal"]
CALENDARS: tuple[Calendar, ...] = ("us_federal",)
FIRST_YEAR = 1971  # the Monday Holidays Act's effective year; the calendar's form before it is not computed here


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th given weekday (Monday 0) of a month; n = -1 is the last."""
    if n > 0:
        first = date(year, month, 1)
        return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    last = date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


MONDAY, THURSDAY = 0, 3


def legal_public_holidays(year: int) -> dict[date, str]:
    """5 U.S.C. §6103(a)'s holidays of a year, on their legal dates (before any observed-day shift)."""
    if year < FIRST_YEAR:
        raise ValueError(f"the us_federal calendar is computed from {FIRST_YEAR}; {year} is before it")
    days = {
        date(year, 1, 1): "New Year's Day",
        _nth_weekday(year, 2, MONDAY, 3): "Washington's Birthday",
        _nth_weekday(year, 5, MONDAY, -1): "Memorial Day",
        date(year, 7, 4): "Independence Day",
        _nth_weekday(year, 9, MONDAY, 1): "Labor Day",
        _nth_weekday(year, 10, MONDAY, 2): "Columbus Day",
        _nth_weekday(year, 11, THURSDAY, 4): "Thanksgiving Day",
        date(year, 12, 25): "Christmas Day",
    }
    if year >= 1986:
        days[_nth_weekday(year, 1, MONDAY, 3)] = "Birthday of Martin Luther King, Jr."
    if year >= 2021:
        days[date(year, 6, 19)] = "Juneteenth National Independence Day"
    if year >= 1978:
        days[date(year, 11, 11)] = "Veterans Day"
    else:
        days[_nth_weekday(year, 10, MONDAY, 4)] = "Veterans Day"
    return days


def _observed(day: date) -> date:
    """The day a Monday-to-Friday week observes a holiday: Saturday to the Friday before, Sunday to the Monday after."""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


@cache
def observed_holidays(year: int, calendar: Calendar = "us_federal") -> frozenset[date]:
    """The days of a year on which the calendar's holidays are observed. A holiday of the next year can be observed in
    this one (New Year's Day on a Saturday is observed on December 31), so both years' holidays are shifted."""
    if calendar not in CALENDARS:
        raise ValueError(f"no holiday calendar {calendar!r} is registered; registered: {list(CALENDARS)}")
    years = [y for y in (year, year + 1) if y >= FIRST_YEAR]
    if year < FIRST_YEAR:
        raise ValueError(f"the us_federal calendar is computed from {FIRST_YEAR}; {year} is before it")
    shifted = {_observed(d) for y in years for d in legal_public_holidays(y)}
    return frozenset(d for d in shifted if d.year == year)


def is_business_day(day: date, calendar: Calendar) -> bool:
    return day.weekday() < 5 and day not in observed_holidays(day.year, calendar)


def add_business_days(day: date, n: int, calendar: Calendar) -> date:
    """The n-th business day after `day`. A lag of zero business days is the day itself when it is a business day, and
    otherwise the next business day (a business-day lag never lands on a weekend or holiday)."""
    if n < 0:
        raise ValueError(f"a lag is a non-negative number of days, not {n}")
    d = day
    while not is_business_day(d, calendar):
        d += timedelta(days=1)
    if d != day and n > 0:
        n -= 1  # rolling forward to the first business day already moved past `day`: it is the first one after it
    for _ in range(n):
        d += timedelta(days=1)
        while not is_business_day(d, calendar):
            d += timedelta(days=1)
    return d


def add_days(day: date, n: int, convention: Literal["calendar", "business"], calendar: Calendar | None) -> date:
    """`n` days after `day` in the contract's day count (core.contract.Timing.lag_days)."""
    if n < 0:
        raise ValueError(f"a lag is a non-negative number of days, not {n}")
    if convention == "calendar":
        return day + timedelta(days=n)
    if calendar is None:
        raise ValueError("a business-day count names its holiday calendar")
    return add_business_days(day, n, calendar)
