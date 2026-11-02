"""The us_federal business-day calendar (requisite/contract/calendar.py): 5 U.S.C. §6103(a)'s holidays computed by rule,
observed on the Friday before a Saturday (§6103(b)(1)) and the Monday after a Sunday (Executive Order 11582), checked
against the Office of Personnel Management's published schedules (research/sources/u-s-office-of-personnel-management-*)."""

from datetime import date

import pytest

from requisite.contract import calendar


def test_the_observed_holidays_are_opms_published_schedules():
    # OPM, 2026 Holiday Schedule: Independence Day (Saturday) is observed Friday July 3.
    assert calendar.observed_holidays(2026) == {
        date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3),
        date(2026, 9, 7), date(2026, 10, 12), date(2026, 11, 11), date(2026, 11, 26), date(2026, 12, 25),
    }  # fmt: skip
    # OPM, 2027 Holiday Schedule: Juneteenth and Christmas fall on Saturdays, Independence Day on a Sunday. And OPM's 2028
    # schedule observes New Year's Day 2028, a Saturday, on Friday December 31, 2027: a holiday observed in 2027.
    assert calendar.observed_holidays(2027) == {
        date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5),
        date(2027, 9, 6), date(2027, 10, 11), date(2027, 11, 11), date(2027, 11, 25), date(2027, 12, 24),
        date(2027, 12, 31),
    }  # fmt: skip
    assert date(2028, 1, 1) not in calendar.observed_holidays(2028)  # a Saturday, observed the year before
    # OPM, 2021: Juneteenth's first year, observed Friday June 18.
    assert date(2021, 6, 18) in calendar.observed_holidays(2021)
    assert date(2020, 6, 19) not in calendar.observed_holidays(2020)


def test_each_year_is_computed_under_the_law_in_force_for_it():
    assert calendar.legal_public_holidays(1975)[date(1975, 10, 27)] == "Veterans Day"  # fourth Monday of October
    assert date(1975, 11, 11) not in calendar.legal_public_holidays(1975)
    assert date(1978, 11, 11) in calendar.legal_public_holidays(1978)
    assert "Birthday of Martin Luther King, Jr." not in calendar.legal_public_holidays(1985).values()
    assert calendar.legal_public_holidays(1986)[date(1986, 1, 20)] == "Birthday of Martin Luther King, Jr."
    assert len(calendar.legal_public_holidays(2030)) == 11
    with pytest.raises(ValueError, match="computed from 1971"):
        calendar.observed_holidays(1970)


def test_inauguration_day_is_not_in_the_calendar():
    # 5 U.S.C. §6103(c) makes it a holiday only in and near the District of Columbia. January 20, 2033 is a Thursday.
    assert calendar.is_business_day(date(2033, 1, 20), "us_federal")


def test_business_days_count_past_weekends_and_holidays():
    # Friday July 2, 2027; Monday July 5 observes Independence Day.
    assert calendar.add_business_days(date(2027, 7, 2), 1, "us_federal") == date(2027, 7, 6)
    assert calendar.add_business_days(date(2027, 7, 2), 0, "us_federal") == date(2027, 7, 2)
    assert calendar.add_business_days(date(2027, 7, 3), 0, "us_federal") == date(2027, 7, 6)  # a Saturday rolls forward
    assert calendar.add_business_days(date(2027, 7, 3), 1, "us_federal") == date(2027, 7, 6)  # and is the first one after
    assert calendar.add_business_days(date(2027, 12, 23), 1, "us_federal") == date(2027, 12, 27)
    assert calendar.add_days(date(2027, 7, 2), 3, "calendar", None) == date(2027, 7, 5)
    with pytest.raises(ValueError, match="no holiday calendar"):
        calendar.observed_holidays(2027, "uk")  # type: ignore[arg-type]
