"""The business's dated cash, day by day, is deep discovery's monthly model at a day's resolution
(requisite/library/nodes/cash.py): each end of an exposure's daily change sums, month by month, to deep.cash_impact's
end for that month, and a driver's daily flows to its monthly amount, exactly, over every combination of the model's
inputs drawn."""

from __future__ import annotations

from datetime import date, timedelta
from fractions import Fraction as F

from hypothesis import given, settings
from hypothesis import strategies as st

from requisite.library.nodes import cash, materiality

DAYS = (date(2026, 11, 1), date(2027, 10, 31))
FIRST, LAST = DAYS


def amounts(seed: int) -> tuple[tuple[date, F], ...]:
    months = materiality._months(*DAYS)  # noqa: SLF001
    return tuple((m, F(1000 + 37 * ((seed + i) % 11), 3)) for i, m in enumerate(months))


bounds = st.one_of(st.none(), st.fractions(min_value=-1, max_value=2, max_denominator=40))
days = st.integers(min_value=0, max_value=400)


@settings(max_examples=150, deadline=None)
@given(seed=st.integers(0, 50), a=bounds, b=bounds, early=days, late=st.one_of(st.none(), days),
       line=st.sampled_from(["cost", "revenue"]))  # fmt: skip
def test_each_end_of_the_daily_change_sums_to_deep_monthly_cash_impact(seed, a, b, early, late, line):
    lo, hi = (a, b) if a is None or b is None or a <= b else (b, a)
    earliest = FIRST + timedelta(days=early)
    latest = None if late is None else earliest + timedelta(days=late)
    amt = amounts(seed)
    daily = cash.daily_impact(amt, DAYS, line, (lo, hi), (earliest, latest))
    monthly = materiality.cash_impact(amt, DAYS, line, (lo, hi), (earliest, latest))
    assert [d for d, _, _ in daily] == cash.horizon_days(DAYS)
    for month, m_lo, m_hi in monthly:
        mine = [(x, y) for d, x, y in daily if (d.year, d.month) == (month.year, month.month)]
        low = None if any(x is None for x, _ in mine) else sum((x for x, _ in mine), F(0))
        high = None if any(y is None for _, y in mine) else sum((y for _, y in mine), F(0))
        assert (low, high) == (m_lo, m_hi), month


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 50), line=st.sampled_from(["cost", "revenue"]))
def test_a_drivers_daily_flows_sum_to_its_monthly_amounts(seed, line):
    amt = amounts(seed)
    flows = cash.daily_flows(amt, DAYS, line)
    sign = 1 if line == "revenue" else -1
    for month, amount in amt:
        assert sum((x for d, x in flows if (d.year, d.month) == (month.year, month.month)), F(0)) == sign * amount
