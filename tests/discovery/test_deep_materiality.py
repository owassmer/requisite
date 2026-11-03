"""The materiality nodes' intervals (requisite/library/nodes/materiality.py): for every value a band allows and every day
the change can start, a month's cash delta lies within the month's interval, and the interval's ends are attained at
the corners the module names, jointly across months, so its sums are exact bounds (addendum A04 §4.1: intervals, never a
midpoint)."""

from datetime import UTC, date, datetime, timedelta
from fractions import Fraction as F

from hypothesis import given, settings
from hypothesis import strategies as st

from requisite import core
from requisite.library.nodes import materiality as m

HORIZON = core.TimeWindow(start=datetime(2026, 11, 1, tzinfo=UTC), end=datetime(2027, 11, 1, tzinfo=UTC),
                          start_closed=True, end_closed=False, tz="America/Los_Angeles")  # fmt: skip
AS_OF = datetime(2026, 11, 1, 12, tzinfo=UTC)

bands = st.tuples(st.integers(0, 60), st.integers(0, 60)).map(lambda t: (F(min(t), 100), F(max(t), 100)))
lags = st.tuples(st.integers(0, 14), st.integers(0, 14)).map(lambda t: (min(t), max(t)))
windows = st.tuples(st.integers(0, 400), st.integers(0, 400)).map(lambda t: (min(t), max(t)))


def window(first: int, last: int) -> core.TimeWindow:
    start = datetime(2026, 11, 2, tzinfo=UTC) + timedelta(days=first)
    return core.TimeWindow(start=start, end=datetime(2026, 11, 2, tzinfo=UTC) + timedelta(days=last),
                           start_closed=True, end_closed=True)  # fmt: skip


def impact(band, lag, win, terms, line, amount, direction="increase"):
    days = m.horizon_days(HORIZON)
    amounts = tuple((mo, amount) for mo, _ in m.monthly_amounts(1, 1, "month", [1] * 12, days))
    effect = m.effect([{"direction": direction}], [{"magnitude": [str(band[0]), str(band[1])]}],
                      [{"conditional": False}])  # fmt: skip
    start = m.cash_start(AS_OF, win, [{"lag_months": [str(lag[0]), str(lag[1])]}], terms, HORIZON)
    return m.cash_impact(amounts, days, line, effect, start)


@settings(max_examples=60, deadline=None)
@given(band=bands, lag=lags, win=windows, terms=st.integers(0, 90), amount=st.integers(1, 10**6),
       line=st.sampled_from(["cost", "revenue"]), direction=st.sampled_from(["increase", "decrease"]),
       pick=st.tuples(st.fractions(0, 1), st.fractions(0, 1), st.fractions(0, 1)))  # fmt: skip
def test_every_point_the_bands_allow_lies_within_the_interval_and_the_ends_are_attained(
    band, lag, win, terms, amount, line, direction, pick
):
    coarse = impact(band, lag, window(*win), terms, line, amount, direction)
    # A point inside the band, a whole number of months inside the lag band, and a day inside the window.
    mag = band[0] + (band[1] - band[0]) * pick[0]
    months = lag[0] + int((lag[1] - lag[0]) * pick[1])
    day = win[0] + int((win[1] - win[0]) * pick[2])
    point = impact((mag, mag), (months, months), window(day, day), terms, line, amount, direction)
    for (month, lo, hi), (month2, x, y) in zip(coarse, point, strict=True):
        assert month == month2 and x == y and lo <= x <= hi
    # The ends: every month's lower end at one corner, every upper end at another, so the sums are attained.
    ends = [impact((b, b), (lg, lg), window(w, w), terms, line, amount, direction)
            for b in band for lg in lag for w in win]  # fmt: skip
    for i, (_, lo, hi) in enumerate(coarse):
        assert lo == min(e[i][1] for e in ends) and hi == max(e[i][2] for e in ends)
    totals = [sum(x for _, x, _ in e) for e in ends]
    assert sum(lo for _, lo, _ in coarse) == min(totals) and sum(hi for _, _, hi in coarse) == max(totals)


def test_an_open_band_leaves_the_loss_unbounded_and_a_decrease_stops_at_the_whole_amount():
    days = m.horizon_days(HORIZON)
    # Days are the horizon's own zone's: 00:00 UTC on November 1 is 17:00 on October 31 in Los Angeles, a day the
    # horizon holds; an open end at a local midnight excludes that day.
    assert days == (date(2026, 10, 31), date(2027, 10, 31))
    utc = HORIZON.model_copy(update={"tz": "UTC"})
    assert m.horizon_days(utc) == (date(2026, 11, 1), date(2027, 10, 31))
    up = m.effect([{"direction": "increase"}], [{"magnitude": ["0.40", None]}], [{"conditional": False}])
    down = m.effect([{"direction": "decrease"}], [{"magnitude": ["0.40", None]}], [{"conditional": True}])
    assert up == (F(2, 5), None) and down == (F(-1), F(0))
    loss = m.horizon_loss(m.cash_impact(((date(2027, 1, 1), F(100)),), days, "cost", up, (date(2027, 1, 1), None)))
    assert loss == (F(0), None)  # the start may fall after the horizon, and the rise has no upper end
    assert m.expected_impact(F(1, 2), loss) == (F(0), None) and m.tail_impact(F(1, 2), F(9, 10), loss) == (F(0), None)


def test_the_tail_is_cvar_of_the_two_point_loss():
    # Probability 1/20 of a 100 loss at alpha 0.9: the worst tenth holds all of the loss's mass and half a tenth of
    # zeros, so CVaR is 50 (addendum A06 §6.3; oracle check 5).
    assert m.tail_impact(F(1, 20), F(9, 10), (F(100), F(100))) == (F(50), F(50))
    assert m.tail_impact(F(1, 20), None, (F(-5), F(100))) == (F(0), F(100))
