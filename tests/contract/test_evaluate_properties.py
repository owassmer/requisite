"""Properties of the settlement function at its limits (spec I6; addendum A05 §5.3-5.4, A09 §10.3), over random contracts:
the trigger's truth is monotone in information; a contract whose every input is known and admissible never yields an
unknown; inclusive and exclusive cutoffs differ exactly at the cutoff instant; a business-day lag never lands on a
weekend or holiday.

The random contracts read two observables on a daily grid of six instants: x under a "final" revision policy, whose
non-final values are unknown, and y under require_agreement of two sources, whose value is unknown while only one has
published. That is how a selected value is made unknown while its instant stays among the published ones: removing a
record altogether is adding or removing a raw publication, which can change what a policy selects and is outside the
monotonicity property (the task's statement of it)."""

from datetime import date, datetime, timedelta
from decimal import Decimal as D
from zoneinfo import ZoneInfo

from hypothesis import given, settings
from hypothesis import strategies as st
from irgen import CUTOFF, GRID, T0, WHEN, contracts, grid_contract, worlds
from irkit import obs

from requisite import core as c
from requisite.contract import calendar
from requisite.contract.evaluate import evaluate, lower
from requisite.core.kleene import Kleene


@settings(max_examples=2000, deadline=None)
@given(data=st.data())
def test_the_trigger_is_monotone_in_information(data):
    # A selected value, an interpretation or a reading going from unknown to known never turns a true or false trigger
    # into its opposite: whatever the partial world decides, the full world decides the same.
    ir = data.draw(contracts())
    (full, given_full), (partial, given_partial) = data.draw(worlds(ir))
    when = data.draw(st.sampled_from(WHEN))
    known = evaluate(ir, full, given_full, at=when)
    less = evaluate(ir, partial, given_partial, at=when)
    if less.trigger_truth is not Kleene.UNKNOWN:
        assert known.trigger_truth is less.trigger_truth
    if less.procedure_status in ("resolved_yes", "resolved_no") and less.trigger_truth is not Kleene.UNKNOWN:
        assert known.procedure_status == less.procedure_status


@settings(max_examples=1000, deadline=None)
@given(data=st.data())
def test_a_known_truth_is_never_reversed_as_publications_arrive(data):
    # One world, evaluated at two instants: what is true or false on the records published by the first stays so on
    # those published by the second (the evaluator's open instants before the cutoff, and its policies, see to it).
    ir = data.draw(contracts())
    (records, given), _ = data.draw(worlds(ir))
    first, second = sorted(data.draw(st.lists(st.sampled_from(WHEN), min_size=2, max_size=2)))
    earlier = evaluate(ir, records, given, at=first)
    if earlier.trigger_truth is not Kleene.UNKNOWN:
        assert evaluate(ir, records, given, at=second).trigger_truth is earlier.trigger_truth


@settings(max_examples=1000, deadline=None)
@given(data=st.data())
def test_a_contract_whose_every_input_is_known_never_yields_unknown(data):
    # Every value published and final, sources agreeing, every interpretation affirmative or negative and every reading
    # one band that decides the comparisons made on it, evaluated once the determination has passed.
    ir = data.draw(contracts(decidable=True))
    (full, given), _ = data.draw(worlds(ir, complete=True))
    ev = evaluate(ir, full, given, at=data.draw(st.sampled_from([CUTOFF, CUTOFF + timedelta(days=3)])))
    assert ev.trigger_truth is not Kleene.UNKNOWN, ev.unresolved
    assert ev.procedure_status in ("resolved_yes", "resolved_no")
    if ir.resolution.early_expiration == "none":
        assert ev.unresolved == ()
        assert len(ev.transfers) == (1 if ev.procedure_status == "resolved_yes" else 0)


OFFSETS = st.one_of(st.integers(-3, 3).map(lambda k: timedelta(microseconds=k)),
                    st.integers(-200_000, 200_000).map(lambda s: timedelta(seconds=s)))  # fmt: skip


@settings(max_examples=300, deadline=None)
@given(deadline=st.integers(1, 4), offset=OFFSETS, lower_bound=st.booleans())
def test_inclusive_and_exclusive_cutoffs_differ_exactly_at_the_cutoff_instant(deadline, offset, lower_bound):
    # A By whose condition is shown at one instant t near its bound (and fails at an instant well inside): the closed
    # and the open bound give the same truth unless t is the bound itself. So does a Throughout broken at t near an end.
    bound = GRID[deadline] + timedelta(hours=7)
    t = bound + offset
    inside = bound + (timedelta(days=1) if lower_bound else -timedelta(days=1))
    shown, fails = c.Cmp(id="hot", observable="y", op=">", value=c.Number(value=D(5))), 1

    def by(closed):
        if lower_bound:
            return c.By(id="by", expr=shown, since=bound, since_closed=closed, deadline=bound + timedelta(days=3), closed=True)
        return c.By(id="by", expr=shown, since=None, since_closed=None, deadline=bound, closed=closed)

    def throughout(closed):
        w = (c.TimeWindow(start=bound, end=bound + timedelta(days=3), start_closed=closed, end_closed=True) if lower_bound
             else c.TimeWindow(start=bound - timedelta(days=3), end=bound, start_closed=True, end_closed=closed))  # fmt: skip
        return c.Throughout(id="th", expr=shown, window=w)

    def judged(trigger, at_t, at_inside):
        records = [obs(v, when, T0, oid="y", src=s) for v, when in ((at_t, t), (at_inside, inside)) for s in ("a", "b")]
        return evaluate(grid_contract(trigger), records, {}, at=CUTOFF + timedelta(days=9)).trigger_truth

    differ = offset == timedelta(0)
    assert (judged(by(True), 11, fails) is judged(by(False), 11, fails)) is not differ
    assert (judged(throughout(True), fails, 11) is judged(throughout(False), fails, 11)) is not differ
    # The lowering's boundary points for the bound say the same as the evaluator at every point.
    for closed in (True, False):
        trigger = by(closed)
        for comparison in lower(grid_contract(trigger)).comparisons:
            for point in comparison.points:
                if comparison.kind == "cutoff" and point.threshold == bound:
                    t_point = point.value
                    records = [obs(11, t_point, T0, oid="y", src=s) for s in ("a", "b")]
                    records += [obs(fails, inside, T0, oid="y", src=s) for s in ("a", "b")]
                    got = evaluate(grid_contract(trigger), records, {}, at=CUTOFF + timedelta(days=9)).trigger_truth
                    assert got is (Kleene.TRUE if point.holds else Kleene.FALSE)


@settings(max_examples=500, deadline=None)
@given(day=st.dates(date(1971, 1, 1), date(2099, 12, 1)), n=st.integers(0, 40))
def test_a_business_day_lag_never_lands_on_a_weekend_or_holiday(day, n):
    landed = calendar.add_business_days(day, n, "us_federal")
    assert landed.weekday() < 5 and landed not in calendar.observed_holidays(landed.year)
    between = [day + timedelta(days=k) for k in range(1, (landed - day).days + 1)]
    counted = sum(calendar.is_business_day(d, "us_federal") for d in between)
    if n == 0:
        assert landed >= day and counted == (0 if landed == day else 1)  # the day itself, or the next business day
    else:
        assert counted == n  # exactly n business days after the day, the last of them the landing day


@settings(max_examples=200, deadline=None)
@given(day=st.dates(date(2026, 1, 1), date(2032, 12, 20)), hour=st.integers(0, 23), lags=st.tuples(st.integers(0, 6),
       st.integers(0, 6)), tz=st.sampled_from(["America/New_York", "America/Los_Angeles", "UTC", "Pacific/Auckland"]))  # fmt: skip
def test_a_settlement_paid_on_business_days_lands_on_one_in_the_contracts_zone(day, hour, lags, tz):
    expiry = datetime(day.year, day.month, day.day, hour, tzinfo=ZoneInfo(tz))
    seen = expiry - timedelta(days=1)
    trigger = c.At(id="at", expr=c.Cmp(id="hot", observable="y", op=">", value=c.Number(value=D(5))), at=seen)
    ir = grid_contract(trigger, open=seen, close=expiry, expiration=expiry, observation_cutoff=expiry,
                       settlement_lag_days=lags[0], payment_lag_days=lags[1], lag_days="business",
                       holiday_calendar="us_federal", tz=tz)  # fmt: skip
    records = [obs(11, seen, seen, oid="y", src=s) for s in ("a", "b")]
    (paid,) = evaluate(ir, records, {}, at=expiry).transfers
    local = paid.at.astimezone(ZoneInfo(tz)).date()
    assert calendar.is_business_day(local, "us_federal") and local >= expiry.date()
