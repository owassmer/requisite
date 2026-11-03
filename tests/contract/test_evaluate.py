"""The single settlement function (requisite/contract/evaluate.py; spec I6, I8, §6.3; addendum A05): what it reads,
how its policies select values, its temporal operators and numbers in three-valued logic, the procedure, and the dated
transfers. Each expectation is worked by hand beside it."""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal as D
from fractions import Fraction as F
from zoneinfo import ZoneInfo

import pytest
from irkit import (
    EXPIRY,
    NY,
    T0,
    at,
    band,
    binary,
    contract,
    days,
    num,
    obs,
    observable,
    procedure,
    reading,
    timing,
)

from requisite import core as c
from requisite.contract.evaluate import IRError, evaluate
from requisite.core.kleene import Kleene

W = T0 + timedelta(days=5)  # the instant most single-value tests read: 2027-03-06 00:00 UTC
HOT = c.Cmp(id="hot", observable="x", op=">", value=num(10))
TRUE, FALSE, UNKNOWN = Kleene.TRUE, Kleene.FALSE, Kleene.UNKNOWN


def truth(ir, observations, interpretations=None, when=EXPIRY):
    return evaluate(ir, observations, interpretations or {}, at=when).trigger_truth


def window(start, end, *, tz="UTC", start_closed=True, end_closed=False):
    return c.TimeWindow(start=start, end=end, start_closed=start_closed, end_closed=end_closed, tz=tz)


def per_unit(quantity, unit="1.00", **kw):
    fields = {"kind": "per_unit", "unit_value": D(unit), "quantity": quantity, "currency": "USD",
              "rounding_quantum": D("0.01"), "rounding_mode": "half_even", "payment": "at_settlement"}  # fmt: skip
    return c.Payout(**{**fields, **kw})


# ---------------------------------------------------------------- what the evaluator reads


def test_observed_and_synthetic_observations_are_never_mixed():
    ir = contract(at("x", ">", 10, W))
    real, synthetic = obs(11, W), obs(11, W, synthetic=True)
    assert truth(ir, [real]) is TRUE and truth(ir, [synthetic]) is TRUE
    with pytest.raises(TypeError, match="never mixed"):
        evaluate(ir, [real, synthetic], {}, at=EXPIRY)


def test_inputs_it_cannot_read_one_way_are_refused_not_guessed():
    ir = contract(c.And(id="a", args=[at("x", ">", 10, W), c.Interpretive(id="q", question_template="t", bindings={})]))
    with pytest.raises(ValueError, match="not stated at it"):  # x is published to the cent; 10.005 is not a print
        evaluate(ir, [obs("10.005", W)], {"q": TRUE}, at=EXPIRY)
    with pytest.raises(ValueError, match="does not declare"):
        evaluate(ir, [obs(11, W, oid="y")], {"q": TRUE}, at=EXPIRY)
    with pytest.raises(ValueError, match="no interpretation given"):  # an unknown is given as UNKNOWN, never omitted
        evaluate(ir, [obs(11, W)], {}, at=EXPIRY)
    with pytest.raises(TypeError, match="Kleene truth"):
        evaluate(ir, [obs(11, W)], {"q": "affirmative"}, at=EXPIRY)
    with pytest.raises(IRError, match="no instant"):  # a comparison outside a temporal operator reads no instant
        evaluate(contract(HOT), [], {}, at=EXPIRY)


def test_an_observable_whose_sources_the_procedure_never_consults_is_refused():
    obs_x = observable(sources=("agency", "blog"))
    ir = contract(at("x", ">", 10, W), observables=[obs_x, observable("y", sources=("blog",))],
                  resolution=procedure(precedence=["agency"]))  # fmt: skip
    assert truth(ir, [obs(11, W, src="blog")]) is UNKNOWN  # x's blog value is never consulted; agency has published none
    reads_y = contract(at("y", ">", 10, W), observables=[obs_x, observable("y", sources=("blog",))],
                       resolution=procedure(precedence=["agency"]))  # fmt: skip
    with pytest.raises(IRError, match="consults none of its sources"):
        evaluate(reads_y, [], {}, at=EXPIRY)


# ---------------------------------------------------------------- revision and conflict policies (A05 §5.4)


def test_each_revision_policy_selects_its_vintage():
    first, revised = obs(11, W, W, final=False), obs(9, W, W + timedelta(days=3), final=False)
    late_final = obs(12, W, W + timedelta(days=20), final=True)
    after_expiry = obs(8, W, EXPIRY + timedelta(hours=1))

    def with_policy(policy, **kw):
        return contract(at("x", ">", 10, W), observables=[observable(policy=policy, **kw)],
                        timing=timing(observation_cutoff=EXPIRY + timedelta(days=2)))  # fmt: skip

    # first_print: the first published value, whatever came after.
    assert truth(with_policy("first_print"), [first, revised, late_final], when=EXPIRY + timedelta(days=3)) is TRUE
    # as_of_expiry: the latest published by expiration; a revision after it is not the expiry vintage.
    assert truth(with_policy("as_of_expiry"), [first, revised], when=EXPIRY + timedelta(days=3)) is FALSE
    assert truth(with_policy("as_of_expiry"), [first, revised, late_final], when=EXPIRY + timedelta(days=3)) is TRUE
    assert truth(with_policy("as_of_expiry"), [first, revised, after_expiry], when=EXPIRY + timedelta(days=3)) is FALSE
    # final with a deadline and no fallback: only a final value published by the deadline settles.
    deadline = with_policy("final", final_deadline=W + timedelta(days=30))
    assert truth(deadline, [first, revised], when=W + timedelta(days=10)) is UNKNOWN  # not final yet
    assert truth(deadline, [first, revised, late_final], when=EXPIRY) is TRUE
    ev = evaluate(deadline, [first, revised], {}, at=EXPIRY + timedelta(days=3))
    assert ev.trigger_truth is UNKNOWN and any("no final value" in u for u in ev.unresolved)
    # final with a fallback: the fallback vintage settles once the determination passes with no final value.
    fallback = with_policy("final", final_fallback="as_of_expiry")
    assert truth(fallback, [first, revised], when=EXPIRY) is UNKNOWN  # values may still be final by the cutoff
    assert truth(fallback, [first, revised], when=EXPIRY + timedelta(days=3)) is FALSE  # as of expiry: 9
    ev = evaluate(fallback, [first, revised], {}, at=EXPIRY + timedelta(days=3))
    assert c.Ref(id="k/observables/x/final_fallback/as_of_expiry", version=1) in ev.policies


def test_precedence_settles_on_the_highest_source_that_has_published():
    two = observable(sources=("a", "b"))
    ir = contract(at("x", ">", 10, W), observables=[two], resolution=procedure(precedence=["a", "b"]))
    assert truth(ir, [obs(9, W, src="b")]) is FALSE  # a has published nothing by the cutoff: b settles
    assert truth(ir, [obs(9, W, src="b"), obs(11, W, src="a")]) is TRUE  # a has: a settles
    # Before the cutoff, b's value is provisional: a may still publish and settle instead.
    later = contract(at("x", ">", 10, W), observables=[two], resolution=procedure(precedence=["a", "b"]),
                     timing=timing(observation_cutoff=EXPIRY + timedelta(days=3)))  # fmt: skip
    b_only = evaluate(later, [obs(9, W, src="b")], {}, at=EXPIRY + timedelta(days=1))
    assert b_only.trigger_truth is UNKNOWN and b_only.procedure_status == "pending"
    assert any("a, ranked above it, may still publish" in u for u in b_only.unresolved)
    a_late = [obs(9, W, src="b"), obs(11, W, EXPIRY + timedelta(days=2), src="a")]
    assert evaluate(later, a_late, {}, at=EXPIRY + timedelta(days=3)).procedure_status == "resolved_yes"
    assert evaluate(later, [obs(9, W, src="b")], {}, at=EXPIRY + timedelta(days=3)).procedure_status == "resolved_no"


def test_sources_that_must_agree_are_unknown_until_both_publish_and_disputed_when_they_disagree():
    two = observable(sources=("a", "b"))
    agree = c.ConflictPolicy(procedure="require_agreement", sources=["a", "b"])
    ir = contract(at("x", ">", 10, W), observables=[two], resolution=procedure(precedence=["a", "b"], conflict_policy=agree))
    assert truth(ir, [obs(11, W, src="a"), obs(11, W, src="b")]) is TRUE
    one = evaluate(ir, [obs(11, W, src="a")], {}, at=EXPIRY)
    assert one.trigger_truth is UNKNOWN and one.procedure_status == "resolved_no"  # the unknown rule applies
    split = evaluate(ir, [obs(11, W, src="a"), obs(9, W, src="b")], {}, at=EXPIRY)
    assert split.trigger_truth is UNKNOWN and split.procedure_status == "disputed"  # a distinct state, not NO
    assert any("disagree: a gives 11, b gives 9" in u for u in split.unresolved)
    assert c.Ref(id="k/resolution/conflict_policy/require_agreement", version=1) in split.policies
    # Before the determination a dispute is still pending; a dispute that cannot change the outcome is no dispute.
    assert evaluate(ir, [obs(11, W, src="a"), obs(9, W, src="b")], {}, at=W).procedure_status == "pending"
    either = c.Or(id="o", args=[at("x", ">", 10, W), at("x", ">", 0, W + timedelta(days=1), nid="later")])
    settled = contract(either, observables=[two], resolution=procedure(precedence=["a", "b"], conflict_policy=agree))
    later = [obs(1, W + timedelta(days=1), src=s) for s in ("a", "b")]
    assert evaluate(settled, [obs(11, W, src="a"), obs(9, W, src="b"), *later], {}, at=EXPIRY).procedure_status == "resolved_yes"


def test_a_value_not_published_by_the_instant_it_is_needed_is_unknown():
    ir = contract(at("x", ">", 10, W))
    late = obs(11, W, W + timedelta(days=2))
    assert truth(ir, [late], when=W + timedelta(days=1)) is UNKNOWN
    assert truth(ir, [late], when=W + timedelta(days=2)) is TRUE
    after_cutoff = obs(11, W, EXPIRY + timedelta(seconds=1))  # the contract considers nothing published after its cutoff
    assert truth(ir, [after_cutoff], when=EXPIRY + timedelta(days=9)) is UNKNOWN


# ---------------------------------------------------------------- temporal operators in three-valued logic


def test_by_counts_observations_within_its_bounds_with_their_stated_inclusion():
    d = T0 + timedelta(days=10)
    closed = c.By(id="by", expr=HOT, since=T0 + timedelta(days=2), since_closed=True, deadline=d, closed=True)
    open_ = closed.model_copy(update={"since_closed": False, "closed": False})
    for by, at_deadline, at_since in ((closed, TRUE, TRUE), (open_, FALSE, FALSE)):
        ir = contract(by)
        assert truth(ir, [obs(9, T0 + timedelta(days=3)), obs(11, d)]) is at_deadline
        assert truth(ir, [obs(11, T0 + timedelta(days=2)), obs(9, T0 + timedelta(days=5))]) is at_since
        outside = [obs(11, T0 + timedelta(days=1)), obs(11, d + timedelta(seconds=1))]
        assert truth(ir, [*outside, obs(9, T0 + timedelta(days=5))]) is FALSE  # the true values are outside its bounds
        assert truth(ir, outside) is UNKNOWN  # and with none inside, nothing establishes it either way
    # Before the observation cutoff, a value for an earlier instant may still be published: By is true or unknown.
    ir = contract(closed)
    assert truth(ir, [obs(9, T0 + timedelta(days=3))], when=d + timedelta(days=5)) is UNKNOWN
    assert truth(ir, [obs(11, T0 + timedelta(days=3))], when=d + timedelta(days=5)) is TRUE
    # Once closed, a By over no observation at all is unknown: nothing established it (I6).
    assert truth(ir, []) is UNKNOWN


def test_throughout_fails_on_one_breaking_value_and_holds_only_once_the_instants_are_closed():
    th = c.Throughout(id="th", expr=HOT, window=window(T0, T0 + timedelta(days=5)))
    ir = contract(th)
    five = [obs(11, d) for d in days(5)]
    assert truth(ir, five) is TRUE
    assert truth(ir, [*five, obs(9, T0 + timedelta(days=5))]) is TRUE  # the window's end is open
    assert truth(ir, [*five[:4], obs(9, T0 + timedelta(days=4))]) is FALSE
    assert truth(ir, five, when=T0 + timedelta(days=6)) is UNKNOWN  # more may be published until the cutoff
    assert truth(ir, [*five[:4], obs(9, T0 + timedelta(days=4))], when=T0 + timedelta(days=6)) is FALSE


@pytest.mark.covers("A.T17")
def test_unknown_observations_in_a_count_give_an_interval_never_false_or_zero():
    # Five days, the oracle's pattern (check 12): true, false, none published, a value not final, true. So k = 2 true days
    # and u = 2 unknown ones: the count is [2, 4] (A05 §5.3), never 2.
    final_x = observable(policy="final", final_deadline=EXPIRY)
    five = window(T0, T0 + timedelta(days=5))
    count = c.CountTrue(id="count", expr=HOT, days=five)
    d = days(5)
    records = [obs(11, d[0]), obs(9, d[1]), obs(11, d[3], final=False), obs(11, d[4])]

    def evaluated(op, k, payout=None):
        trigger = c.NumCmp(id="n", expr=count, op=op, value=D(k))
        return evaluate(contract(trigger, observables=[final_x], payout=payout), records, {}, at=EXPIRY)

    assert evaluated(">=", 1).trigger_truth is TRUE  # every value in [2, 4] is at least 1
    assert evaluated(">=", 3).trigger_truth is UNKNOWN  # 2 is not, 4 is
    assert evaluated(">=", 5).trigger_truth is FALSE
    assert evaluated("==", 0).trigger_truth is FALSE  # unknown days are not zero days
    assert evaluated("==", 2).trigger_truth is UNKNOWN  # nor false ones
    # As a payout quantity it stays an interval at settlement: the amount is not a guess, and no transfer is made.
    ev = evaluated(">=", 1, payout=per_unit(count, unit="10.00"))
    assert ev.procedure_status == "resolved_yes" and ev.quantity == (D(2), D(4)) and ev.transfers == ()
    assert any("between 20.00 and 40.00 USD per unit" in u and "2027-03-03" in u for u in ev.unresolved)
    # With every day unknown the count is [0, 5], not 0.
    nothing = evaluate(contract(c.NumCmp(id="n", expr=count, op=">=", value=D(1)), observables=[final_x],
                                payout=per_unit(count)), [], {}, at=EXPIRY)  # fmt: skip
    assert nothing.quantity == (D(0), D(5)) and nothing.trigger_truth is UNKNOWN


def test_a_count_is_over_the_calendar_days_of_its_zone_and_open_until_the_cutoff():
    # Midnight UTC is 19:00 the day before in New York: the window [Mar 1 00:00Z, Mar 6 00:00Z) holds the New York days
    # Mar 1 to Mar 4 whole (Mar 5 ends after it), and a value at Mar 2 00:00Z falls on New York's March 1.
    count = c.CountTrue(id="count", expr=HOT, days=window(T0, T0 + timedelta(days=5), tz=NY))
    ir = contract(c.NumCmp(id="n", expr=count, op=">=", value=D(0)), payout=per_unit(count))
    d = days(6)
    ev = evaluate(ir, [obs(11, x) for x in d], {}, at=EXPIRY)
    assert ev.quantity == (D(4), D(4))  # d[0] is February 28 in New York, d[5] March 5: outside
    ev = evaluate(ir, [obs(11, d[1]), obs(9, d[2]), obs(11, d[3])], {}, at=EXPIRY)
    assert ev.quantity == (D(2), D(3))  # Mar 1 true, Mar 2 false, Mar 3 true, Mar 4 unknown
    # Before the cutoff a day not yet shown true may still be: Mar 2's false value does not close it.
    assert evaluate(ir, [obs(11, d[1]), obs(9, d[2]), obs(11, d[3])], {}, at=T0 + timedelta(days=9)).quantity == (D(2), D(4))


def test_a_duration_is_measured_in_its_unit_across_a_daylight_saving_change():
    # New York moves its clocks forward on Sunday, March 14, 2027: that calendar day has 23 hours.
    ny = ZoneInfo(NY)
    start, end = datetime(2027, 3, 13, tzinfo=ny), datetime(2027, 3, 16, tzinfo=ny)
    on = datetime(2027, 3, 14, tzinfo=ny)  # the condition holds from the start of March 14 to the window's end
    records = [obs(9, start), obs(11, on)]

    def measured(unit):
        q = c.DurationIn(id="dur", expr=HOT, window=window(start, end, tz=NY), unit=unit)
        return evaluate(contract(c.NumCmp(id="n", expr=q, op=">=", value=D(0)), payout=per_unit(q)), records, {},
                        at=EXPIRY).quantity  # fmt: skip

    assert measured("hour") == (D(47), D(47))  # 23 + 24 hours
    assert measured("calendar_day") == (D(2), D(2))  # March 14 and March 15, one unit each
    elapsed = measured("elapsed_day")  # 47/24 of a 24-hour day, which has no finite decimal: rounded outward
    assert F(elapsed[0]) < F(47, 24) < F(elapsed[1]) and elapsed[1] - elapsed[0] < D("1e-30")
    # A stretch whose value is unknown is in the upper bound only; before the cutoff any stretch may still change.
    final_x = observable(policy="final", final_deadline=EXPIRY)
    q = c.DurationIn(id="dur", expr=HOT, window=window(start, end, tz=NY), unit="hour")
    ir = contract(c.NumCmp(id="n", expr=q, op=">=", value=D(0)), observables=[final_x], payout=per_unit(q))
    assert evaluate(ir, [obs(9, start), obs(11, on, final=False)], {}, at=EXPIRY).quantity == (D(0), D(47))
    assert evaluate(ir, [obs(9, start), obs(11, on)], {}, at=on + timedelta(days=5)).quantity == (D(0), D(71))


# ---------------------------------------------------------------- numbers: readings, published values, nodes


def test_a_comparison_over_an_interval_is_true_or_false_only_where_the_whole_interval_decides_it():
    r = reading()  # bands [0, 10), [10, 20), [20, 30]

    def judged(op, v, given):
        return truth(contract(c.NumCmp(id="n", expr=r, op=op, value=D(v)), observables=[observable()]), [], {"r": given})

    assert judged("<", 10, band(0, 10)) is TRUE  # every value in [0, 10) is below 10
    assert judged("<=", 10, band(10, 20)) is UNKNOWN  # 10 itself is, 15 is not
    assert judged(">=", 10, band(10, 20)) is TRUE
    assert judged(">", 30, band(20, 30)) is FALSE  # the last band is closed: 30 is in it, and is not above 30
    assert judged(">=", 30, band(20, 30)) is UNKNOWN
    assert judged("<", 20, band(0, 20)) is TRUE  # a run of bands
    assert judged("<", 20, band(0, 30)) is UNKNOWN  # the whole range: the question is not established
    with pytest.raises(ValueError, match="not a band"):
        judged("<", 20, band(5, 15))


def test_a_computed_number_is_its_node_at_the_corners_where_declared_monotone_and_unknown_otherwise():
    a, b = reading("a"), reading("b")
    value_at = c.ValueAt(id="v", observable="x", at=W)

    def excess(node, inputs, op=">=", v=0):
        return c.NumCmp(id="n", expr=c.Computed(id="f", node=c.Ref(id=node, version=1), inputs=inputs), op=op, value=D(v))

    diff = contract(excess("tests.contract.difference", {"a": a, "b": value_at}))
    assert truth(diff, [obs(5, W)], {"a": band(10, 20)}) is TRUE  # a - x over [10, 20] - 5 = [5, 15]
    assert truth(diff, [obs(15, W)], {"a": band(10, 20)}) is UNKNOWN  # [-5, 5]
    assert truth(diff, [obs(25, W)], {"a": band(10, 20)}) is FALSE  # [-15, -5)... at most -5
    assert truth(diff, [], {"a": band(10, 20)}) is UNKNOWN  # x not published: unbounded, never a default
    both = contract(excess("tests.contract.difference", {"a": a, "b": b}, ">", 0))
    assert truth(both, [], {"a": band(20, 30), "b": band(0, 10)}) is TRUE  # [20, 30] - [0, 10) is above 10
    # A node not declared monotone in an open input is unknown over it, though every corner would agree.
    square = contract(excess("tests.contract.square", {"a": a}, ">=", 0))
    assert truth(square, [], {"a": band(10, 20)}) is UNKNOWN
    assert truth(square, [], {"a": band(20, 30)}) is UNKNOWN
    # A node that declares a direction it does not have is found out at its corners.
    lies = contract(excess("tests.contract.misdeclared", {"a": a}))
    with pytest.raises(ValueError, match="not monotone as it declares"):
        evaluate(lies, [], {"a": band(10, 20)}, at=EXPIRY)
    with pytest.raises(IRError, match="not registered"):
        evaluate(contract(excess("tests.contract.nowhere", {"a": a})), [], {"a": band(10, 20)}, at=EXPIRY)


# ---------------------------------------------------------------- the procedure (A05 §5.1-5.2)


@pytest.mark.covers("A.T16")
def test_an_unknown_trigger_that_defaults_to_no_keeps_its_truth_unknown():
    ir = contract(c.And(id="both", args=[at("x", ">", 10, W), c.Interpretive(id="q", question_template="t", bindings={})]))
    ev = evaluate(ir, [obs(11, W)], {"q": UNKNOWN}, at=EXPIRY)
    assert ev.trigger_truth is UNKNOWN  # never rewritten to false (A05 §5.2)
    assert ev.procedure_status == "resolved_no"
    assert c.Ref(id="k/resolution/on_unknown_at_expiry/no", version=1) in ev.policies  # the policy that decided it
    assert ev.transfers == () and any(u.startswith("q (t)") for u in ev.unresolved)  # and what stayed unknown
    # A different condition that is definitively false still gives a definitive NO, by the trigger, not the rule.
    no = evaluate(ir, [obs(9, W)], {"q": UNKNOWN}, at=EXPIRY)
    assert no.trigger_truth is FALSE and no.procedure_status == "resolved_no"
    assert c.Ref(id="k/resolution/on_unknown_at_expiry/no", version=1) not in no.policies


def test_before_expiration_an_unknown_or_known_trigger_is_pending():
    ir = contract(at("x", ">", 10, W))
    for records in ([], [obs(11, W)], [obs(9, W)]):
        ev = evaluate(ir, records, {}, at=W + timedelta(days=1))
        assert ev.procedure_status == "pending" and ev.transfers == ()
    # At expiration an unknown trigger waits for the observation cutoff, when the contract considers values until then.
    later = contract(at("x", ">", 10, W), timing=timing(observation_cutoff=EXPIRY + timedelta(days=3)))
    assert evaluate(later, [], {}, at=EXPIRY + timedelta(days=1)).procedure_status == "pending"
    assert evaluate(later, [], {}, at=EXPIRY + timedelta(days=3)).procedure_status == "resolved_no"


def test_the_unknown_rule_yes_void_and_extend_each_settle_as_the_contract_states():
    trigger = at("x", ">", 10, W)
    yes = evaluate(contract(trigger, resolution=procedure(on_unknown_at_expiry="yes")), [], {}, at=EXPIRY)
    assert yes.trigger_truth is UNKNOWN and yes.procedure_status == "resolved_yes" and yes.transfers[0].amount == D("1.00")
    # A void returns what its transfer policy states, and nothing it does not (A05 §5.2).
    fixed = procedure(on_unknown_at_expiry="void", void_transfer=c.VoidTransfer(returns="fixed_amount", amount=D("0.40")))
    void = evaluate(contract(trigger, resolution=fixed), [], {}, at=EXPIRY)
    assert void.procedure_status == "void" and [(t.amount, t.basis) for t in void.transfers] == [(D("0.40"), "void refund")]
    premium = procedure(on_unknown_at_expiry="void", void_transfer=c.VoidTransfer(returns="premium"))
    refund = evaluate(contract(trigger, resolution=premium), [], {}, at=EXPIRY)
    assert refund.transfers == () and any("premium paid per unit is returned" in u for u in refund.unresolved)
    nothing = evaluate(contract(trigger, resolution=procedure(on_unknown_at_expiry="void")), [], {}, at=EXPIRY)
    assert nothing.procedure_status == "void" and nothing.transfers == ()
    # An extension: extended while it runs (its cutoff moves with it), then settled on what it found, or by the rule
    # for after it. Five business days from Tuesday June 1, 2027 is Tuesday June 8; payment one business day each later.
    ext = contract(trigger, timing=timing(lag_days="business", holiday_calendar="us_federal", settlement_lag_days=1,
                                          payment_lag_days=1),
                   resolution=procedure(on_unknown_at_expiry="extend", extension_days=5, after_extension="void",
                                        void_transfer=c.VoidTransfer(returns="fixed_amount", amount=D("0.40"))))  # fmt: skip
    found = [obs(11, W, EXPIRY + timedelta(days=2))]  # published during the extension
    assert evaluate(ext, found, {}, at=EXPIRY + timedelta(days=3)).procedure_status == "extended"
    settled = evaluate(ext, found, {}, at=EXPIRY + timedelta(days=30))
    assert settled.procedure_status == "resolved_yes"
    assert settled.transfers[0].at == datetime(2027, 6, 10, 10, tzinfo=ZoneInfo(NY))
    voided = evaluate(ext, [], {}, at=EXPIRY + timedelta(days=30))
    assert voided.procedure_status == "void" and voided.trigger_truth is UNKNOWN
    assert {r.id for r in voided.policies} >= {"k/resolution/on_unknown_at_expiry/extend", "k/resolution/after_extension/void"}


def test_early_expiration_at_first_true_resolves_at_that_instant():
    by = c.By(id="by", expr=HOT, since=T0, since_closed=True, deadline=EXPIRY, closed=True)
    early = procedure(early_expiration="when_true", early_expiration_at="first_true")
    ir = contract(by, resolution=early, timing=timing(settlement_lag_days=1, payment_lag_days=2))
    shown = W + timedelta(days=1, hours=3)  # the value for March 7 00:00Z is published at 03:00Z: 22:00 March 6 in NY
    records = [obs(9, W), obs(11, W + timedelta(days=1), shown)]
    assert evaluate(ir, records, {}, at=shown - timedelta(seconds=1)).procedure_status == "pending"
    for when in (shown, EXPIRY + timedelta(days=9)):  # the same resolution, whenever it is evaluated
        ev = evaluate(ir, records, {}, at=when)
        assert ev.procedure_status == "resolved_yes" and ev.unresolved == ()
        assert [t.at for t in ev.transfers] == [datetime(2027, 3, 9, 22, tzinfo=ZoneInfo(NY))]  # + 1 + 2 calendar days
    # A final policy's deadline changes what is known with no publication: the trigger is first true when it passes.
    fallback = observable(policy="final", final_deadline=W + timedelta(days=4), final_fallback="first_print")
    ir = contract(by, observables=[fallback], resolution=early)
    ev = evaluate(ir, [obs(11, W, final=False)], {}, at=W + timedelta(days=9))
    assert ev.transfers[0].at == (W + timedelta(days=4)).astimezone(ZoneInfo(NY))


def test_early_expiration_by_an_exchange_rule_leaves_its_date_and_its_transfers_unresolved():
    by = c.By(id="by", expr=HOT, since=None, since_closed=None, deadline=EXPIRY, closed=True)
    rule = procedure(early_expiration="when_true", early_expiration_at="per_exchange_rule", early_expiration_rule="Rule 7.2")
    ev = evaluate(contract(by, resolution=rule), [obs(11, W)], {}, at=W + timedelta(days=2))
    assert ev.trigger_truth is TRUE and ev.procedure_status == "resolved_yes"
    assert ev.transfers == ()  # no date is taken from the scheduled expiration in its place
    assert [u.split(":")[0] for u in ev.unresolved] == ["expiration date", "payout"]
    assert all("Rule 7.2" in u for u in ev.unresolved) and "1.00 USD per unit" in ev.unresolved[1]
    # With interpretations given only as of the evaluation, even first_true cannot say when the trigger became true.
    q = c.Interpretive(id="q", question_template="t", bindings={})
    first = procedure(early_expiration="when_true", early_expiration_at="first_true")
    ev = evaluate(contract(c.And(id="a", args=[by, q]), resolution=first), [obs(11, W)], {"q": TRUE}, at=W + timedelta(days=2))
    assert ev.procedure_status == "resolved_yes" and ev.transfers == ()
    assert "between 2027-03-06T00:00:00+00:00 and 2027-03-08T00:00:00+00:00" in ev.unresolved[0]


# ---------------------------------------------------------------- transfers: amounts and dates (A05 §5.1, §5.3)


def test_payout_amounts_by_kind_rounded_by_the_contracts_quantum_and_mode():
    count = c.CountTrue(id="count", expr=HOT, days=window(T0, T0 + timedelta(days=6)))
    trigger = c.NumCmp(id="any", expr=count, op=">=", value=D(1))
    d = days(6)

    def paid(payout, n_true):
        records = [obs(11 if i < n_true else 9, x) for i, x in enumerate(d)]
        return [t.amount for t in evaluate(contract(trigger, payout=payout), records, {}, at=EXPIRY).transfers]

    assert paid(binary("2.50"), 3) == [D("2.50")]
    assert paid(per_unit(count, "1.005", rounding_mode="half_up"), 3) == [D("3.02")]  # 3.015 rounds away from zero
    assert paid(per_unit(count, "1.005", rounding_mode="half_even"), 3) == [D("3.02")]  # 3.015: 2 is even
    assert paid(per_unit(count, "1.005", rounding_mode="half_even"), 1) == [D("1.00")]  # 1.005: 0 is even
    assert paid(per_unit(count, "1.005", rounding_mode="down"), 3) == [D("3.01")]
    assert paid(per_unit(count, "1.001", rounding_mode="up"), 1) == [D("1.01")]
    capped = per_unit(count, "3", kind="capped_linear", cap=D("10"), rounding_quantum=D("1"))
    assert paid(capped, 2) == [D("6")] and paid(capped, 5) == [D("10")]
    # A capped amount is resolved when every quantity the interval allows reaches the cap.
    partial = [obs(11, x) for x in d[:4]]
    ev = evaluate(contract(trigger, payout=capped), partial, {}, at=EXPIRY)
    assert ev.quantity == (D(4), D(6)) and [t.amount for t in ev.transfers] == [D("10")]


def test_business_day_lags_skip_weekends_and_federal_holidays_in_the_contracts_zone():
    # Expiration Friday July 2, 2027, 10:00 in New York. Independence Day, Sunday July 4, is observed on Monday July 5.
    expiry = datetime(2027, 7, 2, 10, tzinfo=ZoneInfo(NY))
    ir = contract(at("x", ">", 10, W), timing=timing(close=expiry, expiration=expiry, observation_cutoff=expiry,
                                                     settlement_lag_days=1, payment_lag_days=1, lag_days="business",
                                                     holiday_calendar="us_federal"))  # fmt: skip
    ev = evaluate(ir, [obs(11, W)], {}, at=expiry)
    assert [t.at for t in ev.transfers] == [datetime(2027, 7, 7, 10, tzinfo=ZoneInfo(NY))]  # Tuesday 6th, then 7th
    calendar_days = contract(at("x", ">", 10, W), timing=timing(close=expiry, expiration=expiry, observation_cutoff=expiry,
                                                                settlement_lag_days=1, payment_lag_days=1))  # fmt: skip
    assert evaluate(calendar_days, [obs(11, W)], {}, at=expiry).transfers[0].at.date() == date(2027, 7, 4)


def test_a_per_period_payment_pays_each_period_after_its_end():
    p1, p2 = window(T0, T0 + timedelta(days=3)), window(T0 + timedelta(days=3), T0 + timedelta(days=6))
    count = c.CountTrue(id="count", expr=HOT, days=window(T0, T0 + timedelta(days=6)))
    strip = per_unit(count, "2.00", payment="per_period", periods=[p1, p2])
    ir = contract(c.NumCmp(id="any", expr=count, op=">=", value=D(1)), payout=strip,
                  timing=timing(settlement_lag_days=2, lag_days="business", holiday_calendar="us_federal"))  # fmt: skip
    d = days(6)
    ev = evaluate(ir, [obs(11, x) for x in d[:5]] + [obs(9, d[5])], {}, at=EXPIRY)
    # The first period ends Thursday March 4, 00:00Z (Wednesday 19:00 in New York): paid two business days later. The
    # second ends Sunday March 7, 00:00Z (Saturday 19:00 in New York): Monday and Tuesday are its two business days.
    assert [(t.at, t.amount) for t in ev.transfers] == [
        (datetime(2027, 3, 5, 19, tzinfo=ZoneInfo(NY)), D("6.00")),
        (datetime(2027, 3, 9, 19, tzinfo=ZoneInfo(NY)), D("4.00")),
    ]
    # A period whose count is still open is unresolved, saying on what; the other is paid.
    ev = evaluate(ir, [obs(11, x) for x in d[:4]], {}, at=EXPIRY)
    assert [t.amount for t in ev.transfers] == [D("6.00")]
    assert len(ev.unresolved) == 1 and "between 2.00 and 6.00 USD per unit" in ev.unresolved[0]


def test_a_definitions_gap_is_refused_rather_than_guessed():
    count = c.CountTrue(id="count", expr=HOT, days=window(T0, T0 + timedelta(days=6)))
    p = [window(T0, T0 + timedelta(days=3)), window(T0 + timedelta(days=3), T0 + timedelta(days=6))]
    capped_strip = per_unit(count, "1", kind="capped_linear", cap=D(2), payment="per_period", periods=p)
    with pytest.raises(IRError, match="cap is per period or in total"):
        evaluate(contract(c.NumCmp(id="n", expr=count, op=">=", value=D(1)), payout=capped_strip), [], {}, at=EXPIRY)


def test_every_evaluation_records_its_inputs_and_the_rules_it_applied():
    ir = contract(at("x", ">", 10, W))
    a = evaluate(ir, [obs(11, W)], {}, at=EXPIRY)
    assert a == evaluate(ir, [obs(11, W)], {}, at=EXPIRY)  # deterministic
    assert a.evidence_state_hash != evaluate(ir, [obs(11, W)], {}, at=EXPIRY + timedelta(seconds=1)).evidence_state_hash
    assert a.evidence_state_hash != evaluate(ir, [obs(11, W, W + timedelta(hours=1))], {}, at=EXPIRY).evidence_state_hash
    assert a.contract == c.Ref(id="k", version=1)
    assert [r.id for r in a.policies] == [
        "requisite.contract.evaluate", "k/observables/x/revision_policy/first_print",
        "k/resolution/conflict_policy/precedence_first_available", "k/payout/rounding/half_even/0.01",
        "k/timing/lags/calendar/0+0/America/New_York",
    ]  # fmt: skip
    assert datetime.fromisoformat(a.transfers[0].at.isoformat()).astimezone(UTC) == EXPIRY
