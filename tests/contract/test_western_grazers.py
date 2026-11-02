"""The Western Grazers contract (spec §9.3's reference specimen), hand-expressed in the IR (tests/contract/
western_grazers.py), evaluated on the real structure: qualifying actions, effects, scope, timing, operative status and
the computed Net Monthly Cost test. The outside check is the real market, KXCAGOAT-26SEP30, which closed early and
settled YES on September 19, 2026, the day after California AB 187 became law."""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal as D
from fractions import Fraction as F
from pathlib import Path

import western_grazers as wg

from requisite import core as c
from requisite.compute import registry
from requisite.contract.evaluate import evaluate
from requisite.core.kleene import Kleene

ROOT = Path(__file__).resolve().parents[2]
MARKET = json.loads((ROOT / "fixtures/kalshi/KXCAGOAT/markets.json").read_text())["body"]["markets"][0]
NODE = registry.get(c.Ref(id="cagoatwage.net_monthly_cost_excess", version=1))
ZERO = dict.fromkeys(NODE.inputs, 0)


def excess(**terms):
    return NODE.fn(**{**ZERO, "minimum_wage": D("16.90"), "sheepherder_monthly_wage": D("4938.21"), **terms})


def test_the_net_monthly_cost_node_computes_the_terms_own_worked_figures():
    # "The monthly minimum wage applicable to sheepherders in effect on September 30, 2026 is $4,938.21, so the Net
    # Monthly Cost test threshold is $5,678.94, and the bill's $5,200 Net Monthly Cost does not exceed it." The terms
    # state no rounding: the threshold is 115% of $4,938.21 exactly, $5,678.9415, which the example prints to the cent.
    assert excess() == -F(5678_9415, 10_000)
    assert excess(monthly_wage=D(5200)) == F(5200) - F(5678_9415, 10_000) < 0
    # "... a monthly wage of no less than $12,000 ... the bill's $12,000 Net Monthly Cost exceeds it."
    assert excess(monthly_wage=D(12000)) > 0
    # "Does not exceed": a cost equal to the threshold passes, a cent more does not.
    assert excess(monthly_wage=D("5678.9415")) == 0
    assert excess(monthly_wage=D("5678.95")) > 0
    # The sheepherder framework: the Net Monthly Cost is the published monthly minimum, overtime included.
    assert excess(sheepherder_framework=1) == F(493821, 100) - F(5678_9415, 10_000)
    # The hourly defaults: "40 hours at the regular rate, 40 hours at one and one half times the regular rate, and 88
    # hours at two times the regular rate; and one calendar month equals 52/12 workweeks", at the minimum wage.
    hourly = excess(regular_hours=40, overtime_hours=40, double_time_hours=88)
    assert hourly == F(1690, 100) * (40 + 60 + 176) * F(52, 12) - F(5678_9415, 10_000) > 0
    # A credit available as of right is subtracted.
    assert excess(monthly_wage=D(6000), monthly_credit=D(400)) == F(6000 - 400) - F(5678_9415, 10_000) < 0


def test_the_terms_resolve_yes_when_a_qualifying_actions_conditions_hold_and_its_cost_is_within_the_threshold():
    # AB 187's interpretations supplied as affirmative, its terms read as the sheepherder framework with nothing of its
    # own, and the two rates DIR published: the Net Monthly Cost, computed by the registered node over the readings'
    # bands at their corners, is at most the threshold, so the trigger is true.
    ev = evaluate(wg.ir(wg.AB187), wg.rates(), wg.ab187(), at=wg.SETTLED)
    assert ev.trigger_truth is Kleene.TRUE and ev.procedure_status == "resolved_yes"
    # The same instrument with a monthly wage of its own above the threshold fails the test, and the trigger with it.
    dear = {**wg.ab187(), "read_sheepherder_framework": wg.FLAG[0], "read_monthly_wage": (D(10000), D(15000))}
    assert evaluate(wg.ir(wg.AB187), wg.rates(), dear, at=wg.EXPIRATION).trigger_truth is Kleene.FALSE
    # Unless it enjoins the obligations in their entirety, which the terms deem to satisfy the test.
    enjoined = {**dear, "enjoins_entirely": Kleene.TRUE}
    assert evaluate(wg.ir(wg.AB187), wg.rates(), enjoined, at=wg.EXPIRATION).trigger_truth is Kleene.TRUE
    # Without the rates, the cost is not computed from a default: the test is unknown, and so is the trigger.
    assert evaluate(wg.ir(wg.AB187), [], wg.ab187(), at=wg.SETTLED).trigger_truth is Kleene.UNKNOWN


def test_the_market_closed_early_and_settled_yes_as_the_ir_resolves():
    # The outside check: the market's own record.
    assert MARKET["ticker"] == "KXCAGOAT-26SEP30" and MARKET["result"] == "yes" and MARKET["status"] == "finalized"
    assert MARKET["can_close_early"] is True
    k = wg.ir(wg.AB187)
    assert datetime.fromisoformat(MARKET["expiration_time"]) == k.timing.expiration  # the latest expiration
    assert datetime.fromisoformat(MARKET["open_time"]) == k.timing.open
    assert D(MARKET["settlement_value_dollars"]) == k.payout.unit_value
    closed = datetime.fromisoformat(MARKET["close_time"])
    assert closed == wg.EARLY_CLOSE < k.timing.expiration  # it closed early: Rule 7.2
    # Once the qualifying action is evidenced, the IR resolves YES by early expiration, before the scheduled one. The
    # early date is the exchange's under Rule 7.2, so it stays unresolved, named, and so does the payment's: no date
    # is taken from the scheduled expiration in its place.
    for when in (closed, wg.SETTLED, wg.EXPIRATION):
        ev = evaluate(k, wg.rates(), wg.ab187(), at=when)
        assert ev.procedure_status == "resolved_yes" and ev.trigger_truth is Kleene.TRUE
        assert ev.transfers == ()
        assert ev.unresolved[0].startswith("expiration date: the early expiration, whose date Kalshi Rulebook Rule 7.2")
        assert ev.unresolved[1].startswith("payout: 1.00 USD per unit, paid 1 + 0 calendar days after the early")
        assert c.Ref(id="KXCAGOAT-26SEP30/resolution/early_expiration/when_true/per_exchange_rule", version=1) in ev.policies
    # Before AB 187 became law its conditions were not established: the market traded, and the IR is pending.
    before = evaluate(k, wg.rates(), wg.ab187(Kleene.UNKNOWN), at=wg.AB187_LAW - timedelta(hours=1))
    assert before.procedure_status == "pending" and before.trigger_truth is Kleene.UNKNOWN


def test_pending_before_expiration_and_the_contracts_rule_at_expiration_where_nothing_is_established():
    k = wg.ir(wg.AB187)
    pending = evaluate(k, wg.rates(), wg.ab187(Kleene.UNKNOWN), at=datetime(2026, 9, 10, tzinfo=UTC))
    assert pending.procedure_status == "pending" and pending.trigger_truth is Kleene.UNKNOWN
    assert pending.unresolved[0] == "the procedure resolves at expiration, 2026-10-07T14:00:00+00:00"
    # At the expiration, the action still not established: "Only actions documented by a Source Agency on or before the
    # Expiration Date at the Expiration time will be considered", so NO, with the trigger's truth kept unknown.
    at_expiry = evaluate(k, wg.rates(), wg.ab187(Kleene.UNKNOWN), at=wg.EXPIRATION)
    assert at_expiry.trigger_truth is Kleene.UNKNOWN and at_expiry.procedure_status == "resolved_no"
    assert c.Ref(id="KXCAGOAT-26SEP30/resolution/on_unknown_at_expiry/no", version=1) in at_expiry.policies
    assert at_expiry.transfers == ()
    # An instrument limited to the parties of an individual case (the terms' last NO example) is definitively not one.
    individual = {**wg.ab187(), "statewide_scope": Kleene.FALSE}
    no = evaluate(k, wg.rates(), individual, at=wg.EXPIRATION)
    assert no.trigger_truth is Kleene.FALSE and no.procedure_status == "resolved_no"
    assert c.Ref(id="KXCAGOAT-26SEP30/resolution/on_unknown_at_expiry/no", version=1) not in no.policies


def test_every_operative_exclusion_names_the_condition_that_carries_it():
    k = wg.ir(wg.AB187)
    nodes = {n.id for n in c.walk(k.trigger)}
    operative = [x for x in k.exclusions if x.operative]
    assert len(operative) == 13 and all(x.ir_node in nodes for x in operative)
    assert {x.ir_node for x in operative} == {"instrument_kind", "qualifying_effect"}
