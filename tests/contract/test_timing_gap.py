"""A payout that matches the loss in total but arrives after the cash loss (addendum §13 row 6; A05 §5.3; A06 §6.6): the
evaluator's dated transfers say when the payout arrives, and the liquidity optimizer, given those dated transfers on its
grid, shows the timing gap that the terminal match hides. The dates are the evaluator's; the consequence is the
optimizer's."""

from datetime import date, datetime
from decimal import Decimal as D
from fractions import Fraction as F
from zoneinfo import ZoneInfo

import pytest
from irkit import at, contract, obs, observable, timing

from requisite import core as c
from requisite.contract.evaluate import evaluate
from requisite.optimize import Candidate, LiquidityProblem, solve, try_candidate
from requisite.optimize import evaluate as cash

NY = ZoneInfo("America/New_York")
GRID = (date(2027, 1, 1), date(2027, 1, 31), date(2027, 3, 2))  # day 0, day 30 (the cash loss), day 60
EVENT = datetime(2027, 1, 25, 12, tzinfo=NY)


def protection(settlement_lag_days: int) -> c.ContractIR:
    """Pays 120 per unit if the loss event is recorded, determined at expiration, January 31, 2027 at 10:00 in New York."""
    expiry = datetime(2027, 1, 31, 10, tzinfo=NY)
    return contract(
        at("x", ">=", 1, EVENT), observables=[observable()],
        payout=c.Payout(kind="binary", unit_value=D("120"), currency="USD", rounding_quantum=D("0.01"),
                        rounding_mode="half_even", payment="at_settlement"),
        timing=timing(open=datetime(2027, 1, 1, tzinfo=NY), close=expiry, expiration=expiry, observation_cutoff=expiry,
                      settlement_lag_days=settlement_lag_days, tz="America/New_York"),
    )  # fmt: skip


def receipts_on_grid(transfers: tuple[c.CashTransfer, ...]) -> list[D]:
    """The evaluator's dated transfers as per-unit receipts on the grid's dates (the optimizer's h_j,s,tau, A06 §6.1):
    each transfer's date in the contract's zone must be a grid date, so nothing is moved to fit."""
    out = [D(0)] * len(GRID)
    for t in transfers:
        day = t.at.astimezone(NY).date()
        assert day in GRID, f"the transfer on {day} is not on the grid {GRID}"
        out[GRID.index(day)] += t.amount
    return out


@pytest.mark.covers("A.T06")
def test_a_payout_matching_the_loss_but_paid_after_it_leaves_the_timing_gap():
    loss_event = [obs(1, EVENT)]
    # The evaluator dates the payout: thirty days of settlement lag put it on March 2, day 60; none, on day 30.
    late = evaluate(protection(30), loss_event, {}, at=datetime(2027, 3, 3, tzinfo=NY))
    prompt = evaluate(protection(0), loss_event, {}, at=datetime(2027, 3, 3, tzinfo=NY))
    assert late.procedure_status == prompt.procedure_status == "resolved_yes"
    assert [(t.at, t.amount) for t in late.transfers] == [(datetime(2027, 3, 2, 10, tzinfo=NY), D("120.00"))]
    assert [(t.at, t.amount) for t in prompt.transfers] == [(datetime(2027, 1, 31, 10, tzinfo=NY), D("120.00"))]

    # The business: cash 100, a loss of 120 on day 30, the protection costing 10 on day 0 (A06 §6.6's fixture).
    business = LiquidityProblem(
        dates=GRID, probabilities=(F(1),), initial_cash=D(100), operating=[[0, 0, 0]], loss=[[D(0), D(120), D(0)]],
        reserve=[0, 0, 0], alpha=F(9, 10), budget=D(10),
    )  # fmt: skip
    late_j = Candidate("late", D(10), GRID[0], [receipts_on_grid(late.transfers)], D(1))
    prompt_j = Candidate("prompt", D(10), GRID[0], [receipts_on_grid(prompt.transfers)], D(1))
    # Held, the late payout matches the loss in total: the terminal residual is the premium alone...
    held = cash(business.with_candidate(late_j), [1])
    assert held.loss == held.receipts == (D(120),) and held.residual == (D(10),)
    # ...yet on day 30 the business is 30 short, worse than unhedged (20): the terminal match is not protection.
    assert cash(business, []).deficit == (D(20),)
    assert held.deficit == (D(30),) and held.breach_dates == ((GRID[1],),)
    # The optimizer buys none of it; the same payout dated on day 30 closes the gap.
    before = solve(business)
    assert try_candidate(before, late_j).after.quantities == (0,)
    with_prompt = try_candidate(before, prompt_j).after
    assert with_prompt.hedged.breach_dates == ((),) and with_prompt.hedged.breach_probability == 0
