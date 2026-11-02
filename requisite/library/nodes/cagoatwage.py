"""The Net Monthly Cost test of Kalshi's CAGOATWAGE contract (series KXCAGOAT, "California goat herder wage
obligations"), the Western Grazers contract spec §9.3 names as its reference specimen. Its terms are the filed PDF in
fixtures/kalshi/KXCAGOAT/contract_terms.pdf; every rule below quotes them.

The test: "An instrument satisfies the Net Monthly Cost test if the Net Monthly Cost under the instrument does not exceed
the monthly minimum wage applicable to sheepherders under California Labor Code Section 2695.2 and Section 4(E) of
Industrial Welfare Commission Wage Order No. 14-2001, plus 15%, as in effect on the last day of <time period>."

"Net Monthly Cost means the employer's minimum required wage and overtime cost of compensating a goat herder employed on
a regularly scheduled 24-hour shift, seven days per week, for one calendar month, computed under the wage and overtime
rules as modified, limited, or rendered unenforceable by the instrument, minus the monthly value of any credit,
reimbursement, grant, subsidy, payment, or contract price adjustment legally available to the employer under the
instrument."

Where compensation is hourly, "all 24 hours of each shift are treated as compensable; the applicable rate is the
California minimum wage in effect on the last day of <time period>; each workweek consists of 40 hours at the regular
rate, 40 hours at one and one half times the regular rate, and 88 hours at two times the regular rate; and one calendar
month equals 52/12 workweeks", "except to the extent the instrument itself legally modifies the number of compensable
hours, the applicable hourly wage rate, or the applicable premium pay requirements, in which case the instrument's
modified terms govern".

So the instrument's terms (read from its text, by a resolver: ReadValue) and two published rates (ValueAt) determine the
cost:

    hourly at the minimum wage  = minimum_wage x (regular_hours + 1.5 overtime_hours + 2 double_time_hours) x 52/12
    hourly at a special rate    = special_rate x (special_regular_hours + 1.5 special_overtime_hours
                                                  + 2 special_double_time_hours) x 52/12
    Net Monthly Cost            = monthly_wage + sheepherder_framework x sheepherder_monthly_wage
                                  + hourly at the minimum wage + hourly at a special rate - monthly_credit
    excess                      = Net Monthly Cost - 115/100 x sheepherder_monthly_wage

and the test holds when the excess is at most zero ("does not exceed"). The inputs, per goat herder:

- monthly_wage: a monthly wage, in USD a month, that the instrument requires in lieu of hourly pay for all hours (zero
  where it sets none).
- sheepherder_framework: 1 where the instrument makes the sheepherder alternative monthly wage (Labor Code §§2695.1 and
  2695.2) the goat herder's minimum, 0 where it does not. Its amount is the published sheepherder monthly minimum, which
  includes the overtime the framework requires (DIR publishes it as one total; the terms' own example reads that total,
  $4,938.21 for 2026, as "the monthly minimum wage applicable to sheepherders").
- regular_hours, overtime_hours, double_time_hours: compensable hours a week paid at 1, 1.5 and 2 times the California
  minimum wage under the instrument (the terms' defaults 40, 40 and 88 where the instrument leaves hourly rules in place;
  zero where a monthly wage replaces them).
- special_rate and special_regular_hours, special_overtime_hours, special_double_time_hours: an hourly rate the
  instrument sets in place of the minimum wage ("a special minimum wage"), in USD an hour, and the hours paid at it.
- monthly_credit: the monthly value of any credit, reimbursement, grant, subsidy, payment or contract price adjustment
  available as of right under the instrument, in USD a month.
- minimum_wage: the California minimum wage in effect on the last day of the period, in USD an hour (published by DIR).
- sheepherder_monthly_wage: the sheepherder monthly minimum in effect on the last day of the period, in USD a month
  (published by DIR).

The computation is exact (Fractions): the terms state no rounding of the cost or of the threshold. Every input is
non-negative and sheepherder_framework is at most one, which the node checks; on that domain the excess moves with each
input in one direction (it is linear in each), as declared below, so over interval inputs the evaluator's two corners
bound it exactly. "An instrument that enjoins or prohibits the enforcement or collection of the applicable minimum wage
and overtime obligations in their entirety is deemed to satisfy this test" is a condition of its own in the IR, beside
this node, not part of the arithmetic.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

from requisite.compute.registry import node

WEEKS_PER_MONTH = Fraction(52, 12)  # "one calendar month equals 52/12 workweeks"
THRESHOLD_FACTOR = Fraction(115, 100)  # "plus 15%"

INPUTS = [
    "monthly_wage",
    "sheepherder_framework",
    "regular_hours",
    "overtime_hours",
    "double_time_hours",
    "special_rate",
    "special_regular_hours",
    "special_overtime_hours",
    "special_double_time_hours",
    "monthly_credit",
    "minimum_wage",
    "sheepherder_monthly_wage",
]


def _exact(name: str, x: object) -> Fraction:
    if isinstance(x, bool) or isinstance(x, float) or not isinstance(x, Fraction | Decimal | int):
        raise TypeError(f"{name} is an exact number (Fraction, Decimal or int), not {type(x).__name__}")
    value = Fraction(x)
    if value < 0:
        raise ValueError(f"{name} is not negative: {value}")
    return value


@node(
    id="cagoatwage.net_monthly_cost_excess",
    version=1,
    inputs=INPUTS,
    monotone={
        "monthly_wage": "increasing",
        "sheepherder_framework": "increasing",
        "regular_hours": "increasing",
        "overtime_hours": "increasing",
        "double_time_hours": "increasing",
        "special_rate": "increasing",
        "special_regular_hours": "increasing",
        "special_overtime_hours": "increasing",
        "special_double_time_hours": "increasing",
        "monthly_credit": "decreasing",
        "minimum_wage": "increasing",
        "sheepherder_monthly_wage": "decreasing",
    },
)
def net_monthly_cost_excess(**inputs: object) -> Fraction:
    """The Net Monthly Cost under the instrument less the test's threshold, in USD a month; at most zero passes."""
    if set(inputs) != set(INPUTS):
        raise TypeError(f"the Net Monthly Cost node takes exactly {INPUTS}")
    x = {k: _exact(k, v) for k, v in inputs.items()}
    if x["sheepherder_framework"] > 1:
        raise ValueError("sheepherder_framework is 0 or 1: whether the instrument makes the sheepherder wage apply")
    at_minimum_wage = x["minimum_wage"] * (x["regular_hours"] + Fraction(3, 2) * x["overtime_hours"] + 2 * x["double_time_hours"])
    at_special_rate = x["special_rate"] * (
        x["special_regular_hours"] + Fraction(3, 2) * x["special_overtime_hours"] + 2 * x["special_double_time_hours"]
    )
    net_monthly_cost = (
        x["monthly_wage"]
        + x["sheepherder_framework"] * x["sheepherder_monthly_wage"]
        + (at_minimum_wage + at_special_rate) * WEEKS_PER_MONTH
        - x["monthly_credit"]
    )
    return net_monthly_cost - THRESHOLD_FACTOR * x["sheepherder_monthly_wage"]
