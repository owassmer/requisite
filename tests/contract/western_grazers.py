"""The Western Grazers contract (Kalshi rulebook CAGOATWAGE, series KXCAGOAT), market KXCAGOAT-26SEP30, expressed by hand
in the IR from its filed terms (fixtures/kalshi/KXCAGOAT/contract_terms.pdf, the certification contract.pdf, and the
market's fields in markets.json). Spec §9.3 names it as the reference specimen; this IR is the test's, written so that
the evaluator is proven on the real structure. Every element quotes the terms.

<time period>: the market's is "before Oct 1, 2026" (its title and rules_primary), so the end of the period is
"11:59:59 PM ET on the final day of the specified period", September 30, 2026. "The latest Expiration Date of the Contract
shall be one week after the end of <time period>"; "The Expiration time of the Contract shall be 10:00 AM ET": October 7,
2026, 10:00 ET, the market's expiration_time (2026-10-07T14:00:00Z).

The Payout Criterion: "a Qualifying Action has been taken <time period>": an instrument of one of five kinds, "Which has
at least one of the following effects", "And which" applies statewide, was enacted by the end of the period, took
effect by it, is not stayed "as of the Expiration Date at the Expiration time", and "satisfies the Net Monthly Cost
test", which "An instrument that enjoins or prohibits the enforcement or collection of the applicable minimum wage and
overtime obligations in their entirety is deemed to satisfy". Each condition a resolver judges from text is an
Interpretive node; the Net Monthly Cost test is a NumCmp over the registered node cagoatwage.net_monthly_cost_excess
(requisite/library/nodes/cagoatwage.py), on the instrument's terms (ReadValue) and two rates DIR publishes (ValueAt).

What the IR cannot carry exactly, so this IR rests on these assumptions (the grammar findings are in the task report):
- The criterion quantifies over instruments ("a Qualifying Action": some instrument satisfying every condition). The IR
  has no existential over documents, so `ir(instrument)` is the criterion's body for one named instrument, the one the
  evidence identifies. It is exact for YES (that instrument qualifies) and, for NO, speaks only for that instrument.
- "If an Expiration Value cannot be determined on the Expiration Date, Kalshi has the right to determine payouts pursuant
  to Rule 7.1": an exchange-discretion outcome the procedure cannot name. The IR takes the terms' "Only actions
  documented by a Source Agency on or before the Expiration Date at the Expiration time will be considered" as its rule
  for an unknown at expiry: an action not established is not considered, so the contract resolves NO.
- "Where the records of the Source Agencies conflict ... the official records of the body that took the action govern":
  not one of the registered conflict procedures; it governs the evidence of the interpretive conditions, which the IR's
  sources do not model. The two numeric observables have one source, DIR, so their procedure is precedence.
- "The Settlement Date of the Contract shall be no later than the day after the Expiration Date": a deadline, taken as a
  lag of one calendar day (the latest the terms allow).
- The terms state no void path and no rounding; the IR's void_transfer (never reached, since its unknown rule is NO) and
  rounding mode are required by the definition and are not the terms'.
- The Net Monthly Cost's readings are bands (core.ReadValue): an answer such as "none" is the band [0, 1), not 0, since
  only the last band of a reading can be a single value.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal as D
from typing import Any
from zoneinfo import ZoneInfo

from requisite import core as c
from requisite.core.evidence import artifact_id
from requisite.core.kleene import Kleene

ET = ZoneInfo("America/New_York")
PT = ZoneInfo("America/Los_Angeles")
OPEN = datetime(2026, 7, 31, 20, 45, tzinfo=UTC)  # markets.json open_time
PERIOD_END = datetime(2026, 9, 30, 23, 59, 59, tzinfo=ET)  # "11:59:59 PM ET on the final day of the specified period"
EXPIRATION = datetime(2026, 10, 7, 10, 0, tzinfo=ET)  # "one week after the end of <time period>", "10:00 AM ET"
EARLY_CLOSE = datetime(2026, 9, 19, 15, 10, 59, tzinfo=UTC)  # markets.json close_time: the market closed early
SETTLED = datetime(2026, 9, 19, 15, 41, 4, 503741, tzinfo=UTC)  # markets.json settlement_ts, result "yes"
AB187_LAW = datetime(2026, 9, 18, tzinfo=PT)  # AB 187 became law on the Governor's signature, September 18, 2026
RULE = "Kalshi Rulebook Rule 7.2"

DIR = c.Source(id="ca_dir", name="California Department of Industrial Relations", url="https://www.dir.ca.gov",
               rank=5, role="official_record")  # fifth in the terms' "hierarchical order" of Source Agencies  # fmt: skip
MINIMUM_WAGE = c.Observable(
    id="ca_minimum_wage", description="The California minimum wage in effect on a day (General Minimum Wage Order)",
    value_type="number", sources=[DIR], schedule="annual, by General Minimum Wage Order",
    revision_policy="as_of_expiry", unit="USD/hour", precision=D("0.01"),
)  # "Revisions to the Underlying made after Expiration will not be accounted for"  # fmt: skip
SHEEPHERDER_MINIMUM = c.Observable(
    id="ca_sheepherder_monthly_minimum",
    description="The monthly minimum wage applicable to sheepherders under Labor Code §2695.2 and §4(E) of IWC Wage "
                "Order 14-2001, with required overtime, in effect on a day",
    value_type="number", sources=[DIR], schedule="annual, Labor Commissioner's chart",
    revision_policy="as_of_expiry", unit="USD/month", precision=D("0.01"),
)  # fmt: skip


def question(nid: str, template: str, text: str, instrument: str) -> c.Interpretive:
    return c.Interpretive(id=nid, question_template=template,
                          bindings={"instrument": instrument, "condition": text, "period_end": PERIOD_END.isoformat()})  # fmt: skip


HOURS = [(D(0), D(1)), (D(1), D(8)), (D(8), D(16)), (D(16), D(24)), (D(24), D(32)), (D(32), D(40)), (D(40), D(41)),
         (D(41), D(56)), (D(56), D(88)), (D(88), D(89)), (D(89), D(168))]  # fmt: skip
MONTHLY = [(D(0), D(1)), (D(1), D(1000)), (D(1000), D(2000)), (D(2000), D(3000)), (D(3000), D(4000)),
           (D(4000), D(5000)), (D(5000), D(5500)), (D(5500), D(6000)), (D(6000), D(8000)), (D(8000), D(10000)),
           (D(10000), D(15000)), (D(15000), D(100000))]  # fmt: skip
RATE = [(D(0), D("0.01")), (D("0.01"), D(5)), (D(5), D(10)), (D(10), D(15)), (D(15), D(20)), (D(20), D(100))]
CREDIT = [(D(0), D(1)), (D(1), D(500)), (D(500), D(1000)), (D(1000), D(2000)), (D(2000), D(5000)),
          (D(5000), D(100000))]  # fmt: skip
FLAG = [(D(0), D(1)), (D(1), D(1))]

# Each reading: its node input, unit, bands, and the term of the instrument it reads (the node's docstring states each).
READINGS: dict[str, tuple[str, list[tuple[D, D]], str]] = {
    "monthly_wage": ("USD/month", MONTHLY, "the monthly wage the instrument requires in lieu of hourly pay for all hours"),
    "sheepherder_framework": ("flag", FLAG, "whether the instrument makes the sheepherder alternative monthly wage "
                                            "framework (Labor Code §§2695.1, 2695.2) the goat herder's minimum"),
    "regular_hours": ("hours/week", HOURS, "compensable hours a week paid at the minimum wage"),
    "overtime_hours": ("hours/week", HOURS, "compensable hours a week paid at one and one half times the minimum wage"),
    "double_time_hours": ("hours/week", HOURS, "compensable hours a week paid at two times the minimum wage"),
    "special_rate": ("USD/hour", RATE, "an hourly rate the instrument sets in place of the minimum wage"),
    "special_regular_hours": ("hours/week", HOURS, "compensable hours a week paid at that rate"),
    "special_overtime_hours": ("hours/week", HOURS, "compensable hours a week paid at one and one half times that rate"),
    "special_double_time_hours": ("hours/week", HOURS, "compensable hours a week paid at two times that rate"),
    "monthly_credit": ("USD/month", CREDIT, "the monthly value of any credit, reimbursement, grant, subsidy, payment "
                                            "or contract price adjustment available as of right under the instrument"),
}  # fmt: skip


def ir(instrument: str, *, examples: tuple[c.ScenarioExample, ...] = ()) -> c.ContractIR:
    """The Payout Criterion for one instrument, with the contract's timing, payout and procedure."""
    q = lambda nid, text: question(nid, f"cagoatwage.{nid}", text, instrument)  # noqa: E731
    readings = {
        name: c.ReadValue(id=f"read_{name}", question_template=f"cagoatwage.read_{name}",
                          bindings={"instrument": instrument, "term": term}, unit=unit, bands=bands)
        for name, (unit, bands, term) in READINGS.items()
    }  # fmt: skip
    cost = c.Computed(
        id="net_monthly_cost_excess", node=c.Ref(id="cagoatwage.net_monthly_cost_excess", version=1),
        inputs={
            **readings,
            "minimum_wage": c.ValueAt(id="minimum_wage_at_period_end", observable="ca_minimum_wage", at=PERIOD_END),
            "sheepherder_monthly_wage": c.ValueAt(id="sheepherder_minimum_at_period_end",
                                                  observable="ca_sheepherder_monthly_minimum", at=PERIOD_END),
        },
    )  # fmt: skip
    trigger = c.And(id="qualifying_action", args=[
        q("instrument_kind", "a California statute that has become law; a regulation finally approved by the Office of "
          "Administrative Law and filed with the Secretary of State, or an IWC wage order amended or republished that "
          "has taken effect; a written regulation, binding directive, final order or formal statewide enforcement "
          "policy of a California agency with authority over the subject matter; an executive order or proclamation of "
          "the Governor; or a written order of a California state court or a United States federal court"),
        q("qualifying_effect", "has at least one of the six effects of the Payout Criterion on goat herder minimum "
          "wage or overtime obligations"),
        q("statewide_scope", "applies statewide, or is generally available to employers of goat herders as a class, "
          "and is not limited to a named employer, a named individual, or the parties to an individual, non-class "
          "case"),
        q("dated_by_period_end", "was enacted, approved and filed, issued, or entered, as applicable, on or before the "
          "end of <time period>, by its legal date"),
        q("effective_by_period_end", "has qualifying provisions that have taken legal effect, or by their express "
          "terms apply to compensation for work performed, on or before the end of <time period>"),
        q("operative_at_expiration", "has qualifying provisions that have not been stayed, enjoined, suspended, "
          "invalidated, rescinded, or superseded as of the Expiration Date at the Expiration time"),
        c.Or(id="net_monthly_cost_test", args=[
            c.NumCmp(id="net_monthly_cost_at_most_threshold", expr=cost, op="<=", value=D("0.00")),  # to the cent
            q("enjoins_entirely", "enjoins or prohibits the enforcement or collection of the applicable minimum wage "
              "and overtime obligations in their entirety"),
        ]),
    ])  # fmt: skip
    exclusions = [
        *(c.Exclusion(text=t, operative=True, ir_node="instrument_kind") for t in (
            "a bill that has not become law, including a bill passed by one or both chambers and a bill vetoed and not "
            "overridden",
            "a proposed regulation, rulemaking notice, petition, or draft policy that has not been finally approved and "
            "filed",
            "the filing or pendency of litigation without a written order of a California state court or a United "
            "States federal court",
            "a nonbinding advisory, an opinion letter that neither binds the issuing body nor expressly states that "
            "body's statewide enforcement position, a staff communication, a frequently-asked-questions page, or a "
            "website statement",
            "a statement, press release, hearing, study, report, recommendation, or political agreement",
            "federal legislation and federal executive or administrative action, provided that a written order of a "
            "United States federal court is not excluded by this item",
            "action by a city, county, fire district, or other local government, unless it implements a qualifying "
            "state program",
            "a private settlement or agreement not entered as a written order of a California state court or a United "
            "States federal court",
        )),
        *(c.Exclusion(text=t, operative=True, ir_node="qualifying_effect") for t in (
            "an authorization of a future program that requires a further appropriation, implementing legislation, or "
            "discretionary agency approval before relief becomes legally available",
            "a general appropriation or funding for agriculture, grazing, fire prevention, or vegetation management not "
            "expressly tied to goat herder minimum wage or overtime obligations",
            "a reenactment limited to the definitional, housing, meal period, safety, communication, transportation, or "
            "other non-wage provisions formerly contained in California Labor Code Sections 2695.3 and 2695.4",
            "relief consisting solely of a limitation on damages, penalties, or interest, without any of (a) any "
            "reduction of the underlying wage, (b) any reduction of overtime obligation, OR (c) any qualifying payment",
            "A stay entered solely on procedural grounds, including arbitration, venue, discovery, bankruptcy, or case "
            "management, is not a Qualifying Action unless the order itself bars enforcement or collection of the "
            "underlying wage or overtime obligation.",
        )),
        c.Exclusion(text="Persons who are employed by any of the Source Agencies are not permitted to trade on the "
                         "Contract.", operative=False),
    ]  # fmt: skip
    return c.ContractIR(
        id="KXCAGOAT-26SEP30", version=1, observables=[MINIMUM_WAGE, SHEEPHERDER_MINIMUM], trigger=trigger,
        payout=c.Payout(kind="binary", unit_value=D("1.00"), currency="USD", rounding_quantum=D("0.01"),
                        rounding_mode="half_even", payment="at_settlement"),  # "The Settlement Value ... is $1.00"
        timing=c.Timing(open=OPEN, close=EXPIRATION, expiration=EXPIRATION, observation_cutoff=EXPIRATION,
                        settlement_lag_days=1, payment_lag_days=0, lag_days="calendar", holiday_calendar=None,
                        tz="America/New_York"),
        resolution=c.ResolutionProcedure(
            precedence=["ca_dir"], conflict_policy=c.ConflictPolicy(procedure="precedence_first_available"),
            on_unknown_at_expiry="no", early_expiration="when_true", early_expiration_at="per_exchange_rule",
            early_expiration_rule=RULE, void_transfer=c.VoidTransfer(returns="nothing"),
        ),
        examples=examples, exclusions=exclusions,
    )  # fmt: skip


# ---------------------------------------------------------------- the published rates (ValueAt's observations)
# Each artifact id is that of the passage as read from its page (research/sources/ keeps it); the pages themselves are
# frozen by a harvest run through requisite/evidence, not here. Each published_at is an upper bound the source supports.

MW_PASSAGE = b"https://www.dir.ca.gov/IWC/MW-2026.pdf\nEffective January 1, 2026, Minimum Wage: $16.90 per hour"
SHEEP_PASSAGE = (b"https://www.dir.ca.gov/dlse/Sheepherders-owed-as-a-result-of-AB-1066s-Overtime-Phase-In.html\n"
                 b"2026 $3,004.50 monthly minimum wage plus $1,933.71 required overtime pay, for a total of $4,938.21 "
                 b"per month")  # fmt: skip


def rates() -> list[c.Observation]:
    """The two rates in effect on the last day of the period, as DIR published them."""
    return [
        # MW-2026: "These Amendments to the Wage Orders shall be in effect as of January 1, 2026", so published by then.
        c.Observation(observable="ca_minimum_wage", source="ca_dir", at=PERIOD_END, value=c.Number(value=D("16.90")),
                      published_at=datetime(2026, 1, 1, tzinfo=PT), final=True, artifact=artifact_id(MW_PASSAGE)),
        # The Labor Commissioner's chart, its version dated "August 2026" (the figure is also in the terms filed July 31).
        c.Observation(observable="ca_sheepherder_monthly_minimum", source="ca_dir", at=PERIOD_END,
                      value=c.Number(value=D("4938.21")), published_at=datetime(2026, 8, 31, 23, 59, 59, tzinfo=PT),
                      final=True, artifact=artifact_id(SHEEP_PASSAGE)),
    ]  # fmt: skip


# ---------------------------------------------------------------- the qualifying action's inputs: AB 187 (2026)

AB187 = "California AB 187 (2026), signed by the Governor on September 18, 2026"
NONE = {"monthly_wage": MONTHLY[0], "regular_hours": HOURS[0], "overtime_hours": HOURS[0],
        "double_time_hours": HOURS[0], "special_rate": RATE[0], "special_regular_hours": HOURS[0],
        "special_overtime_hours": HOURS[0], "special_double_time_hours": HOURS[0], "monthly_credit": CREDIT[0]}  # fmt: skip


def ab187(truth: Kleene = Kleene.TRUE) -> dict[str, Any]:
    """AB 187's interpretations as of the market's early close: each condition affirmative (or `truth`, for the state in
    which none is established yet), "enjoins entirely" negative, and its terms read as the sheepherder framework with no
    monthly wage, hourly pay or credit of its own. Where each comes from is in the task report and in research/sources:
    AllCiv (law on signing September 18, effective immediately, extends the sheepherder framework to goat herders
    statewide) and the California Farm Bureau (reenacts Labor Code §2695.4)."""
    unknown = truth is Kleene.UNKNOWN
    out: dict[str, Any] = {
        nid: truth
        for nid in ("instrument_kind", "qualifying_effect", "statewide_scope", "dated_by_period_end",
                    "effective_by_period_end", "operative_at_expiration")
    }  # fmt: skip
    out["enjoins_entirely"] = Kleene.UNKNOWN if unknown else Kleene.FALSE
    for name, (_, bands, _) in READINGS.items():
        if unknown:
            out[f"read_{name}"] = (bands[0][0], bands[-1][1])  # not established: the whole range
        elif name == "sheepherder_framework":
            out[f"read_{name}"] = FLAG[1]  # [1, 1]: the framework applies
        else:
            out[f"read_{name}"] = NONE[name]
    return out
