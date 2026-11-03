"""Construction's structure templates (spec §8.2; spec §9: a ClauseTemplate is "an IR fragment, a prose template and an
example generator", and these are its IR fragments): each builds a candidate contract's IR from candidate observables
(core.ObservableCandidate), the event's window and the contract's open, by kind, never by case (design/posture.md).

Families (spec §8.2): a digital pays a fixed amount if the event's adverse state shows by a deadline; a time-in-state
strip pays per day the adverse state holds, per period, since for policy risk "a single digital pays the same whether
relief lasts one month or five years"; a compound strip pays per day two events' states hold together, or either holds.
A ladder needs an observable whose level the scenario model varies across several adverse values, and none does: each
candidate observable publishes one value while its event holds (core.ObservableCandidate.occurs), so a ladder's rungs
would pay together, as one digital. Variants (spec §8.2: "retroactivity, partial relief and caps"): the stated revision
policy, or retroactive (as_of_expiry: a later record that the state held on a day counts, until expiry); the full
adverse state only, or also its partial form where the observable has one; caps by term (a strip's term caps the days it
pays for: six, twelve or twenty-four months, ours, a half-year to two years over which relief questions run); and the
same twelve-month count paid per period or once at expiry (A05 §5.3: "A count paid once at expiry and a strip paid
periodically can have identical terminal amounts and very different liquidity consequences").

Terms that depend on nothing in a case (ours, each stated in every IR): one unit pays one unit of the case's currency
(per day for a strip), rounded to the cent half-even; settlement one calendar day after resolution (the reference
specimen, fixtures/kalshi/KXCAGOAT: "The Settlement Date of the Contract shall be no later than the day after the
Expiration Date") and payment on it; the observable's sources consulted in their rank (precedence_first_available); an
adverse state not established by the cutoff pays nothing (on_unknown_at_expiry "no": the specimen's "Only actions
documented ... will be considered"; the trigger's truth stays unknown, A05 §5.2); no void path. Every date is in the
event window's zone. A contract opens at the decision time and observes from the first midnight at or after it.

No term reads a business's planning horizon, so moving the horizon changes what a contract does for the business (which
of its payments fall within the horizon) and never what it says (spec §15 beat 5: "exposure and optimizer nodes changing
while settlement logic does not").
"""

from __future__ import annotations

import calendar as cal
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, Literal
from zoneinfo import ZoneInfo

from requisite import core
from requisite.core.contract import Number

UNIT = Decimal(1)
QUANTUM = Decimal("0.01")
SETTLEMENT_LAG_DAYS = 1
TERMS = (6, 12, 24)  # a strip's terms in months
BASE_TERM = 12  # the term the timing, retroactivity, partial and compound variants are written over

DIGITAL = core.Ref(id="structure.digital", version=1)
STRIP = core.Ref(id="structure.time_in_state_strip", version=1)
COUNT = core.Ref(id="structure.time_in_state_count", version=1)
COMPOUND = core.Ref(id="structure.compound_strip", version=1)

Based = tuple[str, core.ObservableCandidate]  # a candidate observable's proposal id and record


def _midnight(day: date, tz: str) -> datetime:
    return datetime.combine(day, time(0), tzinfo=ZoneInfo(tz))


def first_midnight(at: datetime, tz: str) -> datetime:
    """The first local midnight at or after an instant."""
    local = at.astimezone(ZoneInfo(tz))
    m = _midnight(local.date(), tz)
    return m if m >= at else _midnight(local.date() + timedelta(days=1), tz)


def add_months(day: date, months: int) -> date:
    y, m = divmod(day.month - 1 + months, 12)
    year, month = day.year + y, m + 1
    return date(year, month, min(day.day, cal.monthrange(year, month)[1]))


def _tz(window: core.TimeWindow | None) -> str:
    return "UTC" if window is None else window.tz


def adverse(c: core.ObservableCandidate, nid: str, *, partial: bool = False) -> Any:
    """The condition that the observable shows its event's adverse state (in full, or also in part)."""
    o = c.observable
    if isinstance(c.occurs, Number):
        assert isinstance(c.absent, Number)
        op: Literal[">=", "<="] = ">=" if c.occurs.value > c.absent.value else "<="
        threshold = c.partial if partial and c.partial is not None else c.occurs
        return core.Cmp(id=nid, observable=o.id, op=op, value=threshold)
    full = core.Cmp(id=nid, observable=o.id, op="==", value=c.occurs)
    if not partial or c.partial is None:
        return full
    part = core.Cmp(id=f"{nid}_partial", observable=o.id, op="==", value=c.partial)
    return core.Or(id=f"{nid}_full_or_partial", args=[full, part])


def _observable(c: core.ObservableCandidate, retroactive: bool) -> core.Observable:
    if not retroactive:
        return c.observable
    fields = {**c.observable.model_dump(mode="json"), "revision_policy": "as_of_expiry", "final_deadline": None,
              "final_fallback": None}  # fmt: skip
    return core.Observable.model_validate_json(core.canonical_json(fields))


def _procedure(observables: Sequence[core.Observable], *, early: bool = False) -> core.ResolutionProcedure:
    sources = sorted({s.id: s for o in observables for s in o.sources}.values(), key=lambda s: (s.rank, s.id))
    return core.ResolutionProcedure(
        precedence=[s.id for s in sources], conflict_policy=core.ConflictPolicy(procedure="precedence_first_available"),
        on_unknown_at_expiry="no", early_expiration="when_true" if early else "none",
        early_expiration_at="first_true" if early else None, void_transfer=core.VoidTransfer(returns="nothing"),
    )  # fmt: skip


def _timing(open: datetime, end: datetime, tz: str) -> core.Timing:
    return core.Timing(open=open, close=end, expiration=end, observation_cutoff=end,
                       settlement_lag_days=SETTLEMENT_LAG_DAYS, payment_lag_days=0, lag_days="calendar",
                       holiday_calendar=None, tz=tz)  # fmt: skip


def _ref_id(*parts: str) -> str:
    return ".".join(parts)


def digital(base: Based, window: core.TimeWindow | None, open: datetime, currency: str, *, early: bool = False,
            retroactive: bool = False, partial: bool = False) -> core.ContractIR | str:  # fmt: skip
    """Pays one unit if the observable shows the adverse state on a day by the end of the event's window; at expiration
    plus the settlement lag, or, expiring early, from the first day it shows it. A reason where it cannot be written."""
    _, c = base
    tz = _tz(window)
    if window is None or window.end is None:
        return f"the event {c.event}'s window has no end, so no deadline by which a digital over {c.observable.id} pays"
    start = first_midnight(open, tz)
    if window.end <= start:
        return f"the event {c.event}'s window ends before the contract over {c.observable.id} first observes"
    o = _observable(c, retroactive)
    cond = adverse(c, "adverse", partial=partial)
    tag = "digital" + (".early" if early else "") + (".retroactive" if retroactive else "") + (".partial" if partial else "")
    return core.ContractIR(
        id=_ref_id(tag, c.observable.id), version=1, observables=[o],
        trigger=core.By(id="by_window_end", expr=cond, since=start, since_closed=True, deadline=window.end,
                        closed=window.end_closed),
        payout=core.Payout(kind="binary", unit_value=UNIT, currency=currency, rounding_quantum=QUANTUM,
                           rounding_mode="half_even", payment="at_settlement"),
        timing=_timing(open, window.end, tz), resolution=_procedure([o], early=early),
    )  # fmt: skip


def _periods(start: datetime, months: int, tz: str) -> list[core.TimeWindow]:
    day = start.astimezone(ZoneInfo(tz)).date()
    edges = [_midnight(add_months(day, i), tz) for i in range(months + 1)]
    return [core.TimeWindow(start=a, end=b, start_closed=True, end_closed=False, tz=tz) for a, b in zip(edges, edges[1:], strict=False)]


def strip(bases: Sequence[Based], tz: str, open: datetime, currency: str, *, months: int, per_period: bool = True,
          retroactive: bool = False, partial: bool = False, joined: str | None = None) -> core.ContractIR:  # fmt: skip
    """Pays one unit per day the adverse state holds within a term of `months` months from the first midnight the
    contract observes: per calendar-month period after the period ends, or once at expiry. Over two observables, `joined`
    says whether a day counts when both states hold ("and") or either does ("or")."""
    start = first_midnight(open, tz)
    periods = _periods(start, months, tz)
    end = periods[-1].end
    assert end is not None
    coverage = core.TimeWindow(start=start, end=end, start_closed=True, end_closed=False, tz=tz)
    obs = [_observable(c, retroactive) for _, c in bases]
    if joined is None:
        ((_, c),) = bases
        cond = adverse(c, "adverse", partial=partial)
        tag = f"strip{months}m" if per_period else f"count{months}m"
        name = c.observable.id
    else:
        (_, a), (_, b) = bases
        parts = [adverse(a, "adverse_a", partial=partial), adverse(b, "adverse_b", partial=partial)]
        cond = (core.And if joined == "and" else core.Or)(id=f"{'both' if joined == 'and' else 'either'}", args=parts)
        tag = f"compound.{joined}.strip{months}m"
        name = f"{a.observable.id}+{b.observable.id}"
    tag += (".retroactive" if retroactive else "") + (".partial" if partial else "")
    count = core.CountTrue(id="days_adverse", expr=cond, days=coverage)
    return core.ContractIR(
        id=_ref_id(tag, name), version=1, observables=obs,
        trigger=core.By(id="adverse_in_term", expr=cond, since=start, since_closed=True, deadline=end, closed=False),
        payout=core.Payout(kind="per_unit", unit_value=UNIT, quantity=count, currency=currency, rounding_quantum=QUANTUM,
                           rounding_mode="half_even", payment="per_period" if per_period else "at_settlement",
                           periods=periods if per_period else []),
        timing=_timing(open, end, tz), resolution=_procedure(obs),
    )  # fmt: skip


def _candidate(ir: core.ContractIR, family: str, template: core.Ref, variant: str, description: str,
               bases: Sequence[Based]) -> core.ConstructedCandidate:  # fmt: skip
    return core.ConstructedCandidate(
        ir=ir, family=family, template=template, variant=variant, description=description,  # type: ignore[arg-type]
        observables=[p for p, _ in bases], bases=[c for _, c in bases], events=sorted({c.event for _, c in bases}),
    )  # fmt: skip


def generate(observables: Sequence[Based], windows: Sequence[tuple[str, core.TimeWindow | None]], open: datetime,
             currency: str) -> core.CandidateSet:  # fmt: skip
    """Every candidate the templates give over the observables that passed their screens, and what is not generated,
    with why."""
    win = dict(windows)
    out: list[core.ConstructedCandidate] = []
    skipped: list[str] = []
    for base in observables:
        pid, c = base
        tz = _tz(win.get(c.event))
        o = c.observable.id
        stated = c.observable.revision_policy
        for early in (False, True):
            ir = digital(base, win.get(c.event), open, currency, early=early)
            when = "from the first day it shows it (expiring early)" if early else "at the window's end"
            if isinstance(ir, str):
                skipped.append(f"digital{' (early)' if early else ''}: {ir}")
                continue
            out.append(_candidate(ir, "digital", DIGITAL, "early" if early else "at_expiration",
                                  f"pays one unit if {o} shows {c.event}'s adverse state by the end of its window, paid "
                                  f"{when}", [base]))  # fmt: skip
        for months in TERMS:
            out.append(_candidate(strip([base], tz, open, currency, months=months), "time_in_state", STRIP,
                                  f"term_{months}m", f"pays one unit per day {o} shows {c.event}'s adverse state, over "
                                  f"{months} months, after each month", [base]))  # fmt: skip
        out.append(_candidate(strip([base], tz, open, currency, months=BASE_TERM, per_period=False), "time_in_state",
                              COUNT, f"count_{BASE_TERM}m_at_expiry", f"pays one unit per day {o} shows {c.event}'s "
                              f"adverse state over {BASE_TERM} months, once at expiry", [base]))  # fmt: skip
        if stated == "as_of_expiry":
            skipped.append(f"retroactive variants over {o}: its stated revision policy already counts later records")
        else:
            out.append(_candidate(strip([base], tz, open, currency, months=BASE_TERM, retroactive=True), "time_in_state",
                                  STRIP, f"term_{BASE_TERM}m_retroactive", f"as the {BASE_TERM}-month strip over {o}, "
                                  "counting a day a later record shows in the adverse state, until expiry", [base]))  # fmt: skip
            ir = digital(base, win.get(c.event), open, currency, retroactive=True)
            if not isinstance(ir, str):
                out.append(_candidate(ir, "digital", DIGITAL, "at_expiration_retroactive", f"as the digital over {o} at "
                                      "the window's end, counting a later record of the adverse state, until expiry",
                                      [base]))  # fmt: skip
        if c.partial is None:
            skipped.append(f"partial-relief variants over {o}: the observable states no partial value")
        else:
            out.append(_candidate(strip([base], tz, open, currency, months=BASE_TERM, partial=True), "time_in_state",
                                  STRIP, f"term_{BASE_TERM}m_partial", f"as the {BASE_TERM}-month strip over {o}, counting "
                                  "a day the state holds in part", [base]))  # fmt: skip
            ir = digital(base, win.get(c.event), open, currency, partial=True)
            if not isinstance(ir, str):
                out.append(_candidate(ir, "digital", DIGITAL, "at_expiration_partial", f"as the digital over {o}, paying "
                                      "also where the state shows in part", [base]))  # fmt: skip
    pairs = [(a, b) for i, a in enumerate(observables) for b in observables[i + 1 :] if a[1].event != b[1].event]
    if not pairs:
        events = sorted({c.event for _, c in observables})
        skipped.append(f"compound: needs passing observables of two events; they are of {len(events)} event(s)"
                       f"{' (' + ', '.join(events) + ')' if events else ''}")  # fmt: skip
    for a, b in pairs:
        if a[1].observable.id == b[1].observable.id:
            skipped.append(f"compound over {a[1].observable.id}: one observable cannot report two events here")
            continue
        tz = _tz(win.get(a[1].event))
        if _tz(win.get(b[1].event)) != tz:
            skipped.append(f"compound over {a[1].observable.id} and {b[1].observable.id}: their events' windows are in "
                           "different zones, and a day is counted in one")  # fmt: skip
            continue
        for joined in ("and", "or"):
            which = "both" if joined == "and" else "either"
            out.append(_candidate(strip([a, b], tz, open, currency, months=BASE_TERM, joined=joined), "compound",
                                  COMPOUND, f"{joined}_{BASE_TERM}m", f"pays one unit per day {which} of "
                                  f"{a[1].observable.id} and {b[1].observable.id} shows its event's adverse state, over "
                                  f"{BASE_TERM} months, after each month", [a, b]))  # fmt: skip
    skipped.append("ladder: needs an observable whose level the scenario model takes at two or more adverse values; each "
                   "candidate observable publishes one value while its event holds, so a ladder's rungs would pay "
                   "together, as one digital")  # fmt: skip
    return core.CandidateSet(candidates=out, not_generated=skipped)
