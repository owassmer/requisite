"""Construction's computations as registered nodes of the case's computation graph (spec §5: "settlement is itself a
computation node"; I7: payoffs, cash flows, aggregation and optimization are computed by registered nodes), so a change
upstream recomputes exactly what depends on it and the diff shows it (spec §15 beat 5; requisite/compute/runtime.py).

What each node computes, in the graph's order:
- the scenario model, from the events' ProbabilityEstimates (independent events, or a committed event tree), and the
  scenarios' assignments of the events alone, so a changed forecast reaches prices and solves, never settlement;
- when each event happens in each endpoint model (core.EventDay);
- the candidate contracts, from the screened observables by the library's structure templates (requisite/library/
  structures.py), each accepted by the evaluator;
- each candidate's settlement in each scenario of each endpoint model by the single settlement function (I8), called
  here inside the node: evaluated at the contract's final instant on the daily observations the scenario realizes;
- the business's dated cash in both endpoint models (from requisite/library/nodes/cash.py's paths);
- each candidate's receipts on the business's grid, its price, the classes of candidates the solve cannot tell apart;
- the liquidity program under each endpoint model (requisite/optimize), its cash recomputed exactly, its reduced costs
  beside their re-solves' measured improvements, its scenario table, uncovered scenarios and timing gaps;
- whether the adverse solve is the worst case over the intervals, the chosen design repriced over the loading range,
  and its robustness to each forecast's band, one at a time.

Nothing here reads committed state, a provider, an agent or a clock (requisite/compute/purity.py): every input arrives as
a vertex's value.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from decimal import Context, Decimal, Inexact, InvalidOperation, Overflow, localcontext
from fractions import Fraction
from typing import Any
from zoneinfo import ZoneInfo

from requisite import core
from requisite.compute.registry import node
from requisite.contract.evaluate import IRError, accept, evaluate, final_instant
from requisite.core.construction import Model
from requisite.core.evidence import synthetic_artifact_id
from requisite.library import structures
from requisite.library.nodes.cash import Change, event_day, horizon_days, local_day
from requisite.library.nodes.materiality import cash_start
from requisite.optimize import (
    Candidate,
    CashEvaluation,
    LiquidityProblem,
    LiquiditySolution,
    reduced_cost,
    try_candidate,
)
from requisite.optimize import evaluate as cash_of
from requisite.optimize import solve as solve_lp

EXACT = Context(prec=200, traps=[Inexact, InvalidOperation, Overflow])
CENT = Fraction(1, 100)
PRICE_QUANTUM = Fraction(1, 10**6)
DAY_COUNT = 365
STEPS = 4  # the loading range is repriced at its ends and at the points that split it into this many equal steps

Assignment = tuple[str, tuple[tuple[str, bool], ...]]


# ---------------------------------------------------------------- exact money


def _decimal(n: int, places: int) -> Decimal:
    """n x 10^-places, exactly (Decimal's scaleb rounds to the context's precision; a string never does)."""
    return Decimal(f"{n}E-{places}")


def floor_to(x: Fraction, quantum: Fraction, places: int) -> Decimal:
    return _decimal(math.floor(x / quantum), places)


def ceil_to(x: Fraction, quantum: Fraction, places: int) -> Decimal:
    return _decimal(math.ceil(x / quantum), places)


def exact(x: Fraction) -> Decimal:
    """A rational whose denominator divides a power of ten, as the Decimal it is, exactly (money computed from Decimal
    amounts); any other is refused, never rounded."""
    k, den = 0, x.denominator
    while den % 10 == 0 or den % 2 == 0 or den % 5 == 0:
        if den % 10 == 0:
            den //= 10
        elif den % 2 == 0:
            den //= 2
        else:
            den //= 5
        k += 1
    if den != 1:
        raise ValueError(f"{x} has no finite decimal form")
    scaled = x * 10**k
    assert scaled.denominator == 1
    return _decimal(scaled.numerator, k)


def fq(x: Any) -> Fraction:
    if isinstance(x, bool) or isinstance(x, float):
        raise TypeError(f"an exact number, not {type(x).__name__}")
    return Fraction(x)


# ---------------------------------------------------------------- the scenario model (addendum A04 §4.1-4.2)


def scenario_id(assignment: Sequence[tuple[str, bool]]) -> str:
    return ",".join(f"{e}:{'occurs' if v else 'not'}" for e, v in assignment)


def independent_assignments(events: Sequence[str]) -> list[Assignment]:
    """Every combination of the events occurring or not, in a fixed order (each event's occurring first)."""
    out = []
    for values in itertools.product((True, False), repeat=len(events)):
        a = tuple(zip(events, values, strict=True))
        out.append((scenario_id(a), a))
    return out


INDEPENDENCE = ("the events occur independently of one another: no committed event tree states how they depend (task "
                "004-1), so each joint scenario's probability is the product of the events' own estimates, a "
                "hypothetical joint")  # fmt: skip


@node(id="construct.independent_model", version=1, inputs=["events", "estimates"])
def independent_model(events: Sequence[str], estimates: Sequence[core.ProbabilityEstimate]) -> core.ScenarioModel:
    """The joint of the events from each one's estimate of its occurring: one event's own model, or several joined as
    independent, named as the assumption it is (task 004-1)."""
    by = {pe.proposition: pe for pe in estimates}
    missing = [e for e in events if e not in by]
    if missing:
        raise ValueError(f"no probability estimate for {missing}: the scenario model is not solved without them")
    for pe in by.values():
        if pe.branch_value != "true" or pe.conditioning:
            raise ValueError(f"{pe.id} is not the unconditional probability that {pe.proposition} occurs")
        if pe.meaning == "pricing":
            raise ValueError(f"{pe.id} is a pricing quantity, not a real-world probability (I9; A06 §6.7)")
    scenarios = []
    for sid, a in independent_assignments(events):
        p = Fraction(1)
        factors = []
        for e, occurs in a:
            pe = by[e]
            p *= pe.value if occurs else 1 - pe.value
            factors.append((pe.id, not occurs))
        scenarios.append(core.Scenario(id=sid, occurs=dict(a), probability=p, factors=factors))
    single = len(events) == 1
    meaning = by[events[0]].meaning if single else "hypothetical"
    return core.ScenarioModel(
        kind="single_event" if single else "independent_events", events=list(events),
        estimates=[by[e].id for e in events], meaning=meaning,  # type: ignore[arg-type]
        assumption=None if single else INDEPENDENCE, scenarios=scenarios,
    )  # fmt: skip


def tree_paths(tree: core.EventTree) -> list[list[core.Branch]]:
    out: list[list[core.Branch]] = []

    def walk(bp: core.BranchPoint, path: list[core.Branch]) -> None:
        for b in bp.branches:
            if b.then is None:
                out.append([*path, b])
            else:
                walk(b.then, [*path, b])

    walk(tree.root, [])
    return out


def tree_assignments(events: Sequence[str], tree: core.EventTree) -> list[Assignment]:
    """Each path of the tree as an assignment of the events, which every path must make whole (A04 §4.2)."""
    out = []
    for path in tree_paths(tree):
        values: dict[str, bool] = {}
        for b in path:
            if b.value not in ("true", "false") or b.proposition in values:
                raise ValueError(f"tree {tree.id}: a path's branches set each event true or false once, not {b.value!r}")
            values[b.proposition] = b.value == "true"
        if set(values) != set(events):
            raise ValueError(f"tree {tree.id}: a path assigns {sorted(values)}, not every event of the case {list(events)}")
        a = tuple((e, values[e]) for e in events)
        out.append((scenario_id(a), a))
    return out


@node(id="construct.tree_model", version=1, inputs=["events", "tree", "estimates"])
def tree_model(events: Sequence[str], tree: core.EventTree, estimates: Sequence[core.ProbabilityEstimate]
               ) -> core.ScenarioModel:  # fmt: skip
    """A committed event tree's paths as the joint: each scenario's probability the product of its branches' conditional
    probabilities (A04 §4.2), each branch point a partition (its probabilities sum to exactly one, never normalized). A
    binary branch point's second branch enters as the complement of its first, which its estimate must equal, so varying
    the first alone keeps the partition (A06 §6.7)."""
    by = {pe.id: pe for pe in estimates}

    def check(bp: core.BranchPoint) -> None:
        missing = [b.probability for b in bp.branches if b.probability not in by]
        if missing:
            raise ValueError(f"tree {tree.id}: no estimate {missing}")
        core.base.exact_sum_is_one([by[b.probability].value for b in bp.branches], f"branch point {bp.id}")
        for b in bp.branches:
            if b.then is not None:
                check(b.then)

    check(tree.root)
    first: dict[str, tuple[str, str]] = {}  # a binary point's second branch estimate -> (its first's, the point)

    def pair(bp: core.BranchPoint) -> None:
        if len(bp.branches) == 2:
            a, b = bp.branches
            first[b.probability] = (a.probability, bp.id)
        for x in bp.branches:
            if x.then is not None:
                pair(x.then)

    pair(tree.root)
    scenarios = []
    for path, (sid, a) in zip(tree_paths(tree), tree_assignments(events, tree), strict=True):
        p = Fraction(1)
        factors = []
        for b in path:
            p *= by[b.probability].value
            if b.probability in first:
                f = first[b.probability][0]
                if by[b.probability].value != 1 - by[f].value:
                    raise ValueError(f"{b.probability} is not the complement of {f}")
                factors.append((f, True))
            else:
                factors.append((b.probability, False))
        scenarios.append(core.Scenario(id=sid, occurs=dict(a), probability=p, factors=factors))
    used = sorted({f for s in scenarios for f, _ in s.factors})
    meanings = {by[x].meaning for x in used}
    return core.ScenarioModel(kind="event_tree", events=list(events), estimates=used, tree=core.Ref(id=tree.id,
                              version=tree.version), meaning="physical" if meanings == {"physical"} else "hypothetical",
                              assumption=None, scenarios=scenarios)  # fmt: skip


@node(id="construct.assignments", version=1, inputs=["model"])
def assignments(model: core.ScenarioModel) -> tuple[Assignment, ...]:
    """The scenarios as assignments of the events, without their probabilities: what settlement reads."""
    return tuple((s.id, tuple((e, bool(s.occurs[e])) for e in model.events)) for s in model.scenarios)


def probabilities(model: core.ScenarioModel, values: Mapping[str, Fraction]) -> list[Fraction]:
    """Each scenario's probability from its factors at these estimate values."""
    out = []
    for s in model.scenarios:
        p = Fraction(1)
        for f, complement in s.factors:
            p *= 1 - values[f] if complement else values[f]
        out.append(p)
    return out


@node(id="construct.event_days", version=1, inputs=["as_of", "windows"])
def event_days(as_of: datetime, windows: Sequence[tuple[str, core.TimeWindow | None]]) -> tuple[core.EventDay, ...]:
    """When each event happens in each endpoint model (core.EventDay), from as_of and its window."""
    out = []
    for e, w in windows:
        tz, first, last = event_day(as_of, w)
        out.append(core.EventDay(event=e, tz=tz, earliest=first, latest=last))
    return tuple(out)


# ---------------------------------------------------------------- candidates and their settlement (spec §8.2-8.3; I8)


REFUSED = "the evaluator refuses it"


@node(id="construct.candidates", version=1, inputs=["observables", "windows", "open", "currency"])
def candidates(observables: Sequence[tuple[str, core.ObservableCandidate]],
               windows: Sequence[tuple[str, core.TimeWindow | None]], open: datetime, currency: str) -> core.CandidateSet:  # fmt: skip
    """The structure templates' candidates over the screened observables, each an IR the evaluator accepts (spec §8.2);
    one it refuses is listed as not generated, with the evaluator's reason."""
    made = structures.generate(list(observables), list(windows), open, currency)
    kept, refused = [], list(made.not_generated)
    for c in made.candidates:
        try:
            accept(c.ir)
        except IRError as e:
            refused.append(f"{c.ir.id}: {REFUSED}: {e}")
            continue
        kept.append(c)
    return core.CandidateSet(candidates=kept, not_generated=refused)


def realize(c: core.ConstructedCandidate, occurs: Mapping[str, bool], day_of: Mapping[str, date | None]
            ) -> list[core.SyntheticObservation]:  # fmt: skip
    """The observations a scenario gives a candidate's observables: a value for every day the contract observes, at the
    day's midnight in its zone, from the observable's highest-ranked source the contract consults, published that
    instant and final: the event's adverse value from the day the event happens in this model where it occurs in the
    scenario, its absent value otherwise. They are a simulation's, so synthetic (I4). Ours: the observable is taken to
    report its event as the scenario has it, so a basis gap between the two is outside the scenario model."""
    ir = c.ir
    tz = ir.timing.tz
    first = structures.first_midnight(ir.timing.open, tz)
    cutoff = ir.timing.observation_cutoff
    out = []
    for base in c.bases:
        o = next(x for x in ir.observables if x.id == base.observable.id)
        own = {s.id for s in o.sources}
        source = next(s for s in ir.resolution.precedence if s in own)
        when = day_of[base.event] if occurs[base.event] else None
        day = first.astimezone(structures.ZoneInfo(tz)).date()
        while (at := structures._midnight(day, tz)) <= cutoff:
            holds = when is not None and day >= when
            value = base.occurs if holds else base.absent
            key = core.canonical_json(["construction scenario", o.id, source, at.isoformat(), value.model_dump(mode="json")])
            out.append(core.SyntheticObservation(observable=o.id, source=source, at=at, value=value, published_at=at,
                                                 final=True, artifact=synthetic_artifact_id(key.encode())))  # fmt: skip
            day += timedelta(days=1)
    return out


def _settle(model: Model, cs: core.CandidateSet, assigned: Sequence[Assignment],
            days: Sequence[core.EventDay]) -> core.Settlement:  # fmt: skip
    day_of = {d.event: (d.earliest if model == "adverse" else d.latest) for d in days}
    settled, excluded = [], {}
    for c in cs.candidates:
        mine = []
        for sid, a in assigned:
            ev = evaluate(c.ir, realize(c, dict(a), day_of), {}, at=final_instant(c.ir))
            why = core.payment_why(ev, c.ir)
            if why is not None:
                excluded[c.ir.id] = f"in scenario {sid} {why}"
                break
            mine.append(core.Settled(candidate=c.ir.id, scenario=sid, evaluation=ev))
        else:
            settled += mine
    return core.Settlement(model=model, event_days=day_of, settled=settled, excluded=excluded)


@node(id="construct.settle_adverse", version=1, inputs=["candidates", "assignments", "event_days"])
def settle_adverse(candidates: core.CandidateSet, assignments: Sequence[Assignment],
                   event_days: Sequence[core.EventDay]) -> core.Settlement:  # fmt: skip
    """Every candidate's settlement in every scenario, each event on its earliest day (the adverse model)."""
    return _settle("adverse", candidates, assignments, event_days)


@node(id="construct.settle_favourable", version=1, inputs=["candidates", "assignments", "event_days"])
def settle_favourable(candidates: core.CandidateSet, assignments: Sequence[Assignment],
                      event_days: Sequence[core.EventDay]) -> core.Settlement:  # fmt: skip
    """Every candidate's settlement in every scenario, each event on its latest day (the favourable model)."""
    return _settle("favourable", candidates, assignments, event_days)


def transfers(s: core.Settlement) -> dict[str, dict[str, tuple[core.CashTransfer, ...]]]:
    """Each candidate's transfers by scenario."""
    out: dict[str, dict[str, tuple[core.CashTransfer, ...]]] = {}
    for x in s.settled:
        out.setdefault(x.candidate, {})[x.scenario] = tuple(x.evaluation.transfers)
    return out


# ---------------------------------------------------------------- the business's dated cash (addendum A06 §6.1)

CONVENTION = ("by calendar day in the horizon's zone: each day's operating flows and modeled loss, and each transfer paid "
              "that day, count on that day; every day's flow is rounded to the cent toward less cash (operating flow "
              "down, loss up), so the computed cash never exceeds the exact model's; the premium is paid on the "
              "decision day, or the horizon's first day where the decision comes before it")  # fmt: skip


def _start_in(as_of: datetime, window: core.TimeWindow | None, lag: Any, terms_days: Any, horizon: core.TimeWindow,
              latest: bool) -> tuple[date, date | None]:  # fmt: skip
    """When an exposure's change first moves cash in one endpoint model: from the day that model puts its event on (its
    earliest day in the adverse model, its latest in the favourable: core.EventDay), plus the exposure's own lag and
    payment terms (deep.cash_start over that one day). So settlement and cash read one history of the event in each
    model, and only the business model's intervals (the lag, the share) take their ends (definition review, November 2:
    a cash-raising share once started on the event's latest day while settlement read its earliest). An event the
    favourable model puts after its window's end (a window with none) starts nothing within the horizon."""
    tz, first, last = event_day(as_of, window)
    day = last if latest else first
    assert horizon.end is not None  # construction's horizon is bounded (deep.horizon_days)
    if day is None:
        after = local_day(horizon.end, horizon.tz) + timedelta(days=1)
        return after, after
    at = datetime.combine(day, time(0), ZoneInfo(tz))
    one = core.TimeWindow(start=at, end=at, start_closed=True, end_closed=True, tz=tz)
    return cash_start(as_of, one, lag, terms_days, horizon)


@node(id="construct.start_adverse", version=1, inputs=["as_of", "window", "lag", "terms_days", "horizon"])
def start_adverse(as_of: datetime, window: core.TimeWindow | None, lag: Any, terms_days: Any,
                  horizon: core.TimeWindow) -> tuple[date, date | None]:  # fmt: skip
    return _start_in(as_of, window, lag, terms_days, horizon, latest=False)


@node(id="construct.start_favourable", version=1, inputs=["as_of", "window", "lag", "terms_days", "horizon"])
def start_favourable(as_of: datetime, window: core.TimeWindow | None, lag: Any, terms_days: Any,
                     horizon: core.TimeWindow) -> tuple[date, date | None]:  # fmt: skip
    return _start_in(as_of, window, lag, terms_days, horizon, latest=True)


@node(id="construct.collect_first", version=1, inputs=["item"])
def collect_first(item: Any) -> tuple[Any, ...]:
    return (item,)


@node(id="construct.collect", version=1, inputs=["items", "item"])
def collect(items: Sequence[Any], item: Any) -> tuple[Any, ...]:
    return (*items, item)


@node(id="construct.business", version=1,
      inputs=["days", "operating", "liquidity", "reserve", "adverse", "favourable", "as_of", "horizon", "assignments",
              "currency"])  # fmt: skip
def business(days: tuple[date, date], operating: Sequence[tuple[date, Fraction]], liquidity: Fraction,
             reserve: Sequence[tuple[date, Fraction]], adverse: Sequence[Sequence[Change]],
             favourable: Sequence[Sequence[Change]], as_of: datetime, horizon: core.TimeWindow,
             assignments: Sequence[Assignment], currency: str) -> core.BusinessCash:  # fmt: skip
    """The business on construction's grid: every day of the horizon; liquidity at its start; the operating flow before
    the modeled loss; the floor; and each scenario's loss in each endpoint model: minus the change at its least end on
    the adverse model's change path (each event on its earliest day: _start_in), or at its greatest on the favourable
    model's (each event on its latest day)."""
    grid = horizon_days(days)
    if [d for d, _ in operating] != grid or [d for d, _ in reserve] != grid:
        raise ValueError("operating flow and reserve are stated over the horizon's days")
    if len(adverse) != len(assignments) or len(favourable) != len(assignments):
        raise ValueError("one change path per scenario in each endpoint model")
    premium = max(grid[0], local_day(as_of, horizon.tz))
    if premium > grid[-1]:
        raise ValueError("the decision comes after the horizon ends: there is no day within it to pay a premium")
    losses: dict[str, list[list[Decimal]]] = {"adverse": [], "favourable": []}
    for (sid, _), a_path, f_path in zip(assignments, adverse, favourable, strict=True):
        for model, index, path in (("adverse", 1, a_path), ("favourable", 2, f_path)):
            if [d for d, _, _ in path] != grid:
                raise ValueError("a scenario's change is stated over the horizon's days")
            row = []
            for when, least, greatest in path:
                end = least if index == 1 else greatest
                if end is None:
                    raise ValueError(f"in scenario {sid} the {model} change on {when} has no bound")
                row.append(ceil_to(-end, CENT, 2))
            losses[model].append(row)
    return core.BusinessCash(
        tz=horizon.tz, currency=currency, grid=grid, premium_day=premium, initial=exact(fq(liquidity)),
        operating=[floor_to(x, CENT, 2) for _, x in operating], reserve=[ceil_to(x, CENT, 2) for _, x in reserve],
        scenarios=[sid for sid, _ in assignments],
        models=[core.EndpointLoss(model=m, loss=losses[m]) for m in core.construction.MODELS], convention=CONVENTION,
    )  # fmt: skip


# ---------------------------------------------------------------- receipts on the grid


@node(id="construct.receipts", version=1, inputs=["settlement", "business"])
def receipts(settlement: core.Settlement, business: core.BusinessCash) -> core.Receipts:
    """Each settled candidate's per-unit transfers on the business's grid, summed per paid day in the grid's zone; a
    transfer paid before the grid's first day is in hand on it, and one paid after its last day is summed apart."""
    out = []
    for cid, by_scenario in transfers(settlement).items():
        rows, after = [], []
        for sid in business.scenarios:
            row, late = core.construction.place(by_scenario[sid], business.grid, business.tz, business.currency)
            rows.append(row)
            after.append(late)
        out.append(core.CandidateReceipts(candidate=cid, rows=rows, after_horizon=after))
    return core.Receipts(model=settlement.model, scenarios=list(business.scenarios), candidates=out)


# ---------------------------------------------------------------- prices (spec §8.3; I9; A06 §6.7)


def discount(day: date, premium: date, rate: Fraction) -> Fraction:
    """The factor taking a transfer paid on `day` to the premium day: simple interest at `rate` a year, actual/365; a
    transfer before the premium day is carried forward to it the same way."""
    n = (day - premium).days
    growth = 1 + rate * Fraction(abs(n), DAY_COUNT)
    if growth <= 0:
        raise ValueError(f"simple interest at {rate} a year gives no positive discount factor over {n} days")
    return 1 / growth if n >= 0 else growth


def expected_payout(by_scenario: Mapping[str, Sequence[core.CashTransfer]], probs: Mapping[str, Fraction],
                    premium: date, rate: Fraction, tz: str) -> Fraction:  # fmt: skip
    """The expected discounted payout per unit under the scenario probabilities, from the evaluator's dated transfers
    (A06 §6.7: expected payout, not one event probability times a payout scalar)."""
    return sum((probs[s] * sum((Fraction(t.amount) * discount(local_day(t.at, tz), premium, rate) for t in ts),
                               Fraction(0)) for s, ts in by_scenario.items()), Fraction(0))  # fmt: skip


def indicative(e: Fraction, loading: Fraction) -> Decimal:
    return core.construction.indicative_price(e, loading)


def same_predicate(a: core.ContractIR, b: core.ContractIR) -> bool:
    """Whether two IRs settle alike: every term but their names and published examples (spec §10: a quote prices a
    candidate only where it settles on the same predicate)."""

    def terms(ir: core.ContractIR) -> str:
        d = ir.model_dump(mode="json")
        for k in ("id", "version", "examples"):
            d.pop(k, None)
        return core.canonical_json(d)

    return terms(a) == terms(b)


Listed = tuple[str, core.ContractIR, core.Quote | None]


@node(id="construct.prices", version=1, inputs=["settlement", "candidates", "model", "business", "parameters", "listed"])
def prices(settlement: core.Settlement, candidates: core.CandidateSet, model: core.ScenarioModel,
           business: core.BusinessCash, parameters: core.ConstructionParameters, listed: Sequence[Listed]) -> core.Prices:  # fmt: skip
    """Each settled candidate's price per unit in this model: the ask of a listed market settling on the same predicate
    where the case holds one with a quote, otherwise indicative (spec §8.3; I9)."""
    probs = {s.id: s.probability for s in model.scenarios}
    cands = {c.ir.id: c.ir for c in candidates.candidates}
    out = []
    unpriced: dict[str, str] = {}
    for cid, by in transfers(settlement).items():
        e = expected_payout(by, probs, business.premium_day, parameters.discount_rate, business.tz)
        same = [(m, q) for m, ir, q in listed if same_predicate(ir, cands[cid])]
        offered = [(m, q) for m, q in same if q is not None and q.yes_ask > 0 and q.yes_ask_size > 0]
        if offered and offered[0][1] is not None and offered[0][1].fee is not None:
            m, q = offered[0]
            assert q is not None and q.fee is not None
            with localcontext(EXACT):  # exactly: a fee is never rounded away (spec §10)
                quoted = q.yes_ask + q.fee
            out.append(core.Price(candidate=cid, pricing="quoted", price=quoted, expected_payout=None,
                                  loading=None, quote=q,
                                  convention=f"quoted: the YES ask of listed market {m}, which settles on this "
                                             f"candidate's predicate, as served at {q.at} ({q.locator}), and the "
                                             f"exchange's fee {q.fee} to buy at it"))  # fmt: skip
            continue
        if offered:
            m, q = offered[0]
            assert q is not None
            unpriced[cid] = (f"listed market {m} settles on its predicate at a YES ask of {q.yes_ask} as served at "
                             f"{q.at}, but the exchange's fee to buy at it is not known here: the ask alone is no "
                             "acquisition cost (spec §10), and an indicative price would stand in for the market's "
                             "(spec §8.3)")  # fmt: skip
            continue
        else:
            listed_note = ("no listed market in the case settles on its predicate" if not same else
                           f"listed market(s) {', '.join(m for m, _ in same)} settle on its predicate but served no YES "
                           "ask for the decision")  # fmt: skip
        out.append(core.Price(
            candidate=cid, pricing="indicative", price=indicative(e, parameters.loading), expected_payout=e,
            loading=parameters.loading,
            convention=(f"indicative, not a quote: the expected payout under the scenario model ({model.kind}, "
                        f"{len(model.scenarios)} scenarios) from the evaluator's transfers, each discounted to the premium "
                        f"day {business.premium_day.isoformat()} by simple interest at {_show(parameters.discount_rate)} a "
                        f"year (actual/{DAY_COUNT}), times (1 + {_show(parameters.loading)}), rounded up to a millionth "
                        f"of the currency; {listed_note}"),
        ))  # fmt: skip
    held = len(listed)
    note = (f"the case holds {held} listed market(s); {sum(1 for p in out if p.pricing == 'quoted')} settle on a "
            "candidate's predicate with a quote") if held else "the case holds no listed contract, so no quote settles on a candidate's predicate"  # fmt: skip
    return core.Prices(model=settlement.model, prices=out, listed=note, unpriced=unpriced)


def _show(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


# ---------------------------------------------------------------- classes the solve cannot tell apart


def _vector(c: core.CandidateReceipts) -> dict[tuple[int, str], Fraction]:
    v = {(i, d.isoformat()): Fraction(a) for i, row in enumerate(c.rows) for d, a in row if a}
    v |= {(i, "after"): Fraction(a) for i, a in enumerate(c.after_horizon) if a}
    return v


def _value(p: core.Price) -> Fraction:
    """What a class compares besides the receipts: the price the program subtracts (core.PayoffClass)."""
    return Fraction(p.price)


@node(id="construct.classes", version=1, inputs=["receipts", "prices", "parameters"])
def classes(receipts: core.Receipts, prices: core.Prices, parameters: core.ConstructionParameters) -> core.PayoffClasses:
    """The candidates the solve cannot tell apart, grouped (core.PayoffClass): receipts on the grid and after it in every
    scenario a positive multiple of the representative's, and the price's basis (the expected payout, or the quote) the
    same multiple. A candidate paying nothing in any scenario, or nothing in expectation, is apart; with eligibility
    "listed", so is every candidate no listed market quotes (spec §8)."""
    price_of = {p.candidate: p for p in prices.prices}
    reps: list[tuple[str, dict[tuple[int, str], Fraction], Fraction, str]] = []
    members: dict[str, list[tuple[str, Fraction]]] = {}
    nothing, ineligible = [], []
    for c in receipts.candidates:
        if c.candidate in prices.unpriced:  # its acquisition cost is not determined: it enters no class (core.Prices)
            continue
        p = price_of[c.candidate]
        if parameters.eligibility == "listed" and p.pricing != "quoted":
            ineligible.append(c.candidate)
            continue
        vec, value = _vector(c), _value(p)
        if not vec or value == 0:
            nothing.append(c.candidate)
            continue
        for rid, rvec, rvalue, pricing in reps:
            if pricing != p.pricing or set(rvec) != set(vec):
                continue
            k = next(iter(vec))
            factor = vec[k] / rvec[k]
            if factor > 0 and all(vec[x] == factor * rvec[x] for x in vec) and value == factor * rvalue:
                members[rid].append((c.candidate, factor))
                break
        else:
            reps.append((c.candidate, vec, value, p.pricing))
            members[c.candidate] = [(c.candidate, Fraction(1))]
    return core.PayoffClasses(model=receipts.model,
                              classes=[core.PayoffClass(representative=r, members=members[r]) for r, _, _, _ in reps],
                              pays_nothing=nothing, ineligible=ineligible)  # fmt: skip


# ---------------------------------------------------------------- the liquidity program (addendum A06; C01 §7.2)

Decision = tuple[str, int, str, Any, Any]  # the decision spec's id, version, objective, alpha and tolerance


def tail(decision: Decision) -> Fraction:
    """The program's alpha: the decision's for a CVaR objective; zero for expected liquidity shortfall, whose CVaR at
    alpha zero is E[D] exactly (A06 §6.3, since D >= 0)."""
    _, _, objective, alpha, _ = decision
    if objective == "liquidity_cvar":
        return fq(alpha)
    if objective == "expected_liquidity_shortfall":
        return Fraction(0)
    raise ValueError(f"construction solves the liquidity program over dated cash (A06 §6.2); the decision's objective "
                     f"{objective!r} is not one of its two (liquidity_cvar, expected_liquidity_shortfall)")  # fmt: skip


def program(business: core.BusinessCash, receipts: core.Receipts, classes: core.PayoffClasses,
            price_of: Mapping[str, Decimal], probs: Sequence[Fraction], parameters: core.ConstructionParameters,
            decision: Decision, only: Sequence[str] | None = None) -> LiquidityProblem:  # fmt: skip
    """The liquidity program of one endpoint model (A06 §6.1-6.2): the business's grid, liquidity, operating flow, loss
    and floor, and one column per class (its representative's receipts and price), each capped at the units the budget
    buys (rounded down to a millionth), so no cap binds before the budget does."""
    m = core.construction.MODELS.index(receipts.model)
    index = {d: i for i, d in enumerate(business.grid)}
    rows = {c.candidate: c for c in receipts.candidates}
    cands = []
    for cl in classes.classes:
        rep = cl.representative
        if only is not None and rep not in only:
            continue
        price = price_of[rep]
        if price <= 0:
            continue
        dense = [[Decimal(0)] * len(business.grid) for _ in business.scenarios]
        for s, row in enumerate(rows[rep].rows):
            for d, a in row:
                dense[s][index[d]] += a
        cap = floor_to(Fraction(parameters.budget) / Fraction(price), PRICE_QUANTUM, 6)
        cands.append(Candidate(rep, price, business.premium_day, tuple(tuple(r) for r in dense), cap))
    return LiquidityProblem(
        dates=tuple(business.grid), probabilities=tuple(probs), initial_cash=business.initial,
        operating=tuple(tuple(business.operating) for _ in business.scenarios),
        loss=tuple(tuple(row) for row in business.models[m].loss),
        reserve=tuple(business.reserve), alpha=tail(decision), budget=parameters.budget, candidates=tuple(cands),
    )  # fmt: skip


def structure(sol: LiquiditySolution) -> list[str]:
    """The classes a solution buys: those whose premium exceeds its money tolerance (a smaller position is within the
    solve's own error)."""
    return [c.ref for c, q in zip(sol.problem.candidates, sol.quantities, strict=True) if c.price * q > sol.tolerance]


def restricted(problem: LiquidityProblem, keep: Sequence[str], epsilon: Decimal) -> LiquiditySolution:
    return solve_lp(LiquidityProblem(
        dates=problem.dates, probabilities=problem.probabilities, initial_cash=problem.initial_cash,
        operating=problem.operating, loss=problem.loss, reserve=problem.reserve, alpha=problem.alpha,
        budget=problem.budget, candidates=tuple(c for c in problem.candidates if c.ref in keep),
    ), epsilon)  # fmt: skip


def summary(ev: Any, refs: Sequence[str]) -> core.CashSummary:
    return core.CashSummary(
        quantities=dict(zip(refs, ev.quantities, strict=True)), premium=ev.premium, cash=ev.cash, deficit=ev.deficit,
        breach_tolerance=ev.breach_tolerance, breach_dates=ev.breach_dates, loss=ev.loss, receipts=ev.receipts,
        residual=ev.residual, cvar=ev.cvar, expected_liquidity_shortfall=ev.expected_liquidity_shortfall,
        breach_probability=ev.breach_probability,
    )  # fmt: skip


def gaps(problem: LiquidityProblem, hedged: CashEvaluation, receipts: core.Receipts,
         scenarios: Sequence[str]) -> list[core.TimingGap]:  # fmt: skip
    """Each scenario whose cash with the held portfolio falls short, beyond its breach tolerance, on days when the
    portfolio's later receipts (within the horizon or after it) would have covered the shortfall (A06 §6.6): the payout
    is enough, but it arrives too late."""
    after = {c.candidate: c.after_horizon for c in receipts.candidates}
    out = []
    held = [(c, q) for c, q in zip(problem.candidates, hedged.quantities, strict=True) if q]
    tol = hedged.breach_tolerance
    for s, sid in enumerate(scenarios):
        paid = [sum((Fraction(c.receipts[s][t]) * Fraction(q) for c, q in held), Fraction(0))
                for t in range(len(problem.dates))]  # fmt: skip
        later = sum((Fraction(after[c.ref][s]) * Fraction(q) for c, q in held), Fraction(0))
        found = []
        for t in range(len(problem.dates) - 1, -1, -1):
            short = Fraction(problem.reserve[t]) - Fraction(hedged.cash[s][t])
            if short > tol and later >= short:
                found.append((problem.dates[t], short, later))
            later += paid[t]
        if found:
            worst = max(found, key=lambda x: (x[1], x[0]))
            out.append(core.TimingGap(scenario=sid, first_day=min(f[0] for f in found), last_day=max(f[0] for f in found),
                                      days=len(found), worst_day=worst[0], shortfall=exact(worst[1]),
                                      later=exact(worst[2])))  # fmt: skip
    return out


@node(id="construct.solve", version=1,
      inputs=["business", "receipts", "prices", "classes", "candidates", "model", "parameters", "decision"])  # fmt: skip
def solve_model(business: core.BusinessCash, receipts: core.Receipts, prices: core.Prices,
                classes: core.PayoffClasses, candidates: core.CandidateSet, model: core.ScenarioModel,
                parameters: core.ConstructionParameters, decision: Decision) -> core.EndpointSolve:  # fmt: skip
    """The liquidity program under one endpoint model (A06 §6.2-6.4): the CVaR solve, then the least premium within its
    tolerance; the chosen portfolio's cash recomputed exactly and the unhedged business's beside it; each unbought
    class's reduced cost against the duals of the program without it, beside the measured improvement of re-solving with
    it (C01 §7.2: a reduced cost is a diagnostic, never shown alone); the scenario table, the scenarios left short and the
    timing gaps."""
    probs = [s.probability for s in model.scenarios]
    price_of = {p.candidate: p.price for p in prices.prices}
    problem = program(business, receipts, classes, price_of, probs, parameters, decision)
    sol = solve_lp(problem, parameters.epsilon)
    refs = [c.ref for c in problem.candidates]
    bought = structure(sol)
    reduced = []
    for cand in problem.candidates:
        if cand.ref in bought:
            continue
        master = restricted(problem, [r for r in refs if r != cand.ref], parameters.epsilon)
        trial = try_candidate(master, cand)
        reduced.append(core.ReducedCost(candidate=cand.ref, value=reduced_cost(master, cand),
                                        measured_improvement=trial.improvement))  # fmt: skip
    pricing = {p.candidate: p for p in prices.prices}
    irs = {c.ir.id: c.ir for c in candidates.candidates}
    dec_id, dec_version, objective, _, _ = decision
    res = sol
    result = core.OptimizationResult(
        decision=core.Ref(id=f"construct.{dec_id}", version=dec_version), objective=objective,  # type: ignore[arg-type]
        alpha=tail(decision),
        scenarios_hash=model.content_hash(), business_hash=business.content_hash(), receipts_hash=receipts.content_hash(),
        probabilities_hash=core.probabilities_hash(probs),
        candidates=[core.PricedCandidate(ref=c.ref, ir=irs[c.ref].ref, price=c.price, pricing=pricing[c.ref].pricing,
                                         price_source=pricing[c.ref].convention, cap=c.cap) for c in problem.candidates],
        dates=list(problem.dates), solver=res.solver, epsilon=res.epsilon, scale=res.scale,
        primary_status=res.primary_status, secondary_status=res.secondary_status,
        primary_objective=res.primary_objective, secondary_objective=res.secondary_objective,
        recomputed_objective=res.hedged.cvar,
        primary_residuals={k: float(v) for k, v in vars(res.primary_residuals).items() if v is not None},
        secondary_residuals={k: float(v) for k, v in vars(res.secondary_residuals).items() if v is not None},
        hedged=summary(res.hedged, refs), unhedged=summary(res.unhedged, refs), cash_duals=res.cash_duals,
        budget_dual=res.budget_dual, reduced_costs=reduced,
    )  # fmt: skip
    table = []
    for s, sid in enumerate(business.scenarios):
        h, u = res.hedged, res.unhedged
        table.append(core.ScenarioRow(
            scenario=sid, probability=probs[s], loss=h.loss[s], receipts=h.receipts[s], premium=h.premium,
            residual=h.residual[s], deficit_unhedged=u.deficit[s], deficit_hedged=h.deficit[s],
            first_breach_unhedged=u.breach_dates[s][0] if u.breach_dates[s] else None,
            first_breach_hedged=h.breach_dates[s][0] if h.breach_dates[s] else None,
        ))  # fmt: skip
    uncovered = [sid for s, sid in enumerate(business.scenarios) if res.hedged.breach_dates[s]]
    return core.EndpointSolve(model=receipts.model, result=result, structure=bought, table=table, uncovered=uncovered,
                              timing_gaps=gaps(problem, res.hedged, receipts, business.scenarios))  # fmt: skip


@node(id="construct.worst_case", version=1, inputs=["adverse", "favourable"])
def worst_case(adverse: core.Settlement, favourable: core.Settlement) -> core.WorstCase:
    """Whether the adverse model's solve is the worst case over every combination of the interval values (task 004-1):
    only where every candidate's transfers are the same in both models, so its payoff depends on the events alone; no
    candidate here settles on the business's own loss (each settles on a published observable)."""
    a, f = transfers(adverse), transfers(favourable)
    moved = sorted(c for c in set(a) | set(f) if a.get(c) != f.get(c))
    excluded = sorted(set(adverse.excluded) | set(favourable.excluded))
    # Equal transfers at the two endpoints do not show equal transfers on every event day between them (core.WorstCase);
    # where every event happens on one known day, no day lies between, and the transfers depend on the events alone.
    moving = sorted(e for e, d in adverse.event_days.items() if favourable.event_days.get(e) != d)
    if not moving and not excluded:
        return core.WorstCase(made=True, reason=(
            "every event of the scenario model happens on one known day, the same in both endpoint models, and every "
            "candidate settles in every scenario on observables those events date, never on the business's own loss: "
            "its transfers depend on which events occur alone, each day's deficit is nondecreasing in each loss, and the "
            "adverse model's solve is the worst case over every combination of the interval values"))  # fmt: skip
    why = []
    if moving:
        why.append(f"{', '.join(moving)} can happen on more than one day of {'its' if len(moving) == 1 else 'their'} "
                   "window, and a candidate settled on its observable pays on a day that moves with it; only the two "
                   "endpoint days were evaluated, which do not show the transfers for the days between")  # fmt: skip
    if moved:
        why.append(f"{len(moved)} candidate(s) pay differently in the two endpoint models ({', '.join(moved)}): their "
                   "transfers depend on when within its window the event happens, as the loss does")  # fmt: skip
    if excluded:
        why.append(f"{len(excluded)} candidate(s) do not settle in every scenario of one model ({', '.join(excluded)})")
    return core.WorstCase(made=False, reason="; ".join(why) + (". So the two solves are each that endpoint model's, and "
                                                             "the adverse one is not claimed to be the worst case"))  # fmt: skip


def chosen_value(problem: LiquidityProblem, solve: core.EndpointSolve, epsilon: Decimal
                 ) -> tuple[Fraction, Fraction, Fraction]:  # fmt: skip
    """The chosen design in another problem over the same classes: the chosen portfolio (the adverse solve's quantities)
    held as one instrument, its classes in their chosen proportions, sized by the same program (A06 §6.2) within the
    budget; its size (in chosen portfolios) and its objective, recomputed exactly. Ours: a design is its instruments and
    their proportions, and its size may change (A06 §6.7: "Quantity ... may change even if the structure family does
    not"). And the objective of buying nothing, beside it."""
    held = solve.result.hedged.quantities
    parts = [(c, held.get(c.ref, Decimal(0))) for c in problem.candidates]
    parts = [(c, q) for c, q in parts if q]
    base = LiquidityProblem(dates=problem.dates, probabilities=problem.probabilities, initial_cash=problem.initial_cash,
                            operating=problem.operating, loss=problem.loss, reserve=problem.reserve, alpha=problem.alpha,
                            budget=problem.budget)  # fmt: skip
    unhedged = cash_of(base, []).cvar
    if not parts:
        return Fraction(0), unhedged, unhedged
    with localcontext(EXACT):  # the chosen portfolio exactly: a rounding here raises
        price = sum((c.price * q for c, q in parts), Decimal(0))
        receipts = tuple(tuple(sum((c.receipts[s][t] * q for c, q in parts), Decimal(0))
                               for t in range(len(problem.dates))) for s in range(len(problem.probabilities)))  # fmt: skip
    one = Candidate("the chosen portfolio", price, parts[0][0].premium_date, receipts,
                    floor_to(Fraction(problem.budget) / Fraction(price), PRICE_QUANTUM, 6))  # fmt: skip
    sol = solve_lp(base.with_candidate(one), epsilon)
    return Fraction(sol.quantities[0]), sol.hedged.cvar, unhedged


@node(id="construct.repricing", version=1,
      inputs=["business", "receipts", "prices", "classes", "model", "parameters", "decision", "solve"])  # fmt: skip
def repricing(business: core.BusinessCash, receipts: core.Receipts, prices: core.Prices, classes: core.PayoffClasses,
              model: core.ScenarioModel, parameters: core.ConstructionParameters, decision: Decision,
              solve: core.EndpointSolve) -> core.Repricing:  # fmt: skip
    """The chosen design repriced over the loading range (spec §8.3; A06 §6.7: "Reprice over loading/quote ranges before
    claiming one design dominates another"): at the range's ends, the points splitting it into equal steps, and the
    loading used, the adverse model re-solved at the indicative prices of that loading, and the chosen design at those
    prices, sized by the program, beside it (chosen_value)."""
    lo, hi = parameters.loading_range
    loadings = sorted({lo + (hi - lo) * Fraction(i, STEPS) for i in range(STEPS + 1)} | {parameters.loading})
    probs = [s.probability for s in model.scenarios]
    tol = Fraction(decision[4])
    out = []
    for loading in loadings:
        price_of = {p.candidate: (indicative(p.expected_payout, loading) if p.expected_payout is not None else p.price)
                    for p in prices.prices}  # fmt: skip
        problem = program(business, receipts, classes, price_of, probs, parameters, decision)
        sol = solve_lp(problem, parameters.epsilon)
        scale, value, unhedged = chosen_value(problem, solve, parameters.epsilon)
        out.append(core.Repriced(loading=loading, structure=structure(sol), optimum=sol.hedged.cvar, scale=scale,
                                 chosen_portfolio=value, unhedged=unhedged,
                                 holds=core.construction.design_holds(solve.structure, value, unhedged,
                                                                      sol.hedged.cvar, exact(tol))))  # fmt: skip
    return core.Repricing(chosen=solve.structure, loadings=loadings, tolerance=exact(tol), repriced=out)


@node(id="construct.robustness", version=1,
      inputs=["business", "receipts", "prices", "classes", "model", "estimates", "forecasts", "parameters", "decision",
              "solve"])  # fmt: skip
def robustness(business: core.BusinessCash, receipts: core.Receipts, prices: core.Prices, classes: core.PayoffClasses,
               model: core.ScenarioModel, estimates: Sequence[core.ProbabilityEstimate],
               forecasts: Sequence[tuple[str, core.Ref]], parameters: core.ConstructionParameters, decision: Decision,
               solve: core.EndpointSolve) -> core.Robustness:  # fmt: skip
    """Each forecast the scenario model rests on, varied alone to each end of its band (spec §8.3; task 004-1: one at a
    time), the adverse model re-solved at the record's prices, and the chosen design, sized by the program, beside it
    (chosen_value): it holds where its objective is within the decision's tolerance of the re-solved optimum (A03
    §3.3)."""
    tol = Fraction(decision[4])
    by_id = {pe.id: pe for pe in estimates}
    question = dict(forecasts)
    base = {x: by_id[x].value for x in model.estimates}
    price_of = {p.candidate: p.price for p in prices.prices}
    bands, variations = [], []
    for x in model.estimates:
        pe = by_id[x]
        band = pe.interval
        ends = [] if band is None else [(n, v) for n, v in (("low", band[0]), ("high", band[1])) if v != pe.value]
        if band is None:
            reason = "no band: the estimate states no uncertainty, so it cannot be varied"
        elif not ends:
            reason = f"its band is the one value {_show(pe.value)}: the configurations agreed, so there is nothing to vary"
        else:
            reason = "varied to " + " and ".join(f"its {n} end {_show(v)}" for n, v in ends)
        varied = False
        at_ends = [(end, v, probabilities(model, {**base, x: v})) for end, v in ends]
        if any(sum(probs, Fraction(0)) != 1 for _, _, probs in at_ends):
            reason = ("varying it alone would leave its branch point no partition (A06 §6.7: perturb a coherent model, "
                      "never nudge one marginal into an impossible distribution), so it is not varied")  # fmt: skip
            at_ends = []
        for end, v, probs in at_ends:
            varied = True
            problem = program(business, receipts, classes, price_of, probs, parameters, decision)
            sol = solve_lp(problem, parameters.epsilon)
            scale, value, unhedged = chosen_value(problem, solve, parameters.epsilon)
            variations.append(core.Variation(
                event=pe.proposition, estimate=pe.id, end=end, probability=v, structure=structure(sol), optimum=sol.hedged.cvar,  # type: ignore[arg-type]
                scale=scale, chosen_portfolio=value, unhedged=unhedged,
                holds=core.construction.design_holds(solve.structure, value, unhedged, sol.hedged.cvar, exact(tol)),
            ))  # fmt: skip
        bands.append(core.EstimateBand(event=pe.proposition, estimate=pe.id, value=pe.value, band=band,
                                       question=question[pe.proposition], varied=varied, reason=reason))  # fmt: skip
    flips = sorted({v.estimate for v in variations if not v.holds})
    named = {b.estimate: b.question for b in bands}
    return core.Robustness(
        method=("one forecast at a time: each estimate the scenario model rests on set to each end of its band (the range "
                "of its configurations' probabilities), every other at its value, the scenario probabilities recomputed "
                "from the model's factors; the adverse model, the record's candidate set and prices held as recorded "
                "(price is varied by the repricing); the chosen design (the chosen portfolio held as one instrument, its "
                "size chosen by the program) holds where its objective is within the decision's tolerance of the "
                "re-solved optimum, an acceptable action (A03 §3.3; core.construction.design_holds)"),
        chosen=solve.structure, tolerance=exact(tol), bands=bands, variations=variations, flips=flips,
        questions=[named[e] for e in flips],
    )  # fmt: skip
