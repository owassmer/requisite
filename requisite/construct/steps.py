"""Construction's steps, as the CLI runs them (`requisite construct <step> --case-dir <dir> --case <name>`; spec §8.1-8.3,
outcome 004). A deep case's requisite run has chosen its top k (requisite/discovery/deep.py); construction protects the
business against the events of those exposures.

1. `observables`: for each event of the top k, the observable_scout agent proposes candidate observables (proposal kind
   "observable", through the store's gates: spec I1), and the judges answer the hedgeability questions of each active
   one (spec §8.1: client_can_influence, source_authoritative, timely) through the recorded provider layer, in spec
   §2.1's stability configurations, resolved under the application policy. The screening is committed.
2. `candidates`: the contracts the library's structure templates give over the observables that passed (spec §8.2),
   each accepted by the evaluator; shown, as the graph computes them.
3. `solve`: the step's parameters committed, then the case's computation graph with construction's vertices beside
   deep's recomputed (requisite/construct/graph.py): the scenario model, the business's dated cash in both endpoint
   models, every candidate's settlement in every scenario (I8), prices, the liquidity program under each model, the
   repricing over the loading range and the robustness to each forecast's band. The construction record is committed.
4. `report`: the latest construction record, as the dossier will show it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from typing import Any

from requisite import core
from requisite.agents import runner
from requisite.agents.answers import ObservablesDraft
from requisite.agents.trace import made_from, record_run
from requisite.compute.runtime import Instance, Runtime
from requisite.construct import graph as cgraph
from requisite.construct import nodes
from requisite.discovery import deep, wide
from requisite.judgment import providers
from requisite.judgment.render import TEXT
from requisite.library import templates as library
from requisite.optimize import DEFAULT_EPSILON
from requisite.stability import diagnose
from requisite.store import Refused, Store

SCOUT = "observable_scout"
RECORD = "construction"
QuoteRead = Callable[[str, datetime], core.Quote | None]


def decision_ref(case: deep.Case) -> str:
    return f"construct.{case.decision.id}"


def _exposures(case: deep.Case) -> dict[str, tuple[core.Driver, core.Proposition]]:
    return {f"{d.id}.{e.id}": (d, e) for d, e in case.exposures}


def _rows(store: Store, run: deep.RequisiteRun) -> dict[str, core.RegisterRow]:
    rows = {r.content_hash(): r for r in store.all("register_row", core.RegisterRow)}
    return {f"{r.driver}.{r.event}": r for h in run.rows if (r := rows.get(h)) is not None}


# ---------------------------------------------------------------- 1. observables (spec §8.1)


def observable_tests(store: Store, proposal: core.Proposal) -> core.GateResult:
    """An observable proposal's tests gate (core.REQUIRED_GATES): its event is a pending event of the case, its
    exposures are the case's exposures of that event, and the structure templates give, over it, IRs the evaluator
    accepts (spec §8.2: "Each candidate is a ContractIR the evaluator accepts")."""
    c = core.ObservableCandidate.model_validate_json(core.canonical_json(dict(proposal.payload)))
    case = deep.open_deep(store)
    events = {e.id: e for e in case.events}
    problems = []
    if c.event not in events:
        problems.append(f"{c.event} is not a pending event of the case")
    else:
        unknown = sorted(set(c.exposures) - {x for x, (_, e) in _exposures(case).items() if e.id == c.event})
        if unknown:
            problems.append(f"{unknown} are not the case's exposures of {c.event}")
        made: core.CandidateSet = nodes.candidates(((proposal.id, c),), ((c.event, events[c.event].window),),
                                                   case.decision.decision_time, case.model.currency)  # fmt: skip
        problems += [x for x in made.not_generated if nodes.REFUSED in x]
        if not made.candidates:
            problems.append("the structure templates give no contract over it")
    return core.GateResult(gate="tests", passed=not problems, detail="; ".join(problems) or (
        f"an observable of pending event {c.event}; the structure templates give {len(made.candidates)} contract(s) over "
        "it, each accepted by the evaluator"))  # fmt: skip


def _show_value(v: core.CmpValue) -> str:
    return str(v.value).lower() if isinstance(v, core.Flag) else str(v.value)


def observable_text(c: core.ObservableCandidate, event: core.Proposition) -> str:
    """The observable as a contract over it would define it, for the judges (spec §6.1: what the question is about)."""
    o = c.observable
    unit = f" in {o.unit}, published to {o.precision}" if o.unit else ""
    partial = f", and {_show_value(c.partial)} on a day it holds in part" if c.partial is not None else ""
    return (f"{o.description} ({o.value_type}{unit}); published {o.schedule}; revision policy {o.revision_policy}; its "
            f"value for a day is {_show_value(c.occurs)} on a day the adverse state of this event holds: "
            f"{deep.event_text(event)}; {_show_value(c.absent)} otherwise{partial}")  # fmt: skip


def sources_text(o: core.Observable) -> str:
    return "; ".join(f"{s.rank}. {s.name} ({s.role}{', ' + s.url if s.url else ''})"
                     for s in sorted(o.sources, key=lambda s: (s.rank, s.id)))  # fmt: skip


def _question(t: str, bindings: Mapping[str, str], qid: str, reaches: str) -> core.AtomicQuestion:
    q = library.instantiate(library.template(t), bindings, question_id=qid, paraphrase_set=f"ps.{t}")
    return q.model_copy(update={"reaches": (reaches,)})


def _scout_input(case: deep.Case, e: core.Proposition, xs: Sequence[str], rows: Mapping[str, core.RegisterRow]
                 ) -> dict[str, Any]:  # fmt: skip
    by = _exposures(case)
    exposures = []
    for x in xs:
        d, _ = by[x]
        r = rows.get(x)
        entry: dict[str, Any] = {"id": x, "line": deep.driver_text(d)}
        if r is not None:
            lo = -sum((m.high for m in r.cash_impact), Fraction(0))
            hi = -sum((m.low for m in r.cash_impact), Fraction(0))
            entry |= {"direction": r.direction, "mechanism": r.mechanism, "magnitude_band": r.magnitude_band,
                      "loss_over_horizon_if_it_occurs": [f"{float(lo):,.2f}", f"{float(hi):,.2f}"]}  # fmt: skip
        exposures.append(entry)
    return {
        "event": {"id": e.id, "statement": e.text, "window": deep._window(e.window)},  # noqa: SLF001
        "exposures": exposures, "loss_definition": case.model.loss_definition, "currency": case.model.currency,
        "horizon": deep._window(case.model.horizon), "evidence": case.rendered(TEXT).text,  # noqa: SLF001
    }  # fmt: skip


def _candidate(e: str, draft: Any) -> core.ObservableCandidate:
    o = draft.observable
    oid = o.id if o.id.startswith(f"{e}.") else f"{e}.{o.id}"
    fields = {**o.model_dump(mode="json"), "id": oid}
    return core.ObservableCandidate(
        event=e, exposures=list(draft.exposures), mechanism=draft.mechanism, occurs=draft.occurs, absent=draft.absent,
        partial=draft.partial, observable=core.Observable.model_validate_json(core.canonical_json(fields)),
    )  # fmt: skip


def observables(store: Store, k: int, *, judge: deep.JudgeCall | None = None, agent: deep.AgentCall | None = None
                ) -> core.ObservableScreening:  # fmt: skip
    """Step 1: candidate observables for each event of the top k, proposed and screened."""
    case = deep.open_deep(store)
    run = deep.latest_run(store, k)
    xs = cgraph.top_k(run)
    by = _exposures(case)
    events = sorted({by[x][1].id for x in xs})
    rows = _rows(store, run)
    store.register_gate("observable", "tests", observable_tests)
    call = agent or runner.run
    runs, proposals = [], []
    for eid in events:
        e = next(x for x in case.events if x.id == eid)
        mine = [x for x in xs if by[x][1].id == eid]
        given = _scout_input(case, e, mine, rows)
        result = call(SCOUT, given, ObservablesDraft, effort="high")
        runs.append(result.run[:16])
        done = made_from(store, result)
        if done:
            proposals += [p.id for p in done]  # this run's answer is already proposed (agents/trace.py)
            continue
        submitted = []
        for draft in result.output.candidates:
            payload: dict[str, Any]
            try:
                payload = _candidate(eid, draft).model_dump(mode="json")
            except ValueError:
                # The definition refuses the draft: it is proposed as drafted, and the schema gate rejects it with why.
                payload = {"event": eid, **draft.model_dump(mode="json")}
            submitted.append(store.submit_proposal(core.Proposal(
                id=f"observable-{core.sha256(core.canonical_json(payload))[:16]}", case=store.case, kind="observable",
                payload=payload, author=result.run[:16], rationale=result.output.rationale, status="proposed")))  # fmt: skip
        record_run(store, result, given, [p.id for p in submitted])
        proposals += [p.id for p in submitted]
    active = [p for p in (store.proposal(pid) for pid in dict.fromkeys(proposals)) if p is not None and p.status == "active"]
    screens = screen(case, active, deep.Judge(judge or providers.judge))
    screening = core.ObservableScreening(k=k, run=run.content_hash(), exposures=xs, events=events, agent_runs=runs,
                                         proposals=list(dict.fromkeys(proposals)), screens=screens)  # fmt: skip
    n = len(store.keys(cgraph.SCREENING))
    store.commit(cgraph.SCREENING, f"{k}|{n + 1:06d}|{screening.content_hash()}", screening)
    return screening


def screen(case: deep.Case, active: Sequence[core.Proposal], judge: deep.Judge) -> list[core.ObservableScreen]:
    """The hedgeability questions of each active candidate observable, judged in spec §2.1's stability configurations and
    resolved under the application policy (A04 §4.1), each answer's truth as applied (I6)."""
    events = {e.id: e for e in case.events}
    reach = decision_ref(case)
    asked: list[tuple[core.Proposal, core.ObservableCandidate, dict[str, deep.Asked]]] = []
    for p in active:
        c = core.ObservableCandidate.model_validate_json(core.canonical_json(dict(p.payload)))
        e = events[c.event]
        text = observable_text(c, e)
        bindings = {
            "client_can_influence": {"observable": text},
            "source_authoritative": {"observable": text, "sources": sources_text(c.observable)},
            "timely": {"observable": text, "event": deep.event_text(e), "window": deep._window(case.model.horizon)},  # noqa: SLF001
        }
        qs = {t: deep.Asked(_question(t, b, f"construct.{t}.{p.id}", reach), t, c.event, observable=p.id)
              for t, b in bindings.items()}  # fmt: skip
        asked.append((p, c, qs))
    js = deep.judge_questions(case, [q for _, _, qs in asked for q in qs.values()], judge)
    out = []
    for p, c, qs in asked:
        answers = {}
        for t, q in qs.items():
            r = None
            if len(diagnose.usable(js[q.id])) >= 2:
                r = diagnose.resolve(q.question, case.state_hash, js[q.id])
                case.store.commit("resolved_judgment", r.content_hash(), r)
            truth = deep.applied_truth(q, r)
            applied = r.application_state if r else "review"
            status = None if applied == "review" else next(a.status for a in q.question.answers if a.value == applied)
            answers[t] = core.HedgeabilityAnswer(question=q.question.ref, resolved=r.content_hash() if r else None,
                                                 applied=applied, status=status, truth=truth)  # fmt: skip
        passes = core.conj([answers["source_authoritative"].truth, core.neg(answers["client_can_influence"].truth),
                            answers["timely"].truth])  # fmt: skip
        out.append(core.ObservableScreen(proposal=p.id, observable=c.observable.id, event=c.event, answers=answers,
                                         passes=passes))  # fmt: skip
    return out


# ---------------------------------------------------------------- 2. candidates (spec §8.2)


def candidates(store: Store, k: int) -> core.CandidateSet:
    """Step 2: the candidate contracts over the observables that passed, as construction's node computes them."""
    case = deep.open_deep(store)
    screening = cgraph.latest_screening(store, k)
    if screening is None:
        raise Refused(f"no observable screening for the top {k}: run `requisite construct observables` first")
    passing = cgraph.screened(store, screening)
    windows = tuple((e.id, e.window) for e in case.events)
    made: core.CandidateSet = nodes.candidates(tuple(passing), windows, case.decision.decision_time, case.model.currency)
    return made


# ---------------------------------------------------------------- 3. solve (spec §8.3; addendum A06)


PRICES = ("yes_bid_dollars", "yes_ask_dollars", "no_bid_dollars", "no_ask_dollars", "last_price_dollars",
          "yes_bid_size_fp", "yes_ask_size_fp", "volume_fp", "open_interest_fp")  # what requisite.market.kalshi.quote reads


def kalshi_quote(market: str, at: datetime) -> core.Quote | None:
    """A listed market's quote as Kalshi's read-only API serves it, the request keyed by the decision's instant `at`
    (spec §10; I9); None where the market is not served (a 404 is the source's answer) or its record carries no prices,
    which the price then says (construct.prices)."""
    from requisite.market import kalshi

    client = kalshi.Kalshi(snapshot=at)
    try:
        fetched = client.market(market)
        record = fetched.json().get("market") or {} if fetched.status == 200 else {}
        if any(record.get(f) is None for f in PRICES):
            return None
        q = kalshi.quote(record, fetched)
    finally:
        client.close()
    return core.Quote(market=q.ticker, yes_bid=q.yes_bid, yes_ask=q.yes_ask, no_bid=q.no_bid, no_ask=q.no_ask,
                      last=q.last, yes_ask_size=q.yes_ask_size, at=q.at.isoformat(), locator=q.locator)  # fmt: skip


def solve(store: Store, *, k: int, budget: Decimal, loading: Fraction, loading_range: tuple[Fraction, Fraction],
          discount_rate: Fraction, eligibility: str, epsilon: Decimal = DEFAULT_EPSILON,
          judge: deep.JudgeCall | None = None, quote: QuoteRead | None = None) -> core.ConstructionRecord:  # fmt: skip
    """Step 3: the construction, through the case's computation graph, committed as a record."""
    case = deep.open_deep(store)
    res = deep.resolve_case(case, deep.Judge(judge or providers.judge))
    listed = sorted(p.id for p in wide.proposals(store, "contract") if p.status == "active")
    params = core.ConstructionParameters(
        case=store.case, decision_spec=core.Ref(id=case.decision.id, version=case.decision.version), k=k, budget=budget,
        loading=loading, loading_range=loading_range, discount_rate=discount_rate, epsilon=epsilon,
        eligibility=eligibility, listed=listed,  # type: ignore[arg-type]
    )  # fmt: skip
    read = quote or kalshi_quote
    when = case.decision.decision_time
    for pid in listed:
        p = store.proposal(pid)
        assert p is not None
        family = core.ContractFamily.model_validate_json(core.canonical_json(dict(p.payload)))
        for ir in family.markets:
            key = cgraph.quote_key(ir.id, when)
            if store.get(cgraph.QUOTE, key, core.Quote) is None and (q := read(ir.id, when)) is not None:
                store.commit(cgraph.QUOTE, key, q)
    graph = cgraph.full_graph(case, res, params)  # refused here, naming what is missing, before anything is committed
    n = len(store.keys(cgraph.PARAMETERS))
    store.commit(cgraph.PARAMETERS, f"{n + 1:06d}|{params.content_hash()}", params)
    rt = Runtime(store)
    rt.recompute(graph)
    record = assemble(case, params, rt)
    store.commit(RECORD, f"{k}|{record.content_hash()}", record)
    return record


def construction_vertices(rt: Runtime) -> list[str]:
    return [v.id for v in rt.graph if v.id.startswith(("v:construct.", "v:cash.", "v:business."))
            or (isinstance(v, Instance) and v.node.id.startswith(("cash.", "construct.")))]  # fmt: skip


def assemble(case: deep.Case, params: core.ConstructionParameters, rt: Runtime) -> core.ConstructionRecord:
    """The construction record from the graph's values (every part as its vertex computed it)."""
    v = rt.value
    models = core.construction.MODELS
    run = deep.latest_run(case.store, params.k)
    screening = cgraph.latest_screening(case.store, params.k)
    assert screening is not None
    d = case.decision
    solves = [v(f"v:construct.solve.{m}") for m in models]
    model: core.ScenarioModel = v("v:construct.model")
    rec = core.ConstructionRecord(
        case=case.store.case, decision=core.Ref(id=decision_ref(case), version=d.version), parameters=params,
        entity_model=core.Ref(id=case.model.id, version=case.model.version), decision_spec=core.Ref(id=d.id,
        version=d.version), objective=d.objective, alpha=nodes.tail(cgraph.decision_value(d)),  # type: ignore[arg-type]
        tolerance=d.tolerance, requisite_run=run.content_hash(), run_status=run.status, top_k=cgraph.top_k(run),
        evidence_hash=case.state_hash, screening=screening.content_hash(), screens=screening.screens,
        scenario_model=model, event_days=v("v:construct.event_days"), business=v("v:construct.business"),
        composition=composition(case, model, rt),
        candidates=v("v:construct.candidates"), settlements=[v(f"v:construct.settle.{m}") for m in models],
        receipts=[v(f"v:construct.receipts.{m}") for m in models], prices=[v(f"v:construct.prices.{m}") for m in models],
        classes=[v(f"v:construct.classes.{m}") for m in models], worst_case=v("v:construct.worst_case"), solves=solves,
        chosen=solves[0].structure, repricing=v("v:construct.repricing"), robustness=v("v:construct.robustness"),
        vertices={vid: rt.state(vid).value_hash for vid in construction_vertices(rt)}, statement="",
    )  # fmt: skip
    dec_id, dec_version, _, _, tolerance = cgraph.decision_value(d)
    recomputed(rec, (dec_id, dec_version, rec.objective, rec.alpha, tolerance))  # at the record's own objective and alpha
    return rec.model_copy(update={"statement": statement(rec, case)})


def recomputed(rec: core.ConstructionRecord, decision: nodes.Decision) -> None:
    """Every part of the record derived from its other parts, recomputed from them by the same nodes and equal to the
    record's; refused otherwise (core.ConstructionRecord): each model's receipts from its settlements, its payoff classes
    from its receipts and prices, each indicative price's expected payout from the settlements and the scenario
    probabilities, and each solve's cash, deficits and objective from the record's business, receipts, prices and
    quantities by the one cash accounting of A06 §6.1 (optimize.evaluate; A06 §6.4)."""
    from requisite.optimize import evaluate

    probs = [s.probability for s in rec.scenario_model.scenarios]
    by_id = {s.id: s.probability for s in rec.scenario_model.scenarios}
    for m, settlement in enumerate(rec.settlements):
        if nodes.receipts(settlement, rec.business) != rec.receipts[m]:
            raise Refused(f"{settlement.model}'s receipts are not its settlements placed on the business's grid")
        if nodes.classes(rec.receipts[m], rec.prices[m], rec.parameters) != rec.classes[m]:
            raise Refused(f"{settlement.model}'s payoff classes are not its receipts and prices grouped")
        paid = nodes.transfers(settlement)
        for p in rec.prices[m].prices:
            if p.expected_payout is not None and p.expected_payout != nodes.expected_payout(
                    paid[p.candidate], by_id, rec.business.premium_day, rec.parameters.discount_rate, rec.business.tz):
                raise Refused(f"{p.candidate}'s expected payout in {settlement.model} is not its settled transfers "
                              "discounted and weighted by the scenario model")  # fmt: skip
    for m, sv in enumerate(rec.solves):
        price_of = {p.candidate: p.price for p in rec.prices[m].prices}
        problem = nodes.program(rec.business, rec.receipts[m], rec.classes[m], price_of, probs, rec.parameters, decision)
        refs = [c.ref for c in problem.candidates]
        res = sv.result
        tol = res.hedged.breach_tolerance
        hedged = nodes.summary(evaluate(problem, [res.hedged.quantities[r] for r in refs], tol), refs)
        unhedged = nodes.summary(evaluate(problem, [0] * len(refs), tol), refs)
        if hedged != res.hedged or unhedged != res.unhedged:
            raise Refused(f"{sv.model}'s solve is not its quantities' cash recomputed from the record's own inputs "
                          "(A06 §6.4)")  # fmt: skip


def composition(case: deep.Case, model: core.ScenarioModel, rt: Runtime) -> str | None:
    """The drivers two or more events of the scenario model move (their computed effects not both zero), whose changes
    the business model adds where the events occur together: an assumption, stated (A06 §6.1)."""
    shared = []
    for d in case.model.drivers:
        moving = [e for e in model.events if rt.value(f"v:{d.id}.{e}.effect") != (0, 0)]
        if len(moving) >= 2:
            shared.append(f"{d.id} ({', '.join(moving)})")
    if not shared:
        return None
    return (f"where events occur together, their changes to a driver add, each a share of the driver's current amount as "
            f"the sensitivity questions ask it: {'; '.join(shared)}. Where two events would move a driver by one "
            "underlying change, this counts it twice")  # fmt: skip


def _money(x: Any) -> str:
    return f"{float(x):,.2f}"


def _frac(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def statement(r: core.ConstructionRecord, case: deep.Case) -> str:
    """The optimality claim as the purpose phrases it: within the declared tolerance, under this evidence, business model,
    candidate set, pricing policy and scenario model; robust only to what was tested."""
    adverse = r.solves[0]
    res = adverse.result
    cur = case.model.currency
    classes = r.classes[0].classes
    prices = r.prices[0]
    pricing: list[str] = sorted({p.pricing for p in prices.prices}) or ["none"]
    chosen = ", ".join(f"{_money(q)} units of {ref}" for ref, q in res.hedged.quantities.items()
                       if ref in adverse.structure)  # fmt: skip
    sm = r.scenario_model
    held = sum(1 for x in r.repricing.repriced if x.holds)
    varied = [x for x in r.robustness.variations]
    kept = sum(1 for x in varied if x.holds)
    return (
        f"Under the evidence as of {case.state.as_of.isoformat()} (state {r.evidence_hash[:12]}), the business model of "
        f"entity model {r.entity_model.id} v{r.entity_model.version} in its adverse endpoint model, the candidate set of "
        f"{len(r.candidates.candidates)} contract(s) generated from {sum(1 for s in r.screens if s.passes is core.Kleene.TRUE)} "
        f"screened observable(s) ({len(classes)} the solve tells apart), the pricing policy ({', '.join(pricing)}; loading "
        f"{_frac(r.parameters.loading)}, discount {_frac(r.parameters.discount_rate)} a year simple, actual/365) and the "
        f"scenario model ({sm.kind}, {len(sm.scenarios)} scenarios{'; ' + sm.assumption if sm.assumption else ''}"
        f"{'; and the business model assumes ' + r.composition if r.composition else ''}), "
        f"{'buying ' + chosen if chosen else 'buying nothing'} is optimal for {r.objective} at alpha {_frac(r.alpha)} "
        f"within the solver's tolerance of {res.epsilon * res.scale} {cur} (epsilon {res.epsilon} at scale {res.scale}), "
        f"within the budget of {_money(r.parameters.budget)} {cur}. The chosen structure stays within the decision's "
        f"tolerance of {_money(r.tolerance)} {cur} of optimal at {held} of {len(r.repricing.repriced)} loadings tested "
        f"in [{_frac(r.parameters.loading_range[0])}, {_frac(r.parameters.loading_range[1])}], and at {kept} of "
        f"{len(varied)} forecast variations, one forecast at a time across its band. "
        + ("It is the worst case over the business model's intervals: " if r.worst_case.made else
           "The favourable model's solve stands beside it: ") + r.worst_case.reason + "."
    )  # fmt: skip


def latest(store: Store, k: int | None = None) -> core.ConstructionRecord:
    found = [r for r in store.all(RECORD, core.ConstructionRecord) if k is None or r.parameters.k == k]
    if not found:
        raise Refused("no construction is committed in this case: run `requisite construct solve`")
    return found[-1]


# ---------------------------------------------------------------- what the CLI prints


def render_screening(s: core.ObservableScreening, store: Store) -> str:
    lines = [f"observables for the top {s.k} (requisite run {s.run[:12]}): exposures {', '.join(s.exposures)}; events "
             f"{', '.join(s.events)}; {len(s.agent_runs)} agent run(s), {len(s.proposals)} proposal(s)"]  # fmt: skip
    screened = {x.proposal: x for x in s.screens}
    for pid in s.proposals:
        p = store.proposal(pid)
        assert p is not None
        x = screened.get(pid)
        if x is None:
            failed = next((g for g in p.gates if not g.passed), None)
            lines.append(f"  {pid}: {p.status}{': ' + failed.gate + ': ' + failed.detail if failed else ''}")
            continue
        answers = ", ".join(f"{t} {a.applied} ({a.truth.value})" for t, a in sorted(x.answers.items()))
        lines.append(f"  {x.observable} ({pid}): passes {x.passes.value}: {answers}")
    return "\n".join(lines)


def render_candidates(cs: core.CandidateSet) -> str:
    lines = [f"{len(cs.candidates)} candidate contract(s), each accepted by the evaluator:"]
    for c in cs.candidates:
        lines.append(f"  {c.ir.id} [{c.family}, {c.variant}; {c.template.id} v{c.template.version}]: {c.description}")
    lines += [f"  not generated: {x}" for x in cs.not_generated]
    return "\n".join(lines)


def render_record(r: core.ConstructionRecord) -> str:
    p = r.parameters
    cur = r.candidates.candidates[0].ir.payout.currency if r.candidates.candidates else ""
    lines = [
        f"construction for {r.decision.id} v{r.decision.version} in case {r.case}: top {p.k} {', '.join(r.top_k)} "
        f"(requisite run {r.requisite_run[:12]}, {r.run_status}); analysis, not advice; prices are indicative unless quoted",
        f"  parameters: budget {_money(p.budget)} (step); objective {r.objective}, alpha {_frac(r.alpha)}, tolerance "
        f"{_money(r.tolerance)} (decision spec {r.decision_spec.id} v{r.decision_spec.version}); loading {_frac(p.loading)} "
        f"over [{_frac(p.loading_range[0])}, {_frac(p.loading_range[1])}], discount {_frac(p.discount_rate)} a year, "
        f"epsilon {p.epsilon}, eligibility {p.eligibility} (step)",
    ]  # fmt: skip
    sm = r.scenario_model
    lines.append(f"  scenario model: {sm.kind} ({sm.meaning}) over {', '.join(sm.events)}, from {', '.join(sm.estimates)}"
                 + (f"; assumed: {sm.assumption}" if sm.assumption else ""))  # fmt: skip
    for s in sm.scenarios:
        lines.append(f"    {s.id}: p = {_frac(s.probability)}")
    if r.composition:
        lines.append(f"  composition (assumed): {r.composition}")
    lines.append("  event days: " + "; ".join(f"{d.event} earliest {d.earliest} (adverse), latest "
                                               f"{d.latest or 'after the horizon'} (favourable)" for d in r.event_days))  # fmt: skip
    b = r.business
    lines.append(f"  business: liquidity {_money(b.initial)} at {b.grid[0]}; grid {b.grid[0]} to {b.grid[-1]} "
                 f"({len(b.grid)} days, {b.tz}); premium paid {b.premium_day}; {b.convention}")  # fmt: skip
    lines.append(f"  observables screened: {sum(1 for s in r.screens if s.passes is core.Kleene.TRUE)} of {len(r.screens)} pass")
    lines.append(render_candidates(r.candidates).replace("\n", "\n  ").join(("  ", "")))
    for prices, classes, receipts in zip(r.prices, r.classes, r.receipts, strict=True):
        lines.append(f"  prices ({prices.model} model; {prices.listed}):")
        by = {x.candidate: x for x in prices.prices}
        late = {c.candidate: c.after_horizon for c in receipts.candidates}
        for cl in classes.classes:
            x = by[cl.representative]
            members = ", ".join(f"{m} x{_frac(f)}" for m, f in cl.members if m != cl.representative)
            after = "; ".join(f"{_money(a)} in {sid}" for sid, a in zip(receipts.scenarios, late[cl.representative],
                                                                       strict=True) if a)  # fmt: skip
            lines.append(f"    {cl.representative}: {x.price} {cur} per unit ({x.pricing})"
                         + (f"; the solve cannot tell apart: {members}" if members else "")
                         + (f"; pays per unit after the horizon, outside the business's cash: {after}" if after else ""))  # fmt: skip
        if classes.pays_nothing:
            lines.append(f"    paying nothing under the scenario model: {', '.join(classes.pays_nothing)}")
        if classes.ineligible:
            lines.append(f"    ineligible ({p.eligibility}): {', '.join(classes.ineligible)}")
    if r.prices[0].prices:
        lines.append(f"    convention: {r.prices[0].prices[0].convention}")
    for solved in r.solves:
        lines += render_solve(solved, cur)
    lines.append(f"  worst case: {'made' if r.worst_case.made else 'not made'}: {r.worst_case.reason}")
    lines.append(f"  repriced over the loading range (tolerance {_money(r.repricing.tolerance)}):")
    for rp in r.repricing.repriced:
        lines.append(f"    loading {_frac(rp.loading)}: the re-solve buys {', '.join(rp.structure) or 'nothing'} (optimum "
                     f"{float(rp.optimum):,.2f}); the chosen design, {float(rp.scale):.4f} times the chosen portfolio, "
                     f"{float(rp.chosen_portfolio):,.2f}; buying nothing {float(rp.unhedged):,.2f}: "
                     f"{'holds' if rp.holds else 'DOES NOT HOLD'}")  # fmt: skip
    rb = r.robustness
    lines.append(f"  robustness ({rb.method}):")
    for band in rb.bands:
        lines.append(f"    {band.event} ({band.estimate}, p = {_frac(band.value)}"
                     f"{', band [' + _frac(band.band[0]) + ', ' + _frac(band.band[1]) + ']' if band.band else ''}): "
                     f"{band.reason}")  # fmt: skip
        for va in (v for v in rb.variations if v.event == band.event):
            lines.append(f"      at its {va.end} end {_frac(va.probability)}: the re-solve buys "
                         f"{', '.join(va.structure) or 'nothing'} (optimum {float(va.optimum):,.2f}); the chosen "
                         f"design, {float(va.scale):.4f} times the chosen portfolio, {float(va.chosen_portfolio):,.2f}; buying "
                         f"nothing {float(va.unhedged):,.2f}: {'holds' if va.holds else 'FLIPS'}")  # fmt: skip
    if rb.flips:
        lines.append("    high-priority questions (the forecast flips the choice): "
                     + ", ".join(f"{q.id} v{q.version}" for q in rb.questions))  # fmt: skip
    lines.append(f"  {r.statement}")
    return "\n".join(lines)


def render_solve(s: core.EndpointSolve, cur: str) -> list[str]:
    res = s.result
    h, u = res.hedged, res.unhedged
    chosen = ", ".join(f"{_money(q)} units of {ref}" for ref, q in h.quantities.items() if ref in s.structure)
    lines = [
        f"  {s.model} model{' (the chosen structure)' if s.model == 'adverse' else ' (beside it)'}: "
        f"{'buys ' + chosen if chosen else 'buys nothing'}; premium {_money(h.premium)} {cur}",
        f"    {res.objective}: {float(u.cvar):,.2f} unhedged, {float(h.cvar):,.2f} hedged; E[D] "
        f"{float(u.expected_liquidity_shortfall):,.2f} -> {float(h.expected_liquidity_shortfall):,.2f}; breach "
        f"probability {_frac(u.breach_probability)} -> {_frac(h.breach_probability)} (beyond {h.breach_tolerance})",
        "    scenario: p, loss, receipts, premium, residual; deficit unhedged (first breach) -> hedged (first breach)",
    ]  # fmt: skip
    for row in s.table:
        lines.append(f"      {row.scenario}: {_frac(row.probability)}, {_money(row.loss)}, {_money(row.receipts)}, "
                     f"{_money(row.premium)}, {_money(row.residual)}; {_money(row.deficit_unhedged)} "
                     f"({row.first_breach_unhedged or '-'}) -> {_money(row.deficit_hedged)} "
                     f"({row.first_breach_hedged or '-'})")  # fmt: skip
    lines.append(f"    uncovered (still short with the hedge): {', '.join(s.uncovered) or 'none'}")
    for g in s.timing_gaps:
        lines.append(f"    timing gap in {g.scenario}: short on {g.days} day(s) from {g.first_day} to {g.last_day}, at "
                     f"worst {_money(g.shortfall)} on {g.worst_day}, while the hedge pays {_money(g.later)} after it")  # fmt: skip
    for rc in res.reduced_costs:
        lines.append(f"    unbought {rc.candidate}: reduced cost {float(rc.value):,.4f} beside its re-solve's measured "
                     f"improvement {float(rc.measured_improvement):,.4f} (a diagnostic: duals are not unique)")  # fmt: skip
    return lines
