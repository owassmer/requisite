"""Construction's vertices in the case's computation graph, beside deep discovery's (requisite/discovery/deep.py
build_graph): one graph from the entity model's parameters to the chosen structure, so the runtime commits every
receipt and a change anywhere upstream (a parameter, the planning horizon, a forecast) recomputes exactly what depends on
it, and the diff shows it (spec §5, §15 beat 5).

What enters from outside the graph enters as a source naming its committed records (requisite/compute/runtime.py): the
construction's parameters; the decision spec's objective, alpha and tolerance; the pending events and their windows;
each event's ProbabilityEstimate and the forecast question it came from; a committed event tree where there is one; the
observables the agent proposed that passed the judges' screens; the case's currency; and the listed contracts the case
holds with their quotes. The agent's proposals and the judges' answers stay outside, as deep's judgments do.

How the parts depend on one another, so what a change reaches is read from the edges: the business's cash
(v:business.*, v:cash.*, v:construct.business) depends on the driver tree, the liquidity profile, the floor, the
horizon's days and the exposures' effects and starts; the candidate contracts (v:construct.candidates) and their
settlement (v:construct.settle.*) depend only on the screened observables, the events' windows, the decision time and
the currency, never on the horizon; receipts, prices, classes and solves depend on both.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from requisite import core
from requisite.compute.runtime import Graph, Instance, Source, Vertex
from requisite.construct import nodes
from requisite.discovery import deep
from requisite.store import Refused, Store

PARAMETERS = "construction_parameters"
SCREENING = "observable_screening"
QUOTE = "quote"
TREE = "event_tree"


def _src(id: str, layer: str, value: Any, *records: core.Record) -> Source:
    return Source(id, layer, value, tuple(records))  # type: ignore[arg-type]


def _inst(id: str, node: str, layer: str = "computation", **inputs: str) -> Instance:
    return Instance(id=id, node=core.Ref(id=node, version=1), inputs=inputs, layer=layer)  # type: ignore[arg-type]


def latest_parameters(store: Store) -> core.ConstructionParameters | None:
    found = store.all(PARAMETERS, core.ConstructionParameters)
    return found[-1] if found else None


def latest_screening(store: Store, k: int) -> core.ObservableScreening | None:
    found = [s for s in store.all(SCREENING, core.ObservableScreening) if s.k == k]
    return found[-1] if found else None


def top_k(run: deep.RequisiteRun) -> list[str]:
    """The exposures of the register's top k: every exposure of a minimax-optimal top-k set (deep.analyse), so a tie
    among sets leaves none of them out."""
    return sorted({x for s in run.top_k for x in s})


def decision_value(d: core.DecisionSpec) -> nodes.Decision:
    """The decision spec's id, version, objective, alpha and tolerance: what construction reads of it (its horizon is the
    entity model's, read through the business's days)."""
    return (d.id, d.version, d.objective, d.alpha, d.tolerance)


def screened(store: Store, screening: core.ObservableScreening) -> list[tuple[str, core.ObservableCandidate]]:
    """The candidate observables whose screens pass, each with its active proposal."""
    out = []
    for s in screening.screens:
        if s.passes is not core.Kleene.TRUE:
            continue
        p = store.proposal(s.proposal)
        if p is None or p.status != "active":
            raise Refused(f"screened observable proposal {s.proposal} is not active in the case")
        out.append((p.id, core.ObservableCandidate.model_validate_json(core.canonical_json(dict(p.payload)))))
    return out


def quote_key(market: str, decision_time: datetime) -> str:
    """A quote read for a decision is kept under its market and the decision's time (in UTC)."""
    return f"{market}|{decision_time.astimezone(UTC).isoformat()}"


def listed(store: Store, params: core.ConstructionParameters, decision_time: datetime
           ) -> tuple[tuple[nodes.Listed, ...], tuple[core.Record, ...]]:  # fmt: skip
    """The listed markets of the contract families the case held when the parameters were committed, each with the quote
    read for this decision where one was served, and the records they are read from."""
    out: list[nodes.Listed] = []
    records: list[core.Record] = []
    for pid in params.listed:
        p = store.proposal(pid)
        if p is None or p.kind != "contract":
            raise Refused(f"listed contract {pid} is not a contract proposal of the case")
        family = core.ContractFamily.model_validate_json(core.canonical_json(dict(p.payload)))
        records.append(family)
        for ir in family.markets:
            q = store.get(QUOTE, quote_key(ir.id, decision_time), core.Quote)
            if q is not None:
                records.append(q)
            out.append((ir.id, ir, q))
    return tuple(out), tuple(records) or (params,)


def scenario_tree(store: Store, events: list[str]) -> core.EventTree | None:
    """The committed event tree that joins the case's events, where there is one (A04 §4.2): the latest committed."""
    trees = store.all(TREE, core.EventTree)
    return trees[-1] if trees else None


def vertices(case: deep.Case, res: deep.Resolutions, graph: Graph,
             params: core.ConstructionParameters) -> list[Vertex]:  # fmt: skip
    """Construction's vertices over deep's graph `graph` of the same case and resolutions, for these parameters. Refused,
    naming the missing input, where the scenario model or the business's cash cannot be computed."""
    store, d = case.store, case.decision
    try:
        nodes.tail(decision_value(d))
    except ValueError as e:
        raise Refused(str(e)) from None
    events = [e.id for e in case.events]
    missing = [e for e in events if e not in res.estimates]
    if missing:
        raise Refused("no probability estimate for " + ", ".join(
            f"{e} (its forecast {res.forecast_leaf(e).id} has fewer than two distributions)" for e in missing)
            + ": the scenario model is refused rather than solved over the events that have one (task 004-1)")  # fmt: skip
    values = deep.evaluate(graph)
    for dr, ev in case.exposures:
        x = f"{dr.id}.{ev.id}"
        lo, hi = values[f"v:{x}.effect"]
        if lo is None or hi is None:
            raise Refused(f"the business's cash under {ev.id} has no bound: {deep._unbounded_by(res, x)}")  # noqa: SLF001
    run = deep.latest_run(store, params.k)
    screening = latest_screening(store, params.k)
    if screening is None:
        raise Refused(f"no observable screening for the top {params.k}: run `requisite construct observables` first")
    if list(screening.exposures) != top_k(run):
        raise Refused(f"the observable screening for the top {params.k} served the exposures {list(screening.exposures)}; "
                      f"the latest requisite run's top {params.k} is {top_k(run)}: run `requisite construct observables` "
                      "again")  # fmt: skip
    passing = screened(store, screening)
    estimates = [res.estimates[e] for e in events]
    forecasts = [res.forecast_leaf(e).question for e in events]
    listed_value, listed_records = listed(store, params, d.decision_time)
    out: list[Vertex] = [
        _src("v:construct.parameters", "computation", params, params),
        _src("v:construct.decision", "computation", decision_value(d), d),
        _src("v:construct.events", "proposition", tuple(events), *case.events),
        _src("v:construct.windows", "proposition", tuple((e.id, e.window) for e in case.events), *case.events),
        _src("v:construct.estimates", "proposition", tuple(estimates), *estimates),
        _src("v:construct.forecasts", "proposition", tuple(zip(events, (q.ref for q in forecasts), strict=True)),
             *forecasts),
        _src("v:construct.observables", "evidence", tuple(passing), screening, *(c for _, c in passing)),
        _src("v:construct.currency", "computation", case.model.currency, case.model),
        _src("v:construct.listed", "evidence", listed_value, *listed_records),
    ]  # fmt: skip
    tree = scenario_tree(store, events)
    if tree is None:
        assignments = nodes.independent_assignments(events)
        out.append(_inst("v:construct.model", "construct.independent_model", "proposition",
                         events="v:construct.events", estimates="v:construct.estimates"))  # fmt: skip
    else:
        try:
            assignments = nodes.tree_assignments(events, tree)
        except ValueError as err:
            raise Refused(str(err)) from None
        branch = [store.get("probability_estimate", b.probability, core.ProbabilityEstimate)
                  for path in nodes.tree_paths(tree) for b in path]  # fmt: skip
        if any(x is None for x in branch):
            raise Refused(f"event tree {tree.id} names a branch probability the case has not committed")
        tree_estimates = tuple({x.id: x for x in branch if x is not None}.values())
        out += [
            _src("v:construct.tree", "proposition", tree, tree),
            _src("v:construct.tree_estimates", "proposition", tree_estimates, *tree_estimates),
            _inst("v:construct.model", "construct.tree_model", "proposition", events="v:construct.events",
                  tree="v:construct.tree", estimates="v:construct.tree_estimates"),
        ]  # fmt: skip
    out += [
        _inst("v:construct.assignments", "construct.assignments", "proposition", model="v:construct.model"),
        _inst("v:construct.event_days", "construct.event_days", "proposition", as_of="v:as_of",
              windows="v:construct.windows"),
        _inst("v:construct.candidates", "construct.candidates", "contract", observables="v:construct.observables",
              windows="v:construct.windows", open="v:as_of", currency="v:construct.currency"),
    ]  # fmt: skip
    for m in core.construction.MODELS:
        out.append(_inst(f"v:construct.settle.{m}", f"construct.settle_{m}", "contract",
                         candidates="v:construct.candidates", assignments="v:construct.assignments",
                         event_days="v:construct.event_days"))  # fmt: skip
    out += business(case, assignments, graph)
    for m in core.construction.MODELS:
        out += [
            _inst(f"v:construct.receipts.{m}", "construct.receipts", settlement=f"v:construct.settle.{m}",
                  business="v:construct.business"),
            _inst(f"v:construct.prices.{m}", "construct.prices", settlement=f"v:construct.settle.{m}",
                  candidates="v:construct.candidates", model="v:construct.model", business="v:construct.business",
                  parameters="v:construct.parameters", listed="v:construct.listed"),
            _inst(f"v:construct.classes.{m}", "construct.classes", receipts=f"v:construct.receipts.{m}",
                  prices=f"v:construct.prices.{m}", parameters="v:construct.parameters"),
            _inst(f"v:construct.solve.{m}", "construct.solve", business="v:construct.business",
                  receipts=f"v:construct.receipts.{m}", prices=f"v:construct.prices.{m}",
                  classes=f"v:construct.classes.{m}", candidates="v:construct.candidates", model="v:construct.model",
                  parameters="v:construct.parameters", decision="v:construct.decision"),
        ]  # fmt: skip
    a = "adverse"
    out += [
        _inst("v:construct.worst_case", "construct.worst_case", adverse="v:construct.settle.adverse",
              favourable="v:construct.settle.favourable"),
        _inst("v:construct.repricing", "construct.repricing", business="v:construct.business",
              receipts=f"v:construct.receipts.{a}", prices=f"v:construct.prices.{a}", classes=f"v:construct.classes.{a}",
              model="v:construct.model", parameters="v:construct.parameters", decision="v:construct.decision",
              solve=f"v:construct.solve.{a}"),
        _inst("v:construct.robustness", "construct.robustness", business="v:construct.business",
              receipts=f"v:construct.receipts.{a}",
              prices=f"v:construct.prices.{a}", classes=f"v:construct.classes.{a}", model="v:construct.model",
              estimates="v:construct.estimates" if tree is None else "v:construct.tree_estimates",
              forecasts="v:construct.forecasts", parameters="v:construct.parameters", decision="v:construct.decision",
              solve=f"v:construct.solve.{a}"),
    ]  # fmt: skip
    return out


def business(case: deep.Case, assignments: list[nodes.Assignment], graph: Graph) -> list[Vertex]:
    """The business's dated cash through deep's driver tree (requisite/library/nodes/cash.py): each driver's daily flow,
    their sum, the liquidity, the floor, each exposure's daily change at its two ends, each scenario's change (the
    exposures of the events that occur in it, added), and the business on the grid."""
    out: list[Vertex] = []
    drivers = list(case.model.drivers)
    for dr in drivers:
        out.append(_inst(f"v:{dr.id}.daily", "cash.daily_flows", amounts=f"v:{dr.id}.amounts", days="v:days",
                         line=f"v:{dr.id}.line"))  # fmt: skip
    operating = f"v:{drivers[0].id}.daily"
    for i, dr in enumerate(drivers[1:], 1):
        out.append(_inst(f"v:business.operating.{i}", "cash.add_flows", a=operating, b=f"v:{dr.id}.daily"))
        operating = f"v:business.operating.{i}"
    out += [
        _inst("v:business.liquidity", "cash.liquidity", cash="v:cash", credit="v:credit"),
        _inst("v:business.reserve", "cash.reserve", floor="v:floor", days="v:days"),
        _inst("v:cash.none", "cash.no_change", days="v:days"),
    ]
    # Each exposure's change in each endpoint model, started from the day that model puts its event on (nodes._start_in),
    # with the inputs deep's own start reads (the event's window, the exposure's lag, the line's payment terms).
    for dr, e in case.exposures:
        x = f"{dr.id}.{e.id}"
        deep_start = graph[deep.exposure_vertices(x)["start"]]
        assert isinstance(deep_start, Instance)
        for model in core.construction.MODELS:
            out.append(_inst(f"v:{x}.start.{model}", f"construct.start_{model}", as_of="v:as_of",
                             window=deep_start.inputs["window"], lag=deep_start.inputs["lag"],
                             terms_days=deep_start.inputs["terms_days"], horizon="v:horizon"))  # fmt: skip
            out.append(_inst(f"v:{x}.daily.{model}", "cash.daily_impact", amounts=f"v:{dr.id}.amounts", days="v:days",
                             line=f"v:{dr.id}.line", effect=f"v:{x}.effect", start=f"v:{x}.start.{model}"))  # fmt: skip
    collected: dict[str, str] = {}
    for model in core.construction.MODELS:
        changes = []
        for sid, a in assignments:
            path = "v:cash.none"
            occurring = [e for e, occurs in a if occurs]
            for i, (dr, e) in enumerate((dr, e) for e_id in occurring for dr, e in case.exposures if e.id == e_id):
                vid = f"v:cash.{model}.{sid}.{i + 1}"
                out.append(_inst(vid, "cash.add_change", change=path, impact=f"v:{dr.id}.{e.id}.daily.{model}"))
                path = vid
            changes.append(path)
        last = f"v:construct.changes.{model}.1"
        out.append(_inst(last, "construct.collect_first", item=changes[0]))
        for i, path in enumerate(changes[1:], 2):
            out.append(_inst(f"v:construct.changes.{model}.{i}", "construct.collect", items=last, item=path))
            last = f"v:construct.changes.{model}.{i}"
        collected[model] = last
    out.append(_inst("v:construct.business", "construct.business", days="v:days", operating=operating,
                     liquidity="v:business.liquidity", reserve="v:business.reserve", adverse=collected["adverse"],
                     favourable=collected["favourable"], as_of="v:as_of", horizon="v:horizon",
                     assignments="v:construct.assignments", currency="v:construct.currency"))  # fmt: skip
    return out


def full_graph(case: deep.Case, res: deep.Resolutions, params: core.ConstructionParameters | None = None) -> Graph:
    """Deep's graph, and construction's beside it where the case holds construction parameters (or `params` is given)."""
    graph = deep.build_graph(case, res)
    params = params or latest_parameters(case.store)
    if params is None:
        return graph
    return graph.replace(*vertices(case, res, graph, params))
