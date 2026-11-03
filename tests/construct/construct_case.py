"""A deep case for construction's tests, built by tests/discovery/deep_case.py over the product's own steps (deep's
requisite loop, then construction's observable step), and test doubles of the observable agent and the judges: the
tests' own input, never a cassette.

The business: landscaping contracts bringing in 12 x 4,100 a month and crews costing 12 x 4,000, so 1,200 a month of
operating cash before any rule; 40,000 of cash and no credit line. The overtime rule (E1) raises the crews' wages 5-15%
within one to three months of taking effect and does nothing to the contracts; the judges put its probability at 0.6
unless a test says otherwise. Every candidate observable the agent double proposes is judged authoritative, beyond the
business's influence and timely."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal as D
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import deep_case as dc

from requisite import core
from requisite.agents.answers import ObservableDraft, ObservablesDraft
from requisite.construct import steps
from requisite.discovery import deep
from requisite.judgment import providers
from requisite.store import Store

DRIVERS = (dc.DriverSpec("sales", "landscaping contracts", "12", "4100", "revenue"),
           dc.DriverSpec("labor", "crew wages", "12", "4000"))  # fmt: skip
HEDGEABILITY = ("source_authoritative", "timely")
E2 = core.Proposition(id="E2", text="The fuel tax rule takes effect", window=dc.WINDOW, domain="bool")
SALES = {"sensitivity_direction": "none", "sensitivity_mechanism": "revenue", "sensitivity_magnitude": "under_1pct",
         "sensitivity_lag": "under_1_month", "sensitivity_conditionality": "unconditional"}  # fmt: skip


@dataclass
class Rule:
    """How the judges answer: deep's seeded world (deep_case.base_rule), the contracts unmoved by any rule, every
    observable hedgeable, and each event's probability by provider (a single number where they agree)."""

    forecast: Mapping[str, Mapping[str, float] | float | None] = field(default_factory=dict)
    hedgeable: Mapping[str, str] = field(default_factory=lambda: {"source_authoritative": "authoritative",
                                                                    "timely": "timely"})  # fmt: skip

    def __call__(self, a: providers.Ask) -> Mapping[str, float]:
        values = list(a.asked.answers)
        t = a.question.template_id
        if t in self.hedgeable:
            return dc.leaning(values, self.hedgeable[t])
        if t == deep.FORECAST:
            event = a.question.id.rsplit(".", 1)[1]
            p = self.forecast.get(event, 0.6)
            if p is None:
                return {}  # every configuration refuses: the event gets no probability
            p = p if isinstance(p, float) else p[a.config.provider]
            return {"occurs": p, "does_not_occur": round(1 - p, 2)}
        if t in SALES and ".sales." in a.question.id:
            return dc.leaning(values, SALES[t])
        return dc.base_rule(a)  # each event moves the crews' wages as deep's seeded world has E1 do


def observable(event: str, oid: str, *, value_type: str = "bool") -> core.Observable:
    fr = core.Source(id="federal_register", name="Federal Register", url="https://www.federalregister.gov", rank=1,
                     role="official_record")  # fmt: skip
    agency = core.Source(id="agency", name="The issuing agency's rule page", rank=2, role="agency_record")
    numeric = {"unit": "USD/week", "precision": D("0.01")} if value_type == "number" else {}
    return core.Observable(id=oid, description=f"whether the rule of {event} is in effect on a day, by its final rule",
                           value_type=value_type, sources=[fr, agency], schedule="daily, on business days",
                           revision_policy="first_print", **numeric)  # type: ignore[arg-type]  # fmt: skip


def bool_draft(event: str, exposures: Sequence[str], oid: str = "in_effect") -> ObservableDraft:
    return ObservableDraft(observable=observable(event, oid), exposures=list(exposures),
                           mechanism="the rule raises what the business pays its crews", occurs=core.Flag(value=True),
                           absent=core.Flag(value=False), partial=None)  # fmt: skip


@dataclass
class Scout:
    """The observable_scout's double: for each event, the drafts `drafts` gives (one bool observable of whether its rule
    is in effect, by default), answering every other role as deep's double does."""

    drafts: Any = None
    calls: list[tuple[str, Mapping[str, Any]]] = field(default_factory=list)

    def __call__(self, role: str, input: Mapping[str, Any]) -> Any:
        self.calls.append((role, input))
        if role != steps.SCOUT:
            return dc.refiner(role, input)
        event = input["event"]["id"]
        exposures = [x["id"] for x in input["exposures"]]
        drafts = self.drafts(event, exposures) if self.drafts else [bool_draft(event, exposures)]
        return ObservablesDraft(candidates=drafts, rationale="the Federal Register publishes the final rule")


@dataclass
class Judges(dc.Judges):
    """deep_case's judges, where a rule's empty answer is a refusal: the provider returned no distribution."""

    def __call__(self, a: providers.Ask) -> core.Judgment:
        if self.rule(a):
            return super().__call__(a)
        self.calls.append(a)
        return core.Judgment(question=core.Ref(id=a.question.id, version=a.question.version), state_hash=a.state_hash,
                             config=a.config, reported=None, distribution=None, refusal="the double refuses",
                             precision=0.01, response="double-refusal", latency_ms=0, cost_usd=D("0"), created_at=dc.NOW)  # fmt: skip


@dataclass
class Built:
    store: Store
    judges: dc.Judges
    agent: dc.Agent
    run: deep.RequisiteRun
    screening: core.ObservableScreening


def build(tmp_path: Path, *, cash: str = "40000", credit: str = "0", k: int = 1, rule: Rule | None = None,
          events: Sequence[core.Proposition] = (dc.EVENT,), scout: Scout | None = None,
          drivers: Sequence[dc.DriverSpec] = DRIVERS, tolerance: str = "100") -> Built:  # fmt: skip
    """The deep case, its requisite loop for the top k, and construction's observable step."""
    store = dc.make_case(tmp_path, drivers=drivers, tolerance=tolerance, cash=cash, credit=credit, events=events,
                         templates=HEDGEABILITY)  # fmt: skip
    judges = Judges(rule or Rule())
    agent = dc.Agent(scout or Scout())
    out = deep.loop(store, k, judge=judges, agent=agent, listing=dc.listing)
    screening = steps.observables(store, k, judge=judges, agent=agent)
    return Built(store, judges, agent, out.run, screening)


PARAMETERS: dict[str, Any] = {"budget": D("30000"), "loading": F(1, 5), "loading_range": (F(0), F(1, 2)),
                              "discount_rate": F(1, 20), "eligibility": "bilateral"}  # fmt: skip


def solve(b: Built, k: int = 1, **overrides: Any) -> core.ConstructionRecord:
    return steps.solve(b.store, k=k, judge=b.judges, **{**PARAMETERS, **overrides})


def at(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(tzinfo=UTC)
