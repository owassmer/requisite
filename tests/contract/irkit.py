"""Builders for the contract tests: small IRs over one or two observables, and their observations. Each builder states
every field the definition requires, with defaults a test overrides by keyword."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal as D
from typing import Any

from requisite import core as c
from requisite.core.evidence import artifact_id, synthetic_artifact_id

T0 = datetime(2027, 3, 1, tzinfo=UTC)
EXPIRY = datetime(2027, 6, 1, 14, tzinfo=UTC)
NY = "America/New_York"


def source(sid: str = "agency", rank: int = 1) -> c.Source:
    return c.Source(id=sid, name=sid, rank=rank, role="official_record")


def observable(oid: str = "x", *, value_type: str = "number", sources=("agency",), policy: str = "first_print",
               **kw: Any) -> c.Observable:  # fmt: skip
    numeric = {"unit": "units", "precision": D("0.01")} if value_type == "number" else {}
    return c.Observable(
        id=oid, description=oid, value_type=value_type, sources=[source(s, i + 1) for i, s in enumerate(sources)],
        revision_policy=policy, **{**numeric, **kw},
    )  # fmt: skip


def timing(**kw: Any) -> c.Timing:
    fields: dict[str, Any] = {
        "open": T0, "close": EXPIRY, "expiration": EXPIRY, "observation_cutoff": EXPIRY, "settlement_lag_days": 0,
        "payment_lag_days": 0, "lag_days": "calendar", "holiday_calendar": None, "tz": NY,
    }  # fmt: skip
    return c.Timing(**{**fields, **kw})


def procedure(**kw: Any) -> c.ResolutionProcedure:
    fields: dict[str, Any] = {
        "precedence": ["agency"],
        "conflict_policy": c.ConflictPolicy(procedure="precedence_first_available"),
        "on_unknown_at_expiry": "no",
        "early_expiration": "none",
        "void_transfer": c.VoidTransfer(returns="nothing"),
    }
    return c.ResolutionProcedure(**{**fields, **kw})


def binary(amount: str = "1.00", **kw: Any) -> c.Payout:
    fields: dict[str, Any] = {
        "kind": "binary", "unit_value": D(amount), "currency": "USD", "rounding_quantum": D("0.01"),
        "rounding_mode": "half_even", "payment": "at_settlement",
    }  # fmt: skip
    return c.Payout(**{**fields, **kw})


def contract(trigger: Any, *, observables=None, payout=None, **kw: Any) -> c.ContractIR:
    return c.ContractIR(
        id=kw.pop("id", "k"), version=1, observables=observables if observables is not None else [observable()],
        trigger=trigger, payout=payout or binary(), timing=kw.pop("timing", timing()),
        resolution=kw.pop("resolution", procedure()), **kw,
    )  # fmt: skip


def num(v: str | int) -> c.Number:
    return c.Number(value=D(str(v)))


def at(oid: str, op: str, v: str | int, when: datetime, nid: str | None = None) -> c.At:
    return c.At(id=nid or f"at_{oid}_{op}_{v}", expr=c.Cmp(id=f"cmp_{nid or oid}", observable=oid, op=op, value=num(v)),
                at=when)  # fmt: skip


def obs(value: str | int | bool | c.Number, when: datetime, published: datetime | None = None, *, oid: str = "x",
        src: str = "agency", final: bool = True, synthetic: bool = False) -> c.Observation | c.SyntheticObservation:  # fmt: skip
    v = value if isinstance(value, c.Record) else (c.Flag(value=value) if isinstance(value, bool) else num(value))
    published = published or when
    key = f"{oid}|{src}|{when.isoformat()}|{published.isoformat()}|{v.value}".encode()
    if synthetic:
        return c.SyntheticObservation(observable=oid, source=src, at=when, value=v, published_at=published, final=final,
                                      artifact=synthetic_artifact_id(key))  # fmt: skip
    return c.Observation(observable=oid, source=src, at=when, value=v, published_at=published, final=final,
                         artifact=artifact_id(key))  # fmt: skip


def days(n: int, start: datetime = T0) -> list[datetime]:
    return [start + timedelta(days=i) for i in range(n)]


# Consequence nodes for the tests of Computed: one declared monotone in each input, one declared in none, and one that
# declares a direction it does not have.
from requisite.compute.registry import node  # noqa: E402


@node(id="tests.contract.difference", version=1, inputs=["a", "b"], monotone={"a": "increasing", "b": "decreasing"})
def difference(a, b):
    return a - b


@node(id="tests.contract.weighted", version=1, inputs=["x", "y"], monotone={"x": "increasing", "y": "increasing"})
def weighted(x, y):
    return 2 * x + 3 * y  # z = 2x + 3y: some values it reaches only with both inputs moved (external evaluation, Nov 2)


@node(id="tests.contract.square", version=1, inputs=["a"])
def square(a):
    return a * a


@node(id="tests.contract.misdeclared", version=1, inputs=["a"], monotone={"a": "increasing"})
def misdeclared(a):
    return -a


def reading(rid: str = "r", bands=((0, 10), (10, 20), (20, 30))) -> c.ReadValue:
    return c.ReadValue(id=rid, question_template=f"read.{rid}", bindings={}, unit="units",
                       bands=[(D(lo), D(hi)) for lo, hi in bands])  # fmt: skip


def band(lo: int | str, hi: int | str) -> tuple[D, D]:
    return (D(str(lo)), D(str(hi)))
