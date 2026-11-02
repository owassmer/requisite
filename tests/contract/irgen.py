"""Random contracts and worlds for the property tests of the settlement function (test_evaluate_properties.py,
test_lowering.py): two observables on a daily grid of six instants, x under a "final" revision policy, whose non-final
values are unknown, and y under require_agreement of two sources, whose value is unknown while only one has published."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal as D
from itertools import count

from hypothesis import strategies as st
from irkit import obs, observable, procedure, timing

from requisite import core as c
from requisite.core.kleene import Kleene

T0 = datetime(2027, 3, 1, tzinfo=UTC)
GRID = [T0 + timedelta(days=i) for i in range(6)]
EXPIRY, CUTOFF = T0 + timedelta(days=6), T0 + timedelta(days=7)
BANDS = [(D(0), D(10)), (D(10), D(20)), (D(20), D(30))]
OBSERVABLES = [
    observable("x", sources=("a",), policy="final", final_deadline=T0 + timedelta(days=8), precision=D(1)),
    observable("y", sources=("a", "b"), precision=D(1)),
]
PROCEDURE = dict(precedence=["a", "b"], conflict_policy=c.ConflictPolicy(procedure="require_agreement", sources=["a", "b"]))
OPS = ["<", "<=", "==", "!=", ">=", ">"]
# Before, during and after the grid, the expiry and the cutoff; once the cutoff has passed, as often as before it.
WHEN = [T0 + timedelta(hours=12 * k) for k in range(19)] + [CUTOFF + timedelta(days=k) for k in range(1, 18)]


# ---------------------------------------------------------------- random contracts


@st.composite
def atoms(draw, ids, *, decidable: bool):
    """One atomic condition. `decidable`: every comparison is one its inputs can decide once known (a reading compared
    at its bands' edges with < or >=, as a band [low, high) decides; a node over published values only)."""
    n = next(ids)
    o = draw(st.sampled_from(["x", "y"]))
    cmp = c.Cmp(id=f"c{n}", observable=o, op=draw(st.sampled_from(OPS)), value=c.Number(value=D(draw(st.integers(0, 10)))))
    i = draw(st.integers(0, 5))
    j = draw(st.integers(i, 5))
    kind = draw(st.sampled_from(["at", "by", "throughout", "count", "duration", "value_at", "interp", "read", "computed"]))
    if kind == "at":
        return c.At(id=f"a{n}", expr=cmp, at=GRID[i])
    if kind == "by":
        since = draw(st.booleans())
        closed = draw(st.booleans())
        if decidable:  # an instant of the grid within its bounds, so that some observation establishes it
            since_closed = True
            closed = closed if j > i else True
        else:
            since_closed = draw(st.booleans())
        return c.By(id=f"b{n}", expr=cmp, since=GRID[i] if since else None, since_closed=since_closed if since else None,
                    deadline=GRID[j], closed=closed)  # fmt: skip
    if kind == "throughout":
        start_closed, end_closed = (True, True) if decidable else (draw(st.booleans()), draw(st.booleans()))
        return c.Throughout(id=f"t{n}", expr=cmp, window=c.TimeWindow(start=GRID[i], end=GRID[j],
                            start_closed=start_closed, end_closed=end_closed))  # fmt: skip
    window = c.TimeWindow(start=GRID[i], end=GRID[j] + timedelta(days=1) if decidable else GRID[j], start_closed=True,
                          end_closed=False)  # fmt: skip
    if kind == "count":
        return c.NumCmp(id=f"n{n}", expr=c.CountTrue(id=f"k{n}", expr=cmp, days=window), op=draw(st.sampled_from(OPS)),
                        value=D(draw(st.integers(-1, 7))))  # fmt: skip
    if kind == "duration":
        return c.NumCmp(id=f"n{n}", expr=c.DurationIn(id=f"d{n}", expr=cmp, window=window, unit="hour"),
                        op=draw(st.sampled_from(OPS)), value=D(draw(st.sampled_from([0, 12, 24, 36, 48, 120]))))  # fmt: skip
    if kind == "value_at":
        return c.NumCmp(id=f"n{n}", expr=c.ValueAt(id=f"v{n}", observable=o, at=GRID[i]), op=draw(st.sampled_from(OPS)),
                        value=D(draw(st.integers(0, 10))))  # fmt: skip
    if kind == "interp":
        return c.Interpretive(id=f"q{n}", question_template=f"t{n}", bindings={})
    read = c.ReadValue(id=f"r{n}", question_template=f"r{n}", bindings={}, unit="u", bands=BANDS)
    if kind == "read":
        if decidable:
            return c.NumCmp(id=f"n{n}", expr=read, op=draw(st.sampled_from(["<", ">="])),
                            value=D(draw(st.sampled_from([-5, 0, 10, 20, 35]))))  # fmt: skip
        return c.NumCmp(id=f"n{n}", expr=read, op=draw(st.sampled_from(OPS)), value=D(draw(st.integers(-5, 35))))
    other = c.ValueAt(id=f"w{n}", observable=draw(st.sampled_from(["x", "y"])), at=GRID[j])
    a = c.ValueAt(id=f"u{n}", observable=o, at=GRID[i]) if decidable else read
    node = c.Computed(id=f"f{n}", node=c.Ref(id="tests.contract.difference", version=1), inputs={"a": a, "b": other})
    return c.NumCmp(id=f"n{n}", expr=node, op=draw(st.sampled_from(OPS)), value=D(draw(st.integers(-10, 30))))


@st.composite
def triggers(draw, ids, depth=0, *, decidable=False):
    shape = draw(st.sampled_from(["atom"] * 3 + (["not", "and", "or"] if depth < 2 else [])))
    if shape == "atom":
        return draw(atoms(ids, decidable=decidable))
    if shape == "not":
        return c.Not(id=f"not{next(ids)}", arg=draw(triggers(ids, depth + 1, decidable=decidable)))
    args = draw(st.lists(triggers(ids, depth + 1, decidable=decidable), min_size=2, max_size=3))
    return (c.And if shape == "and" else c.Or)(id=f"{shape}{next(ids)}", args=args)


@st.composite
def contracts(draw, *, decidable=False):
    trigger = draw(triggers(count(), decidable=decidable))
    early = draw(st.sampled_from(["none", "first_true", "per_exchange_rule"]))
    proc = procedure(**PROCEDURE, **({} if early == "none" else {
        "early_expiration": "when_true", "early_expiration_at": early,
        **({"early_expiration_rule": "Rule 7.2"} if early == "per_exchange_rule" else {})}))  # fmt: skip
    return c.ContractIR(
        id="random", version=1, observables=OBSERVABLES, trigger=trigger,
        payout=c.Payout(kind="binary", unit_value=D("1.00"), currency="USD", rounding_quantum=D("0.01"),
                        rounding_mode="half_even", payment="at_settlement"),
        timing=timing(expiration=EXPIRY, close=EXPIRY, observation_cutoff=CUTOFF, tz="UTC"), resolution=proc,
    )  # fmt: skip


def hidden_nodes(ir):
    nodes = list(c.walk(ir.trigger))
    return [n for n in nodes if isinstance(n, c.Interpretive)], [n for n in nodes if isinstance(n, c.ReadValue)]


@st.composite
def worlds(draw, ir, *, complete=False):
    """A full world (every selected value known) and a partial one that hides some of it, with the same instants."""
    # Each observable at each grid instant (the instants At and ValueAt read), and some at noon between them, so that
    # a day can have two values and a stretch can be split by a value published late.
    noon = timedelta(hours=12)
    slots = [(o, GRID[i]) for o in ("x", "y") for i in range(6)] + [(o, GRID[i] + noon) for o in ("x", "y")
                                                                      for i in range(6)]  # fmt: skip
    if complete:
        present = slots[:12]
    else:
        present = [s for s in slots[:12] if draw(st.integers(0, 4))] + [s for s in slots[12:] if draw(st.booleans())]
    values = {s: draw(st.integers(0, 10)) for s in present}
    offsets = [0, 0, 0, 6, 30, 200]  # hours after the instant: mostly at once, some late, some after the cutoff
    published = {s: s[1] + (timedelta(0) if complete else timedelta(hours=draw(st.sampled_from(offsets))))
                 for s in present}  # fmt: skip
    hide = {s for s in present if draw(st.booleans())}
    full, partial = [], []
    for o, when in present:
        v, p, hidden = values[(o, when)], published[(o, when)], (o, when) in hide
        if o == "x":
            full.append(obs(v, when, p, oid="x", src="a"))
            partial.append(obs(draw(st.integers(0, 10)) if hidden else v, when, p, oid="x", src="a",
                               final=not hidden))  # a non-final value: unknown under "final"  # fmt: skip
        else:
            full += [obs(v, when, p, oid="y", src="a"), obs(v, when, p, oid="y", src="b")]
            partial += [obs(v, when, p, oid="y", src="a")] + ([] if hidden else [obs(v, when, p, oid="y", src="b")])
    interps, reads = hidden_nodes(ir)
    given_full, given_partial = {}, {}
    for q in interps:
        given_full[q.id] = draw(st.sampled_from([Kleene.TRUE, Kleene.FALSE]))
        given_partial[q.id] = Kleene.UNKNOWN if draw(st.booleans()) else given_full[q.id]
    for r in reads:
        k = draw(st.integers(0, 2))
        given_full[r.id] = BANDS[k]
        coarse = draw(st.sampled_from(["same", "run", "whole"]))
        lo, hi = draw(st.integers(0, k)), draw(st.integers(k, 2))
        given_partial[r.id] = {"same": BANDS[k], "run": (BANDS[lo][0], BANDS[hi][1]), "whole": (D(0), D(30))}[coarse]
    return (full, given_full), (partial, given_partial)


def grid_contract(trigger, **timing_fields):
    """A binary contract over the grid's observables, expiring after the grid and closing its instants a day later."""
    fields = {"expiration": EXPIRY, "close": EXPIRY, "observation_cutoff": CUTOFF, "tz": "UTC", **timing_fields}
    return c.ContractIR(id="k", version=1, observables=OBSERVABLES, trigger=trigger,
                        payout=c.Payout(kind="binary", unit_value=D(1), currency="USD", rounding_quantum=D("0.01"),
                                        rounding_mode="half_even", payment="at_settlement"),
                        timing=timing(**fields), resolution=procedure(**PROCEDURE))  # fmt: skip
