"""The IR's conditions stated in words by code, for the rendering check and the scenario renderer (spec §8.6 steps 3-4).

The rendering check asks whether a packet establishes each atomic condition the scenario fixes. Its question states the
condition as the IR means it, written here from the IR's structure: the comparison, the instant or window with each end
included or not, and how the contract selects the values it reads (its sources' order or required agreement, which of a
source's vintages counts, and its observation cutoff). That is state rendered into words by code, never words read into
state (design/posture.md: words become state only through a decision model). It is not the contract's prose: the
prose is the clause drafter's, and the resolvers read only the prose, so the rendering check and the resolution test
different things (spec §8.6: the check recovers the scenario from the packet; the resolution tests the prose on it).

An Interpretive condition asks its own library template's question where the library holds it; where it does not, its
question is the condition_holds template over the node's template name and bindings, the words the contract reader wrote
for it (requisite/agents/prompts/contract_reader.md: "a question template name and its bindings, stating the condition
in the filing's words"; the clause drafter states it the same way). A reading (ReadValue) asks for its band among its
own closed set of bands.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from typing import Any

from requisite import core

Sources = tuple[Mapping[str, tuple[str, ...]], Mapping[str, tuple[str, ...]]]  # consulted, must agree (the evaluator's)

OP_WORDS = {"<": "less than", "<=": "at most", "==": "equal to", "!=": "not equal to", ">=": "at least", ">": "greater than"}


def iso(t: datetime) -> str:
    return t.isoformat()


def _end(t: datetime | None, closed: bool | None, side: str) -> str:
    if t is None:
        return "with no lower bound" if side == "from" else "with no upper bound"
    return f"{side} {iso(t)} ({'included' if closed else 'excluded'})"


def window(w: core.TimeWindow) -> str:
    return f"{_end(w.start, w.start_closed, 'from')} {_end(w.end, w.end_closed, 'to')}, days counted in {w.tz}"


def value(v: core.CmpValue, obs: core.Observable | None) -> str:
    if isinstance(v, core.Number):
        return f"{v.value} {obs.unit}" if obs is not None and obs.unit else str(v.value)
    if isinstance(v, core.Flag):
        return "true" if v.value else "false"
    if isinstance(v, core.OnDay):
        return v.value.isoformat()
    return f'"{v.value}"'


def bindings(b: Mapping[str, Any]) -> str:
    return "; ".join(f"{k}: {b[k]}" for k in sorted(b))


def _observables(ir: core.ContractIR) -> dict[str, core.Observable]:
    return {o.id: o for o in ir.observables}


def comparison(ir: core.ContractIR, e: Any) -> str:
    """A boolean expression over values at one instant (a Cmp, or And, Or and Not over Cmps)."""
    obs = _observables(ir)
    if isinstance(e, core.Cmp):
        o = obs[e.observable]
        if isinstance(e.value, core.Flag | core.Category) and e.op in ("==", "!="):
            verb = "is" if e.op == "==" else "is not"
            return f"{o.description} [{o.id}] {verb} {value(e.value, o)}"
        return f"{o.description} [{o.id}] is {OP_WORDS[e.op]} {value(e.value, o)}"
    if isinstance(e, core.And):
        return " and ".join(f"({comparison(ir, a)})" for a in e.args)
    if isinstance(e, core.Or):
        return "at least one of: " + "; ".join(f"({comparison(ir, a)})" for a in e.args)
    if isinstance(e, core.Not):
        return f"it is not the case that ({comparison(ir, e.arg)})"
    raise TypeError(f"{getattr(e, 'id', e)!r} is not compared at an instant")


def number(ir: core.ContractIR, e: Any) -> str:
    """A numeric expression of the IR."""
    obs = _observables(ir)
    if isinstance(e, core.ValueAt):
        o = obs[e.observable]
        return f"the value of {o.description} [{o.id}] for {iso(e.at)}, in {o.unit}"
    if isinstance(e, core.ReadValue):
        return f"the number {e.question_template} reads ({bindings(e.bindings)}), in {e.unit}"
    if isinstance(e, core.CountTrue):
        return (f"the number of calendar days wholly within the window {window(e.days)} on which, at some instant of "
                f"the day for which a value is published, {comparison(ir, e.expr)}")  # fmt: skip
    if isinstance(e, core.DurationIn):
        unit = {"calendar_day": "calendar days", "elapsed_day": "periods of 24 hours", "hour": "hours"}[e.unit]
        return (f"the time, in {unit}, within the window {window(e.window)} during which {comparison(ir, e.expr)}, "
                "each published value holding from its instant until the next")  # fmt: skip
    if isinstance(e, core.Computed):
        parts = "; ".join(f"{k} = {number(ir, v)}" for k, v in sorted(e.inputs.items()))
        return f"the value computed by {e.node.id} (version {e.node.version}) from {parts}"
    raise TypeError(f"{getattr(e, 'id', e)!r} is not a number")


def condition(ir: core.ContractIR, node: Any) -> str:
    """An atomic condition as the IR means it (an Interpretive's statement is its template name and bindings)."""
    if isinstance(node, core.At):
        return f"For the instant {iso(node.at)}: {comparison(ir, node.expr)}."
    if isinstance(node, core.By):
        since = _end(node.since, node.since_closed, "from")
        return (f"At some instant {since} {_end(node.deadline, node.closed, 'up to')} for which a value is published: "
                f"{comparison(ir, node.expr)}.")  # fmt: skip
    if isinstance(node, core.Throughout):
        return (f"At every instant {_end(node.window.start, node.window.start_closed, 'from')} "
                f"{_end(node.window.end, node.window.end_closed, 'to')} for which a value is published, and a value is "
                f"published for at least one such instant: {comparison(ir, node.expr)}.")  # fmt: skip
    if isinstance(node, core.NumCmp):
        return f"{number(ir, node.expr)} is {OP_WORDS[node.op]} {node.value}."
    if isinstance(node, core.Interpretive):
        return f"{node.question_template}: {bindings(node.bindings)}."
    raise TypeError(f"{getattr(node, 'id', node)!r} is not an atomic condition")


def read_observables(node: Any) -> list[str]:
    """The observables a node reads, in order of first appearance."""
    return list(dict.fromkeys(n.observable for n in core.walk(node) if isinstance(n, core.Cmp | core.ValueAt)))


def policy(ir: core.ContractIR, observable: str, sources: Sources) -> str:
    """How the contract selects the values of one observable it reads (spec §4 Observable; A05 §5.4), with the sources
    the procedure consults for it, as the evaluator consults them (scenarios.sources)."""
    o = _observables(ir)[observable]
    consulted, agree = sources
    names = {s.id: s for s in o.sources}
    shown = [f"{names[s].name} [{s}]" for s in consulted[observable]]
    if observable in agree:
        chosen = f"the sources {', '.join(shown)} must agree, or the value is disputed"
    elif len(shown) > 1:
        chosen = f"the value of the highest-ranked of {', '.join(shown)} (in that order) that has published one"
    else:
        chosen = f"the value {shown[0]} publishes"
    t = ir.timing
    if o.revision_policy == "first_print":
        vintage = "the first published value counts"
    elif o.revision_policy == "as_of_expiry":
        vintage = f"the latest value published by the expiration, {iso(t.expiration)}, counts"
    else:
        by = f" by {iso(o.final_deadline)}" if o.final_deadline is not None else ""
        fallback = {None: "", "first_print": "; if none is final by then, the first published value counts",
                    "as_of_expiry": f"; if none is final by then, the latest published by {iso(t.expiration)} counts"}  # fmt: skip
        vintage = f"the value the source marks final{by} counts{fallback[o.final_fallback]}"
    unit = f", in {o.unit}, published to a precision of {o.precision}" if o.value_type == "number" else ""
    return (f"Values of {o.description} [{o.id}]{unit}: for each instant, {chosen}; of one source's values for an "
            f"instant, {vintage}; values published after {iso(t.observation_cutoff)} are not considered.")  # fmt: skip


def checked(ir: core.ContractIR, node: Any, sources: Sources) -> str:
    """A condition with the selection policy of each observable it reads, as the rendering check asks it."""
    return " ".join([condition(ir, node), *(policy(ir, o, sources) for o in read_observables(node))])


def published_value(ir: core.ContractIR, observable: str, at: datetime, v: core.CmpValue, sources: Sources) -> str:
    """A published value as a condition, for a computed comparison's inputs: the value the contract reads for an instant."""
    o = _observables(ir)[observable]
    return (f"The value of {o.description} [{o.id}] for the instant {iso(at)} is {value(v, o)}. "
            f"{policy(ir, observable, sources)}")  # fmt: skip


def band(lo: Decimal, hi: Decimal, last: bool, unit: str) -> str:
    return f"at least {lo} and {'at most' if last else 'less than'} {hi} {unit}"


def _money(amount: Decimal, currency: str) -> str:
    return f"{amount.normalize():f} {currency}"


def payment_terms(ir: core.ContractIR) -> dict[str, str]:
    """The IR's payment terms, each as one statement a reader can check against the prose (core.TermReading): what one
    contract pays when it resolves YES and when it resolves NO, when it is paid, and what a void returns. Stated from
    the IR's own fields."""
    p, t, v = ir.payout, ir.timing, ir.resolution.void_transfer
    if p.kind == "binary":
        pays = f"when it resolves YES, one contract pays {_money(p.unit_value, p.currency)}"
    else:
        assert p.quantity is not None
        each = f"{_money(p.unit_value, p.currency)} per unit of {number(ir, p.quantity)}"
        cap = f", at most {_money(p.cap, p.currency)}" if p.cap is not None else ""
        pays = f"when it resolves YES, one contract pays {each}{cap}"
    if p.payment == "per_period":
        pays += ", paid per period: " + "; ".join(window(w) for w in p.periods)
    nothing = "when it resolves NO, one contract pays nothing"
    days = "business days (United States federal holidays excluded)" if t.lag_days == "business" else "calendar days"
    when = (f"the outcome is determined {t.settlement_lag_days} {days} after expiration ({iso(t.expiration)}) and payment "
            f"is made {t.payment_lag_days} {days} after that, in the {t.tz} time zone")
    out = {"payout_yes": pays, "payout_no": nothing, "payment_date": when}
    if ir.resolution.on_unknown_at_expiry == "void" or ir.resolution.after_extension == "void":
        refund = {"premium": "the premium paid is returned", "nothing": "nothing is returned",
                  "fixed_amount": f"{_money(v.amount, p.currency) if v.amount is not None else ''} is returned"}[v.returns]
        out["void_refund"] = f"if the contract is void, {refund}"
    return out


def reading_question(node: core.ReadValue, question_id: str) -> core.AtomicQuestion:
    """A reading's own question: its template name and bindings, answered as one of its bands (core.ReadValue: "Decision
    models answer it as a closed set of bands"), or not established."""
    last = len(node.bands) - 1
    answers = [
        core.AnswerValue(value=f"band{i}", meaning=band(lo, hi, i == last, node.unit), status="affirmative")
        for i, (lo, hi) in enumerate(node.bands)
    ]
    answers.append(core.AnswerValue(value="not_established", status="not_established",
                                    meaning="the documents do not establish the number"))  # fmt: skip
    return core.AtomicQuestion(
        id=question_id, version=1, task="classify",
        target=(f"According to the documents under OBSERVED, what number does the following read, in {node.unit}?\n"
                f"Reading: {node.question_template}: {bindings(node.bindings)}"),
        answers=answers,
    )  # fmt: skip
