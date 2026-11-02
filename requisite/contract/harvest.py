"""A filed series read into the contract IR, and checked against how its markets settled (spec §9.3, §12 F1, §13 WP12;
addendum A05 §5.5, A09 §10.5).

A Kalshi series is one set of filed terms listed as many markets that differ only in their listing fields (a strike, the
times, the event), so it is read as an instrument family:
1. Its filed documents (the contract terms and the product certification) are read through requisite/evidence/extract.py
   and the extractions committed. Its markets' records are the ones F1 committed (core.FiledSeries). Nothing committed is
   fetched again.
2. The contract_reader agent proposes the family (agents/answers.HarvestDraft): the support classification, with its
   assumptions and unsupported terms in the filing's words; one representative market's IR (the first harvested market
   in ticker order); the bindings, which say which IR value each per-market listing field fills; and what a market
   record's settlement value reports. The agent reads the filings and the markets' listing fields (kalshi.LISTING_FIELDS),
   never an outcome or a price, so the check below is not one it could have fitted.
3. Code gives each market its IR (`instantiate`: the representative with each binding's slot filled from the market's
   record, a pure function of the two) and submits the family as a Proposal of kind "contract" (core.ContractFamily),
   whose gates the store runs:
   - schema (`schema_check`, after the record validates): the family covers exactly the harvested markets; its bindings
     read listing fields that every market's record carries, in their slots' types; the representative is what its
     bindings make of its own market's record; each market IR re-derives from the representative and its committed
     record; and the evaluator accepts every IR (evaluate.accept: no IRError).
   - tests (`tests_check`): `check`, below. An unsupported series has nothing to test, so its tests gate does not pass,
     and its rejected proposal stands as the record of the finding.
   When both pass, the HarvestedContract and each market IR are committed.

The check. For each market whose record shows it settled by the draw's instant, the evaluator settles the market's IR on
the settlement value its record states (expiration_value, "the value that was considered for the settlement"): an
Observation of the observable the bindings name, from that observable's highest-precedence consulted source, for the
instant the bindings name, with the market record as its artifact (the exchange's statement of the source's value). The
market reproduces when the IR's procedure status is the record's result (yes: resolved_yes, no: resolved_no, scalar:
either) and the payout the evaluation states is the record's settlement value. A market whose IR has interpretive
conditions or readings cannot be checked from its record, which states no resolver's judgment, and neither can one whose
record does not state a value of the observable's type; each is named with why. The gate passes when at least one
market was checked and every checked market reproduces, and the family's equivalence (A09 §10.5) is the one the check
gives: tested_equivalent_on_suite, counterexample_found, or unsupported_or_unassessed where nothing was checked.

Two choices, both so that the check tests what its record can speak to and nothing else (ours):
- The observation is taken as published at the instant it is for. The record states the value the exchange settled on,
  not when the source published it, and its own date, the settlement time, comes after the expiration (the exchange
  settles after it: "The Settlement Date of the Contract shall be no later than the day after the Expiration Date"), so
  dating the value by it would put every value after the observation cutoff of an IR that reads the terms' "the value
  of the Underlying as documented by the Source Agency on the Expiration Date at the Expiration time" exactly.
- The IR is settled at its final instant (evaluate.final_instant: after its expiration and observation cutoff, and any
  extension), never at the settlement time. The exchange settles some markets before their filed expiration by a rule
  outside the filing (the terms' "expiration will be moved to an earlier date and time in accordance with Rule 7.2"),
  and an observed settlement is a procedural outcome that may rest on decisions the filed terms do not entail (A09
  §10.5). So the check compares outcomes, not dates: neither when the exchange settled nor when the IR pays.
Whether a checked market lay at or next to its comparison's boundary is found by the evaluator too: a market lies at it
when making its comparison inclusive instead of strict (or the reverse) changes its outcome, which only a settlement
value equal to the threshold does; next to it when filling its strike from the adjacent market's instead changes its
outcome. Only such markets discriminate an inclusive bound or an off-by-one strike, and the detail says whether any did.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from requisite import core
from requisite.agents import runner
from requisite.agents.answers import HarvestDraft
from requisite.agents.trace import made_from, record_run
from requisite.contract.evaluate import IRError, accept, consulted, evaluate, final_instant
from requisite.core import (
    Category,
    Cmp,
    CmpValue,
    ContractFamily,
    ContractIR,
    CountTrue,
    FiledSeries,
    GateResult,
    HarvestedContract,
    Interpretive,
    ListingBinding,
    Number,
    NumCmp,
    Observable,
    Observation,
    OnDay,
    Proposal,
    ReadValue,
    ResolutionEvaluation,
    walk,
)
from requisite.core.contract import TIMING_SLOTS
from requisite.evidence import extract
from requisite.market import f1
from requisite.market.kalshi import LISTING_FIELDS, Settled, instant, listing
from requisite.store import Refused, Store

ROLE = "contract_reader"
EXAMPLES = 5  # a field that varies across a series' markets is shown at up to this many markets, spread over them
ORDERED = {"<": "<=", "<=": "<", ">": ">=", ">=": ">"}  # each ordered comparison and its strict or inclusive twin
SHOWN = 10  # markets named in a sentence before "and N more"

Agent = Callable[..., runner.AgentResult[HarvestDraft]]


class BindingError(ValueError):
    """A binding that cannot fill its slot from a market's record."""


# ---------------------------------------------------------------- what is committed


def records(store: Store, filed: FiledSeries) -> dict[str, dict[str, Any]]:
    """Each harvested market's record, by ticker, its numbers read exactly (a strike as a Decimal, never a float)."""
    return {t: json.loads(store.content(a), parse_float=Decimal)["market"] for t, a in sorted(filed.markets.items())}


def filed_documents(filed: FiledSeries) -> list[tuple[str, str]]:
    """The filed documents read, terms first: (what it is, artifact)."""
    out = [("contract terms", filed.terms)]
    if filed.contract is not None:
        out.append(("product certification", filed.contract))
    return out


def read_filings(store: Store, filed: FiledSeries, *, root: Path | None = None) -> list[dict[str, Any]]:
    """The filed documents' text by the evidence package's PDF reader, each extraction committed; what could not be read
    is stated with why, never dropped."""
    out = []
    for what, aid in filed_documents(filed):
        artifact = store.get("artifact", aid, core.Artifact)
        assert artifact is not None  # core.FiledSeries names committed artifacts (f1.retrieve)
        x = extract.extract("pdf", aid, store.content(aid), root=root)
        if x.id is not None and x.text is not None:
            store.commit_extraction(aid, x.id, x.text)
        out.append({"document": what, "url": artifact.locator, "artifact": aid, "extraction": x.id, "text": x.text,
                    "unreadable": list(x.unreadable)})  # fmt: skip
    return out


# ---------------------------------------------------------------- the agent's reading


def agent_input(series: Mapping[str, Any], filings: Sequence[Mapping[str, Any]],
                markets: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:  # fmt: skip
    """What the contract_reader reads: the series record, the filings' text, the representative market's listing fields,
    and how the listing fields vary over the series' markets (each varying field's number of distinct values and its
    values at up to EXAMPLES markets spread evenly over ticker order, the representative first). Market records as
    served (a strike as the JSON number it is), never an outcome or a price."""
    order = sorted(markets)
    listed = {t: listing(markets[t]) for t in order}
    n = len(order)
    spread = order if n <= EXAMPLES else [order[round(i * (n - 1) / (EXAMPLES - 1))] for i in range(EXAMPLES)]
    varying = {}
    for k in LISTING_FIELDS:
        values = {core.canonical_json(listed[t].get(k)) for t in order}
        if len(values) > 1:
            varying[k] = {"distinct": len(values), "examples": {t: listed[t].get(k) for t in dict.fromkeys(spread)}}
    return {
        "series": dict(series),
        "filings": [{k: f[k] for k in ("document", "url", "text", "unreadable")} for f in filings],
        "representative_market": listed[order[0]],
        "markets": {"count": n, "tickers": order, "varying_fields": varying},
        "listing_fields": list(LISTING_FIELDS),
    }


# ---------------------------------------------------------------- instantiation


def _field(record: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return False, None
        value = value[part]
    return True, value


def _set_field(record: Mapping[str, Any], path: str, value: Any) -> dict[str, Any]:
    """A copy of the record with the field at `path` replaced."""
    head, _, rest = path.partition(".")
    out = dict(record)
    out[head] = _set_field(record[head], rest, value) if rest else value
    return out


def _instant(raw: Any, what: str) -> str:
    if not isinstance(raw, str):
        raise BindingError(f"{what} is {raw!r}, not an instant")
    try:
        at = instant(raw)
    except ValueError as e:
        raise BindingError(f"{what} is {raw!r}, not an instant") from e
    if at.utcoffset() is None:
        raise BindingError(f"{what} is {raw!r}, an instant with no time zone")
    return at.isoformat()


def _number(raw: Any, what: str) -> Decimal:
    if isinstance(raw, bool) or not isinstance(raw, int | Decimal | str):
        raise BindingError(f"{what} is {raw!r}, not a number")
    try:
        value = Decimal(raw)
    except InvalidOperation as e:
        raise BindingError(f"{what} is {raw!r}, not a number") from e
    if not value.is_finite():
        raise BindingError(f"{what} is {raw!r}, not a finite number")
    return value


def _threshold(node: Any, raw: Any, what: str) -> Any:
    """A comparison's value from a field, in the comparison's type, as the IR's JSON form."""
    if isinstance(node, NumCmp):
        return str(_number(raw, what))
    kind = node.value.kind
    if kind == "number":
        return {"kind": "number", "value": str(_number(raw, what))}
    if kind == "text":
        if not isinstance(raw, str):
            raise BindingError(f"{what} is {raw!r}, not a text")
        return {"kind": "text", "value": raw}
    if kind == "bool":
        if not isinstance(raw, bool):
            raise BindingError(f"{what} is {raw!r}, not a truth")
        return {"kind": "bool", "value": raw}
    try:
        return {"kind": "date", "value": date.fromisoformat(raw).isoformat()}
    except (TypeError, ValueError) as e:
        raise BindingError(f"{what} is {raw!r}, not a date") from e


def _json_nodes(tree: Any) -> Iterator[dict[str, Any]]:
    """Every node of an expression in its JSON form (a mapping with a kind and an id)."""
    if isinstance(tree, dict):
        if "kind" in tree and "id" in tree:
            yield tree
        for v in tree.values():
            yield from _json_nodes(v)
    elif isinstance(tree, list):
        for v in tree:
            yield from _json_nodes(v)


def instantiate(representative: ContractIR, bindings: Sequence[ListingBinding], record: Mapping[str, Any]) -> ContractIR:
    """A market's IR: the representative with each binding's slot filled from the market's record, the field's value in
    the slot's type, and the market's ticker as its id. A pure function of the IR, the bindings and the record; a binding
    that cannot fill its slot raises BindingError."""
    ticker = record.get("ticker")
    if not isinstance(ticker, str):
        raise BindingError("a market record names its ticker")
    if representative.examples:
        raise BindingError("the representative carries examples: a market IR's examples are its evaluator's outcomes "
                           "for fuzzed scenarios (spec §8.5, I8), which no other market's IR inherits")  # fmt: skip
    data = representative.model_dump(mode="json")
    data["id"] = ticker
    typed = {n.id: n for e in (representative.trigger, representative.payout.quantity) if e is not None for n in walk(e)}
    trees = [data["trigger"], data["payout"]["quantity"]]
    for b in bindings:
        if b.field.split(".")[0] not in LISTING_FIELDS:
            raise BindingError(f"{b.field} is not a listing field: a binding reads the terms a market was listed under "
                               "(kalshi.LISTING_FIELDS), never its outcome or prices")  # fmt: skip
        found, raw = _field(record, b.field)
        what = f"{ticker}'s {b.field}"
        if not found or raw is None:
            raise BindingError(f"{ticker}'s record carries no {b.field}")
        if b.node is None:
            data["timing"][b.slot] = _instant(raw, what)
            continue
        node = typed.get(b.node)
        if node is None:
            raise BindingError(f"{b.field} fills {b.slot} of {b.node}, which is not a node of the IR")
        if b.slot == "threshold":
            value = _threshold(node, raw, what)
        else:
            value = _instant(raw, what)
        for j in (n for t in trees for n in _json_nodes(t) if n["id"] == b.node):
            if b.slot == "threshold":
                j["value"] = value
            elif b.slot in ("at", "since", "deadline"):
                j[b.slot] = value
            else:
                j["days" if isinstance(node, CountTrue) else "window"]["start" if b.slot == "window_start" else "end"] = value
    try:
        return ContractIR.model_validate_json(core.canonical_json(data))
    except ValidationError as e:
        raise BindingError(f"{ticker}: its bound values make no valid IR: {e}") from e


# ---------------------------------------------------------------- the flow step


def _proposals(store: Store, filing: str) -> list[Proposal]:
    return [p for p in store.all("proposal", Proposal)
            if p.kind == "contract" and p.payload.get("contract", {}).get("filing") == filing]  # fmt: skip


def family_of(store: Store, filed: FiledSeries, draft: HarvestDraft, version: int) -> ContractFamily:
    """The family a reading proposes: code names the representative (the first harvested market) and the version, gives
    each market its IR, and states the equivalence the check gives (A09 §10.5), which the tests gate checks again."""
    rep = None if draft.ir is None else ContractIR.model_validate_json(core.canonical_json(
        {**draft.ir.model_dump(mode="json"), "id": min(filed.markets), "version": version}))  # fmt: skip
    contract = HarvestedContract(
        filing=filed.key, filed_terms=[aid for _, aid in filed_documents(filed)], support=draft.support,
        ir=rep.ref if rep is not None else None, assumptions=draft.assumptions,
        unsupported_terms=draft.unsupported_terms, equivalence="unsupported_or_unassessed",
    )  # fmt: skip
    if rep is None:
        return ContractFamily(contract=contract, representative=None)
    markets = []
    for record in records(store, filed).values():
        try:
            markets.append(instantiate(rep, draft.bindings, record))
        except BindingError:
            continue  # the schema gate re-derives every market's IR and says why this one has none
    family = ContractFamily(contract=contract, representative=rep, bindings=draft.bindings, settlement=draft.settlement,
                            markets=markets)  # fmt: skip
    if _problems(store, filed, family):
        return family  # the schema gate refuses it, saying why; nothing is assessed
    equivalence = check(store, filed, family, boundary=False).equivalence
    contract = HarvestedContract.model_validate({**contract.model_dump(), "equivalence": equivalence})
    return ContractFamily.model_validate({**family.model_dump(), "contract": contract})


def read(store: Store, series: str, *, as_of: datetime | None = None, agent: Agent | None = None,
         root: Path | None = None) -> Proposal:  # fmt: skip
    """The series read into the IR by the contract_reader (`agent`, by default runner.run) and submitted as a contract
    proposal, whose gates run here; when they pass, the HarvestedContract and each market IR are committed."""
    agent = agent or runner.run
    register(store)
    filed = f1.filed_series(store, series, as_of)
    if not filed.markets:
        raise Refused(f"{filed.key}: no market of the series is harvested, so there is none to read the family from")
    filings = read_filings(store, filed, root=root)
    served = {t: json.loads(store.content(a))["market"] for t, a in sorted(filed.markets.items())}
    input = agent_input(json.loads(store.content(filed.record))["series"], filings, served)
    result = agent(ROLE, input, HarvestDraft, effort="high", root=root)
    if done := made_from(store, result):
        return done[0]  # this reading is already proposed, and its gates have run (agents/trace.py)
    draft = result.output
    family = family_of(store, filed, draft, 1 + len(_proposals(store, filed.key)))
    payload = family.model_dump(mode="json")
    lists = [("assumptions", draft.assumptions), ("unsupported terms", draft.unsupported_terms)]
    rationale = f"{series}: {draft.support}. " + " ".join(f"{name}: {'; '.join(xs) if xs else 'none'}."
                                                          for name, xs in lists)  # fmt: skip
    proposal = store.submit_proposal(Proposal(
        id=f"contract-{core.sha256(core.canonical_json(payload))[:16]}", case=store.case, kind="contract",
        payload=payload, author=result.run[:16], rationale=rationale, status="proposed",
    ))  # fmt: skip
    record_run(store, result, input, [proposal.id])
    if proposal.status == "active":
        commit(store, family)
    return proposal


def commit(store: Store, family: ContractFamily) -> None:
    """An active family's records: its HarvestedContract and each market IR."""
    assert family.representative is not None  # an unsupported family never passes its tests gate
    store.commit("harvested_contract", f"{family.contract.filing}|v{family.representative.version}", family.contract)
    for ir in family.markets:
        store.commit("contract", f"{ir.id}|{ir.version}", ir)


# ---------------------------------------------------------------- the gates


def register(store: Store) -> None:
    store.register_gate("contract", "schema", schema_check)
    store.register_gate("contract", "tests", tests_check)


def _family(store: Store, proposal: Proposal) -> tuple[ContractFamily, FiledSeries | None]:
    family = ContractFamily.model_validate_json(core.canonical_json(dict(proposal.payload)))
    return family, store.get("filed_series", family.contract.filing, FiledSeries)


def _grouped(problems: Sequence[tuple[str, str]]) -> list[str]:
    """Problems that recur across markets, each once with the markets it holds for."""
    by: dict[str, list[str]] = {}
    for ticker, text in problems:
        by.setdefault(text, []).append(ticker)
    return [f"{text} ({_names(tickers)})" if tickers != [""] else text for text, tickers in by.items()]


def _names(tickers: Sequence[str]) -> str:
    shown = ", ".join(tickers[:SHOWN])
    return shown if len(tickers) <= SHOWN else f"{shown} and {len(tickers) - SHOWN} more"


def _difference(a: Any, b: Any, path: str = "") -> str:
    """The first place two IRs differ, by value as their equality compares them (a number by its value, an instant as
    an instant)."""
    if isinstance(a, core.Record) and type(a) is type(b):
        for k in type(a).model_fields:
            if getattr(a, k) != getattr(b, k):
                return _difference(getattr(a, k), getattr(b, k), f"{path}.{k}" if path else k)
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                return _difference(a.get(k), b.get(k), f"{path}.{k}" if path else str(k))
    if isinstance(a, tuple | list) and isinstance(b, tuple | list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            if x != y:
                return _difference(x, y, f"{path}[{i}]")

    def show(v: Any) -> str:
        return v.isoformat() if isinstance(v, datetime | date) else v.canonical() if isinstance(v, core.Record) else str(v)

    return f"{path}: {show(a)} in the IR, {show(b)} from its record"


def schema_check(store: Store, proposal: Proposal) -> GateResult:
    """A contract family's schema beyond its record (module docstring)."""
    family, filed = _family(store, proposal)
    if filed is None:
        return GateResult(gate="schema", passed=False, detail=f"{family.contract.filing} names no committed harvest")
    problems = _problems(store, filed, family)
    if problems:
        return GateResult(gate="schema", passed=False, detail="; ".join(_grouped(problems)))
    if family.representative is None:
        return GateResult(gate="schema", passed=True, detail="a valid ContractFamily: unsupported, with no IR, naming "
                          f"its unsupported terms: {'; '.join(family.contract.unsupported_terms)}")  # fmt: skip
    return GateResult(gate="schema", passed=True, detail=(
        f"a valid ContractFamily of {len(family.markets)} market IRs: each is what the representative and its bindings "
        "make of the market's committed record, the bindings read listing fields every market carries, and the "
        "evaluator accepts every IR"))  # fmt: skip


def _problems(store: Store, filed: FiledSeries, family: ContractFamily) -> list[tuple[str, str]]:
    """What keeps a family from being the harvest's, read as its bindings say: each with the market it holds for, or
    "" for the family as a whole."""
    problems: list[tuple[str, str]] = []
    terms = [aid for _, aid in filed_documents(filed)]
    if list(family.contract.filed_terms) != terms:
        problems.append(("", f"the filed terms read are {terms}, not {list(family.contract.filed_terms)}"))
    rep = family.representative
    if rep is None:
        return problems
    exact = records(store, filed)
    stated = {m.id: m for m in family.markets}
    derived: dict[str, ContractIR] = {}
    for t, r in exact.items():
        try:
            derived[t] = instantiate(rep, family.bindings, r)
        except BindingError as e:
            problems.append((t, str(e).replace(f"{t}'s ", "its ").replace(f"{t}: ", "")))
            continue
        if t not in stated:
            problems.append((t, "its IR is missing from the family"))
        elif derived[t].content_hash() != stated[t].content_hash():
            problems.append((t, "its IR is not what the representative and the bindings make of its committed record"))
    extra = sorted(set(stated) - set(exact))
    if extra:
        problems.append(("", f"IRs of markets not harvested: {_names(extra)}"))
    if rep.id not in exact:
        problems.append(("", f"the representative, {rep.id}, is not a harvested market"))
    elif rep.id in derived and derived[rep.id] != rep:
        problems.append(("", "the representative is not what its bindings make of its own market's record: "
                             + _difference(rep, derived[rep.id])))  # fmt: skip
    if family.settlement is not None:
        at_field = family.settlement.at_field
        if at_field.split(".")[0] not in LISTING_FIELDS:
            problems.append(("", f"the settlement value's instant is read from {at_field}, which is not a listing field"))
        else:
            for t, r in exact.items():
                found, raw = _field(r, at_field)
                if not found:
                    problems.append((t, f"its record carries no {at_field}, the settlement value's instant"))
                    continue
                try:
                    _instant(raw, f"its {at_field}")
                except BindingError as e:
                    problems.append((t, str(e)))
    for ir in [rep, *family.markets]:
        try:
            accept(ir)
        except IRError as e:
            problems.append((ir.id, f"the evaluator cannot give its IR one meaning: {e}"))
    return problems


def tests_check(store: Store, proposal: Proposal) -> GateResult:
    """The settled-market check (module docstring)."""
    family, filed = _family(store, proposal)
    assert filed is not None  # the schema gate passed
    if family.representative is None:
        return GateResult(gate="tests", passed=False, detail=(
            "nothing was tested: the series is read as unsupported (addendum A05 §5.5), so it has no IR to settle. Its "
            f"unsupported terms, in the filing's words: {'; '.join(family.contract.unsupported_terms)}"))  # fmt: skip
    result = check(store, filed, family)
    claim = family.contract.equivalence
    passed = result.checked > 0 and result.not_reproduced == 0 and claim == result.equivalence
    lines = result.render()
    if claim != result.equivalence:
        lines.insert(1, f"The family states its equivalence as {claim}; the check gives {result.equivalence}.")
    elif result.checked == 0:
        lines.insert(1, "No market was checked, so nothing reproduced: the gate does not pass.")
    return GateResult(gate="tests", passed=passed, detail="\n".join(lines))


# ---------------------------------------------------------------- the check


@dataclass(frozen=True)
class MarketResult:
    """One market's place in the check: reproduced, not reproduced, not checkable or not settled by the draw's instant,
    with what was compared or why not; for a checked market, the comparisons at whose boundary its settlement value lay
    and the strike fields an off-by-one of which would change its outcome."""

    ticker: str
    kind: Literal["reproduced", "not_reproduced", "not_checkable", "not_settled"]
    text: str
    at_boundary: tuple[str, ...] = ()
    next_to: tuple[str, ...] = ()


@dataclass(frozen=True)
class Check:
    as_of: datetime
    observable: str | None
    source: str | None
    at_field: str | None
    thresholds: tuple[str, ...]  # the fields that fill a comparison's threshold
    markets: tuple[MarketResult, ...]
    boundary: bool

    def count(self, kind: str) -> int:
        return sum(1 for m in self.markets if m.kind == kind)

    @property
    def checked(self) -> int:
        return self.count("reproduced") + self.count("not_reproduced")

    @property
    def not_reproduced(self) -> int:
        return self.count("not_reproduced")

    @property
    def equivalence(self) -> core.contract.Equivalence:
        if self.not_reproduced:
            return "counterexample_found"
        return "tested_equivalent_on_suite" if self.checked else "unsupported_or_unassessed"

    def render(self) -> list[str]:
        settled = len(self.markets) - self.count("not_settled")
        lines = [
            f"{self.checked} checked of {settled} markets settled by the draw's instant, {self.as_of.isoformat()}: "
            f"{self.count('reproduced')} reproduced, {self.not_reproduced} not reproduced; {self.count('not_checkable')} "
            f"not checkable; {self.count('not_settled')} not settled by the instant. Equivalence: {self.equivalence}.",
            "Scope (addendum A09 §10.5): each checked market's settlement value is the exchange's record of it "
            f"(expiration_value, in the market record), entered as an observation of {self.observable or '(none named)'}"
            f", the observable the bindings name, from {self.source or '(none)'}, its highest-precedence consulted "
            f"source, for the instant in the market's {self.at_field or '(none named)'}, taken as published then; the "
            "market's IR is settled on it alone at the IR's final instant, after its expiration and observation cutoff. "
            "So the check tests the IR's predicate, procedure and bindings and the payout it states; it does not test "
            "the IR's choice of observable, source or instant, when the source published, or when the exchange "
            "settled or paid, which the clause-by-clause reading and the fuzzer test. An exchange's settlement is an "
            "observed procedural outcome, so a match is tested equivalence on these markets, not universal "
            "equivalence.",
        ]
        if self.boundary:
            lines += self._boundary()
        lines.append("Markets:")
        lines += [f"  {m.ticker}: {m.text}" for m in self.markets]
        return lines

    def _boundary(self) -> list[str]:
        checked = [m for m in self.markets if m.kind in ("reproduced", "not_reproduced")]
        if not checked:
            return ["Boundary: no market was checked, so none lay at or next to a comparison's boundary."]
        at = [m for m in checked if m.at_boundary]
        near = [m for m in checked if m.next_to]
        out = []
        if at:
            nodes = sorted({n for m in at for n in m.at_boundary})
            out.append(f"Boundary: {_names([m.ticker for m in at])} lay at their comparison's boundary ({', '.join(nodes)}"
                       "): its settlement value is the threshold, so making the comparison inclusive instead of strict, "
                       "or the reverse, changes its outcome, and the check discriminates an inclusive bound.")  # fmt: skip
        else:
            out.append("Boundary: no checked market's settlement value lay at its comparison's threshold, so the check "
                       "does not discriminate an inclusive bound from a strict one.")  # fmt: skip
        if not self.thresholds:
            out.append("No binding fills a comparison's threshold, so no market's strike can be off by one.")
        elif near:
            fields = sorted({f for m in near for f in m.next_to})
            out.append(f"{_names([m.ticker for m in near])} lay next to it: filling its {', '.join(fields)} from the "
                       "adjacent market's changes its outcome, so the check discriminates an off-by-one strike.")
        else:
            out.append(f"No checked market lay next to it: filling no market's {', '.join(self.thresholds)} from the "
                       "adjacent market's changes its outcome, so the check does not discriminate an off-by-one strike.")
        return out


def _hidden(ir: ContractIR) -> list[str]:
    roots = [ir.trigger] + ([ir.payout.quantity] if ir.payout.quantity is not None else [])
    return list(dict.fromkeys(n.id for r in roots for n in walk(r) if isinstance(n, Interpretive | ReadValue)))


def _value(observable: Observable, text: str) -> tuple[CmpValue | None, str]:
    """The record's settlement value as a value of the observable, or why it is none."""
    if not text:
        return None, "its record states no settlement value"
    kind = observable.value_type
    if kind == "number":
        try:
            number = Decimal(text)
        except InvalidOperation:
            return None, f"its settlement value, {text!r}, is not a number, which {observable.id} is"
        if not number.is_finite():
            return None, f"its settlement value, {text!r}, is not a finite number"
        assert observable.precision is not None  # core.Observable: a number states its precision
        if number % observable.precision != 0:
            return None, (f"its settlement value, {text!r}, is not stated at {observable.id}'s published precision, "
                          f"{observable.precision}")  # fmt: skip
        return Number(value=number), ""
    if kind in ("text", "category"):
        return Category(value=text), ""
    if kind == "date":
        try:
            return OnDay(value=date.fromisoformat(text)), ""
        except ValueError:
            return None, f"its settlement value, {text!r}, is not a date, which {observable.id} is"
    return None, f"its settlement value, {text!r}, is text, and a truth is not read from text here"


def _paid(ev: ResolutionEvaluation) -> list[Decimal]:
    return [t.amount for t in ev.transfers if t.basis == "payout"]


def _outcome(ev: ResolutionEvaluation, currency: str) -> str:
    paid = _paid(ev)
    if paid:
        return f"{ev.procedure_status}, paying {sum(paid, Decimal(0))} {currency} per unit"
    if ev.procedure_status == "resolved_no":
        return "resolved_no, paying nothing"
    return ev.procedure_status


def _reproduces(ev: ResolutionEvaluation, outcome: Settled) -> tuple[bool, str]:
    """Whether the evaluation is the record's outcome: its result as the procedure status, and the settlement value as
    the payout the evaluation states. Where the evaluation states a payout whose date it leaves unresolved, it states no
    transfer, and the amount is not compared (said so)."""
    status = ev.procedure_status
    if outcome.result == "scalar":
        status_ok = status in ("resolved_yes", "resolved_no")
    else:
        status_ok = status == {"yes": "resolved_yes", "no": "resolved_no"}[outcome.result]
    paid = _paid(ev)
    value = outcome.settlement_value
    if value is None:
        return status_ok, "the record states no settlement value, so only the result is compared"
    if paid or status == "resolved_no":
        return status_ok and sum(paid, Decimal(0)) == value, ""
    if status == "resolved_yes" and outcome.result != "scalar":
        return status_ok, "the evaluation leaves the payout's date unresolved, so it states no transfer and the amount is not compared"
    return False, ""


def _differs(a: ResolutionEvaluation, b: ResolutionEvaluation) -> bool:
    return (a.procedure_status, sorted(_paid(a))) != (b.procedure_status, sorted(_paid(b)))


def _toggled(ir: ContractIR, node: str) -> ContractIR:
    data = ir.model_dump(mode="json")
    for j in (n for t in (data["trigger"], data["payout"]["quantity"]) for n in _json_nodes(t) if n["id"] == node):
        j["op"] = ORDERED[j["op"]]
    return ContractIR.model_validate_json(core.canonical_json(data))


def _orderable(raw: Any) -> Decimal | str | None:
    """A strike's value where strikes are ordered: a number, or an instant or date given as text (compared as the
    instant it names); None for a value with no order (a text naming an option)."""
    if isinstance(raw, int | Decimal) and not isinstance(raw, bool):
        return Decimal(raw)
    if isinstance(raw, str):
        try:
            return instant(raw).isoformat() if "T" in raw else date.fromisoformat(raw).isoformat()
        except ValueError:
            return None
    return None


def _neighbours(field: str, value: Any, values: Sequence[Any]) -> list[Any]:
    """The adjacent distinct values of an ordered strike field below and above this market's, among the series'."""
    key = _orderable(value)
    if key is None:
        return []
    keyed = sorted({k: v for v in values if (k := _orderable(v)) is not None and type(k) is type(key)}.items())
    keys = [k for k, _ in keyed]
    i = keys.index(key)
    return [keyed[j][1] for j in (i - 1, i + 1) if 0 <= j < len(keyed)]


def check(store: Store, filed: FiledSeries, family: ContractFamily, *, boundary: bool = True) -> Check:
    """The settled-market check of a supported family (module docstring). `boundary`: also find which checked markets
    lay at or next to their comparison's boundary."""
    rep = family.representative
    assert rep is not None
    exact = records(store, filed)
    s = family.settlement
    observables = {o.id: o for o in rep.observables}
    sources, agree = consulted(rep)
    source = sources[s.observable][0] if s is not None and sources[s.observable] and s.observable not in agree else None
    unsourced = ""
    if s is not None and source is None:
        unsourced = (f"its procedure has {s.observable}'s sources agree, and the record states one value"
                     if s.observable in agree else f"its procedure consults none of {s.observable}'s sources")
    thresholds = tuple(dict.fromkeys(b.field for b in family.bindings if b.slot == "threshold"))
    strikes = {f: [found[1] for r in exact.values() if (found := _field(r, f))[0]] for f in thresholds}
    results = []
    for ir in sorted(family.markets, key=lambda m: m.id):
        results.append(_market(store, filed, family, ir, exact, observables, (source, unsourced), strikes, boundary))
    return Check(filed.as_of, s.observable if s else None, source, s.at_field if s else None, thresholds,
                 tuple(results), boundary)  # fmt: skip


def _market(store: Store, filed: FiledSeries, family: ContractFamily, ir: ContractIR, exact: Mapping[str, Any],
            observables: Mapping[str, Observable], sourced: tuple[str | None, str], strikes: Mapping[str, Sequence[Any]],
            boundary: bool) -> MarketResult:  # fmt: skip
    t = ir.id
    source, unsourced = sourced
    artifact = filed.markets[t]
    outcome = f1.known_outcome(store, artifact, filed.as_of)
    if outcome is None:
        return MarketResult(t, "not_settled", "not settled by the draw's instant")

    def not_checkable(why: str) -> MarketResult:
        return MarketResult(t, "not_checkable", f"not checkable: {why} (the exchange settled {outcome.result or '(no result)'})")

    hidden = _hidden(ir)
    if hidden:
        return not_checkable(f"its IR has interpretive conditions or readings ({', '.join(hidden)}), which its market "
                             "record does not settle")  # fmt: skip
    s = family.settlement
    if s is None:
        return not_checkable("the bindings name no observable that its settlement value reports")
    if source is None:
        return not_checkable(unsourced)
    if outcome.result not in ("yes", "no", "scalar"):
        return not_checkable(f"its record states no result it settled to ({outcome.result!r})")
    value, why = _value(observables[s.observable], outcome.expiration_value)
    if value is None:
        return not_checkable(why)
    record = exact[t]
    at = instant(str(_field(record, s.at_field)[1]))
    observation = Observation(observable=s.observable, source=source, at=at, value=value, published_at=at, final=True,
                              artifact=artifact)  # fmt: skip
    final = final_instant(ir)
    ev = evaluate(ir, [observation], {}, at=final)
    ok, note = _reproduces(ev, outcome)
    actual = f"the exchange settled {outcome.result}" + (
        f" at {outcome.settlement_value} per unit" if outcome.settlement_value is not None else "")
    shown = f"{_outcome(ev, ir.payout.currency)} on {s.observable} = {outcome.expiration_value}"
    if ok:
        text = f"reproduced: {shown}; {actual}" + (f" ({note})" if note else "")
    else:
        why_open = f"; the evaluation: {'; '.join(ev.unresolved)}" if ev.unresolved else ""
        text = f"NOT reproduced: the IR gives {shown}, where {actual}{why_open}"
    if not boundary:
        return MarketResult(t, "reproduced" if ok else "not_reproduced", text)
    at_boundary = tuple(
        n.id for n in dict.fromkeys(n for n in walk(ir.trigger) if isinstance(n, Cmp | NumCmp) and n.op in ORDERED)
        if _differs(evaluate(_toggled(ir, n.id), [observation], {}, at=final), ev)
    )
    rep = family.representative
    assert rep is not None
    next_to = []
    for f, values in strikes.items():
        for other in _neighbours(f, _field(record, f)[1], values):
            shifted = instantiate(rep, family.bindings, _set_field(record, f, other))
            if _differs(evaluate(shifted, [observation], {}, at=final_instant(shifted)), ev):
                next_to.append(f)
                break
    marks = ([f"at the boundary of {', '.join(at_boundary)}"] if at_boundary else []) + (
        [f"next to it in {', '.join(next_to)}"] if next_to else [])
    text += f" [{'; '.join(marks)}]" if marks else ""
    return MarketResult(t, "reproduced" if ok else "not_reproduced", text, at_boundary, tuple(next_to))


# ---------------------------------------------------------------- shown


def render(proposal: Proposal) -> str:
    """A contract proposal as `requisite show` prints it: its gates (the tests gate's detail gives each market's result),
    the harvested contract, the bindings, the representative IR whole, and each market's bound values."""
    family = ContractFamily.model_validate_json(core.canonical_json(dict(proposal.payload)))
    c = family.contract
    out = [f"proposal {proposal.id}  kind contract  status {proposal.status}  author {proposal.author}"]
    for g in proposal.gates:
        out.append(f"  gate {g.gate}: {'passed' if g.passed else 'not passed'}  {g.detail}")
    out += [
        f"filing {c.filing}; filed terms read: {', '.join(c.filed_terms)}",
        f"support {c.support}; equivalence {c.equivalence}",
        "assumptions:" + ("".join(f"\n  - {a}" for a in c.assumptions) or " none"),
        "unsupported terms:" + ("".join(f"\n  - {u}" for u in c.unsupported_terms) or " none"),
    ]
    rep = family.representative
    if rep is not None:
        out.append("bindings:" + ("".join(
            f"\n  {b.field} -> {b.slot} of {b.node or 'the timing'}" for b in family.bindings) or " none"))  # fmt: skip
        s = family.settlement
        out.append(f"settlement value: expiration_value reports {s.observable}, for the instant in {s.at_field}"
                   if s else "settlement value: the bindings name no observable it reports")  # fmt: skip
        out.append(f"representative IR {rep.id} v{rep.version}:")
        out.append(json.dumps(rep.model_dump(mode="json"), indent=2))
        out.append(f"market IRs ({len(family.markets)}), each the representative with its bindings filled from the "
                   "market's committed record:")  # fmt: skip
        for m in family.markets:
            out.append(f"  {m.id}: " + ", ".join(dict.fromkeys(_bound(m, b) for b in family.bindings)))
    out.append(f"rationale: {proposal.rationale}")
    return "\n".join(out)


def _bound(ir: ContractIR, b: ListingBinding) -> str:
    """The value a binding filled in a market's IR, as the IR holds it."""
    if b.node is None:
        assert b.slot in TIMING_SLOTS
        return f"{b.field}={getattr(ir.timing, b.slot).isoformat()}"
    nodes = {n.id: n for e in (ir.trigger, ir.payout.quantity) if e is not None for n in walk(e)}
    n = nodes[b.node]
    if b.slot == "threshold":
        v = n.value if isinstance(n, NumCmp) else n.value.value
    elif b.slot in ("at", "since", "deadline"):
        v = getattr(n, b.slot)
    else:
        w = n.days if isinstance(n, CountTrue) else n.window
        v = w.start if b.slot == "window_start" else w.end
    return f"{b.field}={v.isoformat() if isinstance(v, datetime | date) else v}"


__all__ = ["BindingError", "Check", "MarketResult", "check", "instantiate", "read", "register", "render"]
