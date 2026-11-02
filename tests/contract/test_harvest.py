"""A filed series read into the IR as an instrument family (spec §9.3, §13 WP12), and the gates that check its IRs against
how its markets settled: each market's IR instantiated from the representative over a synthetic strike ladder; the
tests gate catching an IR off by one strike, or inclusive where the terms say "above", by the market that settles
differently; the schema gate refusing an IR the evaluator cannot give one meaning to; the support classification
carried as read, never silently simplified (addendum A05 §5.5); and the Western Grazers series (fixtures/kalshi/KXCAGOAT)
read into a proposal whose market its record cannot check. The reader is a test double: each test's own reading, never a
cassette. A harvest is committed here as F1 commits one (requisite/market/f1.py), its artifacts by the same code."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal as D
from pathlib import Path
from typing import Any

import pytest
import western_grazers as wg
from pydantic import ValidationError

from requisite import cassettes, cli, secrets
from requisite import core as c
from requisite.agents import runner
from requisite.agents.answers import HarvestDraft
from requisite.agents.runner import AgentResult
from requisite.contract import harvest
from requisite.contract.harvest import BindingError, instantiate
from requisite.evidence.artifacts import commit
from requisite.evidence.fetched import Fetched
from requisite.market import f1
from requisite.market.kalshi import API, instant, settled_at
from requisite.store import Store

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures" / "kalshi" / "KXCAGOAT"
AS_OF = datetime(2026, 10, 20, 12, tzinfo=UTC)  # the draw's instant
RETRIEVED = AS_OF + timedelta(hours=1)  # a harvest reads the record after the instant (f1: a reconstruction)

# The synthetic ladder: "above <strike>" at 3 PM ET on October 19, 2026, the index settling at exactly 100.00. Its
# cutoff is the expected expiration, five minutes after the close, and the exchange settles later still, as one does.
SERIES = "KXLAD"
OPEN, CLOSE = "2026-10-19T14:00:00Z", "2026-10-19T19:00:00Z"
EXPECTED, LATEST = "2026-10-19T19:05:00Z", "2026-10-26T19:00:00Z"
SETTLED = "2026-10-19T19:12:00Z"
STRIKES = (90, 95, 100, 105, 110)


def load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())


def market(strike: int, *, value: str = "100.00", result: str | None = None, settled: bool = True) -> dict[str, Any]:
    """A market of the ladder in the recorded market shape (fixtures/kalshi/KXCAGOAT/markets.json)."""
    m = copy.deepcopy(load("markets.json")["body"]["markets"][0])
    m.update(
        ticker=f"{SERIES}-26OCT19-T{strike}", event_ticker=f"{SERIES}-26OCT19", title="Index at 3 PM ET on Oct 19, 2026?",
        subtitle=f"Above {strike}", yes_sub_title=f"Above {strike}", no_sub_title=f"Above {strike}",
        rules_primary=f"If the index is above {strike} at 3 PM ET on Oct 19, 2026, then the market resolves to Yes.",
        rules_secondary="The index's level as published by the Index Agency.", strike_type="greater",
        floor_strike=float(strike), cap_strike=float(strike + 5), open_time=OPEN, close_time=CLOSE,
        expected_expiration_time=EXPECTED, expiration_time=LATEST, latest_expiration_time=LATEST,
        occurrence_datetime=CLOSE,
    )  # fmt: skip
    if not settled:
        for k in ("settlement_ts", "settlement_value_dollars"):
            m.pop(k)
        m.update(status="active", result="", expiration_value="")
        return m
    result = result or ("yes" if D(value) > strike else "no")
    m.update(status="finalized", result=result, expiration_value=value, settlement_ts=SETTLED,
             settlement_value_dollars="1.0000" if result == "yes" else "0.0000")  # fmt: skip
    return m


def text_pdf(text: str) -> bytes:
    """A one-page PDF whose text layer is `text`."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def fetched(locator: str, content: bytes, media: str = "application/json") -> Fetched:
    return Fetched(source="Kalshi", locator=locator, status=200, media_type=media, content=content,
                   retrieved_at=RETRIEVED)  # fmt: skip


def commit_harvest(store: Store, series: dict[str, Any], markets: list[dict[str, Any]], terms: bytes,
                   contract: bytes | None = None) -> c.FiledSeries:  # fmt: skip
    """A series' harvest as f1.retrieve commits it: its record, its filings and its markets' records, and the FiledSeries
    naming them at the draw's instant."""
    ticker = series["ticker"]
    record = commit(store, fetched(f"{API}/series/{ticker}", json.dumps({"series": series}).encode()), published_at=None,
                    basis=f1.RECORD_BASIS)  # fmt: skip
    pdf = "application/pdf"
    filed_terms = commit(store, fetched(series["contract_terms_url"], terms, pdf), published_at=None,
                         basis=f1.FILING_BASIS)  # fmt: skip
    filed_contract = (commit(store, fetched(series["contract_url"], contract, pdf), published_at=None,
                             basis=f1.FILING_BASIS) if contract is not None else None)  # fmt: skip
    arts = {}
    for m in markets:
        at = settled_at(m)
        arts[m["ticker"]] = commit(store, fetched(f"{API}/markets/{m['ticker']}", json.dumps({"market": m}).encode()),
                                   published_at=at, basis=f1.MARKET_SETTLED if at else f1.MARKET_OPEN).id  # fmt: skip
    filed = c.FiledSeries(series=ticker, as_of=AS_OF, why="test", record=record.id,
                          contract=filed_contract.id if filed_contract else None, terms=filed_terms.id, markets=arts)  # fmt: skip
    store.commit("filed_series", filed.key, filed)
    return filed


def ladder(store: Store, markets: list[dict[str, Any]] | None = None) -> c.FiledSeries:
    series = {**load("series.json")["body"]["series"], "ticker": SERIES, "title": "Index level",
              "contract_terms_url": f"https://assets.kalshi.com/contract_terms/{SERIES}.pdf", "contract_url": None}  # fmt: skip
    rows = markets if markets is not None else [market(s) for s in STRIKES] + [market(115, settled=False)]
    terms = text_pdf("The Payout Criterion encompasses Expiration Values above <strike> at <time>.")
    return commit_harvest(store, series, rows, terms)


INDEX = c.Observable(
    id="index", description="The index's level at a time", value_type="number",
    sources=[c.Source(id="agency", name="Index Agency", rank=1, role="statistical_release")],
    revision_policy="as_of_expiry", unit="points", precision=D("0.01"),
)  # fmt: skip


def reading(op: str = ">", strike: str = "100") -> c.ContractIR:
    """The ladder's representative market (the first in ticker order, T100) read into the IR."""
    return c.ContractIR(
        id="read", version=9, observables=[INDEX],
        trigger=c.At(id="index_at_close", at=instant(CLOSE),
                     expr=c.Cmp(id="index_above_strike", observable="index", op=op, value=c.Number(value=D(strike)))),
        payout=c.Payout(kind="binary", unit_value=D("1.00"), currency="USD", rounding_quantum=D("0.01"),
                        rounding_mode="half_even", payment="at_settlement"),
        timing=c.Timing(open=instant(OPEN), close=instant(CLOSE), expiration=instant(EXPECTED),
                        observation_cutoff=instant(EXPECTED), settlement_lag_days=1, payment_lag_days=0,
                        lag_days="calendar", holiday_calendar=None, tz="America/New_York"),
        resolution=c.ResolutionProcedure(
            precedence=["agency"], conflict_policy=c.ConflictPolicy(procedure="precedence_first_available"),
            on_unknown_at_expiry="no", early_expiration="none", void_transfer=c.VoidTransfer(returns="nothing")),
    )  # fmt: skip


def bindings(strike_field: str = "floor_strike", expiration_field: str = "expected_expiration_time"
             ) -> list[c.ListingBinding]:  # fmt: skip
    return [
        c.ListingBinding(field=strike_field, node="index_above_strike", slot="threshold"),
        c.ListingBinding(field="close_time", node="index_at_close", slot="at"),
        c.ListingBinding(field="open_time", node=None, slot="open"),
        c.ListingBinding(field="close_time", node=None, slot="close"),
        c.ListingBinding(field=expiration_field, node=None, slot="expiration"),
        c.ListingBinding(field=expiration_field, node=None, slot="observation_cutoff"),
    ]


SETTLEMENT = c.SettlementBinding(observable="index", at_field="close_time")


def draft(ir: c.ContractIR | None = None, *, binds: list[c.ListingBinding] | None = None,
          settlement: c.SettlementBinding | None = SETTLEMENT, support: str = "supported_exact",
          assumptions: tuple[str, ...] = (), unsupported: tuple[str, ...] = ()) -> HarvestDraft:  # fmt: skip
    if support == "unsupported":
        return HarvestDraft(support="unsupported", assumptions=list(assumptions), unsupported_terms=list(unsupported),
                            ir=None, bindings=[], settlement=None)  # fmt: skip
    return HarvestDraft(support=support, assumptions=list(assumptions), unsupported_terms=list(unsupported),  # type: ignore[arg-type]
                        ir=ir or reading(), bindings=binds if binds is not None else bindings(), settlement=settlement)


class Reader:
    """The contract_reader's test double: it answers its test's own reading, and keeps what it was given."""

    def __init__(self, answer: HarvestDraft):
        self.answer, self.inputs = answer, []

    def __call__(self, role: str, input: dict[str, Any], output: type, *, effort: str, root: Path | None = None
                 ) -> AgentResult[HarvestDraft]:  # fmt: skip
        assert role == "contract_reader" and output is HarvestDraft
        self.inputs.append(input)
        run = c.sha256(c.canonical_json(input) + self.answer.model_dump_json())
        return AgentResult(role, self.answer, run, "test-double", effort, False)


@pytest.fixture
def case(monkeypatch, tmp_path):
    """A case and a cassette directory of its own: extraction runs here and is recorded there (it is local, not a
    provider), and spend is logged under the test's directory."""
    monkeypatch.delenv("REQUISITE_REPLAY_ONLY", raising=False)
    monkeypatch.setattr(secrets, "known_values", lambda: [])
    monkeypatch.setattr(cassettes, "SPEND_LOG", tmp_path / "spend.jsonl")
    store = Store(tmp_path / "case", case="f1")
    yield store, tmp_path / "cassettes"
    store.close()


def read(case, answer: HarvestDraft, series: str = SERIES) -> tuple[c.Proposal, Reader]:
    store, root = case
    reader = Reader(answer)
    return harvest.read(store, series, agent=reader, root=root), reader


def gate(p: c.Proposal, name: str) -> c.GateResult:
    (g,) = [g for g in p.gates if g.gate == name]
    return g


def line(detail: str, ticker: str) -> str:
    (found,) = [x for x in detail.splitlines() if x.startswith(f"  {ticker}: ")]
    return found


# ---------------------------------------------------------------- instantiation


def test_each_market_of_a_strike_ladder_gets_the_representatives_ir_with_its_own_strike_and_times():
    rep = reading().model_copy(update={"id": f"{SERIES}-26OCT19-T100", "version": 1})
    for strike in STRIKES:
        record = json.loads(json.dumps(market(strike)), parse_float=D)
        ir = instantiate(rep, bindings(), record)
        assert ir.id == record["ticker"] and ir.version == 1
        assert ir.trigger.expr.value == c.Number(value=D(strike))  # its own strike
        assert ir.trigger.at == instant(CLOSE)
        assert (ir.timing.open, ir.timing.expiration, ir.timing.observation_cutoff) == (
            instant(OPEN), instant(EXPECTED), instant(EXPECTED))
        # Everything not bound is the representative's.
        same = ("observables", "payout", "resolution", "exclusions", "examples")
        assert all(getattr(ir, k) == getattr(rep, k) for k in same)
        assert ir.trigger.expr.model_copy(update={"value": rep.trigger.expr.value}) == rep.trigger.expr
    # The representative is what its bindings make of its own record, and the function is pure: the same IR again.
    own = json.loads(json.dumps(market(100)), parse_float=D)
    assert instantiate(rep, bindings(), own) == rep
    assert instantiate(rep, bindings(), own).content_hash() == instantiate(rep, bindings(), own).content_hash()
    # A binding fills its slot from the terms a market was listed under, in the slot's type, or not at all.
    outcome = [c.ListingBinding(field="expiration_value", node="index_above_strike", slot="threshold")]
    with pytest.raises(BindingError, match="expiration_value is not a listing field"):
        instantiate(rep, outcome, own)
    with pytest.raises(BindingError, match="T100's record carries no functional_strike"):
        instantiate(rep, [c.ListingBinding(field="functional_strike", node="index_above_strike", slot="threshold")], own)
    with pytest.raises(BindingError, match="T100's subtitle is 'Above 100', not a number"):
        instantiate(rep, [c.ListingBinding(field="subtitle", node="index_above_strike", slot="threshold")], own)
    with pytest.raises(BindingError, match="T100's floor_strike is .*, not an instant"):
        instantiate(rep, [c.ListingBinding(field="floor_strike", node="index_at_close", slot="at")], own)
    # A binding names a slot its node has, and the IR values it fills once (the definition: core.ContractFamily).
    with pytest.raises(ValidationError, match="index_at_close, a at node, which has no threshold"):
        draft(binds=[c.ListingBinding(field="floor_strike", node="index_at_close", slot="threshold")])
    with pytest.raises(ValidationError, match="filled by more than one field"):
        draft(binds=[*bindings(), c.ListingBinding(field="cap_strike", node="index_above_strike", slot="threshold")])


# ---------------------------------------------------------------- the gates


def test_a_reading_that_settles_every_market_as_the_exchange_did_becomes_active(case):
    store, _ = case
    filed = ladder(store)
    p, reader = read(case, draft())
    assert p.status == "active" and [(g.gate, g.passed) for g in p.gates] == [("schema", True), ("tests", True)]
    detail = gate(p, "tests").detail
    assert detail.startswith(f"5 checked of 5 markets settled by the draw's instant, {AS_OF.isoformat()}: 5 reproduced, "
                             "0 not reproduced; 0 not checkable; 1 not settled by the instant. Equivalence: "
                             "tested_equivalent_on_suite.")  # fmt: skip
    assert "Scope (addendum A09 §10.5)" in detail and "does not test the IR's choice of observable, source or instant" in detail
    # Settled at 100.00: T100 lies at its comparison's boundary, and T95 and T100 next to it.
    assert f"Boundary: {SERIES}-26OCT19-T100 lay at their comparison's boundary (index_above_strike)" in detail
    assert f"{SERIES}-26OCT19-T100, {SERIES}-26OCT19-T95 lay next to it" in detail
    assert line(detail, f"{SERIES}-26OCT19-T100") == (
        f"  {SERIES}-26OCT19-T100: reproduced: resolved_no, paying nothing on index = 100.00; the exchange settled no at "
        "0.0000 per unit [at the boundary of index_above_strike; next to it in floor_strike]")
    assert "reproduced: resolved_yes, paying 1.00 USD per unit" in line(detail, f"{SERIES}-26OCT19-T95")
    assert line(detail, f"{SERIES}-26OCT19-T115").endswith("not settled by the draw's instant")
    # The payload: the harvested contract as read, the representative under code's id and version, every market's IR.
    family = c.ContractFamily.model_validate_json(c.canonical_json(dict(p.payload)))
    assert family.contract.filing == filed.key and family.contract.filed_terms == (filed.terms,)
    assert family.contract.support == "supported_exact" and family.contract.equivalence == "tested_equivalent_on_suite"
    assert family.representative.id == f"{SERIES}-26OCT19-T100" and family.representative.version == 1
    assert sorted(m.id for m in family.markets) == sorted(filed.markets)
    # Committed when the gates pass: the harvested contract and each market IR.
    assert store.get("harvested_contract", f"{filed.key}|v1", c.HarvestedContract) == family.contract
    assert store.get("contract", f"{SERIES}-26OCT19-T95|1", c.ContractIR) == {m.id: m for m in family.markets}[
        f"{SERIES}-26OCT19-T95"]
    # The filing's extraction is committed; the reader read it and the markets' listing fields, never an outcome.
    (x,) = [f for f in reader.inputs[0]["filings"]]
    assert "above <strike> at <time>" in x["text"]
    (extraction,) = [r for r in store._all("extraction")]  # noqa: SLF001
    assert "above <strike> at <time>" in extraction["text"]
    shown = json.dumps(reader.inputs[0])
    for outcome in ("expiration_value", "settlement_ts", "settlement_value", '"result"', "last_price", "volume"):
        assert outcome not in shown
    assert reader.inputs[0]["markets"]["varying_fields"]["floor_strike"]["distinct"] == 6
    (run,) = store.all("agent_run", c.AgentRun)
    assert run.role == "contract_reader" and run.proposals == (p.id,)


def test_an_ir_off_by_one_strike_is_caught_by_the_market_that_settles_differently(case):
    store, _ = case
    ladder(store)
    # The reading takes each market's threshold from cap_strike, the strike above its own: one strike off. T100's own
    # cap is 105, so the representative is consistent with its bindings, and the schema gate admits it.
    p, _ = read(case, draft(reading(strike="105"), binds=bindings("cap_strike")))
    assert p.status == "rejected" and [(g.gate, g.passed) for g in p.gates] == [("schema", True), ("tests", False)]
    detail = gate(p, "tests").detail
    assert detail.startswith("5 checked of 5 markets settled by the draw's instant, 2026-10-20T12:00:00+00:00: "
                             "4 reproduced, 1 not reproduced;")  # fmt: skip
    # T95's IR asks "above 100" of a value of 100.00, where the exchange settled it YES.
    assert line(detail, f"{SERIES}-26OCT19-T95").startswith(
        f"  {SERIES}-26OCT19-T95: NOT reproduced: the IR gives resolved_no, paying nothing on index = 100.00, where the "
        "exchange settled yes at 1.0000 per unit")
    assert all(line(detail, f"{SERIES}-26OCT19-T{s}").split(": ")[1] == "reproduced" for s in (90, 100, 105, 110))
    family = c.ContractFamily.model_validate_json(c.canonical_json(dict(p.payload)))
    assert family.contract.equivalence == "counterexample_found"
    assert store.all("contract", c.ContractIR) == [] and store.all("harvested_contract", c.HarvestedContract) == []


def test_an_ir_inclusive_where_the_terms_say_above_is_caught_by_the_market_at_its_strike(case):
    store, _ = case
    ladder(store)
    p, _ = read(case, draft(reading(op=">=")))
    assert p.status == "rejected" and not gate(p, "tests").passed
    detail = gate(p, "tests").detail
    assert "4 reproduced, 1 not reproduced;" in detail.splitlines()[0]
    assert line(detail, f"{SERIES}-26OCT19-T100").startswith(
        f"  {SERIES}-26OCT19-T100: NOT reproduced: the IR gives resolved_yes, paying 1.00 USD per unit on index = 100.00, "
        "where the exchange settled no at 0.0000 per unit")


def test_where_no_settled_value_is_a_strike_the_check_says_it_does_not_discriminate_an_inclusive_bound(case):
    store, _ = case
    ladder(store, [market(s, value="101.37") for s in STRIKES])
    p, _ = read(case, draft(reading(op=">=")))  # inclusive, and no market can show it
    assert p.status == "active"
    detail = gate(p, "tests").detail
    assert ("Boundary: no checked market's settlement value lay at its comparison's threshold, so the check does not "
            "discriminate an inclusive bound from a strict one.") in detail  # fmt: skip
    assert f"{SERIES}-26OCT19-T100, {SERIES}-26OCT19-T105 lay next to it" in detail


def test_the_check_compares_outcomes_not_dates(case):
    """An IR that reads the filed latest expiration settles each market at that expiration, after the exchange settled
    it; the outcomes are still compared, at the IR's final instant, and reproduce (module docstring)."""
    store, _ = case
    ladder(store)
    late = reading().model_copy(update={"timing": reading().timing.model_copy(update={
        "expiration": instant(LATEST), "observation_cutoff": instant(LATEST)})})  # fmt: skip
    p, _ = read(case, draft(late, binds=bindings(expiration_field="latest_expiration_time")))
    assert p.status == "active", gate(p, "tests").detail


def test_the_schema_gate_refuses_an_ir_the_evaluator_cannot_give_one_meaning_to(case):
    store, _ = case
    ladder(store)
    # A comparison read at no instant: valid as a record, refused by the evaluator (no By, At or Throughout says which of
    # the observable's values to read).
    bare = reading().model_copy(update={"trigger": c.Cmp(id="index_above_strike", observable="index", op=">",
                                                         value=c.Number(value=D("100")))})  # fmt: skip
    p, _ = read(case, draft(bare, binds=bindings()[:1] + bindings()[2:]))
    assert p.status == "rejected" and [(g.gate, g.passed) for g in p.gates] == [("schema", False)]
    assert "the evaluator cannot give its IR one meaning: index_above_strike compares index outside By, At" in (
        gate(p, "schema").detail)
    family = c.ContractFamily.model_validate_json(c.canonical_json(dict(p.payload)))
    assert family.contract.equivalence == "unsupported_or_unassessed"  # nothing was assessed


def test_the_schema_gate_refuses_bindings_the_harvest_does_not_bear_out(case):
    store, _ = case
    ladder(store)
    # A representative its bindings do not reproduce from its own record: its threshold is not T100's strike.
    p, _ = read(case, draft(reading(strike="90")))
    assert [(g.gate, g.passed) for g in p.gates] == [("schema", False)]
    assert ("the representative is not what its bindings make of its own market's record: "
            "trigger.expr.value.value: 90 in the IR, 100.0 from its record") in gate(p, "schema").detail
    # A field some market lacks: that market has no IR, and the gate says why.
    rows = [market(s) for s in STRIKES]
    rows[-1].pop("cap_strike")
    store2 = Store(case[1].parent / "case2", case="f1")
    ladder(store2, rows)
    p2 = harvest.read(store2, SERIES, agent=Reader(draft(reading(strike="105"), binds=bindings("cap_strike"))),
                      root=case[1])  # fmt: skip
    assert [(g.gate, g.passed) for g in p2.gates] == [("schema", False)]
    assert f"its record carries no cap_strike ({SERIES}-26OCT19-T110)" in gate(p2, "schema").detail


def test_a_settlement_value_not_stated_in_the_observables_type_leaves_its_market_unchecked(case):
    store, _ = case
    ladder(store, [market(s, value="At least 100", result="yes" if s < 100 else "no") for s in STRIKES])
    p, _ = read(case, draft())
    assert p.status == "rejected" and [(g.gate, g.passed) for g in p.gates] == [("schema", True), ("tests", False)]
    detail = gate(p, "tests").detail
    assert detail.splitlines()[0].startswith("0 checked of 5 markets settled by the draw's instant")
    assert "5 not checkable" in detail.splitlines()[0]
    assert "No market was checked, so nothing reproduced: the gate does not pass." in detail
    assert line(detail, f"{SERIES}-26OCT19-T90").endswith(
        "not checkable: its settlement value, 'At least 100', is not a number, which index is (the exchange settled yes)")


@pytest.mark.covers("WP12.harvest_f1")
def test_a_harvested_contract_is_classified_and_never_silently_simplified(case):
    """Each harvested contract is supported exactly, supported with explicit assumptions, or unsupported (addendum A05
    §5.5): the reading's classification is held to what it says, and reaches the committed record as read."""
    # A reading cannot claim exact support while naming assumptions or unsupported terms, support with assumptions
    # without them, or an unsupported contract with an IR or without the terms it cannot carry.
    for bad in (
        dict(support="supported_exact", assumptions=("the cutoff is the expected expiration",)),
        dict(support="supported_exact", unsupported=("Kalshi may determine payouts under Rule 7.1",)),
        dict(support="supported_with_explicit_assumptions"),
    ):
        with pytest.raises(ValidationError):
            draft(**bad)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="an unsupported contract names the terms the IR cannot carry"):
        draft(support="unsupported")
    with pytest.raises(ValidationError, match="an unsupported contract has no IR"):
        HarvestDraft(support="unsupported", assumptions=[], unsupported_terms=["x"], ir=reading(), bindings=[],
                     settlement=None)  # fmt: skip
    with pytest.raises(ValidationError, match="a supported one has one"):
        HarvestDraft(support="supported_exact", assumptions=[], unsupported_terms=[], ir=None, bindings=[],
                     settlement=None)  # fmt: skip

    store, root = case
    filed = ladder(store)
    # Exact, and with explicit assumptions: each active, its classification and assumptions committed as read.
    exact, _ = read(case, draft())
    assumed = "The index's value at the close is the one published at the close."
    with_assumptions, _ = read(case, draft(support="supported_with_explicit_assumptions", assumptions=(assumed,)))
    assert exact.status == with_assumptions.status == "active"
    committed = {h.support: h for h in store.all("harvested_contract", c.HarvestedContract)}
    assert committed["supported_exact"].assumptions == () and committed["supported_exact"].unsupported_terms == ()
    assert committed["supported_with_explicit_assumptions"].assumptions == (assumed,)
    assert store.get("harvested_contract", f"{filed.key}|v2", c.HarvestedContract).support == (
        "supported_with_explicit_assumptions")  # a second reading of one filing is its own version
    # Unsupported: no IR, its unsupported terms in the filing's words; nothing was tested, so its tests gate does not
    # pass, and the rejected proposal stands as the record of the finding.
    term = "If no data is available on the Expiration Date at the Expiration Time, Kalshi determines the payout."
    unsupported, _ = read(case, draft(support="unsupported", unsupported=(term,)))
    assert unsupported.status == "rejected" and [(g.gate, g.passed) for g in unsupported.gates] == [
        ("schema", True), ("tests", False)]
    assert gate(unsupported, "tests").detail.startswith("nothing was tested: the series is read as unsupported")
    assert term in gate(unsupported, "tests").detail
    family = c.ContractFamily.model_validate_json(c.canonical_json(dict(unsupported.payload)))
    assert family.representative is None and family.markets == () and family.contract.unsupported_terms == (term,)
    assert store.proposal(unsupported.id).status == "rejected"


def test_the_western_grazers_series_is_read_into_a_proposal_its_market_record_cannot_check(case):
    """KXCAGOAT's filed terms and its market, as recorded, read with the Western Grazers IR (tests/contract/
    western_grazers.py): its trigger is interpretive, so the market record, which states the exchange's YES and no
    resolver's judgment, cannot check it, and with no market checked its tests gate does not pass."""
    store, _ = case
    series = load("series.json")["body"]["series"]
    filed = commit_harvest(store, series, load("markets.json")["body"]["markets"],
                           (FIXTURES / "contract_terms.pdf").read_bytes(), (FIXTURES / "contract.pdf").read_bytes())  # fmt: skip
    timing = [c.ListingBinding(field="open_time", node=None, slot="open"),
              c.ListingBinding(field="expiration_time", node=None, slot="expiration"),
              c.ListingBinding(field="expiration_time", node=None, slot="observation_cutoff")]  # fmt: skip
    assumptions = ("The criterion quantifies over instruments; the IR is the criterion's body for one named instrument.",
                   "An action not documented by the Expiration Date is not considered, so the contract resolves NO.")
    reading_ = draft(wg.ir(wg.AB187), binds=timing, settlement=None, support="supported_with_explicit_assumptions",
                     assumptions=assumptions)  # fmt: skip
    p, reader = read(case, reading_, series="KXCAGOAT")
    assert p.status == "rejected" and [(g.gate, g.passed) for g in p.gates] == [("schema", True), ("tests", False)]
    detail = gate(p, "tests").detail
    assert detail.startswith("0 checked of 1 markets settled by the draw's instant, 2026-10-20T12:00:00+00:00: "
                             "0 reproduced, 0 not reproduced; 1 not checkable; 0 not settled by the instant. "
                             "Equivalence: unsupported_or_unassessed.")  # fmt: skip
    assert line(detail, "KXCAGOAT-26SEP30").startswith(
        "  KXCAGOAT-26SEP30: not checkable: its IR has interpretive conditions or readings (instrument_kind, "
        "qualifying_effect, statewide_scope,")
    assert line(detail, "KXCAGOAT-26SEP30").endswith("which its market record does not settle (the exchange settled yes)")
    family = c.ContractFamily.model_validate_json(c.canonical_json(dict(p.payload)))
    assert family.contract.filed_terms == (filed.terms, filed.contract)
    assert family.contract.assumptions == assumptions and family.representative.id == "KXCAGOAT-26SEP30"
    # Both filings were read and their extractions committed; the reader was given their text and the market's rules.
    given = reader.inputs[0]
    assert [f["document"] for f in given["filings"]] == ["contract terms", "product certification"]
    assert "CAGOATWAGE" in given["filings"][0]["text"] and not given["filings"][0]["unreadable"]
    assert given["representative_market"]["rules_primary"].startswith("If California authorizes an alternative wage")
    assert "expiration_value" not in given["representative_market"] and "result" not in given["representative_market"]
    assert len(store._all("extraction")) == 2  # noqa: SLF001


def test_the_cli_reads_a_series_and_shows_each_markets_result(case, monkeypatch, capsys):
    store, root = case
    ladder(store)
    store.close()
    monkeypatch.setattr(cassettes, "CASSETTES", root)
    monkeypatch.setattr(runner, "run", Reader(draft()))
    where = ["--case-dir", str(Path(root).parent / "case"), "--case", "f1"]
    assert cli.main(["contract", "read", *where, "--series", SERIES]) == 0
    out = capsys.readouterr().out
    assert "status active" in out and "gate tests: passed  5 checked of 5 markets settled" in out
    pid = out.split()[1]
    # Run again, the same reading (a replayed run) is the same proposal, not a second one.
    assert cli.main(["contract", "read", *where, "--series", SERIES]) == 0
    assert capsys.readouterr().out.split()[1] == pid
    again = Store(Path(root).parent / "case", case="f1")
    assert [p.id for p in again.all("proposal", c.Proposal)] == [pid]
    again.close()
    assert cli.main(["show", *where, pid]) == 0
    shown = capsys.readouterr().out
    assert f"  {SERIES}-26OCT19-T95: reproduced: resolved_yes, paying 1.00 USD per unit on index = 100.00" in shown
    assert "bindings:\n  floor_strike -> threshold of index_above_strike" in shown
    assert f"  {SERIES}-26OCT19-T105: floor_strike=105.0, close_time=2026-10-19T19:00:00+00:00" in shown
    assert "support supported_exact; equivalence tested_equivalent_on_suite" in shown
