"""Construction's observables (spec §8.1) and prices (spec §8.3; I9): an agent's candidate enters the case only through
its gates, the judges' hedgeability answers decide which pass (an unknown is no pass), and a listed market settling on a
candidate's own predicate prices it by its quote, which enters the optimizer as a price."""

from __future__ import annotations

from decimal import Decimal as D

import construct_case as cc
import deep_case as dc

from requisite import core
from requisite.construct import steps
from requisite.discovery import wide


def drafts(event, exposures):
    good = cc.bool_draft(event, exposures)
    unofficial = cc.bool_draft(event, exposures, "blog_says_in_effect")
    unofficial = unofficial.model_copy(update={"observable": unofficial.observable.model_copy(
        update={"description": "whether an unofficial blog says the rule is in effect"})})  # fmt: skip
    unknown = cc.bool_draft(event, exposures, "press_reports")
    unknown = unknown.model_copy(update={"observable": unknown.observable.model_copy(
        update={"description": "whether the press reports the rule in effect"})})  # fmt: skip
    elsewhere = cc.bool_draft(event, ["fuel.E1"], "fuel_rule")  # an exposure the case does not have
    same = cc.bool_draft(event, exposures, "constant").model_copy(update={"absent": core.Flag(value=True)})
    return [good, unofficial, unknown, elsewhere, same]


class Hedging(cc.Rule):
    def __call__(self, a):
        text = str(a.question.bindings.get("observable", ""))
        values = list(a.asked.answers)
        if a.question.template_id == "source_authoritative" and "unofficial blog" in text:
            return dc.leaning(values, "not_authoritative")
        if a.question.template_id == "timely" and "the press reports" in text:
            return dc.leaning(values, "not_established")
        return super().__call__(a)


def test_a_candidate_observable_enters_through_its_gates_and_passes_only_on_every_screen(tmp_path):
    b = cc.build(tmp_path, scout=cc.Scout(drafts), rule=Hedging())
    s = b.screening
    status = {p.payload.get("observable", {}).get("id"): p for p in wide.proposals(b.store, "observable")}
    assert status["E1.fuel_rule"].status == "rejected"
    assert "['fuel.E1'] are not the case's exposures of E1" in status["E1.fuel_rule"].gates[-1].detail
    rejected_schema = [p for p in wide.proposals(b.store, "observable") if p.status == "rejected"
                       and p.gates[-1].gate == "schema"]  # fmt: skip
    assert len(rejected_schema) == 1 and "distinct" in rejected_schema[0].gates[-1].detail
    tests = status["E1.in_effect"].gates
    assert [g.gate for g in tests] == ["schema", "tests"] and tests[-1].detail.endswith("each accepted by the evaluator")
    passes = {x.observable: x.passes for x in s.screens}
    assert passes == {"E1.in_effect": core.Kleene.TRUE, "E1.blog_says_in_effect": core.Kleene.FALSE,
                      "E1.press_reports": core.Kleene.UNKNOWN}  # fmt: skip
    press = next(x for x in s.screens if x.observable == "E1.press_reports")
    assert press.answers["timely"].applied == "not_established" and press.answers["timely"].truth is core.Kleene.UNKNOWN
    # Only the observable that passed every screen carries candidates.
    made = steps.candidates(b.store, 1)
    assert {base.observable.id for c in made.candidates for base in c.bases} == {"E1.in_effect"}
    shown = steps.render_screening(s, b.store)
    assert "E1.press_reports" in shown and "passes unknown" in shown and "rejected: tests:" in shown


def _listed(tmp_path):
    """A case whose candidate digital.E1.in_effect a listed market settles on, its contract active in the case."""
    b = cc.build(tmp_path)
    ir = next(c.ir for c in steps.candidates(b.store, 1).candidates if c.ir.id == "digital.E1.in_effect")
    market = ir.model_copy(update={"id": "KXRULE-27MAR31"})
    family = core.ContractFamily(
        contract=core.HarvestedContract(filing="KXRULE|2026-11-01T00:00:00+00:00", filed_terms=[],
                                        support="supported_exact", ir=market.ref,
                                        equivalence="unsupported_or_unassessed"),
        representative=market, markets=[market],
    )  # fmt: skip
    b.store.register_gate("contract", "tests", lambda s, p: core.GateResult(gate="tests", passed=True,
                                                                             detail="the test's listed contract"))  # fmt: skip
    payload = family.model_dump(mode="json")
    p = b.store.submit_proposal(core.Proposal(id="contract-kxrule", case=b.store.case, kind="contract", payload=payload,
                                              author="a test", rationale="a listed market", status="proposed"))  # fmt: skip
    assert p.status == "active"
    return b


def test_a_quote_on_the_candidates_own_predicate_enters_the_optimizer_as_its_ask_and_fee(tmp_path):
    b = _listed(tmp_path)
    # The market served no prices for the decision: the candidate keeps an indicative price, which says so.
    unquoted = steps.solve(b.store, k=1, judge=b.judges, quote=lambda ticker, at: None, **cc.PARAMETERS)
    p0 = next(x for x in unquoted.prices[0].prices if x.candidate == "digital.E1.in_effect")
    assert p0.pricing == "indicative" and p0.convention.endswith(
        "listed market(s) KXRULE-27MAR31 settle on its predicate but served no YES ask for the decision")
    asked = []

    def quote(ticker, at, fee=D("0.02")):
        asked.append((ticker, at))
        return core.Quote(market=ticker, yes_bid=D("0.50"), yes_ask=D("0.55"), no_bid=D("0.44"), no_ask=D("0.49"),
                          last=D("0.52"), yes_ask_size=D("1000"), at="2026-11-01T00:00:00+00:00",
                          locator="https://external-api.kalshi.com/trade-api/v2/markets/KXRULE-27MAR31", fee=fee)  # fmt: skip

    # A quote whose exchange fee is not known is no acquisition cost (spec §10: fees enter every comparison), and an
    # indicative price would stand in for the market's (spec §8.3): the candidate's cost is not determined, so it is
    # unpriced, with why, and enters no solve (A04 §4.3). Quotes are kept per market and decision time in the case, so
    # this one is read in a case of its own.
    other = _listed(tmp_path / "feeless")
    feeless = steps.solve(other.store, k=1, judge=other.judges, quote=lambda t, at: quote(t, at, fee=None),
                          **cc.PARAMETERS)  # fmt: skip
    for m in range(2):
        why = feeless.prices[m].unpriced["digital.E1.in_effect"]
        assert "at a YES ask of 0.55" in why and "fee to buy at it is not known" in why
        assert "digital.E1.in_effect" not in {x.candidate for x in feeless.prices[m].prices}
        assert "digital.E1.in_effect" not in {c.ref for c in feeless.solves[m].result.candidates}
    asked.clear()

    r = steps.solve(b.store, k=1, judge=b.judges, quote=quote, **{**cc.PARAMETERS, "eligibility": "listed"})
    assert asked == [("KXRULE-27MAR31", dc.AS_OF)]  # read at the decision time
    priced = {x.candidate: x for x in r.prices[0].prices}
    q = priced["digital.E1.in_effect"]
    assert q.pricing == "quoted" and q.quote is not None and q.quote.fee is not None and q.expected_payout is None
    assert q.price == q.quote.yes_ask + q.quote.fee  # the ask and the fee, recorded apart in the quote
    assert q.convention.startswith("quoted: the YES ask of listed market KXRULE-27MAR31")
    assert priced["digital.retroactive.E1.in_effect"].pricing == "indicative"  # another predicate: no quote
    # Listed only: the quoted candidate alone is eligible, and the optimizer reads its quote as its price.
    assert set(r.classes[0].ineligible) == set(priced) - {"digital.E1.in_effect"}
    (column,) = r.solves[0].result.candidates
    assert (column.ref, column.pricing, column.price) == ("digital.E1.in_effect", "quoted", q.price)
    assert r.chosen == ("digital.E1.in_effect",)
    assert r.parameters.listed == ("contract-kxrule",) and "1 settle on a candidate's predicate" in r.prices[0].listed
