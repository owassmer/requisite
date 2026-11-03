"""Construction on a deep case built over the product's own steps (tests/construct/construct_case.py), with the agent and
the judges doubled: every payoff is the evaluator's, the LP's optimum is confirmed independently with its error bound
computed, the cash table recomputes from the chosen quantities, a reduced cost equals its finite difference where the
step keeps the basis, and a late payout shows its timing gap (spec §8.3; addendum A06, C01 §7.2)."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, time, timedelta
from decimal import Decimal as D
from fractions import Fraction as F
from zoneinfo import ZoneInfo

import construct_case as cc
import numpy as np
import pytest
from scipy.optimize import linprog

from requisite import core
from requisite.construct import nodes, steps
from requisite.contract.evaluate import evaluate, final_instant
from requisite.core.evidence import synthetic_artifact_id
from requisite.optimize import evaluate as cash_of
from requisite.optimize import solve


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("REQUISITE_REPLAY_ONLY", "1")
        b = cc.build(tmp_path_factory.mktemp("construct"))
        return b, cc.solve(b)


def decision(r: core.ConstructionRecord) -> nodes.Decision:
    return ("decision", 1, r.objective, r.alpha, r.tolerance)


def problem(r: core.ConstructionRecord, model: int = 0, only=None):
    price_of = {p.candidate: p.price for p in r.prices[model].prices}
    probs = [s.probability for s in r.scenario_model.scenarios]
    return nodes.program(r.business, r.receipts[model], r.classes[model], price_of, probs, r.parameters, decision(r),
                         only=only)  # fmt: skip


# ---------------------------------------------------------------- payoffs (I8)


def realized(c: core.ConstructedCandidate, occurs: dict[str, bool], day_of: dict) -> list[core.SyntheticObservation]:
    """The scenario's observations, written here from the scenario's terms: a value for each day the contract observes,
    at its midnight, from the highest-ranked consulted source, the event's value from its day where it occurs."""
    ir, tz = c.ir, ZoneInfo(c.ir.timing.tz)
    out = []
    for b in c.bases:
        source = next(s for s in ir.resolution.precedence if s in {x.id for x in b.observable.sources})
        day = ir.timing.open.astimezone(tz).date()
        if datetime.combine(day, time(0), tzinfo=tz) < ir.timing.open:
            day += timedelta(days=1)
        while (at := datetime.combine(day, time(0), tzinfo=tz)) <= ir.timing.observation_cutoff:
            on = occurs[b.event] and day_of[b.event] is not None and day >= day_of[b.event]
            value = b.occurs if on else b.absent
            out.append(core.SyntheticObservation(observable=b.observable.id, source=source, at=at, value=value,
                                                 published_at=at, final=True,
                                                 artifact=synthetic_artifact_id(f"{b.observable.id}{at}".encode())))  # fmt: skip
            day += timedelta(days=1)
    return out


def test_every_payoff_in_the_matrix_is_the_evaluators(built):
    _, r = built
    occurs = {s.id: dict(s.occurs) for s in r.scenario_model.scenarios}
    by_id = {c.ir.id: c for c in r.candidates.candidates}
    assert len(by_id) == 8
    tz = ZoneInfo(r.business.tz)
    for settlement, receipts in zip(r.settlements, r.receipts, strict=True):
        assert not settlement.excluded and len(settlement.settled) == len(by_id) * len(occurs)
        on_grid = {x.candidate: x for x in receipts.candidates}
        for x in settlement.settled:
            c = by_id[x.candidate]
            ev = evaluate(c.ir, realized(c, occurs[x.scenario], dict(settlement.event_days)), {}, at=final_instant(c.ir))
            assert x.evaluation.transfers == ev.transfers and x.evaluation.procedure_status == ev.procedure_status
            # Its receipts on the business's grid are those transfers, each on its paid day, the rest after the horizon.
            s = receipts.scenarios.index(x.scenario)
            days: dict = {}
            for t in ev.transfers:
                days[t.at.astimezone(tz).date()] = days.get(t.at.astimezone(tz).date(), D(0)) + t.amount
            inside = {d: a for d, a in days.items() if d <= r.business.grid[-1] and a}
            assert dict(on_grid[x.candidate].rows[s]) == inside
            assert on_grid[x.candidate].after_horizon[s] == sum((a for d, a in days.items() if d > r.business.grid[-1]),
                                                                D(0))  # fmt: skip
    # The adverse model reads the event on its window's first day, the favourable on its last.
    assert dict(r.settlements[0].event_days) == {"E1": cc.at("2026-11-01").date()}
    assert dict(r.settlements[1].event_days) == {"E1": cc.at("2027-03-31").date()}


def test_an_indicative_price_is_the_expected_discounted_payout_times_the_loading_and_says_so(built):
    _, r = built
    probs = {s.id: s.probability for s in r.scenario_model.scenarios}
    paid = nodes.transfers(r.settlements[0])
    premium = r.business.premium_day
    for p in r.prices[0].prices:
        e = sum((probs[s] * sum((F(t.amount) / (1 + F(1, 20) * F((t.at.date() - premium).days, 365)) for t in ts),
                                F(0)) for s, ts in paid[p.candidate].items()), F(0))  # fmt: skip
        assert p.pricing == "indicative" and p.expected_payout == e and p.loading == F(1, 5)
        assert F(p.price) >= e * F(6, 5) > F(p.price) - F(1, 10**6)
        assert p.convention.startswith("indicative, not a quote: the expected payout under the scenario model")
    assert r.prices[0].listed == "the case holds no listed contract, so no quote settles on a candidate's predicate"


# ---------------------------------------------------------------- the LP's optimum, checked independently


def lipschitz(p, j) -> F:
    """max over scenarios and dates of |A_stj|: how far one more unit of candidate j moves any dated deficit, so how far
    it moves CVaR (CVaR is monotone and moves with a constant added to every deficit; A06 §6.3)."""
    return max(abs(F(a)) for row in p.net_cash(p.candidates[j]) for a in row)


def grid_cvar(p, qs: np.ndarray) -> np.ndarray:
    """CVaR_alpha of the maximum dated deficit at each row of quantities `qs`, by the Rockafellar-Uryasev formula over
    the atoms (A06 §6.3), in floats: C_st = B_st + sum_j A_stj q_j, D_s = max(0, max_t (r_t - C_st))."""
    base = np.array([[float(x) for x in row] for row in p.baseline])  # S x T
    nets = np.array([[[float(x) for x in row] for row in p.net_cash(c)] for c in p.candidates])  # J x S x T
    reserve = np.array([float(x) for x in p.reserve])
    cash = base[None] + np.einsum("gj,jst->gst", qs, nets)  # G x S x T
    d = np.maximum(0.0, (reserve[None, None] - cash).max(axis=2))  # G x S
    probs = np.array([float(x) for x in p.probabilities])
    tail = 1 / (1 - float(p.alpha))
    values = [d[:, k] + tail * (probs[None] * np.maximum(d - d[:, k : k + 1], 0.0)).sum(axis=1) for k in range(d.shape[1])]
    return np.min(np.array(values), axis=0)


def test_the_lp_optimum_agrees_with_an_exhaustive_grid_within_its_computed_bound(built):
    _, r = built
    chosen = list(r.chosen)
    assert len(chosen) == 2  # the adverse solve buys two classes: a digital and a six-month strip
    p = problem(r, only=chosen)
    sol = solve(p, r.parameters.epsilon)
    tol = F(sol.tolerance)
    n = 200
    step = [F(c.cap) / n for c in p.candidates]
    grid = np.array([(float(step[0] * i), float(step[1] * k)) for i in range(n + 1) for k in range(n + 1)
                     if step[0] * i * F(p.candidates[0].price) + step[1] * k * F(p.candidates[1].price) <= F(p.budget)])  # fmt: skip
    best = F(float(grid_cvar(p, grid).min()))
    # Rounding the optimum down to the grid stays feasible and moves each quantity by less than one step, and one unit
    # of candidate j moves every dated deficit, so CVaR, by at most L_j: the grid's best is within the bound of the
    # optimum, and no feasible point lies below it. The floats' own error is under a cent at these magnitudes (ours:
    # values below 1e6, float64's relative error 1e-16, a few hundred additions).
    bound = sum((lipschitz(p, j) * step[j] for j in range(2)), F(0))
    floats = F(1, 100)
    assert sol.hedged.cvar - tol - floats <= best <= sol.hedged.cvar + bound + tol + floats
    assert bound < sol.hedged.cvar / 10  # a bound tight enough to say something beside an optimum of this size
    # The full solve chose those two classes, so restricting to them loses nothing beyond the tolerance.
    assert abs(sol.hedged.cvar - r.solves[0].result.hedged.cvar) <= tol


def test_a_second_solver_on_an_independent_formulation_agrees_within_tolerance(built):
    _, r = built
    res = r.solves[0].result
    scale = F(res.scale)
    probs = [s.probability for s in r.scenario_model.scenarios]
    grid = list(r.business.grid)
    rows = {c.candidate: c for c in r.receipts[0].candidates}
    loss = r.business.models[0].loss
    alpha = r.alpha
    n_s, n_j = len(probs), len(res.candidates)
    # Variables: q (J), d (S), zeta, u (S). Rows: d_s >= r_t - C_st(q); u_s >= d_s - zeta; premium <= budget.
    a, b = [], []
    for s in range(n_s):
        cash, cum = F(r.business.initial), {c.ref: F(0) for c in res.candidates}
        recv = {c.ref: dict(rows[c.ref].rows[s]) for c in res.candidates}
        for t, day in enumerate(grid):
            cash += F(r.business.operating[t]) - F(loss[s][t])
            row = np.zeros(n_j + 2 * n_s + 1)
            for j, c in enumerate(res.candidates):
                cum[c.ref] += F(recv[c.ref].get(day, D(0)))
                row[j] = -float((cum[c.ref] - F(c.price)) / scale)  # the premium is paid on the grid's first day
            row[n_j + s] = -1
            a.append(row)
            b.append(float((cash - F(r.business.reserve[t])) / scale))
    for s in range(n_s):
        row = np.zeros(n_j + 2 * n_s + 1)
        row[n_j + s], row[n_j + n_s], row[n_j + n_s + 1 + s] = 1, -1, -1
        a.append(row)
        b.append(0.0)
    row = np.zeros(n_j + 2 * n_s + 1)
    row[:n_j] = [float(F(c.price) / scale) for c in res.candidates]
    a.append(row)
    b.append(float(F(r.parameters.budget) / scale))
    objective = np.zeros(n_j + 2 * n_s + 1)
    objective[n_j + n_s] = 1
    objective[n_j + n_s + 1 :] = [float(p / (1 - alpha)) for p in probs]
    bounds = [(0, float(c.cap)) for c in res.candidates] + [(0, None)] * n_s + [(None, None)] + [(0, None)] * n_s
    ipm = linprog(objective, A_ub=np.array(a), b_ub=np.array(b), bounds=bounds, method="highs-ipm",
                  options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10,
                           "ipm_optimality_tolerance": 1e-10})  # fmt: skip
    assert ipm.status == 0
    # The interior-point solve's own tolerance (1e-10, relative to the scaled objective of order one) and the declared
    # tolerance, both in money at the record's scale.
    error = F(res.epsilon) * scale + F(1, 10**10) * scale * max(1, abs(F(repr(ipm.fun))))
    assert abs(F(repr(ipm.fun)) * scale - F(res.primary_objective)) <= error


# ---------------------------------------------------------------- the reported table, recomputed


def test_dated_cash_and_deficits_recompute_from_the_chosen_quantities(built):
    _, r = built
    for solved, receipts in zip(r.solves, r.receipts, strict=True):
        res, b = solved.result, r.business
        loss = b.models[r.solves.index(solved)].loss
        rows = {c.candidate: c for c in receipts.candidates}
        q = {ref: F(x) for ref, x in res.hedged.quantities.items()}
        price = {c.ref: F(c.price) for c in res.candidates}
        premium = sum((price[j] * q[j] for j in q), F(0))
        assert F(res.hedged.premium) == premium <= F(r.parameters.budget)
        for s, sid in enumerate(b.scenarios):
            cash, paid, deficit, first = F(b.initial), F(0), F(0), None
            for t, day in enumerate(b.grid):
                today = sum((q[j] * F(dict(rows[j].rows[s]).get(day, D(0))) for j in q), F(0))
                paid += today
                cash += F(b.operating[t]) - F(loss[s][t]) + today - (premium if day == b.premium_day else 0)
                assert F(res.hedged.cash[s][t]) == cash
                short = F(b.reserve[t]) - cash
                deficit = max(deficit, short)
                if short > F(res.hedged.breach_tolerance) and first is None:
                    first = day
            assert F(res.hedged.deficit[s]) == deficit
            row = solved.table[s]
            assert row.scenario == sid and F(row.deficit_hedged) == deficit and row.first_breach_hedged == first
            assert F(row.receipts) == paid and F(row.residual) == F(row.loss) - paid + premium
            assert F(row.loss) == sum((F(x) for x in loss[s]), F(0))
    # Covered in the adverse model, the event leaves the business short beside a hedge that pays too late.
    assert r.solves[0].uncovered == ("E1:occurs",)


# ---------------------------------------------------------------- reduced costs (addendum C01 §7.2)


def forced(p, cand, delta: D):
    """The program with delta units of `cand` held outside it: its receipts in the operating flow, its premium paid on
    the first day (the premium day) and out of the budget."""
    assert cand.premium_date == p.dates[0]
    op = tuple(tuple(o + delta * h for o, h in zip(p.operating[s], cand.receipts[s], strict=True))
               for s in range(len(p.probabilities)))  # fmt: skip
    return replace(p, operating=op, initial_cash=p.initial_cash - delta * cand.price, budget=p.budget - delta * cand.price)


def test_a_nonbasic_candidates_reduced_cost_equals_its_finite_difference(built):
    _, r = built
    res = r.solves[0].result
    full = problem(r)
    reduced = {x.candidate: x for x in res.reduced_costs}
    # Every unbought class is shown with its reduced cost beside its re-solve's measured improvement, never alone.
    assert set(reduced) == {c.ref for c in full.candidates} - set(r.chosen)
    j = "strip12m.E1.in_effect"
    cand = next(c for c in full.candidates if c.ref == j)
    master = nodes.program(r.business, r.receipts[0], r.classes[0], {p.candidate: p.price for p in r.prices[0].prices},
                           [s.probability for s in r.scenario_model.scenarios], r.parameters, decision(r),
                           only=[c.ref for c in full.candidates if c.ref != j])  # fmt: skip
    base = solve(master, r.parameters.epsilon)
    assert base.quantities and j not in [c.ref for c in master.candidates]
    diffs = [(F(solve(forced(master, cand, d), r.parameters.epsilon).primary_objective) - F(base.primary_objective)) / F(d)
             for d in (D("0.01"), D("0.02"), D("0.05"))]  # fmt: skip
    # The step keeps the basis (the difference quotient is the same at each step) and equals the reduced cost, so the
    # fixture is nondegenerate here; and adding the candidate improves nothing (a positive reduced cost).
    assert max(diffs) - min(diffs) < F(1, 10**6)
    assert abs(diffs[0] - F(reduced[j].value)) < F(1, 10**6)
    assert reduced[j].value > 0 and abs(reduced[j].measured_improvement) <= F(res.epsilon * res.scale)
    shown = steps.render_record(r)
    for x in res.reduced_costs:
        assert f"unbought {x.candidate}: reduced cost" in shown and "beside its re-solve's measured improvement" in shown


# ---------------------------------------------------------------- timing (addendum A06 §6.6)


def test_a_payout_matching_the_loss_in_total_but_late_shows_its_timing_gap(built):
    _, r = built
    occurs = r.business.scenarios.index("E1:occurs")
    p = problem(r, only=["digital.E1.in_effect"])  # pays one unit on April 1, the window's end and a day
    (cand,) = p.candidates
    total = sum(p.loss[occurs], D(0))
    held = cash_of(p, [total])  # as many units as the year's loss: its payout matches the loss in total
    assert held.receipts[occurs] == held.loss[occurs] == total and held.residual[occurs] == held.premium
    (gap,) = [g for g in nodes.gaps(p, held, r.receipts[0], r.business.scenarios) if g.scenario == "E1:occurs"]
    # ...yet the business is short before the payout arrives: the terminal match is no liquidity protection.
    assert gap.first_day == r.business.grid[0] and gap.last_day == cc.at("2027-03-31").date()
    assert gap.later >= gap.shortfall > 0
    # The record's own chosen portfolio shows its gap the same way.
    (chosen,) = r.solves[0].timing_gaps
    assert chosen.scenario == "E1:occurs" and chosen.later >= chosen.shortfall


# ---------------------------------------------------------------- what the record says


def test_the_record_says_what_produced_each_field(built):
    b, r = built
    assert r.objective == "liquidity_cvar" and r.alpha == F(9, 10) and r.tolerance == D("100")
    assert r.parameters.budget == D("30000") and r.top_k == ("labor.E1",) and r.run_status == "local_requisite"
    assert r.scenario_model.kind == "single_event" and r.scenario_model.meaning == "physical"
    assert [s.probability for s in r.scenario_model.scenarios] == [F(3, 5), F(2, 5)]
    assert r.chosen == ("digital.E1.in_effect", "strip6m.E1.in_effect")
    # Candidates the solve cannot tell apart are one class: a retroactive variant pays as its base does in every
    # modeled scenario; in the favourable model the early digital pays when the late one does.
    classes = {c.representative: dict(c.members) for c in r.classes[0].classes}
    assert classes["strip12m.E1.in_effect"] == {"strip12m.E1.in_effect": 1, "strip12m.retroactive.E1.in_effect": 1}
    fav = {c.representative: dict(c.members) for c in r.classes[1].classes}
    assert set(fav["digital.E1.in_effect"]) == {"digital.E1.in_effect", "digital.early.E1.in_effect",
                                                "digital.retroactive.E1.in_effect"}  # fmt: skip
    # The worst-case claim is not made: strips pay by when the event happens, as the loss does.
    assert not r.worst_case.made and "not claimed to be the worst case" in r.worst_case.reason
    # The optimality statement names the tolerance, evidence, candidate set, pricing policy and scenario model.
    for words in ("within the solver's tolerance of", f"state {r.evidence_hash[:12]}", "8 contract(s) generated",
                  "pricing policy (indicative; loading 1/5", "scenario model (single_event, 2 scenarios)",
                  "one forecast at a time"):  # fmt: skip
        assert words in r.statement
    # Each computed construction vertex has its derivation receipt committed, under its value hash.
    receipts = {x.value_hash for x in b.store.all("derivation", core.DerivationReceipt)}
    for m in ("adverse", "favourable"):
        for part in ("settle", "receipts", "prices", "classes", "solve"):
            assert r.vertices[f"v:construct.{part}.{m}"] in receipts
    for vid in ("v:construct.business", "v:construct.candidates", "v:construct.model", "v:construct.repricing",
                "v:construct.robustness", "v:construct.worst_case"):  # fmt: skip
        assert r.vertices[vid] in receipts
    assert r == steps.latest(b.store)


# ---------------------------------------------------------------- the record holds only its own problem (A06 §6.2; §12)


def refused(r: core.ConstructionRecord, change, match: str) -> None:
    """The record with one part changed, validated again, is refused for the reason given."""
    dumped = json.loads(r.model_dump_json())
    change(dumped)
    with pytest.raises(ValueError, match=match):
        core.ConstructionRecord.model_validate_json(json.dumps(dumped))


def test_a_solve_is_the_records_own_problem_within_its_budget(built):
    _, r = built
    adverse = r.solves[0].result
    assert adverse.hedged.premium > 0  # the chosen design costs something, so a budget below it is a real test
    assert core.ConstructionRecord.model_validate_json(r.model_dump_json()) == r  # the record as built holds

    def budget(d):
        d["parameters"]["budget"] = str(adverse.hedged.premium - D("0.01"))

    def probabilities(d):
        d["solves"][0]["result"]["probabilities_hash"] = core.probabilities_hash([F(1, 2), F(1, 2)])

    def scenarios(d):
        d["solves"][0]["result"]["scenarios_hash"] = "0" * 64

    def price(d):
        c = d["solves"][0]["result"]["candidates"][0]
        c["price"] = str(D(c["price"]) + D("1"))

    def grid(d):
        d["solves"][0]["result"]["dates"] = d["solves"][0]["result"]["dates"][1:]

    def indicative(d):
        p = next(p for p in d["prices"][0]["prices"] if p["pricing"] == "indicative")
        p["price"] = "0"

    def untested(d):
        lo, hi = r.parameters.loading_range
        d["parameters"]["loading_range"] = [str(lo), str(hi + 1)]

    def table(d):
        d["solves"][0]["table"][0]["deficit_hedged"] = str(D(d["solves"][0]["table"][0]["deficit_hedged"]) + 1)

    def negative(d):
        d["solves"][0]["table"][0]["deficit_unhedged"] = "-1"

    def screen(d):
        used = d["candidates"]["candidates"][0]["observables"][0]
        s = next(s for s in d["screens"] if s["proposal"] == used)
        s["answers"]["timely"].update({"applied": "review", "status": None, "truth": "unknown"})
        s["passes"] = "unknown"

    def history(d):
        days = d["settlements"][0]["event_days"]
        event = next(iter(days))
        days[event] = "2027-12-31"

    def business(d):
        d["business"]["initial"] = str(D(d["business"]["initial"]) + 1)

    def tolerance(d):
        d["repricing"]["tolerance"] = "-1"

    def loading(d):
        lo, hi = r.parameters.loading_range
        other = hi if r.parameters.loading != hi else lo
        d["parameters"]["loading"] = f"{other.numerator}/{other.denominator}"

    if r.prices[0].prices and any(p.loading is not None for p in r.prices[0].prices):
        refused(r, loading, "use another loading than the record's")
    def undated(d):
        d["event_days"] = []

    def currency(d):
        d["business"]["currency"] = "EUR"

    refused(r, undated, "dates every event of its scenario model")
    refused(r, currency, "not the business's currency")
    def alpha(d):
        d["alpha"] = "1/2" if r.alpha != F(1, 2) else "1/4"

    refused(r, alpha, "not the record's")
    refused(r, business, "computed from another business")
    refused(r, tolerance, "a tolerance is not negative")
    refused(r, history, "date the events otherwise")
    refused(r, screen, "did not pass their screen")
    refused(r, table, "not its solve's recomputed cash")
    refused(r, negative, "a deficit is not negative")
    refused(r, grid, "other days than the business's grid")
    refused(r, indicative, "expected payout times")
    refused(r, untested, "tests the declared loading range")
    refused(r, budget, "exceeds the budget")
    refused(r, probabilities, "other scenarios or probabilities")
    refused(r, scenarios, "other scenarios or probabilities")
    refused(r, price, "not its recorded price")


def test_a_yes_whose_payout_is_not_paid_is_no_determined_payment(built):
    _, r = built
    ir = {c.ir.id: c.ir for c in r.candidates.candidates}
    yes = next(s for st in r.settlements for s in st.settled if s.evaluation.procedure_status == "resolved_yes")
    assert core.payment_why(yes.evaluation, ir[yes.candidate]) is None
    unpaid = yes.evaluation.model_copy(update={"transfers": (), "unresolved": ("expiration date: left to a rule",)})
    why = core.payment_why(unpaid, ir[yes.candidate])
    assert why is not None and "0 of its" in why and "left to a rule" in why
    # A payment dated only as a window (a settlement stated as a deadline) has no one date: not determined, never placed
    # at one end of it (A04 §4.3).
    (paid,) = [t for t in yes.evaluation.transfers if t.basis == "payout"]
    window = yes.evaluation.model_copy(update={"transfers": (paid.model_copy(update={"earliest": paid.at - timedelta(days=1)}),)})
    why = core.payment_why(window, ir[yes.candidate])
    assert why is not None and "on a date between" in why
    # An open quantity whose payment the contract still fixes (a cap, a rounding) pays it: what counts is the transfer.
    q = paid.amount
    capped = yes.evaluation.model_copy(update={"quantity": (q, q + 1)})
    assert core.payment_why(capped, ir[yes.candidate]) is None
    big = core.construction.indicative_price(F(10**22) + F(1, 10**6), F(0))
    assert big == D("10000000000000000000000.000001")  # exact at any size: never below the formula


def test_a_quoted_price_is_its_ask_and_fee_exactly_never_rounded():
    def priced(price: str, fee: str) -> core.Price:
        q = core.Quote(market="M", yes_bid=D("0.4"), yes_ask=D("1"), no_bid=D("0"), no_ask=D("0.6"), last=D("0.5"),
                       yes_ask_size=D("10"), at="2026-11-01T00:00:00+00:00", locator="https://example/M", fee=D(fee))  # fmt: skip
        return core.Price(candidate="c", pricing="quoted", price=D(price), expected_payout=None, loading=None, quote=q,
                          convention="quoted")  # fmt: skip

    assert priced("1.0000000000000000000000000001", "1e-28").price > 1  # the fee is in the price
    with pytest.raises(ValueError, match="the ask and the exchange's fee"):
        priced("1", "1e-28")  # a price that drops the fee is refused, however small the fee
