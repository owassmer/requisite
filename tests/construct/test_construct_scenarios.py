"""Construction's scenario model and robustness (spec §8.3; addendum A04 §4.1-4.2, A06 §6.7): probabilities only from the
events' ProbabilityEstimates, several events joined as an assumed independent model and said so, a committed event tree
as the joint, a missing estimate refusing the solve by name, and each forecast varied across its band one at a time, the
one that flips the choice named as a high-priority question."""

from __future__ import annotations

from decimal import Decimal as D
from fractions import Fraction as F

import construct_case as cc
import deep_case as dc
import pytest

from requisite import core
from requisite.construct import nodes, steps
from requisite.store import Refused


def test_a_forecast_whose_band_flips_the_choice_is_named(tmp_path):
    # The providers put the rule at 0.6, 0.95 and 0.05: the estimate is their mean, its band their range. With no cash to
    # spare, the premium leaves the business short where the rule does not come, which counts in the tail only where
    # the rule is unlikely.
    # A design holds where, resized, it stays within the decision's tolerance of the re-solved optimum (A03 §3.3; A06
    # §6.7: its size may change while it holds). With 20,000 in cash and a tolerance of 1, at the band's low end the best
    # structure is another (the digital alone), and the chosen mix, however sized, stays materially worse: a flip.
    rule = cc.Rule(forecast={"E1": {"openai_decisions": 0.6, "clef": 0.95, "jev": 0.05}})
    b = cc.build(tmp_path, cash="20000", rule=rule, tolerance="1")
    r = cc.solve(b)
    (band,) = r.robustness.bands
    assert band.band == (F(1, 20), F(19, 20)) and band.value == F(8, 15) and band.varied
    low, high = r.robustness.variations
    assert (low.end, low.probability, high.end, high.probability) == ("low", F(1, 20), "high", F(19, 20))
    # Each end holds exactly where the chosen design, resized, stays within the tolerance of the re-solved optimum.
    for v in (low, high):
        assert v.holds == (v.chosen_portfolio - v.optimum <= F(r.tolerance))
    flipped = [v for v in (low, high) if not v.holds]
    assert flipped, "this test's premise: an end of the band at which the chosen design is no longer acceptable"
    assert r.robustness.flips == (band.estimate,)
    (question,) = r.robustness.questions
    assert question == b.run.questions[[q.question.id for q in b.run.questions].index("deep.event_occurs.E1")].question
    shown = steps.render_record(r)
    assert "at its low end 1/20" in shown and "FLIPS" in shown
    assert "high-priority questions (the forecast flips the choice): deep.event_occurs.E1 v1" in shown
    assert "one forecast at a time" in r.robustness.method and "one forecast at a time" in r.statement


def test_an_estimate_with_no_band_cannot_be_varied_and_says_so(tmp_path):
    b = cc.build(tmp_path)
    r = cc.solve(b)
    (band,) = r.robustness.bands
    # The configurations agreed on 0.6: its band is that one value.
    assert band.band == (F(3, 5), F(3, 5)) and not band.varied and not r.robustness.variations
    assert band.reason == "its band is the one value 3/5: the configurations agreed, so there is nothing to vary"
    # An estimate that states no band at all cannot be varied either, and says so.
    (pe,) = [x for x in b.store.all("probability_estimate", core.ProbabilityEstimate) if x.id == band.estimate]
    bare = pe.model_copy(update={"interval": None})
    rb = nodes.robustness(r.business, r.receipts[0], r.prices[0], r.classes[0], r.scenario_model, (bare,),
                          (("E1", band.question),), r.parameters, ("decision", 1, r.objective, r.alpha, r.tolerance),
                          r.solves[0])  # fmt: skip
    assert rb.bands[0].reason == "no band: the estimate states no uncertainty, so it cannot be varied"
    assert not rb.bands[0].varied and not rb.variations


def test_several_events_are_joined_as_independent_and_the_assumption_is_stated(tmp_path):
    b = cc.build(tmp_path, events=(dc.EVENT, cc.E2), k=2, rule=cc.Rule(forecast={"E2": 0.3}))
    assert set(b.screening.events) == {"E1", "E2"}
    r = cc.solve(b, k=2)
    sm = r.scenario_model
    assert sm.kind == "independent_events" and sm.meaning == "hypothetical" and sm.assumption == nodes.INDEPENDENCE
    probs = {s.id: s.probability for s in sm.scenarios}
    assert probs == {"E1:occurs,E2:occurs": F(3, 5) * F(3, 10), "E1:occurs,E2:not": F(3, 5) * F(7, 10),
                     "E1:not,E2:occurs": F(2, 5) * F(3, 10), "E1:not,E2:not": F(2, 5) * F(7, 10)}  # fmt: skip
    assert sm.assumption in r.statement
    # Both events move the crews' wages: their changes add through the one driver model, an assumption stated.
    assert r.composition is not None and "labor (E1, E2)" in r.composition and r.composition in r.statement
    loss = {sid: row for sid, row in zip(r.business.scenarios, r.business.models[0].loss, strict=True)}
    for t in range(len(r.business.grid)):
        both, one, two = loss["E1:occurs,E2:occurs"][t], loss["E1:occurs,E2:not"][t], loss["E1:not,E2:occurs"][t]
        assert one + two - D("0.01") <= both <= one + two  # each rounded up to the cent, from one exact sum
    # With observables of two events, compound strips are generated: one paying while both rules hold, one either.
    compound = [c for c in r.candidates.candidates if c.family == "compound"]
    assert sorted(c.variant for c in compound) == ["and_12m", "or_12m"]
    both_ = next(c for c in compound if c.variant == "and_12m").ir.id
    paid = nodes.transfers(r.settlements[0])[both_]
    assert paid["E1:occurs,E2:not"] == () and paid["E1:not,E2:not"] == () and paid["E1:occurs,E2:occurs"]


def test_a_committed_event_tree_is_the_joint(tmp_path):
    b = cc.build(tmp_path, events=(dc.EVENT, cc.E2), k=2)

    def pe(pid: str, prop: str, value: str, p: F) -> str:
        est = core.ProbabilityEstimate(id=pid, proposition=prop, branch_value=value, conditioning=[], meaning="physical",
                                       source="judgment", source_ref="a test", value=p, interval=None,
                                       evidence_hash="h", as_of=dc.AS_OF)  # fmt: skip
        b.store.commit("probability_estimate", est.id, est)
        return est.id

    def point(pid: str, prop: str, p: F, then=(None, None)) -> core.BranchPoint:
        return core.BranchPoint(id=pid, branches=[
            core.Branch(proposition=prop, value="true", probability=pe(f"{pid}.t", prop, "true", p), then=then[0]),
            core.Branch(proposition=prop, value="false", probability=pe(f"{pid}.f", prop, "false", 1 - p), then=then[1]),
        ])  # fmt: skip

    tree = core.EventTree(id="tree", version=1, root=point("root", "E1", F(3, 5), (point("e2|e1", "E2", F(7, 10)),
                                                                                 point("e2|not", "E2", F(1, 5)))))  # fmt: skip
    b.store.commit("event_tree", "tree|1", tree)
    r = cc.solve(b, k=2)
    sm = r.scenario_model
    assert sm.kind == "event_tree" and sm.tree == core.Ref(id="tree", version=1) and sm.assumption is None
    probs = {s.id: s.probability for s in sm.scenarios}
    assert probs == {"E1:occurs,E2:occurs": F(21, 50), "E1:occurs,E2:not": F(9, 50), "E1:not,E2:occurs": F(2, 25),
                     "E1:not,E2:not": F(8, 25)}  # fmt: skip
    # Each binary point's second branch enters as the complement of its first, so varying one keeps the partition.
    factors = {s.id: list(s.factors) for s in sm.scenarios}
    assert factors["E1:not,E2:not"] == [("root.t", True), ("e2|not.t", True)]
    assert factors["E1:occurs,E2:occurs"] == [("root.t", False), ("e2|e1.t", False)]


def test_an_event_without_a_probability_refuses_the_solve_and_names_it(tmp_path):
    b = cc.build(tmp_path, events=(dc.EVENT, cc.E2), rule=cc.Rule(forecast={"E2": None}))
    assert set(b.screening.events) == {"E1"}  # E2's exposures have no materiality to enter the top k
    with pytest.raises(Refused, match=r"no probability estimate for E2 \(its forecast deep.event_occurs.E2 has fewer "
                                      r"than two distributions\): the scenario model is refused rather than solved over "
                                      r"the events that have one"):  # fmt: skip
        cc.solve(b)
    assert not b.store.all("construction_parameters", core.ConstructionParameters)  # refused before anything committed


def test_where_no_candidate_pays_by_when_its_event_happens_the_adverse_model_is_the_worst_case(tmp_path):
    # The rule takes effect on one known day: both endpoint models read it on that day, so every candidate pays alike
    # in both and only the business's own intervals (the size of the change, its lag) differ between them.
    day = cc.at("2027-01-15")
    one = core.Proposition(id="E1", text="The overtime rule takes effect", domain="bool",
                           window=core.TimeWindow(start=day, end=day, start_closed=True, end_closed=True))  # fmt: skip
    b = cc.build(tmp_path, events=(one,))
    r = cc.solve(b)
    assert {e.event: (e.earliest, e.latest) for e in r.event_days} == {"E1": (day.date(), day.date())}
    assert r.worst_case.made and r.worst_case.reason.startswith("every event of the scenario model happens on one known")
    assert "It is the worst case over the business model's intervals" in r.statement
    # The adverse model's loss is at least the favourable model's on every day of every scenario.
    adverse, favourable = r.business.models
    assert all(a >= f for ra, rf in zip(adverse.loss, favourable.loss, strict=True)
               for a, f in zip(ra, rf, strict=True))  # fmt: skip
