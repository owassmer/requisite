"""Coverage of the wide-mode interval in simulation, under the exact implemented design (spec §7.4, §13 WP11b; addendum
A07 §8.5), measured by the product's own measure_coverage; and the draws that design makes.

A frame with known labels; a judge whose bias differs by stratum (it over-calls in one docket and misses positives in
another, or misses them below its own threshold); the two-phase design at the flow's sizes (a frame of thousands to tens
of thousands, a phase-1 sample of a few thousand, 100 gold items, two to four phase-2 strata, the model's positives
oversampled). Seven settings: the three below, and the rare-class frame and F4's shape of test_wide_ratio_coverage.py,
each read as two means, its numerator's indicator and its denominator's. The denominator's is the voice indicator, which
the judge separates by more than half: where the interval, when it floored the labels' part of the variance at
1 - 2 lam delta >= 0, over-covered (0.999 at width 0.121 on the rare-class frame, 2,000 draws), now 0.945 at 0.074.

DRAWS draws each, seeded: a coverage near 0.95 has a Monte Carlo standard error of 0.011 at that count, so the bar of 3
points is about 2.7 of them (the full table, at 2,000 draws, SE 0.005, is in the 002-3 report). The naive interval
(the phase-1 mean of the model's probabilities, read as labels) misses; the corrected interval covers within 3 points of
nominal, for lam = 1 and for the cross-fitted lam, overall and for each phase-1 stratum's own interval. Each coverage is
reported with its Monte Carlo standard error, its misses on each side and its mean width, beside what each phase-2
stratum's gold is expected to hold.

These are regression settings, not a guarantee for every design (A07 §8.5). Their allocations give the gold of the
phase-2 strata that hold most of the frame enough expected positives for the bar to be reachable. Where such a stratum
expects well under one positive in its gold, no interval honest about what that gold cannot show comes within 3 points
on both sides; that is a property of the allocation (the coordinator's design decision), not shaped into the interval.
"""

import itertools
import math
from functools import cache

import numpy as np
import pytest
import test_wide_ratio_coverage as ratio_settings  # the ratio's frames, read here as means (same directory)
from scipy import stats

from requisite.discovery.inference import two_phase_difference
from requisite.discovery.sampling import NAIVE, CoverageReport, draw_phase1, draw_phase2, measure_coverage

DRAWS = 400


def _frame(rng, strata):
    """Known labels and the judge's probabilities, one phase-1 stratum at a time: positives' f ~ Beta with mean tp,
    negatives' with mean fp (concentration 6), so the judge's errors differ by stratum. The phase-2 strata are cut on
    the same f."""
    out = {}
    for i, s in enumerate(strata):
        y = rng.random(s["N"]) < s["q"]
        pos = rng.beta(6 * s["tp"], 6 * (1 - s["tp"]), s["N"])
        neg = rng.beta(6 * s["fp"], 6 * (1 - s["fp"]), s["N"])
        f = np.where(y, pos, neg)
        out[f"docket {i}"] = (y.astype(float), f, f)
    return out


SETTINGS = {
    # Two dockets: the judge over-calls in the first (negatives' mean f 0.2) and misses half the positives in the second
    # (positives' mean f 0.5). 4 phase-2 strata; the model's positives oversampled (15 gold of ~230 items against 35 of
    # ~1,270).
    "two dockets, 20,000 items, 3,000 read, 4 strata": dict(
        strata=[dict(N=12000, q=0.25, tp=0.7, fp=0.2), dict(N=8000, q=0.15, tp=0.5, fp=0.05)],
        n_h=[1500, 1500],
        cuts=[0.5],
        allocation={(0, 0): 35, (0, 1): 15, (1, 0): 35, (1, 1): 15},
    ),
    # A frame small enough to read whole: phase 1 is a census and the estimator is A07's finite-frame form. 3 strata.
    "census of 3,000 items, 3 strata": dict(
        strata=[dict(N=3000, q=0.15, tp=0.6, fp=0.1)],
        n_h=[3000],
        cuts=[0.3, 0.7],
        allocation={(0, 0): 50, (0, 1): 25, (0, 2): 25},
    ),
    # A rare class (about 6% of 30,000 items): the judge's positives are a few percent of the phase-1 sample and are
    # oversampled twelvefold; its misses sit in the large stratum below the cut.
    "rare class, 30,000 items, 3,000 read, 2 strata": dict(
        strata=[dict(N=30000, q=0.06, tp=0.7, fp=0.08)],
        n_h=[3000],
        cuts=[0.5],
        allocation={(0, 0): 80, (0, 1): 20},
    ),
}


# The ratio's rare-class frame and F4's shape, each read as two means: its numerator's indicator (stratified, as the
# ratio's is, on the judge's numerator probability) and its denominator's.
RATIO_FRAMES = (
    "rare class, 30,000 items, 3,000 read, 2 strata",
    "F4's shape, denominator about 0.1, 92,500 items, 1,200 read, 130 gold",
)
MEANS_OF_RATIO_FRAMES = [f"ratio's {name}: {part}" for name in RATIO_FRAMES for part in ("numerator", "denominator")]


def _setting(name):
    """The frame and design of a setting: frame[stratum] = (labels, predictions, stratifier)."""
    if name in SETTINGS:
        return _frame(np.random.default_rng(20261101), SETTINGS[name]["strata"]), SETTINGS[name]
    base, part = name.removeprefix("ratio's ").rsplit(": ", 1)
    cfg = ratio_settings.SETTINGS[base]
    frame = ratio_settings._frame(np.random.default_rng(20261101), cfg["strata"])
    if part == "numerator":
        return {s: (v[2]["exposed"][0], v[2]["exposed"][1], v[3]) for s, v in frame.items()}, cfg
    return {s: (v[0], v[1], v[3]) for s, v in frame.items()}, cfg


@cache
def _measured(name: str) -> tuple[CoverageReport, str]:
    frame, cfg = _setting(name)
    names = list(frame)
    report = measure_coverage(
        frame=frame,
        phase1_sample_sizes=dict(zip(names, cfg["n_h"], strict=True)),
        phase2_cuts=cfg["cuts"],
        phase2_allocation={(names[i], b): a for (i, b), a in cfg["allocation"].items()},
        draws=DRAWS,
        seed=20261102,
        unit="comment",
    )
    lines = [f"{name}: truth {report.truth:.4f}, {report.draws} draws"]
    for c in report.intervals:
        lines.append(
            f"  {c.estimator.split(':')[0]}: coverage {c.coverage:.3f} (Monte Carlo SE {c.monte_carlo_se:.3f}), "
            f"missed below {c.missed_below:.3f} above {c.missed_above:.3f} between {c.missed_between:.3f}, mean width "
            f"{c.mean_width:.4f}" + (f", by stratum {[round(x, 3) for x in c.by_stratum]}" if c.by_stratum else "")
        )
    for g in report.gold:
        lines.append(
            f"  gold in ({g.stratum}, bin {g.bin}): {g.frame_share:.2f} of the frame, prevalence {g.prevalence:.3f}, "
            f"expects {g.expected_gold:.1f} gold: {g.expected_positives:.2f} positive, {g.expected_negatives:.2f} negative"
        )
    summary = "\n".join(lines)
    print(summary)
    return report, summary


@pytest.mark.covers("WP11b.coverage")
@pytest.mark.parametrize("name", list(SETTINGS) + MEANS_OF_RATIO_FRAMES)
def test_corrected_interval_covers_and_naive_interval_misses(name):
    report, summary = _measured(name)
    by = {c.estimator: c for c in report.intervals}
    assert by[NAIVE].coverage < 0.5, summary
    band = 0.03 + 1e-12  # the closed band [0.92, 0.98]: 392 of 400 draws is 0.98, which a float may put past it
    for key in ("lam = 1", "cross-fitted lam"):
        assert abs(by[key].coverage - report.level) <= band, summary
        assert all(abs(x - report.level) <= band for x in by[key].by_stratum), summary


# -------------------------------------------------------------------------------------------------------- the draws


def test_phase1_draws_a_simple_random_sample_of_each_stratum():
    picks = draw_phase1(phase1_frame_sizes={"a": 50, "b": 7}, phase1_sample_sizes={"a": 20, "b": 7}, seed=3)
    assert len(picks["a"]) == 20 and len(set(picks["a"].tolist())) == 20 and np.all(np.diff(picks["a"]) > 0)
    assert picks["b"].tolist() == list(range(7))  # a census
    again = draw_phase1(phase1_frame_sizes={"a": 50, "b": 7}, phase1_sample_sizes={"a": 20, "b": 7}, seed=3)
    assert picks["a"].tolist() == again["a"].tolist()
    # Every item of a stratum is equally likely: 5 of 8, over 4,000 seeds, each item about 5/8 of the time.
    hits = np.zeros(8)
    for s in range(4000):
        hits[draw_phase1(phase1_frame_sizes={"a": 8}, phase1_sample_sizes={"a": 5}, seed=s)["a"]] += 1
    assert stats.chisquare(hits, np.full(8, 4000 * 5 / 8)).pvalue > 1e-3


def test_phase2_draws_min_of_allocation_and_items_from_each_stratum_uniformly():
    stratum = np.array(["a"] * 6 + ["b"] * 3)
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9, 0.1, 0.7, 0.2])
    allocation = {("a", 0): 2, ("a", 1): 2, ("b", 0): 3, ("b", 1): 2}
    counts = {}
    for seed in range(6000):
        gold = draw_phase2(
            phase1_stratum=stratum, stratifier=s, phase2_cuts=[0.5], phase2_allocation=allocation, seed=seed
        )
        # a: 4 items below the cut (2 drawn) and 2 above (taken whole); b: 2 below (taken whole), 1 above (taken whole).
        assert gold[:4].sum() == 2 and gold[4:6].all() and gold[6:].all()
        key = tuple(np.flatnonzero(gold[:4]).tolist())
        counts[key] = counts.get(key, 0) + 1
        if seed < 5:  # the estimator accepts what the draw drew: the same design
            two_phase_difference(
                frame_size=20,
                unit="comment",
                phase1_frame_sizes={"a": 12, "b": 8},
                phase1_sample_sizes={"a": 6, "b": 3},
                phase2_cuts=[0.5],
                phase2_allocation=allocation,
                phase1_stratum=stratum,
                stratifier=s,
                f=s,
                gold=gold,
                y=np.where(gold, (s > 0.5).astype(float), np.nan),
            )
    # Every 2-subset of the 4 items below the cut, equally often.
    assert set(counts) == set(itertools.combinations(range(4), 2))
    assert stats.chisquare(list(counts.values())).pvalue > 1e-3
    with pytest.raises(ValueError, match="no allocation"):
        draw_phase2(
            phase1_stratum=stratum,
            stratifier=s,
            phase2_cuts=[0.5],
            phase2_allocation={("a", 0): 2, ("a", 1): 2, ("b", 0): 3},
            seed=0,
        )


def test_expected_gold_is_exact_for_a_census():
    # With phase 1 a census, each phase-2 stratum has exactly its frame items, so it expects min(allocation, items) gold.
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0, 0, 1], dtype=float)
    f = np.array([0.9, 0.1, 0.2, 0.3, 0.1, 0.8, 0.6, 0.2, 0.1, 0.7])
    report = measure_coverage(
        frame={"a": (y, f, f)},
        phase1_sample_sizes={"a": 10},
        phase2_cuts=[0.5],
        phase2_allocation={("a", 0): 3, ("a", 1): 10},
        draws=5,
        seed=1,
    )
    low, high = report.gold
    assert (low.expected_gold, high.expected_gold) == (3.0, 4.0)
    assert low.expected_positives == pytest.approx(3 * 1 / 6) and high.expected_positives == pytest.approx(4 * 3 / 4)
    assert math.isclose(low.frame_share + high.frame_share, 1.0)
