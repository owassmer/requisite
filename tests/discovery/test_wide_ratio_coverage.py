"""Coverage of the ratio's sets in simulation, under the exact implemented design (requisite/discovery/sampling.py,
measure_ratio_coverage; spec §7.4; addendum A07 §8.5): the share of the denominator's items (in a small business's
voice) that are in the numerator (describing exposure to the measure), R = theta_num / theta_den.

A frame with known labels; a judge biased differently in the numerator and the denominator (it over-calls the voice
indicator in one docket and misses exposure in another, and its numerator probability is the product of its two, as
F4's is); the two-phase design at the flow's sizes: the settings test_wide_coverage.py uses, carried over to two
indicators, and F4's (four phase-1 strata of 300 items read, 130 gold split 80/20 between the phase-2 strata below and
above the cut on the judge's numerator probability) with a denominator about a tenth of the frame; 1,000 draws each,
seeded. Each set measured is two_phase_ratio's own output for the draw.

What was measured, and why the interval is the score set (the measurements below, in the order they were made):
- Fieller's set from the design's unbiased variance covers 0.93-0.96 where every heavy phase-2 stratum's gold sees the
  numerator, but 0.91 in F4's shape, every miss above the set: in the draws where the heaviest stratum's 26 gold items
  hold no numerator item, its plain residual variance is only the predictions' noise, so the set is too short above.
  That is the mean interval's failure for a stratum whose gold holds no positive, and has its cure, the variance
  evaluated where the value is tested.
- The score set evaluates the variance at category probabilities restricted to each tested R. Of the paths tried for
  the restricted point, moving the labels only (inference._tilt) covers nearest nominal with the shortest sets; moving
  them toward the extreme category over-covered (0.98-0.996, sets up to a third longer), and folding the predictions'
  covariance into the path either over-covered (0.97-0.99) or, exactly, had no unique solution. Its upper side stays
  conservative where a heavy stratum's gold expects under one numerator item, as the mean interval's does: valid, and
  wide for that reason.
These are regression settings, not a guarantee for every design (A07 §8.5).
"""

from functools import cache

import numpy as np
import pytest

from requisite.discovery.sampling import NAIVE_RATIO, RatioCoverageReport, measure_ratio_coverage


def _frame(rng, strata):
    """Known labels and the judge's probabilities, one phase-1 stratum at a time. An item is in the denominator with
    probability den and, if it is, in the numerator with probability ratio; the judge's voice probability has mean vtp
    for items in the denominator and vfp for others, its exposure probability etp for exposed items and efp for others
    (Beta, concentration 6), and its numerator probability is their product; the phase-2 strata are cut on that."""
    out = {}
    for i, s in enumerate(strata):
        N = s["N"]
        voice = rng.random(N) < s["den"]
        exposed = np.where(voice, rng.random(N) < s["ratio"], rng.random(N) < 0.05)
        p_voice = np.where(voice, rng.beta(6 * s["vtp"], 6 * (1 - s["vtp"]), N), rng.beta(6 * s["vfp"], 6 * (1 - s["vfp"]), N))
        p_exp = np.where(exposed, rng.beta(6 * s["etp"], 6 * (1 - s["etp"]), N), rng.beta(6 * s["efp"], 6 * (1 - s["efp"]), N))
        f_num = p_voice * p_exp
        num = (voice & exposed).astype(float)
        out[f"docket {i}"] = (voice.astype(float), p_voice, {"exposed": (num, f_num)}, f_num)
    return out


SETTINGS = {
    # test_wide_coverage.py's two dockets: the judge over-calls voice in the first and misses exposure in the second.
    "two dockets, 20,000 items, 3,000 read, 4 strata": dict(
        strata=[
            dict(N=12000, den=0.5, ratio=0.5, vtp=0.8, vfp=0.3, etp=0.7, efp=0.2),
            dict(N=8000, den=0.3, ratio=0.5, vtp=0.6, vfp=0.1, etp=0.5, efp=0.05),
        ],
        n_h=[1500, 1500],
        cuts=[0.5],
        allocation={(0, 0): 35, (0, 1): 15, (1, 0): 35, (1, 1): 15},
    ),
    # Its census of 3,000 items, read whole: phase 1 is a census, 3 phase-2 strata.
    "census of 3,000 items, 3 strata": dict(
        strata=[dict(N=3000, den=0.3, ratio=0.5, vtp=0.7, vfp=0.15, etp=0.6, efp=0.1)],
        n_h=[3000],
        cuts=[0.3, 0.7],
        allocation={(0, 0): 50, (0, 1): 25, (0, 2): 25},
    ),
    # Its rare class: the numerator about 6% of 30,000 items, oversampled twelvefold above the cut.
    "rare class, 30,000 items, 3,000 read, 2 strata": dict(
        strata=[dict(N=30000, den=0.2, ratio=0.3, vtp=0.8, vfp=0.1, etp=0.7, efp=0.08)],
        n_h=[3000],
        cuts=[0.5],
        allocation={(0, 0): 80, (0, 1): 20},
    ),
    # F4's shape: four comment-count bands, the largest holding most items and the fewest in a business's voice (about
    # a tenth of the frame overall); 300 read per band, 130 gold, 26 below and 6-7 above the cut in each.
    "F4's shape, denominator about 0.1, 92,500 items, 1,200 read, 130 gold": dict(
        strata=[
            dict(N=1500, den=0.3, ratio=0.4, vtp=0.8, vfp=0.15, etp=0.7, efp=0.1),
            dict(N=6000, den=0.2, ratio=0.4, vtp=0.7, vfp=0.1, etp=0.6, efp=0.1),
            dict(N=25000, den=0.12, ratio=0.4, vtp=0.7, vfp=0.1, etp=0.5, efp=0.05),
            dict(N=60000, den=0.07, ratio=0.4, vtp=0.6, vfp=0.05, etp=0.5, efp=0.05),
        ],
        n_h=[300, 300, 300, 300],
        cuts=[0.5],
        allocation={(0, 0): 26, (0, 1): 7, (1, 0): 26, (1, 1): 7, (2, 0): 26, (2, 1): 6, (3, 0): 26, (3, 1): 6},
    ),
}


@cache
def _measured(name: str) -> tuple[RatioCoverageReport, str]:
    cfg = SETTINGS[name]
    frame = _frame(np.random.default_rng(20261101), cfg["strata"])
    names = list(frame)
    report = measure_ratio_coverage(
        frame=frame,
        phase1_sample_sizes=dict(zip(names, cfg["n_h"], strict=True)),
        phase2_cuts=cfg["cuts"],
        phase2_allocation={(names[i], b): a for (i, b), a in cfg["allocation"].items()},
        draws=1000,
        seed=20261102,
        unit="comment",
    )
    ratio = report.ratios[0]
    lines = [f"{name}: R = {ratio.truth:.4f}, denominator share {report.denominator_truth:.4f}, {report.draws} draws"]
    for c in ratio.intervals:
        lines.append(
            f"  {c.estimator.split(':')[0]}: coverage {c.coverage:.3f} (Monte Carlo SE {c.monte_carlo_se:.3f}), missed "
            f"below {c.missed_below:.3f} above {c.missed_above:.3f} between {c.missed_between:.3f}, mean width within "
            f"[0, 1] {c.mean_width:.3f}, unbounded {c.unbounded:.3f}, all of [0, 1] {c.whole_unit:.3f}"
        )
    for g in report.gold:
        lines.append(
            f"  gold in ({g.stratum}, bin {g.bin}): {g.frame_share:.3f} of the frame, expects {g.expected_gold:.1f} gold: "
            f"{g.expected_denominator:.2f} in the denominator, {g.expected_numerators['exposed']:.2f} in the numerator"
        )
    summary = "\n".join(lines)
    print(summary)
    return report, summary


def _gold_sees_the_numerator(report: RatioCoverageReport) -> bool:
    """Whether every phase-2 stratum holding a tenth of the frame or more expects at least one numerator item in its
    gold: where one does not, no interval honest about what that gold cannot show comes within 3 points of nominal on
    the upper side (the mean interval's case, decisions "Phase-2 allocation")."""
    return all(g.expected_numerators["exposed"] >= 1 for g in report.gold if g.frame_share >= 0.1)


@pytest.mark.parametrize("name", list(SETTINGS))
def test_the_ratio_interval_covers_and_the_naive_ratio_misses(name):
    report, summary = _measured(name)
    by = {c.estimator: c for c in report.ratios[0].intervals}
    assert by[NAIVE_RATIO].coverage < 0.5, summary
    score = by["score"]
    assert score.coverage >= report.level - 0.03, summary
    # neither side misses beyond its nominal share by more than the same 3 points
    assert max(score.missed_below, score.missed_above) <= (1 - report.level) / 2 + 0.03, summary
    if _gold_sees_the_numerator(report):
        assert score.coverage <= report.level + 0.03, summary
    else:  # over-covers for that stated reason, and still bounds the ratio in nearly every draw
        assert score.whole_unit <= 0.05, summary
    assert by["Fieller"].unbounded <= 0.1, summary  # reported beside it: here its set is almost always bounded


def test_f4s_shape_is_the_case_whose_heavy_strata_gold_expects_under_one_numerator_item():
    # The F4-shaped setting is checked under the lower bar only because of this, computed from the design, not chosen.
    report, summary = _measured("F4's shape, denominator about 0.1, 92,500 items, 1,200 read, 130 gold")
    assert not _gold_sees_the_numerator(report), summary
    for name in SETTINGS:
        if not name.startswith("F4"):
            assert _gold_sees_the_numerator(_measured(name)[0]), name


def test_the_naive_ratio_misses_under_a_judge_biased_differently_in_numerator_and_denominator():
    # Spec §7.4's acceptance test, for the ratio: the judge over-calls the denominator (voice, negatives' mean
    # probability 0.3) in one docket and misses the numerator (exposure, positives' mean 0.5) in the other, so the
    # ratio of its probabilities is pulled apart from the frame's; the interval covers near nominal, and so does
    # Fieller's set here, where every heavy phase-2 stratum's gold sees the numerator.
    report, summary = _measured("two dockets, 20,000 items, 3,000 read, 4 strata")
    by = {c.estimator: c for c in report.ratios[0].intervals}
    assert by[NAIVE_RATIO].coverage < 0.05, summary
    assert abs(by["score"].coverage - report.level) <= 0.03, summary
    assert abs(by["Fieller"].coverage - report.level) <= 0.03, summary


def test_a_simulated_frame_with_no_denominator_item_measures_no_coverage_and_still_states_the_gold():
    # The coverage check runs before the gold is drawn; under a label model that puts no item in the denominator the
    # ratio is undefined, which is said, never divided by zero, and never stops the gold that measures the frame
    # (external evaluation, November 2).
    zeros = np.zeros(20)
    frame = {"s": (zeros, np.full(20, 0.1), {"exposed": (zeros, np.full(20, 0.05))}, np.full(20, 0.05))}
    report = measure_ratio_coverage(frame=frame, phase1_sample_sizes={"s": 10}, phase2_cuts=[0.5],
                                    phase2_allocation={("s", 0): 4, ("s", 1): 2}, draws=5, seed=1)  # fmt: skip
    assert report.ratios == () and report.not_measured and "undefined" in report.not_measured
    assert [(g.stratum, g.bin) for g in report.gold] == [("s", 0)] and report.gold[0].expected_gold == pytest.approx(4)
