"""Drawing the two-phase sample that the wide-mode estimators assume, and measuring their intervals' coverage under it.

The draws are the design the estimators in inference.py are correct for, so the flow draws its samples here rather than
by a rule of its own: phase 1 is a simple random sample without replacement of each phase-1 stratum; phase 2 (the gold
sample) is a simple random sample without replacement of min(allocation, items) from each phase-2 stratum, the phase-2
strata defined once, in inference.py. A draw is a pure function of its seed.

measure_coverage is addendum A07 §8.5's repeated-simulation test under the exact implemented design: over a frame whose
labels are known, it draws the design many times and reports each interval's coverage with its Monte Carlo standard
error, its misses on each side, its mean width, the coverage of each phase-1 stratum's own interval, and what each
phase-2 stratum's gold is expected to hold. measure_ratio_coverage is the same for the share of a denominator's items in
each numerator (two_phase_ratio): its score interval, Fieller's set beside it, and the naive ratio of the model's
probabilities. One frame is one setting: a regression result, not a general guarantee.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from requisite.discovery.inference import (
    RatioSet,
    _phase2_strata,
    two_phase_difference,
    two_phase_difference_tuned,
    two_phase_ratio,
)


def _phase1(
    rng: np.random.Generator, frame_sizes: Mapping[str, int], sample_sizes: Mapping[str, int]
) -> dict[str, NDArray[np.intp]]:
    if set(frame_sizes) != set(sample_sizes):
        raise ValueError("phase-1 strata: frame sizes and sample sizes name different strata")
    out = {}
    for name, N in frame_sizes.items():
        n = sample_sizes[name]
        if not 1 <= n <= N:
            raise ValueError(f"phase-1 stratum {name!r} samples {n} of {N} items")
        out[name] = np.sort(rng.choice(N, size=n, replace=False)).astype(np.intp)
    return out


def _phase2(
    rng: np.random.Generator,
    names: tuple[str, ...],
    phase1_stratum: ArrayLike,
    stratifier: ArrayLike,
    phase2_cuts: Sequence[float],
    phase2_allocation: Mapping[tuple[str, int], int],
) -> NDArray[np.bool_]:
    p2 = _phase2_strata(
        names,
        np.asarray(phase1_stratum).astype(str),
        np.asarray(stratifier, dtype=float),
        phase2_cuts,
        phase2_allocation,
    )
    n = len(p2.cell)
    order = np.lexsort((rng.random(n), p2.cell))  # a uniformly random order within each phase-2 stratum
    start = np.concatenate([[0], np.cumsum(np.bincount(p2.cell, minlength=p2.n_cells))[:-1]])
    rank = np.empty(n, dtype=np.intp)
    rank[order] = np.arange(n) - start[p2.cell[order]]
    gold: NDArray[np.bool_] = rank < p2.allocation[p2.cell]
    return gold


def draw_phase1(
    *, phase1_frame_sizes: Mapping[str, int], phase1_sample_sizes: Mapping[str, int], seed: int
) -> dict[str, NDArray[np.intp]]:
    """The phase-1 sample: for each phase-1 stratum, the positions (0 to N_h - 1, in the stratum's frozen frame order)
    of a simple random sample without replacement of n_h items, sorted."""
    return _phase1(np.random.default_rng(seed), phase1_frame_sizes, phase1_sample_sizes)


def draw_phase2(
    *,
    phase1_stratum: Sequence[str] | ArrayLike,
    stratifier: ArrayLike,
    phase2_cuts: Sequence[float],
    phase2_allocation: Mapping[tuple[str, int], int],
    seed: int,
) -> NDArray[np.bool_]:
    """The gold sample: a mask over the phase-1 items, a simple random sample without replacement of
    min(allocation, items) from each phase-2 stratum. The phase-1 strata are those the allocation names."""
    names = tuple(dict.fromkeys(name for name, _ in phase2_allocation))
    return _phase2(np.random.default_rng(seed), names, phase1_stratum, stratifier, phase2_cuts, phase2_allocation)


# ---------------------------------------------------------------------------------------------------------- coverage


def _misses(
    sets: Sequence[RatioSet], truth: float
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """For each set: whether it holds the truth, whether the truth is below every point of it, and above every point."""
    held = np.array([truth in s for s in sets])
    below = np.array([bool(s.pieces) and truth < s.pieces[0][0] for s in sets])
    above = np.array([bool(s.pieces) and truth > s.pieces[-1][1] for s in sets])
    return held, below, above


@dataclass(frozen=True)
class IntervalCoverage:
    """How often one estimator's interval held the frame truth over the draws."""

    estimator: str
    coverage: float
    monte_carlo_se: float  # sqrt(c (1 - c) / draws)
    missed_below: float  # the truth was below every point of the interval
    missed_above: float  # above every point of it
    missed_between: float  # in a gap between its pieces, or the set was empty
    mean_width: float
    by_stratum: tuple[float, ...]  # each phase-1 stratum's own interval holding its own share; () for the naive


@dataclass(frozen=True)
class GoldExpectation:
    """What one phase-2 stratum's gold is expected to hold under the design, from the frame's known labels."""

    stratum: str
    bin: int
    frame_share: float
    prevalence: float
    expected_gold: float  # E[min(allocation, phase-1 items in the stratum)], exactly, over the hypergeometric phase 1
    expected_positives: float
    expected_negatives: float


@dataclass(frozen=True)
class CoverageReport:
    truth: float
    stratum_truth: tuple[float, ...]
    level: float
    draws: int
    seed: int
    intervals: tuple[IntervalCoverage, ...]  # lam = 1, the cross-fitted lam, and the naive comparator
    gold: tuple[GoldExpectation, ...]


NAIVE = "naive: the phase-1 mean of the model's probabilities, read as labels, with its phase-1 sampling error"


def measure_coverage(
    *,
    frame: Mapping[str, tuple[ArrayLike, ArrayLike, ArrayLike]],
    phase1_sample_sizes: Mapping[str, int],
    phase2_cuts: Sequence[float],
    phase2_allocation: Mapping[tuple[str, int], int],
    draws: int,
    seed: int,
    level: float = 0.95,
    unit: str = "item",
) -> CoverageReport:
    """Coverage of the lam = 1 and cross-fitted intervals (and of the naive comparator) under the exact design, over a
    frame with known labels: frame[stratum] = (labels y in {0, 1}, predictions f in [0, 1], stratifier values), one entry
    per frame item. Seeded and reproducible."""
    names = tuple(frame)
    ys = {s: np.asarray(frame[s][0], dtype=float) for s in names}
    fs = {s: np.asarray(frame[s][1], dtype=float) for s in names}
    ss = {s: np.asarray(frame[s][2], dtype=float) for s in names}
    if any(np.any((ys[s] != 0) & (ys[s] != 1)) for s in names):
        raise ValueError("a simulated frame's labels are known, 0 or 1")
    if draws < 1:
        raise ValueError("at least one draw")
    N_h = {s: len(ys[s]) for s in names}
    N = sum(N_h.values())
    truth = sum(float(ys[s].sum()) for s in names) / N
    stratum_truth = tuple(float(ys[s].mean()) for s in names)
    z = float(stats.norm.ppf(0.5 + level / 2))
    W = np.array([N_h[s] / N for s in names])
    n_h = np.array([phase1_sample_sizes[s] for s in names], dtype=float)
    fpc = 1 - n_h / np.array([N_h[s] for s in names])

    rng = np.random.default_rng(seed)
    sets: dict[str, list[RatioSet]] = {"lam = 1": [], "cross-fitted lam": [], NAIVE: []}
    held: dict[str, list[list[bool]]] = {"lam = 1": [], "cross-fitted lam": []}
    for _ in range(draws):
        picks = _phase1(rng, N_h, phase1_sample_sizes)
        stratum = np.concatenate([np.full(len(picks[s]), s) for s in names])
        y = np.concatenate([ys[s][picks[s]] for s in names])
        f = np.concatenate([fs[s][picks[s]] for s in names])
        st = np.concatenate([ss[s][picks[s]] for s in names])
        gold = _phase2(rng, names, stratum, st, phase2_cuts, phase2_allocation)
        lam1 = two_phase_difference(
            frame_size=N,
            unit=unit,
            phase1_frame_sizes=N_h,
            phase1_sample_sizes=phase1_sample_sizes,
            phase2_cuts=phase2_cuts,
            phase2_allocation=phase2_allocation,
            phase1_stratum=stratum,
            stratifier=st,
            f=f,
            gold=gold,
            y=np.where(gold, y, np.nan),
            level=level,
        )
        tuned = two_phase_difference_tuned(
            frame_size=N,
            unit=unit,
            phase1_frame_sizes=N_h,
            phase1_sample_sizes=phase1_sample_sizes,
            phase2_cuts=phase2_cuts,
            phase2_allocation=phase2_allocation,
            phase1_stratum=stratum,
            stratifier=st,
            f=f,
            gold=gold,
            y=np.where(gold, y, np.nan),
            level=level,
        ).result
        for key, r in (("lam = 1", lam1), ("cross-fitted lam", tuned)):
            sets[key].append(r.interval)
            held[key].append([t in b.interval for b, t in zip(r.by_stratum, stratum_truth, strict=True)])
        means = np.array([fs[s][picks[s]].mean() for s in names])
        variances = np.array([fs[s][picks[s]].var(ddof=1) if len(picks[s]) > 1 else 0.0 for s in names])
        naive = float(W @ means)
        se = float(np.sqrt(np.sum(W**2 * fpc * variances / n_h)))
        sets[NAIVE].append(RatioSet(pieces=((naive - z * se, naive + z * se),), kind="interval", reason=NAIVE))

    intervals = []
    for key, rows in sets.items():
        inside, below, above = _misses(rows, truth)
        c = float(inside.mean())
        intervals.append(
            IntervalCoverage(
                estimator=key,
                coverage=c,
                monte_carlo_se=float(np.sqrt(c * (1 - c) / draws)),
                missed_below=float(below.mean()),
                missed_above=float(above.mean()),
                missed_between=float(np.mean(~inside & ~below & ~above)),
                mean_width=float(np.mean([s.width for s in rows])),
                by_stratum=tuple(float(x) for x in np.mean(held[key], axis=0)) if key in held else (),
            )
        )

    expected = []
    for s in names:
        p2 = _phase2_strata(
            (s,), np.full(N_h[s], s), ss[s], phase2_cuts, {k: v for k, v in phase2_allocation.items() if k[0] == s}
        )
        for b in range(p2.bins):
            in_cell = p2.cell == b
            K = int(in_cell.sum())
            if K == 0:
                continue
            m = np.arange(0, min(K, phase1_sample_sizes[s]) + 1)
            pm = stats.hypergeom.pmf(m, N_h[s], K, phase1_sample_sizes[s])
            e_gold = float(np.sum(pm * np.minimum(p2.allocation[b], m)))
            prev = float(ys[s][in_cell].mean())
            expected.append(
                GoldExpectation(
                    stratum=s,
                    bin=b,
                    frame_share=K / N,
                    prevalence=prev,
                    expected_gold=e_gold,
                    expected_positives=e_gold * prev,
                    expected_negatives=e_gold * (1 - prev),
                )
            )
    return CoverageReport(
        truth=truth,
        stratum_truth=stratum_truth,
        level=level,
        draws=draws,
        seed=seed,
        intervals=tuple(intervals),
        gold=tuple(expected),
    )


# ------------------------------------------------------------------------------------------------------------- ratios


@dataclass(frozen=True)
class RatioIntervalCoverage:
    """How often one estimator's set of ratios held the frame's ratio over the draws."""

    estimator: str
    coverage: float
    monte_carlo_se: float  # sqrt(c (1 - c) / draws)
    missed_below: float  # the truth was below every point of the set
    missed_above: float  # above every point of it
    missed_between: float  # in a gap between its pieces (between Fieller's rays), or the set was empty
    mean_width: float  # of the set within [0, 1]
    unbounded: float  # draws whose set is not one bounded interval of the real line (Fieller's rays or line)
    whole_unit: float  # draws whose set holds all of [0, 1]


@dataclass(frozen=True)
class RatioCoverage:
    numerator: str
    truth: float
    intervals: tuple[RatioIntervalCoverage, ...]  # the score interval, Fieller's set, and the naive comparator


@dataclass(frozen=True)
class RatioGoldExpectation:
    """What one phase-2 stratum's gold is expected to hold under the design, from the frame's known labels."""

    stratum: str
    bin: int
    frame_share: float
    expected_gold: float  # E[min(allocation, phase-1 items in the stratum)], exactly, over the hypergeometric phase 1
    expected_denominator: float  # of which in the denominator
    expected_numerators: Mapping[str, float]  # and in each numerator


@dataclass(frozen=True)
class RatioCoverageReport:
    denominator_truth: float  # the frame share of items in the denominator
    level: float
    draws: int
    seed: int
    ratios: tuple[RatioCoverage, ...]
    gold: tuple[RatioGoldExpectation, ...]
    not_measured: str | None = None  # why no coverage is measured (the simulated frame defines no ratio), else None


NAIVE_RATIO = (
    "naive: the ratio of the phase-1 means of the model's probabilities for the numerator and the denominator, read as "
    "labels, with its phase-1 sampling error by the delta method"
)


def _set_coverage(
    estimator: str, sets: Sequence[RatioSet], within: Sequence[RatioSet], truth: float
) -> RatioIntervalCoverage:
    held, below, above = _misses(sets, truth)
    c, n = float(held.mean()), len(sets)
    return RatioIntervalCoverage(
        estimator=estimator,
        coverage=c,
        monte_carlo_se=float(np.sqrt(c * (1 - c) / n)),
        missed_below=float(below.mean()),
        missed_above=float(above.mean()),
        missed_between=float(np.mean(~held & ~below & ~above)),
        mean_width=float(np.mean([s.width for s in within])),
        unbounded=float(np.mean([s.kind not in ("interval", "union", "empty") for s in sets])),
        whole_unit=float(np.mean([any(lo <= 0.0 and hi >= 1.0 for lo, hi in s.pieces) for s in within])),
    )


def _ratio_gold(names: Sequence[str], N_h: Mapping[str, int], N: int, ss: Mapping[str, Any], yd: Mapping[str, Any],
                yn: Mapping[tuple[str, str], Any], numerators: Sequence[str], phase1_sample_sizes: Mapping[str, int],
                phase2_cuts: Sequence[float], phase2_allocation: Mapping[tuple[str, int], int]
                ) -> tuple[RatioGoldExpectation, ...]:  # fmt: skip
    """What each phase-2 stratum's gold is expected to hold under the design, from the simulated frame's labels."""
    expected = []
    for s in names:
        p2 = _phase2_strata(
            (s,), np.full(N_h[s], s), ss[s], phase2_cuts, {key: v for key, v in phase2_allocation.items() if key[0] == s}
        )
        for b in range(p2.bins):
            in_cell = p2.cell == b
            K = int(in_cell.sum())
            if K == 0:
                continue
            m = np.arange(0, min(K, phase1_sample_sizes[s]) + 1)
            pm = stats.hypergeom.pmf(m, N_h[s], K, phase1_sample_sizes[s])
            e_gold = float(np.sum(pm * np.minimum(p2.allocation[b], m)))
            expected.append(
                RatioGoldExpectation(
                    stratum=s,
                    bin=b,
                    frame_share=K / N,
                    expected_gold=e_gold,
                    expected_denominator=e_gold * float(yd[s][in_cell].mean()),
                    expected_numerators={k: e_gold * float(yn[s, k][in_cell].mean()) for k in numerators},
                )
            )
    return tuple(expected)


def measure_ratio_coverage(
    *,
    frame: Mapping[str, tuple[ArrayLike, ArrayLike, Mapping[str, tuple[ArrayLike, ArrayLike]], ArrayLike]],
    phase1_sample_sizes: Mapping[str, int],
    phase2_cuts: Sequence[float],
    phase2_allocation: Mapping[tuple[str, int], int],
    draws: int,
    seed: int,
    level: float = 0.95,
    unit: str = "item",
) -> RatioCoverageReport:
    """Coverage of two_phase_ratio's sets (its score interval and Fieller's set) and of the naive comparator under the
    exact design, over a frame with known labels: frame[stratum] = (denominator labels in {0, 1}, the denominator's
    predictions, {numerator: (labels, predictions)}, stratifier values), one entry per frame item, each numerator's
    items among the denominator's. Each draw is the design's (_phase1, _phase2) and each set the product's own output
    for it. Seeded and reproducible."""
    names = tuple(frame)
    yd = {s: np.asarray(frame[s][0], dtype=float) for s in names}
    fd = {s: np.asarray(frame[s][1], dtype=float) for s in names}
    ss = {s: np.asarray(frame[s][3], dtype=float) for s in names}
    numerators = tuple(frame[names[0]][2])
    if any(tuple(frame[s][2]) != numerators for s in names):
        raise ValueError("every stratum of a simulated frame carries the same numerators")
    yn = {(s, k): np.asarray(frame[s][2][k][0], dtype=float) for s in names for k in numerators}
    fn = {(s, k): np.asarray(frame[s][2][k][1], dtype=float) for s in names for k in numerators}
    for s in names:
        if np.any((yd[s] != 0) & (yd[s] != 1)) or any(np.any((yn[s, k] != 0) & (yn[s, k] != 1)) for k in numerators):
            raise ValueError("a simulated frame's labels are known, 0 or 1")
        if any(np.any(yn[s, k] > yd[s]) for k in numerators):
            raise ValueError("a numerator's items are among the denominator's")
    if draws < 1:
        raise ValueError("at least one draw")
    N_h = {s: len(yd[s]) for s in names}
    N = sum(N_h.values())
    den_total = sum(float(yd[s].sum()) for s in names)
    expected = _ratio_gold(names, N_h, N, ss, yd, yn, numerators, phase1_sample_sizes, phase2_cuts, phase2_allocation)
    if den_total == 0:
        # The simulated frame holds no denominator item, so the ratio it would estimate is undefined: there is no
        # coverage to measure, which is a statement about the label model, never a reason to withhold the gold that
        # measures the frame itself (external evaluation, November 2).
        return RatioCoverageReport(denominator_truth=0.0, level=level, draws=draws, seed=seed, ratios=(), gold=expected,
                                   not_measured="the simulated labels put no item in the denominator, so the ratio is "
                                                "undefined under this label model")  # fmt: skip
    truths = {k: sum(float(yn[s, k].sum()) for s in names) / den_total for k in numerators}
    z = float(stats.norm.ppf(0.5 + level / 2))
    W = np.array([N_h[s] / N for s in names])
    n_h = np.array([phase1_sample_sizes[s] for s in names], dtype=float)
    fpc = 1 - n_h / np.array([N_h[s] for s in names])

    rng = np.random.default_rng(seed)
    score: dict[str, list[RatioSet]] = {k: [] for k in numerators}
    fieller: dict[str, list[RatioSet]] = {k: [] for k in numerators}
    fieller_unit: dict[str, list[RatioSet]] = {k: [] for k in numerators}
    naive: dict[str, list[RatioSet]] = {k: [] for k in numerators}
    for _ in range(draws):
        picks = _phase1(rng, N_h, phase1_sample_sizes)
        stratum = np.concatenate([np.full(len(picks[s]), s) for s in names])
        st = np.concatenate([ss[s][picks[s]] for s in names])
        gold = _phase2(rng, names, stratum, st, phase2_cuts, phase2_allocation)
        y_den = np.concatenate([yd[s][picks[s]] for s in names])
        f_den = np.concatenate([fd[s][picks[s]] for s in names])
        nums = {}
        for k in numerators:
            y = np.concatenate([yn[s, k][picks[s]] for s in names])
            nums[k] = (np.concatenate([fn[s, k][picks[s]] for s in names]), np.where(gold, y, np.nan))
        out = two_phase_ratio(
            frame_size=N,
            unit=unit,
            phase1_frame_sizes=N_h,
            phase1_sample_sizes=phase1_sample_sizes,
            phase2_cuts=phase2_cuts,
            phase2_allocation=phase2_allocation,
            phase1_stratum=stratum,
            stratifier=st,
            gold=gold,
            f_den=f_den,
            y_den=np.where(gold, y_den, np.nan),
            numerators=nums,
            level=level,
        )
        for k in numerators:
            score[k].append(out[k].interval)
            fieller[k].append(out[k].fieller)
            fieller_unit[k].append(out[k].fieller_unit)
            mean_den = np.array([fd[s][picks[s]].mean() for s in names])
            mean_num = np.array([fn[s, k][picks[s]].mean() for s in names])
            if float(W @ mean_den) == 0.0:  # the model puts no probability in the denominator: no ratio to read
                naive[k].append(RatioSet(pieces=(), kind="empty", reason=f"{NAIVE_RATIO}: undefined, the model's "
                                                                          "denominator mean is 0 in this draw"))  # fmt: skip
                continue
            ratio = float(W @ mean_num) / float(W @ mean_den)
            resid = [fn[s, k][picks[s]] - ratio * fd[s][picks[s]] for s in names]
            var = np.array([r.var(ddof=1) if len(r) > 1 else 0.0 for r in resid])
            se = float(np.sqrt(np.sum(W**2 * fpc * var / n_h))) / float(W @ mean_den)
            interval = (ratio - z * se, ratio + z * se)
            naive[k].append(RatioSet(pieces=(interval,), kind="interval", reason=NAIVE_RATIO))

    ratios = []
    for k in numerators:
        clipped = []
        for n in naive[k]:
            if not n.pieces:  # undefined in that draw: nothing to clip
                clipped.append(n)
                continue
            ((lo, hi),) = n.pieces
            clipped.append(RatioSet(((max(lo, 0.0), min(hi, 1.0)),) if lo <= 1.0 and hi >= 0.0 else (), "interval",
                                    NAIVE_RATIO))  # fmt: skip
        ratios.append(
            RatioCoverage(
                numerator=k,
                truth=truths[k],
                intervals=(
                    _set_coverage("score", score[k], score[k], truths[k]),
                    _set_coverage("Fieller", fieller[k], fieller_unit[k], truths[k]),
                    _set_coverage(NAIVE_RATIO, naive[k], clipped, truths[k]),
                ),
            )
        )

    return RatioCoverageReport(
        denominator_truth=den_total / N,
        level=level,
        draws=draws,
        seed=seed,
        ratios=tuple(ratios),
        gold=expected,
    )
