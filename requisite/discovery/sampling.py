"""Drawing the two-phase sample that the wide-mode estimators assume, and measuring their intervals' coverage under it.

The draws are the design the estimators in inference.py are correct for, so the flow draws its samples here rather than
by a rule of its own: phase 1 is a simple random sample without replacement of each phase-1 stratum; phase 2 (the gold
sample) is a simple random sample without replacement of min(allocation, items) from each phase-2 stratum, the phase-2
strata defined once, in inference.py. A draw is a pure function of its seed.

measure_coverage is addendum A07 §8.5's repeated-simulation test under the exact implemented design: over a frame whose
labels are known, it draws the design many times and reports each interval's coverage with its Monte Carlo standard
error, its misses on each side, its mean width, the coverage of each phase-1 stratum's own interval, and what each
phase-2 stratum's gold is expected to hold. One frame is one setting: a regression result, not a general guarantee.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from requisite.discovery.inference import (
    _phase2_strata,
    two_phase_difference,
    two_phase_difference_tuned,
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


@dataclass(frozen=True)
class IntervalCoverage:
    """How often one estimator's interval held the frame truth over the draws."""

    estimator: str
    coverage: float
    monte_carlo_se: float  # sqrt(c (1 - c) / draws)
    missed_below: float  # the truth was below the interval
    missed_above: float
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
    bounds: dict[str, list[tuple[float, float]]] = {"lam = 1": [], "cross-fitted lam": [], NAIVE: []}
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
            bounds[key].append(r.interval)
            held[key].append(
                [b.interval[0] <= t <= b.interval[1] for b, t in zip(r.by_stratum, stratum_truth, strict=True)]
            )
        means = np.array([fs[s][picks[s]].mean() for s in names])
        variances = np.array([fs[s][picks[s]].var(ddof=1) if len(picks[s]) > 1 else 0.0 for s in names])
        naive = float(W @ means)
        se = float(np.sqrt(np.sum(W**2 * fpc * variances / n_h)))
        bounds[NAIVE].append((naive - z * se, naive + z * se))

    intervals = []
    for key, rows in bounds.items():
        a = np.array(rows)
        c = float(np.mean((a[:, 0] <= truth) & (truth <= a[:, 1])))
        intervals.append(
            IntervalCoverage(
                estimator=key,
                coverage=c,
                monte_carlo_se=float(np.sqrt(c * (1 - c) / draws)),
                missed_below=float(np.mean(truth < a[:, 0])),
                missed_above=float(np.mean(truth > a[:, 1])),
                mean_width=float(np.mean(a[:, 1] - a[:, 0])),
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
