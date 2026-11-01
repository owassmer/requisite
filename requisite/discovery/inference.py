"""Wide-mode inference: the frame share of items whose indicator holds, estimated under the design that drew the samples.

The design admitted here is two-phase. Phase 1 is a stratified simple random sample without replacement (SRSWOR) of
the frame; every phase-1 item carries a model prediction f in [0, 1], fixed before any gold label is seen (spec §7.4:
probabilities, never top answers). Phase 2 is the gold sample: within each phase-1 stratum, the phase-1 items are split
into phase-2 strata by thresholding a phase-1 value (the stratifier, a model probability), and each phase-2 stratum is
an SRSWOR of its phase-1 items, of size min(allocation, items) with allocation >= 2, so a phase-2 stratum with no more
items than its allocation is taken whole. A frame small enough to read whole is a phase-1 census, and then everything
here is exactly addendum A07's finite-frame form (§8.2, §8.4).

Sources, each formula beside the one it rests on:
- Addendum A07 §8.2: the difference estimator over a fixed finite frame with fixed predictions is design-unbiased, its
  SRSWOR standard error has no Var(f)/N term, and with every item gold it is the frame truth with zero sampling error.
- Addendum A07 §8.3: fixed lambda; lambda fitted on the gold it corrects changes finite-sample behaviour, so it is fitted
  by cross-fitting here, its clipping policy is explicit, and its finite-sample coverage is measured, not assumed.
- Addendum A07 §8.4: strata weighted by W_h = N_h / N.
- Two-phase sampling, including two-phase sampling for stratification: Särndal, Swensson and Wretman, *Model Assisted
  Survey Sampling* (1992), ch. 9. Our reasoning from it, not the addendum's: the variance is a phase-1 term on the target
  (the variance of the phase-1 expansion estimator of y) plus the expected phase-2 term on the residuals e = y - lam f,
  each estimated without bias, the first through an unbiased estimate of the phase-1 sample variance of y built from
  the gold sample. tests/discovery/test_two_phase_exact.py proves every formula by enumerating every phase-1 sample and
  every phase-2 subsample of small frames, the take-whole rule included.
- Spec §7.4 and PPI++ (Angelopoulos, Duchi and Zrnic, arXiv:2311.01453): the two-sample PPI++ mean, for a design the
  caller declares as two independent samples and only there (A07 §8.2: a gold subset of a frame is another design).

The unit is the item: an estimate is a share of items, never of the businesses that filed them (A07 §8.4).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------------------------------------------- design


@dataclass(frozen=True)
class _Cells:
    """The declared design resolved against the phase-1 data: one row per phase-2 stratum (a cell). Private: the
    canonical design record is the coordinator's SamplingDesign in requisite/core/."""

    names: tuple[str, ...]  # phase-1 strata, in declared order
    frame_sizes: FloatArray  # N_h
    sample_sizes: FloatArray  # n_h
    frame_size: int  # N
    cell_stratum: NDArray[np.intp]  # phase-1 stratum index of each cell
    cell_bin: NDArray[np.intp]  # stratifier bin of each cell
    m: FloatArray  # phase-1 items in each cell
    r: FloatArray  # gold items in each cell
    unit_cell: NDArray[np.intp]  # cell of each phase-1 item
    gold: NDArray[np.bool_]  # which phase-1 items are gold
    y: FloatArray  # gold labels (0 where not gold; never read there)
    f: FloatArray  # predictions of the target indicator

    @property
    def n_cells(self) -> int:
        return len(self.m)

    def cell_weight(self) -> FloatArray:
        """W_h * m_k / n_h: the frame share each cell's phase-1 mean of y stands for."""
        h = self.cell_stratum
        return (self.frame_sizes[h] / self.frame_size) * (self.m / self.sample_sizes[h])


def _resolve(
    *,
    frame_size: int,
    phase1_frame_sizes: Mapping[str, int],
    phase1_sample_sizes: Mapping[str, int],
    phase2_cuts: Sequence[float],
    phase2_allocation: Mapping[tuple[str, int], int],
    phase1_stratum: Sequence[str] | ArrayLike,
    stratifier: ArrayLike,
    f: ArrayLike,
    gold: ArrayLike,
    y: ArrayLike,
) -> _Cells:
    """Check that the data are a sample this design draws, and lay them out by cell. Refuses anything else: the
    estimator is correct only for the design that drew the samples."""
    names = tuple(phase1_frame_sizes)
    if set(phase1_sample_sizes) != set(names):
        raise ValueError("phase-1 strata: frame sizes and sample sizes name different strata")
    N_h = np.array([phase1_frame_sizes[s] for s in names], dtype=float)
    n_h = np.array([phase1_sample_sizes[s] for s in names], dtype=float)
    if int(N_h.sum()) != frame_size:
        raise ValueError(f"phase-1 strata sum to {int(N_h.sum())} items, not the frame's {frame_size}")
    if np.any(n_h < 1) or np.any(n_h > N_h):
        raise ValueError("each phase-1 stratum samples at least 1 and at most all of its items")
    if np.any((n_h < 2) & (n_h < N_h)):
        raise ValueError("a phase-1 stratum sampled short of a census needs at least 2 items to estimate its variance")
    cuts = np.asarray(phase2_cuts, dtype=float)
    if cuts.ndim != 1 or np.any(~np.isfinite(cuts)) or np.any(np.diff(cuts) <= 0):
        raise ValueError("phase-2 cuts are finite and strictly increasing")

    strat = np.asarray(phase1_stratum)
    s = np.asarray(stratifier, dtype=float)
    fv = np.asarray(f, dtype=float)
    g = np.asarray(gold)
    yv = np.asarray(y, dtype=float)
    n = strat.shape[0]
    for name, a in (("stratifier", s), ("f", fv), ("gold", g), ("y", yv)):
        if a.shape != (n,):
            raise ValueError(f"{name} has shape {a.shape}; one value per phase-1 item ({n}) is required")
    if g.dtype != np.bool_:
        raise ValueError("gold is a boolean mask over the phase-1 items")
    if np.any(~np.isfinite(s)):
        raise ValueError("the stratifier is defined for every phase-1 item")
    if np.any(~np.isfinite(fv)) or np.any((fv < 0) | (fv > 1)):
        raise ValueError("f is a probability in [0, 1] for every phase-1 item (spec §7.4)")
    if np.any((yv[g] != 0) & (yv[g] != 1)):
        raise ValueError("every gold item carries an indicator label, 0 or 1 (an unresolved gold label is not one)")

    index = {name: i for i, name in enumerate(names)}
    try:
        h = np.array([index[str(v)] for v in strat], dtype=np.intp)
    except KeyError as e:
        raise ValueError(f"phase-1 item in undeclared stratum {e}") from None
    counts = np.bincount(h, minlength=len(names))
    if np.any(counts != n_h):
        raise ValueError(f"phase-1 items per stratum {counts.tolist()} differ from the declared {n_h.astype(int).tolist()}")

    bins = len(cuts) + 1
    b = np.searchsorted(cuts, s, side="right").astype(np.intp)
    unit_cell = h * bins + b
    n_cells = len(names) * bins
    m = np.bincount(unit_cell, minlength=n_cells).astype(float)
    r = np.bincount(unit_cell[g], minlength=n_cells).astype(float)

    alloc = np.zeros(n_cells)
    for key, a_k in phase2_allocation.items():
        name, bin_ = key
        if name not in index or not 0 <= bin_ < bins:
            raise ValueError(f"phase-2 allocation for an undeclared stratum {key}")
        if a_k < 2:
            raise ValueError(f"phase-2 stratum {key} is allocated {a_k} gold items; the design allocates at least 2")
        alloc[index[name] * bins + bin_] = a_k
    missing = (m > 0) & (alloc == 0)
    if np.any(missing):
        k = int(np.flatnonzero(missing)[0])
        raise ValueError(f"phase-2 stratum {(names[k // bins], k % bins)} has phase-1 items but no allocation")
    want = np.minimum(alloc, m)
    if np.any(r != want):
        k = int(np.flatnonzero(r != want)[0])
        raise ValueError(
            f"phase-2 stratum {(names[k // bins], k % bins)} has {int(r[k])} gold items; its allocation of "
            f"{int(alloc[k])} over {int(m[k])} phase-1 items draws {int(want[k])}"
        )
    cells = np.arange(n_cells)
    return _Cells(
        names=names,
        frame_sizes=N_h,
        sample_sizes=n_h,
        frame_size=frame_size,
        cell_stratum=(cells // bins).astype(np.intp),
        cell_bin=(cells % bins).astype(np.intp),
        m=m,
        r=r,
        unit_cell=unit_cell,
        gold=g,
        y=np.where(g, yv, 0.0),
        f=fv,
    )


def _cell_sum(c: _Cells, values: FloatArray, gold_only: bool) -> FloatArray:
    idx = c.unit_cell[c.gold] if gold_only else c.unit_cell
    v = values[c.gold] if gold_only else values
    return np.bincount(idx, weights=v, minlength=c.n_cells)


def _cell_var(c: _Cells, values: FloatArray) -> FloatArray:
    """Sample variance (ddof 1) of values over each cell's gold items; 0 where a cell has fewer than 2."""
    r = c.r
    mean = _cell_sum(c, values, True) / np.maximum(r, 1)
    dev = values - mean[c.unit_cell]
    ss = _cell_sum(c, dev * dev, True)
    return np.where(r >= 2, ss / np.maximum(r - 1, 1), 0.0)


def _phase1_var(c: _Cells, values: FloatArray) -> FloatArray:
    """Sample variance (ddof 1) of values over each cell's phase-1 items; 0 where a cell has fewer than 2."""
    m = c.m
    mean = _cell_sum(c, values, False) / np.maximum(m, 1)
    dev = values - mean[c.unit_cell]
    ss = _cell_sum(c, dev * dev, False)
    return np.where(m >= 2, ss / np.maximum(m - 1, 1), 0.0)


# ------------------------------------------------------------------------------------------------------------ estimators


@dataclass(frozen=True)
class StratumMean:
    """The share of one phase-1 stratum's items whose indicator holds (a domain of the frame), under the same design."""

    stratum: str
    frame_size: int
    estimate: float
    variance: float
    interval: tuple[float, float]


@dataclass(frozen=True)
class FrameMean:
    """The frame share of items whose indicator holds, with its design-based variance and interval.

    ``variance`` estimates the design variance: phase 1 on the target plus phase 2 on the residuals. For a fixed lam it
    is exactly unbiased. ``interval`` is the construction ``interval_method`` names, whose coverage is measured by
    simulation (requisite.discovery.coverage).
    """

    unit: str
    frame_size: int
    estimate: float
    variance: float
    variance_phase1: float
    variance_phase2: float
    se: float
    interval: tuple[float, float]
    level: float
    interval_method: str
    lam: float
    lam_rule: str
    phase1_items: int
    gold_items: int
    census: bool
    by_stratum: tuple[StratumMean, ...] = field(default=())


@dataclass(frozen=True)
class _CellEstimates:
    mu: FloatArray  # each cell's estimated phase-1 mean of y
    s2e: FloatArray  # gold residual variance in each cell
    lam: FloatArray  # the lam each cell's estimate used


def _z(level: float) -> float:
    if not 0 < level < 1:
        raise ValueError("the confidence level is in (0, 1)")
    return float(stats.norm.ppf(0.5 + level / 2))


def _cell_score_bounds(c: _Cells, est: _CellEstimates, z: float) -> tuple[FloatArray, FloatArray]:
    """Each cell's score interval for its phase-1 mean of y, p: the p with (mu - p)^2 <= z^2 c S2_e(p), where
    S2_e(p) = kappa p (1 - p) (1 - 2 lam delta) + lam^2 S2_f is the cell's residual variance at prevalence p.

    That form is exact for 0/1 labels (S2_y = kappa p(1-p) and S_yf = kappa p(1-p) delta, delta the gap in mean f
    between the cell's positive and negative items; kappa = m / (m - 1)); S2_f is known from phase 1 and delta is
    estimated from the gold, 0 where the gold holds only one label (no reduction claimed), and 1 - 2 lam delta is
    floored at 0 so that the variance never falls below the f term alone. Like Wilson's interval (to which it reduces at
    lam = 0), it keeps a cell whose gold shows no positive from claiming no uncertainty."""
    m, r = c.m, c.r
    open_ = (m > 0) & (r < m)
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = np.where(open_, (1 - r / m) / r, 0.0)
        kappa = np.where(m >= 2, m / (m - 1), 1.0)
    yv, fv = c.y, c.f
    r1 = np.maximum(r, 1)
    ybar = _cell_sum(c, yv, True) / r1
    fbar = _cell_sum(c, fv, True) / r1
    dy, df = yv - ybar[c.unit_cell], fv - fbar[c.unit_cell]
    syy = _cell_sum(c, dy * dy, True)
    syf = _cell_sum(c, dy * df, True)
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.where(syy > 0, syf / syy, 0.0)
    a = np.maximum(1 - 2 * est.lam * delta, 0.0)
    b = est.lam**2 * _phase1_var(c, fv)
    mu = np.clip(est.mu, 0.0, 1.0)
    k = z * z * cc
    A = 1 + k * kappa * a
    B = 2 * mu + k * kappa * a
    C = mu * mu - k * b
    root = np.sqrt(np.maximum(B * B - 4 * A * C, 0.0))
    lo = np.clip((B - root) / (2 * A), 0.0, 1.0)
    hi = np.clip((B + root) / (2 * A), 0.0, 1.0)
    return np.where(open_, lo, est.mu), np.where(open_, hi, est.mu)


INTERVAL = "mover"


def _assemble(c: _Cells, est: _CellEstimates, *, unit: str, lam: float, lam_rule: str, level: float) -> FrameMean:
    """Combine per-cell estimates into the frame estimate, its variance, and its interval."""
    z = _z(level)
    w = c.cell_weight()
    h = c.cell_stratum
    H = len(c.names)
    W = c.frame_sizes / c.frame_size
    n_h = c.sample_sizes
    occupied = c.m > 0
    mu = np.where(occupied, est.mu, 0.0)

    # Phase 2, on the residuals: (W_h m_k / n_h)^2 (1 - r_k / m_k) s2_e,k / r_k over cells; 0 for a cell taken whole.
    with np.errstate(invalid="ignore", divide="ignore"):
        fpc2 = np.where(occupied & (c.r < c.m), (1 - c.r / c.m) / c.r, 0.0)
    v2_cell = w * w * fpc2 * est.s2e

    # Phase 1, on the target: W_h^2 (1 - n_h / N_h) S2_y,h / n_h over strata, with S2_y,h (the phase-1 sample variance
    # of y) estimated without bias from the gold. For 0/1 labels it is [T - (T^2 - V(T)) / n_h] / (n_h - 1), where
    # T = sum_k m_k ybar_k estimates the stratum's phase-1 total of y and V(T) is its phase-2 variance estimate, so that
    # T^2 - V(T) is unbiased for the squared total. It is >= 0, since 0 <= T <= n_h.
    ybar = np.where(c.r > 0, _cell_sum(c, c.y, True) / np.maximum(c.r, 1), 0.0)
    T = np.bincount(h, weights=c.m * ybar, minlength=H)
    VT = np.bincount(h, weights=c.m * c.m * fpc2 * _cell_var(c, c.y), minlength=H)
    with np.errstate(invalid="ignore", divide="ignore"):
        s2y = np.where(n_h >= 2, (T - (T * T - VT) / n_h) / (n_h - 1), 0.0)
    v1_h = W * W * (1 - n_h / c.frame_sizes) * s2y / n_h
    v2_h = np.bincount(h, weights=v2_cell, minlength=H)
    est_h = np.bincount(h, weights=w * mu, minlength=H)

    if INTERVAL == "mover":
        lo_k, hi_k = _cell_score_bounds(c, est, z)
        d_lo = np.where(occupied, np.maximum(mu - lo_k, 0.0), 0.0) * w
        d_hi = np.where(occupied, np.maximum(hi_k - mu, 0.0), 0.0) * w
        dlo_h = np.bincount(h, weights=d_lo * d_lo, minlength=H)
        dhi_h = np.bincount(h, weights=d_hi * d_hi, minlength=H)
    else:
        dlo_h = dhi_h = z * z * v2_h

    def bounds(estimate: float, dlo2: float, dhi2: float, v1: float) -> tuple[float, float]:
        # MOVER (Zou and Donner 2008): each independent component's own interval, combined in quadrature on each side.
        lo = estimate - float(np.sqrt(dlo2 + z * z * v1))
        hi = estimate + float(np.sqrt(dhi2 + z * z * v1))
        return (max(0.0, lo), min(1.0, hi))

    estimate = float(est_h.sum())
    v1, v2 = float(v1_h.sum()), float(v2_h.sum())
    by_stratum = tuple(
        StratumMean(
            stratum=name,
            frame_size=int(c.frame_sizes[i]),
            estimate=float(est_h[i] / W[i]),
            variance=float((v1_h[i] + v2_h[i]) / W[i] ** 2),
            interval=tuple(  # type: ignore[arg-type]
                x / W[i] for x in bounds(float(est_h[i]), float(dlo_h[i]), float(dhi_h[i]), float(v1_h[i]))
            ),
        )
        for i, name in enumerate(c.names)
    )
    return FrameMean(
        unit=unit,
        frame_size=c.frame_size,
        estimate=estimate,
        variance=v1 + v2,
        variance_phase1=v1,
        variance_phase2=v2,
        se=float(np.sqrt(v1 + v2)),
        interval=bounds(estimate, float(dlo_h.sum()), float(dhi_h.sum()), v1),
        level=level,
        interval_method=INTERVAL,
        lam=lam,
        lam_rule=lam_rule,
        phase1_items=int(c.m.sum()),
        gold_items=int(c.r.sum()),
        census=bool(np.all(c.sample_sizes == c.frame_sizes)),
        by_stratum=by_stratum,
    )


def two_phase_difference(
    *,
    frame_size: int,
    unit: str,
    phase1_frame_sizes: Mapping[str, int],
    phase1_sample_sizes: Mapping[str, int],
    phase2_cuts: Sequence[float],
    phase2_allocation: Mapping[tuple[str, int], int],
    phase1_stratum: Sequence[str] | ArrayLike,
    stratifier: ArrayLike,
    f: ArrayLike,
    gold: ArrayLike,
    y: ArrayLike,
    lam: float = 1.0,
    level: float = 0.95,
) -> FrameMean:
    """The difference estimator of the frame share for a fixed lam (lam = 1, the plain difference estimator, is the
    primary): sum_h W_h sum_k (m_k / n_h) [ybar_gold,k + lam (fbar_phase1,k - fbar_gold,k)] over phase-2 strata k.

    Design-unbiased for any fixed lam and any f fixed before the gold is seen, calibrated or not (f changes only the
    variance), with an unbiased variance estimate; both exactly, over every sample the design draws.
    """
    c = _resolve(
        frame_size=frame_size,
        phase1_frame_sizes=phase1_frame_sizes,
        phase1_sample_sizes=phase1_sample_sizes,
        phase2_cuts=phase2_cuts,
        phase2_allocation=phase2_allocation,
        phase1_stratum=phase1_stratum,
        stratifier=stratifier,
        f=f,
        gold=gold,
        y=y,
    )
    if not np.isfinite(lam):
        raise ValueError("lam is a finite number fixed before the gold is seen")
    r1 = np.maximum(c.r, 1)
    ybar = _cell_sum(c, c.y, True) / r1
    f2 = _cell_sum(c, c.f, True) / r1
    f1 = _cell_sum(c, c.f, False) / np.maximum(c.m, 1)
    est = _CellEstimates(mu=ybar + lam * (f1 - f2), s2e=_cell_var(c, c.y - lam * c.f), lam=np.full(c.n_cells, lam))
    return _assemble(c, est, unit=unit, lam=lam, lam_rule="fixed", level=level)


LAM_RULE = (
    "leave-one-out cross-fit: each gold item's residual uses the lam fitted without it, the minimizer of the estimated "
    "phase-2 variance, sum_k q_k S_yf,k / sum_k q_k S2_f,k with q_k = (W_h m_k / n_h)^2 (1 - r_k / m_k) / r_k, S_yf from "
    "the other gold items of cells with at least 2 of them and S2_f from phase 1; clipped to [0, 1]; lam = 1 where the "
    "other gold items cannot estimate it"
)
LAM_ASSUMPTIONS = (
    "The point estimate is exactly design-unbiased: given phase 1 and every other gold item, the held-out item is a "
    "simple random draw from its stratum's remaining phase-1 items, so its expanded residual is unbiased for theirs "
    "whatever lam the others fitted. The variance is the fixed-lam formula at the held-out residuals, valid as the gold "
    "sample grows and the fitted lams settle (PPI++'s asymptotics, arXiv:2311.01453 §6); its finite-sample coverage is "
    "measured by simulation (addendum A07 §8.3)."
)


@dataclass(frozen=True)
class TunedFrameMean:
    """The power-tuned estimate (addendum A07 §8.3), reported beside the primary lam = 1 estimate, never in its place."""

    result: FrameMean
    lam_mean: float
    lam_min: float
    lam_max: float
    lam_untuned_items: int  # gold items whose held-out lam fell back to 1
    assumptions: str


def two_phase_difference_tuned(
    *,
    frame_size: int,
    unit: str,
    phase1_frame_sizes: Mapping[str, int],
    phase1_sample_sizes: Mapping[str, int],
    phase2_cuts: Sequence[float],
    phase2_allocation: Mapping[tuple[str, int], int],
    phase1_stratum: Sequence[str] | ArrayLike,
    stratifier: ArrayLike,
    f: ArrayLike,
    gold: ArrayLike,
    y: ArrayLike,
    level: float = 0.95,
) -> TunedFrameMean:
    """The difference estimator with lam fitted by leave-one-out cross-fitting, never on the gold item it corrects.

    For a gold item i in phase-2 stratum k (m_k phase-1 items, r_k gold), with lam_i fitted on every other gold item,
    the stratum's phase-1 total of y is estimated by sum_{gold k} y + lam_i (F1_k - F2_k) + (m_k - r_k)(y_i - lam_i f_i),
    F1 and F2 the phase-1 and gold totals of f; the stratum's estimate is the mean of these over its gold items. Given
    phase 1 and the other gold items, i is uniform over the m_k - r_k + 1 phase-1 items not among them, so each term is
    unbiased for the stratum total whatever lam_i is: the estimate is exactly design-unbiased (our reasoning; proved by
    enumeration in tests/discovery/test_two_phase_exact.py). With every lam_i equal it is the fixed-lam estimate.
    """
    c = _resolve(
        frame_size=frame_size,
        phase1_frame_sizes=phase1_frame_sizes,
        phase1_sample_sizes=phase1_sample_sizes,
        phase2_cuts=phase2_cuts,
        phase2_allocation=phase2_allocation,
        phase1_stratum=phase1_stratum,
        stratifier=stratifier,
        f=f,
        gold=gold,
        y=y,
    )
    K = c.n_cells
    w = c.cell_weight()
    open_ = (c.m > 0) & (c.r < c.m)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(open_, w * w * (1 - c.r / c.m) / c.r, 0.0)
    s2f = _phase1_var(c, c.f)

    gi = np.flatnonzero(c.gold)
    kg = c.unit_cell[gi]
    yg, fg = c.y[gi], c.f[gi]
    Sy = np.bincount(kg, weights=yg, minlength=K)
    Sf = np.bincount(kg, weights=fg, minlength=K)
    Syf = np.bincount(kg, weights=yg * fg, minlength=K)
    r = c.r

    def cov(n: FloatArray, sy: FloatArray, sf: FloatArray, syf: FloatArray) -> FloatArray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n >= 2, (syf - sy * sf / n) / (n - 1), 0.0)

    usable = (r >= 2) & (q > 0)
    cov_full = cov(r, Sy, Sf, Syf)
    num_full = float(np.sum(np.where(usable, q * cov_full, 0.0)))
    den_full = float(np.sum(np.where(usable, q * s2f, 0.0)))
    # Leave item i out of its own cell; every other cell is as it is.
    rk = r[kg] - 1
    cov_i = cov(rk, Sy[kg] - yg, Sf[kg] - fg, Syf[kg] - yg * fg)
    usable_i = (rk >= 2) & (q[kg] > 0)
    num_i = num_full - np.where(usable[kg], q[kg] * cov_full[kg], 0.0) + np.where(usable_i, q[kg] * cov_i, 0.0)
    den_i = den_full - np.where(usable[kg], q[kg] * s2f[kg], 0.0) + np.where(usable_i, q[kg] * s2f[kg], 0.0)
    fitted = den_i > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        lam_i = np.where(fitted, np.clip(num_i / den_i, 0.0, 1.0), 1.0)

    # Each cell's estimate of its phase-1 mean of y: (1/m)[Y2 + mean(lam_i)(F1 - F2) + (m - r)(ybar2 - mean(lam_i f_i))].
    r1 = np.maximum(r, 1)
    lam_bar = np.bincount(kg, weights=lam_i, minlength=K) / r1
    lf_bar = np.bincount(kg, weights=lam_i * fg, minlength=K) / r1
    F1 = _cell_sum(c, c.f, False)
    m1 = np.maximum(c.m, 1)
    mu = (Sy + lam_bar * (F1 - Sf) + (c.m - r) * (Sy / r1 - lf_bar)) / m1
    resid = np.zeros(len(c.f))
    resid[gi] = yg - lam_i * fg
    est = _CellEstimates(mu=mu, s2e=_cell_var(c, resid), lam=lam_bar)
    lam_mean = float(lam_i.mean()) if len(lam_i) else 1.0
    result = _assemble(c, est, unit=unit, lam=lam_mean, lam_rule=LAM_RULE, level=level)
    return TunedFrameMean(
        result=result,
        lam_mean=lam_mean,
        lam_min=float(lam_i.min()) if len(lam_i) else 1.0,
        lam_max=float(lam_i.max()) if len(lam_i) else 1.0,
        lam_untuned_items=int(np.sum(~fitted)),
        assumptions=LAM_ASSUMPTIONS,
    )
