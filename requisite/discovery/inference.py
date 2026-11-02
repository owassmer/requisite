"""Wide-mode inference: the frame share of items whose indicator holds, and the share of one indicator's items in
another's, estimated under the design that drew the samples.

The design admitted here is two-phase. Phase 1 is a stratified simple random sample without replacement (SRSWOR) of
the frame; every phase-1 item carries a model prediction f in [0, 1], fixed before any gold label is seen (spec §7.4:
probabilities, never top answers). Phase 2 is the gold sample: within each phase-1 stratum, the phase-1 items are split
into phase-2 strata by thresholding a phase-1 value (the stratifier, a model probability), and each phase-2 stratum is
an SRSWOR of its phase-1 items, of size min(allocation, items) with allocation >= 2, so a phase-2 stratum with no more
items than its allocation is taken whole. A frame small enough to read whole is a phase-1 census, and then the estimate,
its variance and its standard error are exactly addendum A07's finite-frame form (§8.2, §8.4).

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
- The interval is a score interval (Wilson 1927, "Probable inference, the law of succession, and statistical
  inference", JASA 22: the variance is evaluated at the value tested, not at the estimate), carried to this design by our
  reasoning: see _score_bounds. Its coverage is measured by simulation (tests/discovery/test_wide_coverage.py), as A07 §8.5
  asks, and is a regression result for the settings simulated, not a guarantee for every design.
- The ratio of two such shares, the numerator's items a subset of the denominator's (two_phase_ratio), is tested
  through the combination y_num - R y_den (Fieller 1954, "Some problems in interval estimation", JRSS B 16): its
  plain difference estimate and the design's own unbiased variance (_design_covariance, bilinear, so exact for every
  fixed R; tests/discovery/test_two_phase_ratio.py enumerates it). Fieller's set is reported as it is, unbounded
  where the denominator is not distinguishable from zero; the interval to read is the score set, the variance evaluated
  at category probabilities restricted to each tested R (_RatioScore, our reasoning from Wilson's, as for the mean, with
  Owen's empirical-likelihood weights, Biometrika 75, 1988, for the path), whose coverage is measured
  (tests/discovery/test_wide_ratio_coverage.py).
- Spec §7.4 and PPI++ (Angelopoulos, Duchi and Zrnic, arXiv:2311.01453): the two-sample PPI++ mean, for a design the
  caller declares as two independent samples and only there (A07 §8.2: a gold subset of a frame is another design).

The unit is the item: an estimate is a share of items, never of the businesses that filed them (A07 §8.4).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

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

    def open_cells(self) -> NDArray[np.bool_]:
        """Cells whose gold is a sample short of the cell, so that phase 2 adds variance there."""
        return (self.m > 0) & (self.r < self.m)

    def phase2_factor(self) -> FloatArray:
        """(1 - r_k / m_k) / r_k: the SRSWOR variance factor of a cell's gold mean; 0 for a cell taken whole."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.open_cells(), (1 - self.r / self.m) / self.r, 0.0)


@dataclass(frozen=True)
class _Phase2Strata:
    """The phase-2 strata of the phase-1 items: each item's phase-1 stratum and cell (phase-1 stratum crossed with the
    stratifier's bin), and each cell's allocation."""

    stratum: NDArray[np.intp]
    cell: NDArray[np.intp]
    bins: int
    n_cells: int
    allocation: FloatArray


def _phase2_strata(
    names: tuple[str, ...],
    phase1_stratum: NDArray[np.str_],
    stratifier: FloatArray,
    phase2_cuts: Sequence[float],
    phase2_allocation: Mapping[tuple[str, int], int],
) -> _Phase2Strata:
    """The one definition of a phase-2 stratum, for drawing the gold sample and for estimating from it: bin b of the
    stratifier holds cuts[b-1] <= value < cuts[b], within each phase-1 stratum; each stratum holding a phase-1 item is
    allocated at least 2 gold items."""
    cuts = np.asarray(phase2_cuts, dtype=float)
    if cuts.ndim != 1 or np.any(~np.isfinite(cuts)) or np.any(np.diff(cuts) <= 0):
        raise ValueError("phase-2 cuts are finite and strictly increasing")
    if stratifier.shape != phase1_stratum.shape:
        raise ValueError("the stratifier has one value per phase-1 item")
    if np.any(~np.isfinite(stratifier)):
        raise ValueError("the stratifier is defined for every phase-1 item")
    declared = np.array(names, dtype=str)
    order = np.argsort(declared)
    pos = np.minimum(np.searchsorted(declared[order], phase1_stratum), len(names) - 1)
    known = declared[order][pos] == phase1_stratum
    if not np.all(known):
        raise ValueError(f"phase-1 item in undeclared stratum {phase1_stratum[~known][0]!r}")
    h = order[pos].astype(np.intp)
    bins = len(cuts) + 1
    n_cells = len(names) * bins
    cell = h * bins + np.searchsorted(cuts, stratifier, side="right").astype(np.intp)
    index = {name: i for i, name in enumerate(names)}
    alloc = np.zeros(n_cells)
    for key, a_k in phase2_allocation.items():
        name, bin_ = key
        if name not in index or not 0 <= bin_ < bins:
            raise ValueError(f"phase-2 allocation for an undeclared stratum {key}")
        if a_k < 2:
            raise ValueError(f"phase-2 stratum {key} is allocated {a_k} gold items; the design allocates at least 2")
        alloc[index[name] * bins + bin_] = a_k
    missing = (np.bincount(cell, minlength=n_cells) > 0) & (alloc == 0)
    if np.any(missing):
        k = int(np.flatnonzero(missing)[0])
        raise ValueError(f"phase-2 stratum {(names[k // bins], k % bins)} has phase-1 items but no allocation")
    return _Phase2Strata(stratum=h, cell=cell, bins=bins, n_cells=n_cells, allocation=alloc)


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
    if not names:
        raise ValueError("the design declares at least one phase-1 stratum")
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

    strat = np.asarray(phase1_stratum).astype(str)
    s = np.asarray(stratifier, dtype=float)
    fv = np.asarray(f, dtype=float)
    g = np.asarray(gold)
    yv = np.asarray(y, dtype=float)
    n = strat.shape[0]
    for label, a in (("stratifier", s), ("f", fv), ("gold", g), ("y", yv)):
        if a.shape != (n,):
            raise ValueError(f"{label} has shape {a.shape}; one value per phase-1 item ({n}) is required")
    if g.dtype != np.bool_:
        raise ValueError("gold is a boolean mask over the phase-1 items")
    if np.any(~np.isfinite(fv)) or np.any((fv < 0) | (fv > 1)):
        raise ValueError("f is a probability in [0, 1] for every phase-1 item (spec §7.4)")
    if np.any((yv[g] != 0) & (yv[g] != 1)):
        raise ValueError("every gold item carries an indicator label, 0 or 1 (an unresolved gold label is not one)")

    p2 = _phase2_strata(names, strat, s, phase2_cuts, phase2_allocation)
    counts = np.bincount(p2.stratum, minlength=len(names))
    if np.any(counts != n_h):
        raise ValueError(
            f"phase-1 items per stratum {counts.tolist()} differ from the declared {n_h.astype(int).tolist()}"
        )
    m = np.bincount(p2.cell, minlength=p2.n_cells).astype(float)
    r = np.bincount(p2.cell[g], minlength=p2.n_cells).astype(float)
    alloc, bins, n_cells, unit_cell = p2.allocation, p2.bins, p2.n_cells, p2.cell
    want = np.minimum(alloc, m)
    if np.any(r != want):
        k = int(np.flatnonzero(r != want)[0])
        raise ValueError(
            f"phase-2 stratum {(names[k // bins], k % bins)} has {int(r[k])} gold items; its allocation of "
            f"{int(alloc[k])} over {int(m[k])} phase-1 items draws {int(want[k])}"
        )
    return _Cells(
        names=names,
        frame_sizes=N_h,
        sample_sizes=n_h,
        frame_size=frame_size,
        cell_stratum=(np.arange(n_cells) // bins).astype(np.intp),
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


def _gold_cov(c: _Cells, u: FloatArray, v: FloatArray) -> FloatArray:
    """Sample covariance (ddof 1) of u and v over each cell's gold items; 0 where a cell has fewer than 2."""
    r1 = np.maximum(c.r, 1)
    du = u - (_cell_sum(c, u, True) / r1)[c.unit_cell]
    dv = v - (_cell_sum(c, v, True) / r1)[c.unit_cell]
    ss = _cell_sum(c, du * dv, True)
    return np.where(c.r >= 2, ss / np.maximum(c.r - 1, 1), 0.0)


def _design_covariance(
    c: _Cells, u: FloatArray, eu: FloatArray, v: FloatArray, ev: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """The design's estimate of Cov(theta_hat_u, theta_hat_v), per phase-1 stratum, phase 1 and phase 2: for two
    difference estimates on the same sample, of targets u and v (read on the gold) with residuals eu and ev. Bilinear
    and symmetric, so the variance of any linear combination of estimates is this of the combination; with u = v,
    eu = ev, it is the estimator's variance. Exactly unbiased for any targets, 0/1 or not.

    Phase 2, on the residuals: sum_k (W_h m_k / n_h)^2 (1 - r_k / m_k) s_eu,ev,k / r_k over cells; 0 for a cell taken
    whole. Phase 1, on the targets: W_h^2 (1 - n_h / N_h) S_uv,h / n_h, with S_uv,h (the phase-1 sample covariance of u
    and v in the stratum) estimated without bias from the gold as [Q - (T_u T_v - C_T) / n_h] / (n_h - 1): Q = sum_k m_k
    mean_gold,k(u v) estimates the stratum's phase-1 total of u v, T_u = sum_k m_k mean_gold,k(u) its total of u, and
    C_T = sum_k m_k^2 (1 - r_k / m_k) s_uv,k / r_k the phase-2 covariance estimate of T_u and T_v, so that T_u T_v - C_T
    is unbiased for the product of the totals. For 0/1 targets with u = v, Q = T and this is [T - (T^2 - V(T)) / n_h] /
    (n_h - 1), which is >= 0 since 0 <= T <= n_h."""
    H = len(c.names)
    h = c.cell_stratum
    w = c.cell_weight()
    q = c.phase2_factor()
    W = c.frame_sizes / c.frame_size
    n_h = c.sample_sizes
    v2 = np.bincount(h, weights=w * w * q * _gold_cov(c, eu, ev), minlength=H)
    r1 = np.maximum(c.r, 1)
    Tu = np.bincount(h, weights=c.m * _cell_sum(c, u, True) / r1, minlength=H)
    Tv = np.bincount(h, weights=c.m * _cell_sum(c, v, True) / r1, minlength=H)
    Q = np.bincount(h, weights=c.m * _cell_sum(c, u * v, True) / r1, minlength=H)
    CT = np.bincount(h, weights=c.m * c.m * q * _gold_cov(c, u, v), minlength=H)
    with np.errstate(invalid="ignore", divide="ignore"):
        s_uv = np.where(n_h >= 2, (Q - (Tu * Tv - CT) / n_h) / (n_h - 1), 0.0)
    v1 = W * W * (1 - n_h / c.frame_sizes) * s_uv / n_h
    return v1, v2


def _phase1_var(c: _Cells, values: FloatArray) -> FloatArray:
    """Sample variance (ddof 1) of values over each cell's phase-1 items; 0 where a cell has fewer than 2."""
    m = c.m
    mean = _cell_sum(c, values, False) / np.maximum(m, 1)
    dev = values - mean[c.unit_cell]
    ss = _cell_sum(c, dev * dev, False)
    return np.where(m >= 2, ss / np.maximum(m - 1, 1), 0.0)


# -------------------------------------------------------------------------------------------------------------- interval


@dataclass(frozen=True)
class RatioSet:
    """A set of ratios: a union of closed intervals in increasing order, an end possibly infinite. ``kind`` names its
    shape, read from the pieces: "empty"; "interval" (one bounded piece, a single point where both ends agree); "ray"
    (one piece with one infinite end); "two_rays" (the line without an open interval); "line" (every real number);
    "union" (any other several pieces). ``reason`` says why it has that shape."""

    pieces: tuple[tuple[float, float], ...]
    kind: Literal["empty", "interval", "ray", "two_rays", "line", "union"]
    reason: str

    def __contains__(self, value: object) -> bool:
        return isinstance(value, int | float) and any(lo <= value <= hi for lo, hi in self.pieces)

    @property
    def width(self) -> float:
        """The total length of the pieces (infinite for an unbounded set)."""
        return float(sum(hi - lo for lo, hi in self.pieces))


def _ratio_set(pieces: Sequence[tuple[float, float]], reason: str) -> RatioSet:
    ps = tuple(sorted((float(lo), float(hi)) for lo, hi in pieces))
    kind: Literal["empty", "interval", "ray", "two_rays", "line", "union"]
    if not ps:
        kind = "empty"
    elif len(ps) == 1:
        lo, hi = ps[0]
        unbounded = int(lo == -np.inf) + int(hi == np.inf)
        kind = ("interval", "ray", "line")[unbounded]
    elif len(ps) == 2 and ps[0][0] == -np.inf and ps[1][1] == np.inf:
        kind = "two_rays"
    else:
        kind = "union"
    return RatioSet(pieces=ps, kind=kind, reason=reason)


# Categories of an item by its two labels, under the subset: 0 neither, 1 denominator only, 2 numerator (and so
# denominator). _LABELS[j, c] is indicator j's label (0 numerator, 1 denominator) in category c.
_LABELS = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]])


def _combination(R: FloatArray) -> FloatArray:
    """y_num - R y_den in each category, one row per R: (0, -R, 1 - R)."""
    out: FloatArray = np.stack([np.zeros_like(R), -R, 1.0 - R], axis=1)
    return out


def _tilt(pihat: FloatArray, lam: FloatArray, z: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The category probabilities pihat (one row per cell or stratum, or one such array per R) tilted by lam (one per
    cell and R) toward the categories where z (one row per R) is larger: pi_c = pihat_c / (1 - lam z_c + s), s fixed by
    the probabilities summing to one with every 1 - lam z_c + s >= 0 (Owen's empirical-likelihood weights, Biometrika
    1988, whose mean of z is monotone in lam); and d(mean of z)/d(lam). sum_c pihat_c / (1 - lam z_c + s) is decreasing
    and convex in s above the largest pole of a category with mass, so it has one root there, reached by Newton's method
    from the left. It is solved for the distance d of that root from the pole, u_c = 1 - lam z_c + s = gap_c + d with
    gap_c = lam z_top - lam z_c, so that a root next to its pole (a category with little mass, its weight pihat_c / d)
    keeps its precision. A category the estimate gives no mass takes the mass left over once the root would pass its
    pole: Wilson's larger root, the score interval's way out of an empty cell.

    Returned with the tilt: d(pi_c)/d(lam) at a fixed base, and d(mean of z)/d(pihat_c) at a fixed lam, (z_c - ds)/u_c
    for a category with mass (ds = ds/dlam; for moves of the base that keep it summing to one), so that a tilt whose
    base moves (a phase-1 stratum's, its base its cells') has its exact slope."""
    la = lam[..., None] * z[:, None, :]
    pos = np.broadcast_to(pihat > 0, la.shape)
    p = np.broadcast_to(pihat, la.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        la_pos = np.where(pos, la, -np.inf)  # a category without mass: an infinite denominator, so no term
        top = np.argmax(la_pos, axis=-1)[..., None]
        la_top = np.take_along_axis(la_pos, top, axis=-1)
        gap = np.where(pos, la_top - la, np.inf)
        # Two points left of the root: the top category's mass (that term alone is 1 there), and the first-order point
        # 1 - sum_c pihat_c gap_c (by Jensen, sum_c pihat_c / (gap_c + d) >= 1 there). The nearer is the larger; Newton
        # converges from it monotonically.
        first = 1.0 - np.sum(np.where(pos, p * gap, 0.0), axis=-1)
        d = np.maximum(np.take_along_axis(p, top, axis=-1)[..., 0], first)
        for _ in range(100):
            ratio = np.where(pos, p / (gap + d[..., None]), 0.0)
            step = (ratio.sum(axis=-1) - 1.0) / np.sum(ratio / (gap + d[..., None]), axis=-1)
            d = d + step
            if np.all(np.abs(step) <= 1e-14 * d):
                break
        pole = np.argmax(la, axis=-1)[..., None]  # the largest pole of any category: where its u is zero
        d_pole = np.take_along_axis(la, pole, axis=-1)[..., 0] - la_top[..., 0]
        at_pole = d < d_pole
        d = np.maximum(d, d_pole)
        u = np.where(pos, gap + d[..., None], 1.0)
        pi = np.where(pos, p / u, 0.0)
        # ds/dlam: off the pole from differentiating sum_c pihat_c / u_c = 1, on it the pole category's z.
        w2 = np.where(pos, p / (u * u), 0.0)
        ds = np.where(at_pole, np.take_along_axis(np.broadcast_to(z[:, None, :], la.shape), pole, axis=-1)[..., 0],
                      np.sum(w2 * z[:, None, :], axis=-1) / np.sum(w2, axis=-1))  # fmt: skip
        dpi = np.where(pos, pi * (z[:, None, :] - ds[..., None]) / u, 0.0)
    rest = np.where(at_pole, np.maximum(1.0 - pi.sum(axis=-1), 0.0), 0.0)  # the empty category at the largest pole
    np.put_along_axis(pi, pole, np.take_along_axis(pi, pole, axis=-1) + rest[..., None], axis=-1)
    absorbed = np.where(at_pole, -dpi.sum(axis=-1), 0.0)
    np.put_along_axis(dpi, pole, np.take_along_axis(dpi, pole, axis=-1) + absorbed[..., None], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sens: FloatArray = np.where(pos, (z[:, None, :] - ds[..., None]) / u, 0.0)
    out: FloatArray = pi / pi.sum(axis=-1, keepdims=True)
    return out, dpi, sens


@dataclass(frozen=True)
class _Score:
    """The variance of theta_num_hat - R theta_den_hat as a function of the category probabilities (neither,
    denominator only, numerator) of each phase-2 stratum (a cell) and of each phase-1 stratum's frame, and the point
    where they are restricted to a tested R: the score test of R. A mean is the case where every item is in the
    denominator (_mean_score).

    Phase 2. In each cell the gold's residuals of y_num - R y_den (label less prediction) are a mixture over the
    categories: category c has mean residual alpha_c = z_c - (mf_num,c - R mf_den,c), z = (0, -R, 1 - R), mf the mean
    predictions of the cell's gold items in each category, and the predictions' within-category covariance is the cell's
    pooled one. So the cell's residual variance at category probabilities pi is kappa Var_pi(alpha) + g' Wc g,
    g = (1, -R), kappa = r / (r - 1), Wc the within-category sum of products over r - 1: non-negative by construction,
    and at the gold's own category frequencies exactly the plain sample variance of the residuals, which the design's
    unbiased variance (_design_covariance) reads. A category the cell's gold has not shown takes, for each indicator's
    prediction, the mean over the cell's gold items sharing its label for that indicator, or the gold's mean where none
    does: a prediction is credited with separating a category only as far as the cell's own gold shows.

    Phase 1: W_h^2 (1 - n_h / N_h) / (n_h - 1) Var_Pi(z) in each phase-1 stratum, at its frame category probabilities
    Pi_h: the phase-1 sample variance of y_num - R y_den, which the design's variance reads, where the frame is Pi_h.

    The restricted point. One t for every cell and phase-1 stratum, the one at which the frame mean of y_num - R y_den,
    sum_h W_h Pi_h . z, is zero. Each cell's probabilities are its estimate tilted toward the categories where
    y_num - R y_den is larger (or smaller) by _tilt with lam_k = t w_k q_k kappa_k, and each phase-1 stratum's frame
    probabilities are its phase-1 items' (its cells' at t, by their shares) tilted the same way with lam_h = t k1_h / W_h:
    each mean of y_num - R y_den moves by about t times its weight in the estimate times its own variance, as a weighted
    least-squares fit at the tested value would move it (Miettinen and Nurminen's restricted estimates for stratified
    rates, Stat Med 1985, are the same idea; their use here is our reasoning). For one cell of a mean it is Wilson's
    path, p - mu = t w q kappa p (1 - p). A cell taken whole does not move, nor does a phase-1 stratum read whole; where
    phase 1 is a sample, the frame moves beyond its phase-1 items, so every R a frame could have is reached. The path
    moves the labels only: the predictions enter the variance, where the gold shows what they separate, not where the
    path goes (the alternatives measured are in test_wide_ratio_coverage.py's notes)."""

    w: FloatArray  # W_h m_k / n_h, 0 where the cell is empty
    q: FloatArray  # (1 - r_k / m_k) / r_k, 0 where taken whole
    kappa: FloatArray  # r_k / (r_k - 1)
    pihat: FloatArray  # (K, 3) the cells' estimated category probabilities
    mf: FloatArray  # (K, 2, 3) category mean predictions (numerator's, denominator's)
    within: FloatArray  # (K, 2, 2)
    stratum: NDArray[np.intp]  # each cell's phase-1 stratum
    share: FloatArray  # m_k / n_h
    W: FloatArray  # (H,) W_h, each phase-1 stratum's weight in the estimate
    k1: FloatArray  # (H,) W_h^2 (1 - n_h / N_h) / (n_h - 1), 0 where read whole
    theta: tuple[float, float]  # theta_num_hat, theta_den_hat

    def _to_stratum(self) -> FloatArray:
        out = np.zeros((len(self.w), len(self.W)))
        out[np.arange(len(self.w)), self.stratum] = self.share
        return out

    def _rates(self) -> tuple[FloatArray, FloatArray]:
        """Per unit of t, each cell's tilt w q kappa and each phase-1 stratum's k1 / W."""
        with np.errstate(divide="ignore", invalid="ignore"):
            strata = np.where(self.W > 0, self.k1 / np.where(self.W > 0, self.W, 1.0), 0.0)
        return self.w * self.q * self.kappa, strata

    def sampled(self) -> bool:
        """Whether anything is sampled short of the frame (else the estimate is the frame's value)."""
        return bool(np.any(self.w * self.q > 0) or np.any(self.k1 > 0))

    def domain(self, i: int, theta: tuple[float, float]) -> _Score:
        """The same test for phase-1 stratum i alone, as a frame of its own (its estimates ``theta``)."""
        k = self.stratum == i
        Wi = float(self.W[i])
        return _Score(
            w=self.w[k] / Wi,
            q=self.q[k],
            kappa=self.kappa[k],
            pihat=self.pihat[k],
            mf=self.mf[k],
            within=self.within[k],
            stratum=np.zeros(int(k.sum()), dtype=np.intp),
            share=self.share[k],
            W=np.ones(1),
            k1=self.k1[i : i + 1] / Wi**2,
            theta=theta,
        )

    def _limit(self, end: NDArray[np.intp]) -> tuple[FloatArray, FloatArray]:
        """The cells' and the strata's probabilities as t grows without bound, one row per entry of ``end``: everything
        that moves ends at category end, a cell taken whole stays as it is, and a phase-1 stratum read whole is its
        cells'."""
        cell, strata = self._rates()
        e = np.eye(3)[end]
        pi = np.where((cell > 0)[None, :, None], e[:, None, :], self.pihat[None])
        P = np.einsum("nkc,kh->nhc", pi, self._to_stratum())
        return pi, np.where((strata > 0)[None, :, None], e[:, None, :], P)

    def reach(self) -> tuple[float, float]:
        """The ratios some arrangement of the categories can make the frame's: at the upper end everything that moves is
        in the numerator, at the lower end in the denominator only."""
        ends = []
        for end, empty in ((1, 0.0), (2, 1.0)):
            Pi = self._limit(np.array([end]))[1][0]
            den = float(self.W @ (Pi[:, 1] + Pi[:, 2]))
            ends.append(float(self.W @ Pi[:, 2]) / den if den > 0 else empty)
        return (ends[0], ends[1])

    def _path(self, t: FloatArray, z: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """At each t (one per R): the cells' and strata's probabilities, the frame mean of y_num - R y_den, and its
        slope in t (each stratum's tilt moving with lam and with its base, its cells')."""
        cell, strata = self._rates()
        to_h = self._to_stratum()
        pi, dpi, _ = _tilt(self.pihat, t[:, None] * cell[None], z)
        P = np.einsum("nkc,kh->nhc", pi, to_h)
        dP = np.einsum("nkc,kh->nhc", dpi * cell[None, :, None], to_h)  # d(P)/dt
        Pi, dPi, sens = _tilt(P, t[:, None] * strata[None], z)
        slope = np.einsum("h,nhc,nc->n", self.W * strata, dPi, z) + np.einsum("h,nhc->n", self.W, sens * dP)
        return pi, Pi, np.einsum("h,nhc,nc->n", self.W, Pi, z), slope

    def restricted(
        self, R: FloatArray, guess: FloatArray | None = None
    ) -> tuple[FloatArray, FloatArray, NDArray[np.bool_], FloatArray]:
        """The cells' and strata's probabilities at the t where the frame mean of y_num - R y_den is zero, one row per
        R, and whether the path reaches it (it does not where no arrangement of the categories makes R the frame's
        ratio, as when what is read whole already holds more of one side than the rest can offset). The null is
        monotone in t; its root is found by Newton's method kept inside the bracket the iterates establish, from
        ``guess`` where one is given (a root at a nearby R), and returned with the probabilities."""
        z = _combination(R)
        cell, strata = self._rates()
        n = len(R)
        # At t = 0 the cells are at their estimate and each frame at its phase-1 items: the null and its slope there.
        P0 = self._to_stratum().T @ self.pihat
        pi0 = np.broadcast_to(self.pihat, (n, *self.pihat.shape))
        Pi0 = np.broadcast_to(P0, (n, *P0.shape))
        zk, zh = self.pihat @ z.T, P0 @ z.T  # (K, n), (H, n)
        f0 = self.W @ zh
        slope = (self.w * cell) @ (self.pihat @ (z * z).T - zk * zk) + (self.W * strata) @ (P0 @ (z * z).T - zh * zh)
        # As |t| grows, everything that moves ends at its category where y_num - R y_den is largest (t > 0: the
        # numerator) or smallest (t < 0: denominator only), unique for R in (0, 1); the null at that limit decides
        # whether a finite t reaches zero.
        lim_pi, lim_Pi = self._limit(np.where(f0 < 0, 2, 1))
        f_limit = np.einsum("h,nhc,nc->n", self.W, lim_Pi, z)
        at_limit = (f0 != 0) & (np.abs(f_limit) <= 1e-13 * np.sum(self.W))  # reached only as t grows without bound
        reached = (f0 == 0) | (np.sign(f_limit) == -np.sign(f0)) | at_limit
        active = reached & ~at_limit & (f0 != 0)
        direction = -np.sign(f0)
        lo = np.zeros_like(R)  # the null keeps f0's sign at lo and has crossed it at hi, once bracketed
        hi = np.full_like(R, np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(slope > 0, -f0 / np.where(slope > 0, slope, 1.0), direction)
        if guess is not None:
            t = np.where(np.isfinite(guess) & (guess * direction > 0), guess, t)
        pi, Pi = pi0, Pi0
        tol = 1e-12 * np.sum(self.W)
        for _ in range(400):
            if not np.any(active):
                break
            pi_t, Pi_t, f_t, d_t = self._path(np.where(active, t, 0.0), z)  # a converged entry's t stays, unused
            pi = np.where(active[:, None, None], pi_t, pi)
            Pi = np.where(active[:, None, None], Pi_t, Pi)
            same = np.sign(f_t) == np.sign(f0)
            lo = np.where(active & same, t, lo)
            hi = np.where(active & ~same, t, hi)
            bracketed = np.isfinite(hi)
            with np.errstate(divide="ignore", invalid="ignore"):
                newton = t - f_t / d_t
                between = (newton - lo) * (newton - hi) < 0
                beyond = (newton - t) * direction > 0
            nxt = np.where(bracketed, np.where(between, newton, 0.5 * (lo + hi)),
                           np.where(beyond & np.isfinite(newton), np.minimum(np.abs(newton), 4 * np.abs(t)) * direction,
                                    2.0 * t))  # fmt: skip
            done = (np.abs(f_t) <= tol) | (bracketed & (np.abs(hi - lo) <= 1e-12 * np.maximum(1.0, np.abs(hi))))
            active &= ~done
            t = np.where(active, nxt, t)
        still = (f0 == 0)[:, None, None]
        pi = np.where(at_limit[:, None, None], lim_pi, np.where(still, pi0, pi))
        Pi = np.where(at_limit[:, None, None], lim_Pi, np.where(still, Pi0, Pi))
        return pi, Pi, reached, np.where(reached & ~at_limit & (f0 != 0), t, np.nan)

    def variance(self, R: FloatArray, pi: FloatArray, Pi: FloatArray) -> FloatArray:
        """The variance of theta_num_hat - R theta_den_hat at the cells' and strata's probabilities (one row per R)."""
        z = _combination(R)
        g = np.stack([np.ones_like(R), -R], axis=1)
        alpha = z[:, None, :] - (self.mf[None, :, 0, :] - R[:, None, None] * self.mf[None, :, 1, :])  # (n, K, 3)
        dev = alpha - np.sum(pi * alpha, axis=2, keepdims=True)
        between = np.sum(pi * dev * dev, axis=2)
        within = np.einsum("nj,kjl,nl->nk", g, self.within, g)
        v2 = (self.kappa[None] * between + within) @ (self.w * self.w * self.q)
        zdev = z[:, None, :] - np.einsum("nhc,nc->nh", Pi, z)[:, :, None]
        out: FloatArray = v2 + np.sum(Pi * zdev * zdev, axis=2) @ self.k1
        return out

    def statistic(self, R: ArrayLike, guess: FloatArray | None = None) -> tuple[FloatArray, FloatArray]:
        """(theta_num_hat - R theta_den_hat)^2 / V_R for each R in [0, 1], V_R the variance at the restricted point,
        infinite where the path does not reach it; and the path's t there (restricted). At R = 0 and 1, where two
        categories tie for the extreme of y_num - R y_den and the path's limit depends on how R approaches, it is the
        limit from inside [0, 1], evaluated 1e-9 inside."""
        Rv = np.clip(np.atleast_1d(np.asarray(R, dtype=float)), 1e-9, 1.0 - 1e-9)
        pi, Pi, reached, t = self.restricted(Rv, guess)
        v = self.variance(Rv, pi, Pi)
        tz = self.theta[0] - Rv * self.theta[1]
        scale = max(abs(self.theta[0]), abs(self.theta[1]), 1e-300)
        exact = np.where(np.abs(tz) <= 1e-12 * scale, 0.0, np.inf)  # no variance: only the estimate itself fits
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = np.where(v > 0, tz * tz / np.where(v > 0, v, 1.0), exact)
        out: FloatArray = np.where(reached, stat, np.inf)
        return out, t


def _score(
    c: _Cells, category: NDArray[np.intp], predictions: FloatArray, pihat: FloatArray, theta: tuple[float, float]
) -> _Score:
    """The score model of a numerator over a denominator on the cells of c: each gold item's ``category`` (0 neither,
    1 denominator only, 2 numerator), the ``predictions`` (2, n) the estimator subtracts from the numerator's and the
    denominator's labels, the cells' estimated category probabilities ``pihat`` (K, 3), and the estimates ``theta``."""
    K = c.n_cells
    mf = np.zeros((K, 2, 3))
    within = np.zeros((K, 2, 2))
    for k in range(K):
        idx = np.flatnonzero(c.gold & (c.unit_cell == k))
        if len(idx) == 0:
            continue
        ck, fk = category[idx], predictions[:, idx]
        for j in range(2):
            for cat in range(3):
                own = ck == cat
                same_label = _LABELS[j, ck] == _LABELS[j, cat]
                pick = own if own.any() else same_label if same_label.any() else np.ones_like(own)
                mf[k, j, cat] = fk[j, pick].mean()
        dev = fk - mf[k][:, ck]
        within[k] = dev @ dev.T / max(len(idx) - 1, 1)
    occupied = c.m > 0
    n_h, N_h = c.sample_sizes, c.frame_sizes
    W = N_h / c.frame_size
    with np.errstate(invalid="ignore", divide="ignore"):
        k1 = np.where(N_h > n_h, W * W * (N_h - n_h) / (n_h * (N_h - 1)), 0.0)
    return _Score(
        w=np.where(occupied, c.cell_weight(), 0.0),
        q=c.phase2_factor(),
        kappa=np.where(c.m >= 2, c.m / np.maximum(c.m - 1, 1), 1.0),
        pihat=np.where(occupied[:, None], _simplex(pihat), np.array([1.0, 0.0, 0.0])),
        mf=mf,
        within=within,
        stratum=c.cell_stratum,
        share=c.m / n_h[c.cell_stratum],
        W=W,
        k1=k1,
        theta=theta,
    )


def _ratio_score(cn: _Cells, cd: _Cells, theta: tuple[float, float]) -> _Score:
    """The score model of one numerator over the shared denominator (the two resolved on the same cells), each
    indicator's prediction its f (lam = 1) and each cell's probabilities its plain difference estimates."""
    xn, xd = _difference_means(cn, 1.0), _difference_means(cd, 1.0)
    return _score(
        cn,
        np.where(cn.gold, cd.y + cn.y, -1).astype(np.intp),
        np.stack([cn.f, cd.f]),
        np.stack([1.0 - xd, xd - xn, xn], axis=1),
        theta,
    )


def _mean_score(c: _Cells, est: _CellEstimates, theta: float) -> _Score:
    """The score model of a frame mean: the ratio over a denominator that holds every item, its label 1 predicted
    exactly, so that y_num - R y_den is y - R and the test is of the frame mean R. A gold item is in category 1 (label 0)
    or 2 (label 1); its prediction is the one the estimator subtracts, y - e for its residual e: lam f for a fixed lam,
    lam_i f_i for the lam cross-fitted without it. So lam enters the mixture through the category means and the
    within-category spread of lam f alone, and a cell's variance at prevalence p is kappa p (1 - p)(1 - lam delta)^2 +
    lam^2 S2_f,within, delta the gap in the mean f of its gold's two labels: exactly the residual variance of y - lam f.
    The cells' probabilities are their estimates of their phase-1 prevalences, within [0, 1]."""
    return _score(
        c,
        np.where(c.gold, 1 + c.y, -1).astype(np.intp),
        np.stack([c.y - est.resid, np.ones_like(c.f)]),
        np.stack([np.zeros_like(est.mu), 1.0 - est.mu, est.mu], axis=1),
        (theta, 1.0),
    )


def _simplex(v: FloatArray) -> FloatArray:
    """Each row's Euclidean projection onto the probability simplex (a difference estimate may leave it)."""
    u = -np.sort(-v, axis=1)
    css = np.cumsum(u, axis=1) - 1.0
    ind = np.arange(1, v.shape[1] + 1)
    rho = v.shape[1] - 1 - np.argmax(((u - css / ind) > 0)[:, ::-1], axis=1)
    shift = css[np.arange(len(v)), rho] / (rho + 1)
    out: FloatArray = np.maximum(v - shift[:, None], 0.0)
    return out


def _score_set(sc: _Score, z: float, estimate: float | None, se: float | None) -> RatioSet:
    """{R in [0, 1]: statistic(R) <= z^2}: the statistic on a grid over [0, 1], with points about the estimate at
    multiples of its linearized standard error so that a short interval is not stepped over, and each crossing between
    grid points refined by Newton's method to 1e-8. The grid spans only the ratios the cells can reach (reach)."""
    low, high = max(sc.reach()[0], 0.0), min(sc.reach()[1], 1.0)  # outside, no arrangement of the cells fits
    if low > high:
        return _ratio_set([], "no arrangement of the strata's categories makes a ratio in [0, 1] the frame's")
    grid = [float(x) for x in np.linspace(low, high, 21)]
    if estimate is not None and np.isfinite(estimate):
        grid.append(estimate)
        if se is not None and np.isfinite(se) and se > 0:  # densest where a bound is likeliest, about z se away
            steps = (0.5, 1.0, 1.5, 0.8 * z, 0.9 * z, z, 1.1 * z, 1.2 * z, 1.35 * z, 1.5 * z, 1.75 * z, 2 * z, 3 * z)
            grid += [estimate + sign * k * se for k in steps for sign in (-1, 1)]
    R = np.unique(np.clip(np.array(grid), low, high))
    z2 = z * z
    stat, t = sc.statistic(R)
    gap = stat - z2
    inside = gap <= 0
    flips = np.flatnonzero(inside[:-1] != inside[1:])
    ends: dict[int, float] = {}
    if len(flips):
        # Each crossing: Newton's method in R on statistic - z^2, its slope by a forward difference evaluated in the
        # same call, kept inside the bracket (a secant or the midpoint where Newton would leave it, or the statistic is
        # not finite at an end); the path's t from the nearer end starts each solve.
        a, fa, ta, b, fb, tb = R[flips], gap[flips], t[flips], R[flips + 1], gap[flips + 1], t[flips + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.where(np.isfinite(fa) & np.isfinite(fb), (a * fb - b * fa) / (fb - fa), 0.5 * (a + b))
        active = np.ones(len(flips), dtype=bool)
        last_step = np.full(len(flips), np.inf)
        end = np.full(len(flips), np.nan)
        for _ in range(100):
            c = np.where((c > a) & (c < b), c, 0.5 * (a + b))
            h = np.minimum(1e-6, 0.5 * (b - a))  # the statistic is good to about 1e-12, so its slope to about 1e-5
            guess = np.where(np.abs(c - a) < np.abs(c - b), ta, tb)
            both, tboth = sc.statistic(np.concatenate([c, c + h]), np.concatenate([guess, guess]))
            fc, fh, tc = both[: len(c)] - z2, both[len(c) :] - z2, tboth[: len(c)]
            left = (fc <= 0) == (fa <= 0)  # the crossing lies right of c
            a, fa, ta = np.where(active & left, c, a), np.where(active & left, fc, fa), np.where(active & left, tc, ta)
            b, fb, tb = np.where(active & ~left, c, b), np.where(active & ~left, fc, fb), np.where(active & ~left, tc, tb)
            with np.errstate(divide="ignore", invalid="ignore"):
                newton = c - fc * h / (fh - fc)
                secant = (a * fb - b * fa) / (fb - fa)
            nxt = np.where(np.isfinite(newton) & (newton > a) & (newton < b), newton,
                           np.where(np.isfinite(secant) & (secant > a) & (secant < b), secant, 0.5 * (a + b)))  # fmt: skip
            slow = np.abs(nxt - c) > 0.5 * last_step  # steps not halving (a jump, not a root, near): bisect, as Brent's
            nxt = np.where(slow, 0.5 * (a + b), nxt)
            last_step = np.abs(nxt - c)
            settled = active & (np.abs(nxt - c) <= 1e-9) & ~slow  # Newton has converged: the crossing is at c
            end = np.where(settled, c, end)
            active &= (b - a > 1e-8) & ~settled
            c = np.where(active, nxt, c)
            if not np.any(active):
                break
        inside_end = np.where(fb <= 0, b, a)  # where the bracket closed instead, its end inside the set
        last_in = np.where(np.isfinite(end), end, inside_end)
        for i, flip_at in enumerate(flips):
            ends[int(flip_at)] = float(last_in[i])
    pieces: list[tuple[float, float]] = []
    start: float | None = float(R[0]) if inside[0] else None
    for i in range(len(R) - 1):
        if inside[i] and not inside[i + 1]:
            pieces.append((start if start is not None else ends[i], ends[i]))
            start = None
        elif not inside[i] and inside[i + 1]:
            start = ends[i]
    if start is not None:
        pieces.append((start, float(R[-1])))
    if not pieces:
        reason = "no ratio in [0, 1] fits the estimates at this level"
    elif pieces == [(0.0, 1.0)]:
        reason = "every ratio in [0, 1] fits the estimates at this level: the gold does not bound the ratio"
    elif pieces == [(low, high)]:
        reason = "every ratio the strata can reach fits the estimates at this level: only the strata read whole bound it"
    else:
        reason = "the ratios in [0, 1] whose score statistic is at most z^2"
    return _ratio_set(pieces, reason)


# ------------------------------------------------------------------------------------------------------------ estimators


@dataclass(frozen=True)
class StratumMean:
    """The share of one phase-1 stratum's items whose indicator holds (a domain of the frame), under the same design."""

    stratum: str
    frame_size: int
    estimate: float
    variance: float
    interval: tuple[float, float]


INTERVAL_METHOD = (
    "score: theta_hat -+ z SD, the SD evaluated at the cells' prevalences restricted to the bound tested (Wilson's "
    "interval for one cell); a cell's f is credited with reducing the variance only as far as its own gold shows"
)


@dataclass(frozen=True)
class FrameMean:
    """The frame share of items whose indicator holds, with its design-based variance and interval.

    ``variance`` estimates the design variance, phase 1 on the target plus phase 2 on the residuals; for a fixed lam it
    is exactly unbiased. ``interval`` is the score interval ``interval_method`` names, whose coverage is measured by
    simulation (tests/discovery/test_wide_coverage.py); it is not estimate -+ z se, since se is evaluated at the estimate.
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
    resid: FloatArray  # each gold item's residual (0 where not gold; never read there)
    lam: FloatArray  # the lam each cell's estimate used


def _z(level: float) -> float:
    if not 0 < level < 1:
        raise ValueError("the confidence level is in (0, 1)")
    return float(stats.norm.ppf(0.5 + level / 2))


def _mean_interval(sc: _Score, z: float, se: float) -> tuple[float, float]:
    theta = sc.theta[0]
    if not sc.sampled():
        return (theta, theta)
    s = _score_set(sc, z, theta, se)
    _EXPERIMENT.append(s)
    if not s.pieces:
        return (theta, theta)
    return (s.pieces[0][0], s.pieces[-1][1])


_EXPERIMENT: list = []


def _assemble(c: _Cells, est: _CellEstimates, *, unit: str, lam: float, lam_rule: str, level: float) -> FrameMean:
    """Combine per-cell estimates into the frame estimate, its variance, and its interval."""
    z = _z(level)
    w = c.cell_weight()
    h = c.cell_stratum
    H = len(c.names)
    W = c.frame_sizes / c.frame_size
    occupied = c.m > 0
    mu = np.where(occupied, est.mu, 0.0)

    # Phase 1 on the target, phase 2 on the residuals (_design_covariance).
    v1_h, v2_h = _design_covariance(c, c.y, est.resid, c.y, est.resid)
    est_h = np.bincount(h, weights=w * mu, minlength=H)

    v1, v2 = float(v1_h.sum()), float(v2_h.sum())
    theta = float(est_h.sum())
    sc = _mean_score(c, est, theta)
    by_stratum = tuple(
        StratumMean(
            stratum=name,
            frame_size=int(c.frame_sizes[i]),
            estimate=float(est_h[i] / W[i]),
            variance=float((v1_h[i] + v2_h[i]) / W[i] ** 2),
            interval=_mean_interval(
                sc.domain(i, (float(est_h[i] / W[i]), 1.0)), z, float(np.sqrt(v1_h[i] + v2_h[i])) / W[i]
            ),
        )
        for i, name in enumerate(c.names)
    )
    return FrameMean(
        unit=unit,
        frame_size=c.frame_size,
        estimate=theta,
        variance=v1 + v2,
        variance_phase1=v1,
        variance_phase2=v2,
        se=float(np.sqrt(v1 + v2)),
        interval=_mean_interval(sc, z, float(np.sqrt(v1 + v2))),
        level=level,
        interval_method=INTERVAL_METHOD,
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

    The design, as plain data: ``frame_size`` items of ``unit`` in phase-1 strata of ``phase1_frame_sizes`` items, each
    sampled to ``phase1_sample_sizes``; phase-2 strata are the bins of the phase-1 ``stratifier`` cut at
    ``phase2_cuts`` (bin b holds cuts[b-1] <= value < cuts[b]) within each phase-1 stratum, allocated
    ``phase2_allocation[(stratum, bin)]`` gold items each. The data, one entry per phase-1 item: its ``phase1_stratum``,
    ``stratifier`` value, ``f`` for the target indicator, whether it is ``gold``, and its gold label ``y`` (read only
    where gold).
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
    est = _CellEstimates(mu=_difference_means(c, lam), resid=c.y - lam * c.f, lam=np.full(c.n_cells, lam))
    return _assemble(c, est, unit=unit, lam=lam, lam_rule="fixed", level=level)


def _difference_means(c: _Cells, lam: float) -> FloatArray:
    """Each cell's difference estimate of its phase-1 mean of y: ybar_gold + lam (fbar_phase1 - fbar_gold)."""
    r1 = np.maximum(c.r, 1)
    ybar = _cell_sum(c, c.y, True) / r1
    f2 = _cell_sum(c, c.f, True) / r1
    f1 = _cell_sum(c, c.f, False) / np.maximum(c.m, 1)
    return np.asarray(ybar + lam * (f1 - f2), dtype=float)


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
    q = w * w * c.phase2_factor()
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
        lam_i = np.where(fitted, np.clip(num_i / np.where(fitted, den_i, 1.0), 0.0, 1.0), 1.0)

    # Each cell's estimate of its phase-1 mean of y: (1/m)[Y2 + mean(lam_i)(F1 - F2) + (m - r)(ybar2 - mean(lam_i f_i))].
    r1 = np.maximum(r, 1)
    lam_bar = np.bincount(kg, weights=lam_i, minlength=K) / r1
    lf_bar = np.bincount(kg, weights=lam_i * fg, minlength=K) / r1
    F1 = _cell_sum(c, c.f, False)
    m1 = np.maximum(c.m, 1)
    mu = (Sy + lam_bar * (F1 - Sf) + (c.m - r) * (Sy / r1 - lf_bar)) / m1
    resid = np.zeros(len(c.f))
    resid[gi] = yg - lam_i * fg
    est = _CellEstimates(mu=mu, resid=resid, lam=lam_bar)
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


# ---------------------------------------------------------------------------------------------------------------- ratios
#
# The share of the denominator's items that are in the numerator: R = theta_num / theta_den, the frame means of two
# indicators observed on the same gold items, the numerator's items a subset of the denominator's. A ratio of two means
# is not a mean, and the ends of two intervals divided give an interval with no stated coverage; so R is tested through
# the linear combination y_num - R y_den, whose frame mean is zero exactly at the true R (Fieller 1954, "Some problems
# in interval estimation", JRSS B 16). Its plain (lam = 1) difference estimate is theta_num_hat - R theta_den_hat, and
# its variance estimate, the design's own (_design_covariance), is a quadratic in R. The point estimate is the ratio of
# the two plain difference estimates, so it lies in every set below that it can lie in.


def _within_unit(s: RatioSet, reason: str) -> RatioSet:
    """The set's intersection with [0, 1], where the ratio lies (the numerator's items are among the denominator's)."""
    pieces = [(max(lo, 0.0), min(hi, 1.0)) for lo, hi in s.pieces if lo <= 1.0 and hi >= 0.0]
    return _ratio_set(pieces, reason)


def _fieller(tn: float, td: float, a: float, b: float, cc: float, z: float) -> RatioSet:
    """{R: (tn - R td)^2 <= z^2 (a - 2 R b + R^2 cc)}: A R^2 - 2 B R + C <= 0 with A = td^2 - z^2 cc, B = tn td - z^2 b,
    C = tn^2 - z^2 a. Since a - 2 R b + R^2 cc >= 0 for every R (it is a variance estimate), the estimate tn / td lies
    in the set whenever td != 0, so A > 0 gives a bounded interval; A < 0 (td^2 < z^2 cc: theta_den is not
    distinguishable from zero at this level) gives two rays, or the whole line when no R is excluded."""
    z2 = z * z
    A, B, C = td * td - z2 * cc, tn * td - z2 * b, tn * tn - z2 * a
    D = B * B - A * C
    near_zero = (
        f"theta_den_hat^2 = {td * td:.3g} {'<' if A < 0 else '='} z^2 Var(theta_den_hat) = {z2 * cc:.3g}: theta_den "
        f"is not distinguishable from zero at this level, so every ratio far enough from the estimate fits the data"
    )
    if A > 0:
        root = float(np.sqrt(max(D, 0.0)))
        qv = B + np.copysign(root, B) if B != 0 else root
        r1, r2 = (qv / A, C / qv) if qv != 0 else (0.0, 0.0)
        return _ratio_set(
            [(min(r1, r2), max(r1, r2))],
            f"theta_den_hat^2 = {td * td:.3g} > z^2 Var(theta_den_hat) = {z2 * cc:.3g}: theta_den is distinguishable "
            "from zero at this level, so the set is a bounded interval",
        )
    if A < 0:
        if D < 0:
            return _ratio_set([(-np.inf, np.inf)], f"{near_zero}, on both sides: no ratio is excluded")
        root = float(np.sqrt(D))
        qv = B + np.copysign(root, B) if B != 0 else root
        r1, r2 = (qv / A, C / qv) if qv != 0 else (0.0, 0.0)
        lo, hi = min(r1, r2), max(r1, r2)
        return _ratio_set(
            [(-np.inf, lo), (hi, np.inf)], f"{near_zero}: the ratios in ({lo:.6g}, {hi:.6g}) are excluded"
        )
    # A == 0: the test is linear in R.
    if B > 0:
        return _ratio_set([(C / (2 * B), np.inf)], f"{near_zero}, above the estimate")
    if B < 0:
        return _ratio_set([(-np.inf, C / (2 * B))], f"{near_zero}, below the estimate")
    if C <= 0:
        return _ratio_set([(-np.inf, np.inf)], f"{near_zero}, on both sides: no ratio is excluded")
    return _ratio_set([], "theta_den_hat is zero with no estimated variance while theta_num_hat is not: no ratio fits")


FIELLER_METHOD = (
    "Fieller: the R for which (theta_num_hat - R theta_den_hat)^2 <= z^2 V(R), V(R) the design's unbiased variance "
    "estimate of the plain difference estimate of y_num - R y_den (prediction f_num - R f_den), a quadratic in R"
)
RATIO_INTERVAL_METHOD = (
    "score: the R in [0, 1] for which (theta_num_hat - R theta_den_hat)^2 <= z^2 V_R, V_R the variance at category "
    "probabilities of the phase-2 strata (neither, denominator only, numerator) restricted so that the frame's "
    "y_num - R y_den is zero: each stratum's estimate tilted toward the categories where y_num - R y_den is larger or "
    "smaller (empirical-likelihood weights) by t w q kappa, one t for every stratum; the variance there is the mixture "
    "of the categories with the gold's mean predictions in each, a category a stratum's gold has not shown credited "
    "no separation by f"
)
MARGINAL = (
    "marginal: each numerator's interval holds its own ratio at the stated level; the intervals of several numerators "
    "from one call are not simultaneous"
)


@dataclass(frozen=True)
class FrameRatio:
    """The share of the denominator's items that are in one numerator, R = theta_num / theta_den, from the plain
    (lam = 1) difference estimates of the two frame means on the same sample.

    ``interval`` is the score set (RATIO_INTERVAL_METHOD), within [0, 1] by construction, the interval to read; its
    coverage is measured by simulation (tests/discovery/test_wide_ratio_coverage.py). ``fieller`` is Fieller's set over
    the real line from the design's unbiased variance (FIELLER_METHOD), as it is, unbounded where theta_den is not
    distinguishable from zero, and ``fieller_unit`` its intersection with [0, 1]: reported beside the interval because
    it under-covers where a heavy phase-2 stratum's gold holds no numerator item. ``variance_at(R)`` is the unbiased
    variance estimate of theta_num_hat - R theta_den_hat. ``coverage`` says each interval is marginal."""

    numerator: str
    unit: str
    frame_size: int
    estimate: float | None  # theta_num_hat / theta_den_hat; None where theta_den_hat is 0
    theta_num: float
    theta_den: float
    variance_num: float
    variance_den: float
    covariance: float
    interval: RatioSet
    interval_method: str
    fieller: RatioSet
    fieller_unit: RatioSet
    fieller_method: str
    level: float
    coverage: str
    phase1_items: int
    gold_items: int
    census: bool

    def variance_at(self, ratio: float) -> float:
        return self.variance_num - 2 * ratio * self.covariance + ratio * ratio * self.variance_den


def two_phase_ratio(
    *,
    frame_size: int,
    unit: str,
    phase1_frame_sizes: Mapping[str, int],
    phase1_sample_sizes: Mapping[str, int],
    phase2_cuts: Sequence[float],
    phase2_allocation: Mapping[tuple[str, int], int],
    phase1_stratum: Sequence[str] | ArrayLike,
    stratifier: ArrayLike,
    gold: ArrayLike,
    f_den: ArrayLike,
    y_den: ArrayLike,
    numerators: Mapping[str, tuple[ArrayLike, ArrayLike]],
    level: float = 0.95,
) -> dict[str, FrameRatio]:
    """The share of the denominator's items that are in each numerator, under the two-phase design: for each
    numerator name, R = theta_num / theta_den, the frame means of the numerator's and the denominator's indicators.

    The design and the per-item data are two_phase_difference's (the same phase-2 strata, _phase2_strata), with the
    denominator's prediction ``f_den`` and gold label ``y_den``, and ``numerators[name] = (f, y)`` for each numerator:
    its prediction and gold label, one entry per phase-1 item. Which answers make an item count in a numerator or the
    denominator is the caller's (the estimand's) definition; a numerator's gold items must be among the denominator's.
    Each numerator's ratio is estimated from the same call against the shared denominator, and each interval is
    marginal (MARGINAL)."""
    def resolve(f: ArrayLike, y: ArrayLike) -> _Cells:
        return _resolve(
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

    if not numerators:
        raise ValueError("a ratio has at least one numerator")
    z = _z(level)
    cd = resolve(f_den, y_den)
    td = float(np.sum(cd.cell_weight() * np.where(cd.m > 0, _difference_means(cd, 1.0), 0.0)))
    ed = cd.y - cd.f
    v1, v2 = _design_covariance(cd, cd.y, ed, cd.y, ed)
    var_den = float(v1.sum() + v2.sum())
    census = bool(np.all(cd.sample_sizes == cd.frame_sizes))
    out = {}
    for name, (f_num, y_num) in numerators.items():
        cn = resolve(f_num, y_num)
        if np.any(cn.y[cn.gold] > cd.y[cn.gold]):
            raise ValueError(
                f"numerator {name!r}: a gold item is in the numerator but not the denominator; the numerator's items "
                "are a subset of the denominator's by the estimand's definition"
            )
        tn = float(np.sum(cn.cell_weight() * np.where(cn.m > 0, _difference_means(cn, 1.0), 0.0)))
        en = cn.y - cn.f
        a1, a2 = _design_covariance(cn, cn.y, en, cn.y, en)
        b1, b2 = _design_covariance(cn, cn.y, en, cd.y, ed)
        var_num, cov = float(a1.sum() + a2.sum()), float(b1.sum() + b2.sum())
        estimate = tn / td if td != 0 else None
        fieller = _fieller(tn, td, var_num, cov, var_den, z)
        sc = _ratio_score(cn, cd, (tn, td))
        if not sc.sampled():  # every phase-2 stratum read whole, phase 1 a census
            exact = [(estimate, estimate)] if estimate is not None and 0.0 <= estimate <= 1.0 else []
            interval = _ratio_set(exact, "every item of the frame is gold: the ratio is the frame's, without error")
        else:
            se = None
            if estimate is not None:
                se = float(np.sqrt(max(var_num - 2 * estimate * cov + estimate**2 * var_den, 0.0))) / abs(td)
            interval = _score_set(sc, z, estimate, se)
        out[name] = FrameRatio(
            numerator=name,
            unit=unit,
            frame_size=frame_size,
            estimate=estimate,
            theta_num=tn,
            theta_den=td,
            variance_num=var_num,
            variance_den=var_den,
            covariance=cov,
            interval=interval,
            interval_method=RATIO_INTERVAL_METHOD,
            fieller=fieller,
            fieller_unit=_within_unit(fieller, f"Fieller's set within [0, 1], where the ratio lies; {fieller.reason}"),
            fieller_method=FIELLER_METHOD,
            level=level,
            coverage=MARGINAL,
            phase1_items=int(cd.m.sum()),
            gold_items=int(cd.r.sum()),
            census=census,
        )
    return out


# ------------------------------------------------------------------------------------------ two independent samples


@dataclass(frozen=True)
class TwoSampleMean:
    """The PPI++ mean of two independent samples (spec §7.4), with the lam it used and how that lam was chosen."""

    estimate: float
    variance: float
    se: float
    interval: tuple[float, float]
    level: float
    lam: float
    lam_rule: str
    labeled: int
    unlabeled: int


PPI_LAM_RULE = (
    "PPI++ plug-in (arXiv:2311.01453, ppi_py's mean): Cov(y, f) over the labeled sample / ((1 + n / N) Var(f) over both "
    "samples), clipped to [0, 1]; fitted on the labeled sample it corrects, so it rests on PPI++'s asymptotics"
)


def two_sample_ppi_mean(
    *,
    design: Literal["independent_samples"],
    y_labeled: ArrayLike,
    f_labeled: ArrayLike,
    f_unlabeled: ArrayLike,
    lam: float | None = None,
    level: float = 0.95,
) -> TwoSampleMean:
    """The prediction-powered mean for two independent samples from one distribution: n labeled items (y, f) and N
    unlabeled items (f), lam f_bar_unlabeled + mean(y - lam f) over the labeled, with variance
    lam^2 s2_f,unlabeled / N + s2_(y - lam f),labeled / n (spec §7.4; sample variances with ddof 1). A fixed lam (lam = 1
    is PPI) or, with lam None, the PPI++ plug-in.

    Only for a design the caller declares as two independent samples (``design="independent_samples"``): a gold sample
    drawn from the frame whose mean is wanted is another design, served by two_phase_difference (addendum A07 §8.2).
    """
    if design != "independent_samples":
        raise ValueError("the two-sample PPI++ mean is for two independent samples only (addendum A07 §8.2)")
    y = np.asarray(y_labeled, dtype=float)
    fl = np.asarray(f_labeled, dtype=float)
    fu = np.asarray(f_unlabeled, dtype=float)
    if y.ndim != 1 or fl.shape != y.shape or fu.ndim != 1:
        raise ValueError("one label and one prediction per labeled item, and one prediction per unlabeled item")
    n, N = len(y), len(fu)
    if n < 2 or N < 2:
        raise ValueError("each sample needs at least 2 items to estimate its variance")
    if np.any(~np.isfinite(y)):
        raise ValueError("every labeled item carries a label")
    for a in (fl, fu):
        if np.any(~np.isfinite(a)) or np.any((a < 0) | (a > 1)):
            raise ValueError("f is a probability in [0, 1] (spec §7.4)")
    if lam is None:
        both = np.concatenate([fl, fu])
        var_f = float(np.var(both, ddof=1))
        cov_yf = float(np.mean((y - y.mean()) * (fl - fl.mean())))
        lam_used = float(np.clip(cov_yf / ((1 + n / N) * var_f), 0.0, 1.0)) if var_f > 0 else 0.0
        rule = PPI_LAM_RULE
    else:
        if not np.isfinite(lam):
            raise ValueError("lam is a finite number")
        lam_used, rule = float(lam), "fixed"
    e = y - lam_used * fl
    estimate = lam_used * float(fu.mean()) + float(e.mean())
    variance = lam_used**2 * float(np.var(fu, ddof=1)) / N + float(np.var(e, ddof=1)) / n
    se = float(np.sqrt(variance))
    z = _z(level)
    return TwoSampleMean(
        estimate=estimate,
        variance=variance,
        se=se,
        interval=(estimate - z * se, estimate + z * se),
        level=level,
        lam=lam_used,
        lam_rule=rule,
        labeled=n,
        unlabeled=N,
    )


__all__ = [
    "FrameMean",
    "FrameRatio",
    "RatioSet",
    "StratumMean",
    "TunedFrameMean",
    "TwoSampleMean",
    "two_phase_difference",
    "two_phase_difference_tuned",
    "two_phase_ratio",
    "two_sample_ppi_mean",
]
