"""The wide-mode estimators proved exactly by enumeration (requisite/discovery/inference.py; addendum A07).

Each expectation here is a sum over every sample the design draws, weighted by its exact probability: every phase-1
sample of every phase-1 stratum, and for each, every phase-2 subsample of every phase-2 stratum (a stratum with no more
phase-1 items than its allocation is taken whole). The oracle's recorded result (spec/verified_synthetic_results.json,
check 10) is written here as numbers, since spec/ is not in the repository.
"""

import itertools
import math
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from requisite.discovery.inference import (
    two_phase_difference,
    two_phase_difference_tuned,
    two_sample_ppi_mean,
)


def _design_samples(strata, cuts, allocation):
    """Every (probability, arguments) pair the two-phase design draws. strata: name -> (y, f, stratifier, n_h)."""
    names = list(strata)
    frame_size = sum(len(strata[s][0]) for s in names)
    phase1 = [itertools.combinations(range(len(strata[s][0])), strata[s][3]) for s in names]
    p1 = Fraction(1, math.prod(math.comb(len(strata[s][0]), strata[s][3]) for s in names))
    for picks in itertools.product(*[list(p) for p in phase1]):
        stratum, y, f, s = [], [], [], []
        for name, idx in zip(names, picks, strict=True):
            ys, fs, ss, _ = strata[name]
            stratum += [name] * len(idx)
            y += [ys[i] for i in idx]
            f += [fs[i] for i in idx]
            s += [ss[i] for i in idx]
        cells: dict[tuple[str, int], list[int]] = {}
        for j, (name, v) in enumerate(zip(stratum, s, strict=True)):
            cells.setdefault((name, int(np.searchsorted(cuts, v, side="right"))), []).append(j)
        choices = [list(itertools.combinations(units, min(allocation[k], len(units)))) for k, units in cells.items()]
        p2 = Fraction(1, math.prod(len(c) for c in choices))
        for combo in itertools.product(*choices):
            gold = np.zeros(len(y), dtype=bool)
            for c in combo:
                gold[list(c)] = True
            yield (
                p1 * p2,
                dict(
                    frame_size=frame_size,
                    unit="comment",
                    phase1_frame_sizes={n: len(strata[n][0]) for n in names},
                    phase1_sample_sizes={n: strata[n][3] for n in names},
                    phase2_cuts=cuts,
                    phase2_allocation=allocation,
                    phase1_stratum=stratum,
                    stratifier=np.array(s, dtype=float),
                    f=np.array(f, dtype=float),
                    gold=gold,
                    y=np.where(gold, np.array(y, dtype=float), np.nan),
                ),
            )


def _moments(samples, estimator):
    """Exact design expectation of the estimate, its design variance, and the expectation of the variance estimate."""
    rows = [(float(p), estimator(kw)) for p, kw in samples]
    assert math.isclose(math.fsum(p for p, _ in rows), 1.0, abs_tol=1e-12)
    mean = math.fsum(p * r.estimate for p, r in rows)
    design_var = math.fsum(p * (r.estimate - mean) ** 2 for p, r in rows)
    mean_var = math.fsum(p * r.variance for p, r in rows)
    return mean, design_var, mean_var, rows


# ------------------------------------------------------------------------------------------------- the oracle, check 10

ORACLE_Y = [0, 0, 0, 1, 1, 1]
ORACLE_F = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9]


def _census_one_stratum(y, f, gold_idx):
    gold = np.zeros(len(y), dtype=bool)
    gold[list(gold_idx)] = True
    return dict(
        frame_size=len(y),
        unit="comment",
        phase1_frame_sizes={"frame": len(y)},
        phase1_sample_sizes={"frame": len(y)},
        phase2_cuts=[],
        phase2_allocation={("frame", 0): len(gold_idx)},
        phase1_stratum=["frame"] * len(y),
        stratifier=np.asarray(f, dtype=float),
        f=np.asarray(f, dtype=float),
        gold=gold,
        y=np.where(gold, np.asarray(y, dtype=float), np.nan),
    )


@pytest.mark.covers("O10")
def test_oracle_check_10_over_every_size_3_sample():
    # verify_math.py check 10: a six-item frame, a phase-1 census, every size-3 gold sample. Recorded: truth 0.5, naive
    # model mean 0.6, mean over all samples 0.5, design variance = mean estimated variance = 0.021333333333333336, all-gold
    # SE 0.
    samples = list(itertools.combinations(range(6), 3))
    assert len(samples) == 20
    results = [two_phase_difference(**_census_one_stratum(ORACLE_Y, ORACLE_F, s)) for s in samples]
    values = np.array([r.estimate for r in results])
    variances = np.array([r.variance for r in results])
    assert np.mean(ORACLE_Y) == 0.5 and np.mean(ORACLE_F) == pytest.approx(0.6, abs=1e-15)
    assert values.mean() == pytest.approx(0.5, abs=1e-15)
    assert np.var(values) == pytest.approx(0.021333333333333336, abs=1e-15)
    assert variances.mean() == pytest.approx(0.021333333333333336, abs=1e-15)
    # Each sample, beside the oracle's srs_difference, transcribed: f_bar_U + mean(y - f) over G, (1 - n/N) s2 / n.
    for s, r in zip(samples, results, strict=True):
        e = np.array([ORACLE_Y[i] - ORACLE_F[i] for i in s])
        assert r.estimate == pytest.approx(np.mean(ORACLE_F) + e.mean(), abs=1e-15)
        assert r.se == pytest.approx(math.sqrt((1 - 3 / 6) * e.var(ddof=1) / 3), abs=1e-15)
        assert r.variance_phase1 == 0.0 and r.census
    everything = two_phase_difference(**_census_one_stratum(ORACLE_Y, ORACLE_F, range(6)))
    assert everything.estimate == pytest.approx(0.5, abs=1e-15)
    assert everything.se == 0.0


# --------------------------------------------------------------------------------------- A07 §8.2: all gold is the truth


@pytest.mark.covers("A.T12")
@settings(max_examples=60, deadline=None)
@given(
    st.integers(2, 40).flatmap(
        lambda n: st.tuples(
            st.lists(st.integers(0, 1), min_size=n, max_size=n),
            st.lists(st.floats(0, 1), min_size=n, max_size=n),
            st.lists(st.floats(0, 1), min_size=n, max_size=n),
        )
    ),
    st.sampled_from([[], [0.5], [0.3, 0.7]]),
    st.floats(-2, 2),
)
def test_a_gold_set_that_is_the_whole_frame_gives_the_truth_and_zero_se(frame, cuts, lam):
    # Addendum §13 row 12: every item of the frame is gold. Then each phase-2 stratum is taken whole: the corrected mean
    # is the frame truth, whatever f and lam are, and the sampling SE, the variance and the interval's width are zero.
    y, f, s = frame
    n = len(y)
    kw = dict(
        frame_size=n,
        unit="comment",
        phase1_frame_sizes={"frame": n},
        phase1_sample_sizes={"frame": n},
        phase2_cuts=cuts,
        phase2_allocation={("frame", b): n for b in range(len(cuts) + 1)},
        phase1_stratum=["frame"] * n,
        stratifier=np.array(s),
        f=np.array(f),
        gold=np.ones(n, dtype=bool),
        y=np.array(y, dtype=float),
    )
    for r in (two_phase_difference(**kw, lam=lam), two_phase_difference_tuned(**kw).result):
        assert r.estimate == pytest.approx(np.mean(y), abs=1e-12)
        assert r.se == 0.0 and r.variance == 0.0
        assert r.interval[0] == pytest.approx(np.mean(y), abs=1e-12)
        assert r.interval[1] == pytest.approx(np.mean(y), abs=1e-12)


# ------------------------------------------------------------- A07 §8.4: unequal label sampling across phase-2 strata


@pytest.mark.covers("A.T13")
def test_unequal_label_sampling_uses_the_design_weights_and_variance():
    # Addendum §13 row 13. A frame read whole, split by the model's label into two phase-2 strata whose gold fractions
    # differ (2 of 6 against 2 of 3), with a judge biased differently in each (it misses positives below the cut and
    # over-calls above it). Over every gold sample: the stratum-weighted estimate is exactly unbiased and its variance
    # estimate is exactly unbiased for the design variance, each sample's variance is A07 §8.4's formula; the unweighted
    # correction (the frame mean of f plus the pooled mean residual of all gold) is biased.
    y = [0, 0, 1, 0, 1, 0, 1, 1, 0]
    f = [0.1, 0.2, 0.15, 0.3, 0.05, 0.4, 0.9, 0.7, 0.8]
    strata = {"frame": (y, f, f, 9)}
    allocation = {("frame", 0): 2, ("frame", 1): 2}
    samples = list(_design_samples(strata, [0.5], allocation))
    assert len(samples) == math.comb(6, 2) * math.comb(3, 2)
    mean, design_var, mean_var, rows = _moments(samples, lambda kw: two_phase_difference(**kw))
    truth = np.mean(y)
    assert mean == pytest.approx(truth, abs=1e-14)
    assert mean_var == pytest.approx(design_var, abs=1e-14)
    yv, fv = np.array(y, dtype=float), np.array(f)
    low = fv < 0.5
    unweighted = []
    for (p, kw), (_, r) in zip(samples, rows, strict=True):
        g = kw["gold"]
        v = 0.0
        for cell in (low, ~low):
            e = (yv - fv)[g & cell]
            N_k, n_k = cell.sum(), (g & cell).sum()
            v += (N_k / 9) ** 2 * (1 - n_k / N_k) * e.var(ddof=1) / n_k
        assert r.variance == pytest.approx(v, abs=1e-15)
        unweighted.append(float(p) * (fv.mean() + (yv - fv)[g].mean()))
    # Equal gold per stratum weights each stratum's mean residual equally, (e_low + e_high) / 2, where the frame weights
    # them 6/9 and 3/9: the unweighted correction is off by exactly that difference, here -2/45.
    e_low, e_high = (yv - fv)[low].mean(), (yv - fv)[~low].mean()
    bias = (e_low + e_high) / 2 - (6 * e_low + 3 * e_high) / 9
    assert math.fsum(unweighted) == pytest.approx(truth + bias, abs=1e-14)
    assert bias == pytest.approx(-2 / 45, abs=1e-14)


# ------------------------------------------------------------------------- the two-phase design, every sample of both

TWO_PHASE = {
    # name: (y, f, stratifier, n_h). The stratifier is a model probability of another indicator, so f and the phase-2
    # strata differ; cut at 0.5 with 2 gold per phase-2 stratum, the cells take every size from empty to beyond the
    # allocation, so the take-whole rule and empty cells are both exercised.
    "A": ([0, 1, 1, 0, 1, 0], [0.1, 0.3, 0.6, 0.7, 0.9, 0.2], [0.2, 0.7, 0.4, 0.8, 0.6, 0.1], 4),
    "B": ([1, 0, 0, 1, 0], [0.2, 0.4, 0.55, 0.8, 0.15], [0.9, 0.3, 0.45, 0.6, 0.05], 3),
}
TWO_PHASE_ALLOCATION = {("A", 0): 2, ("A", 1): 2, ("B", 0): 2, ("B", 1): 2}


@pytest.mark.parametrize("lam", [1.0, 0.0, 0.5, 1.7, -0.4])
def test_two_phase_estimator_and_variance_estimator_are_exactly_unbiased(lam):
    # Särndal, Swensson and Wretman (1992) ch. 9, as derived in inference.py: for any fixed lam, E[estimate] is the frame
    # share and E[variance estimate] is the exact design variance, over every phase-1 sample and phase-2 subsample.
    samples = list(_design_samples(TWO_PHASE, [0.5], TWO_PHASE_ALLOCATION))
    sizes = set()
    for _, kw in samples:
        cells = zip(kw["phase1_stratum"], np.searchsorted([0.5], kw["stratifier"], side="right"), strict=True)
        sizes |= set(np.unique([f"{a}{b}" for a, b in cells], return_counts=True)[1].tolist())
    assert {1, 2, 3} <= sizes  # cells taken whole (1 or 2 items), subsampled (3), and empty on some samples
    mean, design_var, mean_var, rows = _moments(samples, lambda kw: two_phase_difference(**kw, lam=lam))
    y_all = TWO_PHASE["A"][0] + TWO_PHASE["B"][0]
    assert mean == pytest.approx(np.mean(y_all), abs=1e-14)
    assert mean_var == pytest.approx(design_var, abs=1e-14)
    # The phase-1 strata as domains: each unbiased for its own share, with an unbiased variance estimate.
    for i, name in enumerate(TWO_PHASE):
        dm = math.fsum(float(p) * r.by_stratum[i].estimate for p, r in rows)
        dv = math.fsum(float(p) * (r.by_stratum[i].estimate - dm) ** 2 for p, r in rows)
        assert dm == pytest.approx(np.mean(TWO_PHASE[name][0]), abs=1e-14)
        assert math.fsum(float(p) * r.by_stratum[i].variance for p, r in rows) == pytest.approx(dv, abs=1e-14)


def test_the_cross_fitted_lam_keeps_the_estimate_exactly_unbiased():
    # A lam fitted without the gold item it corrects: exactly unbiased over every sample (inference.py's reasoning). Its
    # variance is the fixed-lam formula, asymptotic only, so its mean is not asserted equal to the design variance.
    samples = list(_design_samples(TWO_PHASE, [0.5], TWO_PHASE_ALLOCATION))
    mean, design_var, mean_var, rows = _moments(samples, lambda kw: two_phase_difference_tuned(**kw).result)
    assert mean == pytest.approx(np.mean(TWO_PHASE["A"][0] + TWO_PHASE["B"][0]), abs=1e-14)
    assert all(r.lam_rule.startswith("leave-one-out") for _, r in rows)


@settings(max_examples=12, deadline=None)
@given(
    st.lists(st.tuples(st.integers(0, 1), st.floats(0, 1), st.floats(0, 1)), min_size=4, max_size=5),
    st.lists(st.tuples(st.integers(0, 1), st.floats(0, 1), st.floats(0, 1)), min_size=3, max_size=4),
    st.integers(2, 3),
    st.floats(0, 1.5),
)
def test_exactness_holds_for_any_frame_and_any_f_calibrated_or_not(a, b, alloc, lam):
    # f is any prediction fixed before the gold; calibration changes only the variance, never the unbiasedness.
    strata = {
        "A": ([u[0] for u in a], [u[1] for u in a], [u[2] for u in a], len(a) - 1),
        "B": ([u[0] for u in b], [u[1] for u in b], [u[2] for u in b], 2),
    }
    allocation = {(s, k): alloc for s in strata for k in (0, 1)}
    samples = list(_design_samples(strata, [0.5], allocation))
    mean, design_var, mean_var, _ = _moments(samples, lambda kw: two_phase_difference(**kw, lam=lam))
    truth = (sum(strata["A"][0]) + sum(strata["B"][0])) / (len(a) + len(b))
    assert mean == pytest.approx(truth, abs=1e-13)
    assert mean_var == pytest.approx(design_var, abs=1e-13)
    tuned_mean, _, _, _ = _moments(samples, lambda kw: two_phase_difference_tuned(**kw).result)
    assert tuned_mean == pytest.approx(truth, abs=1e-13)


# --------------------------------------------------------- a phase-1 census is exactly A07's single-phase form (§8.2, §8.4)


@settings(max_examples=60, deadline=None)
@given(st.data())
def test_a_phase1_census_reduces_to_a07_stratified_form(data):
    n = data.draw(st.integers(6, 30))
    y = np.array(data.draw(st.lists(st.integers(0, 1), min_size=n, max_size=n)), dtype=float)
    f = np.array(data.draw(st.lists(st.floats(0, 1), min_size=n, max_size=n)))
    cuts = data.draw(st.sampled_from([[], [0.5], [0.25, 0.75]]))
    lam = data.draw(st.floats(0, 1.5))
    cell = np.searchsorted(cuts, f, side="right")
    allocation, gold = {}, np.zeros(n, dtype=bool)
    for k in range(len(cuts) + 1):
        units = np.flatnonzero(cell == k)
        a = data.draw(st.integers(2, max(2, len(units))))
        allocation[("frame", k)] = a
        gold[data.draw(st.permutations(list(units)))[: min(a, len(units))]] = True
    r = two_phase_difference(
        frame_size=n,
        unit="comment",
        phase1_frame_sizes={"frame": n},
        phase1_sample_sizes={"frame": n},
        phase2_cuts=cuts,
        phase2_allocation=allocation,
        phase1_stratum=["frame"] * n,
        stratifier=f,
        f=f,
        gold=gold,
        y=np.where(gold, y, np.nan),
        lam=lam,
    )
    # A07 §8.3-8.4, transcribed: sum_h W_h [lam f_bar_Uh + mean_Gh(y - lam f)], V = sum_h W_h^2 (1 - n_h/N_h) s2_e,h / n_h.
    est, var = 0.0, 0.0
    for k in range(len(cuts) + 1):
        u = cell == k
        if not u.any():
            continue
        e = (y - lam * f)[u & gold]
        N_k, n_k = u.sum(), (u & gold).sum()
        est += N_k / n * (lam * f[u].mean() + e.mean())
        var += (N_k / n) ** 2 * (1 - n_k / N_k) * (e.var(ddof=1) if n_k > 1 else 0.0) / n_k
    assert r.estimate == pytest.approx(est, abs=1e-12)
    assert r.variance == pytest.approx(var, abs=1e-12)
    assert r.se == pytest.approx(math.sqrt(var), abs=1e-12)
    assert r.variance_phase1 == 0.0 and r.census


# -------------------------------------------------------------------------------------------- the interval at its limits


@settings(max_examples=80, deadline=None)
@given(st.integers(20, 400), st.data())
def test_with_one_cell_and_no_f_term_the_interval_is_wilsons(m, data):
    # lam = 0 over one phase-2 stratum of a phase-1 census: the cell's residual is y itself, and the interval is Wilson's
    # score interval for a finite population, (p - ybar)^2 = z^2 (1 - r/m)/r * m/(m - 1) * p(1 - p), in closed form.
    r = data.draw(st.integers(2, m - 1))
    x = data.draw(st.integers(0, r))
    pos = data.draw(st.integers(x, m - (r - x)))
    y = np.zeros(m)
    y[:pos] = 1
    gold = np.zeros(m, dtype=bool)
    gold[:x] = True
    gold[pos : pos + r - x] = True
    f = np.full(m, 0.5)
    res = two_phase_difference(**{**_census_one_stratum(y, f, np.flatnonzero(gold)), "f": f, "stratifier": f}, lam=0.0)
    z = stats.norm.ppf(0.975)
    k = z * z * (1 - r / m) / r * m / (m - 1)
    ph = x / r
    centre, half = (ph + k / 2) / (1 + k), math.sqrt(k * ph * (1 - ph) + k * k / 4) / (1 + k)
    assert res.estimate == pytest.approx(ph, abs=1e-15)
    assert res.interval[0] == pytest.approx(max(0.0, centre - half), abs=1e-9)
    assert res.interval[1] == pytest.approx(min(1.0, centre + half), abs=1e-9)


def test_the_interval_is_in_the_unit_interval_and_holds_the_estimate():
    samples = list(_design_samples(TWO_PHASE, [0.5], TWO_PHASE_ALLOCATION))
    for _, kw in samples[::7]:
        for r in (two_phase_difference(**kw), two_phase_difference_tuned(**kw).result):
            lo, hi = r.interval
            assert 0.0 <= lo <= hi <= 1.0
            assert lo - 1e-12 <= min(max(r.estimate, 0.0), 1.0) <= hi + 1e-12
            for s in r.by_stratum:
                assert 0.0 <= s.interval[0] <= s.interval[1] <= 1.0


# ---------------------------------------------------------------------------------------------- two independent samples


def test_two_sample_ppi_mean_matches_ppi_py():
    # The reference (spec §7.4), for the design it is for: two independent samples. Same point estimate and the same
    # PPI++ lam; ppi_py's standard error uses ddof 0 where ours uses ddof 1, so the variances agree after that factor.
    ppi = pytest.importorskip("ppi_py")
    rng = np.random.default_rng(7)
    n, N = 120, 5000
    yl = (rng.random(n) < 0.3).astype(float)
    fl = np.clip(0.6 * yl + rng.normal(0.2, 0.15, n), 0, 1)
    fu = np.clip(rng.normal(0.38, 0.25, N), 0, 1)
    for lam in (None, 1.0, 0.4):
        ours = two_sample_ppi_mean(design="independent_samples", y_labeled=yl, f_labeled=fl, f_unlabeled=fu, lam=lam)
        ref = float(np.asarray(ppi.ppi_mean_pointestimate(yl, fl, fu, lam=lam)).ravel()[0])
        assert ours.estimate == pytest.approx(ref, abs=1e-12)
        lo, hi = (float(np.asarray(v).ravel()[0]) for v in ppi.ppi_mean_ci(yl, fl, fu, alpha=0.05, lam=lam))
        ref_var0 = ((hi - lo) / (2 * stats.norm.ppf(0.975))) ** 2
        e = yl - ours.lam * fl
        assert ref_var0 == pytest.approx(ours.lam**2 * fu.var() / N + e.var() / n, rel=1e-9)
        assert ours.variance == pytest.approx(ours.lam**2 * fu.var(ddof=1) / N + e.var(ddof=1) / n, rel=1e-12)
        if lam is None:
            cov = np.mean((yl - yl.mean()) * (fl - fl.mean()))
            assert ours.lam == pytest.approx(
                np.clip(cov / ((1 + n / N) * np.var(np.concatenate([fl, fu]), ddof=1)), 0, 1), abs=1e-15
            )


def test_two_sample_ppi_mean_is_refused_for_any_other_design():
    other: Any = "gold_subset_of_frame"
    with pytest.raises(ValueError, match="independent samples"):
        two_sample_ppi_mean(design=other, y_labeled=[0, 1], f_labeled=[0.1, 0.9], f_unlabeled=[0.2, 0.4])


# ----------------------------------------------------------------------------------- data that this design did not draw


def _valid():
    return dict(
        frame_size=10,
        unit="comment",
        phase1_frame_sizes={"A": 6, "B": 4},
        phase1_sample_sizes={"A": 4, "B": 4},
        phase2_cuts=[0.5],
        phase2_allocation={("A", 0): 2, ("A", 1): 2, ("B", 0): 2, ("B", 1): 2},
        phase1_stratum=["A"] * 4 + ["B"] * 4,
        stratifier=np.array([0.1, 0.2, 0.7, 0.8, 0.1, 0.3, 0.6, 0.9]),
        f=np.array([0.1, 0.2, 0.7, 0.8, 0.1, 0.3, 0.6, 0.9]),
        gold=np.array([True] * 8),
        y=np.array([0, 0, 1, 1, 0, 1, 1, 1], dtype=float),
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"frame_size": 11}, "not the frame's"),
        ({"phase1_sample_sizes": {"A": 4}}, "different strata"),
        ({"phase1_stratum": ["A"] * 4 + ["C"] * 4}, "undeclared stratum"),
        ({"phase1_stratum": ["A"] * 5 + ["B"] * 3}, "differ from the declared"),
        ({"phase2_allocation": {("A", 0): 2, ("A", 1): 2, ("B", 0): 2, ("B", 1): 1}}, "at least 2"),
        ({"phase2_allocation": {("A", 0): 2, ("A", 1): 2, ("B", 0): 2}}, "no allocation"),
        ({"gold": np.array([True] * 7 + [False])}, "draws"),
        ({"f": np.array([0.1, 0.2, 0.7, 1.2, 0.1, 0.3, 0.6, 0.9])}, r"\[0, 1\]"),
        ({"y": np.array([0, 0, 1, 1, 0, 1, 1, np.nan])}, "indicator label"),
        ({"phase1_sample_sizes": {"A": 1, "B": 4}, "phase1_stratum": ["A"] + ["B"] * 4}, "at least 2 items"),
        ({"phase2_cuts": [0.6, 0.5]}, "strictly increasing"),
    ],
)
def test_data_the_design_did_not_draw_are_refused(change, message):
    kw = {**_valid(), **change}
    n = len(kw["phase1_stratum"])
    for key in ("stratifier", "f", "gold", "y"):
        if len(kw[key]) != n:
            kw[key] = kw[key][:n]
    with pytest.raises(ValueError, match=message):
        two_phase_difference(**kw)
