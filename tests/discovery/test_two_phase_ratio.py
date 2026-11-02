"""The share of a denominator's items in a numerator under the two-phase design (requisite/discovery/inference.py,
two_phase_ratio): its variance proved exactly by enumeration, and its sets checked at their limits.

The estimand is R = theta_num / theta_den, the frame means of two nested indicators on the same items. It is tested
through y_num - R y_den, whose plain difference estimate is theta_num_hat - R theta_den_hat; for every fixed R the
design's variance estimate of that combination (variance_at) is exactly unbiased over every phase-1 sample and every
phase-2 subsample, the take-whole rule included.
"""

import itertools
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy import stats
from scipy.optimize import brentq

from requisite.discovery.inference import (
    MARGINAL,
    _combination,
    _design_covariance,
    _difference_means,
    _fieller,
    _ratio_score,
    _resolve,
    _tilt,
    two_phase_ratio,
)

Z = float(stats.norm.ppf(0.975))


def _samples(strata, cuts, allocation):
    """Every (probability, arguments) pair the two-phase design draws. strata: name -> (y_num, y_den, f_num, f_den,
    stratifier, n_h), the numerator's items among the denominator's."""
    names = list(strata)
    frame_size = sum(len(strata[s][0]) for s in names)
    phase1 = [list(itertools.combinations(range(len(strata[s][0])), strata[s][5])) for s in names]
    p1 = Fraction(1, math.prod(len(p) for p in phase1))
    for picks in itertools.product(*phase1):
        cols: list[list[float]] = [[], [], [], [], []]
        stratum: list[str] = []
        for name, idx in zip(names, picks, strict=True):
            stratum += [name] * len(idx)
            for j in range(5):
                cols[j] += [strata[name][j][i] for i in idx]
        yn, yd, fn, fd, s = (np.array(c, dtype=float) for c in cols)
        cells: dict[tuple[str, int], list[int]] = {}
        for j, (name, v) in enumerate(zip(stratum, s, strict=True)):
            cells.setdefault((name, int(np.searchsorted(cuts, v, side="right"))), []).append(j)
        choices = [list(itertools.combinations(units, min(allocation[k], len(units)))) for k, units in cells.items()]
        p2 = Fraction(1, math.prod(len(c) for c in choices))
        for combo in itertools.product(*choices):
            gold = np.zeros(len(yn), dtype=bool)
            for c in combo:
                gold[list(c)] = True
            yield (
                p1 * p2,
                dict(
                    frame_size=frame_size,
                    unit="comment",
                    phase1_frame_sizes={n: len(strata[n][0]) for n in names},
                    phase1_sample_sizes={n: strata[n][5] for n in names},
                    phase2_cuts=cuts,
                    phase2_allocation=allocation,
                    phase1_stratum=stratum,
                    stratifier=s,
                    gold=gold,
                    f_den=fd,
                    y_den=np.where(gold, yd, np.nan),
                    numerators={"exposed": (fn, np.where(gold, yn, np.nan))},
                ),
            )


RATIO_FRAME = {
    # name: (y_num, y_den, f_num, f_den, stratifier, n_h). Cut at 0.5 with 2 gold per phase-2 stratum, the cells take
    # every size from empty to beyond the allocation, so the take-whole rule and empty cells are both exercised.
    "A": (
        [0, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 1, 1],
        [0.1, 0.6, 0.3, 0.2, 0.8, 0.4],
        [0.2, 0.9, 0.5, 0.3, 0.7, 0.6],
        [0.1, 0.6, 0.3, 0.2, 0.8, 0.4],
        4,
    ),
    "B": ([1, 0, 0, 0, 1], [1, 1, 0, 0, 1], [0.7, 0.2, 0.55, 0.1, 0.45], [0.8, 0.6, 0.3, 0.2, 0.9], [0.7, 0.2, 0.55, 0.1, 0.45], 3),
}
RATIO_ALLOCATION = {("A", 0): 2, ("A", 1): 2, ("B", 0): 2, ("B", 1): 2}


@dataclass(frozen=True)
class _Moments:
    """The estimates and variance terms two_phase_ratio reports, computed by the same definitions without its sets
    (the enumeration needs only these); test_the_reported_terms_are_these checks they agree."""

    theta_num: float
    theta_den: float
    variance_num: float
    variance_den: float
    covariance: float

    def variance_at(self, R: float) -> float:
        return self.variance_num - 2 * R * self.covariance + R * R * self.variance_den


def _terms(kw) -> _Moments:
    design = {k: kw[k] for k in kw if k not in ("unit", "f_den", "y_den", "numerators")}
    cd = _resolve(**design, f=kw["f_den"], y=kw["y_den"])
    cn = _resolve(**design, f=kw["numerators"]["exposed"][0], y=kw["numerators"]["exposed"][1])

    def mean(c):
        return float(np.sum(c.cell_weight() * np.where(c.m > 0, _difference_means(c, 1.0), 0.0)))

    def cov(u, v):
        return float(sum(x.sum() for x in _design_covariance(u, u.y, u.y - u.f, v.y, v.y - v.f)))

    return _Moments(mean(cn), mean(cd), cov(cn, cn), cov(cd, cd), cov(cn, cd))


def _moments(samples):
    rows = [(float(p), _terms(kw)) for p, kw in samples]
    assert math.isclose(math.fsum(p for p, _ in rows), 1.0, abs_tol=1e-12)
    return rows


def test_the_reported_terms_are_these():
    for _, kw in list(_samples(RATIO_FRAME, [0.5], RATIO_ALLOCATION))[::97]:
        r, m = two_phase_ratio(**kw)["exposed"], _terms(kw)
        assert (r.theta_num, r.theta_den, r.variance_num, r.variance_den, r.covariance) == (
            m.theta_num, m.theta_den, m.variance_num, m.variance_den, m.covariance,
        )  # fmt: skip
        assert r.estimate == (m.theta_num / m.theta_den if m.theta_den != 0 else None)
        for R in (-0.5, 0.3, 1.7):
            assert r.variance_at(R) == pytest.approx(m.variance_at(R), abs=1e-15)


def test_the_combinations_variance_estimate_is_exactly_unbiased_for_every_fixed_ratio():
    samples = list(_samples(RATIO_FRAME, [0.5], RATIO_ALLOCATION))
    rows = _moments(samples)
    yn = RATIO_FRAME["A"][0] + RATIO_FRAME["B"][0]
    yd = RATIO_FRAME["A"][1] + RATIO_FRAME["B"][1]
    # Each mean's plain difference estimate is exactly unbiased for its frame share.
    assert math.fsum(p * r.theta_num for p, r in rows) == pytest.approx(np.mean(yn), abs=1e-14)
    assert math.fsum(p * r.theta_den for p, r in rows) == pytest.approx(np.mean(yd), abs=1e-14)
    mn = math.fsum(p * r.theta_num for p, r in rows)
    md = math.fsum(p * r.theta_den for p, r in rows)
    # The quadratic's three coefficients: each an unbiased estimate of its design moment.
    assert math.fsum(p * r.variance_num for p, r in rows) == pytest.approx(
        math.fsum(p * (r.theta_num - mn) ** 2 for p, r in rows), abs=1e-14
    )
    assert math.fsum(p * r.variance_den for p, r in rows) == pytest.approx(
        math.fsum(p * (r.theta_den - md) ** 2 for p, r in rows), abs=1e-14
    )
    assert math.fsum(p * r.covariance for p, r in rows) == pytest.approx(
        math.fsum(p * (r.theta_num - mn) * (r.theta_den - md) for p, r in rows), abs=1e-14
    )
    # And so, for every fixed R, the variance estimate of theta_num_hat - R theta_den_hat.
    for R in (-1.0, 0.0, 0.25, np.mean(yn) / np.mean(yd), 0.8, 1.0, 2.5):
        design = math.fsum(p * ((r.theta_num - R * r.theta_den) - (mn - R * md)) ** 2 for p, r in rows)
        assert math.fsum(p * r.variance_at(R) for p, r in rows) == pytest.approx(design, abs=1e-14)
    sizes = set()
    for _, kw in samples:
        cells = zip(kw["phase1_stratum"], np.searchsorted([0.5], kw["stratifier"], side="right"), strict=True)
        sizes |= set(np.unique([f"{a}{b}" for a, b in cells], return_counts=True)[1].tolist())
    assert {1, 2, 3} <= sizes  # cells taken whole (1 or 2 items), subsampled (3), and empty on some samples


@settings(max_examples=10, deadline=None)
@given(
    st.lists(st.tuples(st.integers(0, 2), st.floats(0, 1), st.floats(0, 1), st.floats(0, 1)), min_size=4, max_size=5),
    st.lists(st.tuples(st.integers(0, 2), st.floats(0, 1), st.floats(0, 1), st.floats(0, 1)), min_size=3, max_size=4),
    st.integers(2, 3),
    st.floats(-1, 2),
)
def test_exactness_holds_for_any_nested_frame_and_any_predictions(a, b, alloc, R):
    # Category 0 neither, 1 denominator only, 2 numerator (and so denominator); f for each indicator and the stratifier
    # any numbers in [0, 1] fixed before the gold.
    def stratum(units, n_h):
        return (
            [int(u[0] == 2) for u in units],
            [int(u[0] >= 1) for u in units],
            [u[1] for u in units],
            [u[2] for u in units],
            [u[3] for u in units],
            n_h,
        )

    frame = {"A": stratum(a, len(a) - 1), "B": stratum(b, 2)}
    rows = _moments(list(_samples(frame, [0.5], {(s, k): alloc for s in frame for k in (0, 1)})))
    mn = math.fsum(p * r.theta_num for p, r in rows)
    md = math.fsum(p * r.theta_den for p, r in rows)
    n = len(a) + len(b)
    assert mn == pytest.approx((sum(frame["A"][0]) + sum(frame["B"][0])) / n, abs=1e-13)
    assert md == pytest.approx((sum(frame["A"][1]) + sum(frame["B"][1])) / n, abs=1e-13)
    design = math.fsum(p * ((r.theta_num - R * r.theta_den) - (mn - R * md)) ** 2 for p, r in rows)
    assert math.fsum(p * r.variance_at(R) for p, r in rows) == pytest.approx(design, abs=1e-13)


@settings(max_examples=40, deadline=None)
@given(st.data())
def test_on_a_phase1_census_the_variance_is_a07s_stratified_form_of_the_combination(data):
    n = data.draw(st.integers(8, 30))
    cat = np.array(data.draw(st.lists(st.integers(0, 2), min_size=n, max_size=n)))
    yn, yd = (cat == 2).astype(float), (cat >= 1).astype(float)
    fn = np.array(data.draw(st.lists(st.floats(0, 1), min_size=n, max_size=n)))
    fd = np.array(data.draw(st.lists(st.floats(0, 1), min_size=n, max_size=n)))
    R = data.draw(st.floats(-1, 2))
    cell = (fn >= 0.5).astype(int)
    allocation, gold = {}, np.zeros(n, dtype=bool)
    for k in (0, 1):
        units = np.flatnonzero(cell == k)
        take = data.draw(st.integers(2, max(2, len(units))))
        allocation[("frame", k)] = take
        gold[data.draw(st.permutations(list(units)))[: min(take, len(units))]] = True
    r = two_phase_ratio(
        frame_size=n, unit="comment", phase1_frame_sizes={"frame": n}, phase1_sample_sizes={"frame": n},
        phase2_cuts=[0.5], phase2_allocation=allocation, phase1_stratum=["frame"] * n, stratifier=fn, gold=gold,
        f_den=fd, y_den=np.where(gold, yd, np.nan), numerators={"exposed": (fn, np.where(gold, yn, np.nan))},
    )["exposed"]  # fmt: skip
    # A07 §8.4, transcribed for e = (y_num - f_num) - R (y_den - f_den): sum_k W_k^2 (1 - n_k / N_k) s2_e,k / n_k.
    e = (yn - fn) - R * (yd - fd)
    var = 0.0
    for k in (0, 1):
        u = cell == k
        if not u.any():
            continue
        g = u & gold
        var += (u.sum() / n) ** 2 * (1 - g.sum() / u.sum()) * (e[g].var(ddof=1) if g.sum() > 1 else 0.0) / g.sum()
    assert r.variance_at(R) == pytest.approx(var, abs=1e-12)
    assert r.census


# ------------------------------------------------------------------------------------------------------- Fieller's set


def test_fiellers_set_is_bounded_where_the_denominator_is_distinguishable_from_zero():
    s = _fieller(0.12, 0.30, 0.0004, 0.0003, 0.0009, Z)
    assert s.kind == "interval"
    lo, hi = s.pieces[0]
    assert lo < 0.12 / 0.30 < hi
    for R in (lo, hi):  # the ends are where the test is an equality
        assert (0.12 - R * 0.30) ** 2 == pytest.approx(Z * Z * (0.0004 - 2 * R * 0.0003 + R * R * 0.0009), rel=1e-9)


def test_fiellers_set_is_reported_as_rays_or_the_line_where_the_denominator_is_not_distinguishable_from_zero():
    # theta_den_hat^2 < z^2 Var(theta_den_hat): two rays when some ratios are excluded, the whole line when none is;
    # never truncated to a range.
    rays = _fieller(0.05, 0.02, 0.0001, 0.00002, 0.0004, Z)
    assert rays.kind == "two_rays" and rays.pieces[0][0] == -np.inf and rays.pieces[1][1] == np.inf
    assert "not distinguishable from zero" in rays.reason
    assert 0.05 / 0.02 in rays
    line = _fieller(0.01, 0.01, 0.01, 0.0, 0.01, Z)
    assert line.kind == "line" and line.pieces == ((-np.inf, np.inf),)
    assert "not distinguishable from zero" in line.reason
    ray = _fieller(0.05, 0.02, 0.0001, 0.00002, 0.02**2 / Z**2, Z)  # theta_den_hat^2 = z^2 Var exactly
    assert ray.kind == "ray"


def test_an_unbounded_fieller_set_is_reported_whole_and_within_the_unit_interval_beside_it():
    # A denominator so thin in the gold that its estimate is not distinguishable from zero: the raw set is unbounded,
    # its intersection with [0, 1] is reported beside it, and the score interval stays within [0, 1].
    rng = np.random.default_rng(3)
    n = 400
    yd = np.zeros(n)
    yd[:6] = 1
    yn = np.zeros(n)
    yn[:2] = 1
    fd = np.clip(rng.normal(0.05, 0.03, n), 0, 1)
    fn = fd * 0.4
    gold = np.zeros(n, dtype=bool)
    gold[[0, 7, 9, 11, 13, 15, 17, 19, 21, 23]] = True
    r = two_phase_ratio(
        frame_size=4000, unit="comment", phase1_frame_sizes={"frame": 4000}, phase1_sample_sizes={"frame": n},
        phase2_cuts=[], phase2_allocation={("frame", 0): 10}, phase1_stratum=["frame"] * n, stratifier=fn, gold=gold,
        f_den=fd, y_den=np.where(gold, yd, np.nan), numerators={"exposed": (fn, np.where(gold, yn, np.nan))},
    )["exposed"]  # fmt: skip
    assert r.fieller.kind in ("two_rays", "line", "ray")
    assert any(np.isinf(lo) or np.isinf(hi) for lo, hi in r.fieller.pieces)
    assert all(0.0 <= lo <= hi <= 1.0 for lo, hi in r.fieller_unit.pieces)
    assert all(0.0 <= lo <= hi <= 1.0 for lo, hi in r.interval.pieces)


# ---------------------------------------------------------------------------------------------------- the score interval


def _draw(seed, n_h=300):
    """One two-phase sample of a frame with a thin denominator in a heavy stratum."""
    rng = np.random.default_rng(seed)
    sizes = {"small": 2000, "large": 20000}
    rate = {"small": 0.3, "large": 0.08}
    stratum, yd, yn, fd, fn = [], [], [], [], []
    for s in sizes:
        voice = rng.random(n_h) < rate[s]
        exposed = rng.random(n_h) < 0.4
        p_voice = np.where(voice, rng.beta(4.8, 1.2, n_h), rng.beta(0.6, 5.4, n_h))
        p_exp = np.where(exposed, rng.beta(3, 3, n_h), rng.beta(0.6, 5.4, n_h))
        stratum += [s] * n_h
        yd.append(voice)
        yn.append(voice & exposed)
        fd.append(p_voice)
        fn.append(p_voice * p_exp)
    yd_, yn_, fd_, fn_ = (np.concatenate(x).astype(float) for x in (yd, yn, fd, fn))
    allocation = {(s, b): a for s in sizes for b, a in ((0, 26), (1, 7))}
    gold = np.zeros(len(yd_), dtype=bool)
    st_ = np.array(stratum)
    for s in sizes:
        for b in (0, 1):
            units = np.flatnonzero((st_ == s) & ((fn_ >= 0.5) == bool(b)))
            gold[rng.permutation(units)[: min(allocation[(s, b)], len(units))]] = True
    return dict(
        frame_size=sum(sizes.values()), unit="comment", phase1_frame_sizes=sizes,
        phase1_sample_sizes={s: n_h for s in sizes}, phase2_cuts=[0.5], phase2_allocation=allocation,
        phase1_stratum=stratum, stratifier=fn_, gold=gold, f_den=fd_, y_den=np.where(gold, yd_, np.nan),
        numerators={"exposed": (fn_, np.where(gold, yn_, np.nan))},
    )  # fmt: skip


def _score(kw, name="exposed"):
    design = {k: kw[k] for k in kw if k not in ("unit", "f_den", "y_den", "numerators")}
    cd = _resolve(**design, f=kw["f_den"], y=kw["y_den"])
    cn = _resolve(**design, f=kw["numerators"][name][0], y=kw["numerators"][name][1])
    r = two_phase_ratio(**kw)[name]
    return r, _ratio_score(cn, cd, (r.theta_num, r.theta_den))


@pytest.mark.parametrize("seed", range(8))
def test_the_score_interval_is_the_ratios_in_the_unit_interval_whose_statistic_is_at_most_z_squared(seed):
    r, sc = _score(_draw(seed))
    assert r.interval_method.startswith("score") and r.coverage == MARGINAL
    assert all(0.0 <= lo <= hi <= 1.0 for lo, hi in r.interval.pieces)
    if r.estimate is not None and 0.0 <= r.estimate <= 1.0:
        assert r.estimate in r.interval  # the estimate's own statistic is zero
    low, high = sc.reach()
    for lo, hi in r.interval.pieces:
        for end, outward in ((lo, -1e-6), (hi, 1e-6)):
            if max(low, 0.0) + 1e-6 < end < min(high, 1.0) - 1e-6:  # an end the statistic sets, not [0, 1] or the reach
                stat, _ = sc.statistic(np.array([end, end + outward]))
                assert stat[0] == pytest.approx(Z * Z, rel=1e-5)
                assert stat[1] > Z * Z
    grid = np.linspace(0.0, 1.0, 201)
    stat, _ = sc.statistic(grid)
    held = np.array([R in r.interval for R in grid])
    clear = np.abs(stat - Z * Z) > 1e-4  # away from the ends, membership is the test
    assert np.array_equal(held[clear], (stat <= Z * Z)[clear])


def test_every_item_gold_gives_the_frames_ratio_with_no_sampling_error():
    rng = np.random.default_rng(5)
    n = 40
    cat = rng.integers(0, 3, n)
    yn, yd = (cat == 2).astype(float), (cat >= 1).astype(float)
    fn, fd = rng.random(n), rng.random(n)
    r = two_phase_ratio(
        frame_size=n, unit="comment", phase1_frame_sizes={"frame": n}, phase1_sample_sizes={"frame": n},
        phase2_cuts=[0.5], phase2_allocation={("frame", 0): n, ("frame", 1): n}, phase1_stratum=["frame"] * n,
        stratifier=fn, gold=np.ones(n, dtype=bool), f_den=fd, y_den=yd, numerators={"exposed": (fn, yn)},
    )["exposed"]  # fmt: skip
    truth = yn.sum() / yd.sum()
    assert r.estimate == pytest.approx(truth, abs=1e-12)
    assert r.variance_num == r.variance_den == r.covariance == 0.0
    for s in (r.interval, r.fieller, r.fieller_unit):
        assert s.kind == "interval"
        assert s.pieces[0][0] == pytest.approx(truth, abs=1e-12) and s.pieces[0][1] == pytest.approx(truth, abs=1e-12)


def test_each_numerators_ratio_from_one_call_is_its_own_call_and_marginal():
    kw = _draw(11)
    rng = np.random.default_rng(12)
    fn, yn = kw["numerators"]["exposed"]
    split = rng.random(len(fn)) < 0.5  # two families partitioning the numerator
    families = {
        "overall": (fn, yn),
        "family a": (fn * 0.5, np.where(np.isnan(yn), np.nan, yn * split)),
        "family b": (fn * 0.5, np.where(np.isnan(yn), np.nan, yn * ~split)),
    }
    together = two_phase_ratio(**{**kw, "numerators": families})
    assert list(together) == list(families)
    for name, pair in families.items():
        alone = two_phase_ratio(**{**kw, "numerators": {name: pair}})[name]
        assert together[name] == alone
        assert together[name].coverage == MARGINAL and "not simultaneous" in MARGINAL
    dens = {(r.theta_den, r.variance_den) for r in together.values()}
    assert len(dens) == 1  # one denominator, shared


def test_a_numerator_item_outside_the_denominator_is_refused():
    kw = _draw(2)
    fn, yn = kw["numerators"]["exposed"]
    gold_items = np.flatnonzero(kw["gold"] & (kw["y_den"] == 0))
    bad = yn.copy()
    bad[gold_items[0]] = 1.0
    with pytest.raises(ValueError, match="subset of the denominator"):
        two_phase_ratio(**{**kw, "numerators": {"exposed": (fn, bad)}})
    with pytest.raises(ValueError, match="at least one numerator"):
        two_phase_ratio(**{**kw, "numerators": {}})


@settings(max_examples=30, deadline=None)
@given(st.integers(30, 300), st.data())
def test_where_phase_1_is_a_sample_and_the_gold_reads_it_whole_the_ends_are_the_empirical_likelihood_tests(n, data):
    # Every phase-2 stratum taken whole, phase 1 a sample of n of N: only phase 1 varies, the frame's categories move
    # beyond its phase-1 items, and at each R the restricted frame is the empirical-likelihood one, Pi_c = P_c / (1 +
    # eta z_c) with sum_c Pi_c z_c = 0 (Owen 1988), computed here on its own; the statistic is (theta_num - R
    # theta_den)^2 / ((N - n) / (n (N - 1)) Var_Pi(z)). The path once collapsed the set to the estimate here.
    N = data.draw(st.integers(n + 1, 20 * n))
    counts = data.draw(st.lists(st.integers(1, 100), min_size=3, max_size=3))  # every category has phase-1 items
    cat = np.repeat([0, 1, 2], np.round(np.array(counts) / sum(counts) * n).astype(int))[:n]
    assume(len(cat) == n and len(set(cat.tolist())) == 3)
    yn, yd = (cat == 2).astype(float), (cat >= 1).astype(float)
    f = np.array(data.draw(st.lists(st.floats(0, 1), min_size=n, max_size=n)))
    gold = np.ones(n, dtype=bool)
    r = two_phase_ratio(
        frame_size=N, unit="comment", phase1_frame_sizes={"frame": N}, phase1_sample_sizes={"frame": n},
        phase2_cuts=[], phase2_allocation={("frame", 0): n}, phase1_stratum=["frame"] * n, stratifier=f, gold=gold,
        f_den=f, y_den=yd, numerators={"exposed": (f, yn)},
    )["exposed"]  # fmt: skip
    P = np.bincount(cat, minlength=3) / n

    def statistic(R: float) -> float:
        zc = np.array([0.0, -R, 1.0 - R])
        lo, hi = -1.0 / (1.0 - R), 1.0 / R  # where every 1 + eta z_c stays positive
        eta = brentq(lambda e: float(np.sum(P * zc / (1 + e * zc))), lo * (1 - 1e-12), hi * (1 - 1e-12), xtol=1e-15)
        Pi = P / (1 + eta * zc)
        var = float(np.sum(Pi * zc * zc) - np.sum(Pi * zc) ** 2)
        return (r.theta_num - R * r.theta_den) ** 2 / ((N - n) / (n * (N - 1)) * var)

    assert r.interval.kind == "interval"
    ((lo, hi),) = r.interval.pieces
    assert hi - lo > 0
    for end in (lo, hi):
        if 1e-6 < end < 1 - 1e-6:
            assert statistic(end) == pytest.approx(Z * Z, rel=1e-6)


@settings(max_examples=300, deadline=None)
@given(
    st.lists(st.sampled_from([0.0, 2.8e-17, 1e-9, 0.3, 0.5, 1.0]), min_size=3, max_size=3).filter(lambda p: sum(p) > 0),
    st.floats(1e-6, 1 - 1e-6),
    st.floats(-1e9, 1e9),
)
def test_the_tilt_is_owens_weights_to_rounding_however_small_a_category_or_large_the_tilt(mass, R, lam):
    # pi_c = pihat_c / (1 - lam z_c + s): pihat_c / pi_c + lam z_c is one number, 1 + s, over the categories with mass,
    # and no category's 1 - lam z_c + s is negative; the weights sum to one; and their mean of z is nondecreasing in lam.
    # A category with a mass of 2.8e-17 next to one of nearly 1 once gave NaN weights.
    pihat = np.array(mass) / sum(mass)
    z = _combination(np.array([R]))
    lams = np.array([lam, lam + 1e-6 * (1 + abs(lam))])  # the second the larger
    pi, _, _ = _tilt(pihat[None, :], lams[:, None], np.repeat(z, 2, axis=0))
    pi = pi[:, 0, :]
    assert np.all(np.isfinite(pi)) and np.all(pi >= 0)
    assert np.allclose(pi.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    for row, la in zip(pi, [lams[0] * z[0], lams[1] * z[0]], strict=True):
        has = pihat > 0
        one = pihat[has] / row[has] + la[has]  # 1 + s
        scale = 1.0 + np.max(np.abs(la))
        assert np.ptp(one) <= 1e-9 * scale
        assert np.all(la <= one.mean() + 1e-9 * scale)  # every category's 1 - lam z_c + s >= 0
    means = pi @ z[0]
    assert means[1] >= means[0] - 1e-12
