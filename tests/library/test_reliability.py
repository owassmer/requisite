"""The exact one-sided binomial upper bound on an error rate (requisite/library/reliability.py; addendum A12 §11.4).

The oracle's recorded result (spec/verified_synthetic_results.json, check 13) is written here as a number, since spec/ is
not in the repository.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import binom

from requisite.library.reliability import error_rate_upper_bound


@pytest.mark.covers("O13")
def test_three_flawless_cases_bound_the_error_rate_only_at_0_6316():
    # Oracle check 13: zero errors in three cases, one-sided exact 95%: 1 - 0.05^(1/3) = 0.6315968501359612.
    b = error_rate_upper_bound(0, 3, alpha=0.05)
    assert b.upper == pytest.approx(0.6315968501359612, abs=1e-15)
    assert b.upper == pytest.approx(1 - 0.05 ** (1 / 3), abs=1e-15)


@given(st.integers(1, 200).flatmap(lambda n: st.tuples(st.integers(0, n - 1), st.just(n))), st.sampled_from([0.01, 0.05, 0.1, 0.32]))
def test_the_bound_is_where_seeing_this_few_errors_has_probability_alpha(xn, alpha):
    # The defining equation, checked by an independent computation: P(X <= x | n, p_U) = alpha.
    x, n = xn
    b = error_rate_upper_bound(x, n, alpha=alpha)
    assert 0 < b.upper < 1
    assert binom.cdf(x, n, b.upper) == pytest.approx(alpha, abs=1e-9)
    if x == 0:
        assert b.upper == pytest.approx(1 - alpha ** (1 / n), abs=1e-12)


@given(st.integers(1, 100), st.integers(0, 99))
def test_more_cases_narrow_the_bound_and_more_errors_widen_it(n, x):
    x = min(x, n - 1)
    b = error_rate_upper_bound(x, n, alpha=0.05).upper
    assert error_rate_upper_bound(x, n + 1, alpha=0.05).upper < b
    assert error_rate_upper_bound(x + 1, n, alpha=0.05).upper > b


def test_all_errors_or_no_cases_exclude_no_rate():
    assert error_rate_upper_bound(3, 3, alpha=0.05).upper == 1.0
    assert error_rate_upper_bound(0, 0, alpha=0.05).upper == 1.0


@pytest.mark.parametrize(
    ("errors", "cases", "alpha", "error"),
    [(4, 3, 0.05, ValueError), (-1, 3, 0.05, ValueError), (0, 3, 0.0, ValueError), (0, 3, 1.0, ValueError),
     (0.0, 3, 0.05, TypeError), (True, 3, 0.05, TypeError)],
)
def test_malformed_inputs_are_refused(errors, cases, alpha, error):
    with pytest.raises(error):
        error_rate_upper_bound(errors, cases, alpha=alpha)
