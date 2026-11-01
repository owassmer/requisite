"""CountTrue over three-valued days (requisite/contract/counts.py; addendum A05 §5.3; spec §6.3)."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from requisite.contract.counts import count_true


@pytest.mark.covers("O12")
def test_unknown_days_widen_the_count_instead_of_counting_as_false():
    # Oracle check 12: [True, False, None, None, True] gives [2, 4].
    b = count_true([True, False, None, None, True])
    assert b.interval == (2, 4)
    assert (b.true, b.false, b.unknown) == (2, 1, 2)


@given(st.lists(st.sampled_from([True, False, None]), max_size=200))
def test_the_interval_is_k_to_k_plus_u(days):
    b = count_true(days)
    k, u = sum(d is True for d in days), sum(d is None for d in days)
    assert b.interval == (k, k + u)
    assert b.true + b.false + b.unknown == len(days)
    # Every resolution of the unknown days lands inside the interval, and both ends are reached.
    assert b.low == count_true([d is True for d in days]).low
    assert b.high == count_true([d is not False for d in days]).high


def test_no_unknown_days_gives_an_exact_count():
    assert count_true([True, True, False]).interval == (2, 2)
    assert count_true([]).interval == (0, 0)


@pytest.mark.parametrize("bad", [0, 1, "", "unknown", 0.0])
def test_values_that_are_not_three_valued_are_refused(bad):
    with pytest.raises(TypeError, match="True, False or None"):
        count_true([True, bad])
