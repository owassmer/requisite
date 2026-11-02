"""Kleene's strong three-valued logic (spec §6.3; I6: unknown stays unknown)."""

import itertools

import pytest

from requisite.core.kleene import Kleene, conj, disj, from_status, neg, of, plain

T, F, U = Kleene.TRUE, Kleene.FALSE, Kleene.UNKNOWN
RANK = {F: 0, U: 1, T: 2}  # strong Kleene: AND is the minimum, OR the maximum, in the order false < unknown < true


@pytest.mark.covers("I6")
def test_the_truth_tables_over_every_combination():
    # Spec §6.3's rows, exactly.
    assert (T & U, T | U) == (U, T)
    assert (F & U, F | U) == (F, U)
    assert (U & U, U | U) == (U, U)
    # And every combination, against the order-theoretic definition.
    for a, b in itertools.product((T, F, U), repeat=2):
        assert conj([a, b]) is min(a, b, key=RANK.get)
        assert disj([a, b]) is max(a, b, key=RANK.get)
        assert (a & b) is (b & a) and (a | b) is (b | a)
    assert [neg(v) for v in (T, F, U)] == [F, T, U]
    # A required condition that is unknown never yields a definitive yes; a definitively false one yields a definitive
    # no (spec §6.3).
    for others in itertools.product((T, F, U), repeat=3):
        assert conj([U, *others]) is not T
        assert conj([F, *others]) is F
    # An unknown can never be coerced by Python truthiness, anywhere: if, and, or, all, any, not.
    for v in (T, F, U):
        with pytest.raises(TypeError, match="I6"):
            bool(v)
        with pytest.raises(TypeError):
            all([v])
    # Answer statuses (spec §6.3): not_established and contested are unknown, never false.
    assert [from_status(s) for s in ("affirmative", "negative", "not_established", "contested")] == [T, F, U, U]


def test_empty_and_plain_forms():
    assert conj([]) is T and disj([]) is F
    assert [of(x) for x in (True, False, None)] == [T, F, U]
    assert [plain(v) for v in (T, F, U)] == [True, False, None]
    with pytest.raises(TypeError):
        of(0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        conj([T, True])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        from_status("maybe")  # type: ignore[arg-type]
