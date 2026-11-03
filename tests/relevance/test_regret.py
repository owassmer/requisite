"""Min-max conditional regret over feasible answers (requisite/relevance/regret.py; addendum A03 §3.3).

The oracle's recorded results (spec/verified_synthetic_results.json, checks 3 and 4) are written here as numbers, since
spec/ is not in the repository; they are exact.
"""

from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from requisite.relevance.regret import minmax_regret


def matrix(rows: list[list[float]]) -> dict[str, dict[str, float]]:
    """Rows are answers, columns actions, as the oracle's minmax_answer_regret takes them."""
    return {f"answer{i}": {str(j): c for j, c in enumerate(row)} for i, row in enumerate(rows)}


@pytest.mark.covers("O03")
def test_different_optimal_sets_may_share_one_action():
    # Oracle check 3: conditional decision costs [[0, 0, 3], [3, 0, 0]] give min-max regret 0 at action 1.
    r = minmax_regret(matrix([[0, 0, 3], [3, 0, 0]]), epsilon=0)
    assert r.regret == 0
    assert r.minimax == {"1"}
    assert r.acceptable == {"1"}


@pytest.mark.covers("O04")
def test_a_stable_coarse_question_can_conceal_a_decision_changing_distinction():
    # Oracle check 4: the same coarse answer covers worlds with costs [0, 10] and [100, 10]: min-max regret 10. No
    # action is within a tolerance below 10 under both; at a tolerance of 10, action 1 is.
    r = minmax_regret(matrix([[0, 10], [100, 10]]), epsilon=0)
    assert r.regret == 10
    assert r.minimax == {"1"}
    assert r.acceptable == frozenset()
    assert minmax_regret(matrix([[0, 10], [100, 10]]), epsilon=Fraction(99, 10)).acceptable == frozenset()
    assert minmax_regret(matrix([[0, 10], [100, 10]]), epsilon=10).acceptable == {"1"}


@pytest.mark.covers("A.T04")
def test_answer_specific_optimum_sets_that_share_an_action_are_not_a_material_flip():
    # Addendum §13 row 4. Under "covered" the digital and the strip are both optimal; under "not_covered" the strip and
    # buying nothing are. A solver breaking each answer's tie by menu order picks the digital for one answer and the
    # strip for the other, which v3's raw R_flip would call a decision flip. The strip is optimal under both answers:
    # rho = 0, and the strip is the common acceptable action.
    costs = {
        "covered": {"digital": 40, "strip": 40, "none": 90},
        "not_covered": {"digital": 75, "strip": 60, "none": 60},
    }
    solver_tie_break = {a: next(d for d, c in row.items() if c == min(row.values())) for a, row in costs.items()}
    assert solver_tie_break == {"covered": "digital", "not_covered": "strip"}  # a spurious flip
    r = minmax_regret(costs, epsilon=0)
    assert r.regret == 0
    assert r.acceptable == {"strip"}
    assert r.optimal_sets == {"covered": {"digital", "strip"}, "not_covered": {"strip", "none"}}
    # A solver's near-tie (an objective one part in a billion apart) is the same action within the economic tolerance.
    noisy = {
        "covered": {"digital": 40.0, "strip": 40.00000004, "none": 90.0},
        "not_covered": {"digital": 75.0, "strip": 60.0, "none": 60.00000002},
    }
    assert minmax_regret(noisy, epsilon=0).acceptable == frozenset()
    r = minmax_regret(noisy, epsilon=Decimal("0.01"))
    assert r.acceptable == {"strip"}
    assert r.regret <= Decimal("0.01")


def test_the_tolerance_boundary_is_exact():
    # Costs differing by exactly epsilon are acceptable: no float rounding moves them across.
    costs = {"a": {"x": Decimal("0.3"), "y": Decimal("0.1")}, "b": {"x": Decimal("0.1"), "y": Decimal("0.1")}}
    r = minmax_regret(costs, epsilon=Decimal("0.2"))
    assert r.regret == 0
    assert r.optimal_sets["a"] == {"x", "y"}
    assert minmax_regret(costs, epsilon=Decimal("0.19999999999999999999")).optimal_sets["a"] == {"y"}


def test_a_single_feasible_answer_cannot_flip_a_decision():
    r = minmax_regret({"only": {"x": 5, "y": 3}}, epsilon=0)
    assert r.regret == 0 and r.acceptable == {"y"}


@pytest.mark.parametrize(
    ("costs", "epsilon", "error"),
    [
        ({}, 0, ValueError),
        ({"a": {}}, 0, ValueError),
        ({"a": {"x": 1, "y": 2}, "b": {"x": 1}}, 0, ValueError),
        ({"a": {"x": float("nan")}}, 0, ValueError),
        ({"a": {"x": Decimal("Infinity")}}, 0, ValueError),
        ({"a": {"x": 1}}, -1, ValueError),
        ({"a": {"x": True}}, 0, TypeError),
        ({"a": {"x": "1"}}, 0, TypeError),
    ],
)
def test_malformed_inputs_are_refused(costs, epsilon, error):
    with pytest.raises(error):
        minmax_regret(costs, epsilon=epsilon)


cost_matrices = st.integers(1, 4).flatmap(
    lambda answers: st.integers(1, 4).flatmap(
        lambda actions: st.lists(
            st.lists(st.integers(-50, 50), min_size=actions, max_size=actions), min_size=answers, max_size=answers
        )
    )
)


@given(cost_matrices, st.integers(0, 20))
def test_agrees_with_the_oracles_formula_and_its_own_definitions(rows, eps):
    r = minmax_regret(matrix(rows), epsilon=eps)
    # The oracle's minmax_answer_regret, computed independently with numpy.
    costs = np.asarray(rows, dtype=float)
    worst = (costs - costs.min(axis=1, keepdims=True)).max(axis=0)
    assert r.regret == worst.min()
    assert r.minimax == {str(j) for j in np.flatnonzero(worst == worst.min())}
    # The common acceptable actions are exactly the intersection of each answer's eps-optimal set, and exist exactly
    # when rho <= eps.
    assert r.acceptable == frozenset.intersection(*r.optimal_sets.values())
    assert bool(r.acceptable) == (r.regret <= eps)
    assert r.regret >= 0
