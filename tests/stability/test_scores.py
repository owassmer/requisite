"""S_arg, S_prob and the pooled distribution (requisite/stability/scores.py; spec §2.1; addendum A01 §2.1-2.3).

The oracle's recorded results (spec/verified_synthetic_results.json, checks 1 and 2) are written here as numbers, since
spec/ is not in the repository. They are compared to 1e-12, far inside any rounding a platform's log2 could add.
"""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.spatial.distance import jensenshannon

from requisite.core.judgment import SUM_TOLERANCE, TIE_TOLERANCE  # the oracle's, as the definition holds them
from requisite.stability.scores import (
    EQUAL_PER_CONFIGURATION,
    EQUAL_PER_PROVIDER,
    TOP_SET_EQUALITY,
    NotADistribution,
    check_distribution,
    js_bits,
    pool,
    scores,
    top_set,
)

ORACLE = 1e-12


def dist(*ps: float) -> dict[str, float]:
    return {f"a{i}": p for i, p in enumerate(ps)}


@st.composite
def distributions(draw, k: int):
    """A distribution over k answers, with zeros allowed, summing to one within the declared tolerance."""
    weights = draw(st.lists(st.integers(0, 1000), min_size=k, max_size=k).filter(lambda w: sum(w) > 0))
    total = sum(weights)
    return {f"a{i}": w / total for i, w in enumerate(weights)}


@st.composite
def configurations(draw):
    k = draw(st.integers(2, 5))
    return draw(st.lists(distributions(k), min_size=2, max_size=8))


@pytest.mark.covers("O01")
def test_stable_does_not_mean_certain():
    # Oracle check 1: twelve configurations at (0.5001, 0.4999) are perfectly stable and still a coin flip.
    s = scores([dist(0.5001, 0.4999)] * 12)
    assert s.s_arg == 1.0
    assert s.s_prob == 1.0
    assert s.modal_top_sets == (frozenset({"a0"}),)


@pytest.mark.covers("O02")
def test_same_argmax_unstable_probability():
    # Oracle check 2: six (0.999, 0.001) and six (0.501, 0.499) share their argmax, and S_prob is 0.8333223518437791
    # (Jensen-Shannon divergence, base 2), below the 0.85 starting threshold of spec §2.1.
    s = scores([dist(0.999, 0.001)] * 6 + [dist(0.501, 0.499)] * 6)
    assert s.s_arg == 1.0
    assert s.s_prob == pytest.approx(0.8333223518437791, abs=ORACLE)
    assert s.s_prob < 0.85


def test_s_prob_uses_the_divergence_not_the_distance():
    # One minus the mean JS divergence in bits; the square-root distance would give 1 - 0.5578 here.
    p, q = {"yes": 1.0, "no": 0.0}, {"yes": 0.5, "no": 0.5}
    expected = 1 - (0.5 * math.log2(1 / 0.75) + 0.5 * (0.5 * math.log2(0.5 / 0.75) + 0.5 * math.log2(0.5 / 0.25)))
    assert scores([p, q]).s_prob == pytest.approx(expected, abs=ORACLE)
    assert scores([p, q]).s_prob != pytest.approx(1 - jensenshannon([1, 0], [0.5, 0.5], base=2), abs=1e-3)


def test_zero_probabilities_and_disjoint_supports():
    # 0 log 0 = 0; distributions with disjoint supports are as far apart as JS_2 allows.
    assert js_bits({"yes": 1.0, "no": 0.0}, {"yes": 0.0, "no": 1.0}) == 1.0
    assert scores([{"yes": 1.0, "no": 0.0}, {"yes": 0.0, "no": 1.0}]).s_prob == 0.0
    assert js_bits({"yes": 1.0, "no": 0.0}, {"yes": 1.0, "no": 0.0}) == 0.0


@given(st.integers(2, 6).flatmap(lambda k: st.tuples(distributions(k), distributions(k))))
def test_js_bits_agrees_with_scipy_squared_distance(pq):
    # An independent computation: scipy returns the Jensen-Shannon distance, the square root of the divergence.
    p, q = pq
    keys = sorted(p)
    reference = jensenshannon([p[a] for a in keys], [q[a] for a in keys], base=2) ** 2
    assert js_bits(p, q) == pytest.approx(reference, abs=1e-9)


@given(configurations())
def test_scores_lie_in_the_unit_interval(cs):
    s = scores(cs)
    assert 0.0 <= s.s_arg <= 1.0
    assert 0.0 <= s.s_prob <= 1.0


@given(st.integers(2, 5).flatmap(distributions), st.integers(2, 12))
def test_identical_configurations_score_one(p, k):
    # Tied or not: identical configurations have identical top sets and zero divergence.
    s = scores([p] * k)
    assert s.s_arg == 1.0
    assert s.s_prob == 1.0


@given(configurations(), st.randoms(use_true_random=False))
def test_no_answer_or_configuration_is_chosen_by_order(cs, rnd):
    # Reordering the configurations, or the answers within each, changes neither score.
    shuffled = list(cs)
    rnd.shuffle(shuffled)
    reordered = []
    for p in shuffled:
        items = list(p.items())
        rnd.shuffle(items)
        reordered.append(dict(items))
    a, b = scores(cs), scores(reordered)
    assert (a.s_arg, a.s_prob) == (b.s_arg, b.s_prob)
    assert set(a.modal_top_sets) == set(b.modal_top_sets)


def test_tie_policy_is_named_and_versioned_and_never_picks_by_position():
    assert (TOP_SET_EQUALITY.name, TOP_SET_EQUALITY.version) == ("top_set_equality", 1)
    tied = {"yes": 0.5, "no": 0.5}
    assert top_set(tied) == frozenset({"yes", "no"})
    # Within the oracle's 1e-12 the answers tie; beyond it they do not.
    assert top_set({"yes": 0.5 + TIE_TOLERANCE / 4, "no": 0.5 - TIE_TOLERANCE / 4}) == frozenset({"yes", "no"})
    assert top_set({"yes": 0.5 + 1e-9, "no": 0.5 - 1e-9}) == frozenset({"yes"})
    # Three tied configurations and one that prefers "yes": the tied set is the modal top answer, "yes" disagrees.
    s = scores([tied, tied, tied, {"yes": 0.6, "no": 0.4}])
    assert s.s_arg == 0.75
    assert s.modal_top_sets == (frozenset({"yes", "no"}),)
    # Two top sets equally common: both are modal, neither is chosen.
    s = scores([{"yes": 0.6, "no": 0.4}, {"yes": 0.4, "no": 0.6}])
    assert s.s_arg == 0.5
    assert s.modal_top_sets == (frozenset({"no"}), frozenset({"yes"}))


def test_sums_to_one_is_checked_to_the_oracles_tolerance():
    assert SUM_TOLERANCE == 1e-10
    check_distribution({"yes": 0.5 + SUM_TOLERANCE / 2, "no": 0.5})
    check_distribution({"yes": 0.5 - SUM_TOLERANCE / 2, "no": 0.5})
    for off in (2 * SUM_TOLERANCE, -2 * SUM_TOLERANCE, 0.1):
        with pytest.raises(NotADistribution, match="never normalized"):
            check_distribution({"yes": 0.5 + off, "no": 0.5})


@pytest.mark.parametrize(
    "bad",
    [{}, {"yes": -0.1, "no": 1.1}, {"yes": math.nan, "no": 1.0}, {"yes": math.inf, "no": 0.0}, {"yes": True}, {1: 1.0}],
)
def test_what_is_not_a_distribution_is_refused(bad):
    with pytest.raises(NotADistribution):
        scores([bad, bad])


def test_configurations_must_share_one_answer_space_and_be_at_least_two():
    with pytest.raises(NotADistribution, match="answer spaces differ"):
        scores([{"yes": 1.0, "no": 0.0}, {"yes": 1.0, "maybe": 0.0}])
    with pytest.raises(ValueError, match="at least two"):
        scores([{"yes": 1.0, "no": 0.0}])


def test_pooling_gives_each_provider_equal_weight_split_among_its_configurations():
    # A01 §2.3: equal weight to providers, each provider's weight split among its configurations.
    configs = [
        ("openai_decisions", {"yes": 0.9, "no": 0.1}),
        ("openai_decisions", {"yes": 0.8, "no": 0.2}),
        ("openai_decisions", {"yes": 0.7, "no": 0.3}),
        ("clef", {"yes": 0.2, "no": 0.8}),
    ]
    pooled = pool(configs)
    assert pooled.policy == EQUAL_PER_PROVIDER and pooled.policy.version == 1
    assert pooled.distribution["yes"] == pytest.approx((0.9 + 0.8 + 0.7) / 6 + 0.2 / 2, abs=ORACLE)
    assert [w for _, _, w in pooled.configurations] == pytest.approx([1 / 6, 1 / 6, 1 / 6, 1 / 2])
    # The disagreement stays inspectable: every configuration's distribution is kept, in order.
    assert [(prov, p) for prov, p, _ in pooled.configurations] == configs
    per_config = pool(configs, EQUAL_PER_CONFIGURATION)
    assert per_config.distribution["yes"] == pytest.approx((0.9 + 0.8 + 0.7 + 0.2) / 4, abs=ORACLE)


@given(st.integers(2, 4).flatmap(lambda k: st.lists(distributions(k), min_size=2, max_size=5)), st.integers(1, 4))
def test_adding_paraphrases_to_one_provider_does_not_change_its_influence(ps, copies):
    # Under the default policy, repeating a provider's configuration leaves the pool where it was.
    configs = [(f"provider{i}", p) for i, p in enumerate(ps)]
    more = configs + [configs[0]] * copies
    a, b = pool(configs).distribution, pool(more).distribution
    assert all(a[k] == pytest.approx(b[k], abs=ORACLE) for k in a)
    assert sum(b.values()) == pytest.approx(1.0, abs=SUM_TOLERANCE)


def test_pooling_refuses_what_is_not_a_distribution():
    with pytest.raises(NotADistribution):
        pool([("clef", {"yes": 0.7, "no": 0.7})])
    with pytest.raises(ValueError):
        pool([])

