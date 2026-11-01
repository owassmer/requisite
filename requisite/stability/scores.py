"""Stability scores and the pooled distribution over judge configurations (spec §2.1; addendum A01 §2.1-2.3).

For one question on one pinned state, each judge configuration returns a distribution over the question's closed answer
set. Two diagnostics follow, each over the configurations with equal weight:

- S_arg, the share of configurations whose top answer equals the modal top answer;
- S_prob, one minus the mean pairwise Jensen-Shannon divergence in bits (the divergence, not its square root, the
  Jensen-Shannon distance some libraries return; addendum A01 §2.1).

They are perturbation diagnostics: they measure whether meaning-preserving presentations move the answer, not whether
the question is atomic, adequate or correct, and configurations are not independent witnesses (A01 §2.2-2.3). Twelve
configurations all at (0.5001, 0.4999) score 1 on both and the event is still a coin flip (oracle check 1).

The pooled distribution (A01 §2.3) is a descriptive ensemble estimate under a named, versioned pooling policy, kept
beside the per-configuration distributions so disagreement stays inspectable. It is never a posterior from independent
witnesses.

Distributions are mappings from answer value (spec §4 ``AnswerValue.value``) to probability. A distribution must be
finite, non-negative and sum to one within ``SUM_TOLERANCE``; anything else is refused, never normalized (the oracle's
``distribution``). Turning a provider's reported probabilities into a distribution is the provider layer's work.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum, log2

# The oracle's tolerances (spec/verify_math.py): a distribution sums to one within 1e-10 (`distribution`), and answers
# within 1e-12 of a configuration's maximum are tied (`stability`).
SUM_TOLERANCE = 1e-10
TIE_TOLERANCE = 1e-12

# S_prob lies in [0, 1] for exact distributions (JS_2 is in [0, 1]). Inputs accepted within SUM_TOLERANCE and the
# rounding of logarithms (about 1e-16) can move the computed value outside by at most about SUM_TOLERANCE; that
# excursion is projected back onto [0, 1]. Anything larger cannot come from valid input and is raised as an error.
_RANGE_SLACK = 2 * SUM_TOLERANCE


class NotADistribution(ValueError):
    """A configuration's output is not a distribution over the closed answer set: refused, never normalized."""


@dataclass(frozen=True)
class TiePolicy:
    name: str
    version: int


# A configuration's top answer is the set of its answers tied at the maximum (within TIE_TOLERANCE); S_arg is the share
# of configurations whose top set equals the most common top set. Identical configurations agree whether or not they
# tie, and no answer is ever chosen by its position in a list (A01 §2.1: "Do not silently break ties according to
# answer-list order"). Without ties it is exactly the oracle's S_arg.
TOP_SET_EQUALITY = TiePolicy("top_set_equality", 1)


@dataclass(frozen=True)
class PoolingPolicy:
    name: str
    version: int


# A01 §2.3: "a defensible default is equal weight to providers, with each provider's weight split among its approved
# configurations. Otherwise adding paraphrases to one provider changes its influence."
EQUAL_PER_PROVIDER = PoolingPolicy("equal_per_provider_split_among_configurations", 1)
# Equal weight per configuration: the weighting S_arg and S_prob use. Adding configurations to one provider raises its
# influence, which is why it is not the default.
EQUAL_PER_CONFIGURATION = PoolingPolicy("equal_per_configuration", 1)


@dataclass(frozen=True)
class Scores:
    """The two stability diagnostics for one question on one state, with what they were computed from."""

    s_arg: float
    s_prob: float
    top_sets: tuple[frozenset[str], ...]  # each configuration's top answers, in the order the configurations came
    modal_top_sets: tuple[frozenset[str], ...]  # the most common top set; several when equally common
    tie_policy: TiePolicy


@dataclass(frozen=True)
class Pooled:
    """A pooled distribution with the per-configuration distributions it was formed from."""

    distribution: Mapping[str, float]
    policy: PoolingPolicy
    configurations: tuple[tuple[str, Mapping[str, float], float], ...]  # (provider, distribution, weight), input order


def check_distribution(p: Mapping[str, float]) -> None:
    """Refuse anything that is not a finite, non-negative distribution summing to one within SUM_TOLERANCE."""
    if not p:
        raise NotADistribution("a distribution over an empty answer set")
    for answer, x in p.items():
        if not isinstance(answer, str):
            raise NotADistribution(f"answer {answer!r} is not an answer value (a string)")
        if isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x) or x < 0:
            raise NotADistribution(f"probability of {answer!r} is {x!r}: expected a finite, non-negative number")
    total = fsum(p.values())
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise NotADistribution(
            f"probabilities sum to {total!r}, not to one within {SUM_TOLERANCE}; a distribution is never normalized here"
        )


def _answer_space(ps: Sequence[Mapping[str, float]]) -> tuple[str, ...]:
    for p in ps:
        check_distribution(p)
    space = set(ps[0])
    for p in ps[1:]:
        if set(p) != space:
            raise NotADistribution(f"answer spaces differ: {sorted(space)} and {sorted(p)}")
    return tuple(sorted(space))


def _js(p: Mapping[str, float], q: Mapping[str, float], answers: tuple[str, ...]) -> float:
    # Terms with zero probability contribute nothing (0 log 0 = 0); where p(a) > 0 or q(a) > 0 the midpoint is positive.
    terms = []
    for a in answers:
        m = (p[a] + q[a]) / 2
        if p[a] > 0:
            terms.append(p[a] * log2(p[a] / m))
        if q[a] > 0:
            terms.append(q[a] * log2(q[a] / m))
    return fsum(terms) / 2


def js_bits(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """The Jensen-Shannon divergence in bits, JS_2(p, q) = KL_2(p||m)/2 + KL_2(q||m)/2 with m = (p + q)/2 (A01 §2.1)."""
    return _unit(_js(p, q, _answer_space([p, q])))


def _unit(x: float) -> float:
    if -_RANGE_SLACK <= x < 0:
        return 0.0
    if 1 < x <= 1 + _RANGE_SLACK:
        return 1.0
    if not 0 <= x <= 1:
        raise ArithmeticError(f"{x!r} lies outside [0, 1] by more than the input tolerance allows")
    return x


def top_set(p: Mapping[str, float]) -> frozenset[str]:
    """The answers tied at the maximum within TIE_TOLERANCE (the oracle's np.isclose(p, p.max(), atol=1e-12, rtol=0))."""
    check_distribution(p)
    top = max(p.values())
    return frozenset(a for a, x in p.items() if abs(x - top) <= TIE_TOLERANCE)


def scores(configurations: Sequence[Mapping[str, float]]) -> Scores:
    """S_arg and S_prob over two or more configurations of one question on one state, each weighted equally."""
    k = len(configurations)
    if k < 2:
        raise ValueError(f"stability needs at least two configurations; got {k}")
    answers = _answer_space(configurations)
    tops = tuple(top_set(p) for p in configurations)
    counts = Counter(tops)
    most = max(counts.values())
    modal = tuple(sorted((s for s, c in counts.items() if c == most), key=lambda s: (len(s), sorted(s))))
    divergences = [_js(configurations[i], configurations[j], answers) for i in range(k) for j in range(i + 1, k)]
    s_prob = _unit(1 - fsum(divergences) / len(divergences))
    return Scores(s_arg=most / k, s_prob=s_prob, top_sets=tops, modal_top_sets=modal, tie_policy=TOP_SET_EQUALITY)


def pool(
    configurations: Sequence[tuple[str, Mapping[str, float]]], policy: PoolingPolicy = EQUAL_PER_PROVIDER
) -> Pooled:
    """The pooled distribution over configurations given as (provider, distribution), under a pooling policy."""
    if not configurations:
        raise ValueError("pooling needs at least one configuration")
    answers = _answer_space([p for _, p in configurations])
    if policy == EQUAL_PER_PROVIDER:
        per_provider = Counter(provider for provider, _ in configurations)
        weights = [1 / (len(per_provider) * per_provider[provider]) for provider, _ in configurations]
    elif policy == EQUAL_PER_CONFIGURATION:
        weights = [1 / len(configurations)] * len(configurations)
    else:
        raise ValueError(f"unknown pooling policy {policy}")
    pooled = {a: fsum(w * p[a] for w, (_, p) in zip(weights, configurations, strict=True)) for a in answers}
    kept = tuple((provider, dict(p), w) for w, (provider, p) in zip(weights, configurations, strict=True))
    return Pooled(distribution=pooled, policy=policy, configurations=kept)
