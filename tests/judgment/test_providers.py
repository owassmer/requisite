"""The provider layer (requisite/judgment/providers.py; spec §6.2, §13 WP3): every provider answers the same question
over the same rendered state with the same answer keys, replay is deterministic and never calls out, and a reported
vector is read exactly as declared: round-off corrected and logged, anything else refused."""

import math
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st
from provider_fixture import QUESTION, asks

from requisite import cassettes
from requisite.core.judgment import correct_roundoff
from requisite.judgment.providers import PROVIDERS, ProviderError, judge, read
from requisite.judgment.render import ask
from requisite.library.templates import original

ASKED = ask(QUESTION, original(QUESTION))
VALUES = {a.value for a in QUESTION.answers}


@pytest.mark.covers("WP3.answer_keys")
def test_every_provider_answers_with_the_questions_own_answer_keys():
    judgments = [judge(a) for a in asks()]
    assert {j.config.provider for j in judgments} == set(PROVIDERS)
    for j in judgments:
        assert j.distribution is not None and set(j.distribution) == VALUES, j.config
        assert j.question.id == QUESTION.id and j.config.model_cutoff == "unknown"


@pytest.mark.covers("WP3.replay")
def test_replay_is_deterministic_and_never_calls_a_provider(monkeypatch):
    # The conftest makes every test replay-only; a live call would raise. Twice over, the same judgments.
    def no_network(*args, **kwargs):
        raise AssertionError("a replay reached the network")

    monkeypatch.setattr("httpx.post", no_network)
    first = [judge(a) for a in asks()]
    again = [judge(a) for a in asks()]
    assert [j.content_hash() for j in first] == [j.content_hash() for j in again]
    # A request that was never recorded is refused, not answered.
    changed = asks()[0]
    reworded = type(changed.asked)(changed.asked.name, changed.asked.instructions + " ", changed.asked.answers)
    other = type(changed)(changed.question, changed.state_hash, changed.rendered, reworded, changed.config)
    with pytest.raises(cassettes.CassetteMissing):
        judge(other)


def response(provider: str, probabilities: dict, model: str | None = None) -> dict:
    spec = PROVIDERS[provider]
    if provider == "openai_decisions":
        return {"model": model or spec.model, "answers": [{"type": "choice", "name": ASKED.name, "probabilities": [
            {"value": v, "probability": p} for v, p in probabilities.items()]}]}  # fmt: skip
    return {"model": model or spec.model, "answers": {ASKED.name: {"type": "choice", "probabilities": probabilities}}}


def test_round_off_is_corrected_within_each_values_own_interval_and_kept():
    spec = PROVIDERS["openai_decisions"]
    # Three answers at two decimals: 0.33 each sums to 0.99, a rounding of thirds. Corrected, the reported kept.
    r = read(spec, response("openai_decisions", {"yes": 0.33, "no": 0.33, "not_established": 0.33}), ASKED)
    assert r.distribution == pytest.approx({"yes": 1 / 3, "no": 1 / 3, "not_established": 1 / 3})
    assert r.reported == {"yes": 0.33, "no": 0.33, "not_established": 0.33} and r.precision == 0.01
    # 0.90 in total is no rounding of any distribution: refused with the reason, never normalized.
    r = read(spec, response("openai_decisions", {"yes": 0.5, "no": 0.3, "not_established": 0.1}), ASKED)
    assert r.distribution is None and "not a rounding" in r.refusal
    # A value reported more finely than the declared step is read at the finer step.
    r = read(spec, response("openai_decisions", {"yes": 0.335, "no": 0.333, "not_established": 0.333}), ASKED)
    assert r.precision == 0.001 and r.distribution is not None


@given(st.lists(st.integers(0, 100), min_size=2, max_size=6), st.sampled_from([0.1, 0.01, 0.001]))
def test_every_correction_stays_within_half_a_step_of_each_reported_value(units, step):
    # Each value as a provider reports it at its step: the decimal u x step (0.7, never 70 * 0.01 = 0.7000000000000001,
    # which is finer than the step, and which correct_roundoff reads exactly as it prints).
    reported = [float(Decimal(u) * Decimal(repr(step))) for u in units]
    corrected = correct_roundoff(reported, step)
    feasible = sum(max(0.0, r - step / 2) for r in reported) <= 1 + 1e-12 <= sum(r + step / 2 for r in reported) + 2e-12
    if corrected is None:
        assert not feasible
        return
    assert math.fsum(corrected) == pytest.approx(1.0, abs=1e-12)
    assert all(abs(p - r) <= step / 2 + 1e-12 and p >= 0 for p, r in zip(corrected, reported, strict=True))


def test_what_is_not_an_answer_to_this_question_is_not_read_as_one():
    spec = PROVIDERS["clef"]
    r = read(spec, response("clef", {}), ASKED)
    assert r.distribution is None and "refusal" in r.refusal
    r = read(spec, response("clef", {"yes": 0.5, "no": 0.5}), ASKED)
    assert r.distribution is None and "closed set" in r.refusal
    with pytest.raises(ProviderError, match="answered as"):
        read(PROVIDERS["jev"], response("jev", {"yes": 1, "no": 0, "not_established": 0}, model="other/model"), ASKED)
    assert read(PROVIDERS["jev"], response("jev", {"yes": 1, "no": 0, "not_established": 0},
                                           model="typesafe/jev-1.13-20260917"), ASKED).distribution  # fmt: skip
