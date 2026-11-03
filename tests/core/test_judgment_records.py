"""Judgments pin their model (I10) and carry a distribution or a refusal, never a repaired one."""

import pytest
from core_examples import CONFIG, JUDGMENT, REF, T0
from pydantic import ValidationError

from requisite import core as c


@pytest.mark.covers("I10")
def test_every_judgment_records_provider_model_and_cutoff_with_unknown_recorded_as_unknown():
    assert (JUDGMENT.config.provider, JUDGMENT.config.model, JUDGMENT.config.model_cutoff) == (
        "clef", "@cf/cloudflare/clef", "unknown",
    )  # fmt: skip
    fields = CONFIG.model_dump()
    for missing in ("provider", "model", "model_cutoff"):
        with pytest.raises(ValidationError):
            c.JudgeConfig(**{k: v for k, v in fields.items() if k != missing})
    # The cutoff is a date or the word unknown; it is never left blank or guessed.
    with pytest.raises(ValidationError):
        c.JudgeConfig(**{**fields, "model_cutoff": None})
    assert c.JudgeConfig(**{**fields, "model_cutoff": "2025-06-01"}).model_cutoff.year == 2025


def test_a_distribution_is_checked_never_normalized_and_a_refusal_has_none():
    base = JUDGMENT.model_dump()
    with pytest.raises(ValidationError, match="never normalized"):
        c.Judgment(**{**base, "distribution": {"yes": 0.7, "no": 0.7}})
    with pytest.raises(ValidationError, match="exactly one"):
        c.Judgment(**{**base, "refusal": "declined"})
    refused = c.Judgment(**{**base, "distribution": None, "refusal": "the provider declined"})
    assert refused.distribution is None
    assert REF.version == 1 and T0.tzinfo is not None


def test_a_non_forecast_question_offers_not_established():
    with pytest.raises(ValidationError, match="not_established"):
        c.AtomicQuestion(
            id="q", version=1, task="interpret", target="t",
            answers=[c.AnswerValue(value="y", meaning="m", status="affirmative"),
                     c.AnswerValue(value="n", meaning="m", status="negative")],
        )  # fmt: skip
