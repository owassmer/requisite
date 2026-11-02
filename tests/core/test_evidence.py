"""Evidence identity and the synthetic quarantine (spec I2 content addressing, I4)."""

import pytest
from core_examples import ART, CLM, T0, TEXT
from pydantic import ValidationError

from requisite import core as c
from requisite.core.evidence import artifact_id, claim_id


@pytest.mark.covers("I4")
def test_a_synthetic_claim_in_a_real_state_fails_type_checking():
    synthetic = "syn_clm_" + "2" * 64
    with pytest.raises(ValidationError):
        c.JudgmentState(as_of=T0, observed=[synthetic])
    # Nor the other way: an observed claim is not a synthetic case's evidence by accident.
    with pytest.raises(ValidationError):
        c.SyntheticJudgmentState(as_of=T0, observed=[CLM], scenario="s1")
    # The namespaces are fixed on the records themselves.
    with pytest.raises(ValidationError):
        c.Claim(id=CLM, namespace="synthetic", artifact=ART, extraction="plain-v1", span=(4, 10), text=TEXT[4:10],
                authority="x")  # fmt: skip
    state = c.JudgmentState(as_of=T0, observed=[CLM])
    assert state.observed == (CLM,)
    # Nor can a synthetic claim be slipped in after validation: a record is immutable all the way down.
    with pytest.raises(AttributeError):
        state.observed.append(synthetic)  # type: ignore[attr-defined]


def test_artifacts_and_claims_are_content_addressed():
    assert artifact_id(b"a") == artifact_id(b"a") != artifact_id(b"b")
    # A claim's text is exactly its span, and its id is the hash of what it quotes from where.
    with pytest.raises(ValidationError, match="exactly its span"):
        c.Claim(id=CLM, artifact=ART, extraction="plain-v1", span=(4, 11), text=TEXT[4:10], authority="x")
    with pytest.raises(ValidationError, match="hash"):
        c.Claim(id=claim_id(ART, "plain-v1", (4, 10), "other!"), artifact=ART, extraction="plain-v1", span=(4, 10),
                text=TEXT[4:10], authority="x")  # fmt: skip


def test_an_instant_carries_its_zone():
    from datetime import datetime

    with pytest.raises(ValidationError, match="time zone"):
        c.JudgmentState(as_of=datetime(2026, 11, 1, 12, 0), observed=[])
