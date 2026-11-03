"""A small fixed state and question for the provider layer's tests. Its responses are recorded in cassettes/ (by
`uv run python tests/judgment/provider_fixture.py --record`, outside the sandbox), and the tests replay them."""

from datetime import UTC, datetime

from requisite import core as c
from requisite.core.evidence import artifact_id, claim_id
from requisite.judgment.providers import PROVIDERS, Ask
from requisite.judgment.render import RENDERS, ask, render_state
from requisite.library.templates import original

AS_OF = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
TEXT = "Crews must stop outdoor work on any day with at least 0.10 inches of rain."
ART = artifact_id(TEXT.encode())
ARTIFACT = c.Artifact(
    id=ART, locator="fixture:provider-layer", source="test fixture", media_type="text/plain", size=len(TEXT),
    published_at=datetime(2026, 3, 1, tzinfo=UTC), published_at_basis="fixture", observed_at=AS_OF,
    availability="retrieved",
)  # fmt: skip
CLAIM = c.Claim(id=claim_id(ART, "plain-v1", (0, len(TEXT)), TEXT, "a city ordinance (fixture)"), artifact=ART, extraction="plain-v1",
                span=(0, len(TEXT)), text=TEXT, authority="a city ordinance (fixture)")  # fmt: skip
STATE = c.JudgmentState(as_of=AS_OF, observed=[CLAIM.id])
QUESTION = c.AtomicQuestion(
    id="fixture.stop_at_0.15", version=1, task="interpret",
    target="Does the rule require crews to stop outdoor work on a day with 0.15 inches of rain?",
    answers=[
        c.AnswerValue(value="yes", meaning="the rule requires crews to stop that day", status="affirmative"),
        c.AnswerValue(value="no", meaning="the rule does not require it", status="negative"),
        c.AnswerValue(value="not_established", meaning="the state does not establish it", status="not_established"),
    ],
)  # fmt: skip


def asks() -> list[Ask]:
    out = []
    for spec in PROVIDERS.values():
        for render in RENDERS:
            rendered = render_state(STATE, render, claims={CLAIM.id: CLAIM}, artifacts={ART: ARTIFACT}, definitions={})
            config = c.JudgeConfig(provider=spec.name, model=spec.model, model_cutoff=spec.model_cutoff,
                                   paraphrase_id="p0", render_id=render.id)  # fmt: skip
            out.append(Ask(QUESTION, STATE.content_hash(), rendered, ask(QUESTION, original(QUESTION)), config))
    return out


if __name__ == "__main__":
    import sys

    from requisite.judgment.providers import judge_all

    if "--record" not in sys.argv:
        raise SystemExit("usage: provider_fixture.py --record (calls the live providers once and records them)")
    for j in judge_all(asks(), workers=6):
        print(j.config.provider, j.config.render_id, dict(j.distribution or {}), j.refusal, dict(j.reported or {}), j.precision)
