"""Decision providers behind one interface (spec §6.2): OpenAI Decisions, Clef (Cloudflare Workers AI) and Jev
(TypeSafe, through OpenRouter). Each takes a state and one typed question and returns a probability per answer.

One question per request (decisions, "Providers and keys": a question's facts belong in the state, and Slope measured
that batching moved answers). Every request goes through the cassettes (requisite/cassettes.py): replayed where
recorded, and live only outside the sandbox, recorded once, its secrets redacted. Provider-shaped requests and
responses never persist beyond the cassette (spec §4): what the product keeps is the Judgment.

Providers report probabilities at a fixed precision, and a reported vector is a rounding of the provider's
distribution. A vector whose values could be a rounding of a distribution (each within half a step of it) is corrected to
the closest such distribution, and both the reported and corrected values are kept on the judgment (addendum A04 §4.2: "Numerical round-off
corrections need a declared tolerance and log"). A vector beyond that is not a rounding: it is refused, with the reason,
never normalized. The precisions are each provider's as measured on November 1 (two decimals for Decisions, four for
Clef, one for Jev); a response reported more finely than its provider's declared precision is read at the finer step.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, Literal

import httpx

from requisite import cassettes
from requisite.core import AtomicQuestion, JudgeConfig, Judgment, Ref
from requisite.core.judgment import Provider, correct_roundoff
from requisite.judgment.render import AskedQuestion, RenderedState
from requisite.secrets import secret

TIMEOUT = httpx.Timeout(120.0, connect=20.0)
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}


class ProviderError(RuntimeError):
    """The provider failed the request after its retries, or answered in a shape this layer does not read."""


@dataclass(frozen=True)
class ProviderSpec:
    """One provider: where it is, which model, its known training cutoff (I10: "unknown" where none is documented), the
    step at which it reports probabilities, and its price."""

    name: Provider
    model: str
    model_cutoff: date | Literal["unknown"]
    precision: float
    endpoint: str  # the cassette's name for the endpoint: no account id or key in it
    price_usd_per_million_input: Decimal | None  # None: the response states its own cost

    def url(self) -> str:
        if self.name == "openai_decisions":
            return "https://api.openai.com/v1/decisions"
        if self.name == "clef":
            return f"https://api.cloudflare.com/client/v4/accounts/{secret('CLOUDFLARE_ACCOUNT_ID')}/ai/run/@cf/cloudflare/clef"
        if self.name == "jev":
            return "https://openrouter.ai/api/v1/systemone"
        raise ProviderError(f"no endpoint for {self.name}")

    def key(self) -> str:
        return secret(
            {"openai_decisions": "OPENAI_DECISIONS_API_KEY", "clef": "CLOUDFLARE_API_TOKEN", "jev": "OPENROUTER_API_KEY"}[
                self.name
            ]
        )

    def accepts(self, model: object) -> bool:
        """The model the provider says answered is the one asked, or a dated snapshot of it; never assumed."""
        if not isinstance(model, str):
            return False
        if self.name == "clef":
            return model in ("clef", "@cf/cloudflare/clef")
        return model == self.model or (model.startswith(self.model + "-") and model[len(self.model) + 1 :].replace("-", "").isdigit())


PROVIDERS: dict[str, ProviderSpec] = {
    # OpenAI's pricing for /v1/decisions: $0.10 per million input tokens (Slope's config, from the vendor's page).
    "openai_decisions": ProviderSpec(
        "openai_decisions", "gpt-6-luna", "unknown", 0.01, "openai/v1/decisions", Decimal("0.10")
    ),
    # Workers AI: $0.24 per million input tokens (Handoff's judgment layer, October 2026).
    "clef": ProviderSpec("clef", "clef", "unknown", 0.0001, "workers-ai/run/@cf/cloudflare/clef", Decimal("0.24")),
    # OpenRouter reports each request's cost in usage.cost.
    "jev": ProviderSpec("jev", "typesafe/jev-1.13", "unknown", 0.1, "openrouter/api/v1/systemone", None),
}


def request_body(spec: ProviderSpec, rendered: RenderedState, asked: AskedQuestion) -> dict[str, Any]:
    """The provider's request for one question over one rendered state (each vendor's documented shape; spec §6.2)."""
    state = dict(rendered.state)
    if spec.name == "openai_decisions":
        return {
            "model": spec.model,
            "input": [{"role": "user", "content": [{"type": "input_json", "data": state}]}],
            "questions": [
                {
                    "type": "choice",
                    "name": asked.name,
                    "instructions": asked.instructions,
                    "choices": [{"value": v, "description": m} for v, m in asked.answers.items()],
                }
            ],
        }
    return {
        "model": spec.model,
        "state": state,
        "questions": {asked.name: {"type": "choice", "instructions": asked.instructions, "criteria": dict(asked.answers)}},
    }


def _send(spec: ProviderSpec, body: Mapping[str, Any]) -> tuple[Any, float]:
    """One live request, retried on the provider's transient failures; its response and its cost in dollars."""
    attempt = 0
    while True:
        try:
            r = httpx.post(spec.url(), json=body, headers={"Authorization": f"Bearer {spec.key()}"}, timeout=TIMEOUT)
        except httpx.TransportError as e:
            if attempt >= 5:
                raise ProviderError(f"{spec.name}: transport failure after retries: {e!r}") from e
        else:
            if r.status_code < 400:
                data = r.json()
                if spec.name == "clef":
                    if not data.get("success", False) or "result" not in data:
                        raise ProviderError(f"clef: {data.get('errors')}")
                    data = data["result"]
                return data, float(_cost(spec, data))
            if r.status_code not in RETRY_STATUS or attempt >= 5:
                raise ProviderError(f"{spec.name}: HTTP {r.status_code}: {r.text[:300]}")
        attempt += 1
        threading.Event().wait(min(30.0, 2**attempt))


def _cost(spec: ProviderSpec, data: Mapping[str, Any]) -> Decimal:
    usage = data.get("usage") or {}
    if spec.price_usd_per_million_input is None:
        if "cost" not in usage:
            raise ProviderError(f"{spec.name}: the response states no cost, and this provider is priced by its own report")
        return Decimal(str(usage["cost"]))
    tokens = usage.get("input_tokens")
    if not isinstance(tokens, int):
        raise ProviderError(f"{spec.name}: the response states no input tokens to price")
    return Decimal(tokens) * spec.price_usd_per_million_input / Decimal(10**6)


@dataclass(frozen=True)
class Reading:
    """A response read: the probabilities as reported, the distribution they are a rounding of (or the reason there is
    none), the step they were read at, and the model that answered."""

    reported: dict[str, float] | None
    distribution: dict[str, float] | None
    refusal: str | None
    precision: float
    model: str


def _step(values: Sequence[float], declared: float) -> float:
    """The step the values are reported at: the provider's declared precision, or finer where the values are."""
    step = declared
    for v in values:
        while step > 1e-12 and abs(round(v / step) * step - v) > 1e-12:
            step /= 10
    return step


def read(spec: ProviderSpec, data: Mapping[str, Any], asked: AskedQuestion) -> Reading:
    """The provider's answer to the asked question, as a distribution over its answer values."""
    model = data.get("model")
    if not spec.accepts(model):
        raise ProviderError(f"{spec.name} answered as {model!r}, not {spec.model!r}")
    if spec.name == "openai_decisions":
        answers = {a.get("name"): a for a in data.get("answers") or ()}
        answer = answers.get(asked.name)
        probs = None if answer is None else {str(p["value"]): p["probability"] for p in answer.get("probabilities") or ()}
    else:
        answer = (data.get("answers") or {}).get(asked.name)
        probs = None if answer is None else answer.get("probabilities")
    if answer is None:
        raise ProviderError(f"{spec.name}: no answer to {asked.name} in the response")
    if not probs:
        return Reading(None, None, f"{spec.name} answered with no probabilities: a refusal", spec.precision, str(model))
    if set(probs) != set(asked.answers):
        return Reading(None, None, f"{spec.name} answered over {sorted(probs)}, not the closed set {sorted(asked.answers)}",
                       spec.precision, str(model))  # fmt: skip
    values = [probs[v] for v in asked.answers]
    if any(isinstance(p, bool) or not isinstance(p, int | float) or not math.isfinite(p) or p < 0 for p in values):
        return Reading(None, None, f"{spec.name} reported probabilities that are not finite and non-negative: {probs}",
                       spec.precision, str(model))  # fmt: skip
    reported = {v: float(probs[v]) for v in asked.answers}
    step = _step(values, spec.precision)
    corrected = correct_roundoff([reported[v] for v in asked.answers], step)
    if corrected is None:
        return Reading(reported, None, f"{spec.name} reported probabilities summing to {math.fsum(values)!r}: no "
                                       f"distribution rounds to them at step {step}, so they are not a rounding", step,
                       str(model))  # fmt: skip
    return Reading(reported, dict(zip(asked.answers, corrected, strict=True)), None, step, str(model))


@dataclass(frozen=True)
class Ask:
    """One judgment to make: a question in one configuration over one rendered state."""

    question: AtomicQuestion
    state_hash: str
    rendered: RenderedState
    asked: AskedQuestion
    config: JudgeConfig


def judge(ask_: Ask, *, budget: cassettes.Budget | None = None, root: Any = None) -> Judgment:
    """One judgment: the request replayed from its cassette or made once and recorded, then read."""
    spec = PROVIDERS[ask_.config.provider]
    if ask_.config.model != spec.model or ask_.config.model_cutoff != spec.model_cutoff:
        raise ProviderError(f"configuration pins {ask_.config.model} ({ask_.config.model_cutoff}); the provider is "
                            f"{spec.model} ({spec.model_cutoff})")  # fmt: skip
    body = request_body(spec, ask_.rendered, ask_.asked)
    recorded = cassettes.call(spec.name, spec.endpoint, body, lambda: _send(spec, body), budget=budget, root=root)
    reading = read(spec, recorded.response, ask_.asked)
    return Judgment(
        question=Ref(id=ask_.question.id, version=ask_.question.version),
        state_hash=ask_.state_hash,
        config=ask_.config,
        reported=reading.reported,
        distribution=reading.distribution,
        refusal=reading.refusal,
        precision=reading.precision,
        response=cassettes.request_key(spec.name, spec.endpoint, body),
        latency_ms=recorded.latency_ms,
        cost_usd=Decimal(str(recorded.cost_usd)),
        created_at=datetime.fromisoformat(recorded.recorded_at) if recorded.recorded_at else datetime.now(UTC),
    )


def judge_all(
    asks: Sequence[Ask], *, budget: cassettes.Budget | None = None, root: Any = None, workers: int = 16,
    on_done: Callable[[int, Judgment], None] | None = None,
) -> list[Judgment]:
    """Many judgments at once, each its own request (one question per request), in the order asked."""
    budget = budget or cassettes.Budget()
    out: list[Judgment | None] = [None] * len(asks)

    def one(i: int) -> None:
        out[i] = judge(asks[i], budget=budget, root=root)
        if on_done is not None:
            on_done(i, out[i])  # type: ignore[arg-type]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in [pool.submit(one, i) for i in range(len(asks))]:
            f.result()
    return [j for j in out if j is not None]
