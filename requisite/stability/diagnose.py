"""Resolving a question's judgments: its stability, its primary diagnosis and its application state (spec §2.1, §4
ResolvedJudgment; addendum A01 §2.1-2.4, A04 §4.1).

A question asked in K configurations (spec §2.1's axes: judge, paraphrase, render) is stable when S_arg >= 0.9 and
S_prob >= 0.85 (STABILITY_THRESHOLDS, versioned heuristics: A01 §2.1). An unstable one gets probes in spec §2.1's fixed
order and exactly one primary diagnosis, every probe's evidence kept (A01 §2.4: "Rename the result in the record to
primary_diagnosis and retain all probe evidence and competing explanations"):

- the base run's paraphrase-versus-provider split (spec §2.1: "comes free from the base run"): grouped by paraphrase,
  each paraphrase is stable across judges and renders while the whole is not, so the disagreement lies between
  paraphrases (imprecise_target); grouped by provider, each provider is stable across paraphrases and renders while the
  whole is not, so the providers disagree (contested). Where both hold, the table's order decides (imprecise_target).
- the split probe (spec §2.1 compound): a proposed split whose sub-questions are each stable and whose composition is a
  valid decision list over the parent's answers (requisite/library/nodes/compose.py). It runs in the requisite loop,
  where an agent proposes the split; `split_probe` reads its result.
- the evidence and state probes are Tier 1 (spec §13 WP4: "0 for provider, paraphrase and split; 1 for evidence and
  state probes"); none is recorded as run, so none is reported as not having fired.
- none fired: unexplained.

The application policy (A04 §4.1: a provider's probability never by itself makes its top answer a fact; the policy is
named and versioned): a stable question whose pooled distribution has one top answer applies it, established where
that answer asserts or denies and not established where it is the not_established answer; any other question applies
nothing ("review"), held as not established, or as contested where providers disagree (spec §2.1: "hold as
not_established until resolved"; G2). A forecast applies no answer at all: its output is its distribution (A04 §4.1: "A
forecast's top answer must never replace its branch probabilities").
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from requisite import core
from requisite.core.base import FrozenDict, freeze
from requisite.core.judgment import Diagnosis, EvidenceStatus
from requisite.library.nodes.compose import binds_key, decision_list
from requisite.stability.scores import EQUAL_PER_PROVIDER, TOP_SET_EQUALITY, Scores, pool, scores, top_set

S_ARG, S_PROB = 0.9, 0.85  # spec §2.1's starting thresholds
STABILITY_THRESHOLDS = core.Ref(id="stability_thresholds.spec_2_1", version=1)
POOLING = core.Ref(id=EQUAL_PER_PROVIDER.name, version=EQUAL_PER_PROVIDER.version)
TIES = core.Ref(id=TOP_SET_EQUALITY.name, version=TOP_SET_EQUALITY.version)
APPLY_STABLE_TOP = core.Ref(id="application.stable_pooled_top", version=1)
APPLY_FORECAST = core.Ref(id="application.forecast_applies_no_answer", version=1)


def is_stable(sc: Scores) -> bool:
    return sc.s_arg >= S_ARG and sc.s_prob >= S_PROB


def usable(judgments: Sequence[core.Judgment]) -> list[core.Judgment]:
    """The judgments that returned a distribution; a refusal is an answer with no probabilities, never a zero."""
    return [j for j in judgments if j.distribution is not None]


def _dist(j: core.Judgment) -> dict[str, float]:
    assert j.distribution is not None
    return dict(j.distribution)


def _group(judgments: Sequence[core.Judgment], axis: Callable[[core.JudgeConfig], str]) -> dict[str, list[core.Judgment]]:
    out: dict[str, list[core.Judgment]] = {}
    for j in judgments:
        out.setdefault(axis(j.config), []).append(j)
    return out


def _within(groups: Mapping[str, Sequence[core.Judgment]], name: str) -> tuple[bool, dict[str, float], list[str]]:
    """Whether every group is stable within itself; each group's scores; why not, where not."""
    ok, out, why = True, {}, []
    for key, js in sorted(groups.items()):
        if len(js) < 2:
            ok = False
            why.append(f"{name} {key} has {len(js)} configuration(s), too few to score")
            continue
        sc = scores([_dist(j) for j in js])
        out[f"{name}:{key}.s_arg"], out[f"{name}:{key}.s_prob"] = sc.s_arg, sc.s_prob
        if not is_stable(sc):
            ok = False
            why.append(f"{name} {key} is unstable within (S_arg {sc.s_arg:.3f}, S_prob {sc.s_prob:.3f})")
    return ok, out, why


def base_probe(judgments: Sequence[core.Judgment]) -> core.ProbeResult:
    """The paraphrase-versus-provider split of the base run (spec §2.1): which axis the disagreement lies along."""
    js = usable(judgments)
    overall = scores([_dist(j) for j in js])
    by_paraphrase = _group(js, lambda c: c.paraphrase_id)
    by_provider = _group(js, lambda c: c.provider)
    para_ok, para_scores, para_why = _within(by_paraphrase, "paraphrase")
    prov_ok, prov_scores, prov_why = _within(by_provider, "provider")
    unstable = not is_stable(overall)
    imprecise = unstable and len(by_paraphrase) >= 2 and para_ok
    contested = unstable and len(by_provider) >= 2 and prov_ok
    indicates: Diagnosis | None = "imprecise_target" if imprecise else "contested" if contested else None
    if not unstable:
        detail = "the question is stable over every configuration"
    elif indicates == "imprecise_target":
        detail = ("each paraphrase is stable across judges and renders while the whole is not: the disagreement "
                  "concentrates across paraphrases")  # fmt: skip
        if contested:
            detail += "; each provider is stable across paraphrases too (spec §2.1's order puts imprecise_target first)"
    elif indicates == "contested":
        detail = "each provider is stable across paraphrases and renders while the whole is not: the providers disagree"
    else:
        detail = "neither axis explains the disagreement: " + "; ".join(para_why[:3] + prov_why[:3])
    refused = len(judgments) - len(js)
    if refused:
        detail += f" ({refused} configuration(s) refused and are not scored)"
    sc = {"s_arg": overall.s_arg, "s_prob": overall.s_prob, **para_scores, **prov_scores}
    return core.ProbeResult(probe="paraphrase_provider", fired=indicates is not None, indicates=indicates, scores=sc,
                            detail=detail, judgments=[j.content_hash() for j in js])  # fmt: skip


@dataclass(frozen=True)
class SubQuestion:
    """A proposed sub-question as the split probe reads it: the question, its judgments in the stability configurations,
    and for each of its answer values the binds a terminal answer gives the parent (empty: continue)."""

    question: core.AtomicQuestion
    judgments: Sequence[core.Judgment]


def feasible_binds(q: core.AtomicQuestion, values: frozenset[str]) -> tuple[FrozenDict, ...]:
    """The binds of the answers a question's state leaves feasible: `values` read as themselves, or, where they include
    a not_established or contested answer, as every substantive answer (addendum A03 §3.3)."""
    by_value = {a.value: a for a in q.answers}
    substantive = [a for a in q.answers if a.status in ("affirmative", "negative")]
    if any(by_value[v].status in ("not_established", "contested") for v in values):
        chosen = substantive
    else:
        chosen = [by_value[v] for v in sorted(values)]
    return tuple(freeze(dict(a.binds)) for a in chosen)


def composition_problems(parent: core.AtomicQuestion, subs: Sequence[core.AtomicQuestion]) -> list[str]:
    """Why a split's sub-questions do not compose into the parent's answers, or none: the last sub-question's
    substantive answers all bind, every binding names exactly the keys the parent's answers bind, and every value it
    binds is one a parent's substantive answer binds or, for an interval, lies within the parent's (a refinement may
    be finer, never wider)."""
    out: list[str] = []
    if not subs:
        return ["a split has at least one sub-question"]
    parent_binds = [dict(a.binds) for a in parent.answers if a.status in ("affirmative", "negative")]
    keys = {k for b in parent_binds for k in b}
    if not keys:
        return [f"{parent.id}'s answers bind nothing, so no composition can reproduce them"]
    for i, q in enumerate(subs):
        last = i == len(subs) - 1
        for a in q.answers:
            if a.status not in ("affirmative", "negative"):
                if a.binds:
                    out.append(f"{q.id} answer {a.value} is {a.status} and binds {dict(a.binds)}: it asserts nothing")
                continue
            if not a.binds:
                if last:
                    out.append(f"{q.id} is the last sub-question and its answer {a.value} binds nothing")
                continue
            if set(a.binds) != keys:
                out.append(f"{q.id} answer {a.value} binds {sorted(a.binds)}, not the parent's {sorted(keys)}")
                continue
            if not any(_within_parent(dict(a.binds), b) for b in parent_binds):
                out.append(f"{q.id} answer {a.value} binds {dict(a.binds)}, which no answer of {parent.id} contains")
    return out


def _within_parent(sub: Mapping[str, object], parent: Mapping[str, object]) -> bool:
    for k, v in sub.items():
        p = parent[k]
        if isinstance(p, list | tuple) and isinstance(v, list | tuple) and len(p) == len(v) == 2:
            lo, hi = Fraction(str(v[0])), None if v[1] is None else Fraction(str(v[1]))
            plo, phi = Fraction(str(p[0])), None if p[1] is None else Fraction(str(p[1]))
            if lo < plo or (phi is not None and (hi is None or hi > phi)) or (hi is not None and hi < lo):
                return False
        elif v != p:
            return False
    return True


def composed_scores(subs: Sequence[SubQuestion]) -> tuple[float, int]:
    """The composed answer's decision stability: in each configuration every sub-question asked in it answered, its
    sub-questions' top answers composed by the decision list; the share of those configurations whose composed answer
    equals the most common one (spec §2.1 S_arg, read over the composition). Returns the share and the configurations."""
    per_config: dict[str, list[frozenset[str] | None]] = {}
    for i, s in enumerate(subs):
        for j in usable(s.judgments):
            key = j.config.content_hash()
            per_config.setdefault(key, [None] * len(subs))[i] = top_set(_dist(j))
    composed: list[str] = []
    for tops in per_config.values():
        if any(t is None for t in tops):
            continue
        steps = [feasible_binds(s.question, t) for s, t in zip(subs, tops, strict=True) if t is not None]
        composed.append("|".join(binds_key(b) for b in decision_list(steps)))
    if not composed:
        return 0.0, 0
    most = max(composed.count(c) for c in set(composed))
    return most / len(composed), len(composed)


def split_probe(parent: core.AtomicQuestion, subs: Sequence[SubQuestion], proposals: Sequence[str]) -> core.ProbeResult:
    """The split probe (spec §2.1 compound): it fires when every proposed sub-question is stable and their composition
    reproduces the parent's answers (`composition_problems`)."""
    sc: dict[str, float] = {}
    why = composition_problems(parent, [s.question for s in subs])
    stable_all = True
    for s in subs:
        js = usable(s.judgments)
        if len(js) < 2:
            stable_all = False
            why.append(f"{s.question.id} returned {len(js)} distribution(s), too few to score")
            continue
        x = scores([_dist(j) for j in js])
        sc[f"{s.question.id}.s_arg"], sc[f"{s.question.id}.s_prob"] = x.s_arg, x.s_prob
        if not is_stable(x):
            stable_all = False
            why.append(f"{s.question.id} is unstable (S_arg {x.s_arg:.3f}, S_prob {x.s_prob:.3f})")
    composed, n = composed_scores(subs)
    sc["composed.s_arg"] = composed
    fired = stable_all and not why
    detail = (f"the {len(subs)} proposed sub-question(s) are each stable and compose into {parent.id}'s answers; "
              f"the composed answer's S_arg is {composed:.3f} over {n} configurations" if fired
              else "the proposed split does not hold: " + "; ".join(why))  # fmt: skip
    detail += f" (proposals {', '.join(proposals)})" if proposals else ""
    return core.ProbeResult(probe="split", fired=fired, indicates="compound" if fired else None, scores=sc,
                            detail=detail, judgments=[j.content_hash() for s in subs for j in usable(s.judgments)])  # fmt: skip


def resolve(
    question: core.AtomicQuestion,
    state_hash: str,
    judgments: Sequence[core.Judgment],
    *,
    probes: Sequence[core.ProbeResult] = (),
    approvals: Sequence[str] = (),
) -> core.ResolvedJudgment:
    """A question's judgments combined (spec §4 ResolvedJudgment, as amended): pooled, scored, diagnosed and applied.
    `probes` are the probes run after the base run (the split probe, from the requisite loop), in spec §2.1's order."""
    js = usable(judgments)
    if len(js) < 2:
        raise ValueError(f"{question.id} has {len(js)} distribution(s) over {len(judgments)} configurations: stability "
                         "compares at least two")  # fmt: skip
    if any(j.question != question.ref or j.state_hash != state_hash for j in judgments):
        raise ValueError(f"a judgment combined for {question.id} v{question.version} answers another question or state")
    sc = scores([_dist(j) for j in js])
    pooled = pool([(j.config.provider, _dist(j)) for j in js], EQUAL_PER_PROVIDER)
    stable = is_stable(sc)
    base = base_probe(judgments)
    ran = [base, *probes]
    if stable:
        diagnosis: Diagnosis = "stable"
    else:
        diagnosis = next((p.indicates for p in ran if p.fired and p.indicates is not None), "unexplained")
    top = top_set(pooled.distribution)
    by_value = {a.value: a for a in question.answers}
    policy = APPLY_FORECAST if question.task == "forecast" else APPLY_STABLE_TOP
    status: EvidenceStatus
    if question.task == "forecast":
        status, applied = "not_established", "review"
    elif stable and len(top) == 1:
        (applied,) = top
        status = "not_established" if by_value[applied].status in ("not_established", "contested") else "established"
    else:
        status, applied = ("contested" if diagnosis == "contested" else "not_established"), "review"
    return core.ResolvedJudgment(
        question=question.ref, state_hash=state_hash, judgments=[j.content_hash() for j in js],
        pooled=dict(pooled.distribution), pooling_policy=POOLING, s_arg=sc.s_arg, s_prob=sc.s_prob, tie_policy=TIES,
        thresholds=STABILITY_THRESHOLDS, stable=stable, primary_diagnosis=diagnosis, probes=ran,
        evidence_status=status, application_policy=policy, application_state=applied, approvals=list(approvals),
    )  # fmt: skip


# Each primary diagnosis's prescribed action (spec §2.1's table), and whether an agent can take it at Tier 0.
ACTIONS: Mapping[str, tuple[str, str]] = {
    "imprecise_target": ("refine", "agent refines the target and answer meanings; re-test"),
    "contested": ("human", "human review (G2); held as not_established until resolved"),
    "evidence_gap": ("fetch", "agent fetches or attaches the evidence; re-test"),
    "compound": ("split", "replace with sub-questions plus a composition node"),
    "incoherent_state": ("fix_state", "fix the state construction"),
    "unexplained": ("human", "human review; saved as a fixture"),
}
