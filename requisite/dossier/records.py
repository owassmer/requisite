"""Pages for the records every case may hold: evidence (artifacts, extractions, claims, items, and their synthetic
counterparts), judgments, agent runs, approvals, gold labels, and any kind no other renderer names, shown whole."""

from __future__ import annotations

from typing import Any

from requisite import core
from requisite.dossier.html import code, dl, esc, heading, link, para, pre, table, ul, value
from requisite.dossier.refs import refs
from requisite.dossier.site import WALL_CLOCK, Page, renders, shown, summarizes


def generic(page: Page, key: str, body: Any, kind: str) -> str:
    """A record of any kind, each of its fields as committed."""
    b = shown(kind, body)
    if isinstance(b, dict):
        return dl((k, value(v)) for k, v in b.items())
    return value(b)


def omitted(kind: str) -> str:
    drop = WALL_CLOCK.get(kind, ())
    if not drop:
        return ""
    return para(f'<span class="note">Left out: {", ".join(code(f) for f in drop)}, which a build writes from its own '
                "clock rather than from what it recorded, so the same state built twice differs in it.</span>")  # fmt: skip


def when(v: object) -> str:
    return esc(v.isoformat()) if hasattr(v, "isoformat") else value(v)


def judgment_config(c: core.JudgeConfig) -> str:
    cut = c.model_cutoff if isinstance(c.model_cutoff, str) else c.model_cutoff.isoformat()
    parts = [f"{c.provider} {c.model} (training cutoff {cut})", f"paraphrase {c.paraphrase_id}", f"render {c.render_id}"]
    if c.distractor_id:
        parts.append(f"distractor {c.distractor_id}")
    return esc(", ".join(parts))


def config_order(c: core.JudgeConfig) -> tuple[str, ...]:
    """A configuration's place in a list: by provider, model, paraphrase, render and distractor."""
    return (c.provider, c.model, c.paraphrase_id, c.render_id, c.distractor_id or "")


def distribution_table(j: core.Judgment) -> str:
    if j.distribution is None:
        return para(f"No distribution: the provider's answer was refused, as recorded: {esc(j.refusal)}")
    reported = j.reported or {}
    rows = [[esc(a), esc(repr(reported.get(a))), esc(repr(p))] for a, p in j.distribution.items()]
    return table(["answer", f"reported (at a step of {esc(repr(j.precision))})", "distribution"], rows)


def distribution_line(j: core.Judgment) -> str:
    if j.distribution is None:
        return f"refused: {esc(j.refusal)}"
    return esc(", ".join(f"{a} {p!r}" for a, p in j.distribution.items()))


# ---------------------------------------------------------------- evidence


@renders("artifact")
def artifact(page: Page, key: str, body: Any) -> str:
    a = core.Artifact.model_validate_json(core.canonical_json(body))
    site = page.site
    pairs = [
        ("id", code(a.id)), ("source", esc(a.source)), ("retrieved from", code(a.locator)), ("media type", esc(a.media_type)),
        ("size in bytes", esc(a.size)), ("published at", when(a.published_at)),
        ("published at, from", esc(a.published_at_basis)), ("observed at (retrieved)", when(a.observed_at)),
        ("availability", esc(a.availability)),
    ]  # fmt: skip
    if a.archive_locator:
        pairs += [("archive snapshot", code(a.archive_locator)), ("archived at", when(a.archived_at))]
    if a.predecessor:
        pairs.append(("predecessor", page.artifact(a.predecessor)))
    out = [dl(pairs)]
    xs = refs(site).extractions.get(a.id, [])
    if xs:
        out.append(heading(2, "Its extracted text"))
        out.append(ul(page.record("extraction", x, x.split("|", 1)[1]) for x in xs))
    if a.id in refs(site).blobs:
        path = site.text_path("artifact", a.id, ".json" if "json" in a.media_type else ".txt")
        site.files.add(path, site.case.content(a.id))
        out.append(para(link(page.href(path), "Its content as retrieved") + " (a copy of the committed bytes)."))
    else:
        out.append(para('<span class="note">Its bytes are kept in the case\'s content-addressed store under its id; '
                        "the dossier shows its text where an extraction of it is committed.</span>"))  # fmt: skip
    return "".join(out)


@summarizes("artifact")
def _artifact_line(page: Page, key: str, body: Any) -> str:
    return esc(f"{body['source']}, {body['media_type']}, {body['locator']}")


@renders("extraction")
def extraction(page: Page, key: str, body: Any) -> str:
    art, _, ext = key.partition("|")
    site = page.site
    path = site.text_path("extraction", key, ".txt")
    site.files.add(path, str(body["text"]))
    return dl([("artifact", page.artifact(art)), ("extraction", code(ext)),
               ("its text", link(page.href(path), "the extracted text") + " (as committed)")])  # fmt: skip


@summarizes("extraction")
def _extraction_line(page: Page, key: str, body: Any) -> str:
    art, _, ext = key.partition("|")
    return esc(f"{ext} of {art}")


@renders("claim")
def claim(page: Page, key: str, body: Any) -> str:
    c = core.Claim.model_validate_json(core.canonical_json(body))
    return dl([("artifact", page.artifact(c.artifact)),
               ("extraction", page.record("extraction", f"{c.artifact}|{c.extraction}", c.extraction)),
               ("span", esc(f"[{c.span[0]}, {c.span[1]})")), ("authority", esc(c.authority)),
               ("text", pre(c.text))])  # fmt: skip


@renders("synthetic_artifact")
def synthetic_artifact(page: Page, key: str, body: Any) -> str:
    a = core.SyntheticArtifact.model_validate_json(core.canonical_json(body))
    site = page.site
    text = site.case.content(a.id).decode()
    path = site.text_path("synthetic", a.id, ".json" if "json" in a.media_type else ".txt")
    site.files.add(path, text)
    return dl([
        ("id", code(a.id)), ("namespace", esc(a.namespace)), ("scenario", esc(a.scenario)), ("generator", esc(a.generator)),
        ("media type", esc(a.media_type)), ("size in bytes", esc(a.size)),
        ("published at (as the document states)", when(a.published_at)),
        ("content", link(page.href(path), "as generated") + pre(text)),
    ]) + omitted("synthetic_artifact")  # fmt: skip


@summarizes("synthetic_artifact")
def _synthetic_line(page: Page, key: str, body: Any) -> str:
    return esc(f"scenario {body['scenario']}, {body['generator']}")


@renders("synthetic_claim")
def synthetic_claim(page: Page, key: str, body: Any) -> str:
    c = core.SyntheticClaim.model_validate_json(core.canonical_json(body))
    return dl([("artifact", page.artifact(c.artifact)), ("span", esc(f"[{c.span[0]}, {c.span[1]})")),
               ("authority", esc(c.authority)), ("text", pre(c.text))])  # fmt: skip


@renders("item")
def item(page: Page, key: str, body: Any) -> str:
    i = core.Item.model_validate_json(core.canonical_json(body))
    xs = refs(page.site).extractions
    extracted = [page.record("extraction", x, x) for a in i.artifacts for x in xs.get(a, [])
                 if x.split("|", 1)[1] in i.extractions]  # fmt: skip
    return dl([
        ("item", code(i.id)), ("source", esc(i.source)), ("filed in", esc(i.container)), ("posted at", when(i.posted_at)),
        ("submitter", esc(i.submitter) if i.submitter else "<em>no name published</em>"),
        ("artifacts (its record, then each attachment)", ul(page.artifact(a) for a in i.artifacts)),
        ("extractions", ul(extracted) if extracted else "<em>none</em>"),
        ("unreadable", ul(esc(u) for u in i.unreadable) if i.unreadable else "<em>none</em>"),
        ("text (what its submitter filed, each part headed by its artifact)", pre(i.text)),
    ])  # fmt: skip


@summarizes("item")
def _item_line(page: Page, key: str, body: Any) -> str:
    return esc(f"{body['id']} in {body['container']}, posted {body['posted_at']}")


# ---------------------------------------------------------------- judgments and runs


@renders("judgment")
def judgment(page: Page, key: str, body: Any) -> str:
    j = core.Judgment.model_validate_json(core.canonical_json(body))
    return dl([
        ("question", esc(f"{j.question.id} v{j.question.version}")), ("state", code(j.state_hash)),
        ("configuration", judgment_config(j.config)), ("answer", distribution_table(j)),
        ("recorded response (cassette key)", code(j.response)), ("latency (ms)", esc(j.latency_ms)),
        ("cost (USD)", esc(j.cost_usd)), ("answered at", when(j.created_at)),
    ])  # fmt: skip


@summarizes("judgment")
def _judgment_line(page: Page, key: str, body: Any) -> str:
    c = body["config"]
    return esc(f"{body['question']['id']} v{body['question']['version']}: {c['provider']} {c['paraphrase_id']} "
               f"{c['render_id']}")  # fmt: skip


@renders("agent_run")
def agent_run(page: Page, key: str, body: Any) -> str:
    r = core.AgentRun.model_validate_json(core.canonical_json(body))
    return dl([
        ("run", code(r.id)), ("role", esc(r.role)), ("model", esc(r.model)), ("effort", esc(r.effort)),
        ("its input (hash)", code(r.input_hash)), ("its answer (hash)", code(r.output_hash)),
        ("recorded call (cassette key)", code(r.response)),
        ("proposals made from its answer", ul(page.proposal(p) for p in r.proposals) if r.proposals else "<em>none</em>"),
    ]) + omitted("agent_run")  # fmt: skip


@summarizes("agent_run")
def _run_line(page: Page, key: str, body: Any) -> str:
    return esc(f"{body['role']} ({body['model']}, {body['effort']})")


@renders("approval")
def approval(page: Page, key: str, body: Any) -> str:
    a = core.Approval.model_validate_json(core.canonical_json(body))
    subject = (page.proposal(a.subject.id) if a.subject.id in page.site.case.proposals
               else esc(f"{a.subject.id} v{a.subject.version}"))  # fmt: skip
    return dl([
        ("gate", esc(a.gate)), ("subject", subject), ("approves the payload with lineage", code(a.lineage_hash)),
        ("approver", esc(a.approver)), ("at", when(a.at)), ("decision", esc(a.decision)), ("reason", esc(a.reason)),
        ("scope", esc(a.scope)),
    ])  # fmt: skip


@summarizes("approval")
def _approval_line(page: Page, key: str, body: Any) -> str:
    return esc(f"{body['gate']} {body['decision']} of {body['subject']['id']} by {body['approver']}")


@renders("gold_label")
def gold_label(page: Page, key: str, body: Any) -> str:
    g = core.GoldLabel.model_validate_json(core.canonical_json(body))
    keys = refs(page.site).items.get(g.item, [])
    shown_item = ", ".join(page.record("item", k, g.item) for k in keys) if keys else esc(g.item)
    return dl([
        ("item", shown_item), ("question", esc(f"{g.question.id} v{g.question.version}")),
        ("design", esc(f"{g.design.id} v{g.design.version}")), ("stratum", esc(g.stratum)),
        ("inclusion probability (phase 2, given phase 1)", esc(f"{g.inclusion_prob.numerator}/{g.inclusion_prob.denominator}")),
        ("outcome", esc(g.outcome)), ("label", esc(g.label) if g.label is not None else "<em>none: unresolved</em>"),
        ("labeler", esc(g.labeler)), ("labeled at", when(g.labeled_at)), ("blind to every model's answer", value(g.blind)),
    ])  # fmt: skip


@summarizes("gold_label")
def _gold_line(page: Page, key: str, body: Any) -> str:
    return esc(f"{body['item']}: {body['question']['id']} = {body['label'] or 'unresolved'} ({body['labeler']})")


@renders("derivation")
def derivation(page: Page, key: str, body: Any) -> str:
    return generic(page, key, body, "derivation") + omitted("derivation")
