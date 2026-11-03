"""Wide discovery in the dossier (spec §7.2-7.4; design/decisions.md, "F4's sample design"): the objective and its G1
approval, the frame criteria and the frozen frame (its searches, the dockets kept and dropped with their reasons, its
items), the sample draws (strata, sizes, inclusion probabilities), every judgment by question and configuration, and
the gold labels. Each judgment is shown with its provider, model and training cutoff, configuration and distribution."""

from __future__ import annotations

from typing import Any

from requisite import core
from requisite.dossier.contracts import window
from requisite.dossier.html import badge, code, dl, esc, heading, link, para, table, ul
from requisite.dossier.records import config_order, judgment_config
from requisite.dossier.refs import refs
from requisite.dossier.site import Page, Site, renders, renders_payload, summarizes


def fraction(p: object) -> str:
    return f"{p.numerator}/{p.denominator}" if hasattr(p, "numerator") and hasattr(p, "denominator") else str(p)


@renders_payload("objective")
def objective(page: Page, p: core.Proposal) -> str:
    o = core.WideObjective.model_validate_json(core.canonical_json(dict(p.payload)))
    return dl([("objective", esc(f"{o.id} v{o.version}")), ("population", esc(o.population)), ("events", esc(o.events)),
               ("horizon", esc(window(o.horizon))), ("loss definition", esc(o.loss_definition)),
               ("estimand", esc(o.estimand)), ("unit", esc(o.unit))])  # fmt: skip


@renders_payload("frame")
def frame_criteria(page: Page, p: core.Proposal) -> str:
    c = core.FrameCriteria.model_validate_json(core.canonical_json(dict(p.payload)))
    frames = [(k, f) for k, f in page.site.case.records("frame", core.Frame) if f.criteria == core.Ref(id=c.id, version=c.version)]
    return dl([
        ("criteria", esc(f"{c.id} v{c.version}")), ("objective", esc(f"{c.objective.id} v{c.objective.version}")),
        ("source", esc(c.source)), ("as of", esc(c.as_of.isoformat())), ("search terms", ul(esc(t) for t in c.search_terms)),
        ("document types", esc(", ".join(c.document_types))),
        ("posted", esc(f"from {c.posted_from.isoformat()}" + (f" to {c.posted_to.isoformat()}" if c.posted_to else ""))),
        ("agencies", esc(", ".join(c.agencies)) or "<em>any</em>"), ("comment status", esc(c.comment_status)),
        ("at least this many comments", esc(c.min_comments)),
        ("the frame its criteria froze", ul(page.record("frame", k, f.id) for k, f in frames) if frames else
         "<em>none is committed</em>"),
    ])  # fmt: skip


@renders("frame")
def frame(page: Page, key: str, body: Any) -> str:
    f = core.Frame.model_validate_json(core.canonical_json(body))
    out = [dl([("frame", code(f.id)), ("criteria", esc(f"{f.criteria.id} v{f.criteria.version}")),
               ("as of", esc(f.as_of.isoformat())),
               ("retrieved", esc(f"{f.retrieval_started.isoformat()} to {f.retrieval_ended.isoformat()}")),
               ("mode (addendum A11 §11.3)", esc(f.mode))])]  # fmt: skip
    rows = [[esc(s.term), code(s.request), esc(s.documents), ul(page.artifact(a) for a in s.artifacts)] for s in f.searches]
    out.append(heading(2, "Searches") + table(["term", "request", "documents returned", "responses"], rows))
    rows = [[code(d.docket), badge("kept", "met") if d.kept else badge("dropped", "other"), esc(d.reason),
             esc(d.comments) if d.comments is not None else "<em>not listed: its status drops it</em>",
             ul(page.artifact(a) for a in d.artifacts)] for d in f.dockets]  # fmt: skip
    out.append(heading(2, "Dockets reached") + table(["docket", "", "why", "public comments by as of", "listings"], rows))
    items = refs(page.site).items
    rows = [[", ".join(page.record("item", k, i.id) for k in items.get(i.id, [])) or esc(i.id), esc(i.docket),
             esc(i.posted_at.isoformat()), "withdrawn" if i.withdrawn else ""] for i in f.items]  # fmt: skip
    out.append(heading(2, "Items (the frame's items, each linked where it was read)")
               + table(["item", "docket", "posted at", ""], rows))
    return "".join(out)


@summarizes("frame")
def _frame_line(page: Page, key: str, body: Any) -> str:
    return esc(f"as of {body['as_of']}, {body['mode']}")


def strata(draw: core.SampleDraw) -> list[list[str]]:
    by: dict[str, list[core.DrawnItem]] = {}
    for d in draw.items:
        by.setdefault(d.stratum, []).append(d)
    return [[esc(s), esc(len(ds)), esc(", ".join(dict.fromkeys(fraction(d.inclusion_prob) for d in ds)))]
            for s, ds in by.items()]  # fmt: skip


@renders("sample_draw")
def sample_draw(page: Page, key: str, body: Any) -> str:
    d = core.SampleDraw.model_validate_json(core.canonical_json(body))
    items = refs(page.site).items
    rows = [[", ".join(page.record("item", k, i.item) for k in items.get(i.item, [])) or esc(i.item), esc(i.stratum),
             esc(fraction(i.inclusion_prob))] for i in d.items]  # fmt: skip
    return (dl([("design", esc(f"{d.design.id} v{d.design.version}")), ("phase", esc(d.phase)), ("seed", esc(d.seed))])
            + heading(2, "Strata") + table(["stratum", "items drawn", "inclusion probability"], strata(d))
            + heading(2, "Items drawn") + table(["item", "stratum", "inclusion probability"], rows))  # fmt: skip


@summarizes("sample_draw")
def _draw_line(page: Page, key: str, body: Any) -> str:
    return esc(f"phase {body['phase']} of {body['design']['id']}, seed {body['seed']}")


def wide_page(site: Site) -> None:
    path = site.page_path("wide")
    page = Page(site, path)
    case = site.case
    out = [para("The wide flow (spec §7.2): from an objective approved at G1, the frame it reaches in the public record, "
                "its samples by the declared design, and the judges' answers and the gold labels over them.")]  # fmt: skip
    approvals = refs(site).approvals
    out.append(heading(2, "Objective"))
    for p in case.proposals_of("objective"):
        decided = approvals.get(p.id, [])
        rows = [[page.record("approval", k, a.id), esc(a.gate), esc(a.decision), esc(a.approver), esc(a.at.isoformat()),
                 esc(a.reason), esc(a.scope)] for k, a in decided]  # fmt: skip
        out.append(para(page.proposal(p.id)) + objective(page, p)
                   + (table(["approval", "gate", "decision", "approver", "at", "reason", "scope"], rows) if rows
                      else para("<em>No human has decided on it.</em>")))  # fmt: skip
    out.append(heading(2, "Frame"))
    for p in case.proposals_of("frame"):
        tests = [g for g in p.gates if g.gate == "tests"]
        out.append(para(page.proposal(p.id)) + frame_criteria(page, p)
                   + "".join(para("Its tests gate: " + esc(g.detail)) for g in tests))  # fmt: skip
    for k, f in case.records("frame", core.Frame):
        kept = [d.docket for d in f.dockets if d.kept]
        out.append(table(["search term", "request", "documents returned"],
                         [[esc(x.term), code(x.request), esc(x.documents)] for x in f.searches]))
        out.append(para(page.record("frame", k, f.id) + esc(f": as of {f.as_of.isoformat()} ({f.mode}); kept "
                                                           f"{', '.join(kept) or 'no docket'}")))  # fmt: skip
        out.append(table(["docket", "", "why", "public comments by as of"],
                         [[code(d.docket), "kept" if d.kept else "dropped", esc(d.reason),
                           esc(d.comments) if d.comments is not None else "<em>not listed</em>"] for d in f.dockets]))
    out.append(heading(2, "Paraphrase sets"))
    sets = case.proposals_of("paraphrase_set")
    out.append(ul(page.proposal(p.id, text=f"{p.payload.get('id')} v{p.payload.get('version')}") for p in sets)
               if sets else para("None is proposed."))  # fmt: skip
    out.append(heading(2, "Sample draws"))
    for k, d in case.records("sample_draw", core.SampleDraw):
        out.append(heading(3, page.record("sample_draw", k, f"phase {d.phase}: {k}"))
                   + para(esc(f"design {d.design.id} v{d.design.version}, seed {d.seed}"))
                   + table(["stratum", "items drawn", "inclusion probability"], strata(d)))  # fmt: skip
    out.append(heading(2, "Judgments, by question and configuration")
               + para('<span class="note">Every judgment committed in the case: the phase-1 labels and the stability '
                      "configurations alike, each with its provider, model and training cutoff, paraphrase and render, "
                      "and its distribution over the question's answers. No stability score is committed as a record "
                      "in this case, so none is shown.</span>"))  # fmt: skip
    groups: dict[str, dict[tuple[str, ...], list[tuple[str, core.Judgment]]]] = {}
    for k, j in case.records("judgment", core.Judgment):
        groups.setdefault(j.question.id, {}).setdefault(config_order(j.config), []).append((k, j))
    for qid in sorted(groups):
        answers = list(dict.fromkeys(a for by in groups[qid].values() for _, j in by for a in (j.distribution or {})))
        out.append(heading(3, code(qid)))
        for config in sorted(groups[qid]):
            js = sorted(groups[qid][config], key=lambda kj: (kj[1].state_hash, kj[0]))
            rows = [[page.record("judgment", k, k[:12]), code(j.state_hash[:16])]
                    + ([esc(repr(j.distribution[a])) if a in j.distribution else "" for a in answers]
                       if j.distribution is not None else [esc(f"refused: {j.refusal}")] + [""] * (len(answers) - 1))
                    for k, j in js]  # fmt: skip
            out.append(para(judgment_config(js[0][1].config))
                       + table(["judgment", "state", *(code(a) for a in answers)], rows))
    out.append(heading(2, "Gold labels"))
    labels = case.records("gold_label", core.GoldLabel)
    if labels:
        items = refs(site).items
        rows = [[", ".join(page.record("item", ik, g.item) for ik in items.get(g.item, [])) or esc(g.item),
                 esc(f"{g.question.id} v{g.question.version}"), esc(g.stratum), esc(fraction(g.inclusion_prob)),
                 esc(g.outcome), esc(g.label) if g.label is not None else "<em>unresolved</em>", esc(g.labeler),
                 "blind" if g.blind else "not blind", page.record("gold_label", k, "record")] for k, g in labels]  # fmt: skip
        out.append(table(["item", "question", "stratum", "inclusion probability", "outcome", "label", "labeler", "",
                          ""], rows))  # fmt: skip
    else:
        out.append(para("No gold label is committed in this case."))
    if "item" in case.keys:
        out.append(heading(2, "Items read") + para(link(page.href(site.kind_path("item")), "Every item read, by key")))
    site.write(path, "Wide discovery", "".join(out))

