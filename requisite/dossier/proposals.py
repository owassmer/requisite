"""A proposal's page: its kind, author, status, every gate with its detail, the decisions recorded on it, and its payload
rendered for its kind (spec §3 I1: a proposal becomes active only after its required gates pass). A rejected proposal
is shown as committed, marked, never hidden. And the review queue: the proposals awaiting a human gate (spec §2.4)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from requisite import core
from requisite.dossier.html import badge, code, dl, esc, heading, para, pre, table, ul, value
from requisite.dossier.refs import refs
from requisite.dossier.site import PAYLOADS, Page, Site, renders, renders_payload, status_badge, summarizes

# The human gate a proposal of each kind awaits, and what it asks (spec §2.4: "G1: the objective, horizon and loss
# definition"; spec §7.1 step 1: the entity model's parameters, "a human confirms at gate G1"). core.REQUIRED_GATES gives
# a human gate to exactly these kinds; G2-G4 approve committed state rather than a proposal (core.governance).
HUMAN_GATES: dict[str, tuple[str, str]] = {
    "decision_spec": ("G1", "the decision's objective, horizon and loss definition"),
    "objective": ("G1", "the run's objective: its population, events, horizon, loss definition and estimand"),
    "entity_model": ("G1", "the entity model: its driver tree, liquidity profile, horizon and loss definition"),
}


def awaiting(p: core.Proposal) -> bool:
    """Whether a proposal awaits a human's decision: proposed, its last gate the human one, not passed."""
    return p.status == "proposed" and bool(p.gates) and p.gates[-1].gate == "human" and not p.gates[-1].passed


def gates_table(gates: Sequence[core.GateResult]) -> str:
    if not gates:
        return para("<em>No gate has run.</em>")
    rows = [[esc(g.gate), badge("passed", "passed") if g.passed else badge("not passed", "failed"), pre(g.detail)]
            for g in gates]  # fmt: skip
    return table(["gate", "result", "detail"], rows)


def authored(page: Page, author: str) -> str:
    runs = refs(page.site).agent_runs.get(author, [])
    if not runs:
        return f"{code(author)} <span class=\"note\">(names no agent run committed in this case)</span>"
    return ul(page.record("agent_run", key, f"agent run {r.id}") + esc(f": {r.role}, {r.model}, effort {r.effort}")
              for key, r in runs)  # fmt: skip


@renders("proposal")
def proposal(page: Page, key: str, body: Any) -> str:
    case = page.site.case
    p = case.proposals[str(body["id"])]
    out = []
    if p.status == "rejected":
        failed = p.gates[-1]
        out.append(para(badge("rejected", "rejected") + esc(f" Its {failed.gate} gate did not pass; it stands as the "
                                                              "record of what was proposed and why it was refused.")))
    if awaiting(p):
        gate, asks = HUMAN_GATES[p.kind]
        out.append(para(badge("awaiting", "awaiting") + esc(f" Awaits {gate}: a human decides on {asks}, as proposed.")))
    out.append(dl([("proposal", code(p.id)), ("kind", esc(p.kind)), ("status", status_badge(p.status)),
                   ("case", esc(p.case)), ("author", authored(page, p.author))]))  # fmt: skip
    out.append(heading(2, "Gates") + gates_table(p.gates))
    history = case.histories.get(p.id, [])
    if len(history) > 1:
        events = [f"{status_badge(status)} " + (", ".join(esc(f"{g.gate} {'passed' if g.passed else 'not passed'}")
                                                          for g in gates) or "no gate ran")
                  for status, gates in history]  # fmt: skip
        out.append(heading(2, "Its decisions, in order") + "<ol>" + "".join(f"<li>{e}</li>" for e in events) + "</ol>")
    approvals = refs(page.site).approvals.get(p.id, [])
    if approvals:
        rows = [[page.record("approval", k, a.id), esc(a.gate), esc(a.decision), esc(a.approver), esc(a.at.isoformat()),
                 esc(a.reason), esc(a.scope), code(a.lineage_hash)] for k, a in approvals]  # fmt: skip
        out.append(heading(2, "Human decisions on it") +
                   table(["approval", "gate", "decision", "approver", "at", "reason", "scope", "payload lineage"], rows))
    out.append(heading(2, "Rationale") + pre(p.rationale))
    out.append(heading(2, f"Payload: a {esc(p.kind)}") + PAYLOADS.get(p.kind, payload_fields)(page, p))
    return "".join(out)


@summarizes("proposal")
def _proposal_line(page: Page, key: str, body: Any) -> str:
    p = page.site.case.proposals[str(body["id"])]
    return esc(p.kind) + " " + status_badge(p.status)


def payload_fields(page: Page, p: core.Proposal) -> str:
    return dl((k, value(v)) for k, v in dict(p.payload).items())


@renders_payload("paraphrase_set")
def paraphrase_set(page: Page, p: core.Proposal) -> str:
    s = core.ParaphraseSet.model_validate_json(core.canonical_json(dict(p.payload)))
    out = [dl([("set", esc(f"{s.id} v{s.version}")), ("template", esc(f"{s.template.id} v{s.template.version}")),
               ("slots", esc(", ".join(s.slots))), ("answers", esc(", ".join(s.answers)))])]  # fmt: skip
    for x in s.paraphrases:
        rows = [[esc(a), esc(m)] for a, m in x.meanings.items()]
        out.append(heading(3, esc(f"Paraphrase {x.id}") + (" (the original wording)" if x.id == "p0" else ""))
                   + pre(x.target) + table(["answer", "meaning in this wording"], rows))  # fmt: skip
    return "".join(out)


@renders_payload("question")
def question(page: Page, p: core.Proposal) -> str:
    q = core.AtomicQuestion.model_validate_json(core.canonical_json(dict(p.payload)))
    answers = table(["value", "meaning", "status", "binds"],
                    [[esc(a.value), esc(a.meaning), esc(a.status), value(dict(a.binds) or None)] for a in q.answers])
    return dl([
        ("question", esc(f"{q.id} v{q.version}")), ("template", value(q.template_id)),
        ("bindings", value(dict(q.bindings) or None)), ("task", esc(q.task)), ("target", pre(q.target)),
        ("window", value(q.window.model_dump(mode="json") if q.window else None)), ("answers", answers),
        ("definitions", value(list(q.definitions) or None)), ("evidence roles", value(list(q.evidence_roles) or None)),
        ("paraphrase set", value(q.paraphrase_set)), ("reaches", value(list(q.reaches) or None)),
        ("replaces", value([r.model_dump(mode="json") for r in q.replaces] or None)),
    ])  # fmt: skip


def queue(site: Site) -> None:
    """The review queue: every proposal awaiting a human gate, with what the gate asks."""
    path = site.page_path("queue")
    page = Page(site, path)
    waiting = [p for p in site.case.all_proposals() if awaiting(p)]
    if waiting:
        rows = []
        for p in waiting:
            gate, asks = HUMAN_GATES[p.kind]
            rows.append([page.proposal(p.id), esc(p.kind), esc(gate), esc(f"a decision on {asks}, as proposed"),
                         esc(p.gates[-1].detail)])  # fmt: skip
        body = table(["proposal", "kind", "gate", "what it asks", "the gate's detail"], rows)
    else:
        body = para("No proposal in this case awaits a human gate.")
    rule = para('<span class="note">The four gates that always require a human (spec §2.4): G1, the objective, horizon '
                "and loss definition; G2, any contested diagnosis on a relevant question; G3, contract acceptance; G4, "
                "template promotion. A proposal awaits one when its required gates "
                "(core.REQUIRED_GATES) end at a human gate not yet decided.</span>")  # fmt: skip
    site.write(path, "Review queue", body + rule)
