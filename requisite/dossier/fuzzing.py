"""A fuzz report in the dossier (spec §8.6; addendum A09): its coverage, each scenario with its split, severity, the IR's
outcome and transfers, the prose's outcome and transfers, the resolvers' pooled distribution and its packet, the scores
against spec §8.6's starting thresholds, the hard gates, the equivalence claimed, and the attributions with their
suspected clauses linked to the rulebook's sections. A fuzz report lives in its synthetic case (spec I4), so every page
here is marked synthetic; its contract and rulebook are the real case's, linked there."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from requisite import core
from requisite.dossier.contracts import contract_record_key, node_anchor, section_anchor
from requisite.dossier.html import anchor, badge, code, dl, esc, heading, para, pre, table, ul
from requisite.dossier.records import config_order, distribution_line, judgment_config
from requisite.dossier.refs import refs
from requisite.dossier.site import Page, Site, renders, renders_payload, summarizes
from requisite.fuzz.run import AMBIGUITY, CLARITY, FIDELITY, SECTION_TITLES, THRESHOLDS


def real(site: Site) -> Site:
    """The case a synthetic case fuzzes: its contract and rulebook are committed there."""
    return site.parent or site


def transfers(ts: Sequence[core.CashTransfer]) -> str:
    if not ts:
        return "<em>none</em>"
    return table(["at", "amount", "currency", "basis"],
                 [[esc(t.at.isoformat()), esc(t.amount), esc(t.currency), esc(t.basis)] for t in ts])  # fmt: skip


def money(v: object) -> str:
    return "<em>none: the scenario defines no quantity</em>" if v is None else esc(v)


def against(name: str, measured: float, rule: str, threshold: float, met: bool) -> list[str]:
    return [esc(name), esc(repr(measured)), esc(f"{rule} {threshold}"),
            badge("met", "met") if met else badge("missed", "missed")]  # fmt: skip


def packet_documents(page: Page, packet_pid: str) -> str:
    site = page.site
    p = site.case.proposals.get(packet_pid)
    if p is None:
        return para(f"<em>No packet proposal {esc(packet_pid)} is in this case.</em>")
    out = []
    for aid in p.payload.get("documents", ()):
        text = site.case.content(str(aid)).decode() if site.case.has("synthetic_artifact", str(aid)) else None
        out.append(page.artifact(str(aid)) + (pre(text) if text is not None else ""))
    return ul(out)


@renders("fuzz_report")
def fuzz_report(page: Page, key: str, body: Any) -> str:
    r = core.FuzzReport.model_validate_json(core.canonical_json(body))
    site = page.site
    case = real(site)
    ckey = contract_record_key(r.contract)

    def node(ident: str) -> str:
        return page.record("contract", ckey, ident, site=case, anchor=node_anchor(ident))

    def scenario(sid: str) -> str:
        return f'<a href="#{esc(anchor("scenario", sid))}">{esc(sid)}</a>'

    out = [dl([
        ("contract", page.record("contract", ckey, f"{r.contract.id} v{r.contract.version}", site=case)),
        ("prose fuzzed (its rulebook)", page.proposal(r.prose, site=case)), ("prose hash", code(r.prose_hash)),
        ("resolvers (K configurations)", ul(judgment_config(c) for c in r.resolvers)),
        ("pooling policy", esc(f"{r.pooling_policy.id} v{r.pooling_policy.version}")),
        ("declared domain", esc(r.domain) if r.domain else "<em>none</em>"),
    ])]  # fmt: skip

    out.append(heading(2, "Scores against spec §8.6's starting thresholds"))
    out.append(table(["score", "measured (this report)", "starting threshold", ""], [
        against("fidelity: the share of scored scenarios whose prose outcome is the IR's", r.fidelity, ">=", FIDELITY,
                r.fidelity >= FIDELITY),
        against("clarity: the mean S_arg across configurations", r.clarity, ">=", CLARITY, r.clarity >= CLARITY),
        against("ambiguity: the mean pooled probability of AMBIGUOUS", r.ambiguity, "<=", AMBIGUITY,
                r.ambiguity <= AMBIGUITY),
    ]))  # fmt: skip
    out.append(f'<div class="rule">The thresholds are a rule, not a computation: spec §8.6 step 6 states "Starting '
               f"thresholds over the coverage set are fidelity ≥ 0.95, clarity ≥ 0.9 and ambiguity ≤ 0.05\", held in "
               f"the product as {code('requisite.fuzz.run.FIDELITY')}, {code('CLARITY')} and {code('AMBIGUITY')}, "
               f"versioned as {code(f'{THRESHOLDS.id} v{THRESHOLDS.version}')} (addendum A01 §2.1: configurable, "
               "versioned heuristics). Scores are over the scenarios whose rendering passed (spec §8.6 step 4).</div>")

    out.append(heading(2, "Hard gates (addendum A09 §10.4)"))
    out.append(table(["gate", "result", "detail"], [
        [esc(g.name), badge("passed", "passed") if g.passed else badge("not passed", "failed"), pre(g.detail)]
        for g in r.hard_gates]))  # fmt: skip
    out.append(heading(2, "Equivalence claimed (addendum A09 §10.5)") + para(code(r.equivalence) + esc(
        ": the scope of the claim that the prose resolves as the IR does, over the scenarios run.")))

    out.append(heading(2, "Coverage (addendum A09 §10.2)") + para(esc(f"Whole trigger: {r.whole_trigger}.")))
    rows = [[", ".join(node(c) for c in o.conditions), esc(o.status), ", ".join(scenario(w) for w in o.witnesses)
             or "<em>none</em>", pre(o.proof) if o.proof else "<em>none</em>",
             ", ".join(node(c) for c in o.coupled_group) or "<em>none</em>"] for o in r.obligations]  # fmt: skip
    out.append(heading(3, "Obligations") + table(["conditions", "status", "witnesses", "proof", "coupled with"], rows))
    rows = [[node(b.condition), esc(b.point), badge("feasible", "met") if b.feasible else badge("infeasible", "other"),
             esc("; ".join(b.excluded_by)) or "<em>none</em>", scenario(b.scenario) if b.scenario else "<em>none</em>"]
            for b in r.boundaries]  # fmt: skip
    out.append(heading(3, "Boundary points (addendum A09 §10.3)")
               + table(["condition", "point", "feasibility", "excluded by", "scenario"], rows))

    out.append(heading(2, "Scenarios"))
    definitions = {a.scenario: k for k, a in site.case.records("synthetic_artifact", core.SyntheticArtifact)
                   if a.media_type == "application/json"}  # fmt: skip
    for s in r.scenarios:
        pooled = table(["outcome", "pooled probability"], [[esc(o), esc(repr(v))] for o, v in s.resolver_outcomes.items()])
        defined = page.record("synthetic_artifact", definitions[s.id], "its definition") if s.id in definitions else ""
        out.append(f'<div id="{esc(anchor("scenario", s.id))}">' + heading(3, esc(f"Scenario {s.id}")) + dl([
            ("split", esc(s.split)), ("mandatory (a coverage witness or boundary scenario)", "yes" if s.mandatory else "no"),
            ("severity", esc(s.severity)), ("its abstract state", code(s.state_hash) + " " + defined),
            ("the IR's outcome (the evaluator, I8)", code(s.ir_outcome)),
            ("the IR's evaluation (content hash)", code(s.ir_evaluation)), ("the IR's transfers", transfers(s.ir_transfers)),
            ("rendering check", badge("passed", "passed") if s.rendering_passed else badge("failed", "failed")),
            ("the resolvers' pooled distribution", pooled), ("the prose's outcome", code(s.prose_outcome)),
            ("the prose's transfers", transfers(s.prose_transfers)), ("amount error", money(s.amount_error)),
            ("liquidity discrepancy", money(s.liquidity_discrepancy)),
            ("its packet", page.proposal(s.packet) + packet_documents(page, s.packet)),
        ]) + "</div>")  # fmt: skip

    out.append(heading(2, "Attributions (addendum A09 §10.1: each open until a counterexample or human gold checks it)"))
    rb = case.case.proposals.get(r.prose)
    rows = []
    for a in r.attributions:
        clauses = [page.proposal(r.prose, site=case, anchor=section_anchor(name), text=SECTION_TITLES[name])
                   + esc(f" (Ochiai {score!r})") for name, score in a.suspected_clauses] if rb is not None else [
            esc(f"{name} ({score!r})") for name, score in a.suspected_clauses]  # fmt: skip
        rows.append([scenario(a.scenario), esc(a.cause), esc(a.status), pre(a.evidence),
                     ul(clauses) if clauses else "<em>none</em>"])  # fmt: skip
    out.append(table(["scenario", "cause", "status", "evidence", "suspected clauses, ranked"], rows))

    out.append(heading(2, "The judgments of this run, by question"))
    by_q: dict[str, list[tuple[str, core.Judgment]]] = {}
    for k, j in site.case.records("judgment", core.Judgment):
        by_q.setdefault(j.question.id, []).append((k, j))
    for qid in sorted(by_q):
        js = sorted(by_q[qid], key=lambda kj: (config_order(kj[1].config), kj[1].state_hash, kj[0]))
        rows = [[page.record("judgment", k, k[:12]), judgment_config(j.config), code(j.state_hash[:12]),
                 distribution_line(j)] for k, j in js]  # fmt: skip
        out.append(heading(3, code(qid)) + table(["judgment", "configuration", "state", "distribution"], rows))
    return "".join(out)


@summarizes("fuzz_report")
def _report_line(page: Page, key: str, body: Any) -> str:
    return esc(f"{body['contract']['id']} v{body['contract']['version']}: {body['equivalence']}")


@renders_payload("scenario")
def scenario_packet(page: Page, p: core.Proposal) -> str:
    sp = core.ScenarioPacket.model_validate_json(core.canonical_json(dict(p.payload)))
    site = page.site
    reports = [(k, r) for k, r in site.case.records("fuzz_report", core.FuzzReport)
               if any(s.packet == p.id for s in r.scenarios)]  # fmt: skip
    used = "; ".join(page.record("fuzz_report", k, f"{k}, scenario {sp.scenario}", anchor=anchor("scenario", sp.scenario))
                     for k, _ in reports) or "<em>no fuzz report in this case scores this packet</em>"  # fmt: skip
    return dl([
        ("contract", page.record("contract", contract_record_key(sp.contract), f"{sp.contract.id} v{sp.contract.version}",
                                 site=real(site))),
        ("scenario", esc(sp.scenario)), ("its abstract state", code(sp.state_hash)),
        ("evidence as of", esc(sp.evidence_at.isoformat())), ("scored in", used),
        ("documents (synthetic, spec I4)", packet_documents(page, p.id)),
    ])  # fmt: skip


def fuzz_page(site: Site) -> None:
    """The fuzz reports of a case: its own (a synthetic case's) or those of the synthetic cases beside it."""
    path = site.page_path("fuzz")
    page = Page(site, path)
    rows = []
    for s, k, r in refs(site).family.reports:
        if s is not site and s.parent is not site:
            continue
        rows.append([page.record("fuzz_report", k, k, site=s), esc(f"{r.contract.id} v{r.contract.version}"),
                     esc(repr(r.fidelity)), esc(repr(r.clarity)), esc(repr(r.ambiguity)), esc(r.equivalence)])  # fmt: skip
    site.write(path, "Fuzz reports", para("Each fuzz run tests one contract version's prose against its IR (spec §8.6), "
                                          "in a synthetic case of its own (spec I4).")  # fmt: skip
               + table(["report", "contract", "fidelity", "clarity", "ambiguity", "equivalence"], rows))
