"""The cross references a dossier's pages show, read once from the committed records of a case and the synthetic cases
beside it: who authored a proposal, which approvals decided it, which extractions an artifact has, which rulebook
sections state an IR node, which questions a fuzz run asked of it, which fuzz reports test a contract or a rulebook.
Each is read from what the records name; nothing is recomputed."""

from __future__ import annotations

from dataclasses import dataclass, field

from requisite import core
from requisite.dossier.site import Site
from requisite.fuzz.run import asked_node

ContractKey = tuple[str, int]


@dataclass
class FamilyRefs:
    """Across a case and its synthetic fuzz cases."""

    stated: dict[ContractKey, dict[str, list[tuple[Site, str, str]]]] = field(default_factory=dict)
    asked: dict[ContractKey, dict[str, list[tuple[Site, str, core.Judgment]]]] = field(default_factory=dict)
    reports: list[tuple[Site, str, core.FuzzReport]] = field(default_factory=list)


@dataclass
class SiteRefs:
    family: FamilyRefs
    flows: set[str]
    as_of: list[tuple[str, list[str]]]
    agent_runs: dict[str, list[tuple[str, core.AgentRun]]]
    approvals: dict[str, list[tuple[str, core.Approval]]]
    extractions: dict[str, list[str]]
    items: dict[str, list[str]]
    blobs: set[str]  # observed artifacts whose content a page shows, so the dossier carries a copy
    families: dict[ContractKey, list[str]]  # a contract IR's (id, version) -> the contract proposals whose family holds it


def contract_key(ref: core.Ref) -> ContractKey:
    return (ref.id, ref.version)


def site_refs(site: Site, family: FamilyRefs) -> SiteRefs:
    case = site.case
    runs: dict[str, list[tuple[str, core.AgentRun]]] = {}
    for key, run in case.records("agent_run", core.AgentRun):
        runs.setdefault(run.id, []).append((key, run))
    approvals: dict[str, list[tuple[str, core.Approval]]] = {}
    for key, a in case.records("approval", core.Approval):
        approvals.setdefault(a.subject.id, []).append((key, a))
    extractions: dict[str, list[str]] = {}
    for key in case.keys.get("extraction", []):
        extractions.setdefault(key.split("|", 1)[0], []).append(key)
    items: dict[str, list[str]] = {}
    for key, body in ((k, case.body("item", k)) for k in case.keys.get("item", [])):
        items.setdefault(str(body["id"]), []).append(key)
    blobs: set[str] = set()
    for _, filed in case.records("filed_series", core.FiledSeries):
        blobs.add(filed.record)
        blobs.update(filed.markets.values())
    families: dict[ContractKey, list[str]] = {}
    for p in case.proposals_of("contract"):
        for m in p.payload.get("markets", ()):
            families.setdefault((str(m["id"]), int(m["version"])), []).append(p.id)
    kinds = set(case.keys)
    pkinds = {p.kind for p in case.proposals.values()}
    flows = set()
    if "filed_series" in kinds:
        flows.add("f1")
    if "clause" in pkinds:
        flows.add("rulebooks")
    if "fuzz_report" in kinds or any("fuzz_report" in c.case.keys for c in site.children):
        flows.add("fuzz")
    if {"objective", "frame"} & pkinds or {"frame", "sample_draw", "gold_label"} & kinds:
        flows.add("wide")
    return SiteRefs(family, flows, case.as_of(), runs, approvals, extractions, items, blobs, families)


def family_refs(sites: list[Site]) -> FamilyRefs:
    out = FamilyRefs()
    for site in sites:
        for p in site.case.proposals_of("clause"):
            rb = core.Rulebook.model_validate_json(core.canonical_json(dict(p.payload)))
            nodes = out.stated.setdefault(contract_key(rb.contract), {})
            for section in rb.sections:
                for node in section.ir_nodes:
                    nodes.setdefault(node, []).append((site, p.id, section.name))
        for key, report in site.case.records("fuzz_report", core.FuzzReport):
            out.reports.append((site, key, report))
    contracts = {contract_key(r.contract): r.contract for _, _, r in out.reports}
    for site in sites:
        if not site.case.synthetic:
            continue
        for key, j in site.case.records("judgment", core.Judgment):
            for ck, ref in contracts.items():
                asked = asked_node(ref, j.question.id)
                if asked is not None:
                    out.asked.setdefault(ck, {}).setdefault(asked, []).append((site, key, j))
    return out


def link_all(root: Site) -> FamilyRefs:
    """Every site's references, and the family's, set on each site."""
    family = family_refs(root.family)
    for site in root.family:
        site.refs = site_refs(site, family)
    return family


def reports_of(family: FamilyRefs, *, prose: str | None = None, contract: core.Ref | None = None
               ) -> list[tuple[Site, str, core.FuzzReport]]:  # fmt: skip
    return [(s, k, r) for s, k, r in family.reports
            if (prose is None or r.prose == prose) and (contract is None or r.contract == contract)]  # fmt: skip


def refs(site: Site) -> SiteRefs:
    assert isinstance(site.refs, SiteRefs)  # set by link_all before any page is written
    return site.refs


__all__ = ["FamilyRefs", "SiteRefs", "link_all", "refs", "reports_of"]
