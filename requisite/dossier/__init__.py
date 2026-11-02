"""The static dossier (spec §13 WP16; §8.7): a case's committed state as plain HTML, rendered offline from its store
alone, byte-identical for identical stores.

`render(case_dir)` writes `<case_dir>/dossier/`: an index (the case, its records by kind with counts, the review queue,
its flows), a page for every committed record, reached from the index through its kind's list, the record's JSON as
committed beside it, a copy of every text a page links to (an extraction, a packet document, a market's record), and the
dossiers of the synthetic fuzz cases beside the case (`<case_dir>/fuzz/*`) under `fuzz/`, each page of which is marked
synthetic (spec I4). It reads nothing but the stores: no provider, no network, no clock.

A page for a kind of record is one renderer (site.RECORDS, site.PAYLOADS for a proposal's payload), registered by the
module that knows the kind; a kind with none is shown field by field, whole.
"""

from __future__ import annotations

from pathlib import Path

from requisite.dossier import contracts, fuzzing, proposals, records, wide
from requisite.dossier.html import badge, code, esc, heading, link, para, table, ul
from requisite.dossier.refs import link_all, refs
from requisite.dossier.site import FLOW_PAGES, RECORDS, SUMMARIES, Case, Files, Page, Site, json_copy, slug, write_css
from requisite.store import Refused

FLOW_WRITERS = {
    "f1": contracts.f1_page,
    "rulebooks": contracts.rulebooks_page,
    "fuzz": fuzzing.fuzz_page,
    "wide": wide.wide_page,
}


def render(case_dir: Path, case: str | None = None) -> Path:
    """Render the case at `case_dir` (named `case`, where given) and the synthetic fuzz cases beside it into
    `<case_dir>/dossier/`, replacing an earlier render; returns the index."""
    case_dir = Path(case_dir)
    main = Case(case_dir)
    opened = [main]
    try:
        if case is not None and main.name != case:
            raise Refused(f"{case_dir} holds case {main.name!r}, not {case!r}")
        files = Files()
        root = Site(main, "", files)
        fuzz_dir = case_dir / "fuzz"
        if fuzz_dir.is_dir():
            for d in sorted(p for p in fuzz_dir.iterdir() if (p / "requisite.db").exists()):
                child = Case(d)
                opened.append(child)
                root.children.append(Site(child, f"fuzz/{slug(d.name)}/", files, parent=root))
        link_all(root)
        write_css(files)
        for site in root.family:
            build(site)
        out = case_dir / "dossier"
        files.write(out)
        return out / "index.html"
    finally:
        for c in opened:
            c.close()


def build(site: Site) -> None:
    for kind, keys in site.case.keys.items():
        kind_list(site, kind, keys)
        for key in keys:
            record_page(site, kind, key)
    proposals.queue(site)
    kinds_page(site)
    for name, write in FLOW_WRITERS.items():
        if name in refs(site).flows:
            write(site)
    index(site)


def title_of(site: Site, kind: str, key: str) -> str:
    if kind == "proposal":
        body = site.case.body(kind, key)
        return f"Proposal {body['id']}"
    return f"{kind.replace('_', ' ').capitalize()} {key}"


def record_page(site: Site, kind: str, key: str) -> None:
    path = site.record_path(kind, key)
    page = Page(site, path)
    renderer = RECORDS.get(kind)
    body = site.case.body(kind, key)
    shown = renderer(page, key, body) if renderer else records.generic(page, key, body, kind)
    copy = json_copy(site, kind, key)
    footer = para(link(page.href(copy), "The record as committed (JSON)")
                  + esc(f": kind {kind}, key {key}. ") + link(page.href(site.kind_path(kind)), esc(f"Every {kind}")))
    site.write(path, title_of(site, kind, key), shown + footer, record=(kind, key))


def kind_list(site: Site, kind: str, keys: list[str]) -> None:
    path = site.kind_path(kind)
    page = Page(site, path)
    summary = SUMMARIES.get(kind)
    rows = [[link(page.href(site.record_path(kind, k)), code(k)), summary(page, k, site.case.body(kind, k)) if summary
             else ""] for k in keys]  # fmt: skip
    site.write(path, f"Every {kind} record", para(esc(f"In key order. {len(keys)} records.")) + table(["key", ""], rows))


def kinds_page(site: Site) -> None:
    path = site.page_path("kinds")
    page = Page(site, path)
    site.write(path, "Records by kind", kinds_table(page))


def kinds_table(page: Page) -> str:
    site = page.site
    rows = [[link(page.href(site.kind_path(k)), code(k)), esc(len(keys))] for k, keys in site.case.keys.items()]
    return table(["kind", "records"], rows)


def index(site: Site) -> None:
    path = site.index_path()
    page = Page(site, path)
    case = site.case
    out = []
    if case.synthetic:
        out.append(para(badge("synthetic", "synthetic") + esc(" This case is synthetic (spec I4): a fuzz run's scenarios, "
                                                              "their generated evidence packets and the judges' answers to them.")))  # fmt: skip
    flows = [(name, title) for name, title in FLOW_PAGES if name in refs(site).flows]
    if flows:
        out.append(heading(2, "What the case holds")
                   + ul(link(page.href(site.page_path(name)), esc(title)) for name, title in flows))
    waiting = [p for p in case.all_proposals() if proposals.awaiting(p)]
    out.append(heading(2, "Review queue") + para(
        link(page.href(site.page_path("queue")), "The review queue") + esc(
            f": {len(waiting)} proposals await a human gate." if waiting else ": no proposal awaits a human gate.")))
    if waiting:
        out.append(ul(page.proposal(p.id) + esc(f" ({p.kind}) awaits {proposals.HUMAN_GATES[p.kind][0]}")
                      for p in waiting))  # fmt: skip
    rejected = [p for p in case.all_proposals() if p.status == "rejected"]
    if rejected:
        out.append(heading(2, "Rejected proposals") + para("Shown as committed, with the gate that refused each.")
                   + ul(page.proposal(p.id) + esc(f" ({p.kind}): its {p.gates[-1].gate} gate did not pass")
                        for p in rejected))  # fmt: skip
    out.append(heading(2, "Records by kind") + kinds_table(page))
    if site.children:
        rows = [[link(page.href(c.index_path()), esc(c.case.name)) + page.mark(c),
                 esc(", ".join(f"{k} {len(v)}" for k, v in c.case.keys.items()))] for c in site.children]  # fmt: skip
        out.append(heading(2, "Synthetic cases beside it") + para(
            "Each fuzz run keeps its scenarios and generated evidence in a synthetic case of its own, never in this one "
            "(spec I4).") + table(["case", "records by kind"], rows))  # fmt: skip
    site.write(path, f"Case {case.name}", "".join(out))


__all__ = ["render"]
