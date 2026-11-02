"""The static dossier (spec §13 WP16, §8.7; requisite/dossier): a case's committed state as plain HTML, rendered from its
store alone. Over cases built by the product's own functions (tests/dossier/cases.py): identical stores render
identical bytes, in a replay-only process with the network refused as in one with it; every committed record is reached
from the index; every clause's cited node links to an anchor that exists, and every node lists the sections that state
it, the questions asked of it and, through the fuzz report, the packets that exercised it; a rejected proposal is shown
with its gates; a synthetic case is marked on every page."""

from __future__ import annotations

import html
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from cases import construction, wide_case
from crawl import Parsed, crawl, parse, resolve, tree
from test_run import IR as MERGER

from requisite import core
from requisite.dossier import render
from requisite.dossier.html import anchor
from requisite.dossier.proposals import HUMAN_GATES
from requisite.fuzz.run import asked_node
from requisite.store import Refused, open_case

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """Both cases, built and rendered once in a process that may record (the doubles and the fake API answer)."""
    tmp = tmp_path_factory.mktemp("dossier")
    with pytest.MonkeyPatch.context() as mp:
        made = SimpleNamespace(construction=construction(mp, tmp / "c"), wide=wide_case(mp, tmp / "w"))
        render(made.construction, "construction")
        render(made.wide, "wide")
    return made


def stores(case_dir: Path) -> dict[str, set[tuple[str, str]]]:
    """Every (kind, key) committed in the case and in each synthetic case beside it, by case name."""
    out = {}
    for d in [case_dir, *sorted(p for p in (case_dir / "fuzz").glob("*") if (p / "requisite.db").exists())]:
        s = open_case(d)
        out[s.case] = {(k, key) for k in s.kinds() for key in s.keys(k)}
        s.close()
    return out


def page_of(pages: dict[Path, Parsed], case: str, kind: str, key: str) -> Path:
    (found,) = [p for p, x in pages.items() if x.case == case and x.record == (kind, key)]
    return found


OFFLINE = """
import socket, sys
def refused(*args, **kwargs):
    raise OSError("the network is refused: a dossier renders offline")
socket.socket.connect = refused
socket.socket.connect_ex = refused
socket.create_connection = refused
socket.getaddrinfo = refused
from requisite.cli import main
raise SystemExit(main(["dossier", "--case-dir", sys.argv[1], "--case", sys.argv[2]]))
"""


@pytest.mark.covers("WP16.offline")
def test_identical_stores_render_identical_bytes_offline_in_a_replay_only_process(built, tmp_path):
    for case_dir, name in ((built.construction, "construction"), (built.wide, "wide")):
        first = tree(case_dir / "dossier")
        assert first and "index.html" in first
        # Again in this process, which is replay-only (tests/conftest.py): the same bytes.
        render(case_dir, name)
        assert tree(case_dir / "dossier") == first
        # In a replay-only process whose every connection is refused: the same bytes.
        env = {**os.environ, "REQUISITE_REPLAY_ONLY": "1"}
        subprocess.run([sys.executable, "-c", OFFLINE, str(case_dir), name], env=env, cwd=ROOT, check=True,
                       capture_output=True)  # fmt: skip
        assert tree(case_dir / "dossier") == first
        # The same store elsewhere: the same bytes, so no page holds a path of the machine that rendered it.
        moved = tmp_path / f"moved-{name}"
        shutil.copytree(case_dir, moved, ignore=shutil.ignore_patterns("dossier"))
        render(moved, name)
        assert tree(moved / "dossier") == first
        # Plain HTML and CSS: no script, nothing loaded but the dossier's own stylesheet, every link inside it.
        for path, data in first.items():
            if not path.endswith(".html"):
                continue
            page = parse(case_dir / "dossier" / path)
            assert page.scripts == 0, path
            assert all(s.endswith("style.css") and "://" not in s for s in page.sources), (path, page.sources)
            assert all("://" not in h and not h.startswith("/") for h in page.hrefs), (path, page.hrefs)
            assert str(case_dir).encode() not in data and str(tmp_path).encode() not in data, path


def test_every_committed_record_is_reached_from_the_index_and_every_link_lands(built):
    for case_dir in (built.construction, built.wide):
        root = case_dir / "dossier"
        pages = crawl(root)
        shown = {(p.case, *p.record) for p in pages.values() if p.record is not None}
        committed = {(case, kind, key) for case, records in stores(case_dir).items() for kind, key in records}
        assert shown == committed, (sorted(committed - shown)[:5], sorted(shown - committed)[:5])
        for page, parsed in pages.items():
            for href in parsed.hrefs:
                target, fragment = resolve(root, page, href)
                assert target.is_file(), (page, href)
                if fragment is not None:
                    assert fragment in (parsed.ids if target == page else parse(target).ids), (page, href)


def sections(page: Path) -> dict[str, str]:
    """Each rulebook section's HTML on a clause's page, by its section name."""
    text = page.read_text()
    return {html.unescape(m.group(1)).removeprefix("section-"): m.group(2)
            for m in re.finditer(r'<section class="prose" id="([^"]+)">(.*?)</section>', text, re.S)}  # fmt: skip


def links(fragment_html: str) -> list[str]:
    return [html.unescape(h) for h in re.findall(r'href="([^"]+)"', fragment_html)]


def stated_div(page_html: str, element_id: str) -> str:
    """The 'Stated by' line of an IR node or observable: the first stated line after the element opens."""
    m = re.search(rf'id="{re.escape(html.escape(element_id))}">.*?<div class="stated">(.*?)</div>', page_html, re.S)
    assert m, element_id
    return m.group(1)


@pytest.mark.covers("WP16.links")
def test_each_clause_links_to_its_ir_nodes_and_each_node_to_its_clauses_questions_and_evidence(built):
    root = built.construction / "dossier"
    pages = crawl(root)
    case = open_case(built.construction)
    clauses = [p for p in (case.proposal(k.split("|")[0]) for k in case.keys("proposal")) if p and p.kind == "clause"]
    assert {p.status for p in clauses} == {"active", "rejected"}
    stated: dict[tuple[str, int], dict[str, set[tuple[Path, str]]]] = {}
    for p in clauses:
        rb = core.Rulebook.model_validate_json(core.canonical_json(dict(p.payload)))
        ir = case.get("contract", f"{rb.contract.id}|{rb.contract.version}", core.ContractIR)
        assert ir is not None
        nodes = {n.id for r in (ir.trigger, ir.payout.quantity) if r is not None for n in core.walk(r)}
        observables = {o.id for o in ir.observables}
        clause_page = page_of(pages, case.case, "proposal", f"{p.id}|submitted")
        shown = sections(clause_page)
        assert list(shown) == [s.name for s in rb.sections]  # spec §8.5's order, each section anchored
        for s in rb.sections:
            targets = {resolve(root, clause_page, h) for h in links(shown[s.name])}
            for ident in s.ir_nodes:
                want = ([anchor("node", ident)] if ident in nodes else []) + (
                    [anchor("observable", ident)] if ident in observables else [])
                if not want:  # the rejected rulebook's "closing": no link, and it says why
                    assert "neither a node nor an observable" in shown[s.name]
                for a in want:
                    (target,) = [t for t, f in targets if f == a]
                    assert a in parse(target).ids
                    assert parse(target).record == ("contract", f"{ir.id}|{ir.version}")
                    stated.setdefault((ir.id, ir.version), {}).setdefault(a, set()).add(
                        (clause_page, anchor("section", s.name)))
    # Every node and observable of every contract lists exactly the sections that state it.
    for (cid, version), by_anchor in stated.items():
        ir = case.get("contract", f"{cid}|{version}", core.ContractIR)
        contract_page = page_of(pages, case.case, "contract", f"{cid}|{version}")
        text = contract_page.read_text()
        everything = [anchor("node", n.id) for r in (ir.trigger, ir.payout.quantity) if r is not None
                      for n in core.walk(r)] + [anchor("observable", o.id) for o in ir.observables]  # fmt: skip
        for a in everything:
            div = stated_div(text, a)
            found = {resolve(root, contract_page, h) for h in links(div)}
            assert found == by_anchor.get(a, set()), (a, found)
            if not by_anchor.get(a):
                assert "Stated by no rulebook section" in div
    # Questions: each condition the fuzz run asked of its packets lists the judgments that asked it.
    merger = MERGER
    contract_page = page_of(pages, case.case, "contract", f"{merger.id}|{merger.version}")
    text = contract_page.read_text()
    fuzz_dir = next(p for p in (built.construction / "fuzz").glob("*") if (p / "requisite.db").exists())
    synthetic = open_case(fuzz_dir)
    asked: dict[str, set[str]] = {}
    for k in synthetic.keys("judgment"):
        j = synthetic.get("judgment", k, core.Judgment)
        node = asked_node(merger.ref, j.question.id)
        if node is not None:
            asked.setdefault(node, set()).add(k)
    assert asked, "the fuzz run asked its rendering-check questions of the merger's conditions"
    for node, keys in asked.items():
        m = re.search(rf'id="{re.escape(anchor("node", node))}">(.*?)(?:<ul class="tree">|</li>)', text, re.S)
        assert m and "Asked of fuzz packets" in m.group(1), node
        reached = {parse(resolve(root, contract_page, h)[0]).record for h in links(m.group(1)) if "judgment" in h}
        assert reached == {("judgment", k) for k in keys}, node
    # Evidence: the active rulebook links its fuzz report, whose every scenario shows its packet's documents.
    (active,) = [p for p in clauses if p.status == "active"]
    clause_page = page_of(pages, case.case, "proposal", f"{active.id}|submitted")
    (report_key,) = synthetic.keys("fuzz_report")
    report_page = page_of(pages, synthetic.case, "fuzz_report", report_key)
    assert report_page in {resolve(root, clause_page, h)[0] for h in parse(clause_page).hrefs}
    report = synthetic.get("fuzz_report", report_key, core.FuzzReport)
    shown = report_page.read_text()
    for s in report.scenarios:
        packet = synthetic.proposal(s.packet)
        assert packet is not None and anchor("scenario", s.id) in parse(report_page).ids
        for doc in packet.payload["documents"]:
            assert html.escape(synthetic.content(doc).decode(), quote=True) in shown
    # The attributions' suspected clauses link to the rulebook's sections.
    suspected = {(a.scenario, name) for a in report.attributions for name, _ in a.suspected_clauses}
    targets = {resolve(root, report_page, h) for h in parse(report_page).hrefs}
    for _, name in suspected:
        assert (clause_page, anchor("section", name)) in targets
    synthetic.close()
    case.close()


def test_a_rejected_proposal_is_shown_marked_with_every_gate_and_listed_from_the_index(built):
    root = built.construction / "dossier"
    pages = crawl(root)
    case = open_case(built.construction)
    rejected = [p for p in (case.proposal(k.split("|")[0]) for k in case.keys("proposal")) if p and p.status == "rejected"]
    assert {p.kind for p in rejected} == {"clause", "contract"}
    index = (root / "index.html").read_text()
    for p in rejected:
        page = page_of(pages, case.case, "proposal", f"{p.id}|submitted")
        text = page.read_text()
        assert '<span class="badge rejected">rejected</span>' in text
        assert [g.passed for g in p.gates][-1] is False
        for g in p.gates:
            assert html.escape(g.detail, quote=True) in text, g.gate
            assert f"<td>{g.gate}</td>" in text
        assert html.escape(p.id) in index and "Rejected proposals" in index
    case.close()


def test_a_synthetic_case_is_marked_synthetic_on_every_page_and_wherever_it_is_linked(built):
    root = built.construction / "dossier"
    fuzz_pages = [p for p in root.joinpath("fuzz").rglob("*.html")]
    assert fuzz_pages
    for page in fuzz_pages:
        parsed = parse(page)
        assert "synthetic" in parsed.body_classes, page
        assert '<span class="badge synthetic">synthetic</span>' in page.read_text().split("</header>")[0], page
    linked = 0
    for page in root.rglob("*.html"):
        if page.is_relative_to(root / "fuzz"):
            continue
        text = page.read_text()
        assert "synthetic" not in parse(page).body_classes
        for m in re.finditer(r'<a href="([^"]+)"[^>]*>.*?</a>', text):
            if resolve(root, page, html.unescape(m.group(1)))[0].is_relative_to(root / "fuzz"):
                assert text[m.end():].startswith('<span class="badge synthetic">synthetic</span>'), (page, m.group(0))
                linked += 1
    assert linked  # the index, the rulebook and the contract link into the fuzz case


def test_f1s_readings_show_their_classification_bindings_ir_and_each_markets_check(built):
    root = built.construction / "dossier"
    pages = crawl(root)
    case = open_case(built.construction)
    readings = [p for p in (case.proposal(k.split("|")[0]) for k in case.keys("proposal")) if p and p.kind == "contract"]
    for p in readings:
        family = core.ContractFamily.model_validate_json(core.canonical_json(dict(p.payload)))
        text = page_of(pages, case.case, "proposal", f"{p.id}|submitted").read_text()
        assert html.escape(family.contract.support) in text and html.escape(family.contract.equivalence) in text
        for b in family.bindings:
            assert f"<code>{html.escape(b.field)}</code>" in text
        assert family.representative is not None
        assert anchor("node", family.representative.trigger.id) in parse(
            page_of(pages, case.case, "proposal", f"{p.id}|submitted")).ids
        (tests,) = [g for g in p.gates if g.gate == "tests"]
        for m in family.markets:
            (line,) = [x for x in tests.detail.splitlines() if x.startswith(f"  {m.id}: ")]
            row = re.search(rf"<tr><td><code>{re.escape(m.id)}</code></td>(.*?)</tr>", text, re.S)
            assert row and html.escape(line.split(": ", 1)[1], quote=True) in row.group(1), m.id
    case.close()


def test_the_wide_page_shows_the_objective_frame_draws_judgments_and_gold(built):
    root = built.wide / "dossier"
    text = (root / "wide.html").read_text()
    case = open_case(built.wide)
    (approval,) = [case.get("approval", k, core.Approval) for k in case.keys("approval")
                   if case.get("approval", k, core.Approval).subject.id.startswith("objective-")]  # fmt: skip
    assert approval.gate == "G1" and "<td>G1</td><td>approved</td><td>tester</td>" in text
    (frame_key,) = case.keys("frame")
    frame = case.get("frame", frame_key, core.Frame)
    for d in frame.dockets:
        assert html.escape(d.reason, quote=True) in text
    for k in case.keys("sample_draw"):
        draw = case.get("sample_draw", k, core.SampleDraw)
        for i in draw.items:
            assert f"<td>{html.escape(i.stratum)}</td>" in text
            p = i.inclusion_prob
            assert f"{p.numerator}/{p.denominator}" in text
    judgments = [case.get("judgment", k, core.Judgment) for k in case.keys("judgment")]
    configs = {(j.config.provider, j.config.paraphrase_id, j.config.render_id) for j in judgments}
    assert len(configs) > 1  # the phase-1 configuration and the stability configurations
    for j in judgments:
        assert f"<code>{html.escape(j.question.id)}</code>" in text
        c = j.config
        assert html.escape(f"{c.provider} {c.model} (training cutoff {c.model_cutoff if isinstance(c.model_cutoff, str) else c.model_cutoff.isoformat()})") in text
    labels = [case.get("gold_label", k, core.GoldLabel) for k in case.keys("gold_label")]
    assert labels and all(f"<td>{html.escape(g.label or '')}</td>" in text for g in labels)
    case.close()


def test_the_review_queue_lists_what_awaits_a_human_gate_with_what_it_asks(built):
    text = (built.construction / "dossier" / "queue.html").read_text()
    case = open_case(built.construction)
    (waiting,) = [p for p in (case.proposal(k.split("|")[0]) for k in case.keys("proposal")) if p and p.status == "proposed"]
    assert waiting.kind == "objective" and html.escape(waiting.id) in text and "<td>G1</td>" in text
    # The gate each kind awaits is the definition's: exactly the kinds whose required gates end at a human.
    assert set(HUMAN_GATES) == {k for k, gates in core.REQUIRED_GATES.items() if "human" in gates}
    case.close()


def test_a_render_never_replaces_a_directory_that_is_not_a_dossier_or_renders_another_case(built, tmp_path):
    copy = tmp_path / "case"
    shutil.copytree(built.wide, copy, ignore=shutil.ignore_patterns("dossier"))
    (copy / "dossier").mkdir()
    (copy / "dossier" / "notes.txt").write_text("mine")
    with pytest.raises(Refused, match="not a rendered dossier"):
        render(copy, "wide")
    assert (copy / "dossier" / "notes.txt").read_text() == "mine"
    with pytest.raises(Refused, match="holds case 'wide', not 'other'"):
        render(copy, "other")
