"""The command line: the product as a user runs it (spec §13 WP16: a CLI plus a static HTML dossier per case)."""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

from requisite import registry


def _registry(args: argparse.Namespace) -> int:
    reg, cov = registry.items(), registry.coverage()
    bad = registry.problems(reg, cov)
    print(registry.board(reg, cov))
    for p in bad:
        print(f"PROBLEM {p}")
    return 1 if bad else 0


def _store(args: argparse.Namespace):  # noqa: ANN202
    from requisite.store import Store

    return Store(Path(args.case_dir), case=args.case)


def _show_proposal(p) -> None:  # noqa: ANN001
    import json

    print(f"proposal {p.id}  kind {p.kind}  status {p.status}  author {p.author}")
    for g in p.gates:
        print(f"  gate {g.gate}: {'passed' if g.passed else 'not passed'}  {g.detail}")
    print(json.dumps(dict(p.payload), indent=2, default=str))
    print(f"rationale: {p.rationale}")


def _wide_objective(args: argparse.Namespace) -> int:
    from requisite.discovery import wide

    today = date.fromisoformat(args.today) if args.today else date.today()
    _show_proposal(wide.draft_objective(_store(args), args.brief, today))
    return 0


def _wide_frame(args: argparse.Namespace) -> int:
    from requisite.discovery import wide

    _show_proposal(wide.draft_frame(_store(args), datetime.fromisoformat(args.as_of)))
    return 0


def _evidence_frame(args: argparse.Namespace) -> int:
    import json

    from requisite.core import FrameCriteria
    from requisite.evidence.frame import summary, take_frame
    from requisite.evidence.regulations_gov import RegulationsGov

    criteria = FrameCriteria.model_validate_json(Path(args.criteria).read_text())
    client = RegulationsGov()
    try:
        frame = take_frame(criteria, client, _store(args))
    finally:
        client.close()
    print(json.dumps(summary(frame), indent=2))
    return 0


def _sample_ids(path: Path) -> list[str]:
    """A sample file: a JSON array of comment ids, or one id per line."""
    import json

    text = path.read_text()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return [line.strip() for line in text.splitlines() if line.strip()]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise SystemExit(f"{path}: a sample is a JSON array of comment ids, or one id per line")
    return value


def _evidence_read(args: argparse.Namespace) -> int:
    from requisite.evidence.frame import entries
    from requisite.evidence.items import read_items
    from requisite.evidence.regulations_gov import RegulationsGov

    store = _store(args)
    wanted = entries(store, _sample_ids(Path(args.sample)))
    client = RegulationsGov()
    try:
        items = read_items(client, store, wanted, workers=args.workers)
    finally:
        client.close()
    for item in items:
        files = len(item.artifacts) - 1
        print(f"{item.id}  {item.container}  {item.posted_at.isoformat()}  {item.submitter or '(no name published)'}  "
              f"{files} file(s)  {len(item.text)} chars")  # fmt: skip
        for u in item.unreadable:
            print(f"  unreadable: {u}")
    print(f"{len(items)} items read")
    return 0


def _instant(value: str) -> datetime:
    """An instant given on the command line: ISO 8601 with its zone (spec §14: explicit time zones)."""
    at = datetime.fromisoformat(value)
    if at.utcoffset() is None:
        raise argparse.ArgumentTypeError(f"{value!r} names no time zone; give the instant with its offset or Z")
    return at


def _f1_draw(args: argparse.Namespace) -> int:
    from requisite.market import f1
    from requisite.market.kalshi import Kalshi

    client = Kalshi(snapshot=args.as_of)
    try:
        print(f1.render(f1.draw(client, args.as_of)))
    finally:
        client.close()
    return 0


def _f1_retrieve(args: argparse.Namespace) -> int:
    from requisite.market import f1
    from requisite.market.kalshi import Kalshi

    store = _store(args)
    client = Kalshi(snapshot=args.as_of)
    try:
        d = f1.draw(client, args.as_of)
        harvest = f1.retrieve(client, store, d)
    finally:
        client.close()
    print(f1.render(d))
    print()
    print(f1.render_harvest(harvest, store))
    return 0


def _approve(args: argparse.Namespace) -> int:
    from requisite.discovery import wide

    p = wide.approve(_store(args), args.proposal, gate=args.gate, approver=args.approver, decision=args.decision,
                     reason=args.reason, scope=args.scope)  # fmt: skip
    _show_proposal(p)
    return 0


def _show(args: argparse.Namespace) -> int:
    p = _store(args).proposal(args.proposal)
    if p is None:
        print(f"no proposal {args.proposal}")
        return 1
    _show_proposal(p)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="requisite",
        description="A runtime for discovery and contract construction. Analysis, not advice; prices are indicative unless quoted.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("registry", help="where the sources' enumerated items stand: covered, open, replaced, by tier")
    r.set_defaults(run=_registry)

    def case_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--case-dir", required=True, help="the case's directory (its store)")
        sp.add_argument("--case", required=True, help="the case's name")

    wide = sub.add_parser("wide", help="wide discovery (spec §7.2): objective, frame, sample, labels, gold, leads")
    wsub = wide.add_subparsers(dest="step", required=True)
    w = wsub.add_parser("objective", help="an agent drafts the run's objective from a brief; it awaits G1")
    case_args(w)
    w.add_argument("--brief", required=True)
    w.add_argument("--today", help="the run's date (default: today)")
    w.set_defaults(run=_wide_objective)
    w = wsub.add_parser("frame", help="an agent derives the frame criteria from the approved objective")
    case_args(w)
    w.add_argument("--as-of", required=True, help="the instant the frame is taken at, with its zone")
    w.set_defaults(run=_wide_frame)
    ev = sub.add_parser("evidence", help="the public record retrieved and frozen (spec §7.2 step 1): a frame, its items")
    esub = ev.add_subparsers(dest="step", required=True)
    e = esub.add_parser("frame", help="run frame criteria against regulations.gov and freeze the frame they return")
    case_args(e)
    e.add_argument("--criteria", required=True, help="a FrameCriteria as JSON")
    e.set_defaults(run=_evidence_frame)
    e = esub.add_parser("read", help="read a sample of a frame's comments whole into items")
    case_args(e)
    e.add_argument("--sample", required=True, help="comment ids of this case's frame: a JSON array, or one per line")
    e.add_argument("--workers", type=int, default=8, help="requests in flight at once (default 8)")
    e.set_defaults(run=_evidence_read)
    mk = sub.add_parser("market", help="the market lens, read-only (spec §10): Kalshi's filed record, F1's contracts")
    msub = mk.add_subparsers(dest="step", required=True)
    m = msub.add_parser("f1-draw", help="draw F1's filed contracts at an instant and print the rule's evidence")
    m.add_argument("--as-of", required=True, type=_instant, help="the instant the draw is pinned at, with its zone")
    m.set_defaults(run=_f1_draw)
    m = msub.add_parser("f1-retrieve", help="draw, then commit the drawn series' filings, rules and settled outcomes")
    case_args(m)
    m.add_argument("--as-of", required=True, type=_instant, help="the instant the draw is pinned at, with its zone")
    m.set_defaults(run=_f1_retrieve)
    a = sub.add_parser("approve", help="a human decides at a gate (G1-G4) on one proposal, as it stands")
    case_args(a)
    a.add_argument("proposal")
    a.add_argument("--gate", required=True, choices=["G1", "G2", "G3", "G4"])
    a.add_argument("--approver", required=True)
    a.add_argument("--decision", default="approved", choices=["approved", "approved_with_edits", "rejected"])
    a.add_argument("--reason", required=True)
    a.add_argument("--scope", required=True)
    a.set_defaults(run=_approve)
    sh = sub.add_parser("show", help="a proposal as it stands: payload, gates, rationale")
    case_args(sh)
    sh.add_argument("proposal")
    sh.set_defaults(run=_show)
    args = parser.parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    sys.exit(main())
