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
