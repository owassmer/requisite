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


def _print_item(item) -> None:  # noqa: ANN001
    files = len(item.artifacts) - 1
    print(f"{item.id}  {item.container}  {item.posted_at.isoformat()}  {item.submitter or '(no name published)'}  "
          f"{files} file(s)  {len(item.text)} chars")  # fmt: skip
    for u in item.unreadable:
        print(f"  unreadable: {u}")


def _progress(done: int, total: int) -> None:
    if done == total or done % 100 == 0:
        print(f"  {done} of {total} judgments", file=sys.stderr, flush=True)


def _wide_sample(args: argparse.Namespace) -> int:
    from requisite.discovery import wide

    store = _store(args)
    draw = wide.draw_sample(store)
    sizes = {k: len(v) for k, v in wide.phase1_strata(wide.the_frame(store)).items()}
    print(f"design {draw.design.id}  phase 1  seed {draw.seed}")
    for k, n in sizes.items():
        drawn = [d for d in draw.items if d.stratum == k]
        print(f"  stratum {k}: {len(drawn)} of {n} items, each with probability {drawn[0].inclusion_prob}")
    print(f"{len(draw.items)} items drawn")
    return 0


def _wide_read(args: argparse.Namespace) -> int:
    from requisite.discovery import wide

    items = wide.read_sample(_store(args), workers=args.workers)
    for item in items:
        _print_item(item)
    print(f"{len(items)} items read")
    return 0


def _show_labelled(done) -> None:  # noqa: ANN001
    refused = sum(1 for j in done.judgments if j.distribution is None)
    print(f"{len(done.judgments)} judgments, {refused} refused")
    for item, why in done.unreadable.items():
        print(f"  no judge asked about {item}: {why}")


def _wide_label(args: argparse.Namespace) -> int:
    from requisite.discovery import wide

    _show_labelled(wide.label_phase1(_store(args), workers=args.workers, on_done=_progress))
    return 0


def _wide_stability(args: argparse.Namespace) -> int:
    from requisite.discovery import wide

    stability, unreadable = wide.test_stability(_store(args), workers=args.workers, on_done=_progress)
    for t, st in stability.items():
        print(f"{t}: {st.items} items, {st.stable_share:.0%} stable (S_arg >= 0.9 and S_prob >= 0.85), mean S_arg "
              f"{st.mean_s_arg:.3f}, mean S_prob {st.mean_s_prob:.3f}, {st.refusals} refusals")  # fmt: skip
    for item, why in unreadable.items():
        print(f"  no judge asked about {item}: {why}")
    return 0


def _wide_gold_draw(args: argparse.Namespace) -> int:
    from requisite.discovery import wide

    store = _store(args)
    report = wide.gold_coverage(store)
    print(f"the gold design under labels simulated from the judge's probabilities ({report.draws} draws, seed "
          f"{report.seed}; a statement about the design, not the frame):")  # fmt: skip
    for c in report.intervals:
        print(f"  {c.estimator}: coverage {c.coverage:.3f} (MC se {c.monte_carlo_se:.3f}), missed below "
              f"{c.missed_below:.3f}, above {c.missed_above:.3f}, mean width {c.mean_width:.3f}")  # fmt: skip
    for g in report.gold:
        print(f"  gold in ({g.stratum}, bin {g.bin}): expects {g.expected_gold:.1f} items, {g.expected_positives:.2f} "
              f"positive, {g.expected_negatives:.2f} negative")  # fmt: skip
    gold = wide.draw_gold(store)
    print(f"design {gold.design.id}  phase 2 (gold)  seed {gold.seed}")
    for stratum in dict.fromkeys(d.stratum for d in gold.items):
        drawn = [d for d in gold.items if d.stratum == stratum]
        print(f"  {stratum}: {len(drawn)} drawn, each with probability {drawn[0].inclusion_prob}")
    print(f"{len(gold.items)} items drawn for gold labels")
    return 0


def _paraphrases(args: argparse.Namespace) -> int:
    from requisite.discovery import wide

    _show_proposal(wide.draft_paraphrases(_store(args), args.template))
    return 0


def _gold(args: argparse.Namespace) -> int:
    from requisite.discovery import gold, wide

    store = _store(args)
    gold.label_session(store, wide.drawn(store, 2), wide.approved_objective(store), wide.measures(store),
                       labeler=args.labeler)  # fmt: skip
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
    steps = (
        ("sample", "draw the phase-1 sample of the frozen frame, by the design's rule", _wide_sample, False),
        ("read", "read the phase-1 sample's items whole and freeze them", _wide_read, True),
        ("label", "the phase-1 judge's labels on every phase-1 item", _wide_label, True),
        ("stability", "every judge configuration on the stability subsample (spec §2.1)", _wide_stability, True),
        ("gold-draw", "draw the gold sample from phase 1, by the judge's probabilities", _wide_gold_draw, False),
    )
    for name, text, run, parallel in steps:
        w = wsub.add_parser(name, help=text)
        case_args(w)
        if parallel:
            w.add_argument("--workers", type=int, default=8, help="requests in flight at once (default 8)")
        w.set_defaults(run=run)
    ph = sub.add_parser("paraphrases", help="an agent writes a library template's frozen paraphrase set (spec §2.1)")
    case_args(ph)
    ph.add_argument("--template", required=True, help="the library template's id")
    ph.set_defaults(run=_paraphrases)
    g = sub.add_parser("gold", help="label the gold sample blind, one item at a time (stop and resume at will)")
    case_args(g)
    g.add_argument("--labeler", required=True, help="who is labelling")
    g.set_defaults(run=_gold)
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
