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


def _store(args: argparse.Namespace, *, begins: bool = False):  # noqa: ANN202
    """The case the command works on. A command that begins a case (`wide objective`, `market f1-retrieve`, `contract
    read`) creates it where none is committed; any other opens the case committed at --case-dir, refused where none is or it is another case,
    so a mistyped path is never a new empty case."""
    from requisite.store import Refused, Store, open_case

    if begins:
        return Store(Path(args.case_dir), case=args.case)
    store = open_case(Path(args.case_dir))
    if store.case != args.case:
        raise Refused(f"{args.case_dir} holds case {store.case}, not {args.case}")
    return store


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
    _show_proposal(wide.draft_objective(_store(args, begins=True), args.brief, today))
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
    from requisite.evidence.mirrulations import Mirrulations

    items = wide.read_sample(_store(args), workers=args.workers, mirror=Mirrulations)
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


def _wide_families(args: argparse.Namespace) -> int:
    from requisite.discovery import results

    _show_proposal(results.draft_families(_store(args)))
    return 0


def _wide_estimate(args: argparse.Namespace) -> int:
    from requisite.discovery import results

    est = results.estimate(_store(args))
    print(f"the share of readable items in the population's voice that describe exposure to a pending measure, "
          f"{est.level:.0%} intervals; {est.phase1_items} phase-1 items ({est.unreadable_items} with no readable text, "
          f"outside the scope), {est.gold_items} gold items, {est.unresolved_gold_labels} unresolved gold labels")  # fmt: skip
    for sc in est.scopes:
        rd = f"[{sc.read[0]:.3f}, {sc.read[1]:.3f}]" if sc.read else f"no interval: {sc.reason}"
        lo, hi = sc.lowest.estimate, sc.highest.estimate
        pts = f"{lo:.3f}" if lo == hi and lo is not None else f"{lo} to {hi}"
        print(f"  {sc.scope}: {rd}  (estimate {pts}; Fieller within [0, 1]: {sc.lowest.fieller} to {sc.highest.fieller})")
    return 0


def _wide_leads(args: argparse.Namespace) -> int:
    from requisite.discovery import results

    leads = results.find_leads(_store(args), workers=args.workers)
    for x in leads.leads:
        usd = [f"{float(i.annual):,.0f} {i.currency}/year" for i in x.impact if i.annual is not None]
        print(f"{x.rank:>3}. tier {x.tier}  {x.submitter or '(no name published)'}  {x.item}  {x.family}  "
              f"{x.basis} p={x.probability:.2f}  {', '.join(f'{k}={v}' for k, v in sorted(x.answers.items()))}"
              f"{'  ' + '; '.join(usd) if usd else ''}")  # fmt: skip
    print(f"{len(leads.leads)} leads")
    return 0


def _wide_promote(args: argparse.Namespace) -> int:
    from requisite.discovery import results

    _show_proposal(results.propose_lead(_store(args), args.rank))
    return 0


def _wide_gold_draw(args: argparse.Namespace) -> int:
    from requisite.discovery import wide

    store = _store(args)
    report = wide.gold_coverage(store)
    print(f"the gold design's coverage of the estimand, a ratio, under labels simulated from the judge's probabilities "
          f"({report.draws} draws, seed {report.seed}; a statement about the design, not the frame):")  # fmt: skip
    if report.not_measured:
        print(f"  not measured: {report.not_measured}; the gold is drawn regardless, since it measures the frame itself")
    for r in report.ratios:
        for c in r.intervals:
            print(f"  {c.estimator}: coverage {c.coverage:.3f} (MC se {c.monte_carlo_se:.3f}), missed below "
                  f"{c.missed_below:.3f}, above {c.missed_above:.3f}, mean width in [0, 1] {c.mean_width:.3f}")  # fmt: skip
    for g in report.gold:
        nums = ", ".join(f"{v:.2f} {k}" for k, v in g.expected_numerators.items())
        print(f"  gold in ({g.stratum}, bin {g.bin}): expects {g.expected_gold:.1f} items, {g.expected_denominator:.2f} "
              f"in the population's voice, {nums}")  # fmt: skip
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
    gold.label_session(store, wide.drawn(store, 2), wide.approved_objective(store),
                       wide.measures(store, [d.item for d in wide.drawn(store, 2).items]),
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

    at = args.as_of or f1.INSTANT
    if args.as_of is None:
        print(f"F1's draw, pinned at {at.isoformat()} (requisite.market.f1.INSTANT)")
    client = Kalshi(snapshot=at)
    try:
        print(f1.render(f1.draw(client, at)))
    finally:
        client.close()
    return 0


def _f1_retrieve(args: argparse.Namespace) -> int:
    from requisite.market import f1
    from requisite.market.kalshi import Kalshi

    store = _store(args, begins=True)
    at = args.as_of or f1.INSTANT
    if args.as_of is None:
        print(f"F1's draw, pinned at {at.isoformat()} (requisite.market.f1.INSTANT)")
    client = Kalshi(snapshot=at)
    try:
        d = f1.draw(client, at)
        harvest = f1.retrieve(client, store, d)
    finally:
        client.close()
    print(f1.render(d))
    print()
    print(f1.render_harvest(harvest, store))
    return 0


def _fuzz(args: argparse.Namespace) -> int:
    from requisite.fuzz import revise
    from requisite.fuzz import run as fuzz

    result, store = fuzz.from_case(Path(args.case), args.contract, prose=Path(args.prose) if args.prose else None,
                                   reference=Path(args.reference) if args.reference else None,
                                   providers=args.providers, workers=args.workers)  # fmt: skip
    print(fuzz.summary(result, store))
    print(revise.examples_line(*revise.publish_run(Path(args.case), result)))
    return 0


def _fuzz_revise(args: argparse.Namespace) -> int:
    from requisite.fuzz import revise

    out = revise.from_case(Path(args.case), args.contract, prose=Path(args.prose) if args.prose else None,
                           reference=Path(args.reference) if args.reference else None, providers=args.providers,
                           workers=args.workers)  # fmt: skip
    print(revise.summary(out))
    return 0


def _fuzz_args(fz: argparse.ArgumentParser) -> None:
    fz.add_argument("contract", help="an active contract in the case (id, or id@version), or a ContractIR JSON file")
    fz.add_argument("--case", required=True, help="the case's directory; the run's synthetic case is kept beside it")
    fz.add_argument("--prose", help="a ProseDraft JSON file to test (default: the clause drafter's prose of the IR)")
    fz.add_argument("--reference", help="a JSON list of observations for values the IR reads only through a "
                                         "computation (default: a hundred steps of each one's precision)")
    fz.add_argument("--providers", nargs="+", default=["openai_decisions", "clef", "jev"],
                    choices=["openai_decisions", "clef", "jev"], help="the judges (default: all three)")
    fz.add_argument("--workers", type=int, default=8, help="scenarios worked at once (default 8)")


REVISE = ("revise the clauses a fuzz run suspects from its development scenarios, each section alone, and fuzz again "
          "over the same scenarios and held-out split, by rounds, until the development scenarios pass (spec §8.6 step 7)")


def _revise_parser() -> argparse.ArgumentParser:
    rv = argparse.ArgumentParser(prog="requisite fuzz revise", description=REVISE.capitalize() + ". The run reported "
                                 "is compared with its base on the development and the held-out scenarios apart; a pass "
                                 "counts only on the held-out ones, which the drafter never saw.")  # fmt: skip
    _fuzz_args(rv)
    rv.set_defaults(run=_fuzz_revise)
    return rv


def _approve(args: argparse.Namespace) -> int:
    from requisite.discovery import wide

    p = wide.approve(_store(args), args.proposal, gate=args.gate, approver=args.approver, decision=args.decision,
                     reason=args.reason, scope=args.scope)  # fmt: skip
    _show_proposal(p)
    return 0


def _contract_read(args: argparse.Namespace) -> int:
    from requisite.contract import harvest

    print(harvest.render(harvest.read(_store(args, begins=True), args.series, as_of=args.as_of)))
    return 0


def _dossier(args: argparse.Namespace) -> int:
    from requisite import dossier

    index = dossier.render(Path(args.case_dir), args.case)
    print(f"dossier of case {args.case}: {index.relative_to(Path(args.case_dir))} in {args.case_dir}")
    return 0


def _deep_store(args: argparse.Namespace):  # noqa: ANN202
    from requisite.store import open_case

    return open_case(Path(args.case))


def _floor_point(value: str) -> tuple[str, str] | None:
    """A liquidity floor point given as DAY=AMOUNT, or `none` for no floor."""
    if value == "none":
        return None
    day, sep, amount = value.partition("=")
    if not sep or not day or not amount:
        raise argparse.ArgumentTypeError(f"{value!r} is not DAY=AMOUNT (for example 2026-11-01=0) or none")
    return day, amount


def _deep_open(args: argparse.Namespace) -> int:
    from requisite.discovery import lead
    from requisite.store import Refused

    points = list(args.floor)
    if None in points and len(points) > 1:
        print("requisite deep open: --floor none states there is no floor; it is not given beside floor points",
              file=sys.stderr)  # fmt: skip
        return 2
    choices = lead.Choices(
        objective=args.objective, alpha=None if args.alpha == "none" else args.alpha, tolerance=args.tolerance,
        floor=tuple(p for p in points if p is not None), actions=tuple(args.actions), information=tuple(args.information),
        recourse=args.recourse, currency=args.currency,
    )  # fmt: skip
    try:
        opened = lead.open_lead(Path(args.case_dir), args.case, args.lead, choices)
    except Refused as e:
        print(f"requisite deep open: refused: {e}", file=sys.stderr)
        return 1
    print(lead.render(opened))
    return 0


def _deep_sensitivities(args: argparse.Namespace) -> int:
    from requisite.discovery import deep

    case, res = deep.sensitivities(_deep_store(args))
    print(deep.render_sensitivities(case, res))
    return 0


def _deep_materiality(args: argparse.Namespace) -> int:
    from requisite.discovery import deep

    case, _, rt, _ = deep.recompute(_deep_store(args))
    print(deep.render_materiality(case, rt))
    return 0


def _deep_hedge(args: argparse.Namespace) -> int:
    from requisite.discovery import deep

    case, scr = deep.run_screen(_deep_store(args))
    print(deep.render_screen(case, scr))
    return 0


def _deep_register(args: argparse.Namespace) -> int:
    from requisite.discovery import deep

    an, reg = deep.run_register(_deep_store(args), args.k)
    print(deep.render_register(an, reg))
    return 0


def _deep_loop(args: argparse.Namespace) -> int:
    from requisite.discovery import deep

    out = deep.loop(_deep_store(args), args.k, budget=args.budget)
    print(deep.render_register(out.analysis, out.register))
    print()
    print(deep.render_run(out.run))
    return 0


def _deep_queue(args: argparse.Namespace) -> int:
    from requisite.discovery import deep

    print(deep.render_run(deep.latest_run(_deep_store(args), args.k)))
    return 0


def _deep_approve(args: argparse.Namespace) -> int:
    from requisite.discovery import deep

    a = deep.approve_question(_deep_store(args), args.question, approver=args.approver, decision=args.decision,
                              reason=args.reason, scope=args.scope)  # fmt: skip
    print(f"{a.id}: {a.gate} {a.decision} by {a.approver} at {a.at.isoformat()}, covering lineage {a.lineage_hash[:12]} "
          f"of {a.subject.id}")  # fmt: skip
    return 0


def _deep_diff(args: argparse.Namespace) -> int:
    from requisite.discovery import deep

    diff = deep.compare(_deep_store(args), args.before, args.after)
    print(f"entity model v{args.before} -> v{args.after}: {len(diff.value_changes())} value(s) changed, "
          f"{len(diff.lineage_changes())} justification(s) changed with the same value; recomputed "
          f"{', '.join(diff.recomputed) or 'nothing'}")  # fmt: skip
    for line in diff.lines():
        print(f"  {line}")
    return 0


def _show(args: argparse.Namespace) -> int:
    p = _store(args).proposal(args.proposal)
    if p is None:
        print(f"no proposal {args.proposal}")
        return 1
    if p.kind == "contract":
        from requisite.contract import harvest

        print(harvest.render(p))
        return 0
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
        ("families", "an agent proposes the run's event families over its kept dockets' measures", _wide_families, False),
        ("estimate", "the share, overall and per family, from the gold labels and the judges' labels", _wide_estimate,
         False),
        ("leads", "the businesses in the numerator, asked, their stated impacts extracted, ranked", _wide_leads, True),
    )
    for name, text, run, parallel in steps:
        w = wsub.add_parser(name, help=text)
        case_args(w)
        if parallel:
            w.add_argument("--workers", type=int, default=8, help="requests in flight at once (default 8)")
        w.set_defaults(run=run)
    dp = sub.add_parser("deep", help="deep discovery (spec §7.1): a case opened from a lead approved at G1, then steps "
                        "3-5 and the requisite loop over them")  # fmt: skip
    dsub = dp.add_subparsers(dest="step", required=True)
    d = dsub.add_parser(
        "open", help="open the deep case of a lead approved at G1: its filing as evidence, its measure as the event, "
        "the entity model and the decision spec proposed for G1",
        description="Open the deep case of a lead approved at G1, in <case-dir>/deep/<its case>: the lead's filing, "
        "copied as the wide case committed it, is its evidence; the measure the filing addresses, taking effect within "
        "the approved objective's horizon, is its event; the decomposer proposes the entity model from the filing's "
        "claims (each number quoted from a claim or assumed with its reason); and the decision spec takes its horizon "
        "and loss from the approved objective and every other choice from the flags below, as given. Both await G1.",
    )  # fmt: skip
    d.add_argument("--case-dir", required=True, help="the wide case's directory (its store)")
    d.add_argument("--case", required=True, help="the wide case's name")
    d.add_argument("--lead", required=True, help="the lead proposal's id (requisite wide promote), approved at G1")
    from typing import get_args

    from requisite import core

    d.add_argument("--objective", required=True, choices=get_args(core.DecisionSpec.model_fields["objective"].annotation),
                   help="the decision's risk functional (spec §8.3; addendum §12)")  # fmt: skip
    d.add_argument("--alpha", required=True, help="the tail level, an exact probability below 1 (9/10 or 0.9) where the "
                   "objective is liquidity_cvar; none for another objective")  # fmt: skip
    d.add_argument("--tolerance", required=True, help="the economic tolerance within which actions count as equally "
                   "good, an amount in --currency (addendum A02 §3.2, A03 §3.3)")  # fmt: skip
    d.add_argument("--floor", required=True, action="append", type=_floor_point, metavar="DAY=AMOUNT",
                   help="a liquidity floor point, from that day on, an amount in --currency; repeat for each point, or "
                   "none for no floor")  # fmt: skip
    d.add_argument("--actions", required=True, nargs="+", help="the candidate actions of the decision")
    d.add_argument("--information", required=True, nargs="*", help="what is observable before the decision time "
                   "(addendum A03 §3.6); give the flag with nothing after it where nothing is")  # fmt: skip
    d.add_argument("--recourse", required=True, help="the permissible recourse after the decision")
    d.add_argument("--currency", required=True, help="the ISO 4217 currency the tolerance and the floor are in, which is "
                   "the entity model's")  # fmt: skip
    d.set_defaults(run=_deep_open)
    deep_steps = (
        ("sensitivities", "the sensitivity questions and each event's forecast, judged and resolved", _deep_sensitivities, False),
        ("materiality", "dated cash impacts, expected and tail impact and breach flags, computed by registered nodes",
         _deep_materiality, False),
        ("hedge", "the hedgeability screen: observables, the business's influence, listed markets close", _deep_hedge, False),
        ("register", "the exposure register, every number from the graph and every row traced", _deep_register, True),
        ("loop", "the requisite loop for the decision 'enters the top k of the register'", _deep_loop, True),
        ("queue", "the latest requisite run: its review queue and the questions informational for its decision",
         _deep_queue, None),
    )  # fmt: skip
    for name, text, run, takes_k in deep_steps:
        d = dsub.add_parser(name, help=text)
        d.add_argument("--case", required=True, help="the case's directory (its store)")
        if takes_k:
            d.add_argument("--k", type=int, required=True, help="the register's top k: the decision relevance is for")
        elif takes_k is None:
            d.add_argument("--k", type=int, help="the run's k (default: the latest run)")
        if name == "loop":
            from requisite.discovery.deep import DEFAULT_BUDGET

            d.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="judge and agent calls the loop may make "
                           f"beyond the base run (default {DEFAULT_BUDGET}); spent, the run stops as a blocked draft")  # fmt: skip
        d.set_defaults(run=run)
    d = dsub.add_parser("approve", help="G2: a human decides on a contested relevant question, as it now stands")
    d.add_argument("--case", required=True, help="the case's directory (its store)")
    d.add_argument("--question", required=True, help="the contested question's id")
    d.add_argument("--approver", required=True)
    d.add_argument("--decision", default="approved", choices=["approved", "approved_with_edits", "rejected"])
    d.add_argument("--reason", required=True)
    d.add_argument("--scope", required=True)
    d.set_defaults(run=_deep_approve)
    d = dsub.add_parser("diff", help="what changes between two G1-approved versions of the entity model")
    d.add_argument("--case", required=True, help="the case's directory (its store)")
    d.add_argument("--from", dest="before", type=int, required=True, help="the earlier entity model version")
    d.add_argument("--to", dest="after", type=int, required=True, help="the later entity model version")
    d.set_defaults(run=_deep_diff)
    ph = sub.add_parser("paraphrases", help="an agent writes a library template's frozen paraphrase set (spec §2.1)")
    case_args(ph)
    ph.add_argument("--template", required=True, help="the library template's id")
    ph.set_defaults(run=_paraphrases)
    pr = wsub.add_parser("promote", help="propose one lead for a deep case; a human decides at G1 (requisite approve)")
    case_args(pr)
    pr.add_argument("--rank", type=int, required=True, help="the lead's rank")
    pr.set_defaults(run=_wide_promote)
    g = sub.add_parser("gold", help="label the gold sample blind, one item at a time (stop and resume at will)")
    case_args(g)
    g.add_argument("--labeler", required=True, help="who is labelling")
    g.set_defaults(run=_gold)
    mk = sub.add_parser("market", help="the market lens, read-only (spec §10): Kalshi's filed record, F1's contracts")
    msub = mk.add_subparsers(dest="step", required=True)
    m = msub.add_parser("f1-draw", help="draw F1's filed contracts at an instant and print the rule's evidence")
    m.add_argument("--as-of", type=_instant, default=None,
                   help="the instant the draw is pinned at, with its zone; F1's own draw is at 2026-11-02T00:00:00Z "
                        "(requisite.market.f1.INSTANT, fixed before the draw ran), which is what this takes when none is given")  # fmt: skip
    m.set_defaults(run=_f1_draw)
    m = msub.add_parser("f1-retrieve", help="draw, then commit the drawn series' filings, rules and settled outcomes")
    case_args(m)
    m.add_argument("--as-of", type=_instant, default=None,
                   help="the instant the draw is pinned at, with its zone; F1's own draw is at 2026-11-02T00:00:00Z "
                        "(requisite.market.f1.INSTANT, fixed before the draw ran), which is what this takes when none is given")  # fmt: skip
    m.set_defaults(run=_f1_retrieve)
    ct = sub.add_parser("contract", help="filed contracts read into the IR (spec §9.3) and checked against settlements")
    csub = ct.add_subparsers(dest="step", required=True)
    m = csub.add_parser("read", help="the contract_reader reads a harvested series into the IR; its gates check the IRs "
                        "against how its markets settled")  # fmt: skip
    case_args(m)
    m.add_argument("--series", required=True, help="the harvested series' ticker")
    m.add_argument("--as-of", type=_instant, help="the draw's instant, where the series is harvested at more than one")
    m.set_defaults(run=_contract_read)
    fz = sub.add_parser("fuzz", help="resolvability fuzzing (spec §8.6): a contract's prose tested against its IR; "
                        "`requisite fuzz revise <contract> --case <dir>` revises the clauses it suspects",
                        description="Resolvability fuzzing (spec §8.6): a contract's prose tested against its IR. "
                        f"`requisite fuzz revise <contract> --case <dir>` (see its --help): {REVISE}.")  # fmt: skip
    _fuzz_args(fz)
    fz.set_defaults(run=_fuzz)
    a = sub.add_parser("approve", help="a human decides at a gate (G1-G4) on one proposal, as it stands")
    case_args(a)
    a.add_argument("proposal")
    a.add_argument("--gate", required=True, choices=["G1", "G2", "G3", "G4"])
    a.add_argument("--approver", required=True)
    a.add_argument("--decision", default="approved", choices=["approved", "approved_with_edits", "rejected"])
    a.add_argument("--reason", required=True)
    a.add_argument("--scope", required=True)
    a.set_defaults(run=_approve)
    d = sub.add_parser("dossier", help="render the case's static HTML dossier, and its synthetic fuzz cases', into "
                       "<case-dir>/dossier/ from its committed state alone (spec §13 WP16)")  # fmt: skip
    case_args(d)
    d.set_defaults(run=_dossier)
    sh = sub.add_parser("show", help="a proposal as it stands: payload, gates, rationale")
    case_args(sh)
    sh.add_argument("proposal")
    sh.set_defaults(run=_show)
    given = list(sys.argv[1:] if argv is None else argv)
    if given[:2] == ["fuzz", "revise"]:  # a step of fuzz, beside its contract argument
        args = _revise_parser().parse_args(given[2:])
    else:
        args = parser.parse_args(given)
    from requisite.cassettes import ReplayOnly
    from requisite.store import Refused

    try:
        return int(args.run(args))
    except Refused as e:
        # A refusal is the product's answer, said plainly: what it will not do and why (never a crash's traceback).
        print(f"requisite: refused: {e}", file=sys.stderr)
        return 2
    except ReplayOnly as e:
        print(f"requisite: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
