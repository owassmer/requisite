# Decisions

What was decided and why. Each line carries its reason and, where it is Owen's, his words or the date he gave it. The
work is judged against this file; its writer is the coordinator (a worker's report proposes changes in `decisions`).

## Scope

- Requisite builds the runtime of the build spec v3 (November 1, 2026): agents propose structure, decision models test
  bounded judgments, code commits state and computes consequences, humans authorize at G1–G4. Two flows: Discovery (deep
  and wide) and Construction. Out of scope: order placement, RFQ submission, account linking, retrospective critique of
  any specific trade, and anything presented as legal or investment advice. Every market adapter is read-only.
- Binding: the invariants I1–I10 (spec §3) and the interfaces (spec §4) as amended. Discretionary: module internals, as
  long as the acceptance tests pass. When unsure, the smaller implementation that preserves an invariant over a larger one
  that adds a feature (spec §0).
- The mathematical addendum is accepted (Owen, November 1: "1. I accept"). Its amendments A01–A12 enter the canonical §4
  module once, in `requisite/core/`; no second schema. Tier-0 behaviour is implemented where an amendment changes a Tier-0
  acceptance test (WP4, WP5, WP6, WP7, WP8, WP9, WP10, WP11b, WP14); C01's reduced costs are reported at Tier 0, since
  the LP's duals come with the solve, and the agent search loop over them is Tier 1.
- Acceptance tests the addendum replaces, so nothing is built to the old ones: WP10's "perfect instrument: residual equals
  premium" and "uncorrelated instruments: buys nothing" become the dominated-payoff fixture with premium tie-breaking and
  the zero-covariance counterexample (A06 §6.5); WP13's "exclusive set summing to 1.10 is flagged" becomes a quote-aware
  payoff-dominance diagnostic (A04 §4.5); WP8's "at least N+1 scenarios" becomes feasible unique-cause witnesses with
  infeasible obligations recorded (A09 §10.2); §11's "accepted_uncovered can never be a defect" becomes "accepted risk
  and component defects coexist" (A10 §11.1).
- Tiers (spec §0): Tier 0 runs end to end from a clean clone and is due Tue Nov 3, 09:00 ET; Tier 1 starts only after
  Tier 0 is green; the demo is Tue Nov 3, 13:30 ET. Never cut: I1–I10, the three-valued evaluator, the rendering check,
  the diff (spec §13).

## Discovery finds what it analyzes

- The discovery flow chooses its own corpus and its own lead. From the objective a human approves at G1, it defines its
  frame from the public record as of a pinned date, screens and samples it by a declared design, retrieves and freezes
  what it reads, and its leads decide which docket and which business the deep case takes. No one picks a docket
  beforehand (Owen, November 1: "the whole point of discovery is that it finds it independently through it's flow we
  build. If you decide beforehand then 1. we will likely hyperfixate on the docket, and 2. the whole repeatable flow that
  is supposed to find these is not faithfully tested"). This replaces spec §12's "proposes three dockets; Owen picks one"
  and §15's Sun 22:00 docket question, and removes the cut rule "Run F2 from a hand-picked docket comment instead of the
  F4 wide run", which is the hand pick this decision removes.
- An objective says what its brief means, never what today's adapters reach; the frame records which corpora it reached,
  and every estimate is a share of the frame's items with that reach stated beside it (Owen, November 1: "is this just
  one of the discovery flows? Why would we restrict to just federal dockets?"; spec §7.3 lists the corpora in priority
  order). F4's objective, drafted by the objective agent from spec §12's F2/F4 wording, was approved at G1 as drafted
  (Owen, November 1: "Approve as drafted").
- The dockets the coordinator surveyed by hand on November 1 enter nothing: no source identifier is written into code,
  prompts, tasks or fixtures before the flow produces it, and no template, question or structure is shaped to a source
  the flow has not yet chosen.
- Once the flow's version and as_of are frozen and it has run, its result stands. A change made after seeing a result is
  a defect fix, recorded with its cause, and the whole flow runs again from the same objective and as_of; it is never
  re-run until a preferred lead appears (addendum A01 §2.5: tests used adaptively to revise are not independent
  validation).
- The wide flow reads its frame by a two-phase design (coordinator's reasoning, November 1). Reading one item from
  regulations.gov is one request, and the key allows 1,000 an hour (measured November 1), so a broad frame cannot be read
  whole. Phase 1 is a stratified simple random sample of the frame without replacement, read and model-labelled. Phase 2
  is the gold sample Owen labels: a stratified simple random subsample of phase 1, whose strata may come from the model's
  phase-1 labels. A frame small enough to read whole is a phase-1 census, and then the estimator is exactly addendum
  A07's finite-frame form. The plain difference estimator (λ = 1) is the primary estimate, since its design-unbiasedness
  is exact; a tuned λ is reported beside it with its measured coverage (A07 §8.3). Two-phase theory is standard survey
  sampling (Särndal, Swensson and Wretman 1992, ch. 9), not the addendum's, and is proved in the tests by exact
  enumeration. The unit is the item, so a wide prevalence is the frame share of items (comments) whose indicator holds,
  for example "in a business's voice and describing exposure to this event family" (spec §7.2: items are comments,
  letters, filings, posts). A share of businesses would need a cluster design over submitters, which the product does not
  draw and so does not claim. Phase-2 strata are nested in phase-1 strata and formed from the model's phase-1 labels;
  each nonempty phase-2 stratum is allocated at least two gold items, or all of its items when it has fewer, so every
  item has a positive inclusion probability (A07 §8.2: pi_i > 0) and every stratum's residual variance can be
  estimated from its own gold (coordinator's reasoning: one gold item gives no within-stratum variance, and the design's
  unbiased variance estimator needs one; the 000-2 task, November 1).
- Wide inference as built (worker on 000-2, November 1, measured by simulation under the exact design, 1,000 draws).
  The interval is a score interval: each bound is the estimate minus or plus z times the SD evaluated at the phase-2
  stratum prevalences restricted to that bound (Wilson's interval for one stratum); a stratum's f is credited with
  reducing the variance only as far as that stratum's own gold shows, never borrowed across strata, since a judge's
  false negatives sit exactly where its f does not separate the labels. Wald under-covered when a heavy stratum's gold
  held no positive (0.918 at λ = 1, 0.657 cross-fitted); MOVER-of-Wilson missed on one side only. The tuned λ is
  cross-fitted leave-one-out, which keeps its point estimate exactly unbiased (proved by enumeration); its variance is
  asymptotic and its coverage measured by requisite/discovery/sampling.py's measure_coverage. The flow draws both
  phases with that module's draw_phase1 and draw_phase2, so the sample is the design the estimator assumes.
- Phase-2 allocation (coordinator, November 1, from 000-2's numbers). Within each phase-1 stratum, 80% of its gold goes
  to the phase-2 stratum the judge scores below the cut (the large stratum where its missed positives live) and 20% to
  the rest, each at least two. Equal allocation over-covered rare families (0.981–0.986, almost no upper misses) because
  their dominant stratum's gold expected under one positive; 80/20 brought the same scenarios to 0.970–0.978. Before the
  gold is drawn, measure_coverage on the actual phase-1 predictions gives each family's expected gold positives, and the
  run reports beside each family's interval its measured coverage and expected positives. Where a family is so rare in
  the frame that its interval over-covers at any allocation the labelling budget allows, the interval is still valid (it
  covers at least nominally) and wide for a stated reason: the gold saw almost none of it.
- F4's sample design, fixed before its frame's sizes were seen (coordinator, November 1, 21:20 ET, while the frame was
  still being taken). Every number below is a rule over the frozen frame, so nobody chooses a stratum, a seed or an item.
  - Phase-1 strata: each kept docket's comment count (core.FrameDocket.comments) in decades, [min_comments, 100),
    [100, 1,000), [1,000, 10,000) and [10,000, ∞); an item takes its docket's band. The count is known from the frame
    alone, so the strata depend on no label. The bands separate where single filers write from where mass campaigns of
    individuals concentrate (our reasoning: the largest dockets are dominated by form letters), so the voice indicator
    should differ across them, which is when stratifying gains precision. The phase-2 constraint fixes how few: each
    phase-1 stratum carries two phase-2 strata of at least two gold items each, out of about 130.
  - Phase-1 size: 1,200 items, read whole (one API request each, plus attachments from the file CDN), split equally over
    the nonempty strata; a stratum with no more items than its share is read whole, and what it leaves is shared equally
    among the rest, repeated until every share fits, the indivisible remainder going one each to the strata in band order.
    Equal shares read the smaller dockets more than their weight, where leads are likelier; the estimator weights strata
    by N_h/N, so the estimate is unaffected.
  - Seeds: phase 1's is the first 16 hex digits of the frame's content hash read as an integer; the stability subsample's
    and phase 2's are the same of the phase-1 draw's content hash with "stability" or "phase2" appended.
  - Phase-1 labels: both objective questions on every phase-1 item, by labels.PHASE1_CONFIG; exposure_to_measure is bound
    to the docket's measure, which code takes from the frame's committed documents listing: the document that decided the
    docket's comment status (the latest Proposed Rule posted by as_of where it is pending a final rule, otherwise the
    latest-posted document whose comment period runs at as_of), stated by its title, agency, type, id and posting date.
  - Stability (spec §2.1, §7.2 step 3): a simple random subsample of 60 phase-1 items (spec: at least 50; ten more so
    refusals cannot take it under), every configuration of labels.stability_configs, both questions.
  - Phase 2: the stratifier is the judge's probability that the item is in the estimand's numerator, P(in_population) x
    P(exposure in labor, inputs or labor_and_inputs) from the phase-1 labels; the cut is 0.5; within each phase-1 stratum
    130/H gold items (H the nonempty phase-1 strata, rounded down, the remainder one each in band order) split 80/20 by
    the decision above, each phase-2 stratum at least two. Every gold item gets both questions. measure_coverage on the
    actual phase-1 predictions is run and reported before the draw.
- Public sources are reached through their official APIs and the files those APIs link to, identified honestly (spec
  §14). Requisite's User-Agent is `Mozilla/5.0 (compatible; RequisiteResearch/1.0; +https://github.com/owassmer/requisite)`,
  the conventional self-identifying crawler form: regulations.gov's file CDN refuses agents without the Mozilla/5.0
  prefix (measured November 1: 403 for curl, httpx and a bare product token; 200 for this form), and the form names the
  runtime and where to learn about it, so nothing is impersonated. The regulations.gov key goes in the X-Api-Key header,
  never in a URL, so it never reaches a cassette key or a log.
- An item's text is everything its submitter filed: the comment and the text of each attachment. A PDF page without a
  text layer is read by OCR (tesseract, installed on this machine; CI installs it), each page's method recorded in the
  extraction, so a scanned business letter is read rather than counted as silent (coordinator's reasoning: an item the
  judges cannot read would be labelled not_established for a reason of ours, not the world's). Extractions are committed
  state, so a replay reads them from the frozen case rather than re-running OCR on another machine.
- F1's filed contracts are drawn by a rule over the filed record, fixed here before the draw runs (coordinator,
  November 1). The Western Grazers contract is in by name: spec §9.3 names it as the reference specimen, and it is
  Kalshi's series KXCAGOAT, "California goat herder wage obligations" (its product certification and contract terms
  are filed PDFs; found November 1 through the read-only Kalshi API). The others: among Kalshi series with filed contract
  terms (contract_terms_url) and at least one market that has settled, as of a pinned instant, one series per strike
  type present among those settled markets, the one whose ticker has the smallest sha256("requisite-F1|" + ticker);
  where more than four strike types are present, the four with the most settled markets. A series's strike types are
  those of its markets settled by the instant (strike_type is a market field); a series is eligible under each. The
  named series is apart from the draw: it counts toward neither the cap nor any type, and the draw passes over it to
  the next in hash order. The hash makes the choice
  independent of anyone's view of which contracts are interesting; spec §12 asks for 2–5 contracts and spec §9.3 for
  every strike type over the harvest as a whole. The IR grammar comes from spec §8.1 and the definition, never fitted to
  a drawn contract: a filed construct the IR cannot express is a finding about the grammar, fixed there when the
  construct is general, and the contract is otherwise unsupported or supported with explicit assumptions (A05 §5.5).

## No escape routes, loud or quiet

- Owen, November 1: "I am also uncompromisingly averse to 'loud failure' escape routes and deferrals/dismissals/shortcuts
  to reasoning." Making a gap visible (an error, an unknown, a flag, "needs review", a BROKEN line, a "not true" in a
  report, an OPEN item) is not the job done; the bar is no failure (Handoff's decision of October 6 is the same: aiming at
  visible failure gives an exit from the bar). Nothing waits on a later tier, a follow-up, a held list or Owen when
  reasoning can settle it now; nothing is called harmless, flaky, pre-existing or cosmetic without being checked. How it
  reads in the product, where unknowns are real findings, is in design/posture.md.

## Numbers

- Two kinds of probability, held apart (addendum A04 §4.1). Scenario and branch probabilities (a ProbabilityEstimate,
  an event tree's branches, the optimizer's scenarios) are exact: Fraction, Decimal or int, summing to exactly one, never
  normalized (addendum §4.2), with a float refused. Expectations under them (CVaR, E[D], a breach probability) are exact
  Fractions, since a third of a dollar has no finite Decimal. A decision model's answer distribution is the provider's
  float output, read at the step the provider reports at (measured November 1: Decisions two decimals, Clef four, Jev
  one). A reported vector is a rounding when some distribution lies within half a step of every reported value: it is
  corrected to the closest such distribution (an equal shift of every value, clipped to its own interval), and the
  Judgment keeps both the reported and the corrected values (addendum A04 §4.2: "Numerical round-off corrections need a
  declared tolerance and log"). Beyond that tolerance it is no rounding, and the judgment has no
  distribution, with the reason; it is never normalized. The stability kernel then computes over distributions that sum
  to one within the oracle's 1e-10 (worker on 000-1 and coordinator, November 1). Money is Decimal, and the optimizer's
  cash accounting is exact Decimal, in which a rounding raises.
- The liquidity optimizer (000-1, addendum A06, C01 §7.2). Receipts arrive per grid date and the kernel cumulates them,
  so its input is the I8 evaluator's dated transfers placed on the grid. The business arrives as B_0, operating cash
  before the modeled loss, and the loss, so the kernel also computes the loss, payoff and terminal residual of the
  scenario table. Money is scaled by the power of ten at the leading digit of max |B_st − r_t|, the business's own cash,
  so a generous cap cannot widen the tolerance. ε_solver is absolute on the scaled primary objective, default 1e-9 (the
  oracle's). The premium-minimizing solve is held to ε/2, and the chosen portfolio's exactly recomputed CVaR must lie
  within ε × scale of the LP optimum, or the solve fails (HiGHS answers miss an exact ≤ ε by about 1e-16, as in the
  oracle's check 6). Deficits are reported exactly. A breach of a solve's portfolios is a deficit beyond ε × scale,
  because the premium tie-break may leave an otherwise fully funded portfolio that close to the floor. That tolerance is
  far below a cent for any business whose cash is under ten billion dollars; the coordinator's reasoning is that a
  sub-cent deficit is no breach of anything money can measure. Dominance with a dated premium means A_stj ≤ 0.
- Coverage (000-3, addendum A09 §10.2). World constraints may mention conditions that are not in the trigger: a state
  of the trigger's conditions is feasible when some assignment of the others satisfies every constraint, and every
  feasible state the coverage kernel shows carries that assignment as its certificate, checkable by evaluating the
  constraints (coordinator, settled in 000-3: refusing such constraints would leave real contracts unanalysable). A
  minimal group of coupled redundant conditions is a coverage obligation of its own, met by two feasible scenarios that
  agree outside the group and differ in outcome: none of its conditions has an independent effect, and without the
  group obligation a fully coupled trigger would be "covered" by a single scenario (A09 §10.2: "requiring a different
  justified coverage criterion"; worker's reasoning on 000-3). Enumeration gives way to python-sat above 10 conditions,
  where enumeration's self-evident proof still costs milliseconds.
- Computation (000-4, spec §5, addendum A11 §11.2). One graph of sources and instances, each carrying its §5 layer. A
  source is a value read from committed records and names at least one of them, so no constant written into code
  enters the graph; an instance is a node (id, version) over named inputs. The value hash is a typed canonical hash
  (values that print alike but differ in type differ; only immutable data hashes); the lineage hash is, for a source,
  its id, value hash and records' content hashes, and for an instance its node id and version and each input's name,
  value hash and lineage: a DerivationReceipt's fields, so every receipt is checkable. A judgment input's question
  version, state hash and configuration enter its consumer's key (spec §5). Receipts commit through
  Store.commit_derivations, one transaction per recompute, keyed by lineage; a committed lineage with another value is
  refused, the symptom of a node changed without a new version or an impure one. A diff lists an approval where it stops
  covering its vertex's lineage; explain() states any approval's standing. Approvals of graph values name the vertex id;
  vertex ids are written "v:<name>", so none collides with a proposal id. I7's static walk applies its purity rules
  (no provider, cassette, key, agent or store module; no network, process, database, clock, randomness, environment,
  file or dynamic import) to every product module a node's module reaches, since the walk sees modules, not calls.
- The exposure register's dated cash impact is monthly (core.DatedImpact; coordinator, November 1): deep discovery's
  materiality screens which exposures matter, and its drivers are stated per period. Intra-month liquidity timing is not
  the register's to judge: construction's optimizer takes the instrument's dated transfers (I8) and the business's cash
  on its own dated grid (A06 §6.1), where a payout late within a month shows as a breach.
- Reduced costs are diagnostics only. HiGHS's duals are not unique (in the addendum's own timing fixture a late payout
  and an early one both price at −110), so wherever the product shows a reduced cost it shows beside it the re-solve's
  measured improvement (C01 §7.2: a re-solve always runs).
- The oracle's recorded values (spec/verified_synthetic_results.json, local) are written into the tests that reproduce
  them, since spec/ is not in git; each transcription is checked against the JSON (the coordinator re-checked 000-1's:
  checks 5–9 match exactly).

## The factory

- The repository is public on GitHub, created by the coordinator, with no branch protection (Owen, November 1: "make it
  public and I don't need to set up branch protection. Create it yourself"). So CODEOWNERS binds nothing, and the
  coordinator merges factory pull requests itself; changes that loosen a check on its own work still go to Owen first,
  by asking, since nothing in GitHub enforces it.
- Since the repository is public, the source documents (`spec/`: the build spec, which carries the demo plan, and the
  addendum) stay out of git, local to each checkout, and are copied into each role's workspace. Secrets are never in git,
  logs, cassettes or fixtures (spec §14).
- Roles, as in Handoff's factory: the coordinator writes the definition (`requisite/core/`) and anything that holds the
  design; workers (Claude Opus 5.5, xhigh) take work packages that stand apart, each in a sandbox with no network beyond
  loopback; a verifier and a reviewer of the other family (GPT-6 Astra) judge outcomes and each staged change to the
  definition. A fresh reader restates each task before it goes.
- Tier-0 work lands on one branch, `outcome/tier0`, in the main checkout: its outcomes (000-007) are one connected demo,
  each built on the last, so separate branches would only add merges. It goes to main at Gate A, Gate B and Tier 0 green,
  each after the verifier has judged the outcomes it carries (coordinator's reasoning, November 1).

## Providers and keys

- Decision models (spec §6.2): OpenAI Decisions, Clef (Cloudflare Workers AI) and Jev (Typesafe, through OpenRouter),
  using the same keys as Slope and Handoff (Owen, November 1: "Use from same key as Slope/Handoff (cloudflare also for
  Clef) Requisite can use these"). The api.data.gov key is in this checkout's `.env` (GOV_API_KEY). Keys are read at run
  time from the environment, `.env`, Slope's `.env`, then the Keychain (`handoff.<NAME>`), and never printed.
- Spend caps are flexible for now (Owen, November 1); each run still records its calls and dollars (spec §6.6), and live
  calls happen only outside the workers' sandbox, by the coordinator, recorded as cassettes.
- One question per Decisions request, the facts as the state: Slope measured on October 29 that a question's situation
  placed in a batched question's instructions moved its answer (0.52 alone, 0.13 batched), while repeat asks alone were
  identical. Spec §6.2's "batch only mutually independent questions" does not prevent that.
- OpenAI's organization limit is 2,000,000 input tokens a minute, shared with Slope through the same key, so Requisite
  paces its Decisions requests at 1,000,000 a minute (estimated as request bytes over four) and retries a 429 with
  backoff (coordinator, November 1, from Slope's measured limit of October 29).
- Requisite's own agents (spec WP15: discovery, decomposer, clause drafter, scenario renderer, template proposer,
  post-mortem) run on Claude Opus 5.5 through the same OAuth subscription route as the coordinator's claude-cli
  extension, at the reasoning level the work warrants (Owen, November 1). Their outputs are Proposals, recorded, so the
  demo replays offline.
