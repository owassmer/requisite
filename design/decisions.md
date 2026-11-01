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
  draw and so does not claim.
- The same reason holds for F1, by the coordinator's reasoning: the filed contracts are drawn by a rule over the filed
  record (spec §9.3; one per strike type where they exist), plus the Western Grazers specimen the spec names, and the IR
  grammar comes from spec §8.1, never fitted to them. A filed construct the IR cannot express is a finding about the
  grammar, fixed there if the construct is general.

## No escape routes, loud or quiet

- Owen, November 1: "I am also uncompromisingly averse to 'loud failure' escape routes and deferrals/dismissals/shortcuts
  to reasoning." Making a gap visible (an error, an unknown, a flag, "needs review", a BROKEN line, a "not true" in a
  report, an OPEN item) is not the job done; the bar is no failure (Handoff's decision of October 6 is the same: aiming at
  visible failure gives an exit from the bar). Nothing waits on a later tier, a follow-up, a held list or Owen when
  reasoning can settle it now; nothing is called harmless, flaky, pre-existing or cosmetic without being checked. How it
  reads in the product, where unknowns are real findings, is in design/posture.md.

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
- Requisite's own agents (spec WP15: discovery, decomposer, clause drafter, scenario renderer, template proposer,
  post-mortem) run on Claude Opus 5.5 through the same OAuth subscription route as the coordinator's claude-cli
  extension, at the reasoning level the work warrants (Owen, November 1). Their outputs are Proposals, recorded, so the
  demo replays offline.
