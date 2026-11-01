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
