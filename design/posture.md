# How Requisite is built: the posture

The reasoning the product is built from, not a rulebook: each point says what it is for. Read it for its intent, apply it
where nothing here names your case, and say plainly where you think it is wrong for your situation. `decisions.md` is the
record it comes from; the code is the current state. Its form comes from Handoff's factory, whose lessons it carries where
their reasons hold here; its content comes from Requisite's own sources, the build spec and the mathematical addendum.

## What the product is for, and what that asks of the code

Requisite replaces taste with measurement at two seams. A question is refined until independent judges agree on it (the
floor) and no further than any decision needs (the ceiling). A contract is a question addressed to a future resolver, so
its settlement language is tested now against its own formal representation. Both only mean something if every number
the runtime shows was computed by code from committed state, every judgment came from a bounded question over a pinned
state, and every claim the runtime makes is scoped to what was actually tested.

**Each actor has one job and one output type.** Agents propose (a Proposal); decision models return a distribution over
a closed answer set; code validates, composes, computes and commits; humans authorize at G1–G4. Only code writes
committed state (I1). A model that does arithmetic in prose, or an agent that writes a record, has taken another actor's
job, and the result cannot be trusted or recomputed.

**One definition, one evaluator.** The §4 types live once, in `requisite/core/`, and every package imports them. The
same IR evaluator computes payoffs in construction, examples in drafting, ground truth in fuzzing and settlement at
resolution (I8). A second copy of a type or a second evaluator is where the system starts to disagree with itself.

**Code is written over kinds, never instances.** What is true of one docket, one contract or one client is data
(fixtures, library entries, case state), never code. An example drawn from one case and written into code or a prompt
reads elsewhere as how things are.

## Escape routes and bloat: the two failures to keep apart

An *escape route* is anything that lets work end without its purpose met. Owen refuses all of them, loud as much as quiet
(decisions.md, November 1):
- *Hidden failure*: a case made to pass while the gap stays. A default, `or 0`, a caught exception or a fallback that
  turns a failure into a value; code bent to make a test pass where the test or its fixture is what is wrong. In
  Requisite these have names, because the sources name them: an unknown coerced to false or zero (I6: a CountTrue over
  unknown days is an interval, not its lower end); a band's midpoint passed off as an expected value, or a range as a
  distribution (A04); a price read as a probability (I9); incoherent forecasts silently normalized, infeasible scenario
  mass silently dropped (addendum §4.2); an LP's slack reported as realized shortfall, a solver's arbitrary tie read as a
  decision flip (A03, A06); a coverage obligation met by an impossible scenario (A09); a "requisite" or "informational"
  label claimed beyond the decision, tolerance and domain it was tested on.
- *Loud failure*: the gap made visible and the visibility taken for the job. An error raised, an unknown returned, a
  warning in the dossier, "needs human review", a BROKEN line, a "not true" in a report, an OPEN item, where the work
  could have produced the answer. Showing a gap is better than hiding it and is never where work stops: the bar is that
  there is no gap.
- *Deferral*: "later", "Tier 1", "a follow-up", "for now", "a stand-in until", a held list, a question put to Owen that
  reasoning can settle. What a claim needs is built when the claim is made. A finding is traced to its cause and fixed
  when it arrives.
- *Dismissal and shortcut*: "harmless", "flaky", "noise", "pre-existing", "an edge case", "cosmetic", said without
  checking. A rule, a precedent, a pattern or a passing test cited in place of reasoning about the case. A hand pick (a
  docket, an example, a comment) in place of what the flow would produce.

The product does state unknowns, contested readings, uncovered risks and blocked drafts, and these are not failures when
they are true of the world: once the flow has fetched what evidence it can, run its probes and refined the question, the
evidence as of as_of does not establish the fact. Then the label is a finding, and the flow takes the action its
diagnosis prescribes (fetch, refine, split, route to G2); it does not stop at the label. An unknown that comes from work
not done is a defect of ours presented as a fact about the world, which is worse than an error. That covers evidence not
fetched, a probe not built, a budget set too low, a response not recorded.

So the failure is made unable to happen: the type refuses it (I4), the state is read from where it is held, and the work
that produces the answer exists. A guard that refuses a wrong action (a replay-only process asking for a key, an
overwrite of evidence) is structure: it prevents a false state. It is never the answer to work the product exists to do.
*Bloat* is the mirror: a check, refusal or field added wherever a symptom showed. A fix for one must not open the other;
a fix that adds more than it removes probably treated a symptom.

## Finding the cause before choosing the fix

A failing test or a verifier's finding is a symptom. Trace it to the definition: the type, field, table row or invariant
that is missing, defined wrongly or held twice. Fix it there and let what derives from it follow. A stream of small
findings is almost always one cause, usually two copies of one thing disagreeing.

## Meaning, judgment and uncertainty

Words become structured state only through a decision model answering a typed question over a canonically rendered
state, never through a pattern. Each check is of the kind of thing it checks: text is checked as text (a quote is found in
its source, which is provenance, not meaning), an agent or a decision model reads the words into a structured form, and the
structured form is checked as structure. Code never checks deterministically what is inherently unstructured, since a
pattern can neither confirm nor refute what words mean (Owen, November 2). Agreement among judges is a perturbation diagnostic, not proof that a question is atomic,
adequate or correct (addendum A01): stable judges can share a bias, and a stable coarse answer can hide a distinction a
decision needs (A02). Disagreement is diagnosed and routed, never averaged away. An interpretation's distribution, a
future event's probability and an evidence status are three different objects (A04). Configurations are not independent
witnesses; their pooled value is an ensemble estimate under a stated pooling policy.

## Verification and done

Whether the definition says what the sources say is verified by reading each of its statements beside its source (a
section of the spec, an amendment, a cited paper, a vendor's documentation, Owen's decision, or our reasoning marked as
ours). The other model family judges each staged change to the definition before it is committed.

Whether the machinery does what the definition says is verified exactly: truth tables over every combination, the
numerical oracle's 13 results, property tests at the limits, and one test per enumerated item in the registry.

Every other check (a seeded fixture, a reviewer, the verifier, a live provider run) touches correctness at one point.
Followed as a target it adds nuance the world does not have: the addendum's warning that adaptive tests used to revise a
template are not independent validation (A01 §2.5), and the finding spec §10 cites from Paleka et al. (ICLR 2025), that
patching a forecaster to pass the checks it was optimized against "did not generalize to held-out checks or improve
accuracy", are the same lesson. So a finding points at the definition or the machinery; never add a state, a branch or a
question because a check produced it.

Done is the claim true of its object, never a check standing in for it. Every number the product shows recomputes from
committed state to the same value by an independent computation. Every judgment it shows was asked of the provider it
names, over the state it shows, and is recorded. Every statement about the world reads true beside its source. Every
output presented as the flow's (a corpus, a lead, a driver tree, a clause, an example) was produced by the flow, never
authored by the builders. CI green, a covered registry item, the oracle reproduced, a review's HOLDS, a verifier's
WORKS and a worker's report are each necessary and none is the bar. The bar is met from a clean clone, replaying recorded
responses, with the demo's flow run as a user would run it; each outcome's file says what its claims are checked against.

## Working on a task

Read the code you change whole, not by search. Where the code or the sources are not as your task reads them, find what
is true from the sources and the code, and do what the task's reason requires there; your report says what you found
and what you did. A defect you find is fixed at its cause when you find it, wherever in the product it is. What your task
settles is the reason; the mechanism it names is the coordinator's best reading, and where it is wrong for the reason,
follow the reason and say why. Workers never call a live provider: the sandbox has no network beyond loopback and
recorded responses are the only path, so a test never spends and never varies. The coordinator records every response a
task needs before it goes out.
