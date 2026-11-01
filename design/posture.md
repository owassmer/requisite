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

An *escape route* lets work go on without what it needs, or makes a case pass while hiding the gap. In Requisite they
have names, because the sources name them:
- an unknown coerced to false or zero (I6): a CountTrue over unknown days is an interval, not its lower end;
- a band's midpoint passed off as an expected value, or a range as a distribution (addendum A04);
- a price read as a probability, or a probability with no stated risk premium (I9);
- incoherent forecasts silently normalized, infeasible scenario mass silently dropped (addendum §4.2);
- an LP's slack reported as realized shortfall, or a solver's arbitrary tie read as a decision flip (addendum A03, A06);
- a hand-picked example, or a coverage obligation met by an impossible scenario (spec §8.5, addendum A09);
- a default, `or 0`, a caught exception or a fallback that turns a failure into a value;
- code bent to make a test pass where the test or its fixture is what is wrong;
- an "informational" or "requisite" label claimed beyond the decision, tolerance and domain it was tested on.

The aim is that the failure cannot happen: the type does not admit it (a synthetic claim cannot be an observed claim,
I4), the state is read from where it is held, and what cannot be computed stands as an error or an explicit unknown with
its reason. *Bloat* is the mirror: a check, refusal or field added wherever a symptom showed. A fix for one must not open
the other; a fix that adds more than it removes probably treated a symptom.

## Finding the cause before choosing the fix

A failing test or a verifier's finding is a symptom. Trace it to the definition: the type, field, table row or invariant
that is missing, defined wrongly or held twice. Fix it there and let what derives from it follow. A stream of small
findings is almost always one cause, usually two copies of one thing disagreeing.

## Meaning, judgment and uncertainty

Words become structured state only through a decision model answering a typed question over a canonically rendered
state, never through a pattern. Agreement among judges is a perturbation diagnostic, not proof that a question is atomic,
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
template are not independent validation (A01 §2.5), and Paleka et al.'s finding that a forecaster patched to pass its
consistency checks did not improve, are the same lesson. So a finding points at the definition or the machinery; never
add a state, a branch or a question because a check produced it. Done is true in the running product: from a clean
clone, replaying recorded responses, the demo's flow works as a user would run it.

## Working on a task

Read the code you change whole, not by search. Where the code or the sources are not as your task reads them, stop that
part and report rather than improvising. A known defect is fixed or reported, never left for later. What your task
settles is the reason; the mechanism it names is the coordinator's best reading, and where it is wrong for the reason,
say so. Workers never call a live provider: the sandbox has no network beyond loopback, and recorded responses are the
only path, so a test never spends and never varies.
