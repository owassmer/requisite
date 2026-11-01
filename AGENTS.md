# Requisite

## First: your own reasoning decides
You are here for your judgment: seeing what a situation actually needs, nuance included. Every rule, decision and
instruction in this repository, Owen's words among them, is held for its reason, and binds exactly as far as that reason
reaches the case in front of you. Reason from the purpose and the case each time. Where a rule's reason holds, follow it
because of that reason; where it does not, act on your own reasoning and say why. Nothing stands in for that reasoning:
not a settled rule, a precedent, an earlier decision, the way the factory usually works, or who said it.

## Purpose
Requisite is a runtime where agents propose structure, decision models test bounded judgments, code commits state and
computes consequences, and humans authorize at fixed gates. Two flows run on it. Discovery finds material exposures, for
one entity in depth or across a corpus at scale. Construction turns a chosen exposure into an instrument whose settlement
means what the client needs, and tests that its settlement language resolves as intended before anyone prices it.
Working as intended, it can say: under this evidence, business model, candidate set, pricing policy and scenario model,
this structure is optimal within the declared tolerance; these judgments are perturbation-stable; these interpretations
are gold-supported or approved; these details remain unknown; the contract language matches its IR on this feasible
coverage domain; these scenario-date risks are uncovered, and these observables are worth investigating next. It never
places orders, links accounts or gives legal or investment advice; every market adapter is read-only.

## Sources of truth
- The build spec v3 and the mathematical addendum, in `spec/` (local to each checkout and copied into each workspace; not
  in git, since the repository is public). The invariants (spec §3, I1–I10) and the interfaces (spec §4, as amended by the
  addendum, accepted by Owen on November 1) are binding; module internals are the builder's call as long as the
  acceptance tests pass. Section numbers are stable and are cited as `spec §8.6`, `addendum A09`.
- `requisite/core/`: the definition. The one canonical module of the §4 types, the invariants' tables and the hashing.
  Every work package imports it; nothing defines a competing schema. Each field and rule cites its source.
- `requisite/registry.toml`: every item the sources enumerate (invariants, diagnoses, gates, post-mortem classes,
  acceptance tests, the addendum's amended tests, the oracle's checks), each covered by exactly one test.
- `spec/verify_math.py`: the numerical oracle. The production kernels reproduce its 13 results.
- `design/posture.md` (how the work is done, with reasons) and `design/decisions.md` (what was decided and why).

## Map
- `requisite/`: the product (Python 3.12, Pydantic v2, SQLite with a content-addressed blob directory). `core/` is the
  definition; the work packages (spec §13) live beside it: `evidence/`, `judgment/`, `stability/`, `propositions/`,
  `compute/`, `relevance/`, `contract/`, `fuzz/`, `optimize/`, `discovery/`, `library/`, `market/`, `ledger/`,
  `agents/`, `cli.py`, `dossier/`.
- `tests/`: pytest with hypothesis. `fixtures/`: F1–F5 data from public sources, golden outputs. `cassettes/`: recorded
  provider responses; CI and workers only replay.
- `reference/`: provider clients that already work elsewhere (OpenAI Decisions, Clef, Jev) and vendor notes.

## Commands
- Gates, the full lint, type check and test run: `gates` (`tools/gates/gates` from a checkout). One full run at a time on
  this machine. A focused run (`uv run pytest tests/<file>`) needs no turn; run `gates` before you finish.
- Use the product: `uv run requisite --help`.

## How work is done here
An outcome is finished when it is true in the running product: the CLI and the static dossier, from a clean clone,
replaying recorded provider responses. Tests are for exact limits and general properties; behaviour is shown by running.
Before a commitment that is costly to undo, re-read the purpose and your task. When something you expected does not
happen, re-read your task before continuing, since the expectation may be what was wrong.
What keeps your task's done from being true is your work, wherever in the product it is. Your report says what you did
and why, and what you saw that bears on something else the product claims, one line each. The only thing that may stand
in it as not done is a decision that is not yours or something outside your workspace's reach, and the coordinator
closes it before anything builds on it.

## Needs Owen
Deleting data, spending beyond the caps he set, releasing, contacting a real person, changing an outside account
(GitHub settings, Cloudflare, OpenAI), any change to AGENTS.md or the skills, and any change that loosens a check on the
coordinator's own work (the delegation gate, the definition review, the verify tooling, CI). The gates G1–G4 in the
product are his as the human authorizer in the demo.
