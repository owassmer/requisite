"""What construction's optimizer and resolvability fuzzing report, as committed records (addendum §12
OptimizationResult and FuzzReport; spec §8.3, §8.6; addendum A06, A09, C01)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from fractions import Fraction
from typing import Literal

from pydantic import model_validator

from requisite.core.base import Day, Money, Probability, Rational, Record, Ref
from requisite.core.contract import CashTransfer, Equivalence, ProcedureStatus, ProseSectionName
from requisite.core.judgment import SUM_TOLERANCE, JudgeConfig, top_answers

# ---------------------------------------------------------------- the optimizer (addendum A06, C01; spec §8.3)


class PricedCandidate(Record):
    """A candidate instrument as the solve saw it. Its payoff comes from its IR through the evaluator (I8). Its price is
    a listed quote where one exists, otherwise an indicative price, labelled so (spec §8.3; I9); `price_source` names
    the quote or the indicative model and its inputs."""

    ref: str
    ir: Ref
    price: Money
    pricing: Literal["quoted", "indicative"]
    price_source: str
    cap: Money


class CashSummary(Record):
    """One portfolio's cash recomputed exactly from the inputs and quantities, never read from LP slack (addendum A06
    §6.4); `breach_tolerance` is the solve's money tolerance (decisions, "Numbers")."""

    quantities: Mapping[str, Money]
    premium: Money
    cash: Sequence[Sequence[Money]]
    deficit: Sequence[Money]
    breach_tolerance: Money
    breach_dates: Sequence[Sequence[Day]]
    loss: Sequence[Money]
    receipts: Sequence[Money]
    residual: Sequence[Money]
    cvar: Rational
    expected_liquidity_shortfall: Rational
    breach_probability: Probability

    @model_validator(mode="after")
    def _signs(self) -> CashSummary:
        if any(q < 0 for q in self.quantities.values()):
            raise ValueError("quantities are nonnegative (addendum A06 §6.2: 0 <= q_j)")
        if any(d < 0 for d in self.deficit):
            raise ValueError("a deficit is D_s = max(0, ...), never negative (A06 §6.1)")
        if self.premium < 0 or self.breach_tolerance < 0:
            raise ValueError("premium and the breach tolerance are nonnegative")
        return self


class ReducedCost(Record):
    """A candidate's reduced cost against the primary duals, always beside the re-solve's measured improvement (addendum
    C01 §7.2; decisions: the duals are not unique, so a reduced cost is never shown alone)."""

    candidate: str
    value: Money
    measured_improvement: Rational


class OptimizationResult(Record):
    """A solve for one decision (addendum §12: "Candidate/price/probability hashes, exact objective, quantities,
    independently recomputed cash paths, solver tolerances, duals and pricing status"; A06 §6.4: "Store solver status,
    primary objective, secondary objective, constraint residuals, monetary scaling and tolerance"). objective is the
    decision's risk functional (spec §8.3 and its variants); recomputed_objective is that functional evaluated exactly on
    the chosen portfolio's recomputed cash, never read from the solver; the residuals are each constraint family's
    largest violation at the solver's point, in its scaled units."""

    decision: Ref
    objective: Literal["liquidity_cvar", "expected_liquidity_shortfall", "terminal_residual"]
    scenarios_hash: str
    probabilities_hash: str
    candidates: Sequence[PricedCandidate]
    dates: Sequence[Day]
    solver: str
    epsilon: Money
    scale: Money
    primary_status: str
    secondary_status: str
    primary_objective: Money
    secondary_objective: Money
    recomputed_objective: Rational
    primary_residuals: Mapping[str, float]
    secondary_residuals: Mapping[str, float]
    hedged: CashSummary
    unhedged: CashSummary
    cash_duals: Sequence[Sequence[Money]]
    budget_dual: Money
    reduced_costs: Sequence[ReducedCost]

    @model_validator(mode="after")
    def _consistent(self) -> OptimizationResult:
        refs = [c.ref for c in self.candidates]
        if len(set(refs)) != len(refs):
            raise ValueError("candidate refs are distinct")
        if set(self.hedged.quantities) != set(refs):
            raise ValueError("the hedged portfolio gives a quantity for every candidate")
        caps = {c.ref: c.cap for c in self.candidates}
        if any(q > caps[r] for r, q in self.hedged.quantities.items()):
            raise ValueError("no quantity exceeds its candidate's cap (addendum A06 §6.2: q_j <= cap_j)")
        if abs(self.recomputed_objective - Fraction(self.primary_objective)) > Fraction(self.epsilon * self.scale):
            raise ValueError("the chosen portfolio's recomputed objective lies within epsilon x scale of the optimum")
        if self.objective == "liquidity_cvar" and self.recomputed_objective != self.hedged.cvar:
            raise ValueError("for a CVaR objective, the recomputed objective is the hedged portfolio's CVaR")
        if self.objective == "expected_liquidity_shortfall" and (
            self.recomputed_objective != self.hedged.expected_liquidity_shortfall
        ):
            raise ValueError("for an expected-shortfall objective, the recomputed objective is the hedged E[D]")
        return self


# ---------------------------------------------------------------- resolvability fuzzing (addendum A09; spec §8.6)

Cause = Literal[
    "scenario_truth",
    "world_constraints",
    "rendering",
    "source_selection",
    "interpreter",
    "prose_divergence",
    "contract_ambiguity",
]


class CoverageObligation(Record):
    """One MC/DC obligation for one scenario set (addendum A09 §10.2: "Store uncovered and infeasible obligations
    explicitly"). `conditions` is one atomic condition, for its own independent effect, or a minimal group of coupled
    redundant conditions, for the group's joint effect (decisions, "Coverage": the different justified criterion A09
    asks for, so a fully coupled trigger is not "covered" by one scenario). met: the scenarios include a feasible pair
    showing the effect, named in `witnesses`. uncovered: a feasible pair exists, and the scenarios include none.
    redundant (a single condition only): no feasible pair exists, which `proof` records, with the conditions it is
    coupled to. vacuous: a finding about the whole trigger leaves no obligation to meet."""

    conditions: Sequence[str]
    status: Literal["met", "uncovered", "redundant", "vacuous"]
    witnesses: Sequence[str]
    proof: str | None
    coupled_group: Sequence[str] = ()

    @model_validator(mode="after")
    def _account(self) -> CoverageObligation:
        if not self.conditions or len(set(self.conditions)) != len(self.conditions):
            raise ValueError("an obligation is over one condition or a group of distinct conditions")
        if (self.status == "met") != (len(set(self.witnesses)) == 2 == len(self.witnesses)):
            raise ValueError("a met obligation names its witness pair, two distinct scenarios; only a met one names any")
        if (self.status == "redundant") != (self.proof is not None):
            raise ValueError("a redundant condition states its proof, and only a redundant one does")
        if self.status == "redundant" and len(self.conditions) != 1:
            raise ValueError("redundancy is a single condition's; a group's joint effect is met or uncovered")
        return self


class BoundaryPoint(Record):
    """A boundary perturbation (A09 §10.3): the condition's value at the threshold or cutoff. status says what was
    established about it, and nothing more (external evaluation, November 2: a search that failed to build a point had
    been recorded as the point being impossible, and the coverage gate then excused it):
    - feasible: some state of the contract's domain reaches it; `scenario` is the scenario built for it, if one was.
    - infeasible: no state reaches it, proved: `excluded_by` gives the proof (every minimal set of the IR's constraints
      that excludes it, or the point lying outside the values the quantity can take). Only a proof makes a point
      infeasible, and only an infeasible point is excused from coverage.
    - unconstructed: no proof excludes it and no scenario this run built reaches it; `excluded_by` says what was tried
      and why it failed. It is work not done, and the coverage gate counts it unmet."""

    condition: str
    point: str
    status: Literal["feasible", "infeasible", "unconstructed"]
    excluded_by: Sequence[str] = ()
    scenario: str | None = None

    @property
    def feasible(self) -> bool:
        return self.status == "feasible"

    @model_validator(mode="after")
    def _accounted(self) -> BoundaryPoint:
        if (self.status != "feasible") != bool(self.excluded_by):
            raise ValueError("a point not shown feasible says why (its proof, or what was tried), and a feasible one does "
                             "not")  # fmt: skip
        if self.scenario is not None and self.status != "feasible":
            raise ValueError("a scenario realizes only a feasible point")
        return self


JudgeOutcome = Literal["resolved_yes", "resolved_no", "pending", "extended", "void", "disputed", "ambiguous"]


PaymentTerm = Literal["payout_yes", "payout_no", "payment_date", "void_refund"]

# What a settled outcome pays rests on that outcome's own payment term, whether it pays something or nothing (definition
# review, November 2: what NO pays and what a void returns are terms the prose must state too, and a term of another
# outcome does not bear on a scenario: a prose that changes what YES pays does not disagree on a NO scenario).
OUTCOME_TERM: Mapping[str, PaymentTerm] = {"resolved_yes": "payout_yes", "resolved_no": "payout_no", "void": "void_refund"}
UNDETERMINED_PAY = ("ambiguous", "disputed")
TERM_READINGS = ("states", "states_otherwise", "not_stated")  # a payment term question's closed answer set  # outcomes that determine no payment: what the prose pays is not established


def terms_used(outcome: str, transfers: Sequence[CashTransfer] | None) -> tuple[PaymentTerm, ...]:
    """The payment terms the prose must be read to state before the evaluator's transfers for this outcome stand for what
    the prose pays: the outcome's own term (OUTCOME_TERM); and the payment date wherever anything is paid, wherever it
    is not known whether anything is (None), and for a pending or extended outcome, whose paying nothing yet is the
    payment date's own statement that payment follows the outcome's determination (definition review, November 2: an
    interim payment the prose states would otherwise go unread)."""
    own = (OUTCOME_TERM[outcome],) if outcome in OUTCOME_TERM else ()
    dated = transfers is None or bool(transfers) or outcome in ("pending", "extended")
    return own + (("payment_date",) if dated else ())


class TermReading(Record):
    """What the resolvers read the prose to state about one of the IR's payment terms (addendum A09 §10.3: prose and IR
    are compared on their dated payments, not only their procedural outcome). term names it: payout_yes (what one
    contract pays when it resolves YES), payout_no (what it pays when it resolves NO), payment_date (when the outcome is
    determined and paid), void_refund (what a void returns, where the contract can be void). statement is the term as
    code states it from the IR (requisite.fuzz.statements.payment_terms); answers is the resolvers' pooled distribution
    over states (the prose states exactly this term), states_otherwise (it states the term differently) and not_stated;
    read is the pooled top, ambiguous where the top is tied or no reader answered (core.top_answers); answers are a
    distribution over TERM_READINGS, or empty where no reader answered; judgments are the content hashes of the judgments read. A scenario's prose transfers are
    the evaluator's only where every term its outcome rests on is read as "states" (external evaluation, November 2:
    the prose's payment had been taken to be the IR's whenever the outcomes agreed, so a prose paying another amount on
    another date passed). A reading of the whole rulebook establishes a scenario's payment only that way: the prose
    states exactly the IR's term, and the outcome agrees, so the one settlement function gives the prose's transfers.
    It never establishes that a scenario's payment differs, since the prose may state the term otherwise only where it
    does not apply to that scenario (an exception, a rate on a zero quantity, a different split of the lags): a term read
    otherwise leaves the payment unestablished (definition review, November 2). Comparing each scenario's payments
    themselves needs the readers to state what the prose pays on that scenario (A09 §10.1), which this reading is not."""

    term: PaymentTerm
    statement: str
    answers: Mapping[str, float]
    read: Literal["states", "states_otherwise", "not_stated", "ambiguous"]
    judgments: Sequence[str]

    @model_validator(mode="after")
    def _read(self) -> TermReading:
        if self.answers:
            values = list(self.answers.values())
            if set(self.answers) - set(TERM_READINGS):
                raise ValueError(f"a term's answers are among {TERM_READINGS}, not {sorted(set(self.answers))}")
            if not all(math.isfinite(v) and v >= 0 for v in values) or abs(math.fsum(values) - 1) > SUM_TOLERANCE:
                raise ValueError("a term's pooled answers are a distribution: finite, nonnegative, summing to one")
        top = top_answers(self.answers)
        want = next(iter(top)) if len(top) == 1 else "ambiguous"
        if self.read != want:
            raise ValueError(f"a term's reading is its pooled answers' top ({want}), ambiguous where tied or none")
        return self


class FuzzScenario(Record):
    """One scenario's three paths (A09 §10.1): the IR's outcome and transfers, the rendering check, and the resolvers'
    reading of the prose. `split` keeps held-out results apart (A09 §10.4; A01 §2.5). `ambiguous` is the resolvers'
    interpretation result, never the contract's VOID or EXTENDED (A05 §5.1, A09 §10.3). Both the procedural outcomes
    and the dated transfers they produce are compared (A09 §10.3): prose_outcome is the resolvers' outcome under the
    fuzz run's pooling policy. Where the resolvers read the prose to state every payment term the prose outcome rests on
    (FuzzReport.terms; terms_used), prose_transfers are the per-unit transfers the evaluator gives for the prose
    outcome, the one settlement function applied to terms the prose was read to state;
    where it did not, where the outcome determines no payment (ambiguous, disputed), or where the payment's amount is
    not yet determined (an unknown quantity: A05 §5.3 keeps it unknown until the missing-data procedure resolves it),
    what the prose pays is not established and prose_transfers is None: never an empty list, which says the prose pays
    nothing (spec I6: an unknown is not zero; definition review, November 2). Paying nothing is a payment term too
    (terms_used). amount_error
    |sum_t h_prose - sum_t h_IR| and liquidity_discrepancy the dated liquidity-impact difference computed by the same
    business cash engine (A09 §10.4), each None where the scenario defines no quantity or the prose's payment is not
    established. state_hash identifies the scenario's abstract state, so two
    scenarios over one state are one state of the domain."""

    id: str
    state_hash: str
    packet: str
    split: Literal["development", "held_out"]
    mandatory: bool
    severity: Literal["critical", "material", "minor"]
    ir_outcome: ProcedureStatus
    ir_evaluation: str
    ir_transfers: Sequence[CashTransfer]
    rendering_passed: bool
    resolver_outcomes: Mapping[str, float]
    prose_outcome: JudgeOutcome
    prose_transfers: Sequence[CashTransfer] | None
    amount_error: Money | None
    liquidity_discrepancy: Money | None

    @model_validator(mode="after")
    def _payment(self) -> FuzzScenario:
        if self.prose_transfers is None and (self.amount_error is not None or self.liquidity_discrepancy is not None):
            raise ValueError("a payment the prose was not read to state has no discrepancy computed")
        if self.prose_outcome in UNDETERMINED_PAY and self.prose_transfers is not None:
            raise ValueError(f"a {self.prose_outcome} outcome determines no payment: its prose transfers are None")
        return self


class Attribution(Record):
    """A hypothesis about why a scenario's resolvers diverged from the IR; it stays open until a minimal counterexample
    or human gold checks it (A09 §10.1; registry A.T18). suspected_clauses ranks the rulebook's sections the failing
    scenario exercises (spec §8.6 step 7: "Each scenario is tagged with the clauses and conditions it exercises, so a
    failure points to a clause"), each with a score of suspicion over the scored scenarios, a ranking that is not a
    finding (the Ochiai coefficient: ours, design/decisions.md, "Resolvability fuzzing"); empty where the cause is not
    the contract's wording."""

    scenario: str
    cause: Cause
    status: Literal["open", "confirmed", "refuted"]
    evidence: str
    suspected_clauses: Sequence[tuple[ProseSectionName, float]] = ()  # ranked, each with its suspiciousness score


class HardGate(Record):
    name: str
    passed: bool
    detail: str


def same_transfers(scenario: FuzzScenario) -> bool | None:
    """Whether prose and IR give the same dated transfers: the same total amount at each instant, in each currency and
    basis, however the transfers are listed or split (A09 §10.3 compares the transfers' cash, not their serialization;
    ours: two payments of 50 at one instant are one payment of 100 to the holder). None where what the prose pays is
    not established (FuzzScenario.prose_transfers): neither the same nor different."""
    if scenario.prose_transfers is None:
        return None

    def totals(transfers: Sequence[CashTransfer]) -> dict[tuple[datetime, str, str], Fraction]:
        out: dict[tuple[datetime, str, str], Fraction] = {}
        for t in transfers:
            key = (t.at.astimezone(UTC), t.currency, t.basis)  # the instant, whatever offset it is written in
            out[key] = out.get(key, Fraction(0)) + Fraction(t.amount)  # exact: Decimal sums round at 28 digits
        return {k: v for k, v in out.items() if v != 0}

    return totals(scenario.prose_transfers) == totals(scenario.ir_transfers)


HARD_GATES = (
    "no_unreviewed_critical_mismatch",
    "no_silent_source_or_missing_data_substitution",
    "mandatory_coverage_met_or_accepted",
    "accepted_ambiguity_scoped",
    "held_out_reported_separately",
)  # addendum A09 §10.4's five hard gates, in its order


class FuzzReport(Record):
    """A fuzz run of one contract version's prose against its IR (spec §8.6; amended by addendum §12 and A09).
    prose is the clause proposal whose rulebook was fuzzed (core.Rulebook; spec I1) and prose_hash its content's hash;
    resolvers are the K configurations that read it (spec §8.6 step 5: "K configurations (providers × paraphrases of the
    resolver instruction) answer: under these rules and this evidence"). pooling_policy is how the resolvers' outcomes were pooled into each scenario's prose outcome (A01 §2.3). The hard
    gates are A09 §10.4's, every one reported. equivalence is A09 §10.5's scope of the claim, and it must be the one the
    scenarios support: a counterexample where prose and IR disagree on any scenario, a tested equivalence only over a
    suite on which they all agree, an exhaustive one only over the finite domain `domain` describes, whose every state is
    listed in `domain_states` (by state hash) and run. terms are the IR's payment terms as the resolvers read the prose
    to state them (TermReading), so a scenario's prose transfers are what the prose was read to pay."""

    contract: Ref
    prose: str
    prose_hash: str
    resolvers: Sequence[JudgeConfig]
    pooling_policy: Ref
    domain: str | None = None
    domain_states: Sequence[str] = ()
    whole_trigger: Literal["ok", "no_feasible_state", "always_fires", "never_fires"]
    terms: Sequence[TermReading]
    obligations: Sequence[CoverageObligation]
    boundaries: Sequence[BoundaryPoint]
    scenarios: Sequence[FuzzScenario]
    fidelity: float
    clarity: float
    ambiguity: float
    hard_gates: Sequence[HardGate]
    attributions: Sequence[Attribution]
    equivalence: Equivalence

    @model_validator(mode="after")
    def _claims(self) -> FuzzReport:
        if tuple(g.name for g in self.hard_gates) != HARD_GATES:
            raise ValueError(f"a fuzz report reports every hard gate of A09 §10.4, in order: {HARD_GATES}")
        # A scenario whose packet failed the rendering check says nothing about the contract: the rendering is at fault
        # and is fixed (spec §8.6 step 4). Over the rest, prose and IR agree on a scenario when both its procedural
        # outcome and its dated transfers agree (A09 §10.3).
        read = {t.term: t.read for t in self.terms}  # a term with no reading is not read as stated
        if len(read) != len(self.terms):
            raise ValueError("each payment term is read once")
        for s in self.scenarios:
            unread = [term for term in terms_used(s.prose_outcome, s.prose_transfers) if read.get(term) != "states"]
            if s.prose_transfers is not None and unread:
                raise ValueError(f"scenario {s.id}'s prose transfers rest on terms not read as stated: {unread}")
        rendered = [s for s in self.scenarios if s.rendering_passed]
        disagree = [s.id for s in rendered if s.prose_outcome != s.ir_outcome or same_transfers(s) is False]
        unestablished = [s.id for s in rendered if s.id not in disagree and same_transfers(s) is None]
        claims_equivalence = self.equivalence in (
            "tested_equivalent_on_suite", "exhaustive_equivalent_on_declared_finite_domain"
        )  # fmt: skip
        if claims_equivalence and len(rendered) != len(self.scenarios):
            raise ValueError("an equivalence holds only over scenarios whose rendering passed; a failed one is fixed")
        if claims_equivalence and unestablished:
            raise ValueError(f"an equivalence needs what the prose pays read on every scenario; not on {unestablished}")
        if disagree and self.equivalence != "counterexample_found":
            raise ValueError(f"prose and IR disagree on {disagree}: the equivalence is a counterexample")
        if self.equivalence == "counterexample_found" and not disagree:
            raise ValueError("a counterexample is a scenario on which prose and IR disagree")
        if self.equivalence == "tested_equivalent_on_suite" and not self.scenarios:
            raise ValueError("a tested equivalence is over a suite of scenarios")
        if self.equivalence == "exhaustive_equivalent_on_declared_finite_domain":
            # Every state of the declared finite domain was run (A09 §10.5: sampled fuzzing proves no more than it ran).
            ran = {s.state_hash for s in self.scenarios}
            if not self.domain or not self.domain_states or ran != set(self.domain_states):
                raise ValueError("an exhaustive equivalence declares its finite domain, every state of it, and runs "
                                 "exactly those states")  # fmt: skip
        elif self.domain_states and not self.domain:
            raise ValueError("domain states belong to a declared domain")
        return self
