"""What construction's optimizer and resolvability fuzzing report, as committed records (addendum §12
OptimizationResult and FuzzReport; spec §8.3, §8.6; addendum A06, A09, C01)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from fractions import Fraction
from typing import Literal

from pydantic import model_validator

from requisite.core.base import Day, Money, Probability, Rational, Record, Ref
from requisite.core.contract import CashTransfer, Equivalence, ProcedureStatus

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
    """A boundary perturbation (A09 §10.3): the condition's value at the threshold or cutoff, feasible or excluded by
    the named constraints."""

    condition: str
    point: str
    feasible: bool
    excluded_by: Sequence[str] = ()
    scenario: str | None = None


JudgeOutcome = Literal["resolved_yes", "resolved_no", "pending", "extended", "void", "disputed", "ambiguous"]


class FuzzScenario(Record):
    """One scenario's three paths (A09 §10.1): the IR's outcome and transfers, the rendering check, and the resolvers'
    reading of the prose. `split` keeps held-out results apart (A09 §10.4; A01 §2.5). `ambiguous` is the resolvers'
    interpretation result, never the contract's VOID or EXTENDED (A05 §5.1, A09 §10.3). Both the procedural outcomes
    and the dated transfers they produce are compared (A09 §10.3): prose_outcome is the resolvers' outcome under the
    fuzz run's pooling policy, prose_transfers the per-unit transfers the evaluator gives for it, amount_error
    |sum_t h_prose - sum_t h_IR| and liquidity_discrepancy the dated liquidity-impact difference computed by the same
    business cash engine (A09 §10.4), each None where the scenario defines no quantity. state_hash identifies the
    scenario's abstract state, so two scenarios over one state are one state of the domain."""

    id: str
    state_hash: str
    split: Literal["development", "held_out"]
    mandatory: bool
    severity: Literal["critical", "material", "minor"]
    ir_outcome: ProcedureStatus
    ir_evaluation: str
    ir_transfers: Sequence[CashTransfer]
    rendering_passed: bool
    resolver_outcomes: Mapping[str, float]
    prose_outcome: JudgeOutcome
    prose_transfers: Sequence[CashTransfer]
    amount_error: Money | None
    liquidity_discrepancy: Money | None


class Attribution(Record):
    """A hypothesis about why a scenario's resolvers diverged from the IR; it stays open until a minimal counterexample
    or human gold checks it (A09 §10.1; registry A.T18)."""

    scenario: str
    cause: Cause
    status: Literal["open", "confirmed", "refuted"]
    evidence: str


class HardGate(Record):
    name: str
    passed: bool
    detail: str


def same_transfers(scenario: FuzzScenario) -> bool:
    """Whether prose and IR give the same dated transfers: the same total amount at each instant, in each currency and
    basis, however the transfers are listed or split (A09 §10.3 compares the transfers' cash, not their serialization;
    ours: two payments of 50 at one instant are one payment of 100 to the holder)."""

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
    pooling_policy is how the resolvers' outcomes were pooled into each scenario's prose outcome (A01 §2.3). The hard
    gates are A09 §10.4's, every one reported. equivalence is A09 §10.5's scope of the claim, and it must be the one the
    scenarios support: a counterexample where prose and IR disagree on any scenario, a tested equivalence only over a
    suite on which they all agree, an exhaustive one only over the finite domain `domain` describes, whose every state is
    listed in `domain_states` (by state hash) and run."""

    contract: Ref
    prose_hash: str
    pooling_policy: Ref
    domain: str | None = None
    domain_states: Sequence[str] = ()
    whole_trigger: Literal["ok", "no_feasible_state", "always_fires", "never_fires"]
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
        rendered = [s for s in self.scenarios if s.rendering_passed]
        disagree = [s.id for s in rendered if s.prose_outcome != s.ir_outcome or not same_transfers(s)]
        claims_equivalence = self.equivalence in (
            "tested_equivalent_on_suite", "exhaustive_equivalent_on_declared_finite_domain"
        )  # fmt: skip
        if claims_equivalence and len(rendered) != len(self.scenarios):
            raise ValueError("an equivalence holds only over scenarios whose rendering passed; a failed one is fixed")
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
