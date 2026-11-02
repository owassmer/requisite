"""What construction's optimizer and resolvability fuzzing report, as committed records (addendum §12
OptimizationResult and FuzzReport; spec §8.3, §8.6; addendum A06, A09, C01)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    """One atomic condition's MC/DC obligation for one scenario set (addendum A09 §10.2: "Store uncovered and infeasible
    obligations explicitly"). met: the scenarios include a feasible unique-cause pair for it, named in `witnesses`.
    uncovered: a feasible pair exists, and the scenarios do not include one. redundant: no feasible pair exists, which
    `proof` records, with the conditions it is coupled to. vacuous: a finding about the whole trigger (no feasible
    state, or a constant trigger) leaves the condition no obligation to meet."""

    condition: str
    status: Literal["met", "uncovered", "redundant", "vacuous"]
    witnesses: Sequence[str]
    proof: str | None
    coupled_group: Sequence[str] = ()

    @model_validator(mode="after")
    def _account(self) -> CoverageObligation:
        if (self.status == "met") != (len(set(self.witnesses)) == 2 == len(self.witnesses)):
            raise ValueError("a met obligation names its witness pair, two distinct scenarios; only a met one names any")
        if (self.status == "redundant") != (self.proof is not None):
            raise ValueError("a redundant condition states its proof, and only a redundant one does")
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
    business cash engine (A09 §10.4), each None where the scenario defines no quantity."""

    id: str
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


class FuzzReport(Record):
    """A fuzz run of one contract version's prose against its IR (spec §8.6; amended by addendum §12 and A09)."""

    contract: Ref
    prose_hash: str
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
