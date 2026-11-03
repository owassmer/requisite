"""One settlement function (spec I8, never cut): the same evaluator computes payoffs in construction, example outcomes in
drafting, ground truth in fuzzing and settlement at resolution. A published example carries the evaluator's own outcome
for its scenario, and every module of the product that turns a contract and a scenario into an outcome or a payoff does
it by calling requisite.contract.evaluate.evaluate: checked over the source, as tests/store/test_store.py checks the
database boundary."""

import ast
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import pytest
import western_grazers as wg

from requisite import core as c
from requisite.contract.evaluate import check_example, evaluate, example
from requisite.core.kleene import Kleene

ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = "requisite.contract.evaluate"
CALLS = {"evaluate", "example", "check_example"}  # example and check_example call evaluate
DEFINITION = {"requisite.core", "requisite.core.contract", "requisite.core.kleene"}
OUTCOMES = {"ResolutionEvaluation", "CashTransfer", "ScenarioExample"}  # what only the evaluator constructs
EXPRESSIONS = {"Cmp", "And", "Or", "Not", "By", "At", "Throughout", "Interpretive", "CountTrue", "DurationIn", "NumCmp",
               "ReadValue", "ValueAt", "Computed"}  # fmt: skip
COMPOSITION = {("requisite.core.kleene", n) for n in ("conj", "disj", "neg")} | {("requisite.core", n) for n in (
    "conj", "disj", "neg")} | {("requisite.contract.counts", "count_true")}  # fmt: skip
SCENARIO = {"Observation", "SyntheticObservation"}


def module_name(path: Path, root: Path) -> str:
    parts = path.relative_to(root).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def settlement_violations(path: Path, name: str) -> list[str]:
    """What a module does that only the settlement function may do. Names are resolved through the module's imports, so
    an alias, a module attribute (c.CashTransfer) or another module's `evaluate` is read as what it is."""
    tree = ast.parse(path.read_text())
    package = name.rsplit(".", 1)[0] if not path.name == "__init__.py" else name
    names: dict[str, tuple[str, str | None]] = {}  # local name -> (module, attribute or None for a module)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names[a.asname or a.name.split(".")[0]] = (a.name if a.asname else a.name.split(".")[0], None)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = ".".join(package.split(".")[: len(package.split(".")) - node.level + 1] + ([base] if base else []))
            for a in node.names:
                names[a.asname or a.name] = (base, a.name)

    def resolve(expr: ast.expr) -> tuple[str, str] | None:
        """(module, attribute) an expression names: from x import y; import x as m then m.y; from p import m then m.y."""
        if isinstance(expr, ast.Name) and expr.id in names:
            module, attr = names[expr.id]
            return (module, attr) if attr is not None else None
        if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name) and expr.value.id in names:
            module, attr = names[expr.value.id]
            return (module if attr is None else f"{module}.{attr}", expr.attr)
        return None

    used: set[tuple[str, str]] = set()
    calls: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name | ast.Attribute) and (r := resolve(node)) is not None:
            used.add(r)
        if isinstance(node, ast.Call):
            func = node.func
            target = resolve(func)
            if isinstance(func, ast.Attribute) and func.attr in ("model_validate", "model_construct"):
                target = resolve(func.value)  # CashTransfer.model_validate(...), c.CashTransfer.model_construct(...)
            if target is not None:
                calls.add(target)
    defined = {attr for module, attr in used if module in DEFINITION}
    out = []
    forged = sorted({attr for module, attr in calls if module in DEFINITION and attr in OUTCOMES})
    if forged:
        out.append(f"{name} constructs {forged}: an outcome or a payoff is the settlement function's output")
    if defined & EXPRESSIONS and used & COMPOSITION:
        out.append(f"{name} composes three-valued truths or counts over IR expressions: a second settlement function")
    calls_evaluator = any(module == EVALUATOR and attr in CALLS for module, attr in calls)
    if "ContractIR" in defined and defined & SCENARIO and not calls_evaluator:
        out.append(f"{name} holds a contract and a scenario's observations but never calls {EVALUATOR}.evaluate")
    imports_evaluator = any(module == EVALUATOR or (module, attr) == ("requisite.contract", "evaluate")
                            for module, attr in names.values())  # fmt: skip
    if imports_evaluator and not calls_evaluator:
        out.append(f"{name} imports the settlement function and does not call it")
    return out


def product_violations(root: Path) -> list[str]:
    out = []
    for path in sorted(root.rglob("*.py")):
        name = module_name(path, root.parent)
        if name.startswith("requisite.core") or name == EVALUATOR:
            continue  # the definition declares the records; the evaluator is the one that settles
        out += settlement_violations(path, name)
    return out


PLANTED = {
    "second_evaluator": """
        from requisite.core import ContractIR, Interpretive, walk
        from requisite.core.kleene import Kleene, conj

        def settle(ir: ContractIR, truths):
            return conj(truths[n.id] for n in walk(ir.trigger) if isinstance(n, Interpretive))
        """,
    "forged": """
        from requisite import core as c

        def payoff(ir, when):
            return c.CashTransfer(at=when, amount=ir.payout.unit_value, currency="USD", basis="payout")
        """,
    "rebuilt": """
        from requisite.core.contract import ResolutionEvaluation

        def load(data):
            return ResolutionEvaluation.model_validate(data)
        """,
    "wrong_evaluate": """
        from requisite.core import ContractIR, SyntheticObservation
        from requisite.optimize import evaluate

        def outcome(ir: ContractIR, scenario: list[SyntheticObservation]):
            return evaluate(ir, scenario)
        """,
    "compliant": """
        from requisite.contract.evaluate import evaluate
        from requisite.core import ContractIR, SyntheticObservation

        def outcome(ir: ContractIR, scenario: list[SyntheticObservation], truths, at):
            return evaluate(ir, scenario, truths, at=at).procedure_status
        """,
    "through_the_module": """
        from requisite import core
        from requisite.contract import evaluate as settlement

        def ground_truth(ir: core.ContractIR, scenario: list[core.Observation], truths, at):
            return settlement.evaluate(ir, scenario, truths, at=at)
        """,
    "prose": """
        from requisite.core import ContractIR, Interpretive, walk

        def conditions(ir: ContractIR) -> list[str]:
            return [n.question_template for n in walk(ir.trigger) if isinstance(n, Interpretive)]
        """,
    "unused": """
        from requisite.contract.evaluate import evaluate

        def nothing():
            return None
        """,
}


def test_the_source_check_finds_each_way_around_the_settlement_function(tmp_path):
    pkg = tmp_path / "planted"
    pkg.mkdir()
    for name, body in PLANTED.items():
        (pkg / f"{name}.py").write_text(textwrap.dedent(body))
    found = {name: settlement_violations(pkg / f"{name}.py", f"planted.{name}") for name in PLANTED}
    assert [v.split(" ", 1)[1] for v in found["second_evaluator"]] == [
        "composes three-valued truths or counts over IR expressions: a second settlement function"]
    assert "constructs ['CashTransfer']" in found["forged"][0]
    assert "constructs ['ResolutionEvaluation']" in found["rebuilt"][0]
    assert found["wrong_evaluate"] == [
        "planted.wrong_evaluate holds a contract and a scenario's observations but never calls "
        "requisite.contract.evaluate.evaluate"]  # the optimizer's evaluate is not the settlement function
    assert found["compliant"] == [] and found["through_the_module"] == [] and found["prose"] == []
    assert found["unused"] == ["planted.unused imports the settlement function and does not call it"]


@pytest.mark.covers("I8")
def test_one_settlement_function_gives_every_outcome_and_every_examples_outcome():
    # Published examples carry the evaluator's own outcome for their scenario (spec §8.5): each is made by the evaluator,
    # names the evaluation that gave it, and is checked against a fresh evaluation of its scenario.
    scenarios = {
        "ab187-at-settlement": (wg.rates(), wg.ab187(), wg.SETTLED),
        "not-established-before-expiry": (wg.rates(), wg.ab187(Kleene.UNKNOWN), datetime(2026, 9, 10, tzinfo=UTC)),
        "not-established-at-expiry": (wg.rates(), wg.ab187(Kleene.UNKNOWN), wg.EXPIRATION),
    }
    base = wg.ir(wg.AB187)
    examples = tuple(example(base, sid, sid, *inputs[:2], at=inputs[2]) for sid, inputs in scenarios.items())
    assert [e.expected for e in examples] == ["resolved_yes", "pending", "resolved_no"]
    published = wg.ir(wg.AB187, examples=examples)
    for e in published.examples:
        observations, interpretations, at = scenarios[e.scenario_id]
        ev = check_example(published, e, observations, interpretations, at=at)
        assert ev.procedure_status == e.expected and ev.content_hash() == e.evaluation
        assert evaluate(published, observations, interpretations, at=at) == ev
    forged = examples[0].model_copy(update={"expected": "resolved_no"})
    with pytest.raises(ValueError, match="states resolved_no; the evaluator gives resolved_yes"):
        check_example(published, forged, *scenarios["ab187-at-settlement"][:2], at=wg.SETTLED)
    elsewhere = examples[1].model_copy(update={"scenario_id": "ab187-at-settlement"})  # pending, from another scenario
    with pytest.raises(ValueError, match="states pending"):
        check_example(published, elsewhere, *scenarios["ab187-at-settlement"][:2], at=wg.SETTLED)
    assert isinstance(examples[0], c.ScenarioExample)

    # And no module of the product turns a contract and a scenario into an outcome or a payoff any other way.
    assert product_violations(ROOT / "requisite") == []
