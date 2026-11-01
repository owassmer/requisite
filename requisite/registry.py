"""The registry of what the sources enumerate (registry.toml), and which test covers each item.

A test declares the items it is the one test for with ``@pytest.mark.covers("<id>", ...)``. Coverage is read from the
test files' syntax (never by running them), so the board is the same whichever tests a run selects.
"""

from __future__ import annotations

import ast
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = Path(__file__).resolve().parent / "registry.toml"


@dataclass(frozen=True)
class Item:
    id: str
    source: str
    says: str
    tier: int
    never_cut: bool = False
    replaced_by: str | None = None


def items(path: Path = REGISTRY) -> dict[str, Item]:
    raw = tomllib.loads(path.read_text())["item"]
    out: dict[str, Item] = {}
    for r in raw:
        item = Item(**r)
        if item.id in out:
            raise ValueError(f"registry: {item.id} is listed twice")
        out[item.id] = item
    return out


def _covers(decorator: ast.expr) -> list[str] | None:
    """The ids of a ``pytest.mark.covers(...)`` decorator, or None for any other decorator."""
    if not isinstance(decorator, ast.Call):
        return None
    f = decorator.func
    if not (isinstance(f, ast.Attribute) and f.attr == "covers"):
        return None
    ids = []
    for a in decorator.args:
        if not (isinstance(a, ast.Constant) and isinstance(a.value, str)):
            raise ValueError(f"covers takes string literals, line {decorator.lineno}")
        ids.append(a.value)
    return ids


def coverage(tests: Path = ROOT / "tests") -> dict[str, list[str]]:
    """Each covered id, with the tests (file::name) that declare it."""
    found: dict[str, list[str]] = {}
    for file in sorted(tests.rglob("test_*.py")):
        tree = ast.parse(file.read_text(), filename=str(file))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                for d in node.decorator_list:
                    ids = _covers(d)
                    for i in ids or []:
                        found.setdefault(i, []).append(f"{file.relative_to(ROOT)}::{node.name}")
    return found


def problems(reg: dict[str, Item], cov: dict[str, list[str]]) -> list[str]:
    """What makes the registry and the tests disagree: an unknown id, an item covered twice, a replaced item covered,
    a replacement that is not itself an item."""
    out = []
    for i, where in cov.items():
        if i not in reg:
            out.append(f"{i}: covered by {', '.join(where)} but not in the registry")
        elif len(where) > 1:
            out.append(f"{i}: covered by {len(where)} tests ({', '.join(where)}); exactly one test covers an item")
        elif reg[i].replaced_by:
            out.append(f"{i}: replaced by {reg[i].replaced_by} (an accepted amendment), yet covered by {where[0]}")
    for item in reg.values():
        if item.replaced_by and item.replaced_by not in reg:
            out.append(f"{item.id}: replaced by {item.replaced_by}, which is not in the registry")
    return out


def board(reg: dict[str, Item], cov: dict[str, list[str]]) -> str:
    """Where the registry stands, by tier, never-cut items first: covered, open, replaced."""
    lines = []
    for tier in sorted({i.tier for i in reg.values()}):
        tiered = [i for i in reg.values() if i.tier == tier]
        live = [i for i in tiered if not i.replaced_by]
        done = [i for i in live if i.id in cov]
        lines.append(f"Tier {tier}: {len(done)} of {len(live)} covered")
        for item in sorted(live, key=lambda i: (not i.never_cut, i.id in cov, i.id)):
            mark = "covered" if item.id in cov else "OPEN"
            cut = " [never cut]" if item.never_cut else ""
            where = f"  ({cov[item.id][0]})" if item.id in cov else ""
            lines.append(f"  {mark:7} {item.id}{cut}: {item.says[:110]}{where}")
        for item in (i for i in tiered if i.replaced_by):
            lines.append(f"  replaced {item.id} -> {item.replaced_by}")
    return "\n".join(lines)
