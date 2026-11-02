"""I7's static check: a node, registered or built, reaches no provider, cassette or network code, and is pure (spec §3 I7: "Code
owns consequences"; spec §13 WP6: "a static check forbids provider calls inside nodes").

The check reads source, never runs it. From each module a node lives in, it follows every import of the product,
transitively, including imports inside functions and the packages an import executes, and refuses:
- a product module whose job is input or output: providers, recorded responses, keys, agent runs, and the store, since
  committed state reaches a node only as a source of the graph (requisite/compute/runtime.py), never read from elsewhere;
- a network, process or database module, or a provider SDK, imported directly;
- a clock, randomness, the environment, a file opened by the builtin `open`, and the means of a dynamic import
  (`importlib`, `__import__`, `eval`, `exec`, `compile`), since a node's value is fixed by its inputs and the walk cannot
  follow what is imported by name at run time.
The same rules hold in every module the walk reaches, not only the node's own: a node's function runs the functions it
imports, and the walk sees modules, not calls.

What a static walk cannot see: an import made by a route it does not name (`getattr` on a module object, a library that
imports on the node's behalf); what third-party packages do inside, since the walk enters only the product; a file opened
by path through a library (`numpy.load`, a template loader); and a callable or client passed to the node as an input.
The last is refused at run time instead: an input has a canonical value hash, which a function, a client or an open
file does not have (runtime.value_hash). Mutation of an input is prevented the same way, since the runtime freezes
values.
"""

from __future__ import annotations

import ast
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

PRODUCT = Path(__file__).resolve().parents[1]  # requisite/
ROOTS: Mapping[str, Path] = {"requisite": PRODUCT}
# How a node comes to exist: registered with @node, or built directly as a Node and handed to a runtime as its lookup.
REGISTER = frozenset({"requisite.compute.registry.node", "requisite.compute.registry.Node"})

# Product modules (with everything under them) whose job is input or output, and why a node may not reach them.
PRODUCT_IO: Mapping[str, str] = {
    "requisite.judgment.providers": "provider calls",
    "requisite.cassettes": "recorded provider responses",
    "requisite.secrets": "keys",
    "requisite.agents": "agent runs",
    "requisite.store": "committed state, which reaches a node only as a source of the graph",
}

# External modules (with their submodules) a node may not import, by what they do.
MODULES: Mapping[str, str] = {
    **dict.fromkeys(["httpx", "requests", "urllib", "urllib3", "http", "socket", "ssl", "aiohttp", "websockets"], "network"),
    **dict.fromkeys(["subprocess", "multiprocessing", "os", "sys"], "process and environment"),
    **dict.fromkeys(["openai", "anthropic", "cloudflare"], "provider SDK"),
    "sqlite3": "database",
    "time": "clock",
    **dict.fromkeys(["random", "secrets", "uuid", "numpy.random"], "randomness"),
    **dict.fromkeys(["pathlib", "io", "shutil", "tempfile", "glob"], "files"),
    **dict.fromkeys(["importlib", "builtins"], "dynamic import"),  # builtins.open, builtins.eval by attribute
}
BUILTINS: Mapping[str, str] = {
    "open": "files",
    "input": "process and environment",
    **dict.fromkeys(["__import__", "eval", "exec", "compile"], "dynamic import"),
}
CLOCK_ATTRIBUTES = frozenset({"now", "utcnow", "today"})  # datetime.now(), date.today(), on a class or an instance


@dataclass(frozen=True)
class Violation:
    module: str
    line: int
    rule: str
    detail: str
    via: tuple[str, ...]  # the import path from the node's module to `module`

    def __str__(self) -> str:
        path = " -> ".join(self.via)
        return f"{self.module}:{self.line}: {self.rule}: {self.detail} (reached by {path})"


def _module_file(name: str, roots: Mapping[str, Path]) -> Path | None:
    top, *rest = name.split(".")
    if top not in roots:
        return None
    base = roots[top].joinpath(*rest)
    for path in (base.with_suffix(".py") if rest else None, base / "__init__.py"):
        if path is not None and path.is_file():
            return path
    return None


def _module_name(path: Path, roots: Mapping[str, Path]) -> str:
    for top, root in roots.items():
        if path.is_relative_to(root):
            parts = [top, *path.relative_to(root).with_suffix("").parts]
            return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
    raise ValueError(f"{path} is under none of {dict(roots)}")


def _matches(name: str, table: Mapping[str, str]) -> str | None:
    for prefix, what in table.items():
        if name == prefix or name.startswith(prefix + "."):
            return what
    return None


def _imports(tree: ast.Module, module: str, package: bool) -> Iterator[tuple[str, int, str | None]]:
    """Each name an import statement brings in: (dotted module or attribute, line, the local alias it binds or None)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname is not None:
                    yield a.name, node.lineno, a.asname
                else:  # `import a.b.c` imports a.b.c and binds `a` to the package a
                    yield a.name, node.lineno, None
                    yield a.name.split(".")[0], node.lineno, a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = module.split(".")
                anchor = parts if package else parts[:-1]
                anchor = anchor[: len(anchor) - (node.level - 1)]
                base = ".".join([*anchor, base] if base else anchor)
            yield base, node.lineno, None
            for a in node.names:
                if a.name != "*":
                    yield f"{base}.{a.name}", node.lineno, a.asname or a.name


def _dotted(node: ast.expr, aliases: Mapping[str, str]) -> str | None:
    attrs: list[str] = []
    while isinstance(node, ast.Attribute):
        attrs.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name) and node.id in aliases:
        return ".".join([aliases[node.id], *reversed(attrs)])
    return None


def _scan(path: Path, module: str, roots: Mapping[str, Path]) -> tuple[list[tuple[int, str, str]], list[str]]:
    """One module's own violations (line, rule, detail) and the product modules it imports."""
    tree = ast.parse(path.read_text(), filename=str(path))
    package = path.name == "__init__.py"
    found: list[tuple[int, str, str]] = []
    product: list[str] = []
    aliases: dict[str, str] = {}
    for name, line, alias in _imports(tree, module, package):
        if alias is not None:
            aliases[alias] = name
        top = name.split(".")[0]
        if top in roots:
            if (what := _matches(name, PRODUCT_IO)) is not None:
                found.append((line, what, f"imports {name}"))
            # an import executes the packages above it
            parts = name.split(".")
            product += [".".join(parts[:i]) for i in range(1, len(parts) + 1)]
        elif (what := _matches(name, MODULES)) is not None:
            found.append((line, what, f"imports {name}"))
    bound = _bound(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in BUILTINS and node.id not in bound:
            found.append((node.lineno, BUILTINS[node.id], f"uses {node.id}"))
        elif isinstance(node, ast.Attribute):
            if node.attr in CLOCK_ATTRIBUTES:
                found.append((node.lineno, "clock", f"reads .{node.attr}"))
            dotted = _dotted(node, aliases)
            if dotted is not None and (what := _matches(dotted, MODULES)) is not None:
                found.append((node.lineno, what, f"uses {dotted}"))
    first = {(line, rule): detail for line, rule, detail in reversed(found)}  # one report per rule and line
    return sorted((line, rule, detail) for (line, rule), detail in first.items()), [
        m for m in dict.fromkeys(product) if _module_file(m, roots) is not None
    ]


def _bound(tree: ast.Module) -> set[str]:
    """The names a module binds itself (assignments, parameters, definitions, imports), which shadow a builtin."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
    return names


def check(modules: Iterable[str], roots: Mapping[str, Path] = ROOTS) -> list[Violation]:
    """Every violation in the given node modules and in everything of the product they import, transitively."""
    out: list[Violation] = []
    seen: set[str] = set()
    queue: deque[tuple[str, tuple[str, ...]]] = deque((m, (m,)) for m in modules)
    while queue:
        module, via = queue.popleft()
        if module in seen:
            continue
        seen.add(module)
        path = _module_file(module, roots)
        if path is None:
            raise ValueError(f"no source for module {module} under {dict(roots)}")
        found, imported = _scan(path, module, roots)
        out += [Violation(module, line, rule, detail, via) for line, rule, detail in found]
        queue.extend((m, (*via, m)) for m in imported if m not in seen)
    return out


def node_modules(roots: Mapping[str, Path] = ROOTS) -> set[str]:
    """The modules under `roots` that define a node: those that call requisite.compute.registry.node (as a decorator or
    not) or build a registry.Node directly, by whatever name they import it under."""
    found = set()
    for root in roots.values():
        for path in sorted(root.rglob("*.py")):
            module = _module_name(path, roots)
            tree = ast.parse(path.read_text(), filename=str(path))
            aliases = {alias: name for name, _, alias in _imports(tree, module, path.name == "__init__.py") if alias}
            if any(isinstance(n, ast.Call) and _dotted(n.func, aliases) in REGISTER for n in ast.walk(tree)):
                found.add(module)
    return found
