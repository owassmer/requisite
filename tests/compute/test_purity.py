"""I7's static check (requisite/compute/purity.py): a registered node reaches no provider, cassette or network code, and
reads no clock, randomness or environment (spec §3 I7; spec §13 WP6: "a static check forbids provider calls inside
nodes")."""

import importlib
import textwrap
from pathlib import Path

import pytest

from requisite.compute import purity, registry

PRODUCT = purity.PRODUCT


def product_modules() -> list[str]:
    """Every module of the product, read from its files (a package without __init__.py included)."""
    return sorted({purity._module_name(p, purity.ROOTS) for p in PRODUCT.rglob("*.py")})


PLANTED = {
    "ok": """
        from datetime import datetime
        from decimal import Decimal

        from requisite.compute.registry import node
        from requisite.core import Money

        @node(id="t.ok", version=1, inputs=["a", "b"])
        def ok(a: Decimal, b: Decimal) -> Decimal:
            input = a + b  # a local named like a builtin is not the builtin
            return input if datetime(2026, 1, 1).year else Decimal(0)
        """,
    "net": """
        import httpx
        from requisite.compute import registry

        @registry.node(id="t.net", version=1, inputs=["url"])
        def fetch(url):
            return httpx.get(url).text
        """,
    "provider": """
        from requisite.judgment import providers
        from requisite.compute.registry import node as register

        register(id="t.provider", version=1, inputs=["q"])(lambda q: providers)
        """,
    "built": """
        from requisite.compute import registry as r

        LOOKUP = {("t.built", 1): r.Node("t.built", 1, ("x",), {}, lambda x: x)}
        """,
    "deep": "from planted import helper\n",
    "helper": "import socket\n",
    "clock": "from datetime import datetime\n\ndef f():\n    return datetime.now()\n",
    "rand": "import numpy as np\n\ndef f():\n    return np.random.default_rng().random()\n",
    "env": "import os\n\ndef f():\n    return os.environ['HOME']\n",
    "dyn": "def f(name):\n    return __import__(name), open(name).read()\n",
    "state": "from requisite import store\n",
    "lazy": "def f():\n    from requisite.cassettes import replay\n    return replay\n",
}


@pytest.fixture
def planted(tmp_path: Path) -> dict[str, Path]:
    pkg = tmp_path / "planted"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for name, body in PLANTED.items():
        (pkg / f"{name}.py").write_text(textwrap.dedent(body))
    return {**purity.ROOTS, "planted": pkg}


def rules(roots, module: str) -> set[tuple[str, str]]:
    return {(v.module, v.rule) for v in purity.check([module], roots)}


@pytest.mark.covers("I7")
def test_a_node_reaches_no_provider_network_clock_randomness_or_environment(planted):
    # The product: every module a node is registered in, found from the source and from the registry once every product
    # module is imported, and the registry module every node imports, reach nothing a node may not reach.
    for module in product_modules():
        importlib.import_module(module)
    registered = {m for n in registry.registered() if (m := n.fn.__module__).split(".")[0] == "requisite"}
    found = purity.node_modules()
    assert registered <= found  # no node escapes the walk by being registered from somewhere it does not look
    assert purity.check(sorted(found | {"requisite.compute.registry"})) == []

    # The walk finds a node registered under any name it is imported by, as a decorator or not, or built directly.
    assert purity.node_modules(planted) >= {"planted.ok", "planted.net", "planted.provider", "planted.built"}
    # Pure code passes: decimals, datetimes built from fields, the registry and the definition.
    assert purity.check(["planted.ok"], planted) == []
    # A network module imported directly.
    assert ("planted.net", "network") in rules(planted, "planted.net")
    # Provider code, and through it the network, the cassettes and the keys, each named with the path that reached it.
    found_provider = purity.check(["planted.provider"], planted)
    assert {("planted.provider", "provider calls"), ("requisite.judgment.providers", "network"),
            ("requisite.judgment.providers", "recorded provider responses"),
            ("requisite.judgment.providers", "keys")} <= {(v.module, v.rule) for v in found_provider}  # fmt: skip
    assert any(v.via == ("planted.provider", "requisite.judgment.providers") for v in found_provider)
    # A network import two steps away, in a module that registers no node.
    deep = purity.check(["planted.deep"], planted)
    assert [(v.module, v.rule, v.via) for v in deep] == [("planted.helper", "network", ("planted.deep", "planted.helper"))]
    # A clock, randomness, the environment, a dynamic import and a file read, in the node's own module.
    assert ("planted.clock", "clock") in rules(planted, "planted.clock")
    assert ("planted.rand", "randomness") in rules(planted, "planted.rand")
    assert ("planted.env", "process and environment") in rules(planted, "planted.env")
    assert {("planted.dyn", "dynamic import"), ("planted.dyn", "files")} <= rules(planted, "planted.dyn")
    # Committed state is read only through the graph's sources; an import inside a function is still an import.
    assert ("planted.state", "committed state, which reaches a node only as a source of the graph") in rules(
        planted, "planted.state"
    )
    assert ("planted.lazy", "recorded provider responses") in rules(planted, "planted.lazy")
