"""The library's consequence nodes: computations a contract or a business model defines, registered as pure nodes (spec
§4 "@node", §12: a contract's own computation is a library addition, not core code). Importing this package registers
every node in it with requisite.compute.registry, so an IR that names one can be evaluated."""

from requisite.library.nodes import cagoatwage, cash, compose, materiality

__all__ = ["cagoatwage", "cash", "compose", "materiality"]
