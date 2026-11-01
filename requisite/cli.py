"""The command line: the product as a user runs it (spec §13 WP16: a CLI plus a static HTML dossier per case)."""

from __future__ import annotations

import argparse
import sys

from requisite import registry


def _registry(args: argparse.Namespace) -> int:
    reg, cov = registry.items(), registry.coverage()
    bad = registry.problems(reg, cov)
    print(registry.board(reg, cov))
    for p in bad:
        print(f"PROBLEM {p}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="requisite",
        description="A runtime for discovery and contract construction. Analysis, not advice; prices are indicative unless quoted.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("registry", help="where the sources' enumerated items stand: covered, open, replaced, by tier")
    r.set_defaults(run=_registry)
    args = parser.parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    sys.exit(main())
