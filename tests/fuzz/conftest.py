"""The fuzz tests build contracts with the contract tests' builders (tests/contract/irkit.py, irgen.py and the Western
Grazers IR), so that directory is importable here."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "contract"))
