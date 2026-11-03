"""The deep case opened from a lead (test_lead.py) is opened from a wide case built by the wide flow's own functions over
the other packages' test doubles and fixtures (tests/dossier/cases.py, and through it tests/evidence, tests/contract and
tests/fuzz), so those directories are importable here."""

import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent.parent
for d in ("contract", "fuzz", "evidence", "dossier"):
    if str(TESTS / d) not in sys.path:
        sys.path.insert(0, str(TESTS / d))
