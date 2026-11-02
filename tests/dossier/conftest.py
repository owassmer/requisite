"""The dossier tests render cases built by the product's own functions over the other packages' test doubles and
fixtures (tests/contract, tests/fuzz, tests/evidence), so those directories are importable here."""

import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent.parent
for d in ("contract", "fuzz", "evidence"):
    if str(TESTS / d) not in sys.path:
        sys.path.insert(0, str(TESTS / d))
