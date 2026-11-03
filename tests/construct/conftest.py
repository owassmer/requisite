"""Construction's tests build their deep cases with deep discovery's builder over the product's own steps
(tests/discovery/deep_case.py), so that directory is importable here."""

import sys
from pathlib import Path

DISCOVERY = Path(__file__).resolve().parent.parent / "discovery"
if str(DISCOVERY) not in sys.path:
    sys.path.insert(0, str(DISCOVERY))
