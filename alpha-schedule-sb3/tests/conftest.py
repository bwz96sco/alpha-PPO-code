from __future__ import annotations

import sys
from pathlib import Path

# Allow `pytest` to run without installing the package.
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

