from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SolveResult:
    best_perm: np.ndarray
    best_wt: float
    wall_time_sec: float
    extra: dict[str, Any] = field(default_factory=dict)

