from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Instance:
    """Scheduling instance for parallel machines with TWT objective.

    Jobs are represented as an (N, 3) array with columns:
    - p: processing time
    - d: due date / deadline
    - w: weight (priority)
    """

    jobs: np.ndarray
    seed: int

    @property
    def part_num(self) -> int:
        return int(self.jobs.shape[0])

