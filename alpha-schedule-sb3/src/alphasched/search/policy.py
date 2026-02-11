from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Policy(Protocol):
    name: str

    def action_probabilities(self, obs: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
        """Return probability vector over actions (length N)."""


@dataclass(frozen=True, slots=True)
class RandomPolicy:
    part_num: int
    name: str = "random"

    def action_probabilities(self, obs: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
        mask = action_mask.astype(bool, copy=False)
        probs = np.zeros(self.part_num, dtype=np.float64)
        valid = np.where(mask)[0]
        if valid.size == 0:
            return probs
        probs[valid] = 1.0 / valid.size
        return probs

