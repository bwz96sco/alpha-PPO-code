from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .instance import Instance

_EPS = 1e-10


@dataclass(frozen=True, slots=True)
class StepOutcome:
    reward: float
    done: bool
    wt_final: float | None
    invalid_action: bool


class ParallelMachineSimulator:
    """Stateful simulator.

    Mirrors the legacy semantics:
    - Each action schedules one job.
    - Job always goes to the least-loaded machine.
    - When only one job remains, it is automatically scheduled.
    """

    def __init__(self, instance: Instance, mach_num: int):
        if mach_num <= 0:
            raise ValueError(f"mach_num must be positive, got {mach_num}")
        self.instance = instance
        self.part_num = int(instance.jobs.shape[0])
        self.mach_num = int(mach_num)
        self.reset()

    def reset(self) -> None:
        self.part = self.instance.jobs.copy()
        self.mach = np.zeros(self.mach_num, dtype=np.float64)
        # start, end, machine, grade(deadline-end), weight
        self.part_log = np.zeros((self.part_num, 5), dtype=np.float64)
        self.remaining = self.part_num

    def clone(self) -> "ParallelMachineSimulator":
        other = ParallelMachineSimulator(self.instance, self.mach_num)
        other.part = self.part.copy()
        other.mach = self.mach.copy()
        other.part_log = self.part_log.copy()
        other.remaining = int(self.remaining)
        return other

    def available_actions(self) -> np.ndarray:
        return np.where(self.part[:, 0] != 0)[0]

    def action_mask(self) -> np.ndarray:
        return (self.part[:, 0] != 0)

    def _schedule_one(self, job_id: int) -> float:
        hours, deadline, weight = self.part[job_id]
        mach_idx = int(np.argmin(self.mach))
        start = float(self.mach[mach_idx])
        end = start + float(hours)
        grade = float(deadline - end)

        self.mach[mach_idx] = end
        self.part[job_id] = 0.0
        self.part_log[job_id] = [start, end, float(mach_idx), grade, float(weight)]
        self.remaining -= 1

        tardiness = max(end - float(deadline), 0.0)
        return -float(weight) * tardiness

    def final_wt(self) -> float:
        grade = self.part_log[:, 3]
        weight = self.part_log[:, 4]
        tardy = grade < 0
        return float(-np.sum(grade[tardy] * weight[tardy]))

    def step(self, action: int, *, invalid_penalty: float = -1e6) -> StepOutcome:
        action = int(action)
        if action < 0 or action >= self.part_num or self.part[action, 0] == 0:
            return StepOutcome(reward=float(invalid_penalty), done=False, wt_final=None, invalid_action=True)

        reward = self._schedule_one(action)

        # Auto-schedule the last remaining job (legacy behavior).
        avail = self.available_actions()
        if avail.size == 1:
            reward += self._schedule_one(int(avail[0]))
            return StepOutcome(reward=float(reward), done=True, wt_final=self.final_wt(), invalid_action=False)

        if avail.size == 0:
            return StepOutcome(reward=float(reward), done=True, wt_final=self.final_wt(), invalid_action=False)

        return StepOutcome(reward=float(reward), done=False, wt_final=None, invalid_action=False)


def evaluate_permutation(instance: Instance, mach_num: int, perm: Sequence[int]) -> float:
    """Evaluate a (possibly length N) permutation; last job may be auto-finished."""
    sim = ParallelMachineSimulator(instance, mach_num)
    for a in perm:
        out = sim.step(int(a))
        if out.invalid_action:
            # Penalize invalid perms by returning a very large objective.
            return float("inf")
        if out.done:
            assert out.wt_final is not None
            return float(out.wt_final)
    # If permutation ended early but episode not finished, finish with random valid actions.
    # This keeps the function total and makes it usable in metaheuristics.
    avail = sim.available_actions()
    for a in avail.tolist():
        out = sim.step(int(a))
        if out.done:
            assert out.wt_final is not None
            return float(out.wt_final)
    return sim.final_wt()
