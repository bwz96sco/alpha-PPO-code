from __future__ import annotations

import time
from typing import Literal

import numpy as np

from alphasched.config.env import ResolvedEnvConfig
from alphasched.core.instance import Instance
from alphasched.core.simulator import ParallelMachineSimulator
from .types import SolveResult

RuleName = Literal["spt", "mp", "wspt", "wmdd", "atc", "wco"]

_EPS = 1e-10


def _select_action(sim: ParallelMachineSimulator, cfg: ResolvedEnvConfig, rule: RuleName) -> int:
    avail = sim.available_actions()
    if avail.size == 0:
        raise RuntimeError("no available actions")

    hours = sim.part[avail, 0]
    deadline = sim.part[avail, 1]
    weight = sim.part[avail, 2]
    cur = float(np.min(sim.mach))

    if rule == "spt":
        return int(avail[int(np.argmin(hours))])
    if rule == "mp":
        return int(avail[int(np.argmax(weight))])

    deadline_cur = (deadline - cur) * (deadline > 0)
    deadline_cur_hours = deadline_cur - hours
    delta = -np.maximum(deadline_cur_hours, 0) / (hours + _EPS)

    if rule == "wspt":
        value = weight / (hours + _EPS)
    elif rule == "wmdd":
        value = -np.maximum(hours, deadline_cur) / (weight + _EPS)
    elif rule == "atc":
        value = np.exp(delta / float(cfg.h)) * weight / (hours + _EPS)
    elif rule == "wco":
        value = np.maximum(1.0 + delta / float(cfg.kt), 0.0) * weight / (hours + _EPS)
    else:
        raise ValueError(f"unknown rule: {rule!r}")

    return int(avail[int(np.argmax(value))])


def solve_rule(instance: Instance, cfg: ResolvedEnvConfig, rule: RuleName) -> SolveResult:
    start = time.time()
    sim = ParallelMachineSimulator(instance, cfg.mach_num)

    actions: list[int] = []
    steps = 0
    while True:
        action = _select_action(sim, cfg, rule)
        out = sim.step(action)
        actions.append(action)
        steps += 1
        if out.done:
            assert out.wt_final is not None
            end = time.time()
            return SolveResult(
                best_perm=np.array(actions, dtype=int),
                best_wt=float(out.wt_final),
                wall_time_sec=end - start,
                extra={"steps": steps, "rule": rule},
            )

