from __future__ import annotations

import time

import numpy as np

from alphasched.config.env import ResolvedEnvConfig
from alphasched.core.instance import Instance
from alphasched.core.simulator import evaluate_permutation
from .types import SolveResult


def solve_pso(
    instance: Instance,
    cfg: ResolvedEnvConfig,
    *,
    pop_size: int,
    iters: int,
    c1: float = 2.0,
    c2: float = 2.1,
    w_start: float = 0.9,
    w_end: float = 0.4,
    rng: np.random.Generator | None = None,
) -> SolveResult:
    rng = rng or np.random.default_rng()
    start = time.time()

    n = cfg.part_num
    pop_size = int(pop_size)
    iters = int(iters)

    population = np.zeros((pop_size, n), dtype=int)
    speed = np.zeros((pop_size, n), dtype=float)
    for i in range(pop_size):
        population[i] = rng.permutation(n)

    indi_best = population.copy()
    indi_best_grade = np.full(pop_size, float("inf"), dtype=float)
    global_best = population[0].copy()
    global_best_grade = float("inf")

    for i in range(iters):
        grades = np.array([evaluate_permutation(instance, cfg.mach_num, ind) for ind in population], dtype=float)
        better = grades < indi_best_grade
        indi_best_grade[better] = grades[better]
        indi_best[better] = population[better]

        best_idx = int(np.argmin(indi_best_grade))
        if float(indi_best_grade[best_idx]) < global_best_grade:
            global_best_grade = float(indi_best_grade[best_idx])
            global_best = indi_best[best_idx].copy()

        w = float(w_end + (w_start - w_end) * (iters - i) / max(iters, 1))
        r1 = rng.random((pop_size, 1))
        r2 = rng.random((pop_size, 1))
        speed = (
            speed * w
            + c1 * r1 * (indi_best.astype(float) - population.astype(float))
            + c2 * r2 * (global_best.astype(float) - population.astype(float))
        )
        middle = population.astype(float) + speed
        for m in range(pop_size):
            population[m] = np.argsort(np.argsort(middle[m])).astype(int)

    end = time.time()
    return SolveResult(best_perm=global_best, best_wt=global_best_grade, wall_time_sec=end - start, extra={"iters": iters})

