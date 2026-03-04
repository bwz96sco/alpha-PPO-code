from __future__ import annotations

import time

import numpy as np

from alphasched.config.env import ResolvedEnvConfig
from alphasched.core.instance import Instance
from alphasched.core.simulator import evaluate_permutation
from .types import SolveResult


def _pso_phase(
    population: np.ndarray,
    speed: np.ndarray,
    instance: Instance,
    n_mach: int,
    rng: np.random.RandomState,
    *,
    iters: int,
    c1: float,
    c2: float,
    w_start: float,
    w_end: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Run one PSO phase (no NatureSelection).

    Replicates ``zhangMulti/pso.py`` iteration() with loggerReset.

    Returns (indi_best, global_best, global_best_grade, final_speed).
    """
    pop_size, n = population.shape

    # loggerReset: reset indiBest/globalBest tracking
    indi_best = np.zeros((pop_size, n), dtype=int)
    indi_best_grade = 1.0 / (np.zeros(pop_size) + 1e-9)  # legacy: ~1e9
    global_best = np.zeros(n, dtype=int)
    global_best_grade = float("inf")

    for i in range(iters):
        # indiBestRefresh
        for m in range(pop_size):
            grade = evaluate_permutation(instance, n_mach, population[m])
            if grade < indi_best_grade[m]:
                indi_best[m] = population[m]
                indi_best_grade[m] = grade

        # globalBestRefresh
        best_idx = int(np.argmin(indi_best_grade))
        if indi_best_grade[best_idx] <= global_best_grade:
            global_best_grade = float(indi_best_grade[best_idx])
            global_best = indi_best[best_idx].copy()

        # weightUpdate
        w = w_end + (w_start - w_end) * (iters - i) / iters

        # speedRefresh (legacy: rand(popSize, 2) then hsplit)
        r1, r2 = np.hsplit(rng.rand(pop_size, 2), 2)
        speed = (
            speed * w
            + c1 * r1 * (indi_best - population)
            + c2 * r2 * (global_best - population)
        )

        # popRefresh
        middle = population + speed
        for m in range(pop_size):
            population[m] = np.argsort(np.argsort(middle[m]))

    return indi_best, global_best, global_best_grade, speed


def solve_mapso(
    instance: Instance,
    cfg: ResolvedEnvConfig,
    *,
    pop_size: int = 400,
    sub_pop_num: int = 2,
    iters: int = 200,
    c1: float = 2.0,
    c2: float = 2.1,
    w_start: float = 0.9,
    w_end: float = 0.4,
    rng: np.random.RandomState | None = None,
    ea_speeds: tuple[np.ndarray, ...] | None = None,
) -> SolveResult:
    """Multi-Agent PSO (paper MAPSO).

    3-phase pipeline replicating ``zhangMulti`` architecture:
      Phase 1: BA initializes population, split into EA1 + EA2 sub-swarms
      Phase 2: EA1 and EA2 run independently (standard PSO, no NatureSelection)
      Phase 3: BA receives individual bests from EA1/EA2, runs final phase

    Parameters
    ----------
    ea_speeds : tuple of ndarray, optional
        Initial speed arrays for EA sub-swarms, carried over from the
        previous instance call (legacy behavior).  When *None*, all
        sub-swarms start with zero speed.  Pass the value returned in
        ``result.extra["ea_speeds"]`` to replicate the legacy state
        carry-over across instances.
    """
    if rng is None:
        rng = np.random.RandomState()
    start = time.time()

    n = cfg.part_num
    n_mach = cfg.mach_num
    sub_pop_size = pop_size // sub_pop_num

    # Phase 0: BA initializes random population (speed = zeros)
    ba_pop = np.zeros((pop_size, n), dtype=int)
    for m in range(pop_size):
        seq = list(range(n))
        rng.shuffle(seq)
        ba_pop[m] = seq

    # Divide into EA1 (first sub_pop_size) and EA2 (rest)
    ea1_pop = ba_pop[:sub_pop_size].copy()
    ea2_pop = ba_pop[sub_pop_size:].copy()

    # Determine initial EA speeds
    ea2_size = pop_size - sub_pop_size
    if ea_speeds is not None:
        ea1_speed = ea_speeds[0]
        ea2_speed = ea_speeds[1]
    else:
        ea1_speed = np.zeros((sub_pop_size, n), dtype=float)
        ea2_speed = np.zeros((ea2_size, n), dtype=float)

    pso_kwargs = dict(iters=iters, c1=c1, c2=c2, w_start=w_start, w_end=w_end)

    # Phase 1: EA1 and EA2 each run iters
    ea1_indi_best, _, _, ea1_final_speed = _pso_phase(
        ea1_pop, ea1_speed, instance, n_mach, rng, **pso_kwargs,
    )
    ea2_indi_best, _, _, ea2_final_speed = _pso_phase(
        ea2_pop, ea2_speed, instance, n_mach, rng, **pso_kwargs,
    )

    # Phase 2: BA receives EA1/EA2 individual bests as new population
    ba_pop[:sub_pop_size] = ea1_indi_best
    ba_pop[sub_pop_size:] = ea2_indi_best

    # Phase 3: BA runs final phase (speed = zeros, fresh each instance)
    _, global_best, global_best_grade, _ = _pso_phase(
        ba_pop,
        np.zeros((pop_size, n), dtype=float),
        instance, n_mach, rng,
        **pso_kwargs,
    )

    end = time.time()
    return SolveResult(
        best_perm=global_best,
        best_wt=global_best_grade,
        wall_time_sec=end - start,
        extra={
            "iters": iters * 3,
            "ea_speeds": (ea1_final_speed, ea2_final_speed),
        },
    )
