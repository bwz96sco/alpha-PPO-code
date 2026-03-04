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
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run one PSO phase with NatureSelection.

    Equivalent to legacy: loggerReset + iteration(iters).

    Returns (indi_best, global_best, global_best_grade).
    """
    pop_size, n = population.shape
    choose_num = int(pop_size * 0.5)

    # loggerReset: reset indiBest/globalBest tracking
    indi_best = np.zeros((pop_size, n), dtype=int)
    indi_best_grade = 1.0 / (np.zeros(pop_size) + 1e-9)  # legacy: ~1e9
    global_best = np.zeros(n, dtype=int)
    global_best_grade = float("inf")

    choose_popu = None
    choose_grade = None

    for i in range(iters):
        # 1. indiBestRefresh: evaluate and update individual bests
        grade = np.array(
            [evaluate_permutation(instance, n_mach, ind) for ind in population],
            dtype=float,
        )
        for m in range(pop_size):
            if grade[m] < indi_best_grade[m]:
                indi_best[m] = population[m]
                indi_best_grade[m] = grade[m]

        # 2. globalBestRefresh (legacy uses <=)
        best_idx = int(np.argmin(indi_best_grade))
        if indi_best_grade[best_idx] <= global_best_grade:
            global_best_grade = float(indi_best_grade[best_idx])
            global_best = indi_best[best_idx].copy()

        # 3. NatureSelection
        if i > 0:
            # Merge: top 50% by current grade + previous roulette pool
            grade_sort_index = np.argsort(grade)
            nice_popu = population[grade_sort_index[:choose_num]]
            nice_grade = grade[grade_sort_index[:choose_num]]
            population[:choose_num] = nice_popu
            population[choose_num:] = choose_popu
            grade[:choose_num] = nice_grade
            grade[choose_num:] = choose_grade

        # Select: roulette wheel selection
        fitness = 1.0 / (grade + 1e-3)
        probs = fitness / np.sum(fitness)
        choose_index = rng.choice(pop_size, size=choose_num, p=probs, replace=True)
        choose_popu = population[choose_index]
        choose_grade = grade[choose_index]

        # 4. Weight update
        w = w_end + (w_start - w_end) * (iters - i) / iters

        # 5. Speed update (legacy: rand(popSize, 2) then hsplit)
        r1, r2 = np.hsplit(rng.rand(pop_size, 2), 2)
        speed = (
            speed * w
            + c1 * r1 * (indi_best - population)
            + c2 * r2 * (global_best - population)
        )

        # 6. Position update
        middle = population + speed
        for m in range(pop_size):
            population[m] = np.argsort(np.argsort(middle[m]))

    return indi_best, global_best, global_best_grade


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
) -> SolveResult:
    """Multi-Agent PSO (paper MAPSO).

    3-phase pipeline replicating legacy BA/SA/EA architecture:
      Phase 1: BA initializes population, split into EA1 + EA2 sub-swarms
      Phase 2: EA1 and EA2 run independently with NatureSelection
      Phase 3: BA receives individual bests from EA1/EA2, runs final phase
    """
    if rng is None:
        rng = np.random.RandomState()
    start = time.time()

    n = cfg.part_num
    n_mach = cfg.mach_num
    sub_pop_size = pop_size // sub_pop_num

    # Phase 0: BA initializes random population
    ba_pop = np.zeros((pop_size, n), dtype=int)
    for m in range(pop_size):
        seq = list(range(n))
        rng.shuffle(seq)
        ba_pop[m] = seq

    # Divide into EA1 (first sub_pop_size) and EA2 (rest)
    ea1_pop = ba_pop[:sub_pop_size].copy()
    ea2_pop = ba_pop[sub_pop_size:].copy()

    # Phase 1: EA1 and EA2 each run iters with NatureSelection
    ea1_indi_best, _, _ = _pso_phase(
        ea1_pop,
        np.zeros((sub_pop_size, n), dtype=float),
        instance, n_mach, rng,
        iters=iters, c1=c1, c2=c2, w_start=w_start, w_end=w_end,
    )
    ea2_indi_best, _, _ = _pso_phase(
        ea2_pop,
        np.zeros((ea2_pop.shape[0], n), dtype=float),
        instance, n_mach, rng,
        iters=iters, c1=c1, c2=c2, w_start=w_start, w_end=w_end,
    )

    # Phase 2: BA receives EA1/EA2 individual bests as new population
    ba_pop[:sub_pop_size] = ea1_indi_best
    ba_pop[sub_pop_size:] = ea2_indi_best

    # Phase 3: BA runs final phase with NatureSelection
    _, global_best, global_best_grade = _pso_phase(
        ba_pop,
        np.zeros((pop_size, n), dtype=float),
        instance, n_mach, rng,
        iters=iters, c1=c1, c2=c2, w_start=w_start, w_end=w_end,
    )

    end = time.time()
    return SolveResult(
        best_perm=global_best,
        best_wt=global_best_grade,
        wall_time_sec=end - start,
        extra={"iters": iters * 3},
    )
