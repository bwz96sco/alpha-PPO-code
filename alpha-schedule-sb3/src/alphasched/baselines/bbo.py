from __future__ import annotations

import time

import numpy as np

from alphasched.config.env import ResolvedEnvConfig
from alphasched.core.instance import Instance
from alphasched.core.simulator import evaluate_permutation
from .types import SolveResult


def _transfer_in(ind: np.ndarray, pos: int, value: int) -> np.ndarray:
    dup_pos = np.where(ind == value)[0]
    if dup_pos.size > 0:
        ind[dup_pos] = ind[pos]
    ind[pos] = value
    return ind


def solve_bbo(
    instance: Instance,
    cfg: ResolvedEnvConfig,
    *,
    pop_size: int,
    iters: int,
    mutate_rate: float = 0.05,
    elite_num: int = 2,
    rng: np.random.Generator | None = None,
) -> SolveResult:
    rng = rng or np.random.default_rng()
    start = time.time()

    n = cfg.part_num
    pop_size = int(pop_size)
    elite_num = int(elite_num)

    population = np.zeros((pop_size, n), dtype=int)
    for i in range(pop_size):
        population[i] = rng.permutation(n)

    mu = np.arange(pop_size, 0, -1, dtype=float) / (pop_size + 1)
    lamb = 1.0 - mu
    prob = mu / mu.sum()

    best_wt = float("inf")
    best_ind = population[0].copy()

    for _ in range(int(iters)):
        grades = np.array([evaluate_permutation(instance, cfg.mach_num, ind) for ind in population], dtype=float)
        sort_idx = np.argsort(grades)
        population = population[sort_idx]
        grades = grades[sort_idx]
        elite = population[:elite_num].copy()

        if float(grades[0]) < best_wt:
            best_wt = float(grades[0])
            best_ind = population[0].copy()

        # migrate
        temp = population.copy()
        lam_rand = rng.random((pop_size, n))
        for i in range(pop_size):
            move_in = np.where(lam_rand[i] < lamb[i])[0]
            if move_in.size == 0:
                continue
            donors = rng.choice(pop_size, size=int(move_in.size), replace=True, p=prob)
            for j, pos in enumerate(move_in.tolist()):
                donor_idx = int(donors[j])
                value = int(population[donor_idx, pos])
                temp[i] = _transfer_in(temp[i], int(pos), value)

        # mutate
        mut_rand = rng.random((pop_size, n))
        for i in range(pop_size):
            pos_list = np.where(mut_rand[i] < mutate_rate)[0]
            if pos_list.size == 1:
                pos = int(pos_list[0])
                new_pos = int(rng.integers(0, n))
                temp[i, [pos, new_pos]] = temp[i, [new_pos, pos]]
            elif pos_list.size > 1:
                vals = temp[i, pos_list].copy()
                rng.shuffle(vals)
                temp[i, pos_list] = vals

        # elite policy: replace worst with elites
        temp[-elite_num:, :] = elite
        population = temp

    # final eval
    grades = np.array([evaluate_permutation(instance, cfg.mach_num, ind) for ind in population], dtype=float)
    arg_best = int(np.argmin(grades))
    if float(grades[arg_best]) < best_wt:
        best_wt = float(grades[arg_best])
        best_ind = population[arg_best].copy()

    end = time.time()
    return SolveResult(best_perm=best_ind, best_wt=best_wt, wall_time_sec=end - start, extra={"iters": iters})

