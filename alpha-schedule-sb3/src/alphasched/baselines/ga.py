from __future__ import annotations

import math
import time

import numpy as np

from alphasched.config.env import ResolvedEnvConfig
from alphasched.core.instance import Instance
from alphasched.core.simulator import evaluate_permutation
from .types import SolveResult


def _crossover(p1: np.ndarray, p2: np.ndarray, fixed_gene: int) -> np.ndarray:
    p1_fill = np.where(p1 == fixed_gene)[0]
    p1_drag = np.where(p1 != fixed_gene)[0]
    p2_drag = p2[np.where(p2 != fixed_gene)[0]]
    child = np.zeros_like(p1)
    child[p1_fill] = fixed_gene
    child[p1_drag] = p2_drag
    return child


def _mutate_swap_by_value(p: np.ndarray, v1: int, v2: int) -> np.ndarray:
    i1 = np.where(p == v1)[0]
    i2 = np.where(p == v2)[0]
    if i1.size == 0 or i2.size == 0:
        return p
    if i1[0] < i2[0]:
        first, second, first_len = v2, v1, i2.size
    else:
        first, second, first_len = v1, v2, i1.size
    idx = np.concatenate((i1, i2))
    idx.sort()
    p[idx[:first_len]] = first
    p[idx[first_len:]] = second
    return p


def solve_ga(
    instance: Instance,
    cfg: ResolvedEnvConfig,
    *,
    pop_size: int,
    iters: int,
    cross_rate: float = 0.6,
    mutate_rate: float = 0.1,
    rng: np.random.Generator | None = None,
) -> SolveResult:
    rng = rng or np.random.default_rng()
    start = time.time()

    n = cfg.part_num
    pop_size = int(pop_size)
    if pop_size % 2 == 1:
        pop_size += 1

    population = np.zeros((pop_size, n), dtype=int)
    for i in range(pop_size):
        population[i] = rng.permutation(n)

    best_wt = float("inf")
    best_ind = population[0].copy()

    def eval_pop(popu: np.ndarray) -> np.ndarray:
        grades = np.zeros(popu.shape[0], dtype=float)
        for j in range(popu.shape[0]):
            grades[j] = evaluate_permutation(instance, cfg.mach_num, popu[j])
        return grades

    for i in range(int(iters)):
        grades = eval_pop(population)
        arg_best = int(np.argmin(grades))
        if float(grades[arg_best]) < best_wt:
            best_wt = float(grades[arg_best])
            best_ind = population[arg_best].copy()

        # roulette selection (legacy-ish)
        m = 1.0 + math.log(max(iters, 2))
        t = min(i + 1, int(iters))
        fitness = (int(t ** (1 / m))) / (grades + 1e-3)
        prob = fitness / np.sum(fitness)
        chosen = rng.choice(pop_size, size=pop_size, replace=True, p=prob)
        temp = population[chosen].copy()

        # crossover
        cross_flags = rng.random(pop_size // 2) <= cross_rate
        for pair_idx, do_cross in enumerate(cross_flags):
            if not do_cross:
                continue
            a, b = 2 * pair_idx, 2 * pair_idx + 1
            p1, p2 = temp[a].copy(), temp[b].copy()
            j1, j2 = rng.choice(n, size=2, replace=False)
            c1 = _crossover(p1, p2, int(j1))
            c2 = _crossover(p2, p1, int(j2))
            temp[a], temp[b] = c1, c2

        # mutation
        mut_flags = rng.random(pop_size) <= mutate_rate
        for idx, do_mut in enumerate(mut_flags):
            if not do_mut:
                continue
            j1, j2 = rng.choice(n, size=2, replace=False)
            temp[idx] = _mutate_swap_by_value(temp[idx], int(j1), int(j2))

        population = temp

    end = time.time()
    return SolveResult(best_perm=best_ind, best_wt=best_wt, wall_time_sec=end - start, extra={"iters": iters})

