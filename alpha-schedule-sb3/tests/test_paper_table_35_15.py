"""Regression tests: verify algorithm results match paper Table (N=35, M=15, high-load).

Paper: "AlphaSchedule" — Mean total weighted tardiness under high-load level.
Column (35,15) from paper Table.

Expected values:
    BBO:    4962.86
    GA:     5148.22
    MAPSO:  6244.35
    SPT:    9520.65
    MP:     7925.10

Rules (SPT, MP) are deterministic and must match exactly.

Stochastic algorithms (GA, BBO, PSO) use legacy-compatible seeding:
the legacy code seeds np.random.seed(instance_id) ONCE per instance,
then createPart() draws 3 * part_num values, and the BBO/GA/PSO
optimization inherits the remaining MT19937 state.  We replicate this
by creating a RandomState(seed), advancing it past the instance-generation
draws, and passing it to the solver.

GA uses a "best of 5 combinations" approach per the paper (line 580):
    (individuals, iterations) chosen from
    {(100,800), (200,400), (285,285), (400,200), (800,100)}
    and the best result is kept per instance.
"""
from __future__ import annotations

import numpy as np
import pytest

from alphasched.baselines import solve_bbo, solve_ga, solve_pso, solve_rule
from alphasched.config.env import EnvConfig
from alphasched.core.generator import InstanceGenerator

# --------------------------------------------------------------------------- #
# Paper Table values for N=35, M=15, dist_type="h"
# --------------------------------------------------------------------------- #
PAPER_BBO = 4962.86
PAPER_GA = 5148.22
PAPER_MAPSO = 6244.35
PAPER_SPT = 9520.65
PAPER_MP = 7925.10

# --------------------------------------------------------------------------- #
# Experiment configuration
# --------------------------------------------------------------------------- #
PART_NUM = 35
MACH_NUM = 15  # auto: (35-5)//10 * 5 = 15
DIST_TYPE = "h"
TEST_NUM = 100

# GA parameter combinations from paper (pop_size, iters).
# All have ~80,000 total evaluations budget.
GA_COMBOS: list[tuple[int, int]] = [
    (100, 800),
    (200, 400),
    (285, 285),
    (400, 200),
    (800, 100),
]


# --------------------------------------------------------------------------- #
# Legacy-compatible RNG helper
# --------------------------------------------------------------------------- #
def _legacy_rng_after_instance(cfg, seed: int) -> np.random.RandomState:
    """Create a RandomState matching the legacy post-createPart() state.

    Legacy flow per instance:
        np.random.seed(seed)
        job_time  = np.random.randint(min_time, max_time, size=part_num)
        tight     = job_time * (1 + np.random.rand(part_num) * tight_factor)
        weight    = np.random.randint(1, priority_max, size=part_num)
        # ... BBO/GA/PSO starts consuming from HERE ...

    We replicate this by creating RandomState(seed) and drawing the same
    3 arrays to advance the state to the correct position.
    """
    rng = np.random.RandomState(seed)
    # Draw the same 3 arrays as createPart() / InstanceGenerator.generate()
    rng.randint(cfg.min_time, cfg.max_time, size=cfg.part_num)   # job_time
    rng.rand(cfg.part_num)                                        # tight factor
    rng.randint(1, cfg.priority_max, size=cfg.part_num)           # weight
    return rng


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def resolved_cfg():
    return EnvConfig(
        part_num=PART_NUM,
        dist_type=DIST_TYPE,
        mach_num=MACH_NUM,
    ).resolved()


@pytest.fixture(scope="module")
def instances(resolved_cfg):
    gen = InstanceGenerator(resolved_cfg, rng_backend="legacy_mt19937")
    return [gen.generate(mode="test", instance_id=i) for i in range(TEST_NUM)]


# --------------------------------------------------------------------------- #
# Deterministic rule tests — must match paper exactly
# --------------------------------------------------------------------------- #
class TestRules:
    """SPT and MP are deterministic; results must match paper values exactly."""

    def test_spt_matches_paper(self, resolved_cfg, instances):
        wt_list = [
            solve_rule(inst, resolved_cfg, "spt").best_wt for inst in instances
        ]
        mean_wt = float(np.mean(wt_list))
        assert mean_wt == pytest.approx(PAPER_SPT, abs=0.01), (
            f"SPT mean WT {mean_wt:.2f} != paper {PAPER_SPT}"
        )

    def test_mp_matches_paper(self, resolved_cfg, instances):
        wt_list = [
            solve_rule(inst, resolved_cfg, "mp").best_wt for inst in instances
        ]
        mean_wt = float(np.mean(wt_list))
        assert mean_wt == pytest.approx(PAPER_MP, abs=0.01), (
            f"MP mean WT {mean_wt:.2f} != paper {PAPER_MP}"
        )


# --------------------------------------------------------------------------- #
# Stochastic algorithm tests — legacy-compatible seeding
# --------------------------------------------------------------------------- #
class TestBBO:
    """BBO: 150 individuals, 400 iterations, 0.05 mutation rate, 2 elites (paper)."""

    @pytest.mark.slow
    def test_bbo_matches_paper(self, resolved_cfg, instances):
        wt_list: list[float] = []
        for inst in instances:
            # Replicate legacy: RNG state = MT19937 after createPart()
            rng = _legacy_rng_after_instance(resolved_cfg, inst.seed)
            result = solve_bbo(
                inst,
                resolved_cfg,
                pop_size=150,
                iters=400,
                mutate_rate=0.05,
                elite_num=2,
                rng=rng,
            )
            wt_list.append(result.best_wt)
        mean_wt = float(np.mean(wt_list))
        print(f"\nBBO mean WT: {mean_wt:.2f} (paper: {PAPER_BBO})")
        assert mean_wt == pytest.approx(PAPER_BBO, abs=0.01), (
            f"BBO mean WT {mean_wt:.2f} != paper {PAPER_BBO}"
        )


class TestGA:
    """GA: best of 5 (pop_size, iters) combos, cross_rate=0.6, mutate_rate=0.1 (paper).

    Per the paper (line 580), (individuals, iterations) is the optimal one
    chosen from {(100,800), (200,400), (285,285), (400,200), (800,100)}.
    One combo is selected that gives the best MEAN WT across all instances.
    """

    @pytest.mark.slow
    def test_ga_best_of_5_matches_paper(self, resolved_cfg, instances):
        # Run each combo on ALL instances, compute mean per combo, pick best mean.
        best_mean = float("inf")
        best_combo = GA_COMBOS[0]
        for pop_size, iters in GA_COMBOS:
            wt_list: list[float] = []
            for inst in instances:
                rng = _legacy_rng_after_instance(resolved_cfg, inst.seed)
                result = solve_ga(
                    inst,
                    resolved_cfg,
                    pop_size=pop_size,
                    iters=iters,
                    cross_rate=0.6,
                    mutate_rate=0.1,
                    rng=rng,
                )
                wt_list.append(result.best_wt)
            combo_mean = float(np.mean(wt_list))
            print(f"\n  GA ({pop_size},{iters}): mean WT = {combo_mean:.2f}")
            if combo_mean < best_mean:
                best_mean = combo_mean
                best_combo = (pop_size, iters)
        print(f"\nGA best combo {best_combo}: mean WT = {best_mean:.2f} (paper: {PAPER_GA})")
        assert best_mean == pytest.approx(PAPER_GA, abs=0.01), (
            f"GA mean WT {best_mean:.2f} != paper {PAPER_GA}"
        )


class TestMAPSO:
    """PSO: 400 particles, 200 iterations, c1=2.0, c2=2.1, w=0.9->0.4 (paper).

    Note: the paper uses a Multi-Agent PSO (MAPSO) with sub-swarms
    (BA/SA/EA architecture).  The current solve_pso is a single-swarm PSO,
    so results may differ from the paper's MAPSO values.
    """

    @pytest.mark.slow
    def test_pso_matches_paper(self, resolved_cfg, instances):
        wt_list: list[float] = []
        for inst in instances:
            rng = _legacy_rng_after_instance(resolved_cfg, inst.seed)
            result = solve_pso(
                inst,
                resolved_cfg,
                pop_size=400,
                iters=200,
                c1=2.0,
                c2=2.1,
                w_start=0.9,
                w_end=0.4,
                rng=rng,
            )
            wt_list.append(result.best_wt)
        mean_wt = float(np.mean(wt_list))
        print(f"\nPSO mean WT: {mean_wt:.2f} (paper MAPSO: {PAPER_MAPSO})")
        # PSO (single-swarm) vs MAPSO (multi-agent) — allow wider tolerance
        assert mean_wt == pytest.approx(PAPER_MAPSO, rel=0.10), (
            f"PSO mean WT {mean_wt:.2f} deviates >10% from paper MAPSO {PAPER_MAPSO}"
        )
