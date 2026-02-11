from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from alphasched.config.env import ResolvedEnvConfig
from alphasched.core.features import FeatureEncoder
from alphasched.core.instance import Instance
from alphasched.core.simulator import ParallelMachineSimulator
from .policy import Policy

_EPS = 1e-12


def _complete_perm(part_num: int, prefix: list[int]) -> np.ndarray:
    missing = [i for i in range(part_num) if i not in set(prefix)]
    return np.array(prefix + missing, dtype=int)


@dataclass(frozen=True, slots=True)
class RolloutResult:
    perm: np.ndarray
    wt: float


def greedy_rollout(instance: Instance, cfg: ResolvedEnvConfig, policy: Policy) -> RolloutResult:
    encoder = FeatureEncoder(cfg)
    sim = ParallelMachineSimulator(instance, cfg.mach_num)
    seq: list[int] = []
    while True:
        obs = encoder.observation(sim)
        mask = encoder.action_mask(sim)
        probs = policy.action_probabilities(obs, mask)
        valid = np.where(mask)[0]
        if valid.size == 0:
            wt = sim.final_wt()
            return RolloutResult(perm=_complete_perm(cfg.part_num, seq), wt=float(wt))
        choice = int(valid[int(np.argmax(probs[valid]))])
        out = sim.step(choice)
        seq.append(choice)
        if out.done:
            assert out.wt_final is not None
            return RolloutResult(perm=_complete_perm(cfg.part_num, seq), wt=float(out.wt_final))


def random_rollout(instance: Instance, cfg: ResolvedEnvConfig, rng: np.random.Generator) -> RolloutResult:
    sim = ParallelMachineSimulator(instance, cfg.mach_num)
    seq: list[int] = []
    while True:
        valid = sim.available_actions()
        if valid.size == 0:
            wt = sim.final_wt()
            return RolloutResult(perm=_complete_perm(cfg.part_num, seq), wt=float(wt))
        choice = int(rng.choice(valid))
        out = sim.step(choice)
        seq.append(choice)
        if out.done:
            assert out.wt_final is not None
            return RolloutResult(perm=_complete_perm(cfg.part_num, seq), wt=float(out.wt_final))

