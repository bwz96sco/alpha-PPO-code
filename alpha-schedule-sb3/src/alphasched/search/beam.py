from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from alphasched.config.env import ResolvedEnvConfig
from alphasched.core.features import FeatureEncoder
from alphasched.core.instance import Instance
from alphasched.core.simulator import ParallelMachineSimulator
from alphasched.baselines.types import SolveResult
from .policy import Policy

_EPS = 1e-12


def _complete_perm(part_num: int, prefix: list[int]) -> np.ndarray:
    missing = [i for i in range(part_num) if i not in set(prefix)]
    return np.array(prefix + missing, dtype=int)


@dataclass(slots=True)
class _Node:
    sim: ParallelMachineSimulator
    seq: list[int]
    logp: float


def beam_search(instance: Instance, cfg: ResolvedEnvConfig, policy: Policy, *, beam_size: int) -> SolveResult:
    """Beam search baseline (paper): expand all feasible actions, keep top-K by trajectory probability."""
    start = time.time()
    encoder = FeatureEncoder(cfg)

    beam: list[_Node] = [_Node(sim=ParallelMachineSimulator(instance, cfg.mach_num), seq=[], logp=0.0)]
    part_num = cfg.part_num

    # Legacy runs for N-1 steps because the last job is auto-scheduled.
    for _ in range(max(part_num - 1, 1)):
        candidates: list[_Node] = []
        for node in beam:
            obs = encoder.observation(node.sim)
            mask = encoder.action_mask(node.sim)
            probs = policy.action_probabilities(obs, mask)
            valid = np.where(mask)[0]
            for a in valid.tolist():
                p = float(probs[a])
                child_sim = node.sim.clone()
                out = child_sim.step(int(a))
                logp = node.logp + float(np.log(max(p, _EPS)))
                child = _Node(sim=child_sim, seq=node.seq + [int(a)], logp=logp)
                candidates.append(child)
        candidates.sort(key=lambda n: n.logp, reverse=True)
        beam = candidates[: int(beam_size)]

    # Pick the most-probable final trajectory and report its WT.
    best = max(beam, key=lambda n: n.logp)
    wt = best.sim.final_wt()
    end = time.time()
    return SolveResult(
        best_perm=_complete_perm(part_num, best.seq),
        best_wt=float(wt),
        wall_time_sec=end - start,
        extra={"beam_size": int(beam_size), "policy_name": getattr(policy, "name", "policy")},
    )

