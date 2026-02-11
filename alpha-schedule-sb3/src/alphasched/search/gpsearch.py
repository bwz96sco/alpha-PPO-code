from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from alphasched.baselines.types import SolveResult
from alphasched.config.env import ResolvedEnvConfig
from alphasched.core.features import FeatureEncoder
from alphasched.core.instance import Instance
from alphasched.core.simulator import ParallelMachineSimulator
from .policy import Policy, RandomPolicy

_EPS = 1e-12


def _complete_perm(part_num: int, prefix: list[int]) -> np.ndarray:
    missing = [i for i in range(part_num) if i not in set(prefix)]
    return np.array(prefix + missing, dtype=int)


@dataclass(slots=True)
class _Node:
    sim: ParallelMachineSimulator
    seq: list[int]
    logp: float


def _greedy_rollout_from(node: _Node, cfg: ResolvedEnvConfig, encoder: FeatureEncoder, policy: Policy) -> tuple[float, list[int]]:
    sim = node.sim.clone()
    seq = node.seq.copy()
    while True:
        mask = encoder.action_mask(sim)
        valid = np.where(mask)[0]
        if valid.size == 0:
            return sim.final_wt(), seq
        obs = encoder.observation(sim)
        probs = policy.action_probabilities(obs, mask)
        choice = int(valid[int(np.argmax(probs[valid]))])
        out = sim.step(choice)
        seq.append(choice)
        if out.done:
            assert out.wt_final is not None
            return float(out.wt_final), seq


def gpsearch(instance: Instance, cfg: ResolvedEnvConfig, policy: Policy, *, beam_size: int) -> SolveResult:
    """Guided Policy Search (paper): expand top-K by policy prob, evaluate children via rollout WT, keep best WT."""
    start = time.time()
    encoder = FeatureEncoder(cfg)
    part_num = cfg.part_num
    beam_size = int(beam_size)

    forest: list[_Node] = [_Node(sim=ParallelMachineSimulator(instance, cfg.mach_num), seq=[], logp=0.0)]
    best_wt = float("inf")
    best_seq: list[int] = []

    for _ in range(max(part_num - 1, 1)):
        children: list[_Node] = []
        for node in forest:
            obs = encoder.observation(node.sim)
            mask = encoder.action_mask(node.sim)
            probs = policy.action_probabilities(obs, mask)
            valid = np.where(mask)[0]
            if valid.size == 0:
                continue
            k = min(beam_size, int(valid.size))
            top_idx = valid[np.argsort(-probs[valid])[:k]]
            for a in top_idx.tolist():
                p = float(probs[int(a)])
                child_sim = node.sim.clone()
                child_sim.step(int(a))
                children.append(_Node(sim=child_sim, seq=node.seq + [int(a)], logp=node.logp + float(np.log(max(p, _EPS)))))

        # Evaluate each child by greedy rollout WT, then keep best K by WT.
        scored: list[tuple[float, _Node, list[int]]] = []
        for child in children:
            wt, seq = _greedy_rollout_from(child, cfg, encoder, policy)
            scored.append((float(wt), child, seq))
            if float(wt) < best_wt:
                best_wt = float(wt)
                best_seq = seq

        scored.sort(key=lambda t: t[0])
        forest = [t[1] for t in scored[:beam_size]] if scored else []
        if not forest:
            break

    end = time.time()
    return SolveResult(
        best_perm=_complete_perm(part_num, best_seq),
        best_wt=float(best_wt),
        wall_time_sec=end - start,
        extra={"beam_size": beam_size, "policy_name": getattr(policy, "name", "policy")},
    )


def random_search(instance: Instance, cfg: ResolvedEnvConfig, *, beam_size: int, rng: np.random.Generator | None = None) -> SolveResult:
    """Random Search (paper): random top-K expansion + random rollout evaluation."""
    rng = rng or np.random.default_rng()
    policy = RandomPolicy(cfg.part_num)
    start = time.time()

    encoder = FeatureEncoder(cfg)
    forest: list[_Node] = [_Node(sim=ParallelMachineSimulator(instance, cfg.mach_num), seq=[], logp=0.0)]
    best_wt = float("inf")
    best_seq: list[int] = []

    for _ in range(max(cfg.part_num - 1, 1)):
        children: list[_Node] = []
        for node in forest:
            mask = encoder.action_mask(node.sim)
            valid = np.where(mask)[0]
            if valid.size == 0:
                continue
            k = min(int(beam_size), int(valid.size))
            chosen = rng.choice(valid, size=k, replace=False)
            for a in chosen.tolist():
                child_sim = node.sim.clone()
                child_sim.step(int(a))
                children.append(_Node(sim=child_sim, seq=node.seq + [int(a)], logp=0.0))

        scored: list[tuple[float, _Node, list[int]]] = []
        for child in children:
            sim = child.sim.clone()
            seq = child.seq.copy()
            while True:
                valid = sim.available_actions()
                if valid.size == 0:
                    wt = sim.final_wt()
                    scored.append((float(wt), child, seq))
                    if float(wt) < best_wt:
                        best_wt = float(wt)
                        best_seq = seq
                    break
                a = int(rng.choice(valid))
                out = sim.step(a)
                seq.append(a)
                if out.done:
                    assert out.wt_final is not None
                    wt = float(out.wt_final)
                    scored.append((wt, child, seq))
                    if wt < best_wt:
                        best_wt = wt
                        best_seq = seq
                    break

        scored.sort(key=lambda t: t[0])
        forest = [t[1] for t in scored[: int(beam_size)]] if scored else []
        if not forest:
            break

    end = time.time()
    return SolveResult(
        best_perm=_complete_perm(cfg.part_num, best_seq),
        best_wt=float(best_wt),
        wall_time_sec=end - start,
        extra={"beam_size": int(beam_size), "policy_name": "random"},
    )

