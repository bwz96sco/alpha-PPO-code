from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alphasched.config.env import ObsConfig, ResolvedEnvConfig
from .simulator import ParallelMachineSimulator

_EPS = 1e-10


@dataclass(frozen=True, slots=True)
class FeatureEncoder:
    cfg: ResolvedEnvConfig
    obs_cfg: ObsConfig = ObsConfig(include_rule_features=True)

    def observation(self, sim: ParallelMachineSimulator) -> np.ndarray:
        """Return obs in shape (C, W, H) matching legacy reshape convention."""
        width, height = self.cfg.reshape_width_height
        area = width * height

        cur = float(np.min(sim.mach))
        hours = sim.part[:, 0:1]
        deadline = sim.part[:, 1:2]
        weight = sim.part[:, 2:3]

        deadline_cur = (deadline - cur) * (deadline > 0)
        deadline_cur_hours = deadline_cur - hours
        slack = -(deadline_cur_hours) / (hours + _EPS)

        if self.obs_cfg.include_rule_features:
            delta = -np.maximum(deadline_cur_hours, 0) / (hours + _EPS)
            wspt = -weight / (hours + _EPS)
            wmdd = np.maximum(hours, deadline_cur) / (weight + _EPS)
            atc = np.exp(delta / float(self.cfg.h)) * weight / (hours + _EPS)
            wco = np.maximum(1.0 + delta / float(self.cfg.kt), 0.0) * weight / (hours + _EPS)
            part_features = np.concatenate(
                (hours, weight, deadline_cur, deadline_cur_hours, slack, wmdd, wspt, atc, wco),
                axis=1,
            )
        else:
            part_features = np.concatenate((hours, weight, deadline_cur, deadline_cur_hours, slack), axis=1)

        job_dim = part_features.shape[1]
        part_obs_flat = np.zeros((job_dim, area), dtype=np.float32)
        n = part_features.shape[0]
        part_obs_flat[:, : min(n, area)] = part_features.T[:, : min(n, area)].astype(np.float32, copy=False)
        part_obs = part_obs_flat.reshape(job_dim, width, height)

        mach_time = (sim.mach - cur).astype(np.float32, copy=False)
        mach_arr = np.zeros(area, dtype=np.float32)
        mach_arr[: min(sim.mach_num, area)] = mach_time[: min(sim.mach_num, area)]
        mach_obs = mach_arr.reshape(1, width, height)

        obs = np.concatenate((part_obs, mach_obs), axis=0)
        return obs.astype(np.float32, copy=False)

    def action_mask(self, sim: ParallelMachineSimulator) -> np.ndarray:
        return sim.action_mask().astype(bool, copy=False)

