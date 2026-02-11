from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from alphasched.config.env import ResolvedEnvConfig, Mode
from alphasched.logging.metrics import MetricsWriter


@dataclass(frozen=True, slots=True)
class EpisodeCsvConfig:
    run_id: str
    algo: str
    mode: Mode
    policy_name: str
    env_cfg: ResolvedEnvConfig


class EpisodeCsvCallback(BaseCallback):
    """Write one CSV row per finished episode using env `info` fields."""

    def __init__(self, writer: MetricsWriter, cfg: EpisodeCsvConfig):
        super().__init__()
        self._writer = writer
        self._cfg = cfg
        self._t0 = time.time()

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if not done:
                continue
            wt = info.get("wt_final")
            if wt is None:
                continue
            self._writer.write(
                {
                    "run_id": self._cfg.run_id,
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "algo": self._cfg.algo,
                    "mode": self._cfg.mode,
                    "part_num": self._cfg.env_cfg.part_num,
                    "mach_num": self._cfg.env_cfg.mach_num,
                    "dist_type": self._cfg.env_cfg.dist_type,
                    "seed": info.get("seed"),
                    "instance_id": info.get("instance_id"),
                    "wt": float(wt),
                    "episode_reward": float(info.get("episode_reward", 0.0)),
                    "steps": None,
                    "wall_time_sec": time.time() - self._t0,
                    "policy_name": self._cfg.policy_name,
                }
            )
        return True


class WallTimeLimitCallback(BaseCallback):
    def __init__(self, *, max_seconds: float):
        super().__init__()
        self._max_seconds = float(max_seconds)
        self._t0 = time.time()

    def _on_step(self) -> bool:
        if (time.time() - self._t0) >= self._max_seconds:
            return False
        return True

