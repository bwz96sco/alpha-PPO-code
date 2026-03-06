from __future__ import annotations

import shutil
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from alphasched.config.env import ResolvedEnvConfig, Mode
from alphasched.logging import update_latest_run
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


@dataclass(frozen=True, slots=True)
class PeriodicModelSaveConfig:
    run_dir: Path
    latest_base_dir: Path
    every_steps: int


def save_run_model_snapshot(model: Any, *, run_dir: Path, latest_base_dir: Path, checkpoint_name: str | None = None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    current_model_path = run_dir / "model.zip"
    model.save(str(current_model_path))
    update_latest_run(base_dir=latest_base_dir, run_dir=run_dir)

    if checkpoint_name is not None:
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoints_dir / checkpoint_name
        shutil.copy2(current_model_path, checkpoint_path)

    return current_model_path


class PeriodicModelSaveCallback(BaseCallback):
    """Save model checkpoints periodically during training."""

    def __init__(self, cfg: PeriodicModelSaveConfig):
        super().__init__()
        self._cfg = cfg
        self._cfg.run_dir.mkdir(parents=True, exist_ok=True)
        self._last_save_steps = 0

    def _save(self, *, suffix: str) -> None:
        assert self.model is not None
        save_run_model_snapshot(
            self.model,
            run_dir=self._cfg.run_dir,
            latest_base_dir=self._cfg.latest_base_dir,
            checkpoint_name=f"model-{suffix}.zip",
        )

    def _on_step(self) -> bool:
        if self._cfg.every_steps <= 0:
            return True
        if (self.num_timesteps - self._last_save_steps) >= int(self._cfg.every_steps):
            self._last_save_steps = int(self.num_timesteps)
            self._save(suffix=f"t{self.num_timesteps}")
        return True


class TrainingLogCallback(BaseCallback):
    """Print per-update training stats matching legacy output format.

    Output format (every ``log_interval`` updates):
        Time MM-DD HHh MMm SSs: Updates N, num timesteps T, Last K training episodes:
        entropy/value/policy/loss E/V/P/L, min/mean/max reward min/mean/max
    """

    def __init__(self, *, log_interval: int = 1, deque_size: int = 100):
        super().__init__()
        self._log_interval = int(log_interval)
        self._deque_size = int(deque_size)
        self._episode_rewards: deque[float] = deque(maxlen=self._deque_size)
        self._t0 = 0.0
        self._updates = 0

    def _on_training_start(self) -> None:
        self._t0 = time.time()

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if not done:
                continue
            ep_info = info.get("episode")
            if ep_info is not None:
                self._episode_rewards.append(ep_info["r"])
        return True

    def _on_rollout_end(self) -> None:
        self._updates += 1
        if self._updates % self._log_interval != 0:
            return
        if len(self._episode_rewards) < 2:
            return

        running_time = time.time() - self._t0
        total_timesteps = self.num_timesteps

        # Get losses from SB3 logger
        entropy = self.model.logger.name_to_value.get("train/entropy_loss", 0.0)
        value_loss = self.model.logger.name_to_value.get("train/value_loss", 0.0)
        policy_loss = self.model.logger.name_to_value.get("train/policy_gradient_loss", 0.0)
        loss = self.model.logger.name_to_value.get("train/loss", 0.0)

        rewards = np.array(self._episode_rewards)
        print(
            f"Time {time.strftime('%m-%d %Hh %Mm %Ss', time.gmtime(running_time))}: "
            f"Updates {self._updates}, num timesteps {total_timesteps}, "
            f"Last {len(self._episode_rewards)} training episodes:\n"
            f"entropy/value/policy/loss {entropy:.3f}/{value_loss:.3f}/{policy_loss:.3f}/{loss:.4f}, "
            f"min/mean/max reward {rewards.min():.3f}/{rewards.mean():.3f}/{rewards.max():.3f}\n"
        )
