from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym  # type: ignore[no-redef]

from alphasched.config.env import EnvConfig, Mode, ObsConfig
from alphasched.core.features import FeatureEncoder
from alphasched.core.generator import InstanceGenerator, RngBackend
from alphasched.core.simulator import ParallelMachineSimulator


@dataclass(frozen=True, slots=True)
class EnvParams:
    env_cfg: EnvConfig
    obs_cfg: ObsConfig = ObsConfig(include_rule_features=True)
    mode: Mode = "train"
    instance_id_start: int = 0
    instance_id_step: int = 1
    seed_offset: int = 0
    rng_backend: RngBackend = "legacy_mt19937"


class ParallelMachineTWTEnv(gym.Env):
    """Gymnasium environment for parallel machine scheduling with TWT objective.

    Observation and reward match the paper + legacy PPO setup:
    - action: pick a job index (Discrete(N))
    - reward: negative weighted tardiness for the scheduled job (Eq. 4)
    - when only one job remains, it is scheduled automatically (legacy behavior)
    """

    metadata = {"render_modes": []}

    def __init__(self, params: EnvParams):
        super().__init__()
        self.params = params
        self.cfg = params.env_cfg.resolved()

        self.generator = InstanceGenerator(self.cfg, rng_backend=params.rng_backend)
        self.encoder = FeatureEncoder(self.cfg, params.obs_cfg)

        width, height = self.cfg.reshape_width_height
        channels = params.obs_cfg.channel_dim

        self.action_space = gym.spaces.Discrete(self.cfg.part_num)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(channels, width, height),
            dtype=np.float32,
        )

        self._mode: Mode = params.mode
        self._instance_id = int(params.instance_id_start)
        self._instance_step = int(params.instance_id_step)
        self._seed_offset = int(params.seed_offset)

        self._sim: ParallelMachineSimulator | None = None
        self._episode_reward = 0.0
        self._current_instance_id: int | None = None
        self._current_seed: int | None = None

    # ---- SB3 action-mask integration ----
    def action_mask(self) -> np.ndarray:
        assert self._sim is not None, "call reset() first"
        return self.encoder.action_mask(self._sim)

    # ---- Gym API ----
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}

        instance_id = int(options.get("instance_id", self._instance_id))
        mode = options.get("mode", self._mode)
        if mode not in ("train", "val", "test"):
            raise ValueError(f"invalid mode: {mode!r}")
        self._mode = mode  # allow mode switching per reset

        instance = self.generator.generate(mode=self._mode, instance_id=instance_id, seed_offset=self._seed_offset)
        self._sim = ParallelMachineSimulator(instance, self.cfg.mach_num)
        self._episode_reward = 0.0
        self._current_instance_id = instance_id
        self._current_seed = instance.seed

        # Advance internal counter only if caller didn't explicitly pin instance_id.
        if "instance_id" not in options:
            self._instance_id += self._instance_step

        obs = self.encoder.observation(self._sim)
        info = {"instance_id": instance_id, "seed": instance.seed, "mode": self._mode}
        return obs, info

    def step(self, action: int):
        assert self._sim is not None, "call reset() first"

        out = self._sim.step(int(action))
        self._episode_reward += float(out.reward)

        obs = self.encoder.observation(self._sim)
        terminated = bool(out.done)
        truncated = False

        info: dict[str, Any] = {
            "invalid_action": bool(out.invalid_action),
            "episode_reward": float(self._episode_reward),
            "instance_id": self._current_instance_id,
            "seed": self._current_seed,
            "mode": self._mode,
        }
        if out.done and out.wt_final is not None:
            info["wt_final"] = float(out.wt_final)
            # legacy-style hook: `info["episode"]["r"]` was used as "grade"/WT
            info["episode"] = {"r": float(out.wt_final)}

        return obs, float(out.reward), terminated, truncated, info
