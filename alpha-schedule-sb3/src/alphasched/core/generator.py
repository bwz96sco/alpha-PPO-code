from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from alphasched.config.env import Mode, ResolvedEnvConfig
from .instance import Instance

RngBackend = Literal["legacy_mt19937", "pcg64"]


def _base_seed(cfg: ResolvedEnvConfig, mode: Mode) -> int:
    if mode == "train":
        return cfg.train_seed
    if mode == "val":
        return cfg.val_seed
    if mode == "test":
        return cfg.test_seed
    raise ValueError(f"unknown mode: {mode!r}")


@dataclass(frozen=True, slots=True)
class InstanceGenerator:
    cfg: ResolvedEnvConfig
    rng_backend: RngBackend = "legacy_mt19937"

    def generate(self, *, mode: Mode, instance_id: int, seed_offset: int = 0) -> Instance:
        """Generate one instance deterministically.

        Seed mapping is designed to match the legacy repo:
        - test instances use seeds: test_seed + instance_id
        - val instances use seeds:  val_seed + instance_id
        """
        if instance_id < 0:
            raise ValueError(f"instance_id must be >= 0, got {instance_id}")
        seed = int(_base_seed(self.cfg, mode) + instance_id + seed_offset)

        if self.rng_backend == "legacy_mt19937":
            rng = np.random.RandomState(seed)
            job_time = rng.randint(self.cfg.min_time, self.cfg.max_time, size=self.cfg.part_num)
            tight_time = job_time * (1.0 + rng.rand(self.cfg.part_num) * self.cfg.tight)
            weight = rng.randint(1, self.cfg.priority_max, size=self.cfg.part_num)
        elif self.rng_backend == "pcg64":
            rng = np.random.default_rng(seed)
            job_time = rng.integers(self.cfg.min_time, self.cfg.max_time, size=self.cfg.part_num, endpoint=False)
            tight_time = job_time * (1.0 + rng.random(self.cfg.part_num) * self.cfg.tight)
            weight = rng.integers(1, self.cfg.priority_max, size=self.cfg.part_num, endpoint=False)
        else:
            raise ValueError(f"unknown rng_backend: {self.rng_backend!r}")

        deadline = np.floor(tight_time)
        jobs = np.vstack((job_time, deadline, weight)).T.astype(np.float64, copy=False)
        return Instance(jobs=jobs, seed=seed)

