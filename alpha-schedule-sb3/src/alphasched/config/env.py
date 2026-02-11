from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

DistType = Literal["h", "m", "l"]
Mode = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class DistributionParams:
    tight: float
    priority_max: int
    max_time: int


# Matches the legacy repo (see `*/venvs/EnvirConf.py`).
DISTRIBUTIONS: dict[DistType, DistributionParams] = {
    "h": DistributionParams(tight=0.3, priority_max=20, max_time=200),
    "m": DistributionParams(tight=0.5, priority_max=12, max_time=125),
    "l": DistributionParams(tight=0.65, priority_max=6, max_time=50),
}


def default_mach_num(part_num: int) -> int:
    group = (part_num - 5) // 10
    return group * 5


def reshape_dims(part_num: int) -> tuple[int, int]:
    """Feature reshape dims (paper Eq. 5–6).

    Legacy code effectively uses N//5 for N divisible by 5. We support any N
    by padding to a 5xk grid when needed.
    """
    if part_num <= 0:
        raise ValueError(f"part_num must be positive, got {part_num}")
    k = part_num // 5 if (part_num % 5 == 0) else math.ceil(part_num / 5)
    width = max(k, 5)
    height = min(k, 5)
    return int(width), int(height)


@dataclass(frozen=True, slots=True)
class EnvConfig:
    part_num: int = 65
    dist_type: DistType = "h"
    mach_num: int | None = None

    # Job generation
    min_time: int = 5

    # Seeds (legacy uses test=0, val=1000; train was random)
    train_seed: int = 12345
    val_seed: int = 1000
    test_seed: int = 0

    # Rule coefficients (legacy defaults)
    kt: float = 1.0
    h: float = 1.0

    def resolved(self) -> "ResolvedEnvConfig":
        if self.dist_type not in DISTRIBUTIONS:
            raise ValueError(f"unknown dist_type: {self.dist_type!r}")
        dist = DISTRIBUTIONS[self.dist_type]
        mach_num = self.mach_num if (self.mach_num and self.mach_num > 0) else default_mach_num(self.part_num)
        return ResolvedEnvConfig(
            part_num=self.part_num,
            dist_type=self.dist_type,
            mach_num=mach_num,
            min_time=self.min_time,
            max_time=dist.max_time,
            tight=dist.tight,
            priority_max=dist.priority_max,
            train_seed=self.train_seed,
            val_seed=self.val_seed,
            test_seed=self.test_seed,
            kt=self.kt,
            h=self.h,
        )


@dataclass(frozen=True, slots=True)
class ResolvedEnvConfig:
    part_num: int
    dist_type: DistType
    mach_num: int

    min_time: int
    max_time: int
    tight: float
    priority_max: int

    train_seed: int
    val_seed: int
    test_seed: int

    kt: float
    h: float

    @property
    def reshape_width_height(self) -> tuple[int, int]:
        return reshape_dims(self.part_num)


@dataclass(frozen=True, slots=True)
class ObsConfig:
    include_rule_features: bool = True

    @property
    def job_feature_dim(self) -> int:
        # Base 5 job features + optional 4 rule-inspired coefficients
        return 9 if self.include_rule_features else 5

    @property
    def channel_dim(self) -> int:
        # +1 machine feature
        return self.job_feature_dim + 1

