from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_FIELDS: tuple[str, ...] = (
    "run_id",
    "timestamp_utc",
    "algo",
    "mode",
    "part_num",
    "mach_num",
    "dist_type",
    "seed",
    "instance_id",
    "wt",
    "episode_reward",
    "steps",
    "wall_time_sec",
    "policy_name",
    "k",
    "beam_size",
    "pop_size",
    "iters",
)


@dataclass(slots=True)
class MetricsWriter:
    path: Path
    fieldnames: tuple[str, ...] = DEFAULT_FIELDS

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=list(self.fieldnames), extrasaction="ignore")
        if self.path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "MetricsWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

