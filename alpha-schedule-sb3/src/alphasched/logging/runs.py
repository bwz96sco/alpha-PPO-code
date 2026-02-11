from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    run_dir: Path

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.csv"

    @property
    def tb_dir(self) -> Path:
        return self.run_dir / "tb"


def create_run_dir(*, base_dir: str | Path = "runs", name: str | None = None) -> RunContext:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    suffix = uuid4().hex[:8]
    run_id = f"{ts}-{suffix}" if not name else f"{ts}-{name}-{suffix}"
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Best-effort "latest" marker.
    try:
        latest = base / "latest"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(run_dir.name)
    except OSError:
        # Fall back to a text file on platforms that dislike symlinks.
        (base / "latest.txt").write_text(run_dir.name, encoding="utf-8")

    return RunContext(run_id=run_id, run_dir=run_dir)

