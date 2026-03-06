from __future__ import annotations

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


def _write_latest_marker(base: Path, run_name: str) -> None:
    latest = base / "latest"
    latest_txt = base / "latest.txt"

    if latest.exists() or latest.is_symlink():
        latest.unlink()

    try:
        latest.symlink_to(run_name)
        if latest_txt.exists():
            latest_txt.unlink()
    except OSError:
        latest_txt.write_text(run_name, encoding="utf-8")


def create_run_dir(*, base_dir: str | Path = "runs", name: str | None = None) -> RunContext:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    suffix = uuid4().hex[:8]
    run_id = f"{ts}-{suffix}" if not name else f"{ts}-{name}-{suffix}"
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(run_id=run_id, run_dir=run_dir)


def update_latest_run(*, base_dir: str | Path = "runs", run_dir: str | Path) -> None:
    base = Path(base_dir)
    run_path = Path(run_dir)
    if run_path.parent.resolve() != base.resolve():
        raise ValueError(f"run_dir must be a direct child of base_dir: {run_path} vs {base}")
    _write_latest_marker(base, run_path.name)
