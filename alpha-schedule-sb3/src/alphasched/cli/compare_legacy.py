from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from alphasched.config.env import EnvConfig
from alphasched.core.generator import InstanceGenerator
from alphasched.core.simulator import evaluate_permutation


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare new generator/simulator vs legacy implementation.")
    p.add_argument("--part-num", type=int, default=65)
    p.add_argument("--mach-num", type=int, default=-1, help="-1 means auto")
    p.add_argument("--dist-type", type=str, default="h", choices=["h", "m", "l"])
    p.add_argument("--instances", type=int, default=10, help="Number of instances to compare (starting at id=0)")
    p.add_argument("--mode", type=str, default="test", choices=["test", "val"])
    p.add_argument("--seed", type=int, default=0, help="Seed for generating comparison permutations")
    return p


def _repo_root() -> Path:
    # alpha-schedule-sb3/src/alphasched/cli/compare_legacy.py -> repo root is 4 parents up
    return Path(__file__).resolve().parents[4]


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    # --- new impl setup ---
    env_cfg = EnvConfig(part_num=args.part_num, dist_type=args.dist_type, mach_num=None if args.mach_num <= 0 else args.mach_num)
    resolved = env_cfg.resolved()
    gen = InstanceGenerator(resolved, rng_backend="legacy_mt19937")

    # --- legacy import setup ---
    repo_root = _repo_root()
    legacy_root = repo_root / "data-mcts" / "mctsAlphaV0.080"
    if not legacy_root.exists():
        raise FileNotFoundError(f"Legacy path not found: {legacy_root}")
    sys.path.insert(0, str(legacy_root))

    from venvs.EnvirConf import config as legacy_config  # type: ignore
    from venvs.scheduler import Scheduler as LegacyScheduler  # type: ignore

    legacy_config.updateParam(partNum=args.part_num, machNum=args.mach_num, distType=args.dist_type)
    legacy_sched = LegacyScheduler(args.mode, 0)

    rng = np.random.default_rng(int(args.seed))
    mismatches = 0

    for i in range(int(args.instances)):
        inst = gen.generate(mode=args.mode, instance_id=i)
        legacy_sched.reset(True)

        legacy_jobs = legacy_sched.part.copy()
        if not np.allclose(legacy_jobs, inst.jobs):
            mismatches += 1
            max_abs = float(np.max(np.abs(legacy_jobs - inst.jobs)))
            print(f"[instance {i}] jobs mismatch (max_abs={max_abs})")

        perm = rng.permutation(resolved.part_num)
        legacy_wt = float(legacy_sched.scheStatic(perm.tolist()))
        new_wt = float(evaluate_permutation(inst, resolved.mach_num, perm.tolist()))
        if abs(legacy_wt - new_wt) > 1e-6:
            mismatches += 1
            print(f"[instance {i}] WT mismatch: legacy={legacy_wt} new={new_wt}")

    if mismatches == 0:
        print("OK: new generator/simulator matches legacy on all checked instances.")
    else:
        raise SystemExit(f"Found {mismatches} mismatches.")


if __name__ == "__main__":  # pragma: no cover
    main()

