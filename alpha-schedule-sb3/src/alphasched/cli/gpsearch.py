from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from alphasched.config.env import EnvConfig
from alphasched.core.generator import InstanceGenerator
from alphasched.logging import MetricsWriter, create_run_dir
from alphasched.rl.sb3_policy import SB3MaskablePolicy
from alphasched.search.beam import beam_search
from alphasched.search.gpsearch import gpsearch, random_search
from alphasched.search.rollout import greedy_rollout


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run search methods (GPSearch / Beam / Rollout) on the test set.")
    p.add_argument("--algo", type=str, default="gpsearch", choices=["gpsearch", "beam", "rollout", "random_search"])
    p.add_argument("--model-path", type=str, help="Required for gpsearch/beam/rollout")
    p.add_argument("--part-num", type=int, default=65)
    p.add_argument("--mach-num", type=int, default=-1, help="-1 means auto")
    p.add_argument("--dist-type", type=str, default="h", choices=["h", "m", "l"])
    p.add_argument("--test-num", type=int, default=100)
    p.add_argument("--beam-size", type=int, default=10)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--run-name", type=str, default="search")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    env_cfg = EnvConfig(part_num=args.part_num, dist_type=args.dist_type, mach_num=None if args.mach_num <= 0 else args.mach_num)
    resolved = env_cfg.resolved()
    gen = InstanceGenerator(resolved, rng_backend="legacy_mt19937")

    if args.algo in ("gpsearch", "beam", "rollout"):
        if not args.model_path:
            raise ValueError("--model-path is required for this algo")
        policy = SB3MaskablePolicy(Path(args.model_path), device=str(args.device))
    else:
        policy = None

    run = create_run_dir(base_dir=args.runs_dir, name=args.run_name)
    t0 = time.time()
    wt_list: list[float] = []

    with MetricsWriter(run.metrics_path) as writer:
        for instance_id in range(int(args.test_num)):
            inst = gen.generate(mode="test", instance_id=instance_id)
            if args.algo == "gpsearch":
                result = gpsearch(inst, resolved, policy, beam_size=int(args.beam_size))  # type: ignore[arg-type]
                best_wt = float(result.best_wt)
            elif args.algo == "beam":
                result = beam_search(inst, resolved, policy, beam_size=int(args.beam_size))  # type: ignore[arg-type]
                best_wt = float(result.best_wt)
            elif args.algo == "rollout":
                rr = greedy_rollout(inst, resolved, policy)  # type: ignore[arg-type]
                best_wt = float(rr.wt)
            elif args.algo == "random_search":
                result = random_search(inst, resolved, beam_size=int(args.beam_size))
                best_wt = float(result.best_wt)
            else:
                raise ValueError(f"unknown algo: {args.algo}")

            wt_list.append(best_wt)
            writer.write(
                {
                    "run_id": run.run_id,
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "algo": args.algo,
                    "mode": "test",
                    "part_num": resolved.part_num,
                    "mach_num": resolved.mach_num,
                    "dist_type": resolved.dist_type,
                    "seed": inst.seed,
                    "instance_id": instance_id,
                    "wt": best_wt,
                    "episode_reward": -best_wt,
                    "steps": None,
                    "wall_time_sec": time.time() - t0,
                    "policy_name": getattr(policy, "name", "random") if policy is not None else "random",
                    "k": int(args.beam_size),
                    "beam_size": int(args.beam_size),
                }
            )

    mean_wt = float(np.mean(wt_list)) if wt_list else float("nan")
    print(f"{args.algo}: evaluated {len(wt_list)} instances. mean WT = {mean_wt:.5f}")
    print(f"Metrics written to: {run.metrics_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
