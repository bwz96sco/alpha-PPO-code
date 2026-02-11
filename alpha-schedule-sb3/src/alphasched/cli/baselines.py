from __future__ import annotations

import argparse
import time

import numpy as np

from alphasched.baselines import solve_bbo, solve_ga, solve_pso, solve_rule
from alphasched.config.env import EnvConfig
from alphasched.core.generator import InstanceGenerator
from alphasched.logging import MetricsWriter, create_run_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run baselines (rules/GA/BBO/PSO) on the deterministic test set.")
    p.add_argument("--algo", type=str, required=True, choices=["rule", "ga", "bbo", "pso"])
    p.add_argument("--rule", type=str, default="wspt", choices=["spt", "mp", "wspt", "wmdd", "atc", "wco"])
    p.add_argument("--part-num", type=int, default=65)
    p.add_argument("--mach-num", type=int, default=-1, help="-1 means auto")
    p.add_argument("--dist-type", type=str, default="h", choices=["h", "m", "l"])
    p.add_argument("--test-num", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--popu", type=int, default=200)
    p.add_argument("--iter", type=int, default=400)

    # PSO
    p.add_argument("--c1", type=float, default=2.0)
    p.add_argument("--c2", type=float, default=2.1)
    p.add_argument("--w-start", type=float, default=0.9)
    p.add_argument("--w-end", type=float, default=0.4)

    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--run-name", type=str, default="baseline")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    env_cfg = EnvConfig(part_num=args.part_num, dist_type=args.dist_type, mach_num=None if args.mach_num <= 0 else args.mach_num)
    resolved = env_cfg.resolved()
    gen = InstanceGenerator(resolved, rng_backend="legacy_mt19937")
    rng = np.random.default_rng(int(args.seed))

    run = create_run_dir(base_dir=args.runs_dir, name=args.run_name)
    t0 = time.time()
    wt_list: list[float] = []

    with MetricsWriter(run.metrics_path) as writer:
        for instance_id in range(int(args.test_num)):
            inst = gen.generate(mode="test", instance_id=instance_id)
            if args.algo == "rule":
                result = solve_rule(inst, resolved, args.rule)  # type: ignore[arg-type]
                pop_size = None
                iters = None
            elif args.algo == "ga":
                result = solve_ga(inst, resolved, pop_size=int(args.popu), iters=int(args.iter), rng=rng)
                pop_size = int(args.popu)
                iters = int(args.iter)
            elif args.algo == "bbo":
                result = solve_bbo(inst, resolved, pop_size=int(args.popu), iters=int(args.iter), rng=rng)
                pop_size = int(args.popu)
                iters = int(args.iter)
            elif args.algo == "pso":
                result = solve_pso(
                    inst,
                    resolved,
                    pop_size=int(args.popu),
                    iters=int(args.iter),
                    c1=float(args.c1),
                    c2=float(args.c2),
                    w_start=float(args.w_start),
                    w_end=float(args.w_end),
                    rng=rng,
                )
                pop_size = int(args.popu)
                iters = int(args.iter)
            else:
                raise ValueError(f"unknown algo: {args.algo}")

            wt_list.append(float(result.best_wt))
            writer.write(
                {
                    "run_id": run.run_id,
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "algo": args.algo if args.algo != "rule" else f"rule:{args.rule}",
                    "mode": "test",
                    "part_num": resolved.part_num,
                    "mach_num": resolved.mach_num,
                    "dist_type": resolved.dist_type,
                    "seed": inst.seed,
                    "instance_id": instance_id,
                    "wt": float(result.best_wt),
                    "episode_reward": -float(result.best_wt),
                    "steps": result.extra.get("steps"),
                    "wall_time_sec": time.time() - t0,
                    "policy_name": None,
                    "pop_size": pop_size,
                    "iters": iters,
                }
            )

    mean_wt = float(np.mean(wt_list)) if wt_list else float("nan")
    print(f"{args.algo}: evaluated {len(wt_list)} instances. mean WT = {mean_wt:.5f}")
    print(f"Metrics written to: {run.metrics_path}")


if __name__ == "__main__":  # pragma: no cover
    main()

