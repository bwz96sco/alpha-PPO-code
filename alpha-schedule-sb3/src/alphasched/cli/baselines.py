from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from alphasched.baselines import solve_bbo, solve_ga, solve_pso, solve_rule
from alphasched.config.env import EnvConfig
from alphasched.core.generator import InstanceGenerator
from alphasched.logging import MetricsWriter, create_run_dir


def _legacy_rng_after_instance(cfg, seed: int) -> np.random.RandomState:
    """Replicate legacy coupling: solver RNG == MT19937 state after createPart().

    Legacy flow per instance:
        np.random.seed(seed)
        job_time  = np.random.randint(min_time, max_time, size=part_num)
        tight     = job_time * (1 + np.random.rand(part_num) * tight_factor)
        weight    = np.random.randint(1, priority_max, size=part_num)
        # ... solver starts consuming from here ...
    """
    rng = np.random.RandomState(int(seed))
    rng.randint(cfg.min_time, cfg.max_time, size=cfg.part_num)   # job_time
    rng.rand(cfg.part_num)                                        # tight factor
    rng.randint(1, cfg.priority_max, size=cfg.part_num)           # weight
    return rng


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run baselines (rules/GA/BBO/PSO) on the deterministic test set.")
    p.add_argument("--algo", type=str, required=True, choices=["rule", "ga", "bbo", "pso"])
    p.add_argument("--rule", type=str, default="wspt", choices=["spt", "mp", "wspt", "wmdd", "atc", "wco"])
    p.add_argument("--part-num", type=int, default=65)
    p.add_argument("--mach-num", type=int, default=-1, help="-1 means auto")
    p.add_argument("--dist-type", type=str, default="h", choices=["h", "m", "l"])
    p.add_argument("--test-num", type=int, default=100)
    p.add_argument(
        "--algo-seed",
        "--seed",
        dest="algo_seed",
        type=int,
        default=0,
        help="Seed for stochastic solvers when --seed-mode=independent. Alias: --seed",
    )
    p.add_argument(
        "--seed-mode",
        type=str,
        default="independent",
        choices=["independent", "legacy_coupled"],
        help=(
            "independent: use --algo-seed once for solver RNG; "
            "legacy_coupled: per-instance MT19937 RNG state after instance-generation draws (matches legacy repo/paper)."
        ),
    )

    p.add_argument("--popu", type=int, default=200)
    p.add_argument("--iter", type=int, default=400)

    # PSO
    p.add_argument("--c1", type=float, default=2.0)
    p.add_argument("--c2", type=float, default=2.1)
    p.add_argument("--w-start", type=float, default=0.9)
    p.add_argument("--w-end", type=float, default=0.4)

    p.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Root run directory. Runs are created under <runs-dir>/<part>-<mach>-<dist>/baselines/",
    )
    p.add_argument("--run-name", type=str, default="baseline")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    env_cfg = EnvConfig(part_num=args.part_num, dist_type=args.dist_type, mach_num=None if args.mach_num <= 0 else args.mach_num)
    resolved = env_cfg.resolved()
    gen = InstanceGenerator(resolved, rng_backend="legacy_mt19937")
    shared_rng = np.random.default_rng(int(args.algo_seed)) if args.seed_mode == "independent" else None

    env_key = f"{resolved.part_num}-{resolved.mach_num}-{resolved.dist_type}"
    run = create_run_dir(base_dir=Path(args.runs_dir) / env_key / "baselines", name=args.run_name)
    t0 = time.time()
    wt_list: list[float] = []

    with MetricsWriter(run.metrics_path) as writer:
        for instance_id in range(int(args.test_num)):
            inst = gen.generate(mode="test", instance_id=instance_id)
            rng = _legacy_rng_after_instance(resolved, inst.seed) if args.seed_mode == "legacy_coupled" else shared_rng
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
            print(f"{instance_id + 1}  bestGrade = {result.best_wt}")
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

    elapsed = time.time() - t0
    mean_wt = float(np.mean(wt_list)) if wt_list else float("nan")
    print(f"num_playouts:, min: {min(wt_list)}, average: {mean_wt}, max:{max(wt_list)}")
    print(f"Execution Time:  {elapsed:.2f}")
    print(f"Metrics written to: {run.metrics_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
