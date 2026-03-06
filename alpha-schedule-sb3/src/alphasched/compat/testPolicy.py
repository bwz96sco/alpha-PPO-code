from __future__ import annotations

import argparse
from pathlib import Path

from alphasched.config.env import EnvConfig
from alphasched.cli.gpsearch import main as search_main


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Legacy-compatible wrapper for testPolicy.py (search inference).")
    p.add_argument("--mode", type=str, default="mcts_policy", help="legacy mode string")
    p.add_argument("--env-name", type=str, default="mach-ex")
    p.add_argument("--resblock-num", type=int, default=9)
    p.add_argument("--beam-size", type=int, default=10)
    p.add_argument("--test-num", type=int, default=100)
    p.add_argument("--part-num", type=int, default=65)
    p.add_argument("--mach-num", type=int, default=-1)
    p.add_argument("--dist-type", type=str, default="h", choices=["h", "m", "l"])
    p.add_argument("--model-path", type=str, default=None, help="Default: runs/<part>-<mach>-<dist>/train-ppo/latest/model.zip")
    p.add_argument("--device", type=str, default="cpu")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    env_cfg = EnvConfig(part_num=args.part_num, dist_type=args.dist_type, mach_num=None if args.mach_num <= 0 else args.mach_num)
    resolved = env_cfg.resolved()
    env_key = f"{resolved.part_num}-{resolved.mach_num}-{resolved.dist_type}"
    default_model_path = Path("runs") / env_key / "train-ppo" / "latest" / "model.zip"

    mode = args.mode.lower()
    if "random" in mode:
        algo = "random_search"
        model_path = None
    elif "pure" in mode:
        algo = "rollout"
        model_path = args.model_path or default_model_path
    elif "beam" in mode:
        algo = "beam"
        model_path = args.model_path or default_model_path
    else:
        algo = "gpsearch"
        model_path = args.model_path or default_model_path

    mapped = [
        "--algo",
        algo,
        "--part-num",
        str(args.part_num),
        "--mach-num",
        str(args.mach_num),
        "--dist-type",
        str(args.dist_type),
        "--test-num",
        str(args.test_num),
        "--beam-size",
        str(args.beam_size),
        "--device",
        str(args.device),
    ]
    if model_path:
        mapped += ["--model-path", str(model_path)]

    search_main(mapped)


if __name__ == "__main__":  # pragma: no cover
    main()
