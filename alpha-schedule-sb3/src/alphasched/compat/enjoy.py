from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alphasched.cli.eval_ppo import main as eval_main


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Legacy-compatible wrapper for PPO evaluation (maps to SB3).")
    p.add_argument("--env-name", type=str, default="sci1-my")
    p.add_argument("--part-num", type=int, default=65)
    p.add_argument("--mach-num", type=int, default=-1)
    p.add_argument("--dist-type", type=str, default="h", choices=["h", "m", "l"])
    p.add_argument("--test-num", type=int, default=100)
    p.add_argument("--num-processes", type=int, default=8)
    p.add_argument("--load-dir", type=str, default="", help="Directory containing `model.zip`")
    p.add_argument("--model-path", type=str, default="", help="Explicit `.zip` path (preferred)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    model_path = args.model_path
    if not model_path and args.load_dir:
        candidate = Path(args.load_dir) / "model.zip"
        if candidate.exists():
            model_path = str(candidate)
    if not model_path:
        raise ValueError("Need --model-path or --load-dir containing model.zip")

    mapped = [
        "--model-path",
        model_path,
        "--part-num",
        str(args.part_num),
        "--mach-num",
        str(args.mach_num),
        "--dist-type",
        str(args.dist_type),
        "--test-num",
        str(args.test_num),
        "--num-envs",
        str(args.num_processes),
        "--device",
        str(args.device),
    ]
    eval_main(mapped)


if __name__ == "__main__":  # pragma: no cover
    main()

