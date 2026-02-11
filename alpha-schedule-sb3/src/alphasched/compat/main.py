from __future__ import annotations

import argparse
import sys

from alphasched.cli.train_ppo import main as train_main


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Legacy-compatible wrapper for PPO training (maps to SB3 MaskablePPO).")
    p.add_argument("--num-processes", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=1024)
    p.add_argument("--num-mini-batch", type=int, default=32)
    p.add_argument("--ppo-epoch", type=int, default=4)

    p.add_argument("--env-name", type=str, default="sci1-my")
    p.add_argument("--part-num", type=int, default=65)
    p.add_argument("--mach-num", type=int, default=-1)
    p.add_argument("--dist-type", type=str, default="h", choices=["h", "m", "l"])

    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--clip-param", type=float, default=0.10)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--value-loss-coef", type=float, default=0.50)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.50)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--run-hours", type=float, default=0.0)
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--excel-save", action="store_true", help="Ignored (use CSV + export script).")
    p.add_argument("--policy-net", type=str, default="resnet", choices=["resnet", "simconv"])
    p.add_argument("--resblocks", type=int, default=9)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    if args.excel_save:
        print("warning: --excel-save is ignored; use unified CSV logs and `alphasched-export-excel`.", file=sys.stderr)

    # Map legacy mini-batches to SB3 batch_size.
    total_batch = int(args.num_processes) * int(args.num_steps)
    batch_size = max(1, total_batch // max(1, int(args.num_mini_batch)))

    mapped = [
        "--part-num",
        str(args.part_num),
        "--mach-num",
        str(args.mach_num),
        "--dist-type",
        str(args.dist_type),
        "--seed",
        str(args.seed),
        "--num-envs",
        str(args.num_processes),
        "--n-steps",
        str(args.num_steps),
        "--batch-size",
        str(batch_size),
        "--n-epochs",
        str(args.ppo_epoch),
        "--learning-rate",
        str(args.lr),
        "--gamma",
        str(args.gamma),
        "--clip-range",
        str(args.clip_param),
        "--gae-lambda",
        str(args.gae_lambda),
        "--vf-coef",
        str(args.value_loss_coef),
        "--ent-coef",
        str(args.entropy_coef),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--policy-net",
        str(args.policy_net),
        "--resblocks",
        str(args.resblocks),
        "--device",
        str(args.device),
        "--total-timesteps",
        str(args.total_timesteps),
    ]
    if args.run_hours and float(args.run_hours) > 0:
        mapped += ["--run-hours", str(args.run_hours)]

    train_main(mapped)


if __name__ == "__main__":  # pragma: no cover
    main()

