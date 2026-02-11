from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from alphasched.config.env import EnvConfig, ObsConfig
from alphasched.envs.parallel_machine_twt import EnvParams, ParallelMachineTWTEnv
from alphasched.logging import MetricsWriter, create_run_dir
from alphasched.rl.callbacks import EpisodeCsvCallback, EpisodeCsvConfig, WallTimeLimitCallback
from alphasched.rl.models import ResNetExtractor, SimConvExtractor


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MaskablePPO on the scheduling environment (SB3).")
    p.add_argument("--part-num", type=int, default=65)
    p.add_argument("--mach-num", type=int, default=-1, help="-1 means auto")
    p.add_argument("--dist-type", type=str, default="h", choices=["h", "m", "l"])
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--run-hours", type=float, default=0.0, help="If >0, stops by wall time as well.")

    # PPO hyperparams (paper Table 4 defaults)
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--clip-range", type=float, default=0.10)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--vf-coef", type=float, default=0.50)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.50)

    # Model
    p.add_argument("--policy-net", type=str, default="resnet", choices=["resnet", "simconv"])
    p.add_argument("--resblocks", type=int, default=9)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    p.add_argument("--run-name", type=str, default="train-ppo")
    p.add_argument("--runs-dir", type=str, default="runs")
    return p


def _make_env_thunk(env_cfg: EnvConfig, obs_cfg: ObsConfig, *, rank: int, seed: int):
    def _thunk():
        from stable_baselines3.common.monitor import Monitor
        from sb3_contrib.common.wrappers import ActionMasker

        params = EnvParams(
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            mode="train",
            instance_id_start=rank * 1_000_000,
            instance_id_step=1,
            seed_offset=0,
        )
        env = ParallelMachineTWTEnv(params)
        env = ActionMasker(env, lambda e: e.action_mask())
        env = Monitor(env)
        return env

    return _thunk


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    env_cfg = EnvConfig(
        part_num=args.part_num,
        dist_type=args.dist_type,
        mach_num=None if args.mach_num <= 0 else args.mach_num,
        train_seed=int(args.seed),
        val_seed=1000,
        test_seed=0,
    )
    obs_cfg = ObsConfig(include_rule_features=True)
    resolved = env_cfg.resolved()

    run = create_run_dir(base_dir=args.runs_dir, name=args.run_name)
    run.run_dir.mkdir(parents=True, exist_ok=True)
    (run.run_dir / "tb").mkdir(parents=True, exist_ok=True)

    # Save run config
    (run.run_dir / "config.json").write_text(
        json.dumps(
            {
                "env": {
                    "part_num": resolved.part_num,
                    "mach_num": resolved.mach_num,
                    "dist_type": resolved.dist_type,
                    "seed": args.seed,
                },
                "ppo": {
                    "learning_rate": args.learning_rate,
                    "gamma": args.gamma,
                    "clip_range": args.clip_range,
                    "n_steps": args.n_steps,
                    "batch_size": args.batch_size,
                    "n_epochs": args.n_epochs,
                    "gae_lambda": args.gae_lambda,
                    "vf_coef": args.vf_coef,
                    "ent_coef": args.ent_coef,
                    "max_grad_norm": args.max_grad_norm,
                },
                "model": {"policy_net": args.policy_net, "resblocks": args.resblocks},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    from stable_baselines3.common.callbacks import CallbackList

    env_fns = [_make_env_thunk(env_cfg, obs_cfg, rank=i, seed=args.seed) for i in range(int(args.num_envs))]
    vec_env = DummyVecEnv(env_fns) if int(args.num_envs) == 1 else SubprocVecEnv(env_fns)

    if args.policy_net == "simconv":
        extractor_cls = SimConvExtractor
        extractor_kwargs = {"features_dim": 128}
        policy_name = "simconv"
    else:
        extractor_cls = ResNetExtractor
        extractor_kwargs = {"filter_num": 128, "resblock_num": int(args.resblocks)}
        policy_name = f"resnet{args.resblocks}"

    policy_kwargs = {
        "features_extractor_class": extractor_cls,
        "features_extractor_kwargs": extractor_kwargs,
        "net_arch": {"pi": [], "vf": []},
    }

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        vec_env,
        learning_rate=float(args.learning_rate),
        gamma=float(args.gamma),
        clip_range=float(args.clip_range),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        n_epochs=int(args.n_epochs),
        gae_lambda=float(args.gae_lambda),
        vf_coef=float(args.vf_coef),
        ent_coef=float(args.ent_coef),
        max_grad_norm=float(args.max_grad_norm),
        tensorboard_log=str(run.tb_dir),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=int(args.seed),
        device=str(args.device),
    )

    with MetricsWriter(run.metrics_path) as writer:
        cb_list = [
            EpisodeCsvCallback(
                writer,
                EpisodeCsvConfig(run_id=run.run_id, algo="ppo", mode="train", policy_name=policy_name, env_cfg=resolved),
            )
        ]
        if args.run_hours and float(args.run_hours) > 0:
            cb_list.append(WallTimeLimitCallback(max_seconds=float(args.run_hours) * 3600.0))
        callbacks = CallbackList(cb_list)

        t0 = time.time()
        model.learn(total_timesteps=int(args.total_timesteps), callback=callbacks, progress_bar=True)
        t1 = time.time()
        print(f"Training finished in {t1 - t0:.1f}s")

    model_path = run.run_dir / "model.zip"
    model.save(str(model_path))
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":  # pragma: no cover
    main()

