from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from alphasched.config.env import EnvConfig, ObsConfig
from alphasched.envs.parallel_machine_twt import EnvParams, ParallelMachineTWTEnv
from alphasched.logging import MetricsWriter, create_run_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained MaskablePPO policy on the deterministic test set.")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--part-num", type=int, default=65)
    p.add_argument("--mach-num", type=int, default=-1, help="-1 means auto")
    p.add_argument("--dist-type", type=str, default="h", choices=["h", "m", "l"])
    p.add_argument("--test-num", type=int, default=100)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--run-name", type=str, default="eval-ppo")
    return p


def _make_eval_env_thunk(env_cfg: EnvConfig, obs_cfg: ObsConfig, *, rank: int, start_id: int):
    def _thunk():
        from stable_baselines3.common.monitor import Monitor
        from sb3_contrib.common.wrappers import ActionMasker

        params = EnvParams(
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            mode="test",
            instance_id_start=start_id,
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
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if int(args.test_num) % int(args.num_envs) != 0:
        raise ValueError("--test-num must be divisible by --num-envs for deterministic coverage.")
    per_env = int(args.test_num) // int(args.num_envs)

    env_cfg = EnvConfig(
        part_num=args.part_num,
        dist_type=args.dist_type,
        mach_num=None if args.mach_num <= 0 else args.mach_num,
        train_seed=0,
        val_seed=1000,
        test_seed=0,
    )
    obs_cfg = ObsConfig(include_rule_features=True)
    resolved = env_cfg.resolved()

    run = create_run_dir(base_dir=args.runs_dir, name=args.run_name)
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    env_fns = [
        _make_eval_env_thunk(env_cfg, obs_cfg, rank=i, start_id=i * per_env) for i in range(int(args.num_envs))
    ]
    vec_env = DummyVecEnv(env_fns) if int(args.num_envs) == 1 else SubprocVecEnv(env_fns)

    model = MaskablePPO.load(str(model_path), device=str(args.device))

    obs = vec_env.reset()
    episodes = 0
    wt_list: list[float] = []
    t0 = time.time()

    with MetricsWriter(run.metrics_path) as writer:
        while episodes < int(args.test_num):
            masks = get_action_masks(vec_env)
            actions, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(actions)

            for info, done in zip(infos, dones):
                if not done:
                    continue
                wt = info.get("wt_final")
                if wt is None:
                    continue
                wt = float(wt)
                wt_list.append(wt)
                episodes += 1
                writer.write(
                    {
                        "run_id": run.run_id,
                        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "algo": "ppo",
                        "mode": "test",
                        "part_num": resolved.part_num,
                        "mach_num": resolved.mach_num,
                        "dist_type": resolved.dist_type,
                        "seed": info.get("seed"),
                        "instance_id": info.get("instance_id"),
                        "wt": wt,
                        "episode_reward": float(info.get("episode_reward", 0.0)),
                        "steps": None,
                        "wall_time_sec": time.time() - t0,
                        "policy_name": "sb3",
                    }
                )
                if episodes >= int(args.test_num):
                    break

    vec_env.close()
    mean_wt = float(np.mean(wt_list)) if wt_list else float("nan")
    print(f"Evaluated {len(wt_list)} instances. mean WT = {mean_wt:.5f}")
    print(f"Metrics written to: {run.metrics_path}")


if __name__ == "__main__":  # pragma: no cover
    main()

