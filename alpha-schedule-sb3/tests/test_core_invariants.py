from __future__ import annotations

import numpy as np

from alphasched.config.env import EnvConfig, ObsConfig, reshape_dims
from alphasched.core.features import FeatureEncoder
from alphasched.core.generator import InstanceGenerator
from alphasched.core.simulator import ParallelMachineSimulator


def test_reproducible_instances():
    cfg = EnvConfig(part_num=65, dist_type="h", mach_num=None, train_seed=123, val_seed=1000, test_seed=0).resolved()
    gen = InstanceGenerator(cfg, rng_backend="legacy_mt19937")
    a = gen.generate(mode="test", instance_id=0).jobs
    b = gen.generate(mode="test", instance_id=0).jobs
    assert np.array_equal(a, b)


def test_reward_sum_matches_negative_wt():
    cfg = EnvConfig(part_num=25, dist_type="m", mach_num=None).resolved()
    gen = InstanceGenerator(cfg, rng_backend="legacy_mt19937")
    inst = gen.generate(mode="test", instance_id=3)

    rng = np.random.default_rng(0)
    perm = rng.permutation(cfg.part_num).tolist()

    sim = ParallelMachineSimulator(inst, cfg.mach_num)
    total_reward = 0.0
    for a in perm:
        out = sim.step(int(a))
        assert not out.invalid_action
        total_reward += float(out.reward)
        if out.done:
            assert out.wt_final is not None
            wt = float(out.wt_final)
            assert abs(total_reward + wt) < 1e-6
            break


def test_action_mask_excludes_scheduled_jobs():
    cfg = EnvConfig(part_num=15, dist_type="l", mach_num=None).resolved()
    gen = InstanceGenerator(cfg, rng_backend="legacy_mt19937")
    inst = gen.generate(mode="test", instance_id=0)
    sim = ParallelMachineSimulator(inst, cfg.mach_num)
    mask0 = sim.action_mask()
    assert mask0.sum() == cfg.part_num

    out = sim.step(0)
    assert not out.invalid_action
    mask1 = sim.action_mask()
    assert not bool(mask1[0])


def test_feature_shape_matches_reshape_dims():
    cfg = EnvConfig(part_num=65, dist_type="h", mach_num=None).resolved()
    gen = InstanceGenerator(cfg, rng_backend="legacy_mt19937")
    inst = gen.generate(mode="test", instance_id=0)
    sim = ParallelMachineSimulator(inst, cfg.mach_num)

    width, height = reshape_dims(cfg.part_num)
    enc = FeatureEncoder(cfg, ObsConfig(include_rule_features=True))
    obs = enc.observation(sim)
    assert obs.shape == (10, width, height)
