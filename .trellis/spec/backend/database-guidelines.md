# Data & Instance Management

> How scheduling instances and experiment data are generated, stored, and managed.

---

## Overview

This project has **no database**. Instead, data management revolves around:

1. **Instance generation** — Deterministic, seed-based job generation (no stored datasets)
2. **Run directories** — Each experiment creates a timestamped directory under `runs/`
3. **CSV metrics** — Structured results logging via `MetricsWriter`
4. **Model checkpoints** — SB3 `.zip` model files saved in run directories

---

## Instance Generation

Instances are generated on-the-fly from seeds, not loaded from files. This ensures exact reproducibility.

```python
# core/generator.py
gen = InstanceGenerator(cfg, rng_backend="legacy_mt19937")
inst = gen.generate(mode="test", instance_id=0)
# inst.jobs is an (N, 3) numpy array: [processing_time, due_date, weight]
# inst.seed is the deterministic seed used
```

### Seed Convention

| Mode | Base seed | Usage |
|------|-----------|-------|
| `train` | `train_seed` (default 12345) | RL training episodes |
| `val` | `val_seed` (default 1000) | Validation during training |
| `test` | `test_seed` (default 0) | Final evaluation (paper results) |

**Key rule**: The `legacy_mt19937` RNG backend must be used for paper-comparable results. It matches the legacy codebase's numpy RandomState behavior.

---

## Run Directory Structure

```
runs/
├── latest -> 20260302-143025-train-ppo-a1b2c3d4/  (symlink)
└── 20260302-143025-train-ppo-a1b2c3d4/
    ├── config.json      # Experiment parameters
    ├── metrics.csv      # Per-instance results
    ├── model.zip        # SB3 model checkpoint (training only)
    └── tb/              # TensorBoard logs (training only)
```

### Naming Convention

Run directories follow: `{YYYYMMDD}-{HHMMSS}-{name}-{uuid8}`

---

## Experiment Configuration Storage

Config is saved as JSON in each run directory:

```json
{
  "env": {
    "part_num": 65,
    "mach_num": 30,
    "dist_type": "h",
    "seed": 0
  },
  "ppo": {
    "learning_rate": 0.00025,
    "gamma": 0.99,
    "clip_range": 0.10
  },
  "model": {
    "policy_net": "resnet",
    "resblocks": 9
  }
}
```

---

## Anti-Patterns

| Anti-pattern | Why | Do instead |
|---|---|---|
| Storing instances as files | Wastes disk, not needed | Generate from seed |
| Hardcoding seeds in code | Breaks experiment tracking | Pass via CLI/config |
| Modifying `legacy_mt19937` behavior | Breaks paper reproducibility | Add new RNG backend if needed |
| Overwriting existing run directories | Loses results | Each run gets unique timestamped dir |
