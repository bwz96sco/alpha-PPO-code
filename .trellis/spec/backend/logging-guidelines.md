# Logging Guidelines

> How logging and metrics recording are done in this project.

---

## Overview

This project does **not** use Python's `logging` module. Instead, it uses:

1. **`MetricsWriter`** — Structured CSV logging for experiment results
2. **TensorBoard** — Training curves via SB3's built-in integration
3. **`print()`** — CLI user feedback for progress/summary messages
4. **`RunContext`** — Run directory management with timestamps

---

## MetricsWriter (Primary Metrics Logging)

All experiment results are written as CSV rows with a fixed schema.

### Schema

```python
# logging/metrics.py:8-27
DEFAULT_FIELDS = (
    "run_id", "timestamp_utc", "algo", "mode",
    "part_num", "mach_num", "dist_type", "seed", "instance_id",
    "wt", "episode_reward", "steps", "wall_time_sec",
    "policy_name", "k", "beam_size", "pop_size", "iters",
)
```

### Usage Pattern (Context Manager)

Always use `MetricsWriter` as a context manager to ensure the file is flushed and closed:

```python
# cli/train_ppo.py:161-162
with MetricsWriter(run.metrics_path) as writer:
    # ... training loop ...
    writer.write({
        "run_id": run.run_id,
        "algo": "ppo",
        "mode": "train",
        "wt": wt_value,
        # ... other fields ...
    })
```

**Key behavior**: Flushes after every row for crash safety (`metrics.py:44-45`).

### Adding New Metrics Fields

If you need a new column:
1. Add it to `DEFAULT_FIELDS` tuple in `logging/metrics.py`
2. Pass it in the `row` dict when calling `writer.write()`
3. Missing fields are silently ignored (`extrasaction="ignore"`)

---

## RunContext (Run Directory Management)

Each experiment run gets a timestamped directory under `runs/`.

```python
# logging/runs.py
run = create_run_dir(base_dir="runs", name="train-ppo")
# Creates: runs/20260302-143025-train-ppo-a1b2c3d4/
#   ├── metrics.csv     (via run.metrics_path)
#   ├── config.json     (saved manually by CLI)
#   └── tb/             (via run.tb_dir, for TensorBoard)

# Also creates symlink: runs/latest -> runs/20260302-...
```

**Convention**: Always save `config.json` in the run directory with the experiment parameters (see `cli/train_ppo.py:90-116`).

---

## TensorBoard (Training Curves)

SB3's MaskablePPO writes TensorBoard logs automatically:

```python
# cli/train_ppo.py:154
model = MaskablePPO(..., tensorboard_log=str(run.tb_dir), ...)
```

View with: `tensorboard --logdir runs/latest/tb`

---

## print() (CLI Feedback)

Used sparingly for user-facing progress in CLI commands:

```python
# cli/train_ppo.py:175-178
print(f"Training finished in {t1 - t0:.1f}s")
print(f"Saved model to: {model_path}")
```

**Guidelines**:
- Use `print()` only in `cli/` modules
- Include timing info for long operations
- Show file paths for saved outputs

---

## What to Log

| Event | Where | Format |
|-------|-------|--------|
| Per-episode WT result | `MetricsWriter` CSV | Row with all fields |
| Training curves (loss, reward) | TensorBoard | Automatic via SB3 |
| Run configuration | `config.json` in run dir | JSON dict |
| CLI progress/summary | `print()` | Human-readable |

---

## What NOT to Log

- Do not use Python's `logging` module — it adds complexity with no benefit for this research workflow
- Do not log raw numpy arrays (too verbose) — log summary statistics instead
- Do not log inside `core/` modules — keep them pure computation
