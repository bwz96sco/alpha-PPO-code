## alpha-schedule-sb3

Rewrite of the legacy `alpha-PPO-code` research repo onto:
- Gymnasium environments
- Stable-Baselines3 (SB3) + `sb3-contrib` (MaskablePPO for action masking)
- Unified CSV + TensorBoard logging (optional Excel export)

### Quick start (after deps installed)

```bash
cd alpha-schedule-sb3
uv sync

# Train PPO (defaults follow paper Table 4 where applicable)
uv run alphasched-train-ppo --part-num 65 --dist-type h --num-envs 8 --total-timesteps 200000

# Evaluate PPO on a deterministic test set
uv run alphasched-eval-ppo --part-num 65 --dist-type h --test-num 100 --model-path runs/latest/model.zip

# Run guided policy search (GPSearch) using the trained policy
uv run alphasched-run-gpsearch --part-num 65 --dist-type h --test-num 100 --beam-size 10 --model-path runs/latest/model.zip

# Run baselines (rules/GA/BBO/PSO)
uv run alphasched-run-baseline --algo ga --part-num 65 --dist-type h --test-num 100 --popu 200 --iter 400
```

### Outputs
- `runs/<run_id>/metrics.csv`: unified per-episode metrics for *all* algorithms.
- `runs/<run_id>/tb/`: TensorBoard logs (PPO training).
- Optional Excel export:
  - `uv run alphasched-export-excel --metrics runs/<run_id>/metrics.csv --out runs/<run_id>/metrics.xlsx`

### Legacy parity check

Compare the new generator/simulator against the legacy implementation on small deterministic seeds:

```bash
cd alpha-schedule-sb3
uv run alphasched-compare-legacy --part-num 65 --dist-type h --instances 10
```
