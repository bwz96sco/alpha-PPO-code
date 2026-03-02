# Frontend Development Guidelines

> **Status: Not Applicable**

---

## Overview

This project is a **pure Python RL research codebase** with no frontend component. There is no web UI, no JavaScript/TypeScript, and no browser-based interface.

All interaction happens through **CLI commands** defined in `pyproject.toml`:

| Command | Purpose |
|---------|---------|
| `alphasched-train-ppo` | Train MaskablePPO agent |
| `alphasched-eval-ppo` | Evaluate trained PPO on test set |
| `alphasched-run-gpsearch` | Run GPSearch / beam / rollout search |
| `alphasched-run-baseline` | Run GA / BBO / PSO / dispatching rules |
| `alphasched-export-excel` | Convert CSV metrics to Excel |
| `alphasched-compare-legacy` | Verify new implementation matches legacy |

Visualization is done via **TensorBoard** (training curves) and **matplotlib** (plots generated in scripts).

---

If a frontend is added in the future (e.g., a dashboard for experiment results), fill in the guideline files in this directory.
