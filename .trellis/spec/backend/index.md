# Backend Development Guidelines

> Conventions for the alpha-schedule-sb3 Python RL research codebase.

---

## Overview

This is a **Python reinforcement learning research project** for solving Parallel Machine Scheduling (TWT objective). The codebase uses Stable-Baselines3 with MaskablePPO, Gymnasium environments, and classical optimization baselines.

**Tech stack**: Python 3.11+, PyTorch, Stable-Baselines3, Gymnasium, NumPy, hatchling (build), uv (package manager), pytest (testing)

---

## Guidelines Index

| Guide | Description | Status |
|-------|-------------|--------|
| [Directory Structure](./directory-structure.md) | Package layout, module hierarchy, naming conventions | Filled |
| [Data & Instance Management](./database-guidelines.md) | Instance generation, run directories, seed conventions | Filled |
| [Error Handling](./error-handling.md) | ValueError, asserts, penalty rewards, graceful fallbacks | Filled |
| [Logging Guidelines](./logging-guidelines.md) | MetricsWriter CSV, TensorBoard, print, RunContext | Filled |
| [Quality Guidelines](./quality-guidelines.md) | Code standards, frozen dataclasses, type hints, testing, forbidden patterns | Filled |

---

## Quick Reference

### Key Conventions

- Every `.py` file starts with `from __future__ import annotations`
- Value objects use `@dataclass(frozen=True, slots=True)`
- Functions with 3+ parameters use `*` keyword-only separator
- CLI `main()` accepts `argv: list[str] | None = None`
- `core/` modules must NOT import from framework layers (`rl/`, `envs/`, etc.)

### Common Commands

```bash
cd alpha-schedule-sb3

# Install dependencies
uv sync

# Run tests
uv run pytest

# Train PPO
uv run alphasched-train-ppo --part-num 65 --dist-type h

# Evaluate
uv run alphasched-eval-ppo --model-path runs/latest/model.zip

# Run baselines
uv run alphasched-run-baseline --algo ga --part-num 65
```

---

**Language**: All code and documentation should be written in **English**.
