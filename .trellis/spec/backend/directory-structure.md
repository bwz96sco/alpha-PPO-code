# Directory Structure

> How the alpha-schedule-sb3 Python package is organized.

---

## Overview

This is a **Python RL research project** for solving the Parallel Machine Scheduling problem with Total Weighted Tardiness (TWT) objective. The active codebase lives in `alpha-schedule-sb3/` using a standard `src` layout with `hatchling` as the build backend.

The project also contains `old_ver_code/` with legacy experiment code (kept for reference only — do not modify).

---

## Directory Layout

```
alpha-schedule-sb3/
├── src/alphasched/           # Main Python package
│   ├── __init__.py
│   ├── config/               # Configuration dataclasses
│   │   └── env.py            # EnvConfig, ResolvedEnvConfig, DistributionParams
│   ├── core/                 # Pure logic (no framework dependencies)
│   │   ├── instance.py       # Instance dataclass (jobs array)
│   │   ├── generator.py      # InstanceGenerator (deterministic seed-based)
│   │   ├── simulator.py      # ParallelMachineSimulator + evaluate_permutation
│   │   └── features.py       # FeatureEncoder (observation tensor + action mask)
│   ├── envs/                 # Gymnasium environment wrappers
│   │   └── parallel_machine_twt.py  # ParallelMachineTWTEnv
│   ├── rl/                   # Stable-Baselines3 integration
│   │   ├── models.py         # CNN extractors (SimConvExtractor, ResNetExtractor)
│   │   ├── callbacks.py      # EpisodeCsvCallback, WallTimeLimitCallback
│   │   └── sb3_policy.py     # SB3MaskablePolicy wrapper for search
│   ├── search/               # Tree search algorithms
│   │   ├── policy.py         # Policy protocol + RandomPolicy
│   │   ├── beam.py           # beam_search
│   │   ├── gpsearch.py       # gpsearch, random_search
│   │   └── rollout.py        # greedy_rollout, random_rollout
│   ├── baselines/            # Classical optimization algorithms
│   │   ├── types.py          # SolveResult dataclass
│   │   ├── ga.py             # Genetic Algorithm (solve_ga)
│   │   ├── bbo.py            # Biogeography-Based Optimization (solve_bbo)
│   │   ├── pso.py            # Particle Swarm Optimization (solve_pso)
│   │   └── rules.py          # Dispatching rules (SPT, WSPT, WMDD, ATC, WCO)
│   ├── logging/              # Metrics and run management
│   │   ├── __init__.py       # Re-exports MetricsWriter, create_run_dir
│   │   ├── metrics.py        # MetricsWriter (CSV logging)
│   │   └── runs.py           # RunContext, create_run_dir
│   ├── cli/                  # CLI entry points (argparse-based)
│   │   ├── train_ppo.py      # alphasched-train-ppo
│   │   ├── eval_ppo.py       # alphasched-eval-ppo
│   │   ├── gpsearch.py       # alphasched-run-gpsearch
│   │   ├── baselines.py      # alphasched-run-baseline
│   │   ├── export_excel.py   # alphasched-export-excel
│   │   └── compare_legacy.py # alphasched-compare-legacy
│   └── compat/               # Legacy CLI argument mapping
├── tests/                    # pytest test suite
│   ├── conftest.py           # sys.path setup
│   └── test_core_invariants.py  # Core module invariant tests
├── pyproject.toml            # Build config, dependencies, CLI entry points
├── .python-version           # Python 3.11
└── README.md
```

---

## Module Organization

### Layer Hierarchy (dependency flows downward)

```
cli/          → Entry points (argparse, orchestration)
  ↓
rl/           → SB3 wrappers (MaskablePPO, callbacks, feature extractors)
search/       → Tree search algorithms (beam, GPSearch, rollout)
baselines/    → Classical optimization (GA, BBO, PSO, rules)
  ↓
envs/         → Gymnasium environment (wraps core/ into gym.Env)
  ↓
core/         → Pure logic (simulator, generator, features) — NO framework deps
  ↓
config/       → Configuration dataclasses
logging/      → Metrics CSV writer, run directory management
```

**Key rule**: `core/` modules must NOT import from `rl/`, `search/`, `baselines/`, `envs/`, or `cli/`. They depend only on `config/` and standard libraries (numpy).

### Where to Put New Code

| Code type | Location | Example |
|-----------|----------|---------|
| New scheduling algorithm | `baselines/` | `solve_sa()` for Simulated Annealing |
| New search strategy | `search/` | A new tree search variant |
| New CNN architecture | `rl/models.py` | A new `BaseFeaturesExtractor` subclass |
| New CLI command | `cli/` + `pyproject.toml` `[project.scripts]` | New entry point |
| New environment variant | `envs/` | Different reward shaping |
| Core logic changes | `core/` | New feature channels, simulator rules |
| Config parameters | `config/env.py` | New fields on `EnvConfig` |

---

## Naming Conventions

| Element | Convention | Examples |
|---------|-----------|----------|
| Modules | `snake_case.py` | `train_ppo.py`, `parallel_machine_twt.py` |
| Classes | `PascalCase` | `ParallelMachineSimulator`, `FeatureEncoder` |
| Functions | `snake_case` | `solve_ga`, `beam_search`, `greedy_rollout` |
| Private helpers | `_snake_case` | `_crossover`, `_schedule_one`, `_conv1x3` |
| Constants | `UPPER_SNAKE` or `_UPPER_SNAKE` | `DEFAULT_FIELDS`, `_EPS` |
| Type aliases | `PascalCase` Literals | `DistType`, `Mode`, `RuleName` |
| CLI entry points | `alphasched-<verb>-<noun>` | `alphasched-train-ppo` |

---

## Examples

- **Well-organized module**: `core/simulator.py` — clean separation of `ParallelMachineSimulator` class and `evaluate_permutation` utility function
- **Config pattern**: `config/env.py` — two-tier `EnvConfig` → `ResolvedEnvConfig` pattern
- **CLI pattern**: `cli/train_ppo.py` — argparse + `main(argv=None)` for testability
