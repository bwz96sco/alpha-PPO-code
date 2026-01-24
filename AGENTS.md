# Repository Guidelines

## Overview
This repository contains research code for job shop scheduling optimization, comparing multiple approaches (PPO, MCTS/GPSearch, GA, BBO, PSO, beam search, and random rollout). Most methods are implemented as self-contained, versioned modules (for example, `...V0.040`) with their own entrypoints and scripts.

## Project Structure & Module Organization
- `sci-1/sci1-ppoV0.040/`: PPO training (`main.py`) and evaluation (`enjoy.py`).
- `data-mcts/mctsAlphaV0.080/`: MCTS/GPSearch evaluation (`testPolicy.py`).
- `BeamAndRandom/beamSearchV0.010/`, `BeamAndRandom/randomRolloutV0.010/`: search baselines (`testPolicy.py`).
- `new-GA-V0.010/`, `bbo/bboParallelV0.020/`, `pso/`, `rule/`: evolutionary/rule baselines and variants.
- Many modules include `venvs/` (shared environment/scheduler code) plus `run.sh` / `test-run.sh` for batch runs.

## Build, Test, and Development Commands
- Python version: `.python-version` is `3.10`. Some module `requirements.txt` files pin older packages (for example `torch==1.4.0`), so you may need a compatible Python/PyTorch combo for those modules.
- Set up an environment (recommended per module):
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r <module>/requirements.txt` (example: `pip install -r data-mcts/mctsAlphaV0.080/requirements.txt`)
  - Optional runner: `uv run ...` (see `CLAUDE.md` for ready-to-use commands).
- Run PPO training (single run): `cd sci-1/sci1-ppoV0.040 && python main.py --env-name sci1-my --part-num 65 --dist-type h`
- Run evaluations: `python enjoy.py ...` (PPO) or `python testPolicy.py ...` (most other modules); for sweeps, use the module’s `run.sh`.

## Coding Style & Naming Conventions
- Python: 4-space indentation; keep changes localized to the module you’re working in.
- Preserve existing conventions: entrypoints like `main.py` / `enjoy.py` / `testPolicy.py`, and versioned folder names (`*V0.0xx`).
- Keep CLI flags consistent across modules (common ones: `--part-num`, `--dist-type`, `--test-num`, `--beam-size`).

## Testing Guidelines
- There is no centralized unit-test suite; treat `testPolicy.py`/`enjoy.py` runs as smoke/evaluation tests.
- Don’t commit generated artifacts (models, Excel outputs, logs); `.gitignore` excludes common outputs like `results/`, `test_results/`, and `*.xlsx`.

## Commit & Pull Request Guidelines
- Existing commit messages are short and imperative (for example, `setup env`, `minor`); keep commits concise and scoped to one module/change.
- PRs should include: affected module path(s), the exact command(s) to reproduce, and any relevant metrics/log excerpts (avoid attaching large binary artifacts).
