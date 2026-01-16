# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for **job shop scheduling optimization** comparing multiple AI-based approaches:
- **PPO** (Proximal Policy Optimization) - Deep RL with CNN-based policy networks
- **MCTS/GPSearch** - Monte Carlo Tree Search with policy guidance
- **Genetic Algorithm (GA)**, **BBO** (Biogeography-Based Optimization), **PSO** (Particle Swarm Optimization)
- **Beam Search** and **Random Rollout** baselines

## Build and Run Commands

Use `uv` for Python package management (Python 3.10):

```bash
# Install dependencies
uv sync
# Or: pip install -r requirements.txt
```

### Training PPO

```bash
cd sci-1/sci1-ppoV0.040
uv run python main.py \
    --num-processes 16 --num-steps 512 --num-mini-batch 32 \
    --env-name sci1-my --part-num 65 --dist-type h \
    --run-hours 24.0 --excel-save
```

### Testing/Evaluation

```bash
# PPO evaluation
cd sci-1/sci1-ppoV0.040
uv run python enjoy.py \
    --env-name sci1-my --part-num 65 --dist-type h \
    --test-num 100 --num-processes 10 \
    --load-dir ./trained_models/ppo --not-eval-load

# MCTS/GPSearch
cd data-mcts/mctsAlphaV0.080
uv run python testPolicy.py --mode mcts_policy --part-num 65 --dist-type h --beam-size 10 --test-num 100

# Beam Search
cd BeamAndRandom/beamSearchV0.010
uv run python testPolicy.py --part-num 65 --dist-type h --test-num 100

# Random Search
cd BeamAndRandom/randomRolloutV0.010
uv run python testPolicy.py --mode mcts_random --part-num 65 --dist-type h --beam-size 10 --test-num 100

# Genetic Algorithm
cd new-GA-V0.010
uv run python gaMain.py --part-num 65 --dist-type h --iter 400 --popu 200 --test-num 100

# BBO
cd bbo/bboParallelV0.020
uv run python bboMain.py --popu 150 --iter 400 --part-num 65 --dist-type h --test-num 100
```

## Architecture

### Directory Structure

Each algorithm variant is self-contained in its own directory with a `venvs/` subfolder containing shared environment code:

- `sci-1/sci1-ppoV0.040/` - Main PPO implementation
- `data-mcts/mctsAlphaV0.080/` - MCTS with policy guidance (GPSearch)
- `BeamAndRandom/` - Beam search and random rollout baselines
- `new-GA-V0.010/` - Genetic Algorithm
- `bbo/bboParallelV0.020/` - Biogeography-Based Optimization
- `pso/` - Particle Swarm Optimization variants

### Core Components

**PPO** (`sci-1/sci1-ppoV0.040/`):
- `main.py` - Training loop
- `enjoy.py` - Evaluation
- `a2c_ppo_acktr/algo/ppo.py` - PPO algorithm
- `a2c_ppo_acktr/model.py` - CNN policy network (SimConvNet/ResNet)

**Environment** (in each module's `venvs/`):
- `EnvirConf.py` - Configuration (part numbers, machine numbers, distributions)
- `scheduler.py` - Scheduling logic and job generation
- `scheEnv.py` - Gym-compatible environment
- `game.py` - Game interface for MCTS
- `excel.py` - Excel logging utilities
- `gantt.py` - Gantt chart visualization

### Common Arguments

| Argument | Description | Values |
|----------|-------------|--------|
| `--part-num` | Number of jobs | 15, 25, 35, 45, 65, 95, 125 |
| `--dist-type` | Distribution type | 'h' (high), 'm' (medium), 'l' (low) |
| `--beam-size` | K parameter for search | default: 10 |
| `--test-num` | Number of test instances | default: 100 |
| `--mach-num` | Number of machines | auto-calculated if -1 |

### Search Methods Comparison

| Method | Expansion | Evaluation | Network Required |
|--------|-----------|------------|------------------|
| GPSearch | Top-K by network probability | Network rollout | Yes |
| Random Search | Random K selection | Random rollout | No |
| Beam Search | All feasible actions | Trajectory probability | Yes |
| Network Rollout | Direct execution | N/A | Yes |

## Key Dependencies

- PyTorch 1.4.0+ (deep learning)
- Gym 0.17.0+ (RL environment)
- NumPy, Pandas (data processing)
- xlrd, xlwt, xlutils, openpyxl (Excel logging)
- Matplotlib (Gantt chart visualization)
