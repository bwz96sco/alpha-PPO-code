# Quality Guidelines

> Code quality standards for this Python RL research project.

---

## Overview

This project follows a **research-grade quality standard**: correctness and reproducibility first, with clean code structure and type safety. The codebase uses modern Python 3.11+ features and frozen dataclasses extensively.

---

## Code Standards

### Every Module Starts With

```python
from __future__ import annotations
```

This enables PEP 604 union syntax (`int | None`) and deferred annotation evaluation.

### Type Annotations

All public functions and class attributes must have type annotations:

```python
# Good — explicit types, keyword-only separator
def solve_ga(
    instance: Instance,
    cfg: ResolvedEnvConfig,
    *,
    pop_size: int,
    iters: int,
    cross_rate: float = 0.6,
    rng: np.random.Generator | None = None,
) -> SolveResult:
```

Type aliases use `Literal` for constrained string values:

```python
DistType = Literal["h", "m", "l"]
Mode = Literal["train", "val", "test"]
RuleName = Literal["SPT", "MP", "WSPT", "WMDD", "ATC", "WCO"]
```

### Frozen Dataclasses

Use `@dataclass(frozen=True, slots=True)` for all value objects:

```python
@dataclass(frozen=True, slots=True)
class Instance:
    jobs: np.ndarray
    seed: int
```

Only use mutable (non-frozen) classes when the object is genuinely stateful:
- `ParallelMachineSimulator` (mutable state during scheduling)
- `MetricsWriter` (wraps file handle)

### Keyword-Only Arguments

Use `*` separator to force keyword arguments for functions with many parameters:

```python
def solve_ga(instance, cfg, *, pop_size, iters, cross_rate=0.6):
```

This prevents positional argument bugs in complex function calls.

### CLI Entry Points

All CLI `main()` functions accept `argv: list[str] | None = None` for testability:

```python
def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
```

### Protocol-Based Abstractions

Use `Protocol` for duck-typing instead of ABC:

```python
class Policy(Protocol):
    def action_probabilities(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray: ...
```

---

## Required Patterns

| Pattern | Where | Example |
|---------|-------|---------|
| `from __future__ import annotations` | Every `.py` file | Top of file |
| `@dataclass(frozen=True, slots=True)` | Value objects | `Instance`, `StepOutcome`, `SolveResult` |
| `*` keyword-only separator | Functions with 3+ params | `solve_ga(..., *, pop_size, iters)` |
| `main(argv=None)` | CLI entry points | `cli/train_ppo.py` |
| Context manager for I/O | File writers | `with MetricsWriter(...) as w:` |
| Explicit `float()` / `int()` casts | numpy → Python boundary | `float(out.wt_final)`, `int(action)` |

---

## Forbidden Patterns

| Pattern | Why | Do Instead |
|---------|-----|------------|
| Global mutable state | Breaks reproducibility | Pass config/state explicitly |
| `import *` | Pollutes namespace | Import specific names |
| Mutable default arguments | Classic Python bug | Use `None` default + `or` |
| `print()` in `core/` or `baselines/` | Side effects in logic modules | Only `print()` in `cli/` |
| `logging.getLogger()` | Not used in this project | Use `MetricsWriter` or `print()` |
| Raw `open()` without `encoding=` | Platform-dependent encoding | Always specify `encoding="utf-8"` |
| Bare `except:` | Hides bugs | Catch specific exceptions |
| Non-frozen dataclass for value objects | Accidental mutation | Use `frozen=True, slots=True` |

---

## Testing Requirements

### Framework

- **pytest >= 8.0** (dev dependency)
- Tests in `alpha-schedule-sb3/tests/`
- Run with: `cd alpha-schedule-sb3 && uv run pytest`

### Test Naming

- Files: `test_<module>_<aspect>.py` (e.g., `test_core_invariants.py`)
- Functions: `test_<what_is_being_tested>` (e.g., `test_reproducible_instances`)

### What to Test

| Must test | Example |
|-----------|---------|
| Core invariants | Reward sum = -WT, action mask correctness |
| Reproducibility | Same seed → same instance |
| Shape consistency | Observation tensor matches expected dims |
| Edge cases | Zero jobs, single machine |

### Test Pattern

Tests follow a consistent pattern: create config → generate instance → run computation → assert invariant:

```python
def test_reward_sum_matches_negative_wt():
    cfg = EnvConfig(part_num=25, dist_type="m").resolved()
    gen = InstanceGenerator(cfg, rng_backend="legacy_mt19937")
    inst = gen.generate(mode="test", instance_id=3)
    sim = ParallelMachineSimulator(inst, cfg.mach_num)
    # ... run simulation ...
    assert abs(total_reward + wt) < 1e-6
```

---

## Dependency Management

- **Build tool**: `uv` (preferred) or `pip`
- **Build backend**: `hatchling`
- **Python**: >= 3.11 (pinned in `.python-version`)
- **Add deps**: Edit `pyproject.toml` `[project.dependencies]`
- **Dev deps**: Edit `pyproject.toml` `[dependency-groups.dev]`
- **Install**: `cd alpha-schedule-sb3 && uv sync`

---

## Code Review Checklist

- [ ] `from __future__ import annotations` at top
- [ ] Type annotations on all public APIs
- [ ] Value objects use frozen dataclasses
- [ ] No mutable default arguments
- [ ] No `print()` outside `cli/`
- [ ] Explicit numpy → Python casts at boundaries
- [ ] Tests for new core logic
- [ ] No new dependencies without justification
