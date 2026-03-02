# Error Handling

> How errors are handled in this project.

---

## Overview

This project uses a **lightweight error handling strategy** typical of research codebases:

- **`ValueError`** for invalid arguments at public API boundaries
- **`assert`** for internal invariants that should never fail
- **Graceful degradation** (penalty values) in simulation/optimization loops
- **No custom exception hierarchy** — standard Python exceptions suffice

---

## Error Types

### 1. ValueError — Invalid Arguments

Used at public function/constructor boundaries for parameter validation.

```python
# config/env.py:37-38
if part_num <= 0:
    raise ValueError(f"part_num must be positive, got {part_num}")

# config/env.py:64-65
if self.dist_type not in DISTRIBUTIONS:
    raise ValueError(f"unknown dist_type: {self.dist_type!r}")

# core/simulator.py:31-32
if mach_num <= 0:
    raise ValueError(f"mach_num must be positive, got {mach_num}")
```

**Pattern**: Always include the invalid value in the error message using f-strings.

### 2. assert — Internal Invariants

Used for conditions that indicate a bug if violated. These should **never** be hit in normal operation.

```python
# core/simulator.py:108-109
assert out.wt_final is not None

# Search loops — after done=True, wt_final must exist
if out.done:
    assert out.wt_final is not None
    return float(out.wt_final)
```

### 3. FileNotFoundError — Missing Resources

Used in eval/CLI code when required model files are missing.

### 4. RuntimeError — Impossible States

Used when code reaches a state that should be unreachable (e.g., no available actions in rule selection).

---

## Error Handling Patterns

### Pattern 1: Penalty Reward for Invalid Actions (Simulation)

Instead of raising on invalid actions, the simulator returns a penalty reward. This keeps the RL training loop stable.

```python
# core/simulator.py:80-83
def step(self, action: int, *, invalid_penalty: float = -1e6) -> StepOutcome:
    action = int(action)
    if action < 0 or action >= self.part_num or self.part[action, 0] == 0:
        return StepOutcome(reward=float(invalid_penalty), done=False, wt_final=None, invalid_action=True)
```

The caller can check `out.invalid_action` to detect this case.

### Pattern 2: Infinity for Invalid Permutations (Optimization)

Metaheuristic solvers need a total function (always returns a value). Invalid permutations return `float("inf")` so they naturally lose in comparisons.

```python
# core/simulator.py:104-106
if out.invalid_action:
    return float("inf")
```

### Pattern 3: Graceful OS Fallback (Symlinks)

For non-critical operations that may fail on some platforms:

```python
# logging/runs.py:34-41
try:
    latest = base / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_dir.name)
except OSError:
    (base / "latest.txt").write_text(run_dir.name, encoding="utf-8")
```

---

## Anti-Patterns (Forbidden)

| Anti-pattern | Why | Do instead |
|---|---|---|
| Bare `except:` | Hides bugs | Catch specific exceptions |
| Raising inside simulation loop | Breaks RL training | Return penalty/inf |
| Silently swallowing errors | Hides data corruption | Log or return sentinel values |
| Custom exception classes (for this project) | Over-engineering for a research codebase | Use `ValueError`, `RuntimeError` |

---

## Common Mistakes

1. **Forgetting to check `invalid_action`** in the step result when iterating over permutations
2. **Using `assert` for input validation** — asserts can be disabled with `-O`; use `ValueError` for external inputs
3. **Not including the bad value in error messages** — always show what was received
