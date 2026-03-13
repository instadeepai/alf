# Training Metrics Logging Design

**Date:** 2026-03-13
**Branch:** feat/log_training_metrics
**Status:** Approved

---

## Problem

`StateLogger` handles experiment-level metrics (environment rewards, episode stats) but there is no structured standard for logging model training metrics (loss, gradients, learning rate). Ad-hoc `logger.info()` calls are used inside model files (e.g. `cnn.py`) instead. Per-epoch training metrics are not captured in any queryable form.

**Acceptance criteria:**
- Model training metrics (loss, spearman, mse per epoch) are logged in a structured, queryable way — not via plain `logger.info()`
- The approach is consistent with how experiment metrics are already logged via `StateLogger`
- TensorBoard and MLflow are supported as logging backends
- No user-facing API change (users still pass a list of loggers)

---

## Scope

This is **PR 1 of 2**:
- PR 1 (this): introduce `EpochMetrics`, `RoundMetrics`, extend logger hierarchy, wire through `Surrogate` → `State`
- PR 2 (follow-on): replace `RoundMetrics.metrics: dict[str, Any]` with typed nested sub-dataclasses (`SurrogateRoundMetrics`, `DatasetRoundMetrics`, etc.)

---

## Data Model

### `EpochMetrics` (new, `alf_core/dataclasses/`)

Typed container for per-epoch training metrics. Explicit fields provide discoverability; `extra` is an escape hatch for future models.

```python
@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float | None = None
    train_spearman: float | None = None
    val_spearman: float | None = None
    train_mse: float | None = None
    val_mse: float | None = None
    extra: dict[str, float] = field(default_factory=dict)
```

### `RoundMetrics` (new, `alf_core/dataclasses/`)

Container for all metrics within a single acquisition round. Nests epoch-level training history alongside the existing round-level metric dict. The `metrics` dict retains the current heterogeneous structure (typed fields deferred to PR 2).

```python
@dataclass
class RoundMetrics:
    round: int
    metrics: dict[str, Any] = field(default_factory=dict)
    training_history: list[EpochMetrics] = field(default_factory=list)
```

### `State` change

`State.round_metrics` type changes from `dict[str, Any]` to `RoundMetrics`. All task code that currently writes `state.round_metrics = {"round": 0, ...}` or `state.round_metrics.update(...)` is updated to work with `RoundMetrics`.

```python
# Before
state.round_metrics: dict[str, Any]

# After
state.round_metrics: RoundMetrics
```

Both `EpochMetrics` and `RoundMetrics` are exported from `alf_core/dataclasses/__init__.py`.

---

## Logger Hierarchy

### `MetricsLogger` (new ABC)

A lightweight base introducing step-based scalar logging. Lives in `alf_core/utils/state_logger.py`.

```python
class MetricsLogger(abc.ABC):
    @abc.abstractmethod
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        namespace: str | None = None,
    ) -> None:
        """Log a flat dict of scalar metrics at a given step."""
        pass
```

### `StateLogger` (extended)

Now extends `MetricsLogger`, inheriting `log_metrics`. The existing `log(state, round_name)` contract is unchanged.

```python
class StateLogger(MetricsLogger):
    @abc.abstractmethod
    def log(self, state: State, round_name: str | None = None) -> None:
        pass
```

### Implementations

| Class | Location | `log()` | `log_metrics()` |
|---|---|---|---|
| `TerminalStateLogger` | `alf_core` | prints round summary (unchanged) | no-op |
| `FileStateLogger` | `alf_core` | writes `metrics.csv` + predictions (unchanged); also writes `training_metrics.csv` from `training_history` | appends row to `training_metrics.csv` |
| `TensorBoardLogger` | `alf_tools` | replays `training_history` via `log_metrics()`; logs `round_metrics.metrics` at `step=round` | calls `writer.add_scalar(f"{namespace}/{key}", value, step)` |
| `MLflowLogger` | `alf_tools` | replays `training_history` via `log_metrics()`; logs `round_metrics.metrics` at `step=round` | calls `mlflow.log_metric(f"{namespace}/{key}", value, step=step)` |

`TensorBoardLogger` and `MLflowLogger` live in `alf_tools` because they introduce optional heavy dependencies (`tensorboard`, `mlflow`). `MetricsLogger` and `StateLogger` remain in `alf_core` as the interface definitions.

**`FileStateLogger` — `training_metrics.csv` schema:**

| column | type | description |
|---|---|---|
| `round` | int | acquisition round number |
| `epoch` | int | epoch within that round |
| `train_loss` | float | |
| `val_loss` | float (nullable) | |
| `train_spearman` | float (nullable) | |
| `val_spearman` | float (nullable) | |
| `train_mse` | float (nullable) | |
| `val_mse` | float (nullable) | |

---

## Wiring: Epoch Metrics → State

The model stays fully decoupled from loggers. Epoch metrics flow upward through the existing call chain.

### `BaseModel` (alf_core)

Gains a new method with a default empty implementation:

```python
def get_epoch_metrics(self) -> list[EpochMetrics]:
    return []
```

### `CNNModel` (alf_tools)

- Adds `self._epoch_metrics: list[EpochMetrics] = []`, reset at the start of each `train()` call
- `_log_epoch_metrics()` is renamed to `_record_epoch_metrics()` and appends an `EpochMetrics` instance instead of calling `logger.info()`
- Overrides `get_epoch_metrics()` to return `self._epoch_metrics`
- Existing `logger.info()` call for epoch progress is removed (metrics are now structured)

### `Surrogate` (alf_core)

`fit()` return type changes from `None` to `list[EpochMetrics]`:

```python
def fit(self, train_data, val_data) -> list[EpochMetrics]:
    self.model.train(train_data, val_data)
    return self.model.get_epoch_metrics()
```

### Tasks (alf_core)

Each task that calls `surrogate.fit()` stores the returned epoch metrics into `state.round_metrics.training_history`:

```python
# SupervisedTask.run()
state.round_metrics = RoundMetrics(round=state.round)
epoch_metrics = state.surrogate.fit(
    train_data=state.dataset.train_dataset,
    val_data=state.dataset.validation_dataset,
)
state.round_metrics.training_history = epoch_metrics
# evaluate() continues to update state.round_metrics.metrics as before

# DesignTask — same pattern in run_initial_train_round() and optimizer.tell() wiring
```

`BaseTask.evaluate()` updates `state.round_metrics.metrics` (the inner dict) rather than `state.round_metrics` directly — a one-line change.

### Full data flow

```
CNNModel._record_epoch_metrics()
  → appends EpochMetrics to self._epoch_metrics each epoch
  → Surrogate.fit() calls get_epoch_metrics(), returns list[EpochMetrics]
  → Task stores in state.round_metrics.training_history
  → state_logger.log(state) called once per round:
      FileStateLogger  → appends rows to training_metrics.csv
      TensorBoardLogger → writer.add_scalar() per epoch metric
      MLflowLogger      → mlflow.log_metric() per epoch metric
```

---

## Testing

### Unit tests — `alf_core`

- `EpochMetrics`: construction, defaults, `extra` field passthrough
- `RoundMetrics`: construction, defaults, `training_history` accumulation
- `MetricsLogger` / `StateLogger`: ABC enforcement — subclasses missing `log_metrics` or `log` raise `TypeError`
- `FileStateLogger.log_metrics()`: appends correct rows to `training_metrics.csv`; creates file on first call; subsequent calls append not overwrite
- `TerminalStateLogger.log_metrics()`: no-op, does not raise
- `State`: existing state tests updated to use `RoundMetrics`

### Unit tests — `alf_tools`

- `CNNModel.get_epoch_metrics()`: returns one `EpochMetrics` per epoch with correct fields; `val_*` fields are `None` when no val data; list resets between `train()` calls
- `Surrogate.fit()`: return value is `list[EpochMetrics]` with correct length
- `TensorBoardLogger.log_metrics()`: calls `writer.add_scalar()` with correct tag (`"{namespace}/{key}"`), value, and step (mock writer)
- `TensorBoardLogger.log()`: replays `training_history` — asserts `log_metrics` called once per epoch
- `MLflowLogger`: same as TensorBoard tests with `mlflow.log_metric` mock

### Integration test — `alf_tools`

- `test_supervised_gfp_cnn_surrogate.py`: assert `training_metrics.csv` exists, has expected columns, and row count equals `num_epochs`

---

## Migration Notes

- `state.round_metrics` type change from `dict` to `RoundMetrics` is a breaking change for any external code accessing `state.round_metrics["key"]` — callers must update to `state.round_metrics.metrics["key"]`
- `Surrogate.fit()` return type changes from `None` to `list[EpochMetrics]` — callers that ignore the return value are unaffected; no breaking change in practice
- `TerminalStateLogger.log_metrics()` is a no-op by design — per-epoch metrics are not printed to terminal to avoid log noise
