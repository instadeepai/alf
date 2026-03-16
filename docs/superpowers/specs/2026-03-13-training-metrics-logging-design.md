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

**Out of scope for PR 1:** File-based logging of per-epoch training history. `FileStateLogger` does not write `training_history` to disk. File sizes would grow large across many rounds × epochs and the format is difficult to analyse interactively. This can be added as a future enhancement.

---

## Data Model

### `EpochMetrics` (new, `alf_core/dataclasses/`)

Typed container for per-epoch training metrics. Explicit fields provide discoverability; `extra` is an escape hatch for future models that produce additional metrics. `None` values are excluded when converting to a flat dict.

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

    def to_metrics_dict(self) -> dict[str, float]:
        """Return a flat dict of non-None metric values merged with extra."""
        result = {"epoch": float(self.epoch), "train_loss": self.train_loss}
        for key in ("val_loss", "train_spearman", "val_spearman", "train_mse", "val_mse"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        result.update(self.extra)
        return result
```

`to_metrics_dict()` is the canonical conversion used by TensorBoard/MLflow loggers. `None` fields are skipped. `extra` keys are passed through to TensorBoard/MLflow.

### `RoundMetrics` (new, `alf_core/dataclasses/`)

Container for all metrics within a single acquisition round. The `metrics` dict retains the current heterogeneous structure (typed fields deferred to PR 2). `training_history` carries per-epoch data for backends that support step-based logging.

```python
@dataclass
class RoundMetrics:
    round: int
    metrics: dict[str, Any] = field(default_factory=dict)
    training_history: list[EpochMetrics] = field(default_factory=list)
```

The `round` field is the canonical location for the round number — it is not duplicated inside `metrics`.

### `State` change

`State.round_metrics` type changes from `dict[str, Any]` to `RoundMetrics`. All task and optimizer code that calls `state.round_metrics = {...}` or `state.round_metrics.update(...)` is updated to use `RoundMetrics` construction or `state.round_metrics.metrics[key] = value` respectively.

Both `EpochMetrics` and `RoundMetrics` are exported from `alf_core/dataclasses/__init__.py`. The `state_logger.py` import list is updated to include them.

---

## Logger Hierarchy

### `StateLogger` — no interface change

`StateLogger` keeps the existing `log(state, round_name)` abstract method unchanged. No new base class is introduced. Each implementation accesses `state.round_metrics.training_history` directly inside `log()` if it needs per-epoch data.

### Implementations

**`TerminalStateLogger`** — two minimal updates to `log()`:
- Iterate `state.round_metrics.metrics.items()` instead of `state.round_metrics.items()`
- Fall back to `str(state.round_metrics.round)` when `round_name is None` (not `str(state.round)`, which is already incremented past the current round)

**`FileStateLogger`** — one minimal update to `log()`:
- Pass `state.round_metrics.metrics` (the inner dict) to the existing `_log_metrics()` method instead of `state.round_metrics` directly
- Also update `round_name is None` fallback to `str(state.round_metrics.round)`
- Training history is **not** written to file in PR 1 (see Scope section)

**`TensorBoardLogger`** (new, `alf_tools`) — extends `StateLogger`, implements `log()` using two private helpers:
- `_log_round_metrics(state)` — calls `writer.add_scalar(f"round/{key}", value, step=state.round_metrics.round)` for each key in `state.round_metrics.metrics`
- `_log_training_history_per_round(state)` — iterates `state.round_metrics.training_history` and calls `writer.add_scalar(f"training/{key}", value, step=e.epoch)` for each key in `e.to_metrics_dict()`

**`MLflowLogger`** (new, `alf_tools`) — same structure as `TensorBoardLogger`:
- `_log_round_metrics(state)` — calls `mlflow.log_metrics({f"round/{k}": v ...}, step=state.round_metrics.round)`
- `_log_training_history_per_round(state)` — iterates `training_history` and calls `mlflow.log_metrics({f"training/{k}": v ...}, step=e.epoch)` per epoch

`TensorBoardLogger` and `MLflowLogger` live in `alf_tools` (optional heavy dependencies).

**Step axis namespacing convention:**
- `training/<metric>` at `step=epoch` — per-epoch training curves
- `round/<metric>` at `step=round` — per-round evaluation metrics

These distinct namespaces prevent TensorBoard/MLflow from conflating epoch-level and round-level step axes.

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
- `_log_epoch_metrics()` renamed to `_record_epoch_metrics()` — appends an `EpochMetrics` instance instead of calling `logger.info()`
- Overrides `get_epoch_metrics()` to return `self._epoch_metrics`

### `Surrogate` (alf_core)

`fit()` return type changes from `None` to `list[EpochMetrics]`:

```python
def fit(self, train_data, val_data) -> list[EpochMetrics]:
    self.model.train(train_data, val_data)
    return self.model.get_epoch_metrics()
```

### `SupervisedTask` (alf_core)

```python
def run(self, state, state_loggers):
    t0 = time.perf_counter()
    epoch_metrics = state.surrogate.fit(
        train_data=state.dataset.train_dataset,
        val_data=state.dataset.validation_dataset,
    )
    t1 = time.perf_counter()
    state.round_metrics = RoundMetrics(
        round=state.round,
        metrics={"tell_time": t1 - t0},
        training_history=epoch_metrics,
    )
    state = self.evaluate(state=state)
    for state_logger in state_loggers:
        state_logger.log(state, round_name="supervised evaluation")
```

### `DesignTask` (alf_core)

`run_initial_train_round()`:

```python
def run_initial_train_round(self, state, state_loggers):
    # Construct RoundMetrics before fit() so state is always typed, even on failure
    state.round_metrics = RoundMetrics(round=0)
    epoch_metrics = state.surrogate.fit(
        train_data=state.dataset.train_dataset,
        val_data=state.dataset.validation_dataset,
    )
    state.round_metrics.training_history = epoch_metrics
    state = self.evaluate(state=state)
    for state_logger in state_loggers:
        state_logger.log(state, round_name="initial_train_round")
    return state
```

Acquisition rounds in `run()`:

```python
for round_i in range(1, self.num_acq_rounds + 1):
    state.round_metrics = RoundMetrics(round=round_i)
    acquired_candidates, state = optimizer.ask(state)
    labelled_candidates, state = oracle.evaluate(acquired_candidates, state)
    state.update(labelled_candidates)          # increments state.round to round_i + 1
    state = optimizer.tell(state=state)        # populates state.round_metrics.training_history
    state = self.evaluate(state=state)
    for state_logger in state_loggers:
        state_logger.log(state)               # state.round_metrics.round == round_i (canonical)
```

**`state.round` vs `state.round_metrics.round`:** After `state.update()`, `state.round` becomes `round_i + 1`, but `state.round_metrics.round` remains `round_i`. `state.round_metrics.round` is the canonical round number for all loggers. `state.round` must not be used for logging the current round.

### `Optimizer` (alf_core)

`ask()` — replace `state.round_metrics.update({"ask_time": ...})` with:
```python
state.round_metrics.metrics["ask_time"] = t1 - t0
```

`tell()` — replace all `state.round_metrics.update(...)` calls:
```python
epoch_metrics = state.surrogate.fit(train_data, val_data)
state.round_metrics.training_history = epoch_metrics  # full replacement, not append
state.round_metrics.metrics["tell_time"] = t1 - t0
state.round_metrics.metrics.update(self.get_metrics(state))
```

`training_history` is exclusively written by `tell()`. `ask()` must never modify it.

**`get_training_summary_metrics()` retention:** Retained in PR 1. `Optimizer.get_metrics()` continues to merge final summary scalars into `round_metrics.metrics`. Consolidation deferred to PR 2.

### `BaseTask.evaluate()` (alf_core)

Each `.update(...)` call becomes `.metrics.update(...)`.

### Full data flow

```
CNNModel._record_epoch_metrics()
  → appends EpochMetrics to self._epoch_metrics each epoch
  → Surrogate.fit() returns list[EpochMetrics]
  → SupervisedTask / DesignTask.run_initial_train_round() / Optimizer.tell()
    stores in state.round_metrics.training_history
  → state_logger.log(state) called once per round:
      TerminalStateLogger → prints state.round_metrics.metrics
      FileStateLogger     → _log_metrics(round_metrics.metrics) → metrics.csv
                            (training_history not written to file)
      TensorBoardLogger   → _log_training_history_per_round() → add_scalar per epoch
                            _log_round_metrics() → add_scalar per round metric
      MLflowLogger        → same with mlflow.log_metrics()
```

---

## Testing

### Unit tests — `alf_core`

- `EpochMetrics.to_metrics_dict()`: None fields excluded; `extra` keys included; `epoch` and `train_loss` always present
- `RoundMetrics`: construction, defaults, `training_history` field
- `StateLogger` ABC: subclasses missing `log()` raise `TypeError`
- `TerminalStateLogger.log()`: iterates `state.round_metrics.metrics.items()` without error; fallback uses `state.round_metrics.round`
- `FileStateLogger.log()`: passes `state.round_metrics.metrics` to `_log_metrics()`; fallback uses `state.round_metrics.round`
- `State`: existing state tests updated to use `RoundMetrics`

### Unit tests — `alf_tools`

- `CNNModel.get_epoch_metrics()`: returns one `EpochMetrics` per epoch; `val_*` fields are `None` when no val data; list resets between `train()` calls
- `Surrogate.fit()`: return value is `list[EpochMetrics]` with length equal to `num_epochs`
- `TensorBoardLogger.log()`: calls `_log_training_history_per_round()` and `_log_round_metrics()`; asserts `writer.add_scalar()` called with correct tag and step (mock writer)
- `MLflowLogger.log()`: same with `mlflow.log_metrics` mock

### Integration test — `alf_tools`

- `test_supervised_gfp_cnn_surrogate.py`: assert `training_history` on `state.round_metrics` has length equal to `num_epochs` after task completes
- `DesignTask` integration test: assert total epoch count across all rounds equals `(num_acq_rounds + 1) * num_epochs` when initial training set is non-empty; `num_acq_rounds * num_epochs` when empty

---

## Migration Notes

- `state.round_metrics` type change from `dict` to `RoundMetrics` is breaking. Callers must update `state.round_metrics["key"]` → `state.round_metrics.metrics["key"]` and `.update(...)` → `.metrics.update(...)`.
- `Surrogate.fit()` return type changes from `None` to `list[EpochMetrics]`. Callers ignoring the return value are unaffected.
- `TerminalStateLogger.log()` and `FileStateLogger.log()`: update `round_name is None` fallback from `str(state.round)` to `str(state.round_metrics.round)`.
- `FileStateLogger.log()`: pass `state.round_metrics.metrics` to `_log_metrics()` instead of `state.round_metrics`.
- `Optimizer.ask()`, `Optimizer.tell()`, `BaseTask.evaluate()`: all `.update(...)` calls on `round_metrics` become `.metrics.update(...)`.
- `DesignTask.run()`: initialize `state.round_metrics = RoundMetrics(round=round_i)` before `optimizer.ask()` each iteration.
