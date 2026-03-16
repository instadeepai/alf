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

`to_metrics_dict()` is the canonical conversion used by all loggers. `None` fields are skipped (not serialized as NaN). `extra` keys are written as dynamic additional columns in `training_metrics.csv` and passed through to TensorBoard/MLflow.

### `RoundMetrics` (new, `alf_core/dataclasses/`)

Container for all metrics within a single acquisition round. Nests epoch-level training history alongside the existing round-level metric dict. The `metrics` dict retains the current heterogeneous structure (typed fields deferred to PR 2).

```python
@dataclass
class RoundMetrics:
    round: int
    metrics: dict[str, Any] = field(default_factory=dict)
    training_history: list[EpochMetrics] = field(default_factory=list)
```

The `round` field on `RoundMetrics` is the canonical location for the round number. It is not duplicated inside `metrics`.

### `State` change

`State.round_metrics` type changes from `dict[str, Any]` to `RoundMetrics`. All task and optimizer code that currently calls `state.round_metrics = {...}` or `state.round_metrics.update(...)` is updated to use `RoundMetrics` construction or `state.round_metrics.metrics[key] = value` respectively.

```python
# Before
state.round_metrics: dict[str, Any]

# After
state.round_metrics: RoundMetrics
```

Both `EpochMetrics` and `RoundMetrics` are exported from `alf_core/dataclasses/__init__.py`. The `state_logger.py` import list must be updated to include `EpochMetrics` and `RoundMetrics` from `alf_core.dataclasses`.

---

## Logger Hierarchy

### `MetricsLogger` (new base class)

A lightweight base introducing step-based scalar logging. Lives in `alf_core/utils/state_logger.py`. `log_metrics()` has a default no-op implementation so that existing `StateLogger` subclasses outside this codebase are not immediately broken. Subclasses that need step-based logging override it.

```python
class MetricsLogger:
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        namespace: str | None = None,
    ) -> None:
        """Log a flat dict of scalar metrics at a given step. No-op by default."""
        pass
```

### `StateLogger` (extended)

Now extends `MetricsLogger`. The existing `log(state, round_name)` abstract contract is unchanged. Existing subclasses that do not override `log_metrics()` inherit the no-op default.

```python
class StateLogger(MetricsLogger, abc.ABC):
    @abc.abstractmethod
    def log(self, state: State, round_name: str | None = None) -> None:
        pass
```

### Implementations

**`TerminalStateLogger`** — `log()` updated to:
- Iterate `state.round_metrics.metrics.items()` instead of `state.round_metrics.items()`
- Fall back to `str(state.round_metrics.round)` when `round_name is None` (not `str(state.round)`, which is already incremented past the current round by the time `log()` is called)

`log_metrics()` inherits the no-op default.

**`FileStateLogger`** — `log()` updated to:
1. Pass `state.round_metrics.metrics` (the inner dict) to the existing private `_log_metrics()` method (for `metrics.csv`)
2. Iterate `state.round_metrics.training_history` and call `self.log_metrics()` per epoch, injecting `{"round": state.round_metrics.round}` alongside `epoch_metrics.to_metrics_dict()`
3. Fall back to `str(state.round_metrics.round)` when `round_name is None` (same fix as `TerminalStateLogger`)

`log_metrics()` is the single write path for `training_metrics.csv`. The private `_log_metrics()` method is kept as-is for `metrics.csv` and is distinct from the public `log_metrics()` ABC method.

`log_metrics()` writes to `training_metrics.csv` using **append mode**: write the header only when the file does not yet exist, then write each row directly without reading the existing file. This avoids the O(n²) read-then-rewrite pattern of `_log_metrics()` (which is acceptable for the low-volume `metrics.csv` but not for per-epoch rows across many rounds).

**`TensorBoardLogger`** (new, `alf_tools`) — overrides both `log()` and `log_metrics()`:
- `log_metrics(metrics, step, namespace)` calls `writer.add_scalar(f"{namespace}/{key}", value, step)` for each key
- `log()` calls `self.log_metrics(e.to_metrics_dict(), step=e.epoch, namespace="training")` per epoch, then calls `self.log_metrics(state.round_metrics.metrics, step=state.round_metrics.round, namespace="round")` for round-level scalars

**`MLflowLogger`** (new, `alf_tools`) — same pattern as `TensorBoardLogger` with `mlflow.log_metric(f"{namespace}/{key}", value, step=step)`.

`TensorBoardLogger` and `MLflowLogger` live in `alf_tools` because they introduce optional heavy dependencies (`tensorboard`, `mlflow`).

**Step axis namespacing convention** — to prevent TensorBoard/MLflow from conflating epoch-level and round-level steps on the same axis, two distinct top-level namespaces are always used:

- `training/<metric>` at `step=epoch` — per-epoch training curves (e.g. `training/train_loss`, `training/val_spearman`)
- `round/<metric>` at `step=round` — per-round evaluation metrics (e.g. `round/surrogate/test_spearman`)

**`FileStateLogger` — `training_metrics.csv` schema:**

`FileStateLogger.log()` calls `self.log_metrics()` per epoch with the dict `{"round": state.round_metrics.round, **epoch_metrics.to_metrics_dict()}` and `step=epoch_metrics.epoch`. `log_metrics()` writes one row per call. The `epoch` value in the row comes from `EpochMetrics.epoch` (included in `to_metrics_dict()` as `"epoch"`); the `step` parameter is not separately written.

When `extra` keys vary across epochs or rounds, `pd.concat` will introduce `NaN` for missing columns in earlier rows. This is acceptable behaviour — sparse columns are expected as different models may report different extra metrics.

| column | type | description |
|---|---|---|
| `round` | int | acquisition round number |
| `epoch` | int | epoch within that round |
| `train_loss` | float | |
| `val_loss` | float (nullable, omitted if None) | |
| `train_spearman` | float (nullable, omitted if None) | |
| `val_spearman` | float (nullable, omitted if None) | |
| `train_mse` | float (nullable, omitted if None) | |
| `val_mse` | float (nullable, omitted if None) | |
| *(extra keys)* | float | any keys from `EpochMetrics.extra` |

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
- Existing `logger.info()` epoch progress call is removed (metrics are now structured)

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

Acquisition rounds in `run()` — `state.round_metrics` must be initialized as `RoundMetrics` before calling `optimizer.ask()`. `state_logger.log(state)` is called without `round_name` in acquisition rounds (falls back to `str(state.round_metrics.round)`), which is intentional:

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

**`state.round` vs `state.round_metrics.round`:** After `state.update()`, `state.round` is incremented to `round_i + 1`, but `state.round_metrics.round` remains `round_i`. `state.round_metrics.round` is the canonical round number used by all loggers for `step` and `round` column values. `state.round` reflects the next round's index and must not be used for logging the current round.

### `Optimizer` (alf_core)

`ask()` currently calls `state.round_metrics.update({"ask_time": ...})`. Updated to:

```python
state.round_metrics.metrics["ask_time"] = t1 - t0
```

`tell()` currently calls `surrogate.fit()`, `state.round_metrics.update({"tell_time": ...})`, and `state.round_metrics.update(self.get_metrics(state))`. Updated to:

```python
epoch_metrics = state.surrogate.fit(train_data, val_data)
state.round_metrics.training_history = epoch_metrics  # full replacement, not append
state.round_metrics.metrics["tell_time"] = t1 - t0
state.round_metrics.metrics.update(self.get_metrics(state))
```

`training_history` is exclusively written by `tell()`. `ask()` must never modify `training_history`.

**`get_training_summary_metrics()` retention:** `BaseModel.get_training_summary_metrics()` and `Surrogate.get_training_summary_metrics()` are retained in PR 1. `Optimizer.get_metrics()` continues to call `get_training_summary_metrics()` and merge final summary scalars (e.g. `surrogate/final_train_loss`) into `round_metrics.metrics`. These summary scalars coexist with the per-epoch `training_history` — they are not redundant because they represent the final-epoch summary for the `metrics.csv` round row. Consolidation or deprecation of `get_training_summary_metrics()` is deferred to PR 2.

### `BaseTask.evaluate()` (alf_core)

Updates `state.round_metrics.metrics` (the inner dict) rather than `state.round_metrics` directly — each `.update(...)` call becomes `.metrics.update(...)`.

### Full data flow

```
CNNModel._record_epoch_metrics()
  → appends EpochMetrics to self._epoch_metrics each epoch
  → Surrogate.fit() calls get_epoch_metrics(), returns list[EpochMetrics]
  → SupervisedTask / DesignTask.run_initial_train_round() / Optimizer.tell()
    stores in state.round_metrics.training_history
  → state_logger.log(state) called once per round:
      FileStateLogger  → _log_metrics(round_metrics.metrics) → metrics.csv
                         iterates training_history, calls log_metrics() per epoch
                         → training_metrics.csv
      TensorBoardLogger → log_metrics() per epoch (namespace="training", step=epoch)
                          log_metrics() per round metric (namespace="round", step=round)
      MLflowLogger      → same as TensorBoard with mlflow.log_metric()
```

---

## Testing

### Unit tests — `alf_core`

- `EpochMetrics.to_metrics_dict()`: None fields are excluded; `extra` keys are included; `epoch` and `train_loss` always present
- `RoundMetrics`: construction, defaults, `training_history` accumulation
- `StateLogger` ABC enforcement: subclasses missing `log()` raise `TypeError`; subclasses missing `log_metrics()` do not raise (it has a default no-op)
- `FileStateLogger.log_metrics()`: appends correct rows to `training_metrics.csv`; creates file on first call; subsequent calls append not overwrite; `extra` keys appear as dynamic columns; `round` key is written correctly
- `FileStateLogger.log()`: passes `state.round_metrics.metrics` to `_log_metrics()`; injects `round` into each row in `training_metrics.csv`
- `TerminalStateLogger.log()`: iterates `state.round_metrics.metrics.items()` without error
- `State`: existing state tests updated to use `RoundMetrics`

### Unit tests — `alf_tools`

- `CNNModel.get_epoch_metrics()`: returns one `EpochMetrics` per epoch with correct fields; `val_*` fields are `None` when no val data; list resets between `train()` calls
- `Surrogate.fit()`: return value is `list[EpochMetrics]` with length equal to `num_epochs`
- `TensorBoardLogger.log_metrics()`: calls `writer.add_scalar()` with correct tag (`"{namespace}/{key}"`), value, and step (mock writer)
- `TensorBoardLogger.log()`: replays `training_history` calling `log_metrics()` once per epoch with `namespace="training"` and `step=epoch`; logs round metrics with `namespace="round"` and `step=round`
- `MLflowLogger`: same as TensorBoard tests with `mlflow.log_metric` mock

### Integration test — `alf_tools`

- `test_supervised_gfp_cnn_surrogate.py`: assert `training_metrics.csv` exists; assert columns match expected schema; assert row count equals `num_epochs` (single training run in supervised task; `num_epochs` is the fixed value from `CNNTrainConfig`)
- `DesignTask` integration test (fixture must have non-empty initial training set): assert row count equals `sum of epochs across all training runs`. For a fixed-epoch model with `num_acq_rounds` rounds and an initial train round, this is `(num_acq_rounds + 1) * num_epochs`. When initial training set is empty, `run_initial_train_round()` is skipped and row count is `num_acq_rounds * num_epochs`. The integration test fixture must document which case it covers.

---

## Migration Notes

- `state.round_metrics` type change from `dict` to `RoundMetrics` is a breaking change. Callers accessing `state.round_metrics["key"]` must update to `state.round_metrics.metrics["key"]`. Callers calling `.update(...)` must update to `.metrics.update(...)`.
- `Surrogate.fit()` return type changes from `None` to `list[EpochMetrics]`. Callers that ignore the return value are unaffected.
- `MetricsLogger.log_metrics()` has a default no-op — existing `StateLogger` subclasses outside this codebase do not need to implement it unless they want step-based logging.
- `TerminalStateLogger.log()` and `FileStateLogger.log()` must update the `round_name is None` fallback from `str(state.round)` to `str(state.round_metrics.round)`. `state.round` is incremented before `log()` is called in `DesignTask` acquisition rounds and would produce the wrong label.
- `TerminalStateLogger.log()` must be updated to iterate `state.round_metrics.metrics.items()`.
- `FileStateLogger.log()` must be updated to pass `state.round_metrics.metrics` to `_log_metrics()`.
- `Optimizer.ask()`, `Optimizer.tell()`, and `BaseTask.evaluate()` all require updates to use `.metrics` on `RoundMetrics`.
- `DesignTask.run()` must initialize `state.round_metrics = RoundMetrics(round=round_i)` at the start of each acquisition loop iteration before calling `optimizer.ask()`.
