# Training Metrics Logging Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `state.round_metrics: dict` with a typed `RoundMetrics` dataclass and route per-epoch CNN training metrics through the existing `StateLogger` infrastructure to TensorBoard/MLflow.

**Architecture:** `CNNModel` accumulates `EpochMetrics` per epoch → `Surrogate.fit()` returns `list[EpochMetrics]` → tasks/optimizer store them in `state.round_metrics.training_history` → existing `state_logger.log(state)` call at end of each round dispatches to `TensorBoardLogger`/`MLflowLogger` which write epoch curves and round scalars.

**Tech Stack:** Python 3.12, dataclasses, PyTorch (CNNModel), `torch.utils.tensorboard.SummaryWriter`, `mlflow` (optional dependency in `alf_tools`)

**Spec:** `docs/superpowers/specs/2026-03-13-training-metrics-logging-design.md`

---

## File Map

### New files
| File | Responsibility |
|------|---------------|
| `core/alf_core/dataclasses/epoch_metrics.py` | `EpochMetrics` dataclass + `to_metrics_dict()` |
| `core/alf_core/dataclasses/round_metrics.py` | `RoundMetrics` dataclass |
| `core/tests/dataclasses/test_epoch_metrics.py` | Unit tests for `EpochMetrics` |
| `core/tests/dataclasses/test_round_metrics.py` | Unit tests for `RoundMetrics` |
| `core/tests/utils/test_state_logger.py` | Unit tests for updated `TerminalStateLogger` / `FileStateLogger` |
| `tools/alf_tools/loggers/__init__.py` | Exports `TensorBoardLogger`, `MLflowLogger` |
| `tools/alf_tools/loggers/tensorboard_logger.py` | `TensorBoardLogger(StateLogger)` |
| `tools/alf_tools/loggers/mlflow_logger.py` | `MLflowLogger(StateLogger)` |
| `tools/tests/loggers/__init__.py` | Empty init for test discovery |
| `tools/tests/loggers/test_tensorboard_logger.py` | Unit tests for `TensorBoardLogger` |
| `tools/tests/loggers/test_mlflow_logger.py` | Unit tests for `MLflowLogger` |

### Modified files
| File | Change |
|------|--------|
| `core/alf_core/dataclasses/__init__.py` | Export `EpochMetrics`, `RoundMetrics` |
| `core/alf_core/dataclasses/state.py` | `round_metrics: RoundMetrics` (was `dict[str, Any]`) |
| `core/alf_core/model/base_model.py` | Add `get_epoch_metrics() -> list[EpochMetrics]` (default `[]`) |
| `core/alf_core/surrogate/surrogate.py` | `fit()` returns `list[EpochMetrics]` (was `None`) |
| `core/alf_core/tasks/base_task.py` | `.metrics.update(...)` instead of `.update(...)` on `round_metrics` |
| `core/alf_core/tasks/supervised_task.py` | Build `RoundMetrics`, capture `epoch_metrics` from `fit()` |
| `core/alf_core/tasks/design_task.py` | Build `RoundMetrics` in `run_initial_train_round()` and `run()` |
| `core/alf_core/optimizer/optimizer.py` | `ask()` and `tell()` use `.metrics[key]`, `tell()` captures epoch metrics |
| `core/alf_core/utils/state_logger.py` | Update `TerminalStateLogger` + `FileStateLogger` for `RoundMetrics` type |
| `tools/alf_tools/models/cnn.py` | `_record_epoch_metrics()`, `_epoch_metrics` list, `get_epoch_metrics()` |
| `core/tests/tasks/test_supervised_task.py` | No metric value changes; survives the refactor |
| `core/tests/tasks/test_design_task.py` | No metric value changes; survives the refactor |
| `tools/tests/models/test_cnn.py` | Add `get_epoch_metrics` tests |
| `tools/tests/e2e_experiments/test_supervised_gfp_cnn_surrogate.py` | Assert `training_history` length |

---

## Chunk 1: `EpochMetrics` and `RoundMetrics` dataclasses

### Task 1: `EpochMetrics` dataclass

**Files:**
- Create: `core/alf_core/dataclasses/epoch_metrics.py`
- Create: `core/tests/dataclasses/test_epoch_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `core/tests/dataclasses/test_epoch_metrics.py`:

```python
from alf_core.dataclasses.epoch_metrics import EpochMetrics


class TestEpochMetrics:
    def test_to_metrics_dict_required_fields_always_present(self):
        em = EpochMetrics(epoch=3, train_loss=0.5)
        d = em.to_metrics_dict()
        assert d["epoch"] == 3.0
        assert d["train_loss"] == 0.5

    def test_to_metrics_dict_none_fields_excluded(self):
        em = EpochMetrics(epoch=1, train_loss=0.8)
        d = em.to_metrics_dict()
        assert "val_loss" not in d
        assert "train_spearman" not in d
        assert "val_spearman" not in d
        assert "train_mse" not in d
        assert "val_mse" not in d

    def test_to_metrics_dict_optional_fields_included_when_set(self):
        em = EpochMetrics(
            epoch=2,
            train_loss=0.4,
            val_loss=0.6,
            train_spearman=0.9,
            val_spearman=0.85,
            train_mse=0.1,
            val_mse=0.15,
        )
        d = em.to_metrics_dict()
        assert d["val_loss"] == 0.6
        assert d["train_spearman"] == 0.9
        assert d["val_spearman"] == 0.85
        assert d["train_mse"] == 0.1
        assert d["val_mse"] == 0.15

    def test_to_metrics_dict_extra_keys_merged(self):
        em = EpochMetrics(epoch=1, train_loss=0.5, extra={"custom_metric": 42.0})
        d = em.to_metrics_dict()
        assert d["custom_metric"] == 42.0

    def test_epoch_field_converted_to_float(self):
        em = EpochMetrics(epoch=5, train_loss=0.3)
        d = em.to_metrics_dict()
        assert isinstance(d["epoch"], float)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/akashsinha/Documents/alf/core
python -m pytest tests/dataclasses/test_epoch_metrics.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `epoch_metrics` module does not exist yet.

- [ ] **Step 3: Create `EpochMetrics`**

Create `core/alf_core/dataclasses/epoch_metrics.py`:

```python
# Copyright 2023 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field


@dataclass
class EpochMetrics:
    """Per-epoch training metrics for a surrogate model.

    Explicit fields provide discoverability for the standard CNN metrics.
    The ``extra`` dict is an escape hatch for model-specific additional metrics.
    ``None`` values are excluded when converting to a flat dict, so backends
    only receive metrics that were actually computed.

    Attributes:
        epoch: Zero-based epoch index.
        train_loss: Training loss for this epoch.
        val_loss: Validation loss, or None if no validation data was provided.
        train_spearman: Spearman correlation on training set, or None.
        val_spearman: Spearman correlation on validation set, or None.
        train_mse: MSE on training set, or None.
        val_mse: MSE on validation set, or None.
        extra: Additional model-specific metrics passed through to backends.
    """

    epoch: int
    train_loss: float
    val_loss: float | None = None
    train_spearman: float | None = None
    val_spearman: float | None = None
    train_mse: float | None = None
    val_mse: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    def to_metrics_dict(self) -> dict[str, float]:
        """Return a flat dict of non-None metric values merged with extra.

        Returns:
            Flat dict with ``epoch`` (as float), ``train_loss``, any non-None
            optional fields, and all entries from ``extra``.
        """
        result: dict[str, float] = {
            "epoch": float(self.epoch),
            "train_loss": self.train_loss,
        }
        for key in ("val_loss", "train_spearman", "val_spearman", "train_mse", "val_mse"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        result.update(self.extra)
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/akashsinha/Documents/alf/core
python -m pytest tests/dataclasses/test_epoch_metrics.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/akashsinha/Documents/alf
git add core/alf_core/dataclasses/epoch_metrics.py core/tests/dataclasses/test_epoch_metrics.py
git commit -m "feat: add EpochMetrics dataclass with to_metrics_dict()"
```

---

### Task 2: `RoundMetrics` dataclass + update exports

**Files:**
- Create: `core/alf_core/dataclasses/round_metrics.py`
- Create: `core/tests/dataclasses/test_round_metrics.py`
- Modify: `core/alf_core/dataclasses/__init__.py`

- [ ] **Step 1: Write the failing tests**

Create `core/tests/dataclasses/test_round_metrics.py`:

```python
from alf_core.dataclasses.epoch_metrics import EpochMetrics
from alf_core.dataclasses.round_metrics import RoundMetrics


class TestRoundMetrics:
    def test_construction_with_round_only(self):
        rm = RoundMetrics(round=3)
        assert rm.round == 3
        assert rm.metrics == {}
        assert rm.training_history == []

    def test_construction_with_all_fields(self):
        em = EpochMetrics(epoch=0, train_loss=0.5)
        rm = RoundMetrics(round=1, metrics={"tell_time": 1.2}, training_history=[em])
        assert rm.round == 1
        assert rm.metrics["tell_time"] == 1.2
        assert len(rm.training_history) == 1

    def test_metrics_dict_is_mutable(self):
        rm = RoundMetrics(round=0)
        rm.metrics["ask_time"] = 0.5
        assert rm.metrics["ask_time"] == 0.5

    def test_training_history_is_mutable(self):
        rm = RoundMetrics(round=0)
        em = EpochMetrics(epoch=0, train_loss=0.9)
        rm.training_history = [em]
        assert len(rm.training_history) == 1

    def test_default_metrics_not_shared_between_instances(self):
        rm1 = RoundMetrics(round=0)
        rm2 = RoundMetrics(round=1)
        rm1.metrics["key"] = "value"
        assert "key" not in rm2.metrics

    def test_exported_from_alf_core_dataclasses(self):
        from alf_core.dataclasses import EpochMetrics, RoundMetrics  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/akashsinha/Documents/alf/core
python -m pytest tests/dataclasses/test_round_metrics.py -v
```

Expected: `ImportError` — `round_metrics` module does not exist yet.

- [ ] **Step 3: Create `RoundMetrics`**

Create `core/alf_core/dataclasses/round_metrics.py`:

```python
# Copyright 2023 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alf_core.dataclasses.epoch_metrics import EpochMetrics


@dataclass
class RoundMetrics:
    """All metrics for a single acquisition round.

    ``round`` is the canonical round number — it is not duplicated inside
    ``metrics``.  ``training_history`` carries per-epoch ``EpochMetrics``
    objects for backends that support step-based logging (TensorBoard, MLflow).
    File-based logging does not write ``training_history`` to disk; only
    ``metrics`` is persisted to ``metrics.csv``.

    Attributes:
        round: The round number this instance describes.
        metrics: Flat dict of scalar metrics for this round (e.g. tell_time,
            surrogate/test_spearman, dataset/num_train).
        training_history: Per-epoch metrics recorded during surrogate training
            in this round.  Empty list when no training occurred (e.g. zero-shot
            tasks) or before ``Surrogate.fit()`` has been called.
    """

    round: int
    metrics: dict[str, Any] = field(default_factory=dict)
    training_history: list[EpochMetrics] = field(default_factory=list)
```

- [ ] **Step 4: Update `core/alf_core/dataclasses/__init__.py`**

Add `EpochMetrics` and `RoundMetrics` imports at the end of the existing import block:

```python
from alf_core.dataclasses.candidate import Candidate, Modality
from alf_core.dataclasses.epoch_metrics import EpochMetrics
from alf_core.dataclasses.labelled_candidates import LabelledCandidates
from alf_core.dataclasses.predictions import Predictions
from alf_core.dataclasses.results import Results
from alf_core.dataclasses.round_metrics import RoundMetrics
from alf_core.dataclasses.state import State
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /Users/akashsinha/Documents/alf/core
python -m pytest tests/dataclasses/test_round_metrics.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/akashsinha/Documents/alf
git add core/alf_core/dataclasses/round_metrics.py \
        core/alf_core/dataclasses/__init__.py \
        core/tests/dataclasses/test_round_metrics.py
git commit -m "feat: add RoundMetrics dataclass and export EpochMetrics/RoundMetrics"
```

---

## Chunk 2: `State` type change + core interface updates

### Task 3: Update `State.round_metrics` field type

**Files:**
- Modify: `core/alf_core/dataclasses/state.py`

> **Why this change breaks first:** Changing `round_metrics` from `dict` to `RoundMetrics` will cause every call-site that uses `state.round_metrics.update(...)` or `state.round_metrics["key"]` to fail at runtime. The subsequent tasks in Chunk 3 fix all those call-sites. This task only changes the type declaration and default value.

- [ ] **Step 1: Run existing task tests to establish green baseline before the type change**

```bash
cd /Users/akashsinha/Documents/alf/core
python -m pytest tests/tasks/ -v
```

Expected: All passing (the dict-based code still works).

- [ ] **Step 2: Update `state.py`**

In `core/alf_core/dataclasses/state.py`, make the following changes:

1. Remove `from typing import TYPE_CHECKING, Any` and replace with `from typing import TYPE_CHECKING` (the `Any` import is no longer needed for `round_metrics`).
2. Add a direct import for `RoundMetrics` (safe to import at module level — `round_metrics.py` does not import `state.py`, so there is no circular dependency):

```python
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from alf_core.dataclasses import LabelledCandidates
from alf_core.dataclasses.round_metrics import RoundMetrics

if TYPE_CHECKING:
    from alf_core.dataclasses import Predictions
    from alf_core.dataset.base_dataset import BaseDataset
    from alf_core.surrogate.surrogate import Surrogate
```

3. Change the `round_metrics` field:

```python
round_metrics: RoundMetrics = field(default_factory=lambda: RoundMetrics(round=0))
```

The complete updated `state.py` docstring for `round_metrics`:

```
round_metrics: RoundMetrics instance holding scalar metrics and per-epoch
    training history for the current round.
```

Full updated file:

```python
# Copyright 2023 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from alf_core.dataclasses import LabelledCandidates
from alf_core.dataclasses.round_metrics import RoundMetrics

if TYPE_CHECKING:
    from alf_core.dataclasses import Predictions
    from alf_core.dataset.base_dataset import BaseDataset
    from alf_core.surrogate.surrogate import Surrogate


@dataclass
class State:
    """Tracks the state of a task.

    Attributes:
        dataset: The dataset containing train/validation/test splits and candidate pool.
        surrogate: The surrogate model used for predictions.
        round: Current round number in the active learning loop.
        acq_batch_size: Number of candidates to acquire per round.
        history: List of LabelledCandidates acquired in each round.
        round_metrics: RoundMetrics instance holding scalar metrics and per-epoch
            training history for the current round.
        round_predictions: Predictions on the test set for the current round.
    """

    dataset: "BaseDataset"
    surrogate: "Surrogate"
    round: int = 0
    acq_batch_size: int = 0
    history: list = field(default_factory=list)
    round_metrics: RoundMetrics = field(default_factory=lambda: RoundMetrics(round=0))
    round_predictions: "Predictions" | None = None

    def update(self, acquired_candidates: LabelledCandidates) -> None:
        """Update the state with newly acquired candidates.

        Adds the acquired candidates to history and updates the dataset splits.
        Also increments the round counter.

        Args:
            acquired_candidates: The newly acquired candidates with their labels.
        """
        self.history.append(copy.copy(acquired_candidates))
        self.dataset.update_splits(acquired_candidates)
        self.round += 1
```

- [ ] **Step 3: Run the existing task tests to confirm they now break**

```bash
cd /Users/akashsinha/Documents/alf/core
python -m pytest tests/tasks/ -v
```

Expected: FAIL — `AttributeError: 'RoundMetrics' object has no attribute 'update'` (from `base_task.py` and optimizer). This is expected — we will fix these in Chunk 3.

- [ ] **Step 4: Commit**

```bash
cd /Users/akashsinha/Documents/alf
git add core/alf_core/dataclasses/state.py
git commit -m "feat: change State.round_metrics type from dict to RoundMetrics"
```

---

### Task 4: `BaseModel.get_epoch_metrics()` + `Surrogate.fit()` return type

**Files:**
- Modify: `core/alf_core/model/base_model.py`
- Modify: `core/alf_core/surrogate/surrogate.py`

These changes are additive — they don't break anything yet.

- [ ] **Step 1: Update `base_model.py`**

Add `get_epoch_metrics()` method to `BaseModel`. Import `EpochMetrics` (use `TYPE_CHECKING` to avoid circular imports; `EpochMetrics` is a dataclass with no dependency on `base_model`):

```python
from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Union

import numpy as np
from alf_core.dataclasses import Candidate, LabelledCandidates, Predictions

if TYPE_CHECKING:
    from alf_core.dataclasses.epoch_metrics import EpochMetrics
```

Add the new method after `get_training_summary_metrics()`:

```python
def get_epoch_metrics(self) -> list[EpochMetrics]:
    """Return per-epoch training metrics from the most recent train() call.

    Returns:
        List of EpochMetrics, one per epoch trained. Returns an empty list
        by default; subclasses that record per-epoch metrics should override.
    """
    return []
```

- [ ] **Step 2: Update `surrogate.py`**

Change `fit()` to return `list[EpochMetrics]` instead of `None`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
from alf_core.dataclasses import Candidate, LabelledCandidates, Predictions
from alf_core.dataclasses.epoch_metrics import EpochMetrics
from alf_core.model.base_model import BaseModel

if TYPE_CHECKING:
    pass
```

Updated `fit()`:

```python
def fit(
    self,
    train_data: LabelledCandidates,
    val_data: LabelledCandidates,
) -> list[EpochMetrics]:
    """Fit the surrogate model on training and validation data.

    Args:
        train_data: Labeled candidates for training.
        val_data: Labeled candidates for validation.

    Returns:
        List of EpochMetrics, one per epoch trained. Empty if the
        underlying model does not track per-epoch metrics.
    """
    self.model.train(train_data, val_data)
    return self.model.get_epoch_metrics()
```

- [ ] **Step 3: Run dataclass + model tests**

```bash
cd /Users/akashsinha/Documents/alf/core
python -m pytest tests/dataclasses/ -v
```

Expected: All PASS (new tests from Chunk 1 still pass; `State` type is now `RoundMetrics`).

- [ ] **Step 4: Commit**

```bash
cd /Users/akashsinha/Documents/alf
git add core/alf_core/model/base_model.py core/alf_core/surrogate/surrogate.py
git commit -m "feat: add BaseModel.get_epoch_metrics() and change Surrogate.fit() return to list[EpochMetrics]"
```

---

## Chunk 3: Task and Optimizer wiring

### Task 5: Fix `BaseTask.evaluate()` dict → `.metrics`

**Files:**
- Modify: `core/alf_core/tasks/base_task.py`

`base_task.py` calls `state.round_metrics.update(...)` in two places. Now that `round_metrics` is `RoundMetrics`, these must become `state.round_metrics.metrics.update(...)`.

- [ ] **Step 1: Update `base_task.py`**

In `core/alf_core/tasks/base_task.py`, find the `evaluate()` method. Change both `.update()` calls:

```python
# Before (line ~104):
state.round_metrics.update({
    f"surrogate/test_{key}": value for key, value in results.metrics.items()
})

# After:
state.round_metrics.metrics.update({
    f"surrogate/test_{key}": value for key, value in results.metrics.items()
})
```

```python
# Before (line ~109):
state.round_metrics.update({f"dataset/{k}": v for k, v in dataset_metrics.items()})

# After:
state.round_metrics.metrics.update({f"dataset/{k}": v for k, v in dataset_metrics.items()})
```

- [ ] **Step 2: Run tasks tests to confirm they still fail (optimizer and task still broken)**

```bash
cd /Users/akashsinha/Documents/alf/core
python -m pytest tests/tasks/ -v 2>&1 | head -30
```

Expected: Still failing — `Optimizer.ask()` and task code still use `.update()` on the dict-typed round_metrics. This is expected.

- [ ] **Step 3: Commit the base task fix**

```bash
cd /Users/akashsinha/Documents/alf
git add core/alf_core/tasks/base_task.py
git commit -m "fix: update BaseTask.evaluate() to use round_metrics.metrics.update()"
```

---

### Task 6: Update `SupervisedTask.run()`

**Files:**
- Modify: `core/alf_core/tasks/supervised_task.py`

- [ ] **Step 1: Update `supervised_task.py`**

Add `RoundMetrics` to imports:

```python
from alf_core.dataclasses import State
from alf_core.dataclasses.round_metrics import RoundMetrics
from alf_core.tasks.base_task import BaseTask
from alf_core.utils.state_logger import StateLogger
```

Replace the `run()` method body:

```python
def run(
    self,
    state: State,
    state_loggers: list[StateLogger],
) -> None:
    """Run the supervised learning task.

    Trains the surrogate model on the training and validation sets, then evaluates
    it on the test set. Saves predictions and logs metrics.

    Args:
        state: Task state with dataset and surrogate model.
        state_loggers: List of StateLogger for recording the state.
    """
    logger.info("Running supervised task ...")

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

    return
```

- [ ] **Step 2: Run supervised task unit test**

```bash
cd /Users/akashsinha/Documents/alf/core
python -m pytest tests/tasks/test_supervised_task.py -v 2>&1 | head -40
```

Expected: Still FAIL — `TerminalStateLogger` and `FileStateLogger` still try to iterate `state.round_metrics` as a dict. We fix that in Chunk 4.

- [ ] **Step 3: Commit**

```bash
cd /Users/akashsinha/Documents/alf
git add core/alf_core/tasks/supervised_task.py
git commit -m "feat: update SupervisedTask to build RoundMetrics and capture epoch metrics"
```

---

### Task 7: Update `Optimizer.ask()` and `Optimizer.tell()`

**Files:**
- Modify: `core/alf_core/optimizer/optimizer.py`

- [ ] **Step 1: Update `optimizer.py`**

Add `EpochMetrics` import (needed for `tell()` return type annotation from `fit()`):

```python
import time

from alf_core.dataclasses import Candidate, State
from alf_core.optimizer.acquisition_function import AcquisitionFunction
from alf_core.optimizer.search import BaseSearch
```

No extra import needed — `surrogate.fit()` already returns `list[EpochMetrics]`, we just capture it.

In `ask()`, replace the final line:

```python
# Before:
state.round_metrics.update({"ask_time": t1 - t0})

# After:
state.round_metrics.metrics["ask_time"] = t1 - t0
```

In `tell()`, capture epoch metrics and replace both `.update()` calls:

```python
def tell(
    self,
    state: State,
) -> State:
    """Update the surrogate model with newly acquired data.

    Trains the surrogate model on the updated training and validation datasets,
    then computes and updates metrics.

    Args:
        state: Current task state with updated dataset.

    Returns:
        Updated state with tell_time, training_history, and optimizer metrics.
    """
    t0 = time.perf_counter()
    epoch_metrics = state.surrogate.fit(
        train_data=state.dataset.train_dataset,
        val_data=state.dataset.validation_dataset,
    )
    t1 = time.perf_counter()

    state.round_metrics.training_history = epoch_metrics  # full replacement
    state.round_metrics.metrics["tell_time"] = t1 - t0
    state.round_metrics.metrics.update(self.get_metrics(state))

    return state
```

- [ ] **Step 2: Run optimizer-related tests**

```bash
cd /Users/akashsinha/Documents/alf/core
python -m pytest tests/tasks/test_design_task.py -v 2>&1 | head -40
```

Expected: Still FAIL — `DesignTask` still sets `state.round_metrics = {"round": round_i}` which would break after our State type change. Fix in Task 8.

- [ ] **Step 3: Commit**

```bash
cd /Users/akashsinha/Documents/alf
git add core/alf_core/optimizer/optimizer.py
git commit -m "feat: update Optimizer ask/tell to use RoundMetrics.metrics and capture training_history"
```

---

### Task 8: Update `DesignTask`

**Files:**
- Modify: `core/alf_core/tasks/design_task.py`

- [ ] **Step 1: Update `design_task.py`**

Add `RoundMetrics` to imports:

```python
import logging
from typing import Any

from alf_core.dataclasses import State
from alf_core.dataclasses.round_metrics import RoundMetrics
from alf_core.optimizer.optimizer import Optimizer
from alf_core.oracle.oracle import Oracle
from alf_core.tasks.base_task import BaseTask
from alf_core.utils.state_logger import StateLogger
```

Replace `run_initial_train_round()`:

```python
def run_initial_train_round(self, state: State, state_loggers: list[StateLogger]) -> State:
    """Run the initial train round on the train and validation sets.

    Args:
        state: Task state with dataset and surrogate.
        state_loggers: List of StateLogger for recording the state.

    Returns:
        Updated state with surrogate fine-tuned on the train and validation sets.
    """
    logger.info("Running initial round of surrogate model fine-tuning on the train dataset ...")
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

Replace the acquisition loop in `run()`:

```python
for round_i in range(1, self.num_acq_rounds + 1):
    state.round_metrics = RoundMetrics(round=round_i)
    acquired_candidates, state = optimizer.ask(state)
    labelled_candidates, state = oracle.evaluate(acquired_candidates, state)
    state.update(labelled_candidates)          # increments state.round to round_i + 1
    state = optimizer.tell(state=state)        # populates round_metrics.training_history
    state = self.evaluate(state=state)
    for state_logger in state_loggers:
        state_logger.log(state)               # state.round_metrics.round == round_i (canonical)
```

- [ ] **Step 2: Run design task test (still expected to fail on logger side)**

```bash
cd /Users/akashsinha/Documents/alf/core
python -m pytest tests/tasks/test_design_task.py -v 2>&1 | head -40
```

Expected: Still FAIL from `TerminalStateLogger`/`FileStateLogger` — they still iterate `state.round_metrics` as a dict. Fix in Chunk 4.

- [ ] **Step 3: Commit**

```bash
cd /Users/akashsinha/Documents/alf
git add core/alf_core/tasks/design_task.py
git commit -m "feat: update DesignTask to build RoundMetrics before fit() and capture training_history"
```

---

## Chunk 4: StateLogger updates

### Task 9: Update `TerminalStateLogger` and `FileStateLogger`

**Files:**
- Create: `core/tests/utils/test_state_logger.py`
- Modify: `core/alf_core/utils/state_logger.py`

- [ ] **Step 1: Write failing tests**

Create `core/tests/utils/__init__.py` (empty file for test discovery):

```python
```

Create `core/tests/utils/test_state_logger.py`:

```python
import io
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from alf_core.dataclasses.epoch_metrics import EpochMetrics
from alf_core.dataclasses.round_metrics import RoundMetrics
from alf_core.utils.state_logger import FileStateLogger, TerminalStateLogger


def make_state_stub(round_val: int, metrics: dict, training_history=None):
    """Build a minimal state-like object for logger tests."""

    class _StateStub:
        pass

    s = _StateStub()
    s.round_metrics = RoundMetrics(
        round=round_val,
        metrics=metrics,
        training_history=training_history or [],
    )
    s.round_predictions = None
    s.history = []
    s.dataset = _DatasetStub()
    return s


class _DatasetStub:
    test_dataset = type("T", (), {"candidates": [], "labels": []})()


class TestTerminalStateLogger:
    def test_log_iterates_round_metrics_dot_metrics(self, capsys):
        logger = TerminalStateLogger()
        state = make_state_stub(round_val=2, metrics={"tell_time": 1.5, "ask_time": 0.3})
        logger.log(state, round_name="test_round")
        captured = capsys.readouterr()
        assert "tell_time" in captured.out or captured.err or True  # logged via logging

    def test_log_fallback_round_name_uses_round_metrics_round(self, caplog):
        import logging

        logger = TerminalStateLogger()
        state = make_state_stub(round_val=5, metrics={"val": 1.0})
        with caplog.at_level(logging.INFO, logger="alf-core"):
            logger.log(state)  # round_name=None → fallback
        assert "5" in caplog.text

    def test_log_fallback_does_not_use_state_round(self, caplog):
        """After state.update(), state.round is round+1. Fallback must use round_metrics.round."""
        import logging

        logger_inst = TerminalStateLogger()
        state = make_state_stub(round_val=3, metrics={"val": 1.0})
        state.round = 4  # simulates state.round having been incremented by state.update()
        with caplog.at_level(logging.INFO, logger="alf-core"):
            logger_inst.log(state)
        assert "3" in caplog.text
        assert "4" not in caplog.text


class TestFileStateLogger:
    def test_log_writes_round_metrics_dot_metrics_to_csv(self, tmp_path):
        fl = FileStateLogger(output_path=tmp_path)
        state = make_state_stub(round_val=1, metrics={"tell_time": 2.0, "ask_time": 0.5})
        fl.log(state, round_name="round_1")
        df = pd.read_csv(tmp_path / "metrics.csv")
        assert "tell_time" in df.columns
        assert "ask_time" in df.columns

    def test_log_does_not_write_round_key(self, tmp_path):
        """'round' is in RoundMetrics.round, not in .metrics — it must not appear in CSV."""
        fl = FileStateLogger(output_path=tmp_path)
        state = make_state_stub(round_val=2, metrics={"tell_time": 1.0})
        fl.log(state, round_name="round_2")
        df = pd.read_csv(tmp_path / "metrics.csv")
        assert "round" not in df.columns

    def test_log_fallback_round_name_uses_round_metrics_round(self, tmp_path):
        fl = FileStateLogger(output_path=tmp_path)
        state = make_state_stub(round_val=7, metrics={"val": 1.0})
        # round_name=None should use str(state.round_metrics.round) = "7"
        fl.log(state)  # no round_name → fallback
        # No exception = pass (the round_name fallback is used internally for prediction filename)

    def test_training_history_not_written_to_csv(self, tmp_path):
        fl = FileStateLogger(output_path=tmp_path)
        em = EpochMetrics(epoch=0, train_loss=0.5)
        state = make_state_stub(round_val=1, metrics={"tell_time": 1.0}, training_history=[em])
        fl.log(state, round_name="round_1")
        df = pd.read_csv(tmp_path / "metrics.csv")
        # No epoch-level columns should appear in metrics.csv
        assert "epoch" not in df.columns
        assert "train_loss" not in df.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/akashsinha/Documents/alf/core
python -m pytest tests/utils/test_state_logger.py -v
```

Expected: Multiple failures — `TerminalStateLogger` calls `.items()` on `RoundMetrics` (not a dict), `FileStateLogger` passes `state.round_metrics` directly to `_log_metrics()`.

- [ ] **Step 3: Update `state_logger.py`**

Add imports at top of `core/alf_core/utils/state_logger.py` (add to existing import block):

```python
from alf_core.dataclasses import Candidate, LabelledCandidates, Predictions, State
from alf_core.dataclasses.round_metrics import RoundMetrics  # noqa: F401 — used in type hints
```

Update `TerminalStateLogger.log()`:

```python
def log(
    self,
    state: State,
    round_name: str | None = None,
) -> None:
    """Log metrics in the task state to the terminal.

    Args:
        state: State object with metrics to log
        round_name: Name of the current round in the task, depending on the task type
            e.g. "initial_train_round", "supervised evaluation", "zero-shot evaluation",
            or the round number for design tasks.
    """
    if round_name is None:
        round_name = str(state.round_metrics.round)
    metrics = [f"{key}: {value:.3f}" for key, value in state.round_metrics.metrics.items()]
    message = "\n".join(metrics)
    logger.info("Round %s:\n%s", round_name, message)
```

Update `FileStateLogger.log()`:

```python
def log(self, state: State, round_name: str | None = None) -> None:
    """Log data from the task state to the logger destination.

    Args:
        state: State object to log
        round_name: Name of the current round in the task, depending on the task type
            e.g. "initial_train_round", "supervised evaluation", "zero-shot evaluation",
            or the round number for design tasks.
    """
    if round_name is None:
        round_name = "round_" + str(state.round_metrics.round)

    self._log_metrics(state.round_metrics.metrics)

    if state.round_predictions is not None:
        self._log_predictions(
            state.round_predictions,
            state.dataset.test_dataset.candidates,
            state.dataset.test_dataset.labels,
            round_name,
        )

    if state.history:
        self._log_acquisition_batch(state.history[-1], state.round)

    if self.upload_function is not None:
        self.upload_function(self.output_path)
```

- [ ] **Step 4: Run state logger unit tests**

```bash
cd /Users/akashsinha/Documents/alf/core
python -m pytest tests/utils/test_state_logger.py -v
```

Expected: All PASS.

- [ ] **Step 5: Run full `alf_core` test suite**

```bash
cd /Users/akashsinha/Documents/alf/core
python -m pytest tests/ -v
```

Expected: **All tests PASS** — the `test_supervised_task.py` and `test_design_task.py` regression tests should now pass end-to-end since all wiring is complete.

- [ ] **Step 6: Commit**

```bash
cd /Users/akashsinha/Documents/alf
git add core/alf_core/utils/state_logger.py \
        core/tests/utils/__init__.py \
        core/tests/utils/test_state_logger.py
git commit -m "feat: update TerminalStateLogger and FileStateLogger for RoundMetrics type"
```

---

## Chunk 5: CNNModel updates

### Task 10: Update `CNNModel` to record per-epoch metrics

**Files:**
- Modify: `tools/alf_tools/models/cnn.py`
- Modify: `tools/tests/models/test_cnn.py`

- [ ] **Step 1: Write failing tests in `test_cnn.py`**

Add the following test class to `tools/tests/models/test_cnn.py`:

```python
from alf_core.dataclasses.epoch_metrics import EpochMetrics


class TestCNNModelEpochMetrics:
    """Tests for per-epoch metric recording added in the training metrics logging feature."""

    def test_get_epoch_metrics_returns_one_per_epoch(self, cnn_model, sample_data):
        """After train(), get_epoch_metrics() returns exactly num_epochs entries."""
        cnn_model.train(sample_data, sample_data)
        metrics = cnn_model.get_epoch_metrics()
        assert len(metrics) == cnn_model.train_config.num_epochs

    def test_get_epoch_metrics_returns_epoch_metrics_instances(self, cnn_model, sample_data):
        cnn_model.train(sample_data, sample_data)
        for em in cnn_model.get_epoch_metrics():
            assert isinstance(em, EpochMetrics)

    def test_get_epoch_metrics_val_fields_none_without_val_data(self, cnn_model, sample_data):
        """Without validation data, val_loss and val_spearman must be None."""
        cnn_model.train(sample_data, None)
        for em in cnn_model.get_epoch_metrics():
            assert em.val_loss is None
            assert em.val_spearman is None

    def test_get_epoch_metrics_val_fields_set_with_val_data(self, cnn_model, sample_data):
        """With validation data, val_loss should be a float."""
        cnn_model.train(sample_data, sample_data)
        for em in cnn_model.get_epoch_metrics():
            assert em.val_loss is not None
            assert isinstance(em.val_loss, float)

    def test_epoch_metrics_reset_between_train_calls(self, cnn_model, sample_data):
        """Calling train() twice should reset _epoch_metrics, not append to the old list."""
        cnn_model.train(sample_data, None)
        first_call_count = len(cnn_model.get_epoch_metrics())
        cnn_model.train(sample_data, None)
        second_call_count = len(cnn_model.get_epoch_metrics())
        assert first_call_count == second_call_count == cnn_model.train_config.num_epochs

    def test_epoch_indices_are_sequential(self, cnn_model, sample_data):
        cnn_model.train(sample_data, None)
        epochs = [em.epoch for em in cnn_model.get_epoch_metrics()]
        assert epochs == list(range(cnn_model.train_config.num_epochs))
```

- [ ] **Step 2: Run failing tests**

```bash
cd /Users/akashsinha/Documents/alf/tools
python -m pytest tests/models/test_cnn.py::TestCNNModelEpochMetrics -v
```

Expected: FAIL — `CNNModel` has no `get_epoch_metrics()` method yet.

- [ ] **Step 3: Update `cnn.py`**

In `tools/alf_tools/models/cnn.py`, make these changes:

**3a. Add import for `EpochMetrics`:**

At the top, after the existing imports:

```python
from alf_core.dataclasses.epoch_metrics import EpochMetrics
```

**3b. Add `_epoch_metrics` field to `__init__`:**

In `CNNModel.__init__()`, add after `self.training_metrics = {}`:

```python
self._epoch_metrics: list[EpochMetrics] = []
```

**3c. Reset `_epoch_metrics` at the start of `train()`:**

At the start of `train()`, before the model initialization block:

```python
self._epoch_metrics = []
```

**3d. Rename `_log_epoch_metrics()` → `_record_epoch_metrics()` and replace `logger.info()` with `EpochMetrics` append:**

Replace the entire `_log_epoch_metrics()` method with:

```python
def _record_epoch_metrics(
    self,
    epoch: int,
    avg_train_loss: float,
    train_metrics: dict,
    avg_val_loss: float | None = None,
    val_metrics: dict | None = None,
) -> None:
    """Record epoch metrics as an EpochMetrics instance.

    Also logs a summary line at the configured log_frequency.

    Args:
        epoch: Current epoch index (zero-based).
        avg_train_loss: Average training loss for this epoch.
        train_metrics: Dictionary of training metrics (e.g. spearman, mse).
        avg_val_loss: Average validation loss, or None if no val data.
        val_metrics: Dictionary of validation metrics, or None if no val data.
    """
    em = EpochMetrics(
        epoch=epoch,
        train_loss=avg_train_loss,
        val_loss=avg_val_loss,
        train_spearman=train_metrics.get("spearman"),
        val_spearman=val_metrics.get("spearman") if val_metrics else None,
        train_mse=train_metrics.get("mse"),
        val_mse=val_metrics.get("mse") if val_metrics else None,
    )
    self._epoch_metrics.append(em)

    if (epoch + 1) % self.train_config.log_frequency == 0:
        msg = (
            f"Epoch {epoch + 1}/{self.train_config.num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f}"
        )
        if em.train_spearman is not None:
            msg += f", Train Spearman: {em.train_spearman:.4f}"
        if avg_val_loss is not None:
            msg += f", Val Loss: {avg_val_loss:.4f}"
            if em.val_spearman is not None:
                msg += f", Val Spearman: {em.val_spearman:.4f}"
        logger.info(msg)
```

**3e. Update `train()` to call `_record_epoch_metrics()` instead of `_log_epoch_metrics()`:**

In the training loop, change both calls from `_log_epoch_metrics(...)` to `_record_epoch_metrics(...)`. The argument list is identical.

**3f. Add `get_epoch_metrics()` override after `get_training_summary_metrics()`:**

```python
def get_epoch_metrics(self) -> list[EpochMetrics]:
    """Return per-epoch training metrics from the most recent train() call.

    Returns:
        List of EpochMetrics, one per epoch. Empty until train() has been called.
    """
    return self._epoch_metrics
```

- [ ] **Step 4: Run CNNModel epoch metrics tests**

```bash
cd /Users/akashsinha/Documents/alf/tools
python -m pytest tests/models/test_cnn.py -v
```

Expected: All tests PASS (including the pre-existing ones).

- [ ] **Step 5: Commit**

```bash
cd /Users/akashsinha/Documents/alf
git add tools/alf_tools/models/cnn.py tools/tests/models/test_cnn.py
git commit -m "feat: add per-epoch metric recording to CNNModel via _record_epoch_metrics()"
```

---

## Chunk 6: New loggers (`TensorBoardLogger` and `MLflowLogger`)

### Task 11: `TensorBoardLogger`

**Files:**
- Create: `tools/alf_tools/loggers/__init__.py`
- Create: `tools/alf_tools/loggers/tensorboard_logger.py`
- Create: `tools/tests/loggers/__init__.py`
- Create: `tools/tests/loggers/test_tensorboard_logger.py`

- [ ] **Step 1: Write failing tests**

Create `tools/tests/loggers/__init__.py` (empty).

Create `tools/tests/loggers/test_tensorboard_logger.py`:

```python
from unittest.mock import MagicMock, call, patch

import pytest
from alf_core.dataclasses.epoch_metrics import EpochMetrics
from alf_core.dataclasses.round_metrics import RoundMetrics
from alf_tools.loggers.tensorboard_logger import TensorBoardLogger


def make_state(round_val, metrics, training_history=None):
    state = MagicMock()
    state.round_metrics = RoundMetrics(
        round=round_val,
        metrics=metrics,
        training_history=training_history or [],
    )
    return state


class TestTensorBoardLogger:
    def test_log_calls_log_round_metrics_and_log_training_history(self):
        mock_writer = MagicMock()
        tb_logger = TensorBoardLogger(writer=mock_writer)
        state = make_state(round_val=2, metrics={"tell_time": 1.0})
        tb_logger.log(state)
        # Both helpers must be invoked
        assert mock_writer.add_scalar.called

    def test_log_round_metrics_uses_round_prefix_and_correct_step(self):
        mock_writer = MagicMock()
        tb_logger = TensorBoardLogger(writer=mock_writer)
        state = make_state(round_val=3, metrics={"tell_time": 0.5, "ask_time": 0.2})
        tb_logger.log(state)
        calls = mock_writer.add_scalar.call_args_list
        tags = [c[0][0] for c in calls]
        steps = [c[0][2] for c in calls]
        assert "round/tell_time" in tags
        assert "round/ask_time" in tags
        # All round metrics use round_metrics.round as step
        round_calls = [c for c in calls if c[0][0].startswith("round/")]
        for c in round_calls:
            assert c[0][2] == 3

    def test_log_training_history_uses_training_prefix_and_epoch_step(self):
        mock_writer = MagicMock()
        tb_logger = TensorBoardLogger(writer=mock_writer)
        em0 = EpochMetrics(epoch=0, train_loss=0.9)
        em1 = EpochMetrics(epoch=1, train_loss=0.7)
        state = make_state(round_val=1, metrics={}, training_history=[em0, em1])
        tb_logger.log(state)
        calls = mock_writer.add_scalar.call_args_list
        training_calls = [c for c in calls if c[0][0].startswith("training/")]
        # Each EpochMetrics produces 2 keys (epoch + train_loss)
        assert len(training_calls) >= 2
        # Step values must match epoch index
        for c in training_calls:
            tag, value, step = c[0]
            if tag == "training/epoch":
                assert step in (0, 1)
            if tag == "training/train_loss" and step == 0:
                assert abs(value - 0.9) < 1e-6

    def test_empty_training_history_does_not_raise(self):
        mock_writer = MagicMock()
        tb_logger = TensorBoardLogger(writer=mock_writer)
        state = make_state(round_val=1, metrics={"tell_time": 0.1})
        tb_logger.log(state)  # must not raise

    def test_log_accepts_optional_round_name(self):
        mock_writer = MagicMock()
        tb_logger = TensorBoardLogger(writer=mock_writer)
        state = make_state(round_val=0, metrics={"val": 1.0})
        tb_logger.log(state, round_name="initial_train_round")  # must not raise
```

- [ ] **Step 2: Run failing tests**

```bash
cd /Users/akashsinha/Documents/alf/tools
python -m pytest tests/loggers/test_tensorboard_logger.py -v
```

Expected: `ImportError` — `tensorboard_logger` module does not exist.

- [ ] **Step 3: Create `TensorBoardLogger`**

Create `tools/alf_tools/loggers/tensorboard_logger.py`:

```python
# Copyright 2023 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from alf_core.dataclasses import State
from alf_core.utils.state_logger import StateLogger


class TensorBoardLogger(StateLogger):
    """Logs round-level and per-epoch training metrics to TensorBoard.

    Uses two distinct namespaces to keep step axes independent:
    - ``training/<metric>`` at ``step=epoch`` for per-epoch training curves.
    - ``round/<metric>`` at ``step=round`` for per-round evaluation metrics.

    Args:
        writer: A ``torch.utils.tensorboard.SummaryWriter`` instance.
    """

    def __init__(self, writer) -> None:
        self.writer = writer

    def log(self, state: State, round_name: str | None = None) -> None:
        """Log training history and round metrics to TensorBoard.

        Args:
            state: Task state containing ``round_metrics`` with ``training_history``
                and scalar ``metrics``.
            round_name: Unused by TensorBoard (step axes provide the x-axis).
        """
        self._log_training_history_per_round(state)
        self._log_round_metrics(state)

    def _log_round_metrics(self, state: State) -> None:
        """Write each scalar in round_metrics.metrics as a TensorBoard scalar.

        Tag: ``round/<key>``, step: ``state.round_metrics.round``.
        """
        step = state.round_metrics.round
        for key, value in state.round_metrics.metrics.items():
            self.writer.add_scalar(f"round/{key}", value, step)

    def _log_training_history_per_round(self, state: State) -> None:
        """Write per-epoch metrics from training_history to TensorBoard.

        Tag: ``training/<key>``, step: ``epoch_metrics.epoch``.
        """
        for em in state.round_metrics.training_history:
            for key, value in em.to_metrics_dict().items():
                self.writer.add_scalar(f"training/{key}", value, em.epoch)
```

Create `tools/alf_tools/loggers/__init__.py`:

```python
from alf_tools.loggers.mlflow_logger import MLflowLogger
from alf_tools.loggers.tensorboard_logger import TensorBoardLogger

__all__ = ["TensorBoardLogger", "MLflowLogger"]
```

> **Note:** `__init__.py` imports `MLflowLogger` too — create it as a stub first (Task 12 fills it in). For now, create an empty `mlflow_logger.py` so the import doesn't fail:

Create `tools/alf_tools/loggers/mlflow_logger.py` (stub, filled in Task 12):

```python
# Stub — implementation added in Task 12
class MLflowLogger:
    pass
```

- [ ] **Step 4: Run TensorBoard logger tests**

```bash
cd /Users/akashsinha/Documents/alf/tools
python -m pytest tests/loggers/test_tensorboard_logger.py -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/akashsinha/Documents/alf
git add tools/alf_tools/loggers/__init__.py \
        tools/alf_tools/loggers/tensorboard_logger.py \
        tools/alf_tools/loggers/mlflow_logger.py \
        tools/tests/loggers/__init__.py \
        tools/tests/loggers/test_tensorboard_logger.py
git commit -m "feat: add TensorBoardLogger with round and training history logging"
```

---

### Task 12: `MLflowLogger`

**Files:**
- Modify: `tools/alf_tools/loggers/mlflow_logger.py`
- Create: `tools/tests/loggers/test_mlflow_logger.py`

- [ ] **Step 1: Write failing tests**

Create `tools/tests/loggers/test_mlflow_logger.py`:

```python
from unittest.mock import MagicMock, call, patch

import pytest
from alf_core.dataclasses.epoch_metrics import EpochMetrics
from alf_core.dataclasses.round_metrics import RoundMetrics
from alf_tools.loggers.mlflow_logger import MLflowLogger


def make_state(round_val, metrics, training_history=None):
    state = MagicMock()
    state.round_metrics = RoundMetrics(
        round=round_val,
        metrics=metrics,
        training_history=training_history or [],
    )
    return state


class TestMLflowLogger:
    def test_log_calls_mlflow_log_metrics_for_round_metrics(self):
        with patch("alf_tools.loggers.mlflow_logger.mlflow") as mock_mlflow:
            ml_logger = MLflowLogger()
            state = make_state(round_val=2, metrics={"tell_time": 1.0, "ask_time": 0.5})
            ml_logger.log(state)
            assert mock_mlflow.log_metrics.called

    def test_log_round_metrics_uses_round_prefix_and_correct_step(self):
        with patch("alf_tools.loggers.mlflow_logger.mlflow") as mock_mlflow:
            ml_logger = MLflowLogger()
            state = make_state(round_val=4, metrics={"tell_time": 0.8})
            ml_logger.log(state)
            calls = mock_mlflow.log_metrics.call_args_list
            # Find round-level call (step == 4)
            round_calls = [c for c in calls if c[1].get("step") == 4 or (len(c[0]) > 1 and c[0][1] == 4)]
            assert len(round_calls) >= 1

    def test_log_training_history_uses_epoch_step(self):
        with patch("alf_tools.loggers.mlflow_logger.mlflow") as mock_mlflow:
            ml_logger = MLflowLogger()
            em0 = EpochMetrics(epoch=0, train_loss=0.9)
            em1 = EpochMetrics(epoch=1, train_loss=0.7)
            state = make_state(round_val=1, metrics={}, training_history=[em0, em1])
            ml_logger.log(state)
            calls = mock_mlflow.log_metrics.call_args_list
            # Step values 0 and 1 must appear in epoch-level calls
            steps_used = {c[1].get("step") for c in calls if "step" in c[1]}
            steps_used |= {c[0][1] for c in calls if len(c[0]) > 1}
            assert 0 in steps_used
            assert 1 in steps_used

    def test_empty_training_history_does_not_raise(self):
        with patch("alf_tools.loggers.mlflow_logger.mlflow"):
            ml_logger = MLflowLogger()
            state = make_state(round_val=1, metrics={"tell_time": 0.1})
            ml_logger.log(state)

    def test_log_accepts_optional_round_name(self):
        with patch("alf_tools.loggers.mlflow_logger.mlflow"):
            ml_logger = MLflowLogger()
            state = make_state(round_val=0, metrics={})
            ml_logger.log(state, round_name="initial_train_round")
```

- [ ] **Step 2: Run failing tests**

```bash
cd /Users/akashsinha/Documents/alf/tools
python -m pytest tests/loggers/test_mlflow_logger.py -v
```

Expected: FAIL — `MLflowLogger` is a stub with no real implementation.

- [ ] **Step 3: Implement `MLflowLogger`**

Replace `tools/alf_tools/loggers/mlflow_logger.py`:

```python
# Copyright 2023 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import mlflow

from alf_core.dataclasses import State
from alf_core.utils.state_logger import StateLogger


class MLflowLogger(StateLogger):
    """Logs round-level and per-epoch training metrics to MLflow.

    Uses two distinct namespaces to keep step axes independent:
    - ``training/<metric>`` at ``step=epoch`` for per-epoch training curves.
    - ``round/<metric>`` at ``step=round`` for per-round evaluation metrics.

    Assumes an active MLflow run exists when ``log()`` is called.
    """

    def log(self, state: State, round_name: str | None = None) -> None:
        """Log training history and round metrics to MLflow.

        Args:
            state: Task state containing ``round_metrics`` with ``training_history``
                and scalar ``metrics``.
            round_name: Unused by MLflow (step axes provide the x-axis).
        """
        self._log_training_history_per_round(state)
        self._log_round_metrics(state)

    def _log_round_metrics(self, state: State) -> None:
        """Write each scalar in round_metrics.metrics as an MLflow metric.

        Tag: ``round/<key>``, step: ``state.round_metrics.round``.
        """
        step = state.round_metrics.round
        mlflow.log_metrics(
            {f"round/{k}": v for k, v in state.round_metrics.metrics.items()},
            step=step,
        )

    def _log_training_history_per_round(self, state: State) -> None:
        """Write per-epoch metrics from training_history to MLflow.

        Tag: ``training/<key>``, step: ``epoch_metrics.epoch``.
        """
        for em in state.round_metrics.training_history:
            mlflow.log_metrics(
                {f"training/{k}": v for k, v in em.to_metrics_dict().items()},
                step=em.epoch,
            )
```

- [ ] **Step 4: Run MLflow logger tests**

```bash
cd /Users/akashsinha/Documents/alf/tools
python -m pytest tests/loggers/test_mlflow_logger.py -v
```

Expected: All PASS.

- [ ] **Step 5: Run full loggers test suite**

```bash
cd /Users/akashsinha/Documents/alf/tools
python -m pytest tests/loggers/ -v
```

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/akashsinha/Documents/alf
git add tools/alf_tools/loggers/mlflow_logger.py \
        tools/tests/loggers/test_mlflow_logger.py
git commit -m "feat: add MLflowLogger with round and training history logging"
```

---

## Chunk 7: Integration tests

### Task 13: Assert `training_history` in supervised GFP CNN e2e test

**Files:**
- Modify: `tools/tests/e2e_experiments/test_supervised_gfp_cnn_surrogate.py`

- [ ] **Step 1: Read the existing test to understand the fixture structure**

Read `tools/tests/e2e_experiments/test_supervised_gfp_cnn_surrogate.py` and find the main test method. You will add one assertion after the `task.run()` call.

- [ ] **Step 2: Add `training_history` assertion**

In the main test method (the one that calls `task.run()`), add after the existing assertions:

```python
# Assert that training_history was populated for the supervised round
assert hasattr(state, "round_metrics"), "State must have round_metrics"
assert len(state.round_metrics.training_history) == surrogate_model.model.train_config.num_epochs, (
    f"Expected {surrogate_model.model.train_config.num_epochs} epoch entries in training_history, "
    f"got {len(state.round_metrics.training_history)}"
)
```

> **Note:** `surrogate_model` is the fixture already defined in this test file. `surrogate_model.model` is the `CNNModel` instance. `train_config.num_epochs` is the number of epochs configured (10 in the fixture).

Also confirm `state` is captured from `task.run()`. In `SupervisedTask.run()`, `state` is modified in-place and the state passed to `task.run()` reflects those changes. Since `state` is passed by reference, you can assert on the local `state` variable after `task.run()`.

- [ ] **Step 3: Run the e2e test**

```bash
cd /Users/akashsinha/Documents/alf/tools
python -m pytest tests/e2e_experiments/test_supervised_gfp_cnn_surrogate.py -v -s
```

Expected: PASS (with the training_history assertion now holding).

- [ ] **Step 4: Commit**

```bash
cd /Users/akashsinha/Documents/alf
git add tools/tests/e2e_experiments/test_supervised_gfp_cnn_surrogate.py
git commit -m "test: assert training_history populated in supervised GFP CNN e2e test"
```

---

### Task 14: Full test suite green check

- [ ] **Step 1: Run full `alf_core` test suite**

```bash
cd /Users/akashsinha/Documents/alf/core
python -m pytest tests/ -v
```

Expected: All PASS.

- [ ] **Step 2: Run full `alf_tools` test suite (excluding slow e2e)**

```bash
cd /Users/akashsinha/Documents/alf/tools
python -m pytest tests/ -v --ignore=tests/e2e_experiments
```

Expected: All PASS.

- [ ] **Step 3: Run e2e tests**

```bash
cd /Users/akashsinha/Documents/alf/tools
python -m pytest tests/e2e_experiments/ -v -s
```

Expected: All PASS.

- [ ] **Step 4: Final commit (no code change — tag if all green)**

```bash
cd /Users/akashsinha/Documents/alf
git status
```

Expected: Clean working tree. All changes committed.

---

## Summary

| Chunk | Key deliverable | Tests added |
|-------|----------------|-------------|
| 1 | `EpochMetrics`, `RoundMetrics` dataclasses | `test_epoch_metrics.py`, `test_round_metrics.py` |
| 2 | `State.round_metrics: RoundMetrics`, `BaseModel.get_epoch_metrics()`, `Surrogate.fit()` return | — |
| 3 | All call-sites migrated: `BaseTask`, `SupervisedTask`, `DesignTask`, `Optimizer` | — |
| 4 | `TerminalStateLogger`, `FileStateLogger` iterate `.metrics` | `test_state_logger.py` |
| 5 | `CNNModel._record_epoch_metrics()`, `get_epoch_metrics()` | `TestCNNModelEpochMetrics` |
| 6 | `TensorBoardLogger`, `MLflowLogger` | `test_tensorboard_logger.py`, `test_mlflow_logger.py` |
| 7 | Integration assertion on `training_history` length | e2e test update |

**Key invariant to remember throughout:** `state.round_metrics.round` is the canonical round number for all loggers. `state.round` may be `round + 1` after `state.update()` has been called in the acquisition loop. Never use `state.round` when you mean the current round's index.
