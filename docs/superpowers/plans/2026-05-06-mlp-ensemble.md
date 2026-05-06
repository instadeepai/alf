# MLP Model & Generic Ensemble Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a flexible MLP surrogate and a generic ensemble wrapper to `alf_tools`, providing configurable MC dropout and deep ensemble uncertainty quantification compatible with all existing acquisition functions.

**Architecture:** `MLPModel(BaseModel)` wraps `MLP(nn.Module)` — lazy init on first `train()` call, featurises pre-computed TABULAR/EMBEDDING vectors as a passthrough. `EnsembleWrapper(BaseModel)` wraps N members via a factory callable, assembling `Predictions.empirical_dist` from per-member outputs for use by Thompson Sampling, UCB, and EI without modification.

**Tech Stack:** Python 3.12, PyTorch ≥ 2.10, NumPy, Pydantic dataclasses, pytest. No new dependencies.

---

## File Map

| Action | Path |
|---|---|
| Create | `tools/alf_tools/models/mlp.py` |
| Create | `tools/alf_tools/models/ensemble.py` |
| Create | `tools/tests/models/test_mlp.py` |
| Create | `tools/tests/models/test_ensemble.py` |
| Modify | `tools/alf_tools/models/__init__.py` |

**Run all tests from the repo root:**
```bash
cd /Users/o.gallup/Code/alf && .venv/bin/pytest tools/tests/models/test_mlp.py tools/tests/models/test_ensemble.py -v
```

---

## Task 1: Config Dataclasses — `MLPModelConfig` and `MLPTrainConfig`

**Files:**
- Create: `tools/alf_tools/models/mlp.py`
- Create: `tools/tests/models/test_mlp.py`

- [ ] **Step 1.1: Write the failing config tests**

Create `tools/tests/models/test_mlp.py`:

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

import numpy as np
import pytest
import torch
from alf_core import Candidate, LabelledCandidates
from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics

from alf_tools.models.mlp import MLP, MLPModel, MLPModelConfig, MLPTrainConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tabular_candidates():
    rng = np.random.RandomState(0)
    return [
        Candidate(data=rng.randn(4).astype(np.float32), modality="tabular")
        for _ in range(8)
    ]


@pytest.fixture
def embedding_candidates():
    rng = np.random.RandomState(1)
    return [
        Candidate(data=rng.randn(4).astype(np.float32), modality="embedding")
        for _ in range(8)
    ]


@pytest.fixture
def labelled_tabular(tabular_candidates):
    rng = np.random.RandomState(2)
    return LabelledCandidates(tabular_candidates, rng.randn(8))


@pytest.fixture
def mlp_model():
    return MLPModel(
        model_config=MLPModelConfig(hidden_dims=[16, 8], model_seed=0),
        train_config=MLPTrainConfig(batch_size=4, num_epochs=2),
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Task 1: Config tests
# ---------------------------------------------------------------------------


class TestMLPModelConfig:
    def test_defaults(self):
        cfg = MLPModelConfig()
        assert cfg.hidden_dims == [256, 128]
        assert cfg.activation == "relu"
        assert cfg.norm == "none"
        assert cfg.dropout == 0.0
        assert cfg.n_mc_passes == 0
        assert cfg.model_seed == 0
        assert cfg.dropout_seed is None

    def test_mc_dropout_requires_dropout_gt_zero(self):
        with pytest.raises(ValueError, match="dropout must be > 0"):
            MLPModelConfig(n_mc_passes=10, dropout=0.0)

    def test_mc_dropout_with_dropout_valid(self):
        cfg = MLPModelConfig(n_mc_passes=10, dropout=0.2)
        assert cfg.n_mc_passes == 10
        assert cfg.dropout == 0.2

    def test_dropout_seed_optional(self):
        cfg = MLPModelConfig(dropout_seed=99)
        assert cfg.dropout_seed == 99


class TestMLPTrainConfig:
    def test_defaults(self):
        cfg = MLPTrainConfig()
        assert cfg.learning_rate == 1e-3
        assert cfg.batch_size == 32
        assert cfg.num_epochs == 50
        assert cfg.optimizer == "adam"
        assert cfg.weight_decay == 0.0
```

- [ ] **Step 1.2: Run tests to confirm they fail (import error)**

```bash
cd /Users/o.gallup/Code/alf && .venv/bin/pytest tools/tests/models/test_mlp.py::TestMLPModelConfig tools/tests/models/test_mlp.py::TestMLPTrainConfig -v
```

Expected: `ModuleNotFoundError: No module named 'alf_tools.models.mlp'`

- [ ] **Step 1.3: Create `mlp.py` with config dataclasses**

Create `tools/alf_tools/models/mlp.py`:

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

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from alf_core import BaseModel, Candidate, LabelledCandidates, Modality, Predictions, Results
from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics
from torch.utils.data import DataLoader, TensorDataset

from alf_tools.models.utils import get_device

logger = logging.getLogger("alf-tools")


@dataclass
class MLPModelConfig:
    """Configuration for MLP model architecture.

    Args:
        hidden_dims: Sizes of hidden layers; length determines depth.
        activation: Activation function applied after each hidden layer's norm.
        norm: Normalisation applied before activation; "none" skips it.
        dropout: Dropout probability applied after activation in each hidden layer.
        n_mc_passes: Number of stochastic forward passes at inference for MC dropout.
            0 disables MC dropout and returns means only.
        model_seed: Global seed for weight initialisation and training data shuffling.
            Also used as the dropout generator seed when dropout_seed is None.
        dropout_seed: If set, overrides model_seed exclusively for the MC dropout
            pass generator.
    """

    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    activation: Literal["relu", "gelu", "silu"] = "relu"
    norm: Literal["none", "batch", "layer"] = "none"
    dropout: float = 0.0
    n_mc_passes: int = 0
    model_seed: int = 0
    dropout_seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_mc_passes > 0 and self.dropout <= 0.0:
            raise ValueError(
                f"dropout must be > 0 when n_mc_passes > 0, got dropout={self.dropout}"
            )


@dataclass
class MLPTrainConfig:
    """Configuration for MLP training.

    Args:
        learning_rate: Learning rate for the optimiser.
        batch_size: Mini-batch size.
        num_epochs: Number of training epochs.
        optimizer: Optimiser type; "adam" or "adamw".
        weight_decay: L2 regularisation coefficient.
    """

    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 50
    optimizer: Literal["adam", "adamw"] = "adam"
    weight_decay: float = 0.0
```

- [ ] **Step 1.4: Run tests to confirm they pass**

```bash
cd /Users/o.gallup/Code/alf && .venv/bin/pytest tools/tests/models/test_mlp.py::TestMLPModelConfig tools/tests/models/test_mlp.py::TestMLPTrainConfig -v
```

Expected: `5 passed`

- [ ] **Step 1.5: Commit**

```bash
cd /Users/o.gallup/Code/alf && git add tools/alf_tools/models/mlp.py tools/tests/models/test_mlp.py && git commit -m "feat: add MLPModelConfig and MLPTrainConfig dataclasses"
```

---

## Task 2: `MLP(nn.Module)` — Pure PyTorch Module

**Files:**
- Modify: `tools/alf_tools/models/mlp.py` (append after configs)
- Modify: `tools/tests/models/test_mlp.py` (append `TestMLP` class)

- [ ] **Step 2.1: Write failing architecture tests**

Append to `tools/tests/models/test_mlp.py`:

```python
# ---------------------------------------------------------------------------
# Task 2: MLP(nn.Module) tests
# ---------------------------------------------------------------------------


class TestMLP:
    def test_forward_shape(self):
        net = MLP(input_dim=16, hidden_dims=[64, 32], activation="relu", norm="none", dropout=0.0)
        out = net(torch.randn(8, 16))
        assert out.shape == (8,)

    def test_single_hidden_layer(self):
        net = MLP(input_dim=8, hidden_dims=[16], activation="relu", norm="none", dropout=0.0)
        assert net(torch.randn(4, 8)).shape == (4,)

    def test_gelu_activation(self):
        net = MLP(input_dim=8, hidden_dims=[16], activation="gelu", norm="none", dropout=0.0)
        assert net(torch.randn(4, 8)).shape == (4,)

    def test_silu_activation(self):
        net = MLP(input_dim=8, hidden_dims=[16], activation="silu", norm="none", dropout=0.0)
        assert net(torch.randn(4, 8)).shape == (4,)

    def test_batch_norm(self):
        net = MLP(input_dim=8, hidden_dims=[16, 8], activation="relu", norm="batch", dropout=0.0)
        net.train()
        assert net(torch.randn(4, 8)).shape == (4,)

    def test_layer_norm(self):
        net = MLP(input_dim=8, hidden_dims=[16], activation="relu", norm="layer", dropout=0.0)
        assert net(torch.randn(4, 8)).shape == (4,)

    def test_with_dropout(self):
        net = MLP(input_dim=8, hidden_dims=[16], activation="relu", norm="none", dropout=0.3)
        net.eval()
        assert net(torch.randn(4, 8)).shape == (4,)

    def test_seed_reproducibility(self):
        def make_net():
            return MLP(
                input_dim=8, hidden_dims=[16], activation="relu",
                norm="none", dropout=0.0, model_seed=42,
            )

        x = torch.randn(4, 8)
        torch.testing.assert_close(make_net()(x), make_net()(x))

    def test_different_seeds_different_weights(self):
        net1 = MLP(input_dim=8, hidden_dims=[16], activation="relu", norm="none", dropout=0.0, model_seed=1)
        net2 = MLP(input_dim=8, hidden_dims=[16], activation="relu", norm="none", dropout=0.0, model_seed=2)
        p1 = list(net1.parameters())[0]
        p2 = list(net2.parameters())[0]
        assert not torch.allclose(p1, p2)
```

- [ ] **Step 2.2: Run tests to confirm they fail**

```bash
cd /Users/o.gallup/Code/alf && .venv/bin/pytest tools/tests/models/test_mlp.py::TestMLP -v
```

Expected: `ImportError` — `MLP` is not yet defined.

- [ ] **Step 2.3: Implement `MLP(nn.Module)`**

Append to `tools/alf_tools/models/mlp.py` (after the `MLPTrainConfig` dataclass):

```python
class MLP(nn.Module):
    """Feedforward MLP for scalar regression on pre-computed feature vectors.

    Architecture:
        input → [Linear → Norm → Activation → Dropout] × depth → Linear → scalar
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation: Literal["relu", "gelu", "silu"],
        norm: Literal["none", "batch", "layer"],
        dropout: float,
        model_seed: int = 0,
    ):
        super().__init__()
        torch.manual_seed(model_seed)

        _activation_map: dict[str, type[nn.Module]] = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        activation_cls = _activation_map[activation]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if norm == "batch":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif norm == "layer":
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.hidden_block = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.hidden_block(x)).squeeze(-1)
```

- [ ] **Step 2.4: Run tests to confirm they pass**

```bash
cd /Users/o.gallup/Code/alf && .venv/bin/pytest tools/tests/models/test_mlp.py::TestMLP -v
```

Expected: `9 passed`

- [ ] **Step 2.5: Commit**

```bash
cd /Users/o.gallup/Code/alf && git add tools/alf_tools/models/mlp.py tools/tests/models/test_mlp.py && git commit -m "feat: add MLP nn.Module with configurable activation, norm, dropout, and seeding"
```

---

## Task 3: `MLPModel` Shell and `featurise()`

**Files:**
- Modify: `tools/alf_tools/models/mlp.py` (append `MLPModel` class)
- Modify: `tools/tests/models/test_mlp.py` (append `TestMLPModelFeaturise`)

- [ ] **Step 3.1: Write failing featurise tests**

Append to `tools/tests/models/test_mlp.py`:

```python
# ---------------------------------------------------------------------------
# Task 3: MLPModel featurise tests
# ---------------------------------------------------------------------------


class TestMLPModelFeaturise:
    def test_featurise_tabular_list(self, mlp_model, tabular_candidates):
        x = mlp_model.featurise(tabular_candidates)
        assert x.shape == (8, 4)
        assert x.dtype == torch.float32

    def test_featurise_embedding_list(self, mlp_model, embedding_candidates):
        x = mlp_model.featurise(embedding_candidates)
        assert x.shape == (8, 4)
        assert x.dtype == torch.float32

    def test_featurise_labelled_candidates(self, mlp_model, labelled_tabular):
        x = mlp_model.featurise(labelled_tabular)
        assert x.shape == (8, 4)

    def test_featurise_rejects_sequence_modality(self, mlp_model):
        candidates = [Candidate(data="ACGT", modality="sequence")]
        with pytest.raises(ValueError, match="TABULAR and EMBEDDING"):
            mlp_model.featurise(candidates)

    def test_featurise_rejects_image_modality(self, mlp_model):
        candidates = [Candidate(data=np.zeros((3, 4), dtype=np.float32), modality="image")]
        with pytest.raises(ValueError, match="TABULAR and EMBEDDING"):
            mlp_model.featurise(candidates)

    def test_featurise_tensor_data(self, mlp_model):
        candidates = [
            Candidate(data=torch.randn(4), modality="tabular") for _ in range(3)
        ]
        x = mlp_model.featurise(candidates)
        assert x.shape == (3, 4)
        assert x.dtype == torch.float32

    def test_get_epoch_metrics_before_train_returns_empty(self, mlp_model):
        assert mlp_model.get_epoch_metrics() == []

    def test_sample_raises_not_implemented(self, mlp_model):
        with pytest.raises(NotImplementedError):
            mlp_model.sample()
```

- [ ] **Step 3.2: Run tests to confirm they fail**

```bash
cd /Users/o.gallup/Code/alf && .venv/bin/pytest tools/tests/models/test_mlp.py::TestMLPModelFeaturise -v
```

Expected: `ImportError` — `MLPModel` not yet defined.

- [ ] **Step 3.3: Implement `MLPModel` shell with `featurise()`**

Append to `tools/alf_tools/models/mlp.py` (after the `MLP` class):

```python
class MLPModel(BaseModel):
    """Surrogate model wrapping MLP for pre-computed vector inputs.

    Featurisation is a passthrough — inputs must arrive as TABULAR or EMBEDDING
    candidates whose data is a numpy array or torch tensor.
    """

    def __init__(
        self,
        name: str = "mlp_model",
        model_config: MLPModelConfig | None = None,
        train_config: MLPTrainConfig | None = None,
        device: str | None = None,
    ):
        self.name = name
        self.model_config = model_config or MLPModelConfig()
        self.train_config = train_config or MLPTrainConfig()
        self.device = get_device(device)
        self.net: MLP | None = None
        self.training_metrics: dict[str, Union[float, int, np.number]] = {}
        self._epoch_metrics: list[SurrogateEpochMetrics] = []

    def featurise(self, inputs: Union[LabelledCandidates, list[Candidate]]) -> torch.Tensor:
        if isinstance(inputs, LabelledCandidates):
            candidates = inputs.candidates
        elif isinstance(inputs, list):
            candidates = inputs
        else:
            raise ValueError("Input must be LabelledCandidates or list of Candidate")

        for c in candidates:
            if c.modality not in (Modality.TABULAR, Modality.EMBEDDING):
                raise ValueError(
                    f"MLPModel only supports TABULAR and EMBEDDING modalities, got {c.modality}"
                )

        arrays = []
        for c in candidates:
            if isinstance(c.data, torch.Tensor):
                arrays.append(c.data.float().cpu().numpy())
            else:
                arrays.append(np.asarray(c.data, dtype=np.float32))

        return torch.tensor(np.stack(arrays), dtype=torch.float32)

    def sample(self, condition: Any | None = None) -> list[Candidate]:
        raise NotImplementedError("Sampling is not implemented for MLPModel.")

    def get_epoch_metrics(self) -> list[SurrogateEpochMetrics]:
        return self._epoch_metrics

    def get_training_summary_metrics(self) -> dict[str, Union[float, int, np.number]]:
        return self.training_metrics

    def train(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates | None = None,
    ) -> None:
        raise NotImplementedError("train() not yet implemented")

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        raise NotImplementedError("predict() not yet implemented")
```

- [ ] **Step 3.4: Run tests to confirm they pass**

```bash
cd /Users/o.gallup/Code/alf && .venv/bin/pytest tools/tests/models/test_mlp.py::TestMLPModelFeaturise -v
```

Expected: `8 passed`

- [ ] **Step 3.5: Commit**

```bash
cd /Users/o.gallup/Code/alf && git add tools/alf_tools/models/mlp.py tools/tests/models/test_mlp.py && git commit -m "feat: add MLPModel shell with featurise() for TABULAR and EMBEDDING inputs"
```

---

## Task 4: `MLPModel.train()`

**Files:**
- Modify: `tools/alf_tools/models/mlp.py` (replace stub `train()`)
- Modify: `tools/tests/models/test_mlp.py` (append `TestMLPModelTrain`)

- [ ] **Step 4.1: Write failing train tests**

Append to `tools/tests/models/test_mlp.py`:

```python
# ---------------------------------------------------------------------------
# Task 4: MLPModel.train() tests
# ---------------------------------------------------------------------------


class TestMLPModelTrain:
    def test_train_initialises_net(self, mlp_model, labelled_tabular):
        assert mlp_model.net is None
        mlp_model.train(labelled_tabular)
        assert mlp_model.net is not None

    def test_train_summary_metrics_contains_train_loss(self, mlp_model, labelled_tabular):
        mlp_model.train(labelled_tabular)
        metrics = mlp_model.get_training_summary_metrics()
        assert "final_train_loss" in metrics
        assert np.isfinite(metrics["final_train_loss"])

    def test_train_with_validation_adds_val_metrics(self, mlp_model, labelled_tabular):
        val_candidates = [
            Candidate(data=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32), modality="tabular")
            for _ in range(3)
        ]
        val_data = LabelledCandidates(val_candidates, np.array([1.0, 2.0, 3.0]))
        mlp_model.train(labelled_tabular, val_data=val_data)
        metrics = mlp_model.get_training_summary_metrics()
        assert "final_val_loss" in metrics
        assert np.isfinite(metrics["final_val_loss"])

    def test_epoch_metrics_length_equals_num_epochs(self, mlp_model, labelled_tabular):
        mlp_model.train(labelled_tabular)
        assert len(mlp_model.get_epoch_metrics()) == mlp_model.train_config.num_epochs

    def test_epoch_metrics_reset_on_retrain(self, mlp_model, labelled_tabular):
        mlp_model.train(labelled_tabular)
        mlp_model.train(labelled_tabular)
        assert len(mlp_model.get_epoch_metrics()) == mlp_model.train_config.num_epochs

    def test_epoch_metrics_are_surrogate_epoch_metrics(self, mlp_model, labelled_tabular):
        mlp_model.train(labelled_tabular)
        for em in mlp_model.get_epoch_metrics():
            assert isinstance(em, SurrogateEpochMetrics)

    def test_epoch_metrics_train_loss_finite(self, mlp_model, labelled_tabular):
        mlp_model.train(labelled_tabular)
        for em in mlp_model.get_epoch_metrics():
            assert np.isfinite(em.train_loss)

    def test_epoch_metrics_val_loss_populated_with_val_data(self, mlp_model, labelled_tabular):
        val_candidates = [
            Candidate(data=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32), modality="tabular")
            for _ in range(3)
        ]
        val_data = LabelledCandidates(val_candidates, np.array([1.0, 2.0, 3.0]))
        mlp_model.train(labelled_tabular, val_data=val_data)
        for em in mlp_model.get_epoch_metrics():
            assert em.val_loss is not None
            assert np.isfinite(em.val_loss)

    def test_epoch_metrics_val_loss_none_without_val_data(self, mlp_model, labelled_tabular):
        mlp_model.train(labelled_tabular)
        for em in mlp_model.get_epoch_metrics():
            assert em.val_loss is None

    def test_adamw_optimizer_trains(self, labelled_tabular):
        model = MLPModel(
            model_config=MLPModelConfig(hidden_dims=[8]),
            train_config=MLPTrainConfig(batch_size=4, num_epochs=2, optimizer="adamw", weight_decay=1e-4),
            device="cpu",
        )
        model.train(labelled_tabular)
        assert model.net is not None
```

- [ ] **Step 4.2: Run tests to confirm they fail**

```bash
cd /Users/o.gallup/Code/alf && .venv/bin/pytest tools/tests/models/test_mlp.py::TestMLPModelTrain -v
```

Expected: `NotImplementedError` on all tests.

- [ ] **Step 4.3: Implement `MLPModel.train()`**

Replace the stub `train()` method in `MLPModel` with:

```python
    def train(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates | None = None,
    ) -> None:
        self._epoch_metrics = []
        np.random.seed(self.model_config.model_seed)
        torch.manual_seed(self.model_config.model_seed)

        if self.net is None:
            input_dim = len(train_data.data[0])
            self.net = MLP(
                input_dim=input_dim,
                hidden_dims=self.model_config.hidden_dims,
                activation=self.model_config.activation,
                norm=self.model_config.norm,
                dropout=self.model_config.dropout,
                model_seed=self.model_config.model_seed,
            ).to(self.device)
            logger.info(
                f"MLPModel '{self.name}' initialised with input_dim={input_dim}, "
                f"hidden_dims={self.model_config.hidden_dims}, "
                f"params={sum(p.numel() for p in self.net.parameters()):,}"
            )

        x_train = self.featurise(train_data).to(self.device)
        y_train = torch.tensor(train_data.labels, dtype=torch.float32).to(self.device)
        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=self.train_config.batch_size,
            shuffle=True,
        )

        val_loader: DataLoader | None = None
        if val_data is not None and len(val_data) > 0:
            x_val = self.featurise(val_data).to(self.device)
            y_val = torch.tensor(val_data.labels, dtype=torch.float32).to(self.device)
            val_loader = DataLoader(
                TensorDataset(x_val, y_val),
                batch_size=self.train_config.batch_size,
                shuffle=False,
            )

        if self.train_config.optimizer == "adamw":
            optimizer: optim.Optimizer = optim.AdamW(
                self.net.parameters(),
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                self.net.parameters(),
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay,
            )

        criterion = nn.MSELoss()

        avg_train_loss = 0.0
        train_metrics: dict[str, float] = {}
        avg_val_loss: float | None = None
        val_metrics: dict[str, float] = {}

        for epoch in range(self.train_config.num_epochs):
            self.net.train()
            train_losses: list[float] = []
            train_preds_list: list[np.ndarray] = []
            train_targets_list: list[np.ndarray] = []

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                preds = self.net(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                train_preds_list.append(preds.detach().cpu().numpy())
                train_targets_list.append(batch_y.detach().cpu().numpy())

            avg_train_loss = float(np.mean(train_losses))
            train_preds = np.concatenate(train_preds_list)
            train_targets = np.concatenate(train_targets_list)

            if len(train_preds) >= 2:
                train_metrics = Results(
                    predictions=Predictions(means=train_preds), targets=train_targets
                ).metrics
            else:
                train_metrics = {"mse": float(np.mean((train_preds - train_targets) ** 2))}

            if val_loader is not None:
                self.net.eval()
                val_losses: list[float] = []
                val_preds_list: list[np.ndarray] = []
                val_targets_list: list[np.ndarray] = []

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        preds = self.net(batch_x)
                        loss = criterion(preds, batch_y)
                        val_losses.append(loss.item())
                        val_preds_list.append(preds.cpu().numpy())
                        val_targets_list.append(batch_y.cpu().numpy())

                avg_val_loss = float(np.mean(val_losses))
                val_preds = np.concatenate(val_preds_list)
                val_targets = np.concatenate(val_targets_list)

                if len(val_preds) >= 2:
                    val_metrics = Results(
                        predictions=Predictions(means=val_preds), targets=val_targets
                    ).metrics
                else:
                    val_metrics = {"mse": float(np.mean((val_preds - val_targets) ** 2))}

            additional: dict[str, float] = {}
            if (v := train_metrics.get("spearman")) is not None:
                additional["train_spearman"] = float(v)
            if (v := train_metrics.get("mse")) is not None:
                additional["train_mse"] = float(v)
            if val_loader is not None:
                if (v := val_metrics.get("spearman")) is not None:
                    additional["val_spearman"] = float(v)
                if (v := val_metrics.get("mse")) is not None:
                    additional["val_mse"] = float(v)

            self._epoch_metrics.append(
                SurrogateEpochMetrics(
                    epoch=epoch,
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss,
                    additional_metrics=additional,
                )
            )

        self.training_metrics = {"final_train_loss": avg_train_loss}
        self.training_metrics.update({f"final_train_{k}": v for k, v in train_metrics.items()})
        if val_loader is not None:
            self.training_metrics["final_val_loss"] = avg_val_loss  # type: ignore[assignment]
            self.training_metrics.update({f"final_val_{k}": v for k, v in val_metrics.items()})

        self.net.eval()
```

- [ ] **Step 4.4: Run tests to confirm they pass**

```bash
cd /Users/o.gallup/Code/alf && .venv/bin/pytest tools/tests/models/test_mlp.py::TestMLPModelTrain -v
```

Expected: `10 passed`

- [ ] **Step 4.5: Commit**

```bash
cd /Users/o.gallup/Code/alf && git add tools/alf_tools/models/mlp.py tools/tests/models/test_mlp.py && git commit -m "feat: implement MLPModel.train() with epoch metrics and Adam/AdamW support"
```

---

## Task 5: `MLPModel.predict()` — Eval Mode

**Files:**
- Modify: `tools/alf_tools/models/mlp.py` (replace stub `predict()`)
- Modify: `tools/tests/models/test_mlp.py` (append `TestMLPModelPredict`)

- [ ] **Step 5.1: Write failing eval predict tests**

Append to `tools/tests/models/test_mlp.py`:

```python
# ---------------------------------------------------------------------------
# Task 5: MLPModel.predict() eval mode tests
# ---------------------------------------------------------------------------


class TestMLPModelPredictEval:
    def test_predict_means_shape(self, mlp_model, tabular_candidates, labelled_tabular):
        mlp_model.train(labelled_tabular)
        preds = mlp_model.predict(tabular_candidates)
        assert preds.means.shape == (8,)

    def test_predict_means_finite(self, mlp_model, tabular_candidates, labelled_tabular):
        mlp_model.train(labelled_tabular)
        preds = mlp_model.predict(tabular_candidates)
        assert np.all(np.isfinite(preds.means))

    def test_predict_no_variances_in_eval_mode(self, mlp_model, tabular_candidates, labelled_tabular):
        mlp_model.train(labelled_tabular)
        preds = mlp_model.predict(tabular_candidates)
        assert preds.variances is None
        assert preds.empirical_dist is None

    def test_predict_before_train_raises_runtime_error(self, mlp_model, tabular_candidates):
        with pytest.raises(RuntimeError, match="not trained"):
            mlp_model.predict(tabular_candidates)
```

- [ ] **Step 5.2: Run tests to confirm they fail**

```bash
cd /Users/o.gallup/Code/alf && .venv/bin/pytest tools/tests/models/test_mlp.py::TestMLPModelPredictEval -v
```

Expected: `NotImplementedError`

- [ ] **Step 5.3: Implement `MLPModel.predict()` (eval path only, MC dropout stub)**

Replace the stub `predict()` in `MLPModel` with the full implementation:

```python
    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        if self.net is None:
            raise RuntimeError("Model not trained. Call train() first.")

        x = self.featurise(candidate_points).to(self.device)

        if self.model_config.n_mc_passes == 0:
            self.net.eval()
            with torch.no_grad():
                preds = self.net(x).cpu().numpy()
            return Predictions(means=preds)

        # MC dropout: keep model in train() mode so dropout is active
        self.net.train()
        seed = (
            self.model_config.dropout_seed
            if self.model_config.dropout_seed is not None
            else self.model_config.model_seed
        )
        torch.manual_seed(seed)
        passes: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(self.model_config.n_mc_passes):
                passes.append(self.net(x).cpu().numpy())

        empirical_dist = np.stack(passes, axis=1)  # (N, T)
        means = empirical_dist.mean(axis=1)
        variances = empirical_dist.var(axis=1)
        return Predictions(means=means, variances=variances, empirical_dist=empirical_dist)
```

- [ ] **Step 5.4: Run tests to confirm they pass**

```bash
cd /Users/o.gallup/Code/alf && .venv/bin/pytest tools/tests/models/test_mlp.py::TestMLPModelPredictEval -v
```

Expected: `4 passed`

- [ ] **Step 5.5: Commit**

```bash
cd /Users/o.gallup/Code/alf && git add tools/alf_tools/models/mlp.py tools/tests/models/test_mlp.py && git commit -m "feat: implement MLPModel.predict() eval mode"
```

---

## Task 6: `MLPModel.predict()` — MC Dropout Mode and Seeding

**Files:**
- Modify: `tools/tests/models/test_mlp.py` (append MC dropout and seeding test classes)

No new implementation needed — `predict()` already handles both paths from Task 5.

- [ ] **Step 6.1: Write MC dropout and seeding tests**

Append to `tools/tests/models/test_mlp.py`:

```python
# ---------------------------------------------------------------------------
# Task 6: MLPModel.predict() MC dropout + seeding tests
# ---------------------------------------------------------------------------


class TestMLPModelPredictMCDropout:
    def test_empirical_dist_shape(self, labelled_tabular, tabular_candidates):
        model = MLPModel(
            model_config=MLPModelConfig(hidden_dims=[16], dropout=0.2, n_mc_passes=8, model_seed=0),
            train_config=MLPTrainConfig(batch_size=4, num_epochs=2),
            device="cpu",
        )
        model.train(labelled_tabular)
        preds = model.predict(tabular_candidates)
        assert preds.empirical_dist.shape == (8, 8)
        assert preds.means.shape == (8,)
        assert preds.variances.shape == (8,)

    def test_means_equal_rowwise_mean_of_empirical_dist(self, labelled_tabular, tabular_candidates):
        model = MLPModel(
            model_config=MLPModelConfig(hidden_dims=[8], dropout=0.3, n_mc_passes=6, model_seed=0),
            train_config=MLPTrainConfig(batch_size=4, num_epochs=2),
            device="cpu",
        )
        model.train(labelled_tabular)
        preds = model.predict(tabular_candidates)
        np.testing.assert_allclose(preds.means, preds.empirical_dist.mean(axis=1), rtol=1e-5)

    def test_variances_equal_rowwise_var_of_empirical_dist(self, labelled_tabular, tabular_candidates):
        model = MLPModel(
            model_config=MLPModelConfig(hidden_dims=[8], dropout=0.3, n_mc_passes=6, model_seed=0),
            train_config=MLPTrainConfig(batch_size=4, num_epochs=2),
            device="cpu",
        )
        model.train(labelled_tabular)
        preds = model.predict(tabular_candidates)
        np.testing.assert_allclose(preds.variances, preds.empirical_dist.var(axis=1), rtol=1e-5)

    def test_mc_dropout_is_stochastic(self, labelled_tabular, tabular_candidates):
        model = MLPModel(
            model_config=MLPModelConfig(hidden_dims=[16], dropout=0.5, n_mc_passes=10, model_seed=0),
            train_config=MLPTrainConfig(batch_size=4, num_epochs=2),
            device="cpu",
        )
        model.train(labelled_tabular)
        preds = model.predict(tabular_candidates)
        assert preds.variances.sum() > 0, "Dropout should introduce non-zero variance"

    def test_same_dropout_seed_reproducible(self, labelled_tabular, tabular_candidates):
        model = MLPModel(
            model_config=MLPModelConfig(
                hidden_dims=[8], dropout=0.3, n_mc_passes=5, model_seed=0, dropout_seed=99
            ),
            train_config=MLPTrainConfig(batch_size=4, num_epochs=2),
            device="cpu",
        )
        model.train(labelled_tabular)
        preds1 = model.predict(tabular_candidates)
        preds2 = model.predict(tabular_candidates)
        np.testing.assert_array_equal(preds1.empirical_dist, preds2.empirical_dist)

    def test_different_dropout_seeds_give_different_passes(self, labelled_tabular, tabular_candidates):
        def make_model(dropout_seed: int) -> MLPModel:
            return MLPModel(
                model_config=MLPModelConfig(
                    hidden_dims=[16], dropout=0.3, n_mc_passes=5,
                    model_seed=42, dropout_seed=dropout_seed,
                ),
                train_config=MLPTrainConfig(batch_size=4, num_epochs=2),
                device="cpu",
            )

        m_a = make_model(10)
        m_b = make_model(20)
        m_a.train(labelled_tabular)
        m_b.train(labelled_tabular)
        p_a = m_a.predict(tabular_candidates)
        p_b = m_b.predict(tabular_candidates)
        assert not np.allclose(p_a.empirical_dist, p_b.empirical_dist)

    def test_dropout_seed_none_falls_back_to_model_seed(self, labelled_tabular, tabular_candidates):
        """dropout_seed=None must give same passes as dropout_seed=model_seed."""
        cfg_none = MLPModelConfig(
            hidden_dims=[8], dropout=0.3, n_mc_passes=5, model_seed=7, dropout_seed=None
        )
        cfg_explicit = MLPModelConfig(
            hidden_dims=[8], dropout=0.3, n_mc_passes=5, model_seed=7, dropout_seed=7
        )
        m_none = MLPModel(model_config=cfg_none, train_config=MLPTrainConfig(batch_size=4, num_epochs=2), device="cpu")
        m_explicit = MLPModel(model_config=cfg_explicit, train_config=MLPTrainConfig(batch_size=4, num_epochs=2), device="cpu")
        m_none.train(labelled_tabular)
        m_explicit.train(labelled_tabular)
        np.testing.assert_array_equal(
            m_none.predict(tabular_candidates).empirical_dist,
            m_explicit.predict(tabular_candidates).empirical_dist,
        )


class TestMLPModelSeeding:
    def test_same_model_seed_same_eval_predictions(self, labelled_tabular, tabular_candidates):
        def make_and_train() -> MLPModel:
            m = MLPModel(
                model_config=MLPModelConfig(hidden_dims=[16], model_seed=42),
                train_config=MLPTrainConfig(batch_size=4, num_epochs=3),
                device="cpu",
            )
            m.train(labelled_tabular)
            return m

        p1 = make_and_train().predict(tabular_candidates)
        p2 = make_and_train().predict(tabular_candidates)
        np.testing.assert_array_equal(p1.means, p2.means)

    def test_different_model_seeds_different_eval_predictions(self, labelled_tabular, tabular_candidates):
        def make_and_train(seed: int) -> MLPModel:
            m = MLPModel(
                model_config=MLPModelConfig(hidden_dims=[16], model_seed=seed),
                train_config=MLPTrainConfig(batch_size=4, num_epochs=3),
                device="cpu",
            )
            m.train(labelled_tabular)
            return m

        p1 = make_and_train(1).predict(tabular_candidates)
        p2 = make_and_train(2).predict(tabular_candidates)
        assert not np.allclose(p1.means, p2.means)
```

- [ ] **Step 6.2: Run all tests**

```bash
cd /Users/o.gallup/Code/alf && .venv/bin/pytest tools/tests/models/test_mlp.py -v
```

Expected: All tests pass (no new implementation required — `predict()` is already complete).

- [ ] **Step 6.3: Commit**

```bash
cd /Users/o.gallup/Code/alf && git add tools/tests/models/test_mlp.py && git commit -m "test: add MC dropout and seeding reproducibility tests for MLPModel"
```

---

## Task 7: `EnsembleWrapperConfig`

**Files:**
- Create: `tools/alf_tools/models/ensemble.py`
- Create: `tools/tests/models/test_ensemble.py`

- [ ] **Step 7.1: Write failing config tests**

Create `tools/tests/models/test_ensemble.py`:

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

import numpy as np
import pytest
from alf_core import Candidate, LabelledCandidates

from alf_tools.models.ensemble import EnsembleWrapper, EnsembleWrapperConfig
from alf_tools.models.mlp import MLPModel, MLPModelConfig, MLPTrainConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tabular_candidates():
    rng = np.random.RandomState(0)
    return [
        Candidate(data=rng.randn(4).astype(np.float32), modality="tabular")
        for _ in range(6)
    ]


@pytest.fixture
def labelled_tabular(tabular_candidates):
    rng = np.random.RandomState(1)
    return LabelledCandidates(tabular_candidates, rng.randn(6))


def mlp_factory(seed: int) -> MLPModel:
    return MLPModel(
        model_config=MLPModelConfig(hidden_dims=[8], model_seed=seed),
        train_config=MLPTrainConfig(batch_size=4, num_epochs=2),
        device="cpu",
    )


def mc_mlp_factory(n_passes: int, dropout: float = 0.3) -> callable:
    def factory(seed: int) -> MLPModel:
        return MLPModel(
            model_config=MLPModelConfig(
                hidden_dims=[8], dropout=dropout, n_mc_passes=n_passes, model_seed=seed
            ),
            train_config=MLPTrainConfig(batch_size=4, num_epochs=2),
            device="cpu",
        )
    return factory


# ---------------------------------------------------------------------------
# Task 7: EnsembleWrapperConfig tests
# ---------------------------------------------------------------------------


class TestEnsembleWrapperConfig:
    def test_base_seed_derives_sequential_seeds(self):
        cfg = EnsembleWrapperConfig(base_seed=10, n_members=3)
        assert cfg.resolve_seeds() == [10, 11, 12]

    def test_member_seeds_used_directly(self):
        cfg = EnsembleWrapperConfig(member_seeds=[5, 10, 15])
        assert cfg.resolve_seeds() == [5, 10, 15]

    def test_member_seeds_determines_count(self):
        cfg = EnsembleWrapperConfig(member_seeds=[1, 2, 3, 4])
        assert len(cfg.resolve_seeds()) == 4

    def test_neither_base_seed_nor_member_seeds_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            EnsembleWrapperConfig()

    def test_both_base_seed_and_member_seeds_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            EnsembleWrapperConfig(base_seed=0, member_seeds=[1, 2])

    def test_base_seed_without_n_members_raises(self):
        with pytest.raises(ValueError, match="n_members must be set"):
            EnsembleWrapperConfig(base_seed=5)

    def test_member_seeds_n_members_ignored(self):
        cfg = EnsembleWrapperConfig(member_seeds=[10, 20], n_members=99)
        assert cfg.resolve_seeds() == [10, 20]
```

- [ ] **Step 7.2: Run tests to confirm they fail**

```bash
cd /Users/o.gallup/Code/alf && .venv/bin/pytest tools/tests/models/test_ensemble.py::TestEnsembleWrapperConfig -v
```

Expected: `ModuleNotFoundError: No module named 'alf_tools.models.ensemble'`

- [ ] **Step 7.3: Create `ensemble.py` with `EnsembleWrapperConfig`**

Create `tools/alf_tools/models/ensemble.py`:

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

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Union

import numpy as np
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions
from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics

logger = logging.getLogger("alf-tools")


@dataclass
class EnsembleWrapperConfig:
    """Configuration for the generic ensemble wrapper.

    Exactly one of `base_seed` or `member_seeds` must be provided.

    Args:
        base_seed: Base integer from which member seeds are derived as
            [base_seed, base_seed+1, ..., base_seed+n_members-1].
            Requires n_members to also be set.
        member_seeds: Explicit list of seeds, one per member.
            len(member_seeds) determines the number of members.
            n_members is ignored when this is set.
        n_members: Number of ensemble members. Only used when base_seed is set.
    """

    base_seed: int | None = None
    member_seeds: list[int] | None = None
    n_members: int | None = None

    def __post_init__(self) -> None:
        if self.base_seed is None and self.member_seeds is None:
            raise ValueError(
                "Exactly one of base_seed or member_seeds must be set; got neither."
            )
        if self.base_seed is not None and self.member_seeds is not None:
            raise ValueError(
                "Exactly one of base_seed or member_seeds must be set; got both."
            )
        if self.base_seed is not None and self.n_members is None:
            raise ValueError("n_members must be set when base_seed is provided.")

    def resolve_seeds(self) -> list[int]:
        """Return the ordered list of per-member seeds."""
        if self.member_seeds is not None:
            return list(self.member_seeds)
        assert self.base_seed is not None and self.n_members is not None
        return [self.base_seed + i for i in range(self.n_members)]
```

- [ ] **Step 7.4: Run tests to confirm they pass**

```bash
cd /Users/o.gallup/Code/alf && .venv/bin/pytest tools/tests/models/test_ensemble.py::TestEnsembleWrapperConfig -v
```

Expected: `7 passed`

- [ ] **Step 7.5: Commit**

```bash
cd /Users/o.gallup/Code/alf && git add tools/alf_tools/models/ensemble.py tools/tests/models/test_ensemble.py && git commit -m "feat: add EnsembleWrapperConfig with base_seed and member_seeds modes"
```

---

## Task 8: `EnsembleWrapper` — Construction, `featurise()`, `train()`, and Metrics

**Files:**
- Modify: `tools/alf_tools/models/ensemble.py` (append `EnsembleWrapper`)
- Modify: `tools/tests/models/test_ensemble.py` (append test classes)

- [ ] **Step 8.1: Write failing construction and train tests**

Append to `tools/tests/models/test_ensemble.py`:

```python
# ---------------------------------------------------------------------------
# Task 8: EnsembleWrapper construction, featurise, train, metrics tests
# ---------------------------------------------------------------------------


class TestEnsembleWrapperConstruction:
    def test_base_seed_creates_correct_member_count(self):
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=3),
        )
        assert len(wrapper.members) == 3

    def test_member_seeds_creates_correct_member_count(self):
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(member_seeds=[10, 20, 30]),
        )
        assert len(wrapper.members) == 3

    def test_featurise_delegates_to_first_member(self, tabular_candidates):
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        x = wrapper.featurise(tabular_candidates)
        assert x.shape == (6, 4)


class TestEnsembleWrapperTrain:
    def test_train_trains_all_members(self, labelled_tabular):
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=3),
        )
        wrapper.train(labelled_tabular)
        for member in wrapper.members:
            assert member.net is not None

    def test_epoch_metrics_tagged_with_member_index(self, labelled_tabular):
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        wrapper.train(labelled_tabular)
        all_keys: set[str] = set()
        for em in wrapper.get_epoch_metrics():
            all_keys.update(em.additional_metrics.keys())
        assert any(k.startswith("member_0/") for k in all_keys)
        assert any(k.startswith("member_1/") for k in all_keys)

    def test_summary_metrics_tagged_with_member_index(self, labelled_tabular):
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        wrapper.train(labelled_tabular)
        summary = wrapper.get_training_summary_metrics()
        assert any(k.startswith("member_0/") for k in summary)
        assert any(k.startswith("member_1/") for k in summary)

    def test_epoch_metrics_total_length(self, labelled_tabular):
        n_members = 3
        num_epochs = 2
        wrapper = EnsembleWrapper(
            model_factory=lambda seed: MLPModel(
                model_config=MLPModelConfig(hidden_dims=[8], model_seed=seed),
                train_config=MLPTrainConfig(batch_size=4, num_epochs=num_epochs),
                device="cpu",
            ),
            config=EnsembleWrapperConfig(base_seed=0, n_members=n_members),
        )
        wrapper.train(labelled_tabular)
        assert len(wrapper.get_epoch_metrics()) == n_members * num_epochs

    def test_sample_raises_not_implemented(self):
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        with pytest.raises(NotImplementedError):
            wrapper.sample()
```

- [ ] **Step 8.2: Run tests to confirm they fail**

```bash
cd /Users/o.gallup/Code/alf && .venv/bin/pytest tools/tests/models/test_ensemble.py::TestEnsembleWrapperConstruction tools/tests/models/test_ensemble.py::TestEnsembleWrapperTrain -v
```

Expected: `ImportError` — `EnsembleWrapper` not yet defined.

- [ ] **Step 8.3: Implement `EnsembleWrapper`**

Append to `tools/alf_tools/models/ensemble.py` (after `EnsembleWrapperConfig`):

```python
class EnsembleWrapper(BaseModel):
    """Generic ensemble wrapper that composes N BaseModel instances.

    Assembles Predictions.empirical_dist from per-member outputs:
    - If a member returns empirical_dist (e.g. MC dropout MLPModel), all its
      columns are concatenated.
    - If a member returns only means, that column is appended as a single column.

    This gives three modes when wrapping MLPModel:
        deep ensemble  : N members, n_mc_passes=0  → empirical_dist (N_cand, N)
        MC dropout     : 1 member,  n_mc_passes=T  → empirical_dist (N_cand, T)
        combined       : N members, n_mc_passes=T  → empirical_dist (N_cand, N*T)
    """

    def __init__(
        self,
        model_factory: Callable[[int], BaseModel],
        config: EnsembleWrapperConfig,
        name: str = "ensemble_wrapper",
    ):
        self.name = name
        self.config = config
        seeds = config.resolve_seeds()
        self.members: list[BaseModel] = [model_factory(seed) for seed in seeds]

    def featurise(self, inputs: Union[LabelledCandidates, list[Candidate]]) -> Any:
        return self.members[0].featurise(inputs)

    def train(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates | None = None,
    ) -> None:
        for i, member in enumerate(self.members):
            logger.info(f"EnsembleWrapper '{self.name}': training member {i + 1}/{len(self.members)}")
            member.train(train_data, val_data)

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        columns: list[np.ndarray] = []
        for member in self.members:
            p_i = member.predict(candidate_points)
            if p_i.empirical_dist is not None:
                columns.append(p_i.empirical_dist)
            else:
                columns.append(p_i.means[:, np.newaxis])

        empirical_dist = np.concatenate(columns, axis=1)
        means = empirical_dist.mean(axis=1)
        variances = empirical_dist.var(axis=1)
        return Predictions(means=means, variances=variances, empirical_dist=empirical_dist)

    def sample(self, condition: Any | None = None) -> list[Candidate]:
        raise NotImplementedError("Sampling is not implemented for EnsembleWrapper.")

    def get_epoch_metrics(self) -> list[SurrogateEpochMetrics]:
        result: list[SurrogateEpochMetrics] = []
        for i, member in enumerate(self.members):
            for em in member.get_epoch_metrics():
                tagged = {f"member_{i}/{k}": v for k, v in em.additional_metrics.items()}
                result.append(
                    SurrogateEpochMetrics(
                        epoch=em.epoch,
                        train_loss=em.train_loss,
                        val_loss=em.val_loss,
                        additional_metrics=tagged,
                    )
                )
        return result

    def get_training_summary_metrics(self) -> dict[str, Union[float, int, np.number]]:
        result: dict[str, Union[float, int, np.number]] = {}
        for i, member in enumerate(self.members):
            for k, v in member.get_training_summary_metrics().items():
                result[f"member_{i}/{k}"] = v
        return result
```

- [ ] **Step 8.4: Run tests to confirm they pass**

```bash
cd /Users/o.gallup/Code/alf && .venv/bin/pytest tools/tests/models/test_ensemble.py::TestEnsembleWrapperConstruction tools/tests/models/test_ensemble.py::TestEnsembleWrapperTrain -v
```

Expected: `10 passed`

- [ ] **Step 8.5: Commit**

```bash
cd /Users/o.gallup/Code/alf && git add tools/alf_tools/models/ensemble.py tools/tests/models/test_ensemble.py && git commit -m "feat: implement EnsembleWrapper with generic factory, train, and tagged metrics"
```

---

## Task 9: `EnsembleWrapper.predict()` — All Uncertainty Modes

**Files:**
- Modify: `tools/tests/models/test_ensemble.py` (append `TestEnsembleWrapperPredict`)

No new implementation — `predict()` is already complete from Task 8.

- [ ] **Step 9.1: Write predict tests for all three modes**

Append to `tools/tests/models/test_ensemble.py`:

```python
# ---------------------------------------------------------------------------
# Task 9: EnsembleWrapper.predict() all modes
# ---------------------------------------------------------------------------


class TestEnsembleWrapperPredict:
    def test_deep_ensemble_empirical_dist_shape(self, tabular_candidates, labelled_tabular):
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=4),
        )
        wrapper.train(labelled_tabular)
        preds = wrapper.predict(tabular_candidates)
        assert preds.empirical_dist.shape == (6, 4)
        assert preds.means.shape == (6,)
        assert preds.variances.shape == (6,)

    def test_deep_ensemble_means_are_rowwise_mean(self, tabular_candidates, labelled_tabular):
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=3),
        )
        wrapper.train(labelled_tabular)
        preds = wrapper.predict(tabular_candidates)
        np.testing.assert_allclose(preds.means, preds.empirical_dist.mean(axis=1), rtol=1e-5)

    def test_deep_ensemble_variances_are_rowwise_var(self, tabular_candidates, labelled_tabular):
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=3),
        )
        wrapper.train(labelled_tabular)
        preds = wrapper.predict(tabular_candidates)
        np.testing.assert_allclose(preds.variances, preds.empirical_dist.var(axis=1), rtol=1e-5)

    def test_mc_dropout_single_member_shape(self, tabular_candidates, labelled_tabular):
        """1 member with n_mc_passes=T → empirical_dist (N_cand, T)."""
        wrapper = EnsembleWrapper(
            model_factory=mc_mlp_factory(n_passes=5),
            config=EnsembleWrapperConfig(member_seeds=[42]),
        )
        wrapper.train(labelled_tabular)
        preds = wrapper.predict(tabular_candidates)
        assert preds.empirical_dist.shape == (6, 5)

    def test_combined_mode_shape(self, tabular_candidates, labelled_tabular):
        """N members each with n_mc_passes=T → empirical_dist (N_cand, N*T)."""
        n_members = 3
        n_passes = 4
        wrapper = EnsembleWrapper(
            model_factory=mc_mlp_factory(n_passes=n_passes),
            config=EnsembleWrapperConfig(base_seed=0, n_members=n_members),
        )
        wrapper.train(labelled_tabular)
        preds = wrapper.predict(tabular_candidates)
        assert preds.empirical_dist.shape == (6, n_members * n_passes)

    def test_different_member_seeds_give_different_columns(self, tabular_candidates, labelled_tabular):
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(member_seeds=[1, 2]),
        )
        wrapper.train(labelled_tabular)
        preds = wrapper.predict(tabular_candidates)
        assert not np.allclose(
            preds.empirical_dist[:, 0], preds.empirical_dist[:, 1]
        ), "Different member seeds should produce different predictions"

    def test_base_seed_and_member_seeds_same_result(self, tabular_candidates, labelled_tabular):
        """base_seed=10, n_members=2 must equal member_seeds=[10, 11]."""
        w1 = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=10, n_members=2),
        )
        w2 = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(member_seeds=[10, 11]),
        )
        w1.train(labelled_tabular)
        w2.train(labelled_tabular)
        p1 = w1.predict(tabular_candidates)
        p2 = w2.predict(tabular_candidates)
        np.testing.assert_array_equal(p1.empirical_dist, p2.empirical_dist)
```

- [ ] **Step 9.2: Run all ensemble tests**

```bash
cd /Users/o.gallup/Code/alf && .venv/bin/pytest tools/tests/models/test_ensemble.py -v
```

Expected: All tests pass.

- [ ] **Step 9.3: Commit**

```bash
cd /Users/o.gallup/Code/alf && git add tools/tests/models/test_ensemble.py && git commit -m "test: add EnsembleWrapper predict tests for deep ensemble, MC dropout, and combined modes"
```

---

## Task 10: Export from `models/__init__.py` and Final Verification

**Files:**
- Modify: `tools/alf_tools/models/__init__.py`

- [ ] **Step 10.1: Add exports to `__init__.py`**

Open `tools/alf_tools/models/__init__.py`. It currently contains:

```python
from alf_tools.models.cnn import CNNModel, CNNModelConfig, CNNTrainConfig
from alf_tools.models.gp import FeaturizerConfig, GPModel, GPModelConfig, GPTrainConfig
from alf_tools.models.utils import (
    create_char_to_idx_mapping,
    extract_sequences_from_inputs,
    get_device,
    one_hot_encode,
)

__all__ = [
    "CNNModel",
    "CNNModelConfig",
    "CNNTrainConfig",
    ...
]
```

Add the new imports and `__all__` entries:

```python
from alf_tools.models.cnn import CNNModel, CNNModelConfig, CNNTrainConfig
from alf_tools.models.ensemble import EnsembleWrapper, EnsembleWrapperConfig
from alf_tools.models.gp import FeaturizerConfig, GPModel, GPModelConfig, GPTrainConfig
from alf_tools.models.mlp import MLP, MLPModel, MLPModelConfig, MLPTrainConfig
from alf_tools.models.utils import (
    create_char_to_idx_mapping,
    extract_sequences_from_inputs,
    get_device,
    one_hot_encode,
)

__all__ = [
    "CNNModel",
    "CNNModelConfig",
    "CNNTrainConfig",
    "EnsembleWrapper",
    "EnsembleWrapperConfig",
    "create_char_to_idx_mapping",
    "extract_sequences_from_inputs",
    "get_device",
    "one_hot_encode",
    "FeaturizerConfig",
    "GPModelConfig",
    "GPModel",
    "GPTrainConfig",
    "MLP",
    "MLPModel",
    "MLPModelConfig",
    "MLPTrainConfig",
]
```

- [ ] **Step 10.2: Verify imports work**

```bash
cd /Users/o.gallup/Code/alf && .venv/bin/python -c "
from alf_tools.models import (
    MLP, MLPModel, MLPModelConfig, MLPTrainConfig,
    EnsembleWrapper, EnsembleWrapperConfig,
)
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 10.3: Run full test suite for new files**

```bash
cd /Users/o.gallup/Code/alf && .venv/bin/pytest tools/tests/models/test_mlp.py tools/tests/models/test_ensemble.py -v
```

Expected: All tests pass.

- [ ] **Step 10.4: Run existing tests to check for regressions**

```bash
cd /Users/o.gallup/Code/alf && .venv/bin/pytest tools/tests/models/test_cnn.py tools/tests/models/test_gp.py -v
```

Expected: All previously passing tests still pass.

- [ ] **Step 10.5: Commit**

```bash
cd /Users/o.gallup/Code/alf && git add tools/alf_tools/models/__init__.py && git commit -m "feat: export MLPModel, MLP, MLPModelConfig, MLPTrainConfig, EnsembleWrapper, EnsembleWrapperConfig from alf_tools.models"
```
