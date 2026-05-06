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
