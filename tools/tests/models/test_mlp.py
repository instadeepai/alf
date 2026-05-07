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
import torch.nn as nn
from alf_core import Candidate, LabelledCandidates
from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics
from alf_tools.models.mlp import MLP, MLPModel, MLPModelConfig, MLPTrainConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tabular_candidates():
    """Return 8 random tabular Candidates with feature dimension 4."""
    rng = np.random.RandomState(0)
    return [Candidate(data=rng.randn(4).astype(np.float32), modality="tabular") for _ in range(8)]


@pytest.fixture
def embedding_candidates():
    """Return 8 random embedding Candidates with feature dimension 4."""
    rng = np.random.RandomState(1)
    return [Candidate(data=rng.randn(4).astype(np.float32), modality="embedding") for _ in range(8)]


@pytest.fixture
def labelled_tabular(tabular_candidates):
    """Return LabelledCandidates wrapping tabular_candidates with random labels."""
    rng = np.random.RandomState(2)
    return LabelledCandidates(tabular_candidates, rng.randn(8))


@pytest.fixture
def val_labelled_tabular():
    """Return 3 labelled validation candidates with varied features and labels."""
    rng = np.random.RandomState(3)
    candidates = [
        Candidate(data=rng.randn(4).astype(np.float32), modality="tabular") for _ in range(3)
    ]
    return LabelledCandidates(candidates, np.array([1.0, 2.0, 3.0]))


@pytest.fixture
def mlp_model():
    """Return a small 2-epoch MLPModel with hidden_dims=[16,8] for fast tests."""
    return MLPModel(
        model_config=MLPModelConfig(hidden_dims=[16, 8], model_seed=0),
        train_config=MLPTrainConfig(batch_size=4, num_epochs=2),
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Task 1: Config tests
# ---------------------------------------------------------------------------


class TestMLPModelConfig:
    """Tests for MLPModelConfig defaults and validation."""

    def test_defaults(self):
        """Default MLPModelConfig must have expected field values."""
        cfg = MLPModelConfig()
        assert cfg.hidden_dims == [256, 128]
        assert cfg.activation == "relu"
        assert cfg.norm == "none"
        assert cfg.dropout == 0.0
        assert cfg.n_mc_passes == 0
        assert cfg.model_seed == 0
        assert cfg.dropout_seed is None

    def test_mc_dropout_requires_dropout_gt_zero(self):
        """n_mc_passes>0 with dropout=0 must raise ValueError."""
        with pytest.raises(ValueError, match="dropout must be > 0"):
            MLPModelConfig(n_mc_passes=10, dropout=0.0)

    def test_mc_dropout_with_dropout_valid(self):
        """n_mc_passes>0 with dropout>0 must not raise."""
        cfg = MLPModelConfig(n_mc_passes=10, dropout=0.2)
        assert cfg.n_mc_passes == 10
        assert cfg.dropout == 0.2

    def test_dropout_seed_optional(self):
        """dropout_seed must default to None and accept an explicit int."""
        cfg = MLPModelConfig(dropout_seed=99)
        assert cfg.dropout_seed == 99


class TestMLPTrainConfig:
    """Tests for MLPTrainConfig defaults and validation."""

    def test_defaults(self):
        """Default MLPTrainConfig must have expected field values."""
        cfg = MLPTrainConfig()
        assert cfg.learning_rate == 1e-3
        assert cfg.batch_size == 32
        assert cfg.num_epochs == 50
        assert cfg.optimizer == "adam"
        assert cfg.weight_decay == 0.0
        assert cfg.log_frequency == 10

    def test_num_epochs_zero_raises_value_error(self):
        """MLPTrainConfig(num_epochs=0) must raise ValueError."""
        with pytest.raises(ValueError, match="num_epochs must be >= 1"):
            MLPTrainConfig(num_epochs=0)

    def test_num_epochs_negative_raises_value_error(self):
        """MLPTrainConfig(num_epochs=-1) must raise ValueError."""
        with pytest.raises(ValueError, match="num_epochs must be >= 1"):
            MLPTrainConfig(num_epochs=-1)


# ---------------------------------------------------------------------------
# Task 2: MLP(nn.Module) tests
# ---------------------------------------------------------------------------


class TestMLP:
    """Tests for the MLP nn.Module: shape, activations, norm, dropout, seeding."""

    def test_forward_shape(self):
        """Multi-layer MLP must output shape (batch,) for a 2D input."""
        net = MLP(input_dim=16, hidden_dims=[64, 32], activation="relu", norm="none", dropout=0.0)
        out = net(torch.randn(8, 16))
        assert out.shape == (8,)

    def test_single_hidden_layer(self):
        """Single hidden layer must produce shape (batch,)."""
        net = MLP(input_dim=8, hidden_dims=[16], activation="relu", norm="none", dropout=0.0)
        assert net(torch.randn(4, 8)).shape == (4,)

    def test_gelu_activation(self):
        """GELU activation must not change output shape."""
        net = MLP(input_dim=8, hidden_dims=[16], activation="gelu", norm="none", dropout=0.0)
        assert net(torch.randn(4, 8)).shape == (4,)

    def test_silu_activation(self):
        """SiLU activation must not change output shape."""
        net = MLP(input_dim=8, hidden_dims=[16], activation="silu", norm="none", dropout=0.0)
        assert net(torch.randn(4, 8)).shape == (4,)

    def test_batch_norm(self):
        """BatchNorm must not change output shape when net is in train mode."""
        net = MLP(input_dim=8, hidden_dims=[16, 8], activation="relu", norm="batch", dropout=0.0)
        net.train()
        assert net(torch.randn(4, 8)).shape == (4,)

    def test_layer_norm(self):
        """LayerNorm must not change output shape."""
        net = MLP(input_dim=8, hidden_dims=[16], activation="relu", norm="layer", dropout=0.0)
        assert net(torch.randn(4, 8)).shape == (4,)

    def test_with_dropout(self):
        """Dropout must not change output shape in eval mode."""
        net = MLP(input_dim=8, hidden_dims=[16], activation="relu", norm="none", dropout=0.3)
        net.eval()
        assert net(torch.randn(4, 8)).shape == (4,)

    def test_seed_reproducibility(self):
        """Same model_seed must produce identical outputs on identical inputs."""

        def make_net():
            return MLP(
                input_dim=8,
                hidden_dims=[16],
                activation="relu",
                norm="none",
                dropout=0.0,
                model_seed=42,
            )

        x = torch.randn(4, 8)
        torch.testing.assert_close(make_net()(x), make_net()(x))

    def test_different_seeds_different_weights(self):
        """Different model_seeds must initialise different weight matrices."""
        net1 = MLP(
            input_dim=8, hidden_dims=[16], activation="relu", norm="none", dropout=0.0, model_seed=1
        )
        net2 = MLP(
            input_dim=8, hidden_dims=[16], activation="relu", norm="none", dropout=0.0, model_seed=2
        )
        p1 = list(net1.parameters())[0]
        p2 = list(net2.parameters())[0]
        assert not torch.allclose(p1, p2)

    def test_empty_hidden_dims(self):
        """hidden_dims=[] must build a direct input-to-output linear layer."""
        net = MLP(input_dim=8, hidden_dims=[], activation="relu", norm="none", dropout=0.0)
        out = net(torch.randn(4, 8))
        assert out.shape == (4,)


# ---------------------------------------------------------------------------
# Task 3: MLPModel featurise tests
# ---------------------------------------------------------------------------


class TestMLPModelFeaturise:
    """Tests for MLPModel.featurise() modality handling and output shape."""

    def test_featurise_tabular_list(self, mlp_model, tabular_candidates):
        """List of TABULAR candidates must produce a (8, 4) float32 tensor."""
        x = mlp_model.featurise(tabular_candidates)
        assert x.shape == (8, 4)
        assert x.dtype == torch.float32

    def test_featurise_embedding_list(self, mlp_model, embedding_candidates):
        """List of EMBEDDING candidates must produce a (8, 4) float32 tensor."""
        x = mlp_model.featurise(embedding_candidates)
        assert x.shape == (8, 4)
        assert x.dtype == torch.float32

    def test_featurise_labelled_candidates(self, mlp_model, labelled_tabular):
        """LabelledCandidates input must produce a (8, 4) tensor."""
        x = mlp_model.featurise(labelled_tabular)
        assert x.shape == (8, 4)

    def test_featurise_rejects_sequence_modality(self, mlp_model):
        """SEQUENCE modality must raise ValueError."""
        candidates = [Candidate(data="ACGT", modality="sequence")]
        with pytest.raises(ValueError, match="TABULAR and EMBEDDING"):
            mlp_model.featurise(candidates)

    def test_featurise_rejects_image_modality(self, mlp_model):
        """IMAGE modality must raise ValueError."""
        candidates = [Candidate(data=np.zeros((3, 4), dtype=np.float32), modality="image")]
        with pytest.raises(ValueError, match="TABULAR and EMBEDDING"):
            mlp_model.featurise(candidates)

    def test_featurise_tensor_data(self, mlp_model):
        """Candidates with torch.Tensor data must featurise correctly."""
        candidates = [Candidate(data=torch.randn(4), modality="tabular") for _ in range(3)]
        x = mlp_model.featurise(candidates)
        assert x.shape == (3, 4)
        assert x.dtype == torch.float32

    def test_get_epoch_metrics_before_train_returns_empty(self, mlp_model):
        """get_epoch_metrics() before train() must return an empty list."""
        assert mlp_model.get_epoch_metrics() == []

    def test_sample_raises_not_implemented(self, mlp_model):
        """sample() must raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            mlp_model.sample()

    def test_cleanup_callable_before_train(self, mlp_model):
        """cleanup() must be callable before train() without raising."""
        mlp_model.cleanup()

    def test_cleanup_callable_after_train(self, mlp_model, labelled_tabular):
        """cleanup() must be callable after train() without raising."""
        mlp_model.train(labelled_tabular)
        mlp_model.cleanup()


# ---------------------------------------------------------------------------
# Task 4: MLPModel.train() tests
# ---------------------------------------------------------------------------


class TestMLPModelTrain:
    """Tests for MLPModel.train(): initialisation, metrics, optimisers."""

    def test_train_initialises_net(self, mlp_model, labelled_tabular):
        """train() must set self.net from None to an MLP instance."""
        assert mlp_model.net is None
        mlp_model.train(labelled_tabular)
        assert mlp_model.net is not None

    def test_train_summary_metrics_contains_train_loss(self, mlp_model, labelled_tabular):
        """Summary metrics must contain a finite 'final_train_loss'."""
        mlp_model.train(labelled_tabular)
        metrics = mlp_model.get_training_summary_metrics()
        assert "final_train_loss" in metrics
        assert np.isfinite(metrics["final_train_loss"])

    def test_train_summary_metrics_contains_spearman_and_mse(self, mlp_model, labelled_tabular):
        """Summary metrics must contain 'final_train_spearman' and 'final_train_mse'."""
        mlp_model.train(labelled_tabular)
        metrics = mlp_model.get_training_summary_metrics()
        assert "final_train_spearman" in metrics
        assert "final_train_mse" in metrics

    def test_train_with_validation_adds_val_metrics(
        self, mlp_model, labelled_tabular, val_labelled_tabular
    ):
        """Passing val_data must add a finite 'final_val_loss' to summary metrics."""
        mlp_model.train(labelled_tabular, val_data=val_labelled_tabular)
        metrics = mlp_model.get_training_summary_metrics()
        assert "final_val_loss" in metrics
        assert np.isfinite(metrics["final_val_loss"])

    def test_epoch_metrics_length_equals_num_epochs(self, mlp_model, labelled_tabular):
        """get_epoch_metrics() must return exactly num_epochs entries."""
        mlp_model.train(labelled_tabular)
        assert len(mlp_model.get_epoch_metrics()) == mlp_model.train_config.num_epochs

    def test_epoch_metrics_reset_on_retrain(self, mlp_model, labelled_tabular):
        """Calling train() twice must reset epoch_metrics to a fresh num_epochs list."""
        mlp_model.train(labelled_tabular)
        mlp_model.train(labelled_tabular)
        assert len(mlp_model.get_epoch_metrics()) == mlp_model.train_config.num_epochs

    def test_epoch_metrics_are_surrogate_epoch_metrics(self, mlp_model, labelled_tabular):
        """Every epoch metric must be a SurrogateEpochMetrics instance."""
        mlp_model.train(labelled_tabular)
        for em in mlp_model.get_epoch_metrics():
            assert isinstance(em, SurrogateEpochMetrics)

    def test_epoch_metrics_train_loss_finite(self, mlp_model, labelled_tabular):
        """Every epoch train_loss must be a finite number."""
        mlp_model.train(labelled_tabular)
        for em in mlp_model.get_epoch_metrics():
            assert np.isfinite(em.train_loss)

    def test_epoch_metrics_additional_metrics_keys(self, mlp_model, labelled_tabular):
        """Every epoch must have 'train_spearman' and 'train_mse' in additional_metrics."""
        mlp_model.train(labelled_tabular)
        for em in mlp_model.get_epoch_metrics():
            assert "train_spearman" in em.additional_metrics
            assert "train_mse" in em.additional_metrics

    def test_epoch_metrics_val_loss_populated_with_val_data(
        self, mlp_model, labelled_tabular, val_labelled_tabular
    ):
        """With val_data, every epoch val_loss must be finite."""
        mlp_model.train(labelled_tabular, val_data=val_labelled_tabular)
        for em in mlp_model.get_epoch_metrics():
            assert em.val_loss is not None
            assert np.isfinite(em.val_loss)

    def test_epoch_metrics_val_loss_none_without_val_data(self, mlp_model, labelled_tabular):
        """Without val_data, every epoch val_loss must be None."""
        mlp_model.train(labelled_tabular)
        for em in mlp_model.get_epoch_metrics():
            assert em.val_loss is None

    def test_adamw_optimizer_trains(self, labelled_tabular):
        """AdamW optimizer must initialise the network successfully."""
        model = MLPModel(
            model_config=MLPModelConfig(hidden_dims=[8]),
            train_config=MLPTrainConfig(
                batch_size=4, num_epochs=2, optimizer="adamw", weight_decay=1e-4
            ),
            device="cpu",
        )
        model.train(labelled_tabular)
        assert model.net is not None

    def test_retrain_preserves_network_identity(self, mlp_model, labelled_tabular):
        """Calling train() twice must re-use the same MLP object (warm start)."""
        mlp_model.train(labelled_tabular)
        net_after_first = mlp_model.net
        mlp_model.train(labelled_tabular)
        assert mlp_model.net is net_after_first


# ---------------------------------------------------------------------------
# Task 5: MLPModel.predict() eval mode tests
# ---------------------------------------------------------------------------


class TestMLPModelPredictEval:
    """Tests for MLPModel.predict() in deterministic eval mode."""

    def test_predict_means_shape(self, mlp_model, tabular_candidates, labelled_tabular):
        """Means must have shape (8,) matching the number of candidates."""
        mlp_model.train(labelled_tabular)
        preds = mlp_model.predict(tabular_candidates)
        assert preds.means.shape == (8,)

    def test_predict_means_finite(self, mlp_model, tabular_candidates, labelled_tabular):
        """All predicted means must be finite."""
        mlp_model.train(labelled_tabular)
        preds = mlp_model.predict(tabular_candidates)
        assert np.all(np.isfinite(preds.means))

    def test_predict_no_variances_in_eval_mode(
        self, mlp_model, tabular_candidates, labelled_tabular
    ):
        """Eval-mode predict() must return variances=None, empirical_dist=None."""
        mlp_model.train(labelled_tabular)
        preds = mlp_model.predict(tabular_candidates)
        assert preds.variances is None
        assert preds.empirical_dist is None

    def test_predict_before_train_raises_runtime_error(self, mlp_model, tabular_candidates):
        """predict() before train() must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="not trained"):
            mlp_model.predict(tabular_candidates)


# ---------------------------------------------------------------------------
# Task 6: MLPModel.predict() MC dropout + seeding tests
# ---------------------------------------------------------------------------


class TestMLPModelPredictMCDropout:
    """Tests for MLPModel.predict() with MC dropout: shapes, statistics, seeding."""

    def test_empirical_dist_shape(self, labelled_tabular, tabular_candidates):
        """n_mc_passes=8 must produce empirical_dist of shape (8, 8)."""
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
        """Means must equal empirical_dist.mean(axis=1)."""
        model = MLPModel(
            model_config=MLPModelConfig(hidden_dims=[8], dropout=0.3, n_mc_passes=6, model_seed=0),
            train_config=MLPTrainConfig(batch_size=4, num_epochs=2),
            device="cpu",
        )
        model.train(labelled_tabular)
        preds = model.predict(tabular_candidates)
        np.testing.assert_allclose(preds.means, preds.empirical_dist.mean(axis=1), rtol=1e-5)

    def test_variances_equal_rowwise_var_of_empirical_dist(
        self, labelled_tabular, tabular_candidates
    ):
        """Variances must equal empirical_dist.var(axis=1)."""
        model = MLPModel(
            model_config=MLPModelConfig(hidden_dims=[8], dropout=0.3, n_mc_passes=6, model_seed=0),
            train_config=MLPTrainConfig(batch_size=4, num_epochs=2),
            device="cpu",
        )
        model.train(labelled_tabular)
        preds = model.predict(tabular_candidates)
        np.testing.assert_allclose(preds.variances, preds.empirical_dist.var(axis=1), rtol=1e-5)

    def test_mc_dropout_is_stochastic(self, labelled_tabular, tabular_candidates):
        """MC dropout must produce non-zero total variance across passes."""
        model = MLPModel(
            model_config=MLPModelConfig(
                hidden_dims=[16], dropout=0.5, n_mc_passes=10, model_seed=0
            ),
            train_config=MLPTrainConfig(batch_size=4, num_epochs=2),
            device="cpu",
        )
        model.train(labelled_tabular)
        preds = model.predict(tabular_candidates)
        assert preds.variances.sum() > 0, "Dropout should introduce non-zero variance"

    def test_same_dropout_seed_reproducible(self, labelled_tabular, tabular_candidates):
        """Same dropout_seed must give identical empirical_dist on repeated predict() calls."""
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

    def test_different_dropout_seeds_give_different_passes(
        self, labelled_tabular, tabular_candidates
    ):
        """Different dropout_seeds must produce different empirical_dist matrices."""

        def make_model(dropout_seed: int) -> MLPModel:
            return MLPModel(
                model_config=MLPModelConfig(
                    hidden_dims=[16],
                    dropout=0.3,
                    n_mc_passes=5,
                    model_seed=42,
                    dropout_seed=dropout_seed,
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
        m_none = MLPModel(
            model_config=cfg_none,
            train_config=MLPTrainConfig(batch_size=4, num_epochs=2),
            device="cpu",
        )
        m_explicit = MLPModel(
            model_config=cfg_explicit,
            train_config=MLPTrainConfig(batch_size=4, num_epochs=2),
            device="cpu",
        )
        m_none.train(labelled_tabular)
        m_explicit.train(labelled_tabular)
        np.testing.assert_array_equal(
            m_none.predict(tabular_candidates).empirical_dist,
            m_explicit.predict(tabular_candidates).empirical_dist,
        )

    def test_batch_norm_running_stats_not_mutated_during_predict(
        self, labelled_tabular, tabular_candidates
    ):
        """BatchNorm running stats must not change during predict() with MC dropout."""
        model = MLPModel(
            model_config=MLPModelConfig(
                hidden_dims=[16], norm="batch", dropout=0.3, n_mc_passes=5, model_seed=0
            ),
            train_config=MLPTrainConfig(batch_size=4, num_epochs=2),
            device="cpu",
        )
        model.train(labelled_tabular)

        # Capture running mean before predict
        bn_before = [
            m.running_mean.clone() for m in model.net.modules() if isinstance(m, nn.BatchNorm1d)
        ]

        model.predict(tabular_candidates)

        # Running stats must be unchanged after predict
        bn_after = [m.running_mean for m in model.net.modules() if isinstance(m, nn.BatchNorm1d)]
        for before, after in zip(bn_before, bn_after):
            torch.testing.assert_close(before, after)


class TestMLPModelSeeding:
    """Tests that model_seed fully determines weight initialisation and training."""

    def test_same_model_seed_same_eval_predictions(self, labelled_tabular, tabular_candidates):
        """Two models with the same seed must predict identical means."""

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

    def test_different_model_seeds_different_eval_predictions(
        self, labelled_tabular, tabular_candidates
    ):
        """Two models with different seeds must predict different means."""

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
