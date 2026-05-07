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

from typing import Callable

import numpy as np
import pytest
import torch
from alf_core import Candidate, LabelledCandidates
from alf_tools.models.cnn import CNNModel, CNNModelConfig, CNNTrainConfig
from alf_tools.models.ensemble import EnsembleWrapper, EnsembleWrapperConfig
from alf_tools.models.mlp import MLPModel, MLPModelConfig, MLPTrainConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tabular_candidates():
    """Return 6 random tabular Candidates with feature dimension 4."""
    rng = np.random.RandomState(0)
    return [Candidate(data=rng.randn(4).astype(np.float32), modality="tabular") for _ in range(6)]


@pytest.fixture
def labelled_tabular(tabular_candidates):
    """Return LabelledCandidates wrapping tabular_candidates with random labels."""
    rng = np.random.RandomState(1)
    return LabelledCandidates(tabular_candidates, rng.randn(6))


def mlp_factory(seed: int) -> MLPModel:
    """Return a small 2-epoch MLPModel seeded with the given seed."""
    return MLPModel(
        model_config=MLPModelConfig(hidden_dims=[8], model_seed=seed),
        train_config=MLPTrainConfig(batch_size=4, num_epochs=2),
        device="cpu",
    )


def mc_mlp_factory(n_passes: int, dropout: float = 0.3) -> Callable[[int], MLPModel]:
    """Return a factory that creates MC-dropout MLPModels with n_passes passes."""

    def factory(seed: int) -> MLPModel:
        return MLPModel(
            model_config=MLPModelConfig(
                hidden_dims=[8], dropout=dropout, n_mc_passes=n_passes, model_seed=seed
            ),
            train_config=MLPTrainConfig(batch_size=4, num_epochs=2),
            device="cpu",
        )

    return factory


@pytest.fixture
def sequence_candidates():
    """Return 6 distinct SEQUENCE Candidates, each a 20-character protein sequence.

    Sequences must be distinct so that per-candidate predictions vary across
    members in seed-diversity assertions.
    """
    seqs = [
        "ACDEFGHIKLMNPQRSTVWY",
        "WYTVRSQPNMLKIHGFEDCA",
        "ACKMNPQRSTVWYDEFGHIL",
        "MNPQRSTVWYACDEFGHIKL",
        "RSTVWYACDEFGHIKLMNPQ",
        "GHIKLMNPQRSTVWYACDEF",
    ]
    return [Candidate(data=s, modality="sequence") for s in seqs]


@pytest.fixture
def labelled_sequences(sequence_candidates):
    """Return LabelledCandidates wrapping sequence_candidates with fixed random labels."""
    rng = np.random.RandomState(2)
    return LabelledCandidates(sequence_candidates, rng.randn(6))


def cnn_factory(seed: int) -> CNNModel:
    """Seed global RNG then return a small, fast CNNModel.

    CNNModelConfig has no model_seed field, so seeding must be done via
    global torch/numpy state before construction — the same approach used in
    TestCNNModelReproducibility.test_reproducibility_with_seed. Different
    integer seeds produce different initial weights.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    return CNNModel(
        model_config=CNNModelConfig(num_filters=16, num_conv_layers=1, fc_hidden_dim=32),
        train_config=CNNTrainConfig(batch_size=4, num_epochs=2),
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Task 7: EnsembleWrapperConfig tests
# ---------------------------------------------------------------------------


class TestEnsembleWrapperConfig:
    """Tests for EnsembleWrapperConfig validation and seed resolution."""

    def test_base_seed_derives_sequential_seeds(self):
        """base_seed=10 with n_members=3 must resolve to [10, 11, 12]."""
        cfg = EnsembleWrapperConfig(base_seed=10, n_members=3)
        assert cfg.resolve_seeds() == [10, 11, 12]

    def test_member_seeds_used_directly(self):
        """member_seeds=[5,10,15] must be returned unchanged by resolve_seeds()."""
        cfg = EnsembleWrapperConfig(member_seeds=[5, 10, 15])
        assert cfg.resolve_seeds() == [5, 10, 15]

    def test_member_seeds_determines_count(self):
        """len(member_seeds) must equal the number of resolved seeds."""
        cfg = EnsembleWrapperConfig(member_seeds=[1, 2, 3, 4])
        assert len(cfg.resolve_seeds()) == 4

    def test_neither_base_seed_nor_member_seeds_raises(self):
        """Omitting both base_seed and member_seeds must raise ValueError."""
        with pytest.raises(ValueError, match="Exactly one"):
            EnsembleWrapperConfig()

    def test_both_base_seed_and_member_seeds_raises(self):
        """Providing both base_seed and member_seeds must raise ValueError."""
        with pytest.raises(ValueError, match="Exactly one"):
            EnsembleWrapperConfig(base_seed=0, member_seeds=[1, 2])

    def test_base_seed_without_n_members_raises(self):
        """base_seed without n_members must raise ValueError."""
        with pytest.raises(ValueError, match="n_members must be set"):
            EnsembleWrapperConfig(base_seed=5)

    def test_member_seeds_n_members_ignored(self):
        """n_members is ignored when member_seeds is provided."""
        cfg = EnsembleWrapperConfig(member_seeds=[10, 20], n_members=99)
        assert cfg.resolve_seeds() == [10, 20]

    def test_n_members_zero_raises(self):
        """n_members=0 with base_seed must raise ValueError."""
        with pytest.raises(ValueError, match="n_members must be >= 1"):
            EnsembleWrapperConfig(base_seed=0, n_members=0)


# ---------------------------------------------------------------------------
# Task 8: EnsembleWrapper construction, featurise, train, metrics tests
# ---------------------------------------------------------------------------


class TestEnsembleWrapperConstruction:
    """Tests for EnsembleWrapper instantiation and featurise delegation."""

    def test_base_seed_creates_correct_member_count(self):
        """base_seed mode must create exactly n_members members."""
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=3),
        )
        assert len(wrapper.members) == 3

    def test_member_seeds_creates_correct_member_count(self):
        """member_seeds mode must create one member per seed."""
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(member_seeds=[10, 20, 30]),
        )
        assert len(wrapper.members) == 3

    def test_featurise_delegates_to_first_member(self, tabular_candidates):
        """featurise() output must match what the first member returns."""
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        x = wrapper.featurise(tabular_candidates)
        assert x.shape == (6, 4)

    def test_featurise_with_labelled_candidates(self, labelled_tabular):
        """featurise() must accept LabelledCandidates, not only list[Candidate]."""
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        x = wrapper.featurise(labelled_tabular)
        assert x.shape == (6, 4)

    def test_get_epoch_metrics_before_train_returns_empty(self):
        """get_epoch_metrics() before train() must return an empty list."""
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        assert wrapper.get_epoch_metrics() == []

    def test_get_training_summary_metrics_before_train_returns_empty(self):
        """get_training_summary_metrics() before train() must return an empty dict."""
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        assert wrapper.get_training_summary_metrics() == {}


class TestEnsembleWrapperTrain:
    """Tests for EnsembleWrapper training and metric aggregation."""

    def test_train_trains_all_members(self, labelled_tabular):
        """After train(), every member must have an initialised network."""
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=3),
        )
        wrapper.train(labelled_tabular)
        for member in wrapper.members:
            assert member.net is not None

    def test_epoch_metrics_tagged_with_member_index(self, labelled_tabular):
        """Epoch metric keys must be prefixed with 'member_0/' and 'member_1/'."""
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
        """Summary metric keys must be prefixed with 'member_0/' and 'member_1/'."""
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        wrapper.train(labelled_tabular)
        summary = wrapper.get_training_summary_metrics()
        assert any(k.startswith("member_0/") for k in summary)
        assert any(k.startswith("member_1/") for k in summary)

    def test_epoch_metrics_total_length(self, labelled_tabular):
        """Total epoch metrics count must equal n_members × num_epochs."""
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
        """sample() must raise NotImplementedError."""
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        with pytest.raises(NotImplementedError):
            wrapper.sample()

    def test_predict_before_train_raises(self, tabular_candidates):
        """predict() before train() must raise RuntimeError."""
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        with pytest.raises(RuntimeError, match="not trained"):
            wrapper.predict(tabular_candidates)

    def test_cleanup_delegates_to_all_members(self, labelled_tabular):
        """cleanup() must call cleanup() on every member without error."""
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=3),
        )
        wrapper.train(labelled_tabular)
        wrapper.cleanup()  # must not raise

    def test_train_with_val_data_populates_val_metrics(self, labelled_tabular, tabular_candidates):
        """Passing val_data to train() must result in finite val_loss in epoch metrics."""
        val_data = LabelledCandidates(
            tabular_candidates[:3],
            np.array([1.0, 2.0, 3.0]),
        )
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        wrapper.train(labelled_tabular, val_data=val_data)
        for em in wrapper.get_epoch_metrics():
            assert em.val_loss is not None
            assert np.isfinite(em.val_loss)


# ---------------------------------------------------------------------------
# Task 9: EnsembleWrapper.predict() all modes
# ---------------------------------------------------------------------------


class TestEnsembleWrapperPredict:
    """Tests for EnsembleWrapper prediction across deep-ensemble, MC, and combined modes."""

    def test_deep_ensemble_empirical_dist_shape(self, tabular_candidates, labelled_tabular):
        """4-member deep ensemble must produce empirical_dist of shape (6, 4)."""
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
        """Means must equal empirical_dist.mean(axis=1) for deep ensemble."""
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=3),
        )
        wrapper.train(labelled_tabular)
        preds = wrapper.predict(tabular_candidates)
        np.testing.assert_allclose(preds.means, preds.empirical_dist.mean(axis=1), rtol=1e-5)

    def test_deep_ensemble_variances_are_rowwise_var(self, tabular_candidates, labelled_tabular):
        """Variances must equal empirical_dist.var(axis=1) for deep ensemble."""
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=3),
        )
        wrapper.train(labelled_tabular)
        preds = wrapper.predict(tabular_candidates)
        np.testing.assert_allclose(preds.variances, preds.empirical_dist.var(axis=1), rtol=1e-5)

    def test_mc_dropout_single_member_shape(self, tabular_candidates, labelled_tabular):
        """1 member, n_mc_passes=5 must produce empirical_dist of shape (6, 5)."""
        wrapper = EnsembleWrapper(
            model_factory=mc_mlp_factory(n_passes=5),
            config=EnsembleWrapperConfig(member_seeds=[42]),
        )
        wrapper.train(labelled_tabular)
        preds = wrapper.predict(tabular_candidates)
        assert preds.empirical_dist.shape == (6, 5)

    def test_combined_mode_shape(self, tabular_candidates, labelled_tabular):
        """N members × T passes must produce empirical_dist of shape (6, N*T)."""
        n_members = 3
        n_passes = 4
        wrapper = EnsembleWrapper(
            model_factory=mc_mlp_factory(n_passes=n_passes),
            config=EnsembleWrapperConfig(base_seed=0, n_members=n_members),
        )
        wrapper.train(labelled_tabular)
        preds = wrapper.predict(tabular_candidates)
        assert preds.empirical_dist.shape == (6, n_members * n_passes)

    def test_different_member_seeds_give_different_columns(
        self, tabular_candidates, labelled_tabular
    ):
        """Different member seeds must produce different prediction columns."""
        wrapper = EnsembleWrapper(
            model_factory=mlp_factory,
            config=EnsembleWrapperConfig(member_seeds=[1, 2]),
        )
        wrapper.train(labelled_tabular)
        preds = wrapper.predict(tabular_candidates)
        assert not np.allclose(preds.empirical_dist[:, 0], preds.empirical_dist[:, 1]), (
            "Different member seeds should produce different predictions"
        )

    def test_base_seed_and_member_seeds_same_result(self, tabular_candidates, labelled_tabular):
        """base_seed=10,n_members=2 and member_seeds=[10,11] must give identical results."""
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


# ---------------------------------------------------------------------------
# EnsembleWrapper with CNNModel
# ---------------------------------------------------------------------------


class TestEnsembleWrapperWithCNN:
    """Confirm EnsembleWrapper works end-to-end with CNNModel (SEQUENCE modality).

    CNNModel differs from MLPModel in two ways relevant here:
      - Accepts SEQUENCE candidates (one-hot encoded internally), not TABULAR/EMBEDDING.
      - Has no model_seed config field; seeding is done via torch.manual_seed in cnn_factory.
    """

    def test_cnn_member_count_with_base_seed(self):
        """base_seed mode must create exactly n_members CNNModel instances."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=3),
        )
        assert len(wrapper.members) == 3
        assert all(isinstance(m, CNNModel) for m in wrapper.members)

    def test_cnn_member_count_with_member_seeds(self):
        """member_seeds mode must create one CNNModel per seed."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(member_seeds=[10, 20]),
        )
        assert len(wrapper.members) == 2
        assert all(isinstance(m, CNNModel) for m in wrapper.members)

    def test_cnn_featurise_returns_tensor(self, sequence_candidates):
        """featurise() must return a float32 tensor of shape (n_candidates, alphabet_size, seq_len).

        The 20-character protein sequence produces shape (6, 20, 20):
        6 candidates, 20 alphabet characters, 20-position sequence.
        """
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        x = wrapper.featurise(sequence_candidates)
        assert isinstance(x, torch.Tensor)
        assert x.shape == (6, 20, 20)
        assert x.dtype == torch.float32

    def test_cnn_predict_before_train_raises(self, sequence_candidates):
        """predict() before train() must raise RuntimeError."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        with pytest.raises(RuntimeError, match="Model not trained"):
            wrapper.predict(sequence_candidates)

    def test_cnn_ensemble_trains_all_members(self, labelled_sequences):
        """After train(), every CNNModel member must have an initialised .model attribute."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=3),
        )
        wrapper.train(labelled_sequences)
        for member in wrapper.members:
            assert member.model is not None

    def test_cnn_ensemble_epoch_metric_tagging(self, labelled_sequences):
        """After train(), epoch metric keys must be prefixed 'member_0/' and 'member_1/'."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        wrapper.train(labelled_sequences)
        all_keys: set[str] = set()
        for em in wrapper.get_epoch_metrics():
            all_keys.update(em.additional_metrics.keys())
        assert any(k.startswith("member_0/") for k in all_keys)
        assert any(k.startswith("member_1/") for k in all_keys)

    def test_cnn_ensemble_cleanup_delegates(self, labelled_sequences):
        """cleanup() must call cleanup() on every CNN member without raising."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        wrapper.train(labelled_sequences)
        wrapper.cleanup()  # must not raise

    def test_cnn_ensemble_empirical_dist_shape(self, sequence_candidates, labelled_sequences):
        """3-member CNN ensemble must produce empirical_dist of shape (6, 3).

        CNNModel.predict() returns means only (no empirical_dist), so
        EnsembleWrapper stacks each member's means as a single column.
        """
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=3),
        )
        wrapper.train(labelled_sequences)
        preds = wrapper.predict(sequence_candidates)
        assert preds.empirical_dist.shape == (6, 3)
        assert preds.means.shape == (6,)
        assert preds.variances.shape == (6,)

    def test_cnn_ensemble_means_are_rowwise_mean(self, sequence_candidates, labelled_sequences):
        """Means must equal empirical_dist.mean(axis=1) for a CNN deep ensemble."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=3),
        )
        wrapper.train(labelled_sequences)
        preds = wrapper.predict(sequence_candidates)
        np.testing.assert_allclose(preds.means, preds.empirical_dist.mean(axis=1), rtol=1e-5)

    def test_cnn_ensemble_variances_are_rowwise_var(self, sequence_candidates, labelled_sequences):
        """Variances must equal empirical_dist.var(axis=1) for a CNN deep ensemble."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=3),
        )
        wrapper.train(labelled_sequences)
        preds = wrapper.predict(sequence_candidates)
        np.testing.assert_allclose(preds.variances, preds.empirical_dist.var(axis=1), rtol=1e-5)

    def test_cnn_different_seeds_give_different_columns(
        self, sequence_candidates, labelled_sequences
    ):
        """Different member seeds must produce different per-member prediction columns.

        cnn_factory(seed) calls torch.manual_seed(seed) before construction, so
        different seeds yield different weight initialisations and thus different
        predictions after identical training data.
        """
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(member_seeds=[1, 2]),
        )
        wrapper.train(labelled_sequences)
        preds = wrapper.predict(sequence_candidates)
        assert not np.allclose(preds.empirical_dist[:, 0], preds.empirical_dist[:, 1]), (
            "Different CNN member seeds must produce different predictions"
        )


# ---------------------------------------------------------------------------
# Explicit MC dropout and combined mode semantics
# ---------------------------------------------------------------------------


class TestEnsembleWrapperMCDropoutExplicit:
    """Explicit stochasticity and aggregation semantics for MC dropout and combined modes.

    These tests are MLP-only: CNNModel has no n_mc_passes parameter, so
    MC dropout and combined modes are not applicable to CNN ensembles.
    """

    def test_mc_dropout_passes_are_stochastic(self, tabular_candidates, labelled_tabular):
        """MC dropout columns within a single ensemble member must differ from each other.

        With n_mc_passes=5 and dropout=0.3, the 5 stochastic forward passes must
        not produce identical outputs. Zero variance across all 5 columns would
        indicate dropout is inactive during inference.
        """
        wrapper = EnsembleWrapper(
            model_factory=mc_mlp_factory(n_passes=5),
            config=EnsembleWrapperConfig(member_seeds=[42]),
        )
        wrapper.train(labelled_tabular)
        preds = wrapper.predict(tabular_candidates)
        col0 = preds.empirical_dist[:, 0]
        col1 = preds.empirical_dist[:, 1]
        assert not np.allclose(col0, col1), (
            "MC dropout passes must be stochastic; identical columns indicate dropout is inactive"
        )

    def test_combined_mode_means_aggregate_all_nt_samples(
        self, tabular_candidates, labelled_tabular
    ):
        """In combined mode, means must equal the grand mean across all N*T columns.

        3 members × 4 MC passes = 12 columns; means must be the mean of all 12,
        not the mean of the 3 per-member averages (which would differ when T > 1).
        """
        n_members = 3
        n_passes = 4
        wrapper = EnsembleWrapper(
            model_factory=mc_mlp_factory(n_passes=n_passes),
            config=EnsembleWrapperConfig(base_seed=0, n_members=n_members),
        )
        wrapper.train(labelled_tabular)
        preds = wrapper.predict(tabular_candidates)
        assert preds.empirical_dist.shape == (6, 12)
        np.testing.assert_allclose(preds.means, preds.empirical_dist.mean(axis=1), rtol=1e-5)

    def test_combined_mode_different_seeds_give_diverse_member_distributions(
        self, tabular_candidates, labelled_tabular
    ):
        """In combined mode, MC pass blocks from different member seeds must differ.

        With member_seeds=[1, 2] and n_mc_passes=4:
          columns 0–3  → MC passes from member seeded with 1
          columns 4–7  → MC passes from member seeded with 2
        The per-member mean predictions must differ, confirming that member seeds
        produce genuinely different models even when MC dropout is active.
        """
        wrapper = EnsembleWrapper(
            model_factory=mc_mlp_factory(n_passes=4),
            config=EnsembleWrapperConfig(member_seeds=[1, 2]),
        )
        wrapper.train(labelled_tabular)
        preds = wrapper.predict(tabular_candidates)
        member0_mean = preds.empirical_dist[:, :4].mean(axis=1)
        member1_mean = preds.empirical_dist[:, 4:].mean(axis=1)
        assert not np.allclose(member0_mean, member1_mean), (
            "Members with different seeds must produce different prediction distributions"
        )
