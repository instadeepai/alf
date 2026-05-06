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
