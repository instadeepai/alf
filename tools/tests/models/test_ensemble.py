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

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from alf_core import Candidate, LabelledCandidates
from alf_tools.models.cnn import CNNModel, CNNModelConfig, CNNTrainConfig
from alf_tools.models.ensemble import EnsembleWrapper, EnsembleWrapperConfig, SubsampleConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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

    Args:
        seed (int): Seed for torch and numpy RNGs to ensure reproducibility.

    Returns:
        CNNModel: A CNNModel instance with a small architecture and 2 epochs.
    """
    # TODO: remove global seeding once CNNModelConfig gains a model_seed parameter
    torch.manual_seed(seed)
    np.random.seed(seed)
    return CNNModel(
        model_config=CNNModelConfig(num_filters=16, num_conv_layers=1, fc_hidden_dim=32),
        train_config=CNNTrainConfig(batch_size=4, num_epochs=2),
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Task 1: SubsampleConfig tests
# ---------------------------------------------------------------------------


class TestSubsampleConfig:
    """Tests for SubsampleConfig validation."""

    def test_fraction_only(self):
        """Defaults: replace=False, seeds=None."""
        cfg = SubsampleConfig(fraction=0.8)
        assert cfg.fraction == 0.8
        assert cfg.replace is False
        assert cfg.seeds is None

    def test_with_replace(self):
        """replace=True is stored correctly."""
        cfg = SubsampleConfig(fraction=0.5, replace=True)
        assert cfg.replace is True

    def test_with_seeds(self):
        """Explicit seeds list is stored correctly."""
        cfg = SubsampleConfig(fraction=0.7, seeds=[1, 2, 3])
        assert cfg.seeds == [1, 2, 3]

    def test_fraction_zero_raises(self):
        """fraction=0.0 must raise ValueError."""
        with pytest.raises(ValueError, match="fraction"):
            SubsampleConfig(fraction=0.0)

    def test_fraction_negative_raises(self):
        """Negative fraction must raise ValueError."""
        with pytest.raises(ValueError, match="fraction"):
            SubsampleConfig(fraction=-0.1)

    def test_fraction_above_one_raises(self):
        """Fraction > 1.0 must raise ValueError."""
        with pytest.raises(ValueError, match="fraction"):
            SubsampleConfig(fraction=1.1)

    def test_empty_seeds_raises(self):
        """seeds=[] must raise ValueError."""
        with pytest.raises(ValueError, match="seeds"):
            SubsampleConfig(fraction=0.5, seeds=[])


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

    def test_member_seeds_with_invalid_n_members_does_not_raise(self):
        """member_seeds takes precedence: n_members < 1 must not raise when member_seeds is set."""
        cfg = EnsembleWrapperConfig(member_seeds=[1, 2], n_members=0)
        assert cfg.resolve_seeds() == [1, 2]

    def test_subsample_seeds_length_mismatch_raises(self):
        """subsample.seeds length != member count must raise ValueError."""
        with pytest.raises(ValueError, match="subsample.seeds"):
            EnsembleWrapperConfig(
                member_seeds=[1, 2, 3],
                subsample=SubsampleConfig(fraction=0.8, seeds=[10, 20]),  # 2 seeds, 3 members
            )


# ---------------------------------------------------------------------------
# Task 8: EnsembleWrapper construction, featurise, train, metrics tests
# ---------------------------------------------------------------------------


class TestEnsembleWrapperConstruction:
    """Tests for EnsembleWrapper instantiation and featurise delegation."""

    def test_base_seed_creates_correct_member_count(self):
        """base_seed mode must create exactly n_members members."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=3),
        )
        assert len(wrapper.members) == 3

    def test_member_seeds_creates_correct_member_count(self):
        """member_seeds mode must create one member per seed."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(member_seeds=[10, 20, 30]),
        )
        assert len(wrapper.members) == 3

    def test_featurise_delegates_to_first_member(self, sequence_candidates):
        """featurise() output must match what the first member returns.

        CNNModel featurise returns a float32 tensor of shape (n_cands, alphabet_size, seq_len).
        For 6 candidates with 20-char sequences over a 20-character alphabet: (6, 20, 20).
        """
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        x = wrapper.featurise(sequence_candidates, model_index=0)
        assert x.shape == (6, 20, 20)

    def test_featurise_with_labelled_candidates(self, labelled_sequences):
        """featurise() must accept LabelledCandidates, not only list[Candidate]."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        x = wrapper.featurise(labelled_sequences, model_index=0)
        assert x.shape == (6, 20, 20)

    def test_get_epoch_metrics_before_train_returns_empty(self):
        """get_epoch_metrics() before train() must return an empty list."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        assert wrapper.get_epoch_metrics() == []

    def test_get_training_summary_metrics_before_train_returns_empty(self):
        """get_training_summary_metrics() before train() must return an empty dict."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        assert wrapper.get_training_summary_metrics() == {}


class TestEnsembleWrapperTrain:
    """Tests for EnsembleWrapper training and metric aggregation."""

    def test_train_trains_all_members(self, labelled_sequences):
        """After train(), every member must have an initialised network."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=3),
        )
        wrapper.train(labelled_sequences)
        for member in wrapper.members:
            assert member.model is not None

    def test_epoch_metrics_tagged_with_member_index(self, labelled_sequences):
        """Epoch metric keys must be prefixed with 'member_0/' and 'member_1/'."""
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

    def test_summary_metrics_tagged_with_member_index(self, labelled_sequences):
        """Summary metric keys must be prefixed with 'member_0/' and 'member_1/'."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        wrapper.train(labelled_sequences)
        summary = wrapper.get_training_summary_metrics()
        assert any(k.startswith("member_0/") for k in summary)
        assert any(k.startswith("member_1/") for k in summary)

    def test_epoch_metrics_total_length(self, labelled_sequences):
        """Total epoch metrics count must equal n_members × num_epochs."""
        n_members = 3
        num_epochs = 2  # matches cnn_factory's CNNTrainConfig(num_epochs=2)
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=n_members),
        )
        wrapper.train(labelled_sequences)
        assert len(wrapper.get_epoch_metrics()) == n_members * num_epochs

    def test_sample_raises_not_implemented(self):
        """sample() must raise NotImplementedError."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        with pytest.raises(NotImplementedError):
            wrapper.sample()

    def test_predict_before_train_raises(self, sequence_candidates):
        """predict() before train() must raise RuntimeError."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        with pytest.raises(RuntimeError, match="not trained"):
            wrapper.predict(sequence_candidates)

    def test_cleanup_delegates_to_all_members(self, labelled_sequences):
        """cleanup() must call cleanup() on every member without error."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=3),
        )
        wrapper.train(labelled_sequences)
        wrapper.cleanup()  # must not raise

    def test_train_with_val_data_populates_val_metrics(
        self, labelled_sequences, sequence_candidates
    ):
        """Passing val_data to train() must result in finite val_loss in epoch metrics."""
        val_data = LabelledCandidates(
            sequence_candidates[:3],
            np.array([1.0, 2.0, 3.0]),
        )
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        wrapper.train(labelled_sequences, val_data=val_data)
        for em in wrapper.get_epoch_metrics():
            assert em.val_loss is not None
            assert np.isfinite(em.val_loss)

    def test_subsample_none_passes_full_data(self, labelled_sequences):
        """subsample=None must pass the original train_data object to every member."""
        wrapper = EnsembleWrapper(
            model_factory=lambda s: MagicMock(),
            config=EnsembleWrapperConfig(base_seed=0, n_members=2),
        )
        wrapper.train(labelled_sequences)
        for member in wrapper.members:
            received = member.train.call_args[0][0]
            assert received is labelled_sequences

    def test_subsample_sizes_match_fraction(self, labelled_sequences):
        """_subsample must return round(fraction * n) candidates."""
        # len(labelled_sequences) == 6, fraction=0.5 → k = round(0.5 * 6) = 3
        cfg = EnsembleWrapperConfig(
            member_seeds=[0],
            subsample=SubsampleConfig(fraction=0.5, seeds=[42]),
        )
        wrapper = EnsembleWrapper(model_factory=lambda s: MagicMock(), config=cfg)
        subset = wrapper._subsample(labelled_sequences, 42)
        assert len(subset) == 3

    def test_subsample_without_replacement_unique_indices(self, labelled_sequences):
        """replace=False must produce a subset with no repeated candidates."""
        cfg = EnsembleWrapperConfig(
            member_seeds=[0],
            subsample=SubsampleConfig(fraction=0.5, replace=False, seeds=[0]),
        )
        wrapper = EnsembleWrapper(model_factory=lambda s: MagicMock(), config=cfg)
        subset = wrapper._subsample(labelled_sequences, 0)
        # Labels are distinct floats from np.random.RandomState(2).randn(6),
        # so duplicate labels mean duplicate indices.
        assert len(set(subset.labels.tolist())) == len(subset)

    def test_subsample_with_replacement_correct_size(self, labelled_sequences):
        """replace=True must still return the correct number of candidates."""
        cfg = EnsembleWrapperConfig(
            member_seeds=[0],
            subsample=SubsampleConfig(fraction=1.0, replace=True, seeds=[0]),
        )
        wrapper = EnsembleWrapper(model_factory=lambda s: MagicMock(), config=cfg)
        subset = wrapper._subsample(labelled_sequences, 0)
        # fraction=1.0, n=6 → k=6
        assert len(subset) == 6

    def test_subsample_reproducible(self, labelled_sequences):
        """Same config and seed must produce identical subsets across calls."""
        cfg = EnsembleWrapperConfig(
            member_seeds=[7],
            subsample=SubsampleConfig(fraction=0.5, seeds=[99]),
        )
        w1 = EnsembleWrapper(model_factory=lambda s: MagicMock(), config=cfg)
        w2 = EnsembleWrapper(model_factory=lambda s: MagicMock(), config=cfg)
        s1 = w1._subsample(labelled_sequences, 99)
        s2 = w2._subsample(labelled_sequences, 99)
        np.testing.assert_array_equal(s1.labels, s2.labels)

    def test_subsample_members_receive_different_subsets(self, labelled_sequences):
        """Explicit subsample seeds must produce different subsets per member."""
        wrapper = EnsembleWrapper(
            model_factory=lambda s: MagicMock(),
            config=EnsembleWrapperConfig(
                member_seeds=[1, 2],
                subsample=SubsampleConfig(fraction=0.5, seeds=[10, 20]),
            ),
        )
        wrapper.train(labelled_sequences)
        subset_0 = wrapper.members[0].train.call_args[0][0]
        subset_1 = wrapper.members[1].train.call_args[0][0]
        assert not np.array_equal(subset_0.labels, subset_1.labels)

    def test_subsample_diversity_from_member_seeds(self, labelled_sequences):
        """Distinct member seeds with no subsample.seeds must produce different subsets.

        This is the core diversity property: different model-init seeds drive
        different data sampling when no explicit subsample seeds are provided.
        """
        wrapper = EnsembleWrapper(
            model_factory=lambda s: MagicMock(),
            config=EnsembleWrapperConfig(
                member_seeds=[10, 20, 30],
                subsample=SubsampleConfig(fraction=0.5),
            ),
        )
        wrapper.train(labelled_sequences)
        subsets = [member.train.call_args[0][0] for member in wrapper.members]
        labels = [s.labels for s in subsets]
        # At least one pair of members must have received different subsets
        assert not all(np.array_equal(labels[0], lb) for lb in labels[1:])

    def test_val_data_passed_through_unmodified(self, labelled_sequences, sequence_candidates):
        """val_data must always be the original object, regardless of subsample config."""
        val_data = LabelledCandidates(
            sequence_candidates[:3],
            np.array([1.0, 2.0, 3.0]),
        )
        wrapper = EnsembleWrapper(
            model_factory=lambda s: MagicMock(),
            config=EnsembleWrapperConfig(
                member_seeds=[1, 2],
                subsample=SubsampleConfig(fraction=0.5),
            ),
        )
        wrapper.train(labelled_sequences, val_data=val_data)
        for member in wrapper.members:
            received_val = member.train.call_args[0][1]
            assert received_val is val_data

    def test_explicit_subsample_seeds_override_member_seeds(self, labelled_sequences):
        """When subsample.seeds is set, member seeds must not affect subsampling.

        Two wrappers with different member seeds but identical subsample.seeds
        must produce identical subsets.
        """

        def make_wrapper(member_seed: int) -> EnsembleWrapper:
            return EnsembleWrapper(
                model_factory=lambda s: MagicMock(),
                config=EnsembleWrapperConfig(
                    member_seeds=[member_seed],
                    subsample=SubsampleConfig(fraction=0.5, seeds=[42]),
                ),
            )

        w1 = make_wrapper(100)
        w2 = make_wrapper(999)
        w1.train(labelled_sequences)
        w2.train(labelled_sequences)
        subset_1 = w1.members[0].train.call_args[0][0]
        subset_2 = w2.members[0].train.call_args[0][0]
        np.testing.assert_array_equal(subset_1.labels, subset_2.labels)


# ---------------------------------------------------------------------------
# Task 9: EnsembleWrapper.predict() all modes
# ---------------------------------------------------------------------------


class TestEnsembleWrapperPredict:
    """Tests for EnsembleWrapper prediction in deep-ensemble mode."""

    def test_deep_ensemble_empirical_dist_shape(self, sequence_candidates, labelled_sequences):
        """4-member deep ensemble must produce empirical_dist of shape (6, 4)."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=4),
        )
        wrapper.train(labelled_sequences)
        preds = wrapper.predict(sequence_candidates)
        assert preds.empirical_dist.shape == (6, 4)
        assert preds.means.shape == (6,)
        assert preds.variances.shape == (6,)

    def test_deep_ensemble_means_are_rowwise_mean(self, sequence_candidates, labelled_sequences):
        """Means must equal empirical_dist.mean(axis=1) for deep ensemble."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=3),
        )
        wrapper.train(labelled_sequences)
        preds = wrapper.predict(sequence_candidates)
        np.testing.assert_allclose(preds.means, preds.empirical_dist.mean(axis=1), rtol=1e-5)

    def test_deep_ensemble_variances_are_rowwise_var(self, sequence_candidates, labelled_sequences):
        """Variances must equal empirical_dist.var(axis=1) for deep ensemble."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=0, n_members=3),
        )
        wrapper.train(labelled_sequences)
        preds = wrapper.predict(sequence_candidates)
        np.testing.assert_allclose(preds.variances, preds.empirical_dist.var(axis=1), rtol=1e-5)

    def test_different_member_seeds_give_different_columns(
        self, sequence_candidates, labelled_sequences
    ):
        """Different member seeds must produce different prediction columns."""
        wrapper = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(member_seeds=[1, 2]),
        )
        wrapper.train(labelled_sequences)
        preds = wrapper.predict(sequence_candidates)
        assert not np.allclose(preds.empirical_dist[:, 0], preds.empirical_dist[:, 1]), (
            "Different member seeds should produce different predictions"
        )

    def test_base_seed_and_member_seeds_same_result(self, sequence_candidates, labelled_sequences):
        """base_seed=10,n_members=2 and member_seeds=[10,11] must give identical results."""
        w1 = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(base_seed=10, n_members=2),
        )
        w2 = EnsembleWrapper(
            model_factory=cnn_factory,
            config=EnsembleWrapperConfig(member_seeds=[10, 11]),
        )
        # TODO: remove global seeding once CNNModelConfig gains a model_seed parameter
        seed = 10
        torch.manual_seed(seed)
        np.random.seed(seed)
        w1.train(labelled_sequences)
        torch.manual_seed(seed)
        np.random.seed(seed)
        w2.train(labelled_sequences)
        # p1 = w1.predict(sequence_candidates)
        # p2 = w2.predict(sequence_candidates)

        assert (
            w1.members[0].model.fc_layers[0].weight.data
            == w2.members[0].model.fc_layers[0].weight.data
        ).all()
        assert (
            w1.members[1].model.fc_layers[0].weight.data
            == w2.members[1].model.fc_layers[0].weight.data
        ).all()


# ---------------------------------------------------------------------------
# EnsembleWrapper with CNNModel
# ---------------------------------------------------------------------------


class TestEnsembleWrapperWithCNN:
    """Confirm EnsembleWrapper works end-to-end with CNNModel (SEQUENCE modality).

    CNNModel characteristics relevant to the ensemble wrapper:
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
        x = wrapper.featurise(sequence_candidates, model_index=0)
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
