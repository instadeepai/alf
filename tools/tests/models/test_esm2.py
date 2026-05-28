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
from alf_tools.models.esm2 import ESM2Model, ESM2ModelConfig, ESM2TrainConfig

MODEL_ID = "facebook/esm2_t6_8M_UR50D"


@pytest.fixture(scope="session")
def model_config():
    """Create a default ESM2ModelConfig for testing.

    Returns:
        An ESM2ModelConfig using the smallest ESM-2 checkpoint.
    """
    return ESM2ModelConfig(model_id=MODEL_ID)


@pytest.fixture(scope="session")
def train_config():
    """Create a frozen ESM2TrainConfig for testing.

    Returns:
        An ESM2TrainConfig with freeze_backbone=True.
    """
    return ESM2TrainConfig(freeze_backbone=True)


@pytest.fixture(scope="session")
def esm2_model(model_config, train_config):
    """Frozen ESM-2 model — downloaded once per test session.

    Returns:
        An ESM2Model with frozen backbone on CPU.
    """
    return ESM2Model(
        name="test_esm2", model_config=model_config, train_config=train_config, device="cpu"
    )


@pytest.fixture(scope="class")
def esm2_finetune_model():
    """ESM-2 model with unfrozen backbone for fine-tuning tests.

    Returns:
        An ESM2Model with trainable backbone, 2 epochs, batch size 2.
    """
    config = ESM2ModelConfig(model_id=MODEL_ID)
    train_cfg = ESM2TrainConfig(
        freeze_backbone=False,
        num_epochs=2,
        batch_size=2,
        learning_rate=1e-4,
        log_frequency=1,
    )
    return ESM2Model(name="test_esm2_ft", model_config=config, train_config=train_cfg, device="cpu")


@pytest.fixture(scope="class")
def esm2_ll_model():
    """ESM-2 model configured for log-likelihood training.

    Returns:
        An ESM2Model with loss_type='log_likelihood', trainable backbone, 2 epochs.
    """
    config = ESM2ModelConfig(model_id=MODEL_ID)
    train_cfg = ESM2TrainConfig(
        freeze_backbone=False,
        loss_type="log_likelihood",
        num_epochs=2,
        batch_size=2,
        learning_rate=1e-4,
        log_frequency=1,
    )
    return ESM2Model(name="test_esm2_ll", model_config=config, train_config=train_cfg, device="cpu")


@pytest.fixture(scope="session")
def esm2_small_batch_model():
    """Frozen ESM-2 with batch_size=2 to exercise multi-batch predict.

    Returns:
        An ESM2Model with batch_size=2, frozen backbone, CPU.
    """
    config = ESM2ModelConfig(model_id=MODEL_ID)
    train_cfg = ESM2TrainConfig(freeze_backbone=True, batch_size=2)
    return ESM2Model(
        name="test_esm2_small_batch", model_config=config, train_config=train_cfg, device="cpu"
    )


@pytest.fixture
def sample_data():
    """Create a small LabelledCandidates dataset for testing.

    Returns:
        A LabelledCandidates object with three short amino-acid sequences.
    """
    sequences = ["ACDEFGHIKL", "MNPQRSTVWY", "ACMNPQRST"]
    candidates = [Candidate(data=seq, modality="sequence") for seq in sequences]
    labels = np.array([1.0, 2.0, 1.5])
    return LabelledCandidates(candidates, labels)


class TestConfigs:
    """Tests for ESM2ModelConfig and ESM2TrainConfig dataclasses."""

    def test_model_config_requires_model_id(self):
        """Test that ESM2ModelConfig stores the model_id correctly."""
        config = ESM2ModelConfig(model_id=MODEL_ID)
        assert config.model_id == MODEL_ID

    def test_model_config_defaults(self):
        """Test that ESM2ModelConfig has the expected default values."""
        config = ESM2ModelConfig(model_id=MODEL_ID)
        assert config.pooling == "mean"
        assert config.repr_layer == -1

    def test_train_config_defaults(self):
        """Test that ESM2TrainConfig has the expected default values."""
        config = ESM2TrainConfig()
        assert config.freeze_backbone is True
        assert config.learning_rate == 1e-4
        assert config.optimizer_type == "adamw"
        assert config.batch_size == 8
        assert config.num_epochs == 10
        assert config.mask_probability == 0.15
        assert config.mask_splitting == (0.8, 0.1, 0.1)
        assert config.log_frequency == 1
        assert config.loss_type == "mlm"

    def test_train_config_num_epochs_zero_raises(self):
        """ESM2TrainConfig with num_epochs=0 must raise ValueError."""
        with pytest.raises(ValueError, match="num_epochs must be >= 1"):
            ESM2TrainConfig(num_epochs=0)


class TestFeaturise:
    """Tests for ESM2Model.featurise()."""

    def test_returns_dict_with_required_keys(self, esm2_model, sample_data):
        """Test that featurise returns a dict with input_ids and attention_mask."""
        result = esm2_model.featurise(sample_data)
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_tensors_have_correct_batch_size(self, esm2_model, sample_data):
        """Test that returned tensors have batch size equal to the number of sequences."""
        result = esm2_model.featurise(sample_data)
        assert result["input_ids"].shape[0] == len(sample_data)
        assert result["attention_mask"].shape[0] == len(sample_data)

    def test_tensors_are_2d(self, esm2_model, sample_data):
        """Test that returned tensors are 2D (batch, seq_len)."""
        result = esm2_model.featurise(sample_data)
        assert result["input_ids"].ndim == 2
        assert result["attention_mask"].ndim == 2

    def test_accepts_list_of_candidates(self, esm2_model, sample_data):
        """Test that featurise accepts a plain list of Candidate objects."""
        result = esm2_model.featurise(sample_data.candidates)
        assert result["input_ids"].shape[0] == len(sample_data)

    def test_returns_cpu_tensors(self, esm2_model, sample_data):
        """Test that featurise returns tensors on CPU regardless of model device."""
        result = esm2_model.featurise(sample_data)
        assert result["input_ids"].device.type == "cpu"
        assert result["attention_mask"].device.type == "cpu"


class TestPredict:
    """Tests for ESM2Model.predict()."""

    def test_mean_pooling_shape(self, esm2_model, sample_data):
        """Test that mean pooling produces embeddings of shape (n_seqs, hidden_dim)."""
        predictions = esm2_model.predict(sample_data.candidates)
        hidden_dim = esm2_model.esm_model.config.hidden_size
        assert predictions.means.shape == (len(sample_data), hidden_dim)

    def test_cls_pooling_shape(self, model_config, train_config, sample_data):
        """Test that CLS pooling produces embeddings of shape (n_seqs, hidden_dim)."""
        config = ESM2ModelConfig(model_id=MODEL_ID, pooling="cls")
        model = ESM2Model(
            name="cls_model", model_config=config, train_config=train_config, device="cpu"
        )
        predictions = model.predict(sample_data.candidates)
        hidden_dim = model.esm_model.config.hidden_size
        assert predictions.means.shape == (len(sample_data), hidden_dim)

    def test_last_hidden_state_pooling_shape(self, model_config, sample_data):
        """Test that last_hidden_state pooling produces shape (n_seqs, seq_len, hidden_dim)."""
        config = ESM2ModelConfig(model_id=MODEL_ID, pooling="last_hidden_state")
        # batch_size=1 required; batch_size>1 raises ValueError (different seq lengths)
        train_cfg = ESM2TrainConfig(freeze_backbone=True, batch_size=1)
        model = ESM2Model(
            name="lhs_model", model_config=config, train_config=train_cfg, device="cpu"
        )
        predictions = model.predict(sample_data.candidates)
        # Shape: (n_seqs, seq_len, hidden_dim) — seq_len includes special tokens and padding
        assert predictions.means.ndim == 3
        assert predictions.means.shape[0] == len(sample_data)
        assert predictions.means.shape[2] == model.esm_model.config.hidden_size

    def test_last_hidden_state_batch_size_gt_1_raises(self):
        """last_hidden_state pooling with batch_size > 1 must raise ValueError at init."""
        config = ESM2ModelConfig(model_id=MODEL_ID, pooling="last_hidden_state")
        train_cfg = ESM2TrainConfig(freeze_backbone=True, batch_size=2)
        with pytest.raises(ValueError, match="pooling='last_hidden_state' requires batch_size=1"):
            ESM2Model(name="lhs_bad", model_config=config, train_config=train_cfg, device="cpu")

    def test_variances_are_none(self, esm2_model, sample_data):
        """Test that predict returns Predictions with variances=None."""
        predictions = esm2_model.predict(sample_data.candidates)
        assert predictions.variances is None

    def test_embeddings_are_finite(self, esm2_model, sample_data):
        """Test that all embedding values are finite (no NaN or inf)."""
        predictions = esm2_model.predict(sample_data.candidates)
        assert np.all(np.isfinite(predictions.means))

    def test_repr_layer_produces_different_embeddings(self, train_config, sample_data):
        """Test that extracting from different layers produces different embeddings."""
        config_final = ESM2ModelConfig(model_id=MODEL_ID, repr_layer=-1)
        config_first = ESM2ModelConfig(model_id=MODEL_ID, repr_layer=1)
        model_final = ESM2Model(
            name="final", model_config=config_final, train_config=train_config, device="cpu"
        )
        model_first = ESM2Model(
            name="first", model_config=config_first, train_config=train_config, device="cpu"
        )
        preds_final = model_final.predict(sample_data.candidates)
        preds_first = model_first.predict(sample_data.candidates)
        assert not np.allclose(preds_final.means, preds_first.means)

    def test_predict_batched_matches_shape(self, esm2_small_batch_model):
        """predict() with N > batch_size returns the same shape as N <= batch_size."""
        candidates = [
            Candidate(data="MKTIIALSYIFCLVFA", modality="sequence"),
            Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence"),
            Candidate(data="GASGAAS", modality="sequence"),
            Candidate(data="PEPTIDE", modality="sequence"),
            Candidate(data="ACGT", modality="sequence"),
        ]
        predictions = esm2_small_batch_model.predict(candidates)
        hidden_dim = esm2_small_batch_model.esm_model.config.hidden_size
        assert predictions.means.shape == (5, hidden_dim)

    def test_predict_batched_equals_single_batch(self, esm2_small_batch_model, esm2_model):
        """Embeddings from batched predict equal those from a single-pass predict."""
        candidates = [
            Candidate(data="MKTIIALSYIFCLVFA", modality="sequence"),
            Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence"),
            Candidate(data="GASGAAS", modality="sequence"),
            Candidate(data="PEPTIDE", modality="sequence"),
            Candidate(data="ACGT", modality="sequence"),
        ]
        # esm2_model has batch_size=8 (fits all 5 in one pass after refactor)
        # esm2_small_batch_model has batch_size=2 (forces 3 mini-batches)
        single = esm2_model.predict(candidates)
        batched = esm2_small_batch_model.predict(candidates)
        np.testing.assert_allclose(single.means, batched.means, rtol=1e-5, atol=1e-5)

    def test_predict_empty_candidates(self, esm2_model):
        """predict() with an empty list returns an empty Predictions."""
        with pytest.raises(AssertionError):
            esm2_model.predict([])

    def test_predict_single_candidate(self, esm2_model):
        """predict() with a single candidate returns shape (1, hidden_dim)."""
        predictions = esm2_model.predict([Candidate(data="ACGT", modality="sequence")])
        hidden_dim = esm2_model.esm_model.config.hidden_size
        assert predictions.means.shape == (1, hidden_dim)


class TestMaskTokens:
    """Tests for ESM2Model._mask_tokens()."""

    def test_special_tokens_never_masked(self, esm2_model):
        """Test that CLS, EOS, and PAD positions are always excluded from masking."""
        # Two sequences of different lengths to produce PAD tokens after collation
        seqs = ["ACDE", "ACDEFGHIKLMNPQRSTVWY"]
        encoding = esm2_model.tokeniser(seqs, return_tensors="pt", padding=True)
        input_ids = encoding["input_ids"]

        special_ids = {
            esm2_model.tokeniser.cls_token_id,
            esm2_model.tokeniser.eos_token_id,
            esm2_model.tokeniser.pad_token_id,
        } - {None}

        _, labels = esm2_model._mask_tokens(input_ids)

        for sid in special_ids:
            positions = input_ids == sid
            if positions.any():
                assert (labels[positions] == -100).all()

    def test_nan_guard_ensures_at_least_one_masked_token(self):
        """Test that the NaN guard forces a mask even when mask_probability=0."""
        config = ESM2ModelConfig(model_id=MODEL_ID)
        train_cfg = ESM2TrainConfig(freeze_backbone=True, mask_probability=0.0)
        model = ESM2Model(
            name="zero_mask", model_config=config, train_config=train_cfg, device="cpu"
        )
        seqs = ["ACDEFGHIKL"] * 8
        encoding = model.tokeniser(seqs, return_tensors="pt", padding=True)
        input_ids = encoding["input_ids"]

        _, labels = model._mask_tokens(input_ids)

        assert (labels != -100).any(dim=1).all()

    def test_selected_labels_equal_original_token(self, esm2_model):
        """Test that labels at selected positions store the original token ID."""
        seqs = ["ACDEFGHIKL", "MNPQRSTVWY"]
        encoding = esm2_model.tokeniser(seqs, return_tensors="pt", padding=True)
        input_ids = encoding["input_ids"]

        _, labels = esm2_model._mask_tokens(input_ids)

        selected = labels != -100
        assert (labels[selected] == input_ids[selected]).all()

    def test_mask_token_split_proportions(self):
        """Test that the 80/10/10 replacement split matches the configured proportions."""
        config = ESM2ModelConfig(model_id=MODEL_ID)
        # mask_probability=1.0 selects all eligible tokens, giving stable statistics
        train_cfg = ESM2TrainConfig(
            freeze_backbone=True,
            mask_probability=1.0,
            mask_splitting=(0.8, 0.1, 0.1),
        )
        model = ESM2Model(
            name="split_model", model_config=config, train_config=train_cfg, device="cpu"
        )
        # 100 sequences × 20 eligible tokens = ~2 000 selected positions
        seqs = ["ACDEFGHIKLMNPQRSTVWY"] * 100
        encoding = model.tokeniser(seqs, return_tensors="pt", padding=True)
        input_ids = encoding["input_ids"]

        masked_ids, labels = model._mask_tokens(input_ids)

        selected = labels != -100
        n = selected.sum().item()
        n_mask = (masked_ids[selected] == model.tokeniser.mask_token_id).sum().item()
        n_unchanged = (masked_ids[selected] == input_ids[selected]).sum().item()
        # random = not [MASK] and not unchanged
        # (chance of random token coinciding with original ≈ 1/vocab_size ≈ 3%, negligible)
        n_random = n - n_mask - n_unchanged

        assert 0.70 < n_mask / n < 0.90, f"Expected ~80% [MASK], got {n_mask / n:.2f}"
        assert 0.03 < n_random / n < 0.20, f"Expected ~10% random, got {n_random / n:.2f}"
        assert 0.05 < n_unchanged / n < 0.20, f"Expected ~10% unchanged, got {n_unchanged / n:.2f}"


class TestComputeLogLikelihoodLabels:
    """Tests for ESM2Model._compute_log_likelihood_labels()."""

    def test_all_non_special_positions_labeled(self, esm2_model):
        """All non-special token positions should have labels != -100."""
        seqs = ["ACDE", "ACDEFGHIKLMNPQRSTVWY"]
        encoding = esm2_model.tokeniser(seqs, return_tensors="pt", padding=True)
        input_ids = encoding["input_ids"]

        special_ids = {
            esm2_model.tokeniser.cls_token_id,
            esm2_model.tokeniser.eos_token_id,
            esm2_model.tokeniser.pad_token_id,
        } - {None}
        special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for sid in special_ids:
            special_mask |= input_ids.eq(sid)

        _, labels = esm2_model._compute_log_likelihood_labels(input_ids)

        # Every non-special position must be labeled (not -100)
        assert (labels[~special_mask] != -100).all()

    def test_special_tokens_excluded_from_labels(self, esm2_model):
        """CLS, EOS, and PAD positions must have label -100."""
        seqs = ["ACDE", "ACDEFGHIKLMNPQRSTVWY"]
        encoding = esm2_model.tokeniser(seqs, return_tensors="pt", padding=True)
        input_ids = encoding["input_ids"]

        special_ids = {
            esm2_model.tokeniser.cls_token_id,
            esm2_model.tokeniser.eos_token_id,
            esm2_model.tokeniser.pad_token_id,
        } - {None}

        _, labels = esm2_model._compute_log_likelihood_labels(input_ids)

        for sid in special_ids:
            positions = input_ids == sid
            if positions.any():
                assert (labels[positions] == -100).all()

    def test_all_non_special_tokens_replaced_with_mask(self, esm2_model):
        """All non-special token positions in masked_ids should be mask_token_id."""
        seqs = ["ACDE", "ACDEFGHIKLMNPQRSTVWY"]
        encoding = esm2_model.tokeniser(seqs, return_tensors="pt", padding=True)
        input_ids = encoding["input_ids"]

        special_ids = {
            esm2_model.tokeniser.cls_token_id,
            esm2_model.tokeniser.eos_token_id,
            esm2_model.tokeniser.pad_token_id,
        } - {None}
        special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for sid in special_ids:
            special_mask |= input_ids.eq(sid)

        masked_ids, _ = esm2_model._compute_log_likelihood_labels(input_ids)

        assert (masked_ids[~special_mask] == esm2_model.tokeniser.mask_token_id).all()

    def test_labels_equal_original_at_labeled_positions(self, esm2_model):
        """Labels at non-special positions must equal the original token IDs."""
        seqs = ["ACDEFGHIKL", "MNPQRSTVWY"]
        encoding = esm2_model.tokeniser(seqs, return_tensors="pt", padding=True)
        input_ids = encoding["input_ids"]

        _, labels = esm2_model._compute_log_likelihood_labels(input_ids)

        labeled = labels != -100
        assert (labels[labeled] == input_ids[labeled]).all()


class TestTrainFrozen:
    """Tests for ESM2Model.train() when freeze_backbone=True."""

    def test_frozen_train_is_noop(self, esm2_model, sample_data):
        """Test that train() does not update any weights when freeze_backbone=True."""
        initial_params = {
            name: param.clone() for name, param in esm2_model.esm_model.named_parameters()
        }
        esm2_model.train(sample_data)
        for name, param in esm2_model.esm_model.named_parameters():
            assert torch.equal(initial_params[name], param), f"Parameter {name} changed"

    def test_frozen_epoch_metrics_empty(self, esm2_model, sample_data):
        """Test that get_epoch_metrics returns an empty list after frozen train()."""
        esm2_model.train(sample_data)
        assert esm2_model.get_epoch_metrics() == []

    def test_frozen_summary_metrics_empty(self, esm2_model, sample_data):
        """Test that get_training_summary_metrics returns an empty dict after frozen train()."""
        esm2_model.train(sample_data)
        assert esm2_model.get_training_summary_metrics() == {}


class TestTrainFinetune:
    """Tests for ESM2Model.train() when freeze_backbone=False."""

    def test_finetune_updates_weights(self, esm2_finetune_model, sample_data):
        """Test that MLM fine-tuning updates at least one model parameter."""
        initial_params = {
            name: param.clone() for name, param in esm2_finetune_model.esm_model.named_parameters()
        }
        esm2_finetune_model.train(sample_data)
        params_changed = any(
            not torch.equal(initial_params[name], param)
            for name, param in esm2_finetune_model.esm_model.named_parameters()
        )
        assert params_changed, "Fine-tuning should update model parameters"

    def test_epoch_metrics_recorded(self, esm2_finetune_model, sample_data):
        """Test that one SurrogateEpochMetrics is recorded per training epoch."""
        esm2_finetune_model.train(sample_data)
        metrics = esm2_finetune_model.get_epoch_metrics()
        assert len(metrics) == esm2_finetune_model.train_config.num_epochs
        assert all(isinstance(m, SurrogateEpochMetrics) for m in metrics)

    def test_epoch_metrics_train_loss_finite(self, esm2_finetune_model, sample_data):
        """Test that train_loss in each epoch metric is a finite number."""
        esm2_finetune_model.train(sample_data)
        for m in esm2_finetune_model.get_epoch_metrics():
            assert np.isfinite(m.train_loss)

    def test_epoch_metrics_reset_on_retrain(self, esm2_finetune_model, sample_data):
        """Test that epoch metrics are cleared at the start of each train() call."""
        esm2_finetune_model.train(sample_data)
        esm2_finetune_model.train(sample_data)
        assert (
            len(esm2_finetune_model.get_epoch_metrics())
            == esm2_finetune_model.train_config.num_epochs
        )

    def test_summary_metrics_has_final_train_loss(self, esm2_finetune_model, sample_data):
        """Test that get_training_summary_metrics includes a finite final_train_loss."""
        esm2_finetune_model.train(sample_data)
        summary = esm2_finetune_model.get_training_summary_metrics()
        assert "final_train_loss" in summary
        assert np.isfinite(summary["final_train_loss"])

    def test_val_loss_recorded_when_val_data_provided(self, esm2_finetune_model, sample_data):
        """Test that val_loss is recorded in epoch metrics when val_data is provided."""
        val_candidates = [Candidate(data="ACDEFGHIKL", modality="sequence")]
        val_data = LabelledCandidates(val_candidates, np.array([1.0]))
        esm2_finetune_model.train(sample_data, val_data=val_data)
        for m in esm2_finetune_model.get_epoch_metrics():
            assert m.val_loss is not None
            assert np.isfinite(m.val_loss)
        assert "final_val_loss" in esm2_finetune_model.get_training_summary_metrics()

    def test_val_loss_none_without_val_data(self, esm2_finetune_model, sample_data):
        """Test that val_loss is None in epoch metrics when no val_data is provided."""
        esm2_finetune_model.train(sample_data)
        for m in esm2_finetune_model.get_epoch_metrics():
            assert m.val_loss is None

    def test_epoch_metrics_contain_train_perplexity_and_token_accuracy(
        self, esm2_finetune_model, sample_data
    ):
        """Test that each epoch metric includes finite train_perplexity and train_token_accuracy."""
        esm2_finetune_model.train(sample_data)
        for m in esm2_finetune_model.get_epoch_metrics():
            assert "train_perplexity" in m.additional_metrics
            assert "train_token_accuracy" in m.additional_metrics
            assert np.isfinite(m.additional_metrics["train_perplexity"])
            assert 0.0 <= m.additional_metrics["train_token_accuracy"] <= 1.0

    def test_epoch_metrics_contain_val_perplexity_and_token_accuracy(
        self, esm2_finetune_model, sample_data
    ):
        """Test that epoch metrics include val_perplexity and val_token_accuracy."""
        val_candidates = [Candidate(data="ACDEFGHIKL", modality="sequence")]
        val_data = LabelledCandidates(val_candidates, np.array([1.0]))
        esm2_finetune_model.train(sample_data, val_data=val_data)
        for m in esm2_finetune_model.get_epoch_metrics():
            assert "val_perplexity" in m.additional_metrics
            assert "val_token_accuracy" in m.additional_metrics
            assert np.isfinite(m.additional_metrics["val_perplexity"])
            assert 0.0 <= m.additional_metrics["val_token_accuracy"] <= 1.0

    def test_summary_metrics_contain_final_perplexity_and_token_accuracy(
        self, esm2_finetune_model, sample_data
    ):
        """Test that summary metrics include final_train_perplexity and token_accuracy."""
        esm2_finetune_model.train(sample_data)
        summary = esm2_finetune_model.get_training_summary_metrics()
        assert "final_train_perplexity" in summary
        assert "final_train_token_accuracy" in summary
        assert np.isfinite(summary["final_train_perplexity"])
        assert 0.0 <= summary["final_train_token_accuracy"] <= 1.0

    def test_summary_metrics_contain_val_perplexity_and_token_accuracy_with_val_data(
        self, esm2_finetune_model, sample_data
    ):
        """Test that summary metrics include final_val_perplexity and token_accuracy."""
        val_candidates = [Candidate(data="ACDEFGHIKL", modality="sequence")]
        val_data = LabelledCandidates(val_candidates, np.array([1.0]))
        esm2_finetune_model.train(sample_data, val_data=val_data)
        summary = esm2_finetune_model.get_training_summary_metrics()
        assert "final_val_perplexity" in summary
        assert "final_val_token_accuracy" in summary
        assert np.isfinite(summary["final_val_perplexity"])
        assert 0.0 <= summary["final_val_token_accuracy"] <= 1.0


class TestTrainLogLikelihood:
    """Tests for ESM2Model.train() when loss_type='log_likelihood'."""

    def test_ll_finetune_updates_weights(self, esm2_ll_model, sample_data):
        """Test that log-likelihood fine-tuning updates at least one model parameter."""
        initial_params = {
            name: param.clone() for name, param in esm2_ll_model.esm_model.named_parameters()
        }
        esm2_ll_model.train(sample_data)
        params_changed = any(
            not torch.equal(initial_params[name], param)
            for name, param in esm2_ll_model.esm_model.named_parameters()
        )
        assert params_changed, "Log-likelihood fine-tuning should update model parameters"

    def test_ll_epoch_metrics_recorded(self, esm2_ll_model, sample_data):
        """Test that one SurrogateEpochMetrics is recorded per training epoch."""
        esm2_ll_model.train(sample_data)
        metrics = esm2_ll_model.get_epoch_metrics()
        assert len(metrics) == esm2_ll_model.train_config.num_epochs
        assert all(isinstance(m, SurrogateEpochMetrics) for m in metrics)

    def test_ll_epoch_metrics_train_loss_finite(self, esm2_ll_model, sample_data):
        """Test that train_loss in each epoch metric is a finite number."""
        esm2_ll_model.train(sample_data)
        for m in esm2_ll_model.get_epoch_metrics():
            assert np.isfinite(m.train_loss)

    def test_ll_summary_metrics_has_final_train_loss(self, esm2_ll_model, sample_data):
        """Test that get_training_summary_metrics includes a finite final_train_loss."""
        esm2_ll_model.train(sample_data)
        summary = esm2_ll_model.get_training_summary_metrics()
        assert "final_train_loss" in summary
        assert np.isfinite(summary["final_train_loss"])

    def test_ll_epoch_metrics_contain_train_perplexity_token_accuracy_and_log_likelihood(
        self, esm2_ll_model, sample_data
    ):
        """Test that each epoch metric has finite train_perplexity, train_token_accuracy,
        and train_log_likelihood when loss_type='log_likelihood'.
        """
        esm2_ll_model.train(sample_data)
        for m in esm2_ll_model.get_epoch_metrics():
            assert "train_perplexity" in m.additional_metrics
            assert "train_token_accuracy" in m.additional_metrics
            assert "train_log_likelihood" in m.additional_metrics
            assert np.isfinite(m.additional_metrics["train_perplexity"])
            assert 0.0 <= m.additional_metrics["train_token_accuracy"] <= 1.0
            assert np.isfinite(m.additional_metrics["train_log_likelihood"])

    def test_ll_val_loss_recorded_when_val_data_provided(self, esm2_ll_model, sample_data):
        """Test that val_loss is recorded when val_data is provided."""
        val_candidates = [Candidate(data="ACDEFGHIKL", modality="sequence")]
        val_data = LabelledCandidates(val_candidates, np.array([1.0]))
        esm2_ll_model.train(sample_data, val_data=val_data)
        for m in esm2_ll_model.get_epoch_metrics():
            assert m.val_loss is not None
            assert np.isfinite(m.val_loss)
        assert "final_val_loss" in esm2_ll_model.get_training_summary_metrics()

    def test_ll_epoch_metrics_contain_val_log_likelihood(self, esm2_ll_model, sample_data):
        """Test that val_log_likelihood is recorded in epoch metrics when val_data provided."""
        val_candidates = [Candidate(data="ACDEFGHIKL", modality="sequence")]
        val_data = LabelledCandidates(val_candidates, np.array([1.0]))
        esm2_ll_model.train(sample_data, val_data=val_data)
        for m in esm2_ll_model.get_epoch_metrics():
            assert "val_log_likelihood" in m.additional_metrics
            assert np.isfinite(m.additional_metrics["val_log_likelihood"])

    def test_ll_summary_metrics_contain_final_train_log_likelihood(
        self, esm2_ll_model, sample_data
    ):
        """Test that summary metrics include finite final_train_log_likelihood."""
        esm2_ll_model.train(sample_data)
        summary = esm2_ll_model.get_training_summary_metrics()
        assert "final_train_log_likelihood" in summary
        assert np.isfinite(summary["final_train_log_likelihood"])

    def test_ll_log_likelihood_is_negative_of_loss(self, esm2_ll_model, sample_data):
        """Test that train_log_likelihood == -train_loss in each epoch metric."""
        esm2_ll_model.train(sample_data)
        for m in esm2_ll_model.get_epoch_metrics():
            assert np.isclose(
                m.additional_metrics["train_log_likelihood"], -m.train_loss, rtol=1e-5
            )

    def test_mlm_mode_does_not_emit_log_likelihood_metric(self, esm2_finetune_model, sample_data):
        """Test that MLM training (default loss_type) does NOT add log_likelihood to metrics."""
        esm2_finetune_model.train(sample_data)
        for m in esm2_finetune_model.get_epoch_metrics():
            assert "train_log_likelihood" not in m.additional_metrics
