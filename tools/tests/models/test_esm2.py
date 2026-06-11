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

pytest.importorskip("transformers", reason="esm2 not installed; install alf_tools[esm2]")

from alf_tools.models.esm2 import ESM2Model, ESM2ModelConfig, ESM2TrainConfig

MODEL_ID = "facebook/esm2_t6_8M_UR50D"


@pytest.fixture(scope="session")
def model_config():
    """Create a default ESM2ModelConfig for testing.

    Returns:
        An ESM2ModelConfig using the smallest ESM-2 checkpoint.
    """
    return ESM2ModelConfig(model_id=MODEL_ID, seed=42)


@pytest.fixture(scope="session")
def train_config():
    """Create a frozen ESM2TrainConfig for testing.

    Returns:
        An ESM2TrainConfig with freeze_backbone=True and scoring_function="pll",
        for zero-shot PLL scoring mode.
    """
    return ESM2TrainConfig(freeze_backbone=True, scoring_function="pll")


@pytest.fixture(scope="session")
def esm2_model(model_config, train_config):
    """Frozen ESM-2 model — downloaded once per test session.

    Returns:
        An ESM2Model with frozen backbone on CPU.
    """
    return ESM2Model(
        name="test_esm2", model_config=model_config, train_config=train_config, device="cpu"
    )


@pytest.fixture(scope="session")
def esm2_small_batch_model():
    """Frozen ESM-2 with batch_size=2 to exercise multi-batch predict.

    Returns:
        An ESM2Model with batch_size=2, frozen backbone,
        and scoring_function="pll", CPU.
    """
    config = ESM2ModelConfig(model_id=MODEL_ID, seed=42)
    train_cfg = ESM2TrainConfig(freeze_backbone=True, batch_size=2, scoring_function="pll")
    return ESM2Model(
        name="test_esm2_small_batch", model_config=config, train_config=train_cfg, device="cpu"
    )


@pytest.fixture(scope="session")
def esm2_linear_head_model():
    """Frozen ESM-2 with linear head and default batch size for batching tests.

    Returns:
        An ESM2Model with default batch_size, frozen backbone,
        and scoring_function="linear_head", CPU.
    """
    config = ESM2ModelConfig(model_id=MODEL_ID, seed=42)
    train_cfg = ESM2TrainConfig(freeze_backbone=True, scoring_function="linear_head")
    return ESM2Model(
        name="test_esm2_linear_head", model_config=config, train_config=train_cfg, device="cpu"
    )


@pytest.fixture(scope="session")
def esm2_linear_head_small_batch_model():
    """Frozen ESM-2 with linear head and batch_size=2 to exercise multi-batch predict.

    Returns:
        An ESM2Model with batch_size=2, frozen backbone,
        and scoring_function="linear_head", CPU.
    """
    config = ESM2ModelConfig(model_id=MODEL_ID, seed=42)
    train_cfg = ESM2TrainConfig(freeze_backbone=True, batch_size=2, scoring_function="linear_head")
    return ESM2Model(
        name="test_esm2_linear_head_small_batch",
        model_config=config,
        train_config=train_cfg,
        device="cpu",
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
        """ESM2ModelConfig stores the model_id correctly."""
        config = ESM2ModelConfig(model_id=MODEL_ID)
        assert config.model_id == MODEL_ID

    def test_model_config_defaults(self):
        """ESM2ModelConfig has expected default values."""
        config = ESM2ModelConfig(model_id=MODEL_ID)
        assert config.pooling == "mean"
        assert config.repr_layer == -1

    def test_train_config_defaults(self):
        """ESM2TrainConfig has expected default values."""
        config = ESM2TrainConfig()
        assert config.mode == "linear_head"
        assert config.freeze_backbone is True
        assert config.loss_fn is None
        assert config.output_dim == 1
        assert config.mask_probability == 0.15
        assert config.mask_splitting == (0.8, 0.1, 0.1)
        assert config.learning_rate == 1e-4
        assert config.optimizer_type == "adamw"
        assert config.batch_size == 8
        assert config.num_epochs == 10
        assert config.log_frequency == 1

    def test_mode_invalid_raises(self):
        """mode with an unknown value raises ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            ESM2TrainConfig(mode="unknown")  # type: ignore[arg-type]

    def test_train_config_num_epochs_zero_raises(self):
        """num_epochs=0 must raise ValueError."""
        with pytest.raises(ValueError, match="num_epochs must be >= 1"):
            ESM2TrainConfig(num_epochs=0)

    def test_loss_fn_mlm_with_linear_head_raises(self):
        """loss_fn='mlm' with mode='linear_head' raises ValueError."""
        with pytest.raises(ValueError, match="loss_fn='mlm' is only valid"):
            ESM2TrainConfig(mode="linear_head", loss_fn="mlm")

    def test_loss_fn_set_with_frozen_esm2_likelihoods_raises(self):
        """loss_fn not None when mode='esm2_likelihoods' + freeze_backbone=True raises."""
        with pytest.raises(ValueError, match="loss_fn cannot be set"):
            ESM2TrainConfig(mode="esm2_likelihoods", freeze_backbone=True, loss_fn="mlm")

    def test_loss_fn_mse_with_frozen_esm2_likelihoods_raises(self):
        """loss_fn='mse' with mode='esm2_likelihoods' + freeze_backbone=True raises."""
        with pytest.raises(ValueError, match="loss_fn cannot be set"):
            ESM2TrainConfig(mode="esm2_likelihoods", freeze_backbone=True, loss_fn="mse")

    def test_esm2_likelihoods_frozen_no_loss_fn_valid(self):
        """mode='esm2_likelihoods' + freeze_backbone=True + loss_fn=None is valid."""
        cfg = ESM2TrainConfig(mode="esm2_likelihoods")
        assert cfg.mode == "esm2_likelihoods"
        assert cfg.loss_fn is None

    def test_esm2_likelihoods_unfrozen_mlm_valid(self):
        """mode='esm2_likelihoods' + freeze_backbone=False + loss_fn='mlm' is valid."""
        cfg = ESM2TrainConfig(mode="esm2_likelihoods", freeze_backbone=False, loss_fn="mlm")
        assert cfg.loss_fn == "mlm"

    def test_esm2_likelihoods_unfrozen_non_mlm_loss_raises(self):
        """mode='esm2_likelihoods' + freeze_backbone=False + loss_fn='mse' raises."""
        with pytest.raises(ValueError, match="MLM loss"):
            ESM2TrainConfig(mode="esm2_likelihoods", freeze_backbone=False, loss_fn="mse")

    def test_esm2_likelihoods_unfrozen_cross_entropy_raises(self):
        """mode='esm2_likelihoods' + freeze_backbone=False + loss_fn='cross_entropy' raises."""
        with pytest.raises(ValueError, match="MLM loss"):
            ESM2TrainConfig(mode="esm2_likelihoods", freeze_backbone=False, loss_fn="cross_entropy")

    def test_esm2_likelihoods_unfrozen_none_loss_fn_raises(self):
        """mode='esm2_likelihoods' + freeze_backbone=False + loss_fn=None raises."""
        with pytest.raises(ValueError, match="MLM loss"):
            ESM2TrainConfig(mode="esm2_likelihoods", freeze_backbone=False, loss_fn=None)

    def test_mask_probability_out_of_range_raises(self):
        """mask_probability outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="mask_probability"):
            ESM2TrainConfig(mode="esm2_likelihoods", freeze_backbone=False, loss_fn="mlm",
                            mask_probability=0.0)

    def test_mask_splitting_not_summing_to_one_raises(self):
        """mask_splitting that does not sum to 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="mask_splitting"):
            ESM2TrainConfig(mode="esm2_likelihoods", freeze_backbone=False, loss_fn="mlm",
                            mask_splitting=(0.8, 0.1, 0.05))

    def test_mask_probability_warning_when_frozen(self):
        """Non-default mask_probability with freeze_backbone=True emits UserWarning."""
        with pytest.warns(UserWarning, match="mask_probability"):
            ESM2TrainConfig(mode="esm2_likelihoods", freeze_backbone=True, mask_probability=0.3)

    def test_mask_splitting_warning_when_frozen(self):
        """Non-default mask_splitting with freeze_backbone=True emits UserWarning."""
        with pytest.warns(UserWarning, match="mask_splitting"):
            ESM2TrainConfig(mode="esm2_likelihoods", freeze_backbone=True,
                            mask_splitting=(0.7, 0.2, 0.1))

    def test_freeze_backbone_false_raises_for_linear_head(self):
        """freeze_backbone=False with mode='linear_head' raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            ESM2TrainConfig(mode="linear_head", freeze_backbone=False)

    def test_last_hidden_state_with_linear_head_mode_raises(self):
        """last_hidden_state pooling with mode='linear_head' raises at model init."""
        config = ESM2ModelConfig(model_id=MODEL_ID, pooling="last_hidden_state")
        train_cfg = ESM2TrainConfig(mode="linear_head", loss_fn="mse", batch_size=1)
        with pytest.raises(ValueError, match="last_hidden_state.*mode"):
            ESM2Model(name="lhs_mlp", model_config=config, train_config=train_cfg, device="cpu")

    def test_invalid_pooling_raises(self):
        """ESM2ModelConfig with an invalid pooling strategy raises ValueError."""
        with pytest.raises(ValueError, match="pooling must be"):
            ESM2ModelConfig(model_id=MODEL_ID, pooling="max")  # type: ignore[arg-type]

    def test_repr_layer_out_of_range_raises(self):
        """ESM2Model raises ValueError when repr_layer is out of range."""
        config = ESM2ModelConfig(model_id=MODEL_ID, repr_layer=999)
        train_cfg = ESM2TrainConfig()
        with pytest.raises(ValueError, match="repr_layer.*out of range"):
            ESM2Model(name="bad_layer", model_config=config, train_config=train_cfg, device="cpu")


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

    def test_non_string_candidate_data_raises(self, esm2_model):
        """featurise() raises ValueError when Candidate.data is not a string."""
        bad_candidate = Candidate(data=123, modality="sequence")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Expected string sequences"):
            esm2_model.featurise([bad_candidate])


class TestPredict:
    """Tests for ESM2Model.predict()."""

    def test_predict_shape(self, esm2_model, sample_data):
        """predict() returns means of shape (n_candidates,) — one scalar per sequence."""
        predictions = esm2_model.predict(sample_data.candidates)
        assert predictions.means.ndim == 1
        assert predictions.means.shape == (len(sample_data),)

    def test_last_hidden_state_batch_size_gt_1_raises(self):
        """last_hidden_state pooling with batch_size > 1 must raise ValueError at init."""
        config = ESM2ModelConfig(model_id=MODEL_ID, pooling="last_hidden_state")
        train_cfg = ESM2TrainConfig(freeze_backbone=True, batch_size=2)
        with pytest.raises(ValueError, match="pooling='last_hidden_state' requires batch_size=1"):
            ESM2Model(name="lhs_bad", model_config=config, train_config=train_cfg, device="cpu")

    def test_variances_are_none(self, esm2_model, sample_data):
        """predict() returns Predictions with variances=None."""
        predictions = esm2_model.predict(sample_data.candidates)
        assert predictions.variances is None

    def test_log_likelihoods_are_finite(self, esm2_model, sample_data):
        """predict() returns finite log-likelihood scores (no NaN or inf)."""
        predictions = esm2_model.predict(sample_data.candidates)
        assert np.all(np.isfinite(predictions.means))

    def test_log_likelihoods_are_negative(self, esm2_model, sample_data):
        """predict() returns negative values since log-probabilities are always <= 0."""
        predictions = esm2_model.predict(sample_data.candidates)
        assert np.all(predictions.means <= 0)

    def test_predict_batched_matches_shape(self, esm2_small_batch_model):
        """predict() with N > batch_size returns shape (N,)."""
        candidates = [
            Candidate(data="MKTIIALSYIFCLVFA", modality="sequence"),
            Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence"),
            Candidate(data="GASGAAS", modality="sequence"),
            Candidate(data="PEPTIDE", modality="sequence"),
            Candidate(data="ACGT", modality="sequence"),
        ]
        predictions = esm2_small_batch_model.predict(candidates)
        assert predictions.means.shape == (5,)

    def test_predict_batched_equals_single_batch(self, esm2_small_batch_model, esm2_model):
        """Log-likelihoods from batched predict equal those from a single-pass predict."""
        candidates = [
            Candidate(data="MKTIIALSYIFCLVFA", modality="sequence"),
            Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence"),
            Candidate(data="GASGAAS", modality="sequence"),
            Candidate(data="PEPTIDE", modality="sequence"),
            Candidate(data="ACGT", modality="sequence"),
        ]
        single = esm2_model.predict(candidates)
        batched = esm2_small_batch_model.predict(candidates)
        np.testing.assert_allclose(single.means, batched.means, rtol=1e-5, atol=1e-5)

    def test_predict_empty_candidates_raises(self, esm2_model):
        """predict() with an empty list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            esm2_model.predict([])

    def test_predict_single_candidate(self, esm2_model):
        """predict() with a single candidate returns shape (1,)."""
        predictions = esm2_model.predict([Candidate(data="ACGT", modality="sequence")])
        assert predictions.means.shape == (1,)


class TestEmbed:
    """Tests for ESM2Model.embed()."""

    def test_embed_mean_pooling_shape(self, esm2_model, sample_data):
        """embed() with mean pooling returns shape (n_seqs, hidden_dim)."""
        embeddings = esm2_model.embed(sample_data.candidates)
        hidden_dim = esm2_model.esm_model.config.hidden_size
        assert embeddings.shape == (len(sample_data), hidden_dim)

    def test_embed_cls_pooling_shape(self, train_config, sample_data):
        """embed() with cls pooling returns shape (n_seqs, hidden_dim)."""
        config = ESM2ModelConfig(model_id=MODEL_ID, pooling="cls")
        model = ESM2Model(
            name="cls_model", model_config=config, train_config=train_config, device="cpu"
        )
        embeddings = model.embed(sample_data.candidates)
        hidden_dim = model.esm_model.config.hidden_size
        assert embeddings.shape == (len(sample_data), hidden_dim)

    def test_embed_last_hidden_state_raises_with_scoring_linear_head(self, sample_data):
        """last_hidden_state pooling is incompatible with scoring_function='linear_head' — raises
        at init.
        """
        config = ESM2ModelConfig(model_id=MODEL_ID, pooling="last_hidden_state")
        train_cfg = ESM2TrainConfig(
            freeze_backbone=True, scoring_function="linear_head", batch_size=1
        )
        with pytest.raises(ValueError, match="pooling='last_hidden_state' is not supported with"):
            ESM2Model(name="lhs_model", model_config=config, train_config=train_cfg, device="cpu")

    def test_embed_last_hidden_state_shape(self, sample_data):
        """embed() with last_hidden_state pooling returns shape (n_seqs, seq_len, hidden_dim)."""
        config = ESM2ModelConfig(model_id=MODEL_ID, pooling="last_hidden_state")
        train_cfg = ESM2TrainConfig(freeze_backbone=True, scoring_function="pll", batch_size=1)
        model = ESM2Model(
            name="lhs_model", model_config=config, train_config=train_cfg, device="cpu"
        )
        embeddings = model.embed(sample_data.candidates)
        assert embeddings.ndim == 3
        assert embeddings.shape[0] == len(sample_data)
        assert embeddings.shape[2] == model.esm_model.config.hidden_size

    def test_embed_empty_candidates(self, esm2_model):
        """embed() with an empty list returns shape (0, hidden_dim)."""
        hidden_dim = esm2_model.esm_model.config.hidden_size
        result = esm2_model.embed([])
        assert result.shape == (0, hidden_dim)

    def test_embed_repr_layer_produces_different_embeddings(self, train_config, sample_data):
        """embed() extracts from different layers, producing different embeddings."""
        config_final = ESM2ModelConfig(model_id=MODEL_ID, repr_layer=-1, seed=42)
        config_first = ESM2ModelConfig(model_id=MODEL_ID, repr_layer=1, seed=42)
        model_final = ESM2Model(
            name="final", model_config=config_final, train_config=train_config, device="cpu"
        )
        model_first = ESM2Model(
            name="first", model_config=config_first, train_config=train_config, device="cpu"
        )
        emb_final = model_final.embed(sample_data.candidates)
        emb_first = model_first.embed(sample_data.candidates)
        assert not np.allclose(emb_final, emb_first)

    def test_embed_finite(self, esm2_model, sample_data):
        """embed() returns finite values (no NaN or inf)."""
        embeddings = esm2_model.embed(sample_data.candidates)
        assert np.all(np.isfinite(embeddings))

    def test_embed_batched_matches_shape(self, esm2_small_batch_model):
        """embed() with N > batch_size returns shape (N, hidden_dim)."""
        candidates = [
            Candidate(data="MKTIIALSYIFCLVFA", modality="sequence"),
            Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence"),
            Candidate(data="GASGAAS", modality="sequence"),
            Candidate(data="PEPTIDE", modality="sequence"),
            Candidate(data="ACGT", modality="sequence"),
        ]
        embeddings = esm2_small_batch_model.embed(candidates)
        hidden_dim = esm2_small_batch_model.esm_model.config.hidden_size
        assert embeddings.shape == (5, hidden_dim)

    def test_embed_batched_equals_single_batch(self, esm2_small_batch_model, esm2_model):
        """Embeddings from batched embed() equal those from a single-pass embed()."""
        candidates = [
            Candidate(data="MKTIIALSYIFCLVFA", modality="sequence"),
            Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence"),
            Candidate(data="GASGAAS", modality="sequence"),
            Candidate(data="PEPTIDE", modality="sequence"),
            Candidate(data="ACGT", modality="sequence"),
        ]
        single = esm2_model.embed(candidates)
        batched = esm2_small_batch_model.embed(candidates)
        np.testing.assert_allclose(single, batched, rtol=1e-5, atol=1e-5)


@pytest.fixture(scope="module")
def frozen_train_model():
    """Module-scoped ESM-2 model with scoring_function=None for TestTrainFrozen.

    Returns:
        An ESM2Model with freeze_backbone=True and scoring_function="pll" on CPU.
    """
    config = ESM2ModelConfig(model_id=MODEL_ID, seed=42)
    train_cfg = ESM2TrainConfig(freeze_backbone=True, scoring_function="pll")
    return ESM2Model(
        name="test_esm2_frozen_train", model_config=config, train_config=train_cfg, device="cpu"
    )


class TestTrainFrozen:
    """Tests for ESM2Model.train() when scoring_function=None."""

    def test_train_without_scoring_function_raises(self, frozen_train_model, sample_data):
        """train() raises NotImplementedError when scoring_function=None."""
        with pytest.raises(NotImplementedError):
            frozen_train_model.train(sample_data)

    def test_epoch_metrics_empty_before_training(self, frozen_train_model):
        """get_epoch_metrics() returns empty list before any training."""
        assert frozen_train_model.get_epoch_metrics() == []

    def test_summary_metrics_empty_before_training(self, frozen_train_model):
        """get_training_summary_metrics() returns empty dict before any training."""
        assert frozen_train_model.get_training_summary_metrics() == {}


@pytest.fixture
def esm2_mlp_model():
    """Function-scoped ESM-2 model with linear regression head (output_dim=1, mse loss).

    Function-scoped so each test gets a fresh, untrained model — prevents
    implicit ordering dependencies from mutation via train() calls.

    Returns:
        An ESM2Model with scoring_function='linear_head', loss_fn='mse', output_dim=1, CPU.
    """
    config = ESM2ModelConfig(model_id=MODEL_ID, seed=42)
    train_cfg = ESM2TrainConfig(
        scoring_function="linear_head",
        loss_fn="mse",
        output_dim=1,
        num_epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        log_frequency=1,
    )
    return ESM2Model(
        name="test_esm2_mlp", model_config=config, train_config=train_cfg, device="cpu"
    )


@pytest.fixture
def esm2_mlp_classification_model():
    """Function-scoped ESM-2 model with linear classification head (output_dim=2, cross_entropy).

    Function-scoped so each test gets a fresh, untrained model — prevents
    implicit ordering dependencies from mutation via train() calls.

    Returns:
        An ESM2Model with scoring_function='linear_head', loss_fn='cross_entropy',
        output_dim=2, CPU.
    """
    config = ESM2ModelConfig(model_id=MODEL_ID, seed=42)
    train_cfg = ESM2TrainConfig(
        scoring_function="linear_head",
        loss_fn="cross_entropy",
        output_dim=2,
        num_epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        log_frequency=1,
    )
    return ESM2Model(
        name="test_esm2_mlp_cls", model_config=config, train_config=train_cfg, device="cpu"
    )


class TestMLPHead:
    """Tests for ESM2Model with scoring_function='linear_head'."""

    def test_head_is_linear_layer(self, esm2_mlp_model):
        """MLP head should be a torch.nn.Linear module."""
        assert esm2_mlp_model._head is not None
        assert isinstance(esm2_mlp_model._head, torch.nn.Linear)

    def test_head_output_dim(self, esm2_mlp_model):
        """Linear head output features should equal output_dim."""
        hidden_dim = esm2_mlp_model.esm_model.config.hidden_size
        assert esm2_mlp_model._head.in_features == hidden_dim
        assert esm2_mlp_model._head.out_features == 1

    def test_backbone_frozen_in_mlp_mode(self, esm2_mlp_model):
        """All ESM-2 backbone parameters should have requires_grad=False."""
        for param in esm2_mlp_model.esm_model.parameters():
            assert not param.requires_grad

    def test_head_on_correct_device(self, esm2_mlp_model):
        """Linear head should be on the same device as the ESM-2 model."""
        head_device = next(esm2_mlp_model._head.parameters()).device
        model_device = next(esm2_mlp_model.esm_model.parameters()).device
        assert head_device == model_device

    def test_no_head_when_scoring_function_none(self, esm2_model):
        """_head should be None when scoring_function=None."""
        assert esm2_model._head is None

    def test_classification_head_output_dim(self, esm2_mlp_classification_model):
        """Classification head should have out_features equal to output_dim."""
        assert esm2_mlp_classification_model._head.out_features == 2

    def test_prepare_data_loader_yields_labels_in_mlp_mode(self, esm2_mlp_model, sample_data):
        """DataLoader in scoring_function='linear_head' mode yields
        (input_ids, attention_mask, targets) triples.
        """
        loader = esm2_mlp_model._prepare_data_loader(sample_data)
        batch = next(iter(loader))
        assert len(batch) == 3  # input_ids, attention_mask, targets
        _, _, targets = batch
        assert targets.dtype == torch.float32

    def test_prepare_data_loader_no_labels_without_scoring_function(self, esm2_model, sample_data):
        """DataLoader with scoring_function=None yields (input_ids, attention_mask) pairs."""
        loader = esm2_model._prepare_data_loader(sample_data)
        batch = next(iter(loader))
        assert len(batch) == 2

    def test_mlp_train_updates_head_weights(self, esm2_mlp_model, sample_data):
        """train() in scoring_function='linear_head' mode updates the linear head parameters."""
        initial_weight = esm2_mlp_model._head.weight.clone()
        esm2_mlp_model.train(sample_data)
        assert not torch.equal(initial_weight, esm2_mlp_model._head.weight)

    def test_mlp_train_does_not_update_backbone(self, esm2_mlp_model, sample_data):
        """train() in scoring_function='linear_head' mode must not change
        any ESM-2 backbone parameters.
        """
        initial_params = {
            name: param.clone() for name, param in esm2_mlp_model.esm_model.named_parameters()
        }
        esm2_mlp_model.train(sample_data)
        for name, param in esm2_mlp_model.esm_model.named_parameters():
            assert torch.equal(initial_params[name], param), f"Backbone param {name} changed"

    def test_mlp_train_records_epoch_metrics(self, esm2_mlp_model, sample_data):
        """train() in scoring_function='linear_head' mode records one
        SurrogateEpochMetrics per epoch.
        """
        esm2_mlp_model.train(sample_data)
        metrics = esm2_mlp_model.get_epoch_metrics()
        # Relies on log_frequency=1 in the fixture so every epoch is logged.
        assert len(metrics) == esm2_mlp_model.train_config.num_epochs
        assert all(isinstance(m, SurrogateEpochMetrics) for m in metrics)

    def test_mlp_train_loss_is_finite(self, esm2_mlp_model, sample_data):
        """Each epoch metric train_loss must be finite."""
        esm2_mlp_model.train(sample_data)
        for m in esm2_mlp_model.get_epoch_metrics():
            assert np.isfinite(m.train_loss)

    def test_mlp_train_summary_has_final_train_loss(self, esm2_mlp_model, sample_data):
        """Summary metrics include a finite final_train_loss."""
        esm2_mlp_model.train(sample_data)
        summary = esm2_mlp_model.get_training_summary_metrics()
        assert "final_train_loss" in summary
        assert np.isfinite(summary["final_train_loss"])

    def test_mlp_val_loss_recorded_when_val_data_provided(self, esm2_mlp_model, sample_data):
        """val_loss is recorded in epoch metrics and final summary when val_data is provided."""
        val_candidates = [Candidate(data="ACDEFGHIKL", modality="sequence")]
        val_data = LabelledCandidates(val_candidates, np.array([1.0]))
        esm2_mlp_model.train(sample_data, val_data=val_data)
        for m in esm2_mlp_model.get_epoch_metrics():
            assert m.val_loss is not None
            assert np.isfinite(m.val_loss)
        summary = esm2_mlp_model.get_training_summary_metrics()
        assert "final_val_loss" in summary
        assert np.isfinite(summary["final_val_loss"])

    def test_mlp_cross_entropy_train_updates_head(self, esm2_mlp_classification_model):
        """train() with cross_entropy loss and integer labels updates the head."""
        candidates = [
            Candidate(data="ACDEFGHIKL", modality="sequence"),
            Candidate(data="MNPQRSTVWY", modality="sequence"),
            Candidate(data="ACMNPQRST", modality="sequence"),
        ]
        labels = np.array([0.0, 1.0, 0.0])  # class indices as floats (cast to long internally)
        sample = LabelledCandidates(candidates, labels)
        initial_weight = esm2_mlp_classification_model._head.weight.clone()
        esm2_mlp_classification_model.train(sample)
        assert not torch.equal(initial_weight, esm2_mlp_classification_model._head.weight)

    def test_mlp_mode_does_not_emit_log_likelihood_metric(self, esm2_mlp_model, sample_data):
        """MLP head mode must NOT include train_log_likelihood in epoch metrics."""
        esm2_mlp_model.train(sample_data)
        for m in esm2_mlp_model.get_epoch_metrics():
            assert "train_log_likelihood" not in m.additional_metrics

    def test_mlp_predict_shape(self, esm2_mlp_model, sample_data):
        """predict() in scoring_function='linear_head' mode returns means
        of shape (n_candidates,).
        """
        predictions = esm2_mlp_model.predict(sample_data.candidates)
        assert predictions.means.ndim == 1
        assert predictions.means.shape == (len(sample_data),)

    def test_mlp_predict_variances_none(self, esm2_mlp_model, sample_data):
        """predict() in scoring_function='linear_head' mode returns Predictions
        with variances=None.
        """
        predictions = esm2_mlp_model.predict(sample_data.candidates)
        assert predictions.variances is None

    def test_mlp_predict_finite(self, esm2_mlp_model, sample_data):
        """predict() in scoring_function='linear_head' mode returns finite values."""
        predictions = esm2_mlp_model.predict(sample_data.candidates)
        assert np.all(np.isfinite(predictions.means))

    def test_predict_linear_head_batched_equals_single_batch(
        self, esm2_linear_head_model, esm2_linear_head_small_batch_model
    ):
        """Linear-head predictions from batched predict equal those from a single-pass predict."""
        candidates = [
            Candidate(data="MKTIIALSYIFCLVFA", modality="sequence"),
            Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence"),
            Candidate(data="GASGAAS", modality="sequence"),
            Candidate(data="PEPTIDE", modality="sequence"),
            Candidate(data="ACGT", modality="sequence"),
        ]
        single = esm2_linear_head_model.predict(candidates)
        batched = esm2_linear_head_small_batch_model.predict(candidates)
        np.testing.assert_allclose(single.means, batched.means, rtol=1e-5, atol=1e-5)

    def test_mlp_predict_empty_raises(self, esm2_mlp_model):
        """predict() with an empty list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            esm2_mlp_model.predict([])

    def test_mlp_classification_predict_returns_class_indices(
        self, esm2_mlp_classification_model, sample_data
    ):
        """predict() with output_dim=2 returns argmax class indices in [0, output_dim)."""
        preds = esm2_mlp_classification_model.predict(sample_data.candidates)
        assert preds.means.shape == (len(sample_data),)
        assert np.all(preds.means >= 0)
        assert np.all(preds.means < 2)  # output_dim=2

    def test_mlp_adam_optimizer_trains(self, sample_data):
        """train() with optimizer_type='adam' completes and updates head weights."""
        config = ESM2ModelConfig(model_id=MODEL_ID)
        train_cfg = ESM2TrainConfig(
            scoring_function="linear_head",
            optimizer_type="adam",
            num_epochs=1,
            batch_size=2,
        )
        model = ESM2Model(
            name="adam_test", model_config=config, train_config=train_cfg, device="cpu"
        )
        initial_weight = model._head.weight.clone()
        model.train(sample_data)
        assert not torch.equal(initial_weight, model._head.weight)

    def test_max_grad_norm_does_not_crash_training(self, sample_data):
        """train() with max_grad_norm set completes without error."""
        config = ESM2ModelConfig(model_id=MODEL_ID)
        train_cfg = ESM2TrainConfig(
            scoring_function="linear_head",
            num_epochs=1,
            batch_size=2,
            max_grad_norm=1.0,
        )
        model = ESM2Model(
            name="grad_clip_test", model_config=config, train_config=train_cfg, device="cpu"
        )
        model.train(sample_data)
        summary = model.get_training_summary_metrics()
        assert np.isfinite(summary["final_train_loss"])

    def test_batch_size_inference_different_from_batch_size(self, sample_data):
        """predict() with batch_size_inference != batch_size produces correct shape."""
        config = ESM2ModelConfig(model_id=MODEL_ID)
        train_cfg = ESM2TrainConfig(
            freeze_backbone=True,
            scoring_function="linear_head",
            batch_size=8,
            batch_size_inference=1,
        )
        model = ESM2Model(
            name="bsi_test", model_config=config, train_config=train_cfg, device="cpu"
        )
        predictions = model.predict(sample_data.candidates)
        assert predictions.means.shape == (len(sample_data),)
        assert np.all(np.isfinite(predictions.means))

    def test_log_frequency_skips_intermediate_epochs(self, sample_data):
        """log_frequency > 1 skips intermediate epochs but always records the final one."""
        config = ESM2ModelConfig(model_id=MODEL_ID)
        train_cfg = ESM2TrainConfig(
            scoring_function="linear_head",
            num_epochs=4,
            batch_size=2,
            log_frequency=3,
        )
        model = ESM2Model(
            name="log_freq_test", model_config=config, train_config=train_cfg, device="cpu"
        )
        model.train(sample_data)
        # Epochs logged: epoch 3 (index 2, (2+1)%3==0) and epoch 4 (index 3, last)
        assert len(model.get_epoch_metrics()) == 2


class TestEmbedBatch:
    """Tests for ESM2Model._embed_batch()."""

    def test_embed_batch_returns_correct_shape(self, esm2_mlp_model, sample_data):
        """_embed_batch returns (batch_size, hidden_dim) tensor."""
        batch = esm2_mlp_model.featurise(sample_data)
        input_ids = batch["input_ids"].to(esm2_mlp_model.device)
        attention_mask = batch["attention_mask"].to(esm2_mlp_model.device)
        hidden_dim = esm2_mlp_model.esm_model.config.hidden_size
        result = esm2_mlp_model._embed_batch(input_ids, attention_mask)
        assert result.shape == (len(sample_data), hidden_dim)

    def test_embed_batch_on_correct_device(self, esm2_mlp_model, sample_data):
        """_embed_batch returns a tensor on the model's device."""
        batch = esm2_mlp_model.featurise(sample_data)
        input_ids = batch["input_ids"].to(esm2_mlp_model.device)
        attention_mask = batch["attention_mask"].to(esm2_mlp_model.device)
        result = esm2_mlp_model._embed_batch(input_ids, attention_mask)
        assert result.device == esm2_mlp_model.device

    def test_embed_batch_matches_embed_output(self, esm2_mlp_model, sample_data):
        """_embed_batch output matches the result from the public embed() method."""
        expected = esm2_mlp_model.embed(sample_data.candidates)
        batch = esm2_mlp_model.featurise(sample_data)
        input_ids = batch["input_ids"].to(esm2_mlp_model.device)
        attention_mask = batch["attention_mask"].to(esm2_mlp_model.device)
        result = esm2_mlp_model._embed_batch(input_ids, attention_mask)
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-5, atol=1e-5)


class TestSeed:
    """Tests for ESM2ModelConfig.seed reproducibility of the linear head."""

    def test_same_seed_produces_identical_head_weights(self):
        """Two linear-head models created with the same seed have identical head weights."""
        train_cfg = ESM2TrainConfig(freeze_backbone=True, scoring_function="linear_head")
        model_a = ESM2Model(
            name="seed_same_a",
            model_config=ESM2ModelConfig(model_id=MODEL_ID, seed=7),
            train_config=train_cfg,
            device="cpu",
        )
        model_b = ESM2Model(
            name="seed_same_b",
            model_config=ESM2ModelConfig(model_id=MODEL_ID, seed=7),
            train_config=train_cfg,
            device="cpu",
        )
        assert torch.equal(model_a._head.weight, model_b._head.weight)
        assert torch.equal(model_a._head.bias, model_b._head.bias)

    def test_different_seeds_produce_different_head_weights(self):
        """Two linear-head models created with different seeds have different head weights."""
        train_cfg = ESM2TrainConfig(freeze_backbone=True, scoring_function="linear_head")
        model_a = ESM2Model(
            name="seed_diff_a",
            model_config=ESM2ModelConfig(model_id=MODEL_ID, seed=0),
            train_config=train_cfg,
            device="cpu",
        )
        model_b = ESM2Model(
            name="seed_diff_b",
            model_config=ESM2ModelConfig(model_id=MODEL_ID, seed=1),
            train_config=train_cfg,
            device="cpu",
        )
        assert not torch.equal(model_a._head.weight, model_b._head.weight)


class TestSample:
    """Tests for ESM2Model.sample()."""

    def test_sample_raises_not_implemented(self, esm2_model):
        """sample() always raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            esm2_model.sample()
