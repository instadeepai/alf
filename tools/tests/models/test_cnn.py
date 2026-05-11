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
from alf_core.utils.enums import ProblemType
from alf_tools.models.cnn import (
    CNNModel,
    CNNModelConfig,
    CNNTrainConfig,
    _apply_activation,  # noqa: PLC2701
)


@pytest.fixture
def sample_data():
    """Create sample training data.

    Returns:
        A LabelledCandidates object containing the training data.
    """
    sequences = ["ACDEFGHIKLMNPQRSTVWY"] * 10  # Simple repeated sequence
    candidates = [Candidate(data=seq, modality="sequence") for seq in sequences]
    labels = np.random.randn(10) * 0.5 + 1.0
    return LabelledCandidates(candidates, labels)


@pytest.fixture
def cnn_model():
    """Create a CNNModel with small settings for fast testing.

    Returns:
        A CNNModel.
    """
    model_config = CNNModelConfig(num_filters=16, num_conv_layers=1, fc_hidden_dim=32)
    train_config = CNNTrainConfig(batch_size=4, num_epochs=2)
    return CNNModel(
        name="test_cnn",
        model_config=model_config,
        train_config=train_config,
        device="cpu",
    )


class TestCNNModel:
    """Test the CNNModel class."""

    def test_train_and_predict(self, cnn_model, sample_data):
        """Test the full training and prediction pipeline."""
        # Train should work without error
        cnn_model.train(sample_data)

        # Model should be initialized
        assert cnn_model.model is not None

        # Should store metrics
        metrics = cnn_model.get_training_summary_metrics()
        assert "final_train_loss" in metrics
        assert "final_train_spearman" in metrics

        # Predictions should work
        test_candidates = [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")] * 3
        predictions = cnn_model.predict(test_candidates)
        assert predictions.means.shape == (3,)
        assert np.all(np.isfinite(predictions.means))

    def test_train_with_validation(self, cnn_model, sample_data):
        """Test training with validation data."""
        val_sequences = ["ACDEFGHIKLMNPQRSTVWY"] * 3
        val_candidates = [Candidate(data=seq, modality="sequence") for seq in val_sequences]
        val_data = LabelledCandidates(val_candidates, np.random.randn(3))

        cnn_model.train(sample_data, val_data=val_data)

        metrics = cnn_model.get_training_summary_metrics()
        assert "final_val_loss" in metrics
        assert "final_val_spearman" in metrics

    def test_predict_before_train_raises_error(self, cnn_model):
        """Test that predicting before train raises an error."""
        candidates = [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")]
        with pytest.raises(RuntimeError, match="Model not trained"):
            cnn_model.predict(candidates)

    def test_one_hot_encoding(self, cnn_model):
        """Test that one-hot encoding produces correct shape and values."""
        sequences = ["ACDE", "FGHI"]
        encoded = cnn_model._one_hot_encode(sequences)

        # Check shape: (batch_size, alphabet_size, seq_length)
        assert encoded.shape == (2, 20, 4)
        # Each position should sum to 1 (one-hot)
        assert encoded.sum(dim=1).allclose(encoded.new_ones(2, 4))

    def test_different_sequence_lengths_fails(self, cnn_model):
        """Test that sequences of different lengths cause issues appropriately."""
        # First train with one length
        data1 = LabelledCandidates([Candidate(data="A" * 10, modality="sequence")], np.array([1.0]))
        cnn_model.train(data1)

        # Trying to predict with different length should fail in one-hot encoding
        different_length_candidates = [Candidate(data="A" * 15, modality="sequence")]
        with pytest.raises((RuntimeError, IndexError, ValueError)):
            cnn_model.predict(different_length_candidates)

    def test_model_parameters_update(self, cnn_model, sample_data):
        """Test that model parameters are actually updated during training."""
        # train the model
        cnn_model.train(sample_data)

        # Get initial parameters
        initial_params = [p.clone() for p in cnn_model.model.parameters()]

        # Train for one more epoch
        cnn_model.train_config.num_epochs = 1
        cnn_model.train(sample_data)

        # Get updated parameters
        updated_params = list(cnn_model.model.parameters())

        # Check that at least one parameter has changed
        params_changed = False
        for initial, updated in zip(initial_params, updated_params):
            if not torch.allclose(initial, updated, atol=1e-6):
                params_changed = True
                break

        assert params_changed, "Model parameters should be updated during training"


class TestCNNModelSurrogateEpochMetrics:
    """Tests for CNNModel.get_epoch_metrics() and per-epoch recording."""

    def test_get_epoch_metrics_returns_list_of_epoch_metrics(self, cnn_model, sample_data):
        """get_epoch_metrics() should return a list of SurrogateEpochMetrics after training."""
        cnn_model.train(sample_data)
        epoch_metrics = cnn_model.get_epoch_metrics()
        assert isinstance(epoch_metrics, list)
        assert all(isinstance(em, SurrogateEpochMetrics) for em in epoch_metrics)

    def test_epoch_metrics_length_matches_num_epochs(self, cnn_model, sample_data):
        """get_epoch_metrics() length must equal num_epochs."""
        cnn_model.train(sample_data)
        assert len(cnn_model.get_epoch_metrics()) == cnn_model.train_config.num_epochs

    def test_epoch_metrics_reset_on_retrain(self, cnn_model, sample_data):
        """Calling train() twice must reset the epoch metrics list."""
        cnn_model.train(sample_data)
        cnn_model.train(sample_data)
        assert len(cnn_model.get_epoch_metrics()) == cnn_model.train_config.num_epochs

    def test_epoch_metrics_train_loss_populated(self, cnn_model, sample_data):
        """Every SurrogateEpochMetrics must have a finite train_loss."""
        cnn_model.train(sample_data)
        for em in cnn_model.get_epoch_metrics():
            assert np.isfinite(em.train_loss)

    def test_epoch_metrics_val_fields_populated_with_val_data(self, cnn_model, sample_data):
        """When val_data is provided, val_loss must be set on every SurrogateEpochMetrics."""
        val_candidates = [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")] * 3
        val_data = LabelledCandidates(val_candidates, np.random.randn(3))
        cnn_model.train(sample_data, val_data=val_data)
        for em in cnn_model.get_epoch_metrics():
            assert em.val_loss is not None
            assert np.isfinite(em.val_loss)

    def test_epoch_metrics_val_fields_none_without_val_data(self, cnn_model, sample_data):
        """Without val_data, val_loss must be None on every SurrogateEpochMetrics."""
        cnn_model.train(sample_data)
        for em in cnn_model.get_epoch_metrics():
            assert em.val_loss is None

    def test_get_epoch_metrics_before_train_returns_empty(self, cnn_model):
        """get_epoch_metrics() before any training must return an empty list."""
        assert cnn_model.get_epoch_metrics() == []


class TestCNNModelReproducibility:
    """Reproducibility tests, extracted for clarity."""

    def test_reproducibility_with_seed(self, sample_data):
        """Test that training is reproducible when using the same seed."""

        def set_seed(seed: int):
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Create two identical surrogates
        model_config = CNNModelConfig(num_filters=16, num_conv_layers=1, fc_hidden_dim=32)
        train_config = CNNTrainConfig(batch_size=4, num_epochs=2)

        set_seed(42)
        surrogate1 = CNNModel(
            name="test_cnn_1",
            model_config=model_config,
            train_config=train_config,
            device="cpu",
        )
        surrogate1.train(sample_data)

        set_seed(42)
        surrogate2 = CNNModel(
            name="test_cnn_2",
            model_config=model_config,
            train_config=train_config,
            device="cpu",
        )
        surrogate2.train(sample_data)

        # Check that model parameters are identical
        params1 = list(surrogate1.model.parameters())
        params2 = list(surrogate2.model.parameters())

        # Parameters should be identical (or very close due to floating point)
        for p1, p2 in zip(params1, params2):
            torch.testing.assert_close(p1, p2, rtol=1e-5, atol=1e-7)


class TestApplyActivation:
    """Tests for the _apply_activation module-level helper."""

    def test_regression_returns_tensor_unchanged(self):
        """Test that regression logits are returned as-is."""
        t = torch.tensor([1.0, -1.0, 0.5])
        result = _apply_activation(t, ProblemType.REGRESSION)
        assert torch.equal(result, t)

    def test_binary_returns_two_class_probs(self):
        """Test that binary logits are converted to (n, 2) probability matrix."""
        t = torch.tensor([0.0, 2.0, -2.0])
        result = _apply_activation(t, ProblemType.BINARY)
        assert result.shape == (3, 2)
        assert torch.allclose(result[:, 1], torch.sigmoid(t))
        assert torch.allclose(result.sum(dim=-1), torch.ones(3))

    def test_multiclass_applies_softmax(self):
        """Test that multiclass logits are converted to softmax probabilities."""
        t = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
        result = _apply_activation(t, ProblemType.MULTICLASS)
        assert torch.allclose(result, torch.softmax(t, dim=-1))
        assert torch.allclose(result.sum(dim=-1), torch.ones(2))


class TestEpochMetricsClassification:
    """Integration tests for epoch metric computation across problem types."""

    def test_binary_training_produces_nonempty_metrics(self):
        """Test that binary training populates accuracy and f1 in the summary."""
        model_config = CNNModelConfig(num_filters=4, num_conv_layers=1, fc_hidden_dim=8)
        train_config = CNNTrainConfig(batch_size=4, num_epochs=1)
        model = CNNModel(
            name="test_binary",
            model_config=model_config,
            train_config=train_config,
            device="cpu",
        )
        sequences = ["ACDEFGHIKLMNPQRSTVWY"] * 8
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
        data = LabelledCandidates(
            [Candidate(data=s, modality="sequence") for s in sequences], labels
        )
        model.train(data, problem_type=ProblemType.BINARY)

        summary = model.get_training_summary_metrics()
        assert "final_train_accuracy" in summary
        assert "final_train_f1" in summary

    def test_regression_one_sample_omits_pearson_and_spearman(self):
        """Test that single-sample regression omits pearson and spearman from summary."""
        model_config = CNNModelConfig(num_filters=4, num_conv_layers=1, fc_hidden_dim=8)
        train_config = CNNTrainConfig(batch_size=1, num_epochs=1)
        model = CNNModel(
            name="test_reg_single",
            model_config=model_config,
            train_config=train_config,
            device="cpu",
        )
        data = LabelledCandidates(
            [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")],
            np.array([1.0]),
        )
        model.train(data)

        summary = model.get_training_summary_metrics()
        assert "final_train_mse" in summary
        assert "final_train_pearson" not in summary
        assert "final_train_spearman" not in summary
