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

from alf.core.dataclasses import Candidate, LabeledCandidates
from alf.tools.models.cnn import (
    CNNModel,
    CNNModelConfig,
    CNNTrainConfig,
)


@pytest.fixture
def sample_data():
    """Create sample training data."""
    sequences = ["ACDEFGHIKLMNPQRSTVWY"] * 10  # Simple repeated sequence
    candidates = [Candidate(data=seq, modality="sequence") for seq in sequences]
    labels = np.random.randn(10) * 0.5 + 1.0
    return LabeledCandidates(candidates, labels)


@pytest.fixture
def cnn_model():
    """Create a CNNSurrogate with small settings for fast testing."""
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
        test_candidates = [
            Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")
        ] * 3
        predictions = cnn_model.predict(test_candidates)
        assert predictions.means.shape == (3,)
        assert np.all(np.isfinite(predictions.means))

    def test_train_with_validation(self, cnn_model, sample_data):
        """Test training with validation data."""
        val_sequences = ["ACDEFGHIKLMNPQRSTVWY"] * 3
        val_candidates = [
            Candidate(data=seq, modality="sequence") for seq in val_sequences
        ]
        val_data = LabeledCandidates(val_candidates, np.random.randn(3))

        cnn_model.train(sample_data, val_data=val_data)

        metrics = cnn_model.get_training_summary_metrics()
        assert "final_val_loss" in metrics
        assert "final_val_spearman" in metrics

    def test_predict_before_train_raises_error(self, cnn_model):
        """Test that predicting before train raises an error."""
        candidates = [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")]
        with pytest.raises(RuntimeError, match="Model not trained"):
            cnn_model.predict(candidates)

    def test_logger_integration(self, cnn_model, sample_data):
        """Test that logger is called during training."""
        from unittest.mock import Mock

        mock_logger = Mock()
        cnn_model.train(sample_data, logger=mock_logger)

        # Logger should be called for each epoch
        assert mock_logger.write.called
        # Check that metrics are logged with correct label
        call_args = mock_logger.write.call_args_list[0]
        assert call_args[1]["label"] == "surrogate"

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
        data1 = LabeledCandidates(
            [Candidate(data="A" * 10, modality="sequence")], np.array([1.0])
        )
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
        model_config = CNNModelConfig(
            num_filters=16, num_conv_layers=1, fc_hidden_dim=32
        )
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
