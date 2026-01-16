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
from alf_core import Candidate, LabeledCandidates
from alf_tools.models.gp import (
    FeaturizerConfig,
    GPModelConfig,
    GPModelTrainer,
    GPTrainConfig,
)


@pytest.fixture
def sample_data():
    """Create sample training data.

    Returns:
        A LabeledCandidates object containing the training data.
    """
    sequences = ["ACDEFGHIKLMNPQRSTVWY"] * 10  # Simple repeated sequence
    candidates = [Candidate(data=seq, modality="sequence") for seq in sequences]
    labels = np.random.randn(10) * 0.5 + 1.0
    return LabeledCandidates(candidates, labels)


@pytest.fixture
def gp_model():
    """Create a GPModel with small settings for fast testing.

    Returns:
        A GPModel.
    """
    model_config = GPModelConfig(kernel_type="rbf", ard=False)
    train_config = GPTrainConfig(num_iterations=10, log_frequency=5)
    featurizer_config = FeaturizerConfig(featurizer_type="one_hot", flatten_one_hot=True)
    return GPModelTrainer(
        name="test_gp",
        model_config=model_config,
        train_config=train_config,
        featurizer_config=featurizer_config,
        device="cpu",
    )


class TestGPModel:
    """Test the GPModel class."""

    def test_train_and_predict(self, gp_model, sample_data):
        """Test the full training and prediction pipeline."""
        # Train should work without error
        gp_model.train(sample_data)

        # Model should be initialized
        assert gp_model.gp_model is not None
        assert gp_model.likelihood is not None

        # Should store metrics
        metrics = gp_model.get_training_summary_metrics()
        assert "final_mll" in metrics
        assert "final_train_spearman" in metrics

        # Predictions should work and include variances
        test_candidates = [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")] * 3
        predictions = gp_model.predict(test_candidates)
        assert predictions.means.shape == (3,)
        assert predictions.variances.shape == (3,)
        assert np.all(np.isfinite(predictions.means))
        assert np.all(np.isfinite(predictions.variances))
        # Variances must be non-negative
        assert np.all(predictions.variances >= 0)

    def test_train_with_validation(self, gp_model, sample_data):
        """Test training with validation data."""
        val_sequences = ["ACDEFGHIKLMNPQRSTVWY"] * 3
        val_candidates = [Candidate(data=seq, modality="sequence") for seq in val_sequences]
        val_data = LabeledCandidates(val_candidates, np.random.randn(3))

        gp_model.train(sample_data, val_data=val_data)

        metrics = gp_model.get_training_summary_metrics()
        assert "final_val_spearman" in metrics

    def test_predict_before_train_raises_error(self, gp_model):
        """Test that predicting before train raises an error."""
        candidates = [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")]
        with pytest.raises(RuntimeError, match="Model not trained"):
            gp_model.predict(candidates)

    def test_one_hot_encoding(self, gp_model):
        """Test that one-hot encoding produces correct shape and values."""
        sequences = ["ACDE", "FGHI"]
        encoded = gp_model._one_hot_encode(sequences)

        # Check shape: (batch_size, alphabet_size * seq_length) when flattened
        assert encoded.shape == (2, 20 * 4)
        # Check that encoding is binary
        assert torch.all((encoded == 0) | (encoded == 1))

    def test_one_hot_encoding_not_flattened(self):
        """Test one-hot encoding without flattening."""
        featurizer_config = FeaturizerConfig(featurizer_type="one_hot", flatten_one_hot=False)
        model_config = GPModelConfig(kernel_type="rbf")
        train_config = GPTrainConfig(num_iterations=5)
        gp_model = GPModelTrainer(
            model_config=model_config,
            train_config=train_config,
            featurizer_config=featurizer_config,
            device="cpu",
        )

        sequences = ["ACDE", "FGHI"]
        encoded = gp_model._one_hot_encode(sequences)

        # Check shape: (batch_size, alphabet_size, seq_length) when not flattened
        assert encoded.shape == (2, 20, 4)

    def test_get_hyperparameters(self, gp_model, sample_data):
        """Test that we can extract learned hyperparameters."""
        gp_model.train(sample_data)
        hyperparams = gp_model.get_hyperparameters()

        assert "noise" in hyperparams
        assert "lengthscale" in hyperparams
        assert "outputscale" in hyperparams
        assert "mean_constant" in hyperparams

        # Check that hyperparameters are reasonable
        assert hyperparams["noise"] > 0
        assert hyperparams["outputscale"] > 0

    def test_different_kernels(self, sample_data):
        """Test that different kernel types work."""
        kernel_types = ["rbf", "matern", "linear", "polynomial", "rbf_linear"]

        for kernel_type in kernel_types:
            model_config = GPModelConfig(kernel_type=kernel_type, ard=False)
            train_config = GPTrainConfig(num_iterations=5, log_frequency=10)
            featurizer_config = FeaturizerConfig(featurizer_type="one_hot", flatten_one_hot=True)
            gp_model = GPModelTrainer(
                model_config=model_config,
                train_config=train_config,
                featurizer_config=featurizer_config,
                device="cpu",
            )

            # Should train without error
            gp_model.train(sample_data)

            # Should predict without error
            test_candidates = [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")]
            predictions = gp_model.predict(test_candidates)
            assert predictions.means.shape == (1,)
            assert predictions.variances.shape == (1,)

    def test_ard_kernel(self, sample_data):
        """Test that ARD (Automatic Relevance Determination) works."""
        model_config = GPModelConfig(kernel_type="rbf", ard=True)
        train_config = GPTrainConfig(num_iterations=10)
        featurizer_config = FeaturizerConfig(featurizer_type="one_hot", flatten_one_hot=True)
        gp_model = GPModelTrainer(
            model_config=model_config,
            train_config=train_config,
            featurizer_config=featurizer_config,
            device="cpu",
        )

        gp_model.train(sample_data)
        hyperparams = gp_model.get_hyperparameters()

        # With ARD, lengthscale should be an array
        assert isinstance(hyperparams["lengthscale"], np.ndarray)
        # Should have multiple lengthscales
        assert len(hyperparams["lengthscale"]) > 1

    def test_custom_featurizer(self, sample_data):
        """Test using a custom featurization function."""

        def custom_featurizer(sequences: list[str]) -> torch.Tensor:
            # Simple custom featurizer: length of sequence
            features = torch.tensor([[len(seq)] for seq in sequences], dtype=torch.float32)
            return features

        featurizer_config = FeaturizerConfig(
            featurizer_type="custom", custom_featurizer=custom_featurizer
        )
        model_config = GPModelConfig(kernel_type="rbf", ard=False)
        train_config = GPTrainConfig(num_iterations=10)
        gp_model = GPModelTrainer(
            model_config=model_config,
            train_config=train_config,
            featurizer_config=featurizer_config,
            device="cpu",
        )

        gp_model.train(sample_data)

        test_candidates = [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")]
        predictions = gp_model.predict(test_candidates)
        assert predictions.means.shape == (1,)
        assert predictions.variances.shape == (1,)

    def test_early_stopping(self, sample_data):
        """Test that early stopping works."""
        model_config = GPModelConfig(kernel_type="rbf", ard=False)
        train_config = GPTrainConfig(
            num_iterations=100,
            early_stopping_patience=5,
            early_stopping_delta=1e-4,
            log_frequency=10,
        )
        featurizer_config = FeaturizerConfig(featurizer_type="one_hot", flatten_one_hot=True)
        gp_model = GPModelTrainer(
            model_config=model_config,
            train_config=train_config,
            featurizer_config=featurizer_config,
            device="cpu",
        )

        gp_model.train(sample_data)
        metrics = gp_model.get_training_summary_metrics()

        # Should stop before 100 iterations
        assert metrics["num_iterations"] < 100

    def test_reproducibility_with_seed(self, sample_data):
        """Test that training is reproducible when using the same seed."""

        def set_seed(seed: int):
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Create two identical models
        model_config = GPModelConfig(kernel_type="rbf", ard=False)
        train_config = GPTrainConfig(num_iterations=10)
        featurizer_config = FeaturizerConfig(featurizer_type="one_hot", flatten_one_hot=True)

        set_seed(42)
        gp_model1 = GPModelTrainer(
            name="test_gp_1",
            model_config=model_config,
            train_config=train_config,
            featurizer_config=featurizer_config,
            device="cpu",
        )
        gp_model1.train(sample_data)

        set_seed(42)
        gp_model2 = GPModelTrainer(
            name="test_gp_2",
            model_config=model_config,
            train_config=train_config,
            featurizer_config=featurizer_config,
            device="cpu",
        )
        gp_model2.train(sample_data)

        # Get predictions from both models
        test_candidates = [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")]
        pred1 = gp_model1.predict(test_candidates)
        pred2 = gp_model2.predict(test_candidates)

        # Predictions should be very close
        np.testing.assert_allclose(pred1.means, pred2.means, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(pred1.variances, pred2.variances, rtol=1e-5, atol=1e-7)

    def test_matern_kernel_smoothness(self, sample_data):
        """Test Matern kernel with different smoothness parameters."""
        nu_values = [0.5, 1.5, 2.5]

        for nu in nu_values:
            model_config = GPModelConfig(kernel_type="matern", matern_nu=nu, ard=False)
            train_config = GPTrainConfig(num_iterations=5)
            featurizer_config = FeaturizerConfig(featurizer_type="one_hot", flatten_one_hot=True)
            gp_model = GPModelTrainer(
                model_config=model_config,
                train_config=train_config,
                featurizer_config=featurizer_config,
                device="cpu",
            )

            # Should train without error
            gp_model.train(sample_data)

            # Should predict without error
            test_candidates = [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")]
            predictions = gp_model.predict(test_candidates)
            assert predictions.means.shape == (1,)

    def test_adam_optimizer(self, sample_data):
        """Test training with Adam optimizer."""
        model_config = GPModelConfig(kernel_type="rbf", ard=False)
        train_config = GPTrainConfig(optimizer_type="adam", learning_rate=0.1, num_iterations=10)
        featurizer_config = FeaturizerConfig(featurizer_type="one_hot", flatten_one_hot=True)
        gp_model = GPModelTrainer(
            model_config=model_config,
            train_config=train_config,
            featurizer_config=featurizer_config,
            device="cpu",
        )

        gp_model.train(sample_data)
        assert gp_model.gp_model is not None

    def test_lbfgs_optimizer(self, sample_data):
        """Test training with L-BFGS optimizer."""
        model_config = GPModelConfig(kernel_type="rbf", ard=False)
        train_config = GPTrainConfig(optimizer_type="lbfgs", learning_rate=0.1, num_iterations=5)
        featurizer_config = FeaturizerConfig(featurizer_type="one_hot", flatten_one_hot=True)
        gp_model = GPModelTrainer(
            model_config=model_config,
            train_config=train_config,
            featurizer_config=featurizer_config,
            device="cpu",
        )

        gp_model.train(sample_data)
        assert gp_model.gp_model is not None
