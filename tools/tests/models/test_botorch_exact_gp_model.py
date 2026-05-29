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

"""Tests for BoTorchGPModel class.

This module tests the BoTorch-based GP model implementation, including:
- Model initialization with various configurations
- Training with different optimizers
- Prediction capabilities
- Error handling and edge cases
- Device management (CPU/CUDA)
"""

import numpy as np
import pytest
import torch
from alf_core import Candidate, LabelledCandidates, Modality
from alf_tools.models.botorch_exact_gp_model import BoTorchGPModel


@pytest.fixture
def simple_2d_training_data():
    """Create simple 2D training data for testing.

    Returns:
        LabelledCandidates with 10 samples in 2D space.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Create 10 random 2D points in [0, 1]
    X = np.random.rand(10, 2).astype(np.float32)
    # Create labels from a simple function
    y = np.sin(X[:, 0] * 3.14159) + np.cos(X[:, 1] * 3.14159)

    candidates = [Candidate(data=x, modality=Modality.TABULAR) for x in X]

    return LabelledCandidates(candidates=candidates, labels=y)


@pytest.fixture
def test_2d_candidates():
    """Create test candidates for prediction.

    Returns:
        List of 5 test candidates in 2D space.
    """
    X_test = np.array(
        [
            [0.5, 0.5],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.3, 0.7],
            [0.7, 0.3],
        ],
        dtype=np.float32,
    )

    return [Candidate(data=x, modality=Modality.TABULAR) for x in X_test]


class TestBoTorchGPModelInitialization:
    """Test BoTorchGPModel initialization and configuration."""

    def test_default_initialization(self):
        """Test model initialization with default parameters."""
        model = BoTorchGPModel()

        assert model.normalise_inputs is True
        assert model.standardise_outputs is True
        assert model.num_iterations == 100
        assert model.learning_rate == 0.1
        assert model.optimizer == "scipy"
        assert model.max_attempts == 5
        assert model.dtype == torch.float32
        assert model.model is None
        assert model.train_X is None
        assert model.train_Y is None
        assert model._training_metrics == {"loss": [], "iteration": []}

    def test_custom_initialization(self):
        """Test model initialization with custom parameters."""
        model = BoTorchGPModel(
            normalise_inputs=False,
            standardise_outputs=False,
            num_iterations=50,
            learning_rate=0.05,
            optimizer="torch",
            max_attempts=3,
            dtype=torch.float64,
        )

        assert model.normalise_inputs is False
        assert model.standardise_outputs is False
        assert model.num_iterations == 50
        assert model.learning_rate == 0.05
        assert model.optimizer == "torch"
        assert model.max_attempts == 3
        assert model.dtype == torch.float64

    def test_device_auto_detection(self):
        """Test automatic device detection."""
        model = BoTorchGPModel()

        # Should auto-detect device
        assert model.device is not None
        assert isinstance(model.device, torch.device)

    def test_explicit_cpu_device(self):
        """Test explicit CPU device specification."""
        model = BoTorchGPModel(device="cpu")

        assert model.device == torch.device("cpu")

    def test_invalid_optimizer_raises_error(self):
        """Test that invalid optimizer raises ValueError."""
        with pytest.raises(ValueError, match="optimizer must be 'scipy' or 'torch'"):
            BoTorchGPModel(optimizer="invalid")


class TestBoTorchGPModelFeaturisation:
    """Test featurisation method."""

    def test_featurise_valid_candidates(self):
        """Test featurisation with valid candidate list."""
        model = BoTorchGPModel()

        candidates = [
            Candidate(data=np.array([0.5, 0.5], dtype=np.float32), modality=Modality.TABULAR),
            Candidate(data=np.array([0.3, 0.7], dtype=np.float32), modality=Modality.TABULAR),
            Candidate(data=np.array([0.8, 0.2], dtype=np.float32), modality=Modality.TABULAR),
        ]

        features = model.featurise(candidates)

        assert isinstance(features, torch.Tensor)
        assert features.shape == (3, 2)
        assert features.dtype == torch.float32

    def test_featurise_single_candidate(self):
        """Test featurisation with single candidate."""
        model = BoTorchGPModel()

        candidates = [
            Candidate(data=np.array([0.5, 0.5], dtype=np.float32), modality=Modality.TABULAR),
        ]

        features = model.featurise(candidates)

        assert features.shape == (1, 2)


class TestBoTorchGPModelTraining:
    """Test training functionality."""

    def test_train_with_valid_data_scipy(self, simple_2d_training_data):
        """Test training with valid data using scipy optimizer."""
        model = BoTorchGPModel(num_iterations=50, optimizer="scipy")

        # Training should complete without error
        model.train(simple_2d_training_data)

        # Model should be initialized
        assert model.model is not None
        assert model.train_X is not None
        assert model.train_Y is not None
        assert model.train_X.shape == (10, 2)
        assert model.train_Y.shape == (10, 1)

        # Training metrics should be recorded
        assert len(model._training_metrics["loss"]) > 0
        assert len(model._training_metrics["iteration"]) > 0

    def test_train_with_valid_data_torch(self, simple_2d_training_data):
        """Test training with valid data using torch optimizer."""
        model = BoTorchGPModel(num_iterations=20, optimizer="torch", learning_rate=0.1)

        # Training should complete without error
        model.train(simple_2d_training_data)

        # Model should be initialized
        assert model.model is not None
        assert model.train_X is not None
        assert model.train_Y is not None

    def test_train_with_validation_data(self, simple_2d_training_data):
        """Test training with validation data (for API compatibility)."""
        model = BoTorchGPModel(num_iterations=20)

        # Create validation data
        X_val = np.random.rand(3, 2).astype(np.float32)
        y_val = np.sin(X_val[:, 0] * 3.14159)
        val_candidates = [Candidate(data=x, modality=Modality.TABULAR) for x in X_val]
        val_data = LabelledCandidates(candidates=val_candidates, labels=y_val)

        # Training should work even though val_data is not used
        model.train(simple_2d_training_data, val_data=val_data)

        assert model.model is not None

    def test_train_with_empty_data_raises_error(self):
        """Test that training with empty data raises ValueError."""
        model = BoTorchGPModel()

        empty_data = LabelledCandidates(candidates=[], labels=np.array([]))

        with pytest.raises(ValueError, match="Training data cannot be empty"):
            model.train(empty_data)

    def test_train_updates_metrics(self, simple_2d_training_data):
        """Test that training updates metrics correctly."""
        model = BoTorchGPModel(num_iterations=30)

        model.train(simple_2d_training_data)

        # Check metrics were recorded
        metrics = model.get_training_summary_metrics()
        assert "final_loss" in metrics
        assert isinstance(metrics["final_loss"], float)
        assert np.isfinite(metrics["final_loss"])

    def test_multiple_train_calls(self, simple_2d_training_data):
        """Test that model can be retrained."""
        model = BoTorchGPModel(num_iterations=20)

        # First training
        model.train(simple_2d_training_data)

        # Second training (should work)
        model.train(simple_2d_training_data)

        # Metrics reset on each train() call, so only 1 entry
        assert len(model._training_metrics["loss"]) == 1

    def test_train_with_1d_output(self):
        """Test training with 1D output (automatically reshaped to 2D)."""
        model = BoTorchGPModel(num_iterations=20)

        # Create simple 1D data
        X = np.random.rand(10, 1).astype(np.float32)
        y = np.sin(X[:, 0] * 3.14159)

        candidates = [Candidate(data=x, modality=Modality.TABULAR) for x in X]
        train_data = LabelledCandidates(candidates=candidates, labels=y)

        # Should train without error
        model.train(train_data)

        assert model.model is not None
        assert model.train_Y.shape == (10, 1)


class TestBoTorchGPModelPrediction:
    """Test prediction functionality."""

    def test_predict_after_training(self, simple_2d_training_data, test_2d_candidates):
        """Test predictions after training."""
        model = BoTorchGPModel(num_iterations=30)
        model.train(simple_2d_training_data)

        predictions = model.predict(test_2d_candidates)

        # Check prediction shape and content
        assert predictions.means is not None
        assert predictions.variances is not None
        assert predictions.means.shape == (5,)
        assert predictions.variances.shape == (5,)

        # Check values are finite
        assert np.all(np.isfinite(predictions.means))
        assert np.all(np.isfinite(predictions.variances))

        # Variances should be non-negative
        assert np.all(predictions.variances >= 0)

    def test_predict_before_training_raises_error(self, test_2d_candidates):
        """Test that predicting before training raises RuntimeError."""
        model = BoTorchGPModel()

        with pytest.raises(RuntimeError, match="Model must be trained before making predictions"):
            model.predict(test_2d_candidates)

    def test_predict_with_empty_candidates_raises_error(self, simple_2d_training_data):
        """Test that predicting with empty candidates raises ValueError."""
        model = BoTorchGPModel(num_iterations=20)
        model.train(simple_2d_training_data)

        with pytest.raises(ValueError, match="Candidates list cannot be empty"):
            model.predict([])

    def test_predict_dimension_mismatch_raises_error(self, simple_2d_training_data):
        """Test that dimension mismatch raises ValueError."""
        model = BoTorchGPModel(num_iterations=20)
        model.train(simple_2d_training_data)

        # Try to predict with 3D candidates (trained on 2D)
        wrong_dim_candidates = [
            Candidate(data=np.array([0.5, 0.5, 0.5], dtype=np.float32), modality=Modality.TABULAR),
        ]

        with pytest.raises(ValueError, match="Input dimension mismatch"):
            model.predict(wrong_dim_candidates)

    def test_predict_single_point(self, simple_2d_training_data):
        """Test prediction on a single point."""
        model = BoTorchGPModel(num_iterations=30)
        model.train(simple_2d_training_data)

        single_candidate = [
            Candidate(data=np.array([0.5, 0.5], dtype=np.float32), modality=Modality.TABULAR),
        ]

        predictions = model.predict(single_candidate)

        assert predictions.means.shape == (1,)
        assert predictions.variances.shape == (1,)

    def test_predictions_on_training_points(self, simple_2d_training_data):
        """Test that predictions on training points have low variance."""
        model = BoTorchGPModel(num_iterations=50)
        model.train(simple_2d_training_data)

        # Predict on first training point
        train_candidates = [simple_2d_training_data.candidates[0]]
        predictions = model.predict(train_candidates)

        # Variance should be very small at training points
        # (not exactly zero due to noise, but should be small)
        assert predictions.variances[0] < 0.1

    def test_predict_returns_numpy_arrays(self, simple_2d_training_data, test_2d_candidates):
        """Test that predictions return numpy arrays, not tensors."""
        model = BoTorchGPModel(num_iterations=20)
        model.train(simple_2d_training_data)

        predictions = model.predict(test_2d_candidates)

        assert isinstance(predictions.means, np.ndarray)
        assert isinstance(predictions.variances, np.ndarray)


class TestBoTorchGPModelMetrics:
    """Test training metrics functionality."""

    def test_get_training_summary_metrics_after_training(self, simple_2d_training_data):
        """Test getting summary metrics after training."""
        model = BoTorchGPModel(num_iterations=30)
        model.train(simple_2d_training_data)

        metrics = model.get_training_summary_metrics()

        assert "final_loss" in metrics
        assert isinstance(metrics["final_loss"], float)
        assert np.isfinite(metrics["final_loss"])

    def test_get_training_summary_metrics_before_training(self):
        """Test getting summary metrics before training returns empty dict."""
        model = BoTorchGPModel()

        metrics = model.get_training_summary_metrics()

        assert metrics == {}

    def test_metrics_updated_after_multiple_trainings(self, simple_2d_training_data):
        """Test that metrics reset and reflect only the most recent training."""
        model = BoTorchGPModel(num_iterations=20)

        # Train twice
        model.train(simple_2d_training_data)
        model.train(simple_2d_training_data)

        # Metrics reset on each train() call, so only 1 entry
        assert len(model._training_metrics["loss"]) == 1
        assert len(model._training_metrics["iteration"]) == 1


class TestBoTorchGPModelSample:
    """Test sample method (should raise NotImplementedError)."""

    def test_sample_raises_not_implemented_error(self):
        """Test that sample method raises NotImplementedError."""
        model = BoTorchGPModel()

        with pytest.raises(
            NotImplementedError,
            match="BoTorchGPModel is a discriminative model and does not support sampling",
        ):
            model.sample()

    def test_sample_with_condition_raises_not_implemented_error(self):
        """Test that sample method with condition raises NotImplementedError."""
        model = BoTorchGPModel()

        with pytest.raises(NotImplementedError):
            model.sample(condition="some_condition")


class TestBoTorchGPModelEdgeCases:
    """Test edge cases and special scenarios."""

    def test_train_with_single_sample(self):
        """Test training with just one sample (edge case)."""
        model = BoTorchGPModel(num_iterations=10)

        # Single training point
        X = np.array([[0.5, 0.5]], dtype=np.float32)
        y = np.array([1.0])

        candidates = [Candidate(data=X[0], modality=Modality.TABULAR)]
        train_data = LabelledCandidates(candidates=candidates, labels=y)

        # Should train without error (though not very useful)
        model.train(train_data)

        assert model.model is not None

    def test_train_with_constant_outputs(self):
        """Test training when all outputs are the same."""
        model = BoTorchGPModel(num_iterations=20)

        # All outputs are 1.0
        X = np.random.rand(10, 2).astype(np.float32)
        y = np.ones(10)

        candidates = [Candidate(data=x, modality=Modality.TABULAR) for x in X]
        train_data = LabelledCandidates(candidates=candidates, labels=y)

        # Should handle this case
        model.train(train_data)

        predictions = model.predict(candidates[:3])

        # Predictions should be close to 1.0
        assert np.allclose(predictions.means, 1.0, atol=0.5)

    def test_high_dimensional_input(self):
        """Test with high-dimensional input."""
        model = BoTorchGPModel(num_iterations=20)

        # 10D input
        X = np.random.rand(20, 10).astype(np.float32)
        y = np.sum(X, axis=1)  # Simple sum function

        candidates = [Candidate(data=x, modality=Modality.TABULAR) for x in X]
        train_data = LabelledCandidates(candidates=candidates, labels=y)

        # Should train without error
        model.train(train_data)

        # Should predict without error
        predictions = model.predict(candidates[:5])
        assert predictions.means.shape == (5,)

    def test_negative_labels(self):
        """Test training with negative labels."""
        model = BoTorchGPModel(num_iterations=20)

        X = np.random.rand(10, 2).astype(np.float32)
        y = -np.ones(10) * 5.0  # All negative

        candidates = [Candidate(data=x, modality=Modality.TABULAR) for x in X]
        train_data = LabelledCandidates(candidates=candidates, labels=y)

        # Should handle negative values
        model.train(train_data)

        predictions = model.predict(candidates[:3])

        # Predictions should be negative
        assert np.all(predictions.means < 0)


class TestBoTorchGPModelReproducibility:
    """Test reproducibility with random seeds."""

    def test_reproducible_training_with_seed(self):
        """Test that training is reproducible with same seed."""

        def train_with_seed(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)

            X = np.random.rand(10, 2).astype(np.float32)
            y = np.sin(X[:, 0] * 3.14159)

            candidates = [Candidate(data=x, modality=Modality.TABULAR) for x in X]
            train_data = LabelledCandidates(candidates=candidates, labels=y)

            model = BoTorchGPModel(num_iterations=30)
            model.train(train_data)

            # Make predictions
            test_candidates = [
                Candidate(data=np.array([0.5, 0.5], dtype=np.float32), modality=Modality.TABULAR)
            ]
            predictions = model.predict(test_candidates)

            return predictions.means[0], predictions.variances[0]

        # Train twice with same seed
        mean1, var1 = train_with_seed(42)
        mean2, var2 = train_with_seed(42)

        # Results should be very close (allowing for small numerical differences)
        assert np.allclose(mean1, mean2, rtol=1e-3)
        assert np.allclose(var1, var2, rtol=1e-3)


class TestBoTorchGPModelIntegration:
    """Integration tests with real scenarios."""

    def test_full_pipeline_scipy_optimizer(self):
        """Test complete pipeline: initialize -> train -> predict with scipy."""
        torch.manual_seed(123)
        np.random.seed(123)

        # Create model
        model = BoTorchGPModel(
            num_iterations=40,
            optimizer="scipy",
            max_attempts=3,
        )

        # Create training data
        X_train = np.random.rand(15, 2).astype(np.float32)
        y_train = np.sin(X_train[:, 0] * 3.14159) + np.cos(X_train[:, 1] * 3.14159)

        train_candidates = [Candidate(data=x, modality=Modality.TABULAR) for x in X_train]
        train_data = LabelledCandidates(candidates=train_candidates, labels=y_train)

        # Train
        model.train(train_data)

        # Create test data
        X_test = np.random.rand(5, 2).astype(np.float32)
        test_candidates = [Candidate(data=x, modality=Modality.TABULAR) for x in X_test]

        # Predict
        predictions = model.predict(test_candidates)

        # Verify results
        assert predictions.means.shape == (5,)
        assert predictions.variances.shape == (5,)
        assert np.all(np.isfinite(predictions.means))
        assert np.all(np.isfinite(predictions.variances))
        assert np.all(predictions.variances > 0)

        # Get metrics
        metrics = model.get_training_summary_metrics()
        assert "final_loss" in metrics

    def test_full_pipeline_torch_optimizer(self):
        """Test complete pipeline: initialize -> train -> predict with torch."""
        torch.manual_seed(456)
        np.random.seed(456)

        # Create model
        model = BoTorchGPModel(
            num_iterations=30,
            optimizer="torch",
            learning_rate=0.1,
            max_attempts=3,
        )

        # Create training data
        X_train = np.random.rand(15, 2).astype(np.float32)
        y_train = np.sin(X_train[:, 0] * 3.14159) + np.cos(X_train[:, 1] * 3.14159)

        train_candidates = [Candidate(data=x, modality=Modality.TABULAR) for x in X_train]
        train_data = LabelledCandidates(candidates=train_candidates, labels=y_train)

        # Train
        model.train(train_data)

        # Create test data
        X_test = np.random.rand(5, 2).astype(np.float32)
        test_candidates = [Candidate(data=x, modality=Modality.TABULAR) for x in X_test]

        # Predict
        predictions = model.predict(test_candidates)

        # Verify results
        assert predictions.means.shape == (5,)
        assert predictions.variances.shape == (5,)
        assert np.all(np.isfinite(predictions.means))
        assert np.all(np.isfinite(predictions.variances))

    def test_branin_function_optimization(self, branin_dataset):
        """Test on real Branin synthetic dataset."""
        model = BoTorchGPModel(num_iterations=50, optimizer="scipy")

        # Train on dataset
        model.train(branin_dataset.train_dataset)

        # Predict on test points
        test_candidates = branin_dataset.test_dataset.candidates[:10]
        predictions = model.predict(test_candidates)

        # Check predictions are reasonable
        assert predictions.means.shape == (10,)
        assert predictions.variances.shape == (10,)
        assert np.all(np.isfinite(predictions.means))
        assert np.all(predictions.variances > 0)

        # Get metrics
        metrics = model.get_training_summary_metrics()
        assert "final_loss" in metrics
        assert np.isfinite(metrics["final_loss"])


class TestBoTorchGPModelKernelTypes:
    """Tests for kernel type selection, ARD, and Hvarfner priors."""

    def test_matern_kernel_trains_without_error(self, simple_2d_training_data, test_2d_candidates):
        """Training with kernel_type='matern' completes without error."""
        model = BoTorchGPModel(kernel_type="matern", num_iterations=20)
        model.train(simple_2d_training_data)
        preds = model.predict(test_2d_candidates)
        assert preds.means.shape == (5,)
        assert np.all(np.isfinite(preds.means))

    def test_rbf_kernel_trains_without_error(self, simple_2d_training_data, test_2d_candidates):
        """Training with kernel_type='rbf' (explicit) completes without error."""
        model = BoTorchGPModel(kernel_type="rbf", num_iterations=20)
        model.train(simple_2d_training_data)
        preds = model.predict(test_2d_candidates)
        assert preds.means.shape == (5,)
        assert np.all(np.isfinite(preds.means))

    def test_ard_kernel_trains_without_error(self, simple_2d_training_data, test_2d_candidates):
        """Training with use_ard=True completes without error."""
        model = BoTorchGPModel(use_ard=True, num_iterations=20)
        model.train(simple_2d_training_data)
        preds = model.predict(test_2d_candidates)
        assert preds.means.shape == (5,)
        assert np.all(np.isfinite(preds.means))

    def test_invalid_kernel_type_raises_value_error(self, simple_2d_training_data):
        """Invalid kernel_type raises ValueError on train(), not silent fallback."""
        model = BoTorchGPModel(num_iterations=10)
        model.kernel_type = "invalid_kernel"  # Override after init
        with pytest.raises(ValueError, match="Invalid kernel_type"):
            model.train(simple_2d_training_data)

    def test_default_kernel_has_hvarfner_priors(self, simple_2d_training_data):
        """Default kernel (kernel_type=None) registers Hvarfner lengthscale prior."""
        model = BoTorchGPModel(kernel_type=None, num_iterations=20)
        model.train(simple_2d_training_data)
        # The covar_module's base kernel should have a lengthscale_prior registered
        base_kernel = model.model.covar_module.base_kernel
        assert hasattr(base_kernel, "lengthscale_prior")
        assert base_kernel.lengthscale_prior is not None

    def test_rbf_kernel_has_hvarfner_priors(self, simple_2d_training_data):
        """Explicit 'rbf' kernel also registers Hvarfner lengthscale prior."""
        model = BoTorchGPModel(kernel_type="rbf", num_iterations=20)
        model.train(simple_2d_training_data)
        base_kernel = model.model.covar_module.base_kernel
        assert hasattr(base_kernel, "lengthscale_prior")
        assert base_kernel.lengthscale_prior is not None

    def test_botorch_model_property_raises_before_training(self):
        """botorch_model property raises RuntimeError if model is not trained."""
        model = BoTorchGPModel()
        with pytest.raises(RuntimeError, match="Model must be trained"):
            _ = model.botorch_model
