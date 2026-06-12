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
from alf_core.dataset.base_dataset import BaseDataset, BaseDatasetConfig
from alf_core.utils.enums import ProblemType
from alf_tools.models.cnn import (
    CNNModel,
    CNNModelConfig,
    CNNTrainConfig,
    _apply_activation,  # noqa: PLC2701
)


def _make_dataset(labels: np.ndarray, problem_type: ProblemType) -> BaseDataset:
    """Create a minimal BaseDataset for model setup in tests.

    Sets _raw_dataset directly so determine_num_classes() works without a full
    dataset load/split cycle.

    Returns:
        A BaseDataset with _raw_dataset pre-populated from the given labels.
    """

    class _TestDataset(BaseDataset):
        def load_dataset(self) -> LabelledCandidates:
            candidates = [
                Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence") for _ in labels
            ]
            return LabelledCandidates(candidates=candidates, labels=labels)

    config = BaseDatasetConfig(
        name="test",
        modality="sequence",
        seed=0,
        train_ratio=0.6,
        validation_frac=0.2,
        test_ratio=0.2,
        problem_type=problem_type,
    )
    dataset = _TestDataset(config)
    dataset._raw_dataset = dataset.load_dataset()
    dataset.num_classes = dataset.determine_num_classes()
    return dataset


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
def val_data():
    """Create sample validation data.

    Returns:
        A LabelledCandidates object containing the validation data.
    """
    sequences = ["ACDEFGHIKLMNPQRSTVWY"] * 2  # Simple repeated sequence
    candidates = [Candidate(data=seq, modality="sequence") for seq in sequences]
    labels = np.random.randn(2) * 0.5 + 1.0
    return LabelledCandidates(candidates, labels)


@pytest.fixture
def empty_data():
    """Create empty sample data for validation testing.

    Returns:
        A LabelledCandidates object containing the empty data.
    """
    candidates = []
    labels = np.array([])
    return LabelledCandidates(candidates, labels)


@pytest.fixture
def cnn_model(sample_data):
    """Create a CNNModel with small settings for fast testing.

    Returns:
        A CNNModel configured and set up for regression.
    """
    model_config = CNNModelConfig(num_filters=16, num_conv_layers=1, fc_hidden_dim=32)
    train_config = CNNTrainConfig(batch_size=4, num_epochs=2)
    model = CNNModel(
        model_config=model_config,
        name="test_cnn",
        train_config=train_config,
        device="cpu",
    )
    model.setup(_make_dataset(sample_data.labels, ProblemType.REGRESSION))
    return model


class TestCNNModel:
    """Test the CNNModel class."""

    def test_train_and_predict(self, cnn_model, sample_data, val_data):
        """Test the full training and prediction pipeline."""
        # Train should work without error
        cnn_model.train(sample_data, val_data)

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

    def test_train_with_validation(self, cnn_model, sample_data, val_data):
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
        val_data1 = LabelledCandidates(
            [Candidate(data="A" * 10, modality="sequence")], np.array([1.0])
        )
        cnn_model.train(data1, val_data=val_data1)

        # Trying to predict with different length should fail in one-hot encoding
        different_length_candidates = [Candidate(data="A" * 15, modality="sequence")]
        with pytest.raises((RuntimeError, IndexError, ValueError)):
            cnn_model.predict(different_length_candidates)

    def test_model_parameters_update(self, cnn_model, sample_data, val_data):
        """Test that model parameters are actually updated during training."""
        # train the model
        cnn_model.train(sample_data, val_data=val_data)

        # Get initial parameters
        initial_params = [p.clone() for p in cnn_model.model.parameters()]

        # Train for one more epoch
        cnn_model.train_config.num_epochs = 1
        cnn_model.train(sample_data, val_data=val_data)

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

    def test_get_epoch_metrics_returns_list_of_epoch_metrics(
        self, cnn_model, sample_data, val_data
    ):
        """get_epoch_metrics() should return a list of SurrogateEpochMetrics after training."""
        cnn_model.train(sample_data, val_data=val_data)
        epoch_metrics = cnn_model.get_epoch_metrics()
        assert isinstance(epoch_metrics, list)
        assert all(isinstance(em, SurrogateEpochMetrics) for em in epoch_metrics)

    def test_epoch_metrics_length_matches_num_epochs(self, cnn_model, sample_data, val_data):
        """get_epoch_metrics() length must equal num_epochs."""
        cnn_model.train(sample_data, val_data=val_data)
        assert len(cnn_model.get_epoch_metrics()) == cnn_model.train_config.num_epochs

    def test_epoch_metrics_reset_on_retrain(self, cnn_model, sample_data, val_data):
        """Calling train() twice must reset the epoch metrics list."""
        cnn_model.train(sample_data, val_data=val_data)
        cnn_model.train(sample_data, val_data=val_data)
        assert len(cnn_model.get_epoch_metrics()) == cnn_model.train_config.num_epochs

    def test_epoch_metrics_train_loss_populated(self, cnn_model, sample_data, val_data):
        """Every SurrogateEpochMetrics must have a finite train_loss."""
        cnn_model.train(sample_data, val_data=val_data)
        for em in cnn_model.get_epoch_metrics():
            assert np.isfinite(em.train_loss)

    def test_epoch_metrics_val_fields_populated_with_val_data(
        self, cnn_model, sample_data, val_data
    ):
        """When val_data is provided, val_loss must be set on every SurrogateEpochMetrics."""
        cnn_model.train(sample_data, val_data=val_data)
        for em in cnn_model.get_epoch_metrics():
            assert em.val_loss is not None
            assert np.isfinite(em.val_loss)

    def test_epoch_metrics_val_fields_none_without_val_data(
        self, cnn_model, sample_data, empty_data
    ):
        """Without val_data, val_loss must be None on every SurrogateEpochMetrics."""
        cnn_model.train(sample_data, val_data=empty_data)
        for em in cnn_model.get_epoch_metrics():
            assert em.val_loss is None

    def test_train_with_val_data_none(self, cnn_model, sample_data):
        """Passing val_data=None must complete without error and produce no val_loss."""
        cnn_model.train(sample_data, val_data=None)
        for em in cnn_model.get_epoch_metrics():
            assert em.val_loss is None

    def test_get_epoch_metrics_before_train_returns_empty(self, cnn_model):
        """get_epoch_metrics() before any training must return an empty list."""
        assert cnn_model.get_epoch_metrics() == []


class TestCNNModelReproducibility:
    """Reproducibility tests, extracted for clarity."""

    def test_reproducibility_with_seed(self, sample_data, val_data):
        """Test that training is reproducible when using the same seed."""

        def set_seed(seed: int):
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        model_config = CNNModelConfig(num_filters=16, num_conv_layers=1, fc_hidden_dim=32)
        train_config = CNNTrainConfig(batch_size=4, num_epochs=2)
        dataset = _make_dataset(sample_data.labels, ProblemType.REGRESSION)

        set_seed(42)
        surrogate1 = CNNModel(
            model_config=model_config,
            name="test_cnn_1",
            train_config=train_config,
            device="cpu",
        )
        surrogate1.setup(dataset)
        surrogate1.train(sample_data, val_data=val_data)

        set_seed(42)
        surrogate2 = CNNModel(
            model_config=model_config,
            name="test_cnn_2",
            train_config=train_config,
            device="cpu",
        )
        surrogate2.setup(dataset)
        surrogate2.train(sample_data, val_data=val_data)

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
            model_config=model_config,
            name="test_binary",
            train_config=train_config,
            device="cpu",
        )
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
        model.setup(_make_dataset(labels, ProblemType.BINARY))
        sequences = ["ACDEFGHIKLMNPQRSTVWY"] * 8
        data = LabelledCandidates(
            [Candidate(data=s, modality="sequence") for s in sequences], labels
        )
        val_data = LabelledCandidates(
            [Candidate(data=s, modality="sequence") for s in sequences[:2]], labels[:2]
        )
        model.train(data, val_data=val_data)

        summary = model.get_training_summary_metrics()
        assert "final_train_accuracy" in summary
        assert "final_train_f1" in summary

    def test_regression_one_sample_omits_pearson_and_spearman(self):
        """Test that single-sample regression omits pearson and spearman from summary."""
        model_config = CNNModelConfig(num_filters=4, num_conv_layers=1, fc_hidden_dim=8)
        train_config = CNNTrainConfig(batch_size=1, num_epochs=1)
        model = CNNModel(
            model_config=model_config,
            name="test_reg_single",
            train_config=train_config,
            device="cpu",
        )
        labels = np.array([1.0])
        model.setup(_make_dataset(labels, ProblemType.REGRESSION))
        data = LabelledCandidates(
            [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")],
            labels,
        )
        val_data = LabelledCandidates(
            [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")],
            np.array([1.0]),
        )
        model.train(data, val_data=val_data)

        summary = model.get_training_summary_metrics()
        assert "final_train_mse" in summary
        assert "final_train_pearson" not in summary
        assert "final_train_spearman" not in summary


class TestCNNLabelDtype:
    """Tests for label_dtype resolution in CNNModel."""

    def test_default_label_dtype_is_float32(self, sample_data):
        """Without explicit label_dtype, label tensors must be float32."""
        model = CNNModel(
            train_config=CNNTrainConfig(num_epochs=1),
            device="cpu",
        )
        train_loader, _ = model._prepare_train_data(sample_data, None)
        _, batch_y = next(iter(train_loader))
        assert batch_y.dtype == torch.float32

    def test_explicit_label_dtype_override_is_used(self, sample_data):
        """When label_dtype=torch.float64 is set, label tensors must be float64."""
        model = CNNModel(
            train_config=CNNTrainConfig(num_epochs=1, label_dtype=torch.float64),
            device="cpu",
        )
        train_loader, _ = model._prepare_train_data(sample_data, None)
        _, batch_y = next(iter(train_loader))
        assert batch_y.dtype == torch.float64

    def test_val_loader_label_dtype_matches_train(self, sample_data):
        """Val label tensors must use the same resolved dtype as train labels."""
        val_candidates = [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")] * 3
        val_data = LabelledCandidates(val_candidates, np.random.randn(3))
        model = CNNModel(
            train_config=CNNTrainConfig(num_epochs=1, label_dtype=torch.float64),
            device="cpu",
        )
        _, val_loader = model._prepare_train_data(sample_data, val_data)
        _, val_batch_y = next(iter(val_loader))
        assert val_batch_y.dtype == torch.float64


class TestCNNNormalisation:
    """Tests for CNNModel input normalisation and output standardisation."""

    def test_cnn_train_config_default_no_input_normalisation(self) -> None:
        """CNNTrainConfig must default normalise_inputs_strategy to None."""
        config = CNNTrainConfig()
        assert config.normalise_inputs_strategy is None

    def test_input_normalisation_enabled(self, sample_data, val_data):
        """An explicit 'zscore' strategy must run without error."""
        model = CNNModel(
            train_config=CNNTrainConfig(num_epochs=2, normalise_inputs_strategy="zscore"),
            device="cpu",
        )
        model.setup(_make_dataset(sample_data.labels, ProblemType.REGRESSION))
        model.train(sample_data, val_data=val_data)
        assert model._input_transform is not None
        assert model._input_transform.is_fitted

        predictions = model.predict(sample_data.candidates)
        assert np.all(np.isfinite(predictions.means))

    def test_input_normalisation_disabled(self, sample_data, val_data):
        """A None strategy leaves _input_transform as None."""
        model = CNNModel(
            train_config=CNNTrainConfig(num_epochs=2, normalise_inputs_strategy=None),
            device="cpu",
        )
        model.setup(_make_dataset(sample_data.labels, ProblemType.REGRESSION))
        model.train(sample_data, val_data=val_data)
        assert model._input_transform is None

        predictions = model.predict(sample_data.candidates)
        assert np.all(np.isfinite(predictions.means))

    def test_output_standardisation_enabled(self, sample_data, val_data):
        """standardise_outputs=True trains in standardised space; predict returns original scale."""
        model = CNNModel(
            train_config=CNNTrainConfig(num_epochs=2, standardise_outputs=True),
            device="cpu",
        )
        model.setup(_make_dataset(sample_data.labels, ProblemType.REGRESSION))
        model.train(sample_data, val_data=val_data)
        assert model._output_standardiser is not None
        assert model._output_standardiser.is_fitted

        predictions = model.predict(sample_data.candidates)
        assert np.all(np.isfinite(predictions.means))
        # Predictions must be in original label scale (mean ≈ label mean, not ~0 from Z-score space)
        label_mean = sample_data.labels.mean()
        label_std = sample_data.labels.std()
        assert np.abs(predictions.means.mean() - label_mean) < label_std * 5

    def test_val_data_with_standardisation(self, sample_data, val_data):
        """Val metrics are in original label scale when standardise_outputs=True."""
        train_data = LabelledCandidates(sample_data.candidates[:6], sample_data.labels[:6])
        model = CNNModel(
            train_config=CNNTrainConfig(num_epochs=2, standardise_outputs=True),
            device="cpu",
        )
        model.setup(_make_dataset(sample_data.labels, ProblemType.REGRESSION))
        model.train(train_data, val_data=val_data)

        history = model.get_epoch_metrics()
        assert len(history) == 2
        for epoch_metrics in history:
            assert epoch_metrics.val_loss is not None
            # val_mse (if present) should be in original label scale
            if "val_mse" in epoch_metrics.additional_metrics:
                val_mse = epoch_metrics.additional_metrics["val_mse"]
                label_range = float(np.ptp(sample_data.labels))
                # MSE in original space: at most (label_range)^2 * 10 (loose upper bound)
                assert val_mse < (label_range**2) * 10

    def test_standardise_outputs_train_and_val_metrics_are_inverse_transformed(self):
        """Both final_train_mse and final_val_mse must be in original label scale.

        Labels have std=50.  In standardised (Z-score) space the CNN predicts near 0
        and MSE against unit-variance targets ≈ 1.  After correct inverse-transform
        both predictions and targets are in original space, so MSE ≈ var(labels) ≈ 2500.
        A threshold of 100 cleanly separates the two cases for both train and val metrics.
        """
        rng = np.random.default_rng(0)
        n_train, n_val = 10, 4
        label_std = 50.0
        train_labels = rng.standard_normal(n_train) * label_std
        val_labels = rng.standard_normal(n_val) * label_std

        train_data = LabelledCandidates(
            [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")] * n_train,
            train_labels,
        )
        val_data = LabelledCandidates(
            [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")] * n_val,
            val_labels,
        )

        model = CNNModel(
            train_config=CNNTrainConfig(num_epochs=5, standardise_outputs=True),
            device="cpu",
        )
        model.setup(_make_dataset(train_labels, ProblemType.REGRESSION))
        model.train(train_data, val_data=val_data)
        metrics = model.get_training_summary_metrics()

        # Correct (original scale): CNN converges toward predicting ≈ label mean in
        # original space; MSE = E[(pred - target)²] ≈ var(labels) ≈ 2500.
        # Bug (standardised scale): predictions ≈ 0 in Z-score space, targets ~ N(0,1)
        # → MSE ≈ 1.  Threshold of 100 cleanly separates both cases.
        assert "final_train_mse" in metrics
        assert "final_val_mse" in metrics
        assert metrics["final_train_mse"] > 100, (
            f"final_train_mse={metrics['final_train_mse']:.3f} — "
            "train metrics appear to be in standardised space, not original label scale"
        )
        assert metrics["final_val_mse"] > 100, (
            f"final_val_mse={metrics['final_val_mse']:.3f} — "
            "val metrics appear to be in standardised space, not original label scale"
        )

    def test_standardise_outputs_raises_for_binary_classification(self):
        """standardise_outputs=True must raise ValueError for BINARY classification."""
        binary_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
        model = CNNModel(
            train_config=CNNTrainConfig(num_epochs=2, standardise_outputs=True),
            device="cpu",
        )
        model.setup(_make_dataset(binary_labels, ProblemType.BINARY))
        candidates = [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")] * 10
        train_data = LabelledCandidates(candidates, binary_labels)

        with pytest.raises(ValueError, match="standardise_outputs"):
            model.train(train_data, val_data=LabelledCandidates([], np.array([])))

    def test_standardise_outputs_raises_for_multiclass_classification(self):
        """standardise_outputs=True must raise ValueError for MULTICLASS classification."""
        multiclass_labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        model = CNNModel(
            train_config=CNNTrainConfig(num_epochs=2, standardise_outputs=True),
            device="cpu",
        )
        model.setup(_make_dataset(multiclass_labels, ProblemType.MULTICLASS))
        candidates = [Candidate(data="ACDEFGHIKLMNPQRSTVWY", modality="sequence")] * 9
        train_data = LabelledCandidates(candidates, multiclass_labels)

        with pytest.raises(ValueError, match="standardise_outputs"):
            model.train(train_data, val_data=LabelledCandidates([], np.array([])))
