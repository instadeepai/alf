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

"""Unit tests for CNNModel in BINARY and MULTICLASS configurations."""

import numpy as np
import pytest
from alf_core import Candidate, LabelledCandidates
from alf_core.utils.enums import ProblemType
from alf_tools.models.cnn import CNNModel, CNNModelConfig, CNNTrainConfig

SEQ = "ACDEFGHIKLMNPQRSTVWY"  # 20-char sequence (protein alphabet)


def make_data(labels: list[int]) -> LabelledCandidates:
    """Create LabelledCandidates with identical sequences and given integer labels.

    Returns:
        A LabelledCandidates with one candidate per label, all using the same sequence.
    """
    candidates = [Candidate(data=SEQ, modality="sequence") for _ in labels]
    return LabelledCandidates(candidates, np.array(labels, dtype=float))


@pytest.fixture
def small_config() -> tuple[CNNModelConfig, CNNTrainConfig]:
    """Return a minimal CNNModelConfig and CNNTrainConfig for fast tests."""
    model_cfg = CNNModelConfig(num_filters=8, num_conv_layers=1, fc_hidden_dim=16, num_classes=3)
    train_cfg = CNNTrainConfig(batch_size=4, num_epochs=1)
    return model_cfg, train_cfg


@pytest.fixture
def binary_model(small_config):
    """Return a CNNModel instance for binary classification tests."""
    model_cfg, train_cfg = small_config
    return CNNModel(model_config=model_cfg, train_config=train_cfg, device="cpu")


@pytest.fixture
def multiclass_model(small_config):
    """Return a CNNModel instance for multiclass classification tests."""
    model_cfg, train_cfg = small_config
    return CNNModel(model_config=model_cfg, train_config=train_cfg, device="cpu")


class TestCNNBinaryClassification:
    """Tests for CNNModel in BINARY mode."""

    def test_train_does_not_raise(self, binary_model):
        """Test that training in BINARY mode completes without error."""
        data = make_data([0, 1, 0, 1, 0, 1, 0, 1])
        binary_model.train(data, problem_type=ProblemType.BINARY)

    def test_problem_type_stored(self, binary_model):
        """Test that the problem type is stored on the model after training."""
        data = make_data([0, 1, 0, 1])
        binary_model.train(data, problem_type=ProblemType.BINARY)
        assert binary_model._problem_type == ProblemType.BINARY

    def test_predict_shape(self, binary_model):
        """Test that binary predictions have shape (n_samples, 2)."""
        data = make_data([0, 1, 0, 1, 0, 1])
        binary_model.train(data, problem_type=ProblemType.BINARY)
        preds = binary_model.predict(data.candidates)
        # Binary: means must be (n_samples, 2)
        assert preds.means.ndim == 2
        assert preds.means.shape == (len(data), 2)

    def test_predict_probabilities_sum_to_one(self, binary_model):
        """Test that binary predicted probabilities sum to 1 for each sample."""
        data = make_data([0, 1, 0, 1, 0, 1])
        binary_model.train(data, problem_type=ProblemType.BINARY)
        preds = binary_model.predict(data.candidates)
        np.testing.assert_allclose(preds.means.sum(axis=1), np.ones(len(data)), atol=1e-5)

    def test_predict_probabilities_in_range(self, binary_model):
        """Test that binary predicted probabilities are in [0, 1]."""
        data = make_data([0, 1, 0, 1])
        binary_model.train(data, problem_type=ProblemType.BINARY)
        preds = binary_model.predict(data.candidates)
        assert np.all(preds.means >= 0.0)
        assert np.all(preds.means <= 1.0)

    def test_output_neuron_count_is_one(self, binary_model):
        """Test that the binary model uses a single output neuron (logit)."""
        data = make_data([0, 1, 0, 1])
        binary_model.train(data, problem_type=ProblemType.BINARY)
        assert binary_model.model is not None
        assert binary_model.model._output_neurons == 1

    def test_uses_bce_loss(self, binary_model):
        """BCEWithLogitsLoss works with float32 labels — no dtype error."""
        data = make_data([0, 1, 0, 1, 0, 1])
        # Would raise RuntimeError if wrong loss or dtype
        binary_model.train(data, problem_type=ProblemType.BINARY)


class TestCNNMulticlassClassification:
    """Tests for CNNModel in MULTICLASS mode."""

    def test_train_does_not_raise(self, multiclass_model):
        """Test that training in MULTICLASS mode completes without error."""
        data = make_data([0, 1, 2, 0, 1, 2])
        multiclass_model.train(data, problem_type=ProblemType.MULTICLASS)

    def test_problem_type_stored(self, multiclass_model):
        """Test that the problem type is stored on the model after training."""
        data = make_data([0, 1, 2, 0, 1, 2])
        multiclass_model.train(data, problem_type=ProblemType.MULTICLASS)
        assert multiclass_model._problem_type == ProblemType.MULTICLASS

    def test_predict_shape(self, multiclass_model):
        """Test that multiclass predictions have shape (n_samples, num_classes)."""
        data = make_data([0, 1, 2, 0, 1, 2])
        multiclass_model.train(data, problem_type=ProblemType.MULTICLASS)
        preds = multiclass_model.predict(data.candidates)
        # Multiclass: means must be (n_samples, num_classes)
        assert preds.means.ndim == 2
        assert preds.means.shape == (len(data), 3)  # num_classes=3

    def test_predict_probabilities_sum_to_one(self, multiclass_model):
        """Test that multiclass predicted probabilities sum to 1 for each sample."""
        data = make_data([0, 1, 2, 0, 1, 2])
        multiclass_model.train(data, problem_type=ProblemType.MULTICLASS)
        preds = multiclass_model.predict(data.candidates)
        np.testing.assert_allclose(preds.means.sum(axis=1), np.ones(len(data)), atol=1e-5)

    def test_output_neuron_count_matches_num_classes(self, multiclass_model):
        """Test that the multiclass model output neuron count equals num_classes."""
        data = make_data([0, 1, 2, 0, 1, 2])
        multiclass_model.train(data, problem_type=ProblemType.MULTICLASS)
        assert multiclass_model.model is not None
        assert multiclass_model.model._output_neurons == 3

    def test_uses_cross_entropy_loss(self, multiclass_model):
        """CrossEntropyLoss works with long labels — no dtype error."""
        data = make_data([0, 1, 2, 0, 1, 2])
        multiclass_model.train(data, problem_type=ProblemType.MULTICLASS)


class TestCNNRegressionUnchanged:
    """Regression behaviour is unchanged after Phase 3 changes."""

    def test_regression_predict_shape_is_1d(self):
        """Test that regression predictions remain 1D arrays."""
        model_cfg = CNNModelConfig(num_filters=8, num_conv_layers=1, fc_hidden_dim=16)
        train_cfg = CNNTrainConfig(batch_size=4, num_epochs=1)
        model = CNNModel(model_config=model_cfg, train_config=train_cfg, device="cpu")
        data = make_data([0, 1, 2, 3, 4, 5])
        data_float = LabelledCandidates(data.candidates, np.array([0.1, 0.5, 0.9, 0.3, 0.7, 0.4]))
        model.train(data_float)
        preds = model.predict(data_float.candidates)
        assert preds.means.ndim == 1
        assert preds.means.shape == (len(data_float),)

    def test_default_problem_type_is_regression(self):
        """Test that the default problem type before training is REGRESSION."""
        model_cfg = CNNModelConfig(num_filters=8, num_conv_layers=1, fc_hidden_dim=16)
        train_cfg = CNNTrainConfig(batch_size=4, num_epochs=1)
        model = CNNModel(model_config=model_cfg, train_config=train_cfg, device="cpu")
        assert model._problem_type == ProblemType.REGRESSION

    def test_output_neurons_is_one_for_regression(self):
        """Test that regression training uses a single output neuron."""
        model_cfg = CNNModelConfig(num_filters=8, num_conv_layers=1, fc_hidden_dim=16)
        train_cfg = CNNTrainConfig(batch_size=4, num_epochs=1)
        model = CNNModel(model_config=model_cfg, train_config=train_cfg, device="cpu")
        data_float = LabelledCandidates(
            [Candidate(data=SEQ, modality="sequence") for _ in range(4)],
            np.array([0.1, 0.5, 0.9, 0.3]),
        )
        model.train(data_float)
        assert model.model is not None
        assert model.model._output_neurons == 1
