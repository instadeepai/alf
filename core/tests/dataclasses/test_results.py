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

"""Unit tests for the Results dataclass."""

import numpy as np
import pytest
import torch
from alf_core.dataclasses import Predictions, Results
from alf_core.utils.enums import ProblemType
from beartype.roar import BeartypeCallHintParamViolation


def get_binary_probs(n: int = 4) -> np.ndarray:
    """Return simple binary probability array."""
    return np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]][:n])


def get_multiclass_probs(n: int = 3) -> np.ndarray:
    """Return simple 3-class probability array."""
    return np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]][:n])


@pytest.fixture
def predictions():
    """Return a Predictions instance with numpy array means."""
    return Predictions(means=np.array([1.0, 2.0, 3.0]))


class TestResultsNumpyTypeValidation:
    """Test that Results rejects non-numpy arrays for targets & non-Predictions for predictions."""

    def test_torch_targets_raises_type_error(self, predictions):
        """Test that torch tensor targets raises BeartypeCallHintParamViolation."""
        targets = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(BeartypeCallHintParamViolation):
            Results(targets=targets, predictions=predictions, problem_type=ProblemType.REGRESSION)

    def test_numpy_targets_accepted(self, predictions):
        """Test that numpy array targets are accepted."""
        targets = np.array([1.0, 2.0, 3.0])
        results = Results(
            targets=targets, predictions=predictions, problem_type=ProblemType.REGRESSION
        )
        assert isinstance(results.targets, np.ndarray)

    def test_non_predictions_raises_type_error(self):
        """Test non-Predictions object for predictions raises BeartypeCallHintParamViolation."""
        targets = np.array([1.0, 2.0, 3.0])
        with pytest.raises(BeartypeCallHintParamViolation):
            Results(
                targets=targets,
                predictions="not_a_predictions_object",
                problem_type=ProblemType.REGRESSION,
            )


class TestResultsValidation:
    """Test cases for Results validation."""

    def test_mismatched_lengths_raises(self, predictions):
        """Test that mismatched targets and predictions lengths raises ValueError."""
        targets = np.array([1.0, 2.0])  # length 2 vs predictions length 3
        with pytest.raises(AssertionError, match="same length"):
            Results(targets=targets, predictions=predictions, problem_type=ProblemType.REGRESSION)

    def test_mismatched_lengths_raises_2d(self):
        """Test length mismatch is also caught for 2D classification means."""
        probs = get_binary_probs(4)
        targets = np.array([0, 1, 0])  # wrong length
        preds = Predictions(means=probs)
        with pytest.raises(AssertionError):
            Results(targets=targets, predictions=preds, problem_type=ProblemType.BINARY)

    def test_results_computes_metrics(self, predictions):
        """Test that Results computes metrics on valid inputs."""
        targets = np.array([1.0, 2.0, 3.0])
        results = Results(
            targets=targets, predictions=predictions, problem_type=ProblemType.REGRESSION
        )
        assert isinstance(results.metrics, dict)
        assert len(results.metrics) > 0


class TestResultsRegressionRouting:
    """Regression routing computes the correct metrics."""

    def test_regression_metrics_computed(self):
        """Test that regression metrics (mse, spearman) are computed for REGRESSION."""
        preds = Predictions(means=np.array([1.0, 2.0, 3.0]))
        results = Results(
            targets=np.array([1.1, 1.9, 3.1]),
            predictions=preds,
            problem_type=ProblemType.REGRESSION,
        )
        assert "mse" in results.metrics
        assert "spearman" in results.metrics

    def test_regression_problem_type_stored(self):
        """Test that the problem type is stored correctly on the Results instance."""
        preds = Predictions(means=np.array([1.0, 2.0]))
        results = Results(
            targets=np.array([1.0, 2.0]), predictions=preds, problem_type=ProblemType.REGRESSION
        )
        assert results.problem_type == ProblemType.REGRESSION

    def test_regression_no_classification_metrics(self):
        """Test that classification metrics are absent for REGRESSION."""
        preds = Predictions(means=np.array([1.0, 2.0, 3.0]))
        results = Results(
            targets=np.array([1.0, 2.0, 3.0]),
            predictions=preds,
            problem_type=ProblemType.REGRESSION,
        )
        assert "accuracy" not in results.metrics
        assert "f1" not in results.metrics

    # The registry runs regret metrics with the default num_acquisitions (100),
    # which exceeds the 4 test samples and triggers the documented fallback warning.
    @pytest.mark.filterwarnings("ignore:num_acquisitions:UserWarning")
    def test_accuracy_metrics_present_when_variances_provided(self):
        """Variance-independent metrics must still be computed when variances exist.

        Regression test: previously the registry was queried either/or, so a
        surrogate that provided variances lost mse/spearman/pearson entirely.
        """
        means = np.array([1.0, 2.0, 3.0, 4.0])
        targets = np.array([1.1, 1.9, 3.2, 3.8])

        without_variances = Results(
            targets=targets,
            predictions=Predictions(means=means),
            problem_type=ProblemType.REGRESSION,
        ).metrics
        with_variances = Results(
            targets=targets,
            predictions=Predictions(means=means, variances=np.array([0.1, 0.2, 0.1, 0.3])),
            problem_type=ProblemType.REGRESSION,
        ).metrics

        # The accuracy metrics must not disappear once variances are present.
        for key in ("mse", "spearman", "pearson"):
            assert key in with_variances

        # Variance metrics are computed in addition to (not instead of) the
        # variance-independent ones.
        assert set(without_variances).issubset(with_variances)
        assert len(with_variances) > len(without_variances)


class TestResultsBinaryRouting:
    """Results correctly routes to classification metrics for BINARY."""

    def test_classification_metrics_computed(self):
        """Test that all classification metrics are computed for BINARY."""
        probs = get_binary_probs()
        targets = np.array([0, 1, 0, 1])
        preds = Predictions(means=probs)
        results = Results(targets=targets, predictions=preds, problem_type=ProblemType.BINARY)
        assert "accuracy" in results.metrics
        assert "f1" in results.metrics
        assert "precision" in results.metrics
        assert "recall" in results.metrics
        assert "auc_roc" in results.metrics

    def test_no_regression_metrics(self):
        """Test that regression metrics are absent for BINARY."""
        probs = get_binary_probs()
        targets = np.array([0, 1, 0, 1])
        preds = Predictions(means=probs)
        results = Results(targets=targets, predictions=preds, problem_type=ProblemType.BINARY)
        assert "mse" not in results.metrics
        assert "spearman" not in results.metrics

    def test_perfect_predictions(self):
        """Test that accuracy is 1.0 for perfect binary predictions."""
        probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
        targets = np.array([0, 1, 0, 1])
        preds = Predictions(means=probs)
        results = Results(targets=targets, predictions=preds, problem_type=ProblemType.BINARY)
        assert results.metrics["accuracy"] == pytest.approx(1.0)


class TestResultsMulticlassRouting:
    """Results correctly routes to classification metrics for MULTICLASS."""

    def test_classification_metrics_computed(self):
        """Test that classification metrics are computed for MULTICLASS."""
        probs = get_multiclass_probs()
        targets = np.array([0, 1, 2])
        preds = Predictions(means=probs)
        results = Results(targets=targets, predictions=preds, problem_type=ProblemType.MULTICLASS)
        assert "accuracy" in results.metrics
        assert "auc_roc" in results.metrics

    def test_perfect_multiclass(self):
        """Test that accuracy is 1.0 for perfect multiclass predictions."""
        probs = get_multiclass_probs()
        targets = np.array([0, 1, 2])
        preds = Predictions(means=probs)
        results = Results(targets=targets, predictions=preds, problem_type=ProblemType.MULTICLASS)
        assert results.metrics["accuracy"] == pytest.approx(1.0)
