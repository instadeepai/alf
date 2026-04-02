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
from alf_core.dataclasses import Predictions, Results
from alf_core.enums import ProblemType


def binary_probs(n: int = 4) -> np.ndarray:
    """Return simple binary probability array."""
    return np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]][:n])


def multiclass_probs(n: int = 3) -> np.ndarray:
    """Return simple 3-class probability array."""
    return np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]][:n])


class TestResultsRegressionRouting:
    """Regression routing computes the correct metrics."""

    def test_regression_metrics_computed(self):
        preds = Predictions(means=np.array([1.0, 2.0, 3.0]))
        results = Results(targets=np.array([1.1, 1.9, 3.1]), predictions=preds, problem_type=ProblemType.REGRESSION)
        assert "mse" in results.metrics
        assert "spearman" in results.metrics

    def test_regression_problem_type_stored(self):
        preds = Predictions(means=np.array([1.0, 2.0]))
        results = Results(targets=np.array([1.0, 2.0]), predictions=preds, problem_type=ProblemType.REGRESSION)
        assert results.problem_type == ProblemType.REGRESSION

    def test_regression_no_classification_metrics(self):
        preds = Predictions(means=np.array([1.0, 2.0, 3.0]))
        results = Results(targets=np.array([1.0, 2.0, 3.0]), predictions=preds, problem_type=ProblemType.REGRESSION)
        assert "accuracy" not in results.metrics
        assert "f1" not in results.metrics


class TestResultsBinaryRouting:
    """Results correctly routes to classification metrics for BINARY."""

    def test_classification_metrics_computed(self):
        probs = binary_probs()
        targets = np.array([0, 1, 0, 1])
        preds = Predictions(means=probs)
        results = Results(targets=targets, predictions=preds, problem_type=ProblemType.BINARY)
        assert "accuracy" in results.metrics
        assert "f1" in results.metrics
        assert "precision" in results.metrics
        assert "recall" in results.metrics
        assert "auc_roc" in results.metrics

    def test_no_regression_metrics(self):
        probs = binary_probs()
        targets = np.array([0, 1, 0, 1])
        preds = Predictions(means=probs)
        results = Results(targets=targets, predictions=preds, problem_type=ProblemType.BINARY)
        assert "mse" not in results.metrics
        assert "spearman" not in results.metrics

    def test_perfect_predictions(self):
        probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
        targets = np.array([0, 1, 0, 1])
        preds = Predictions(means=probs)
        results = Results(targets=targets, predictions=preds, problem_type=ProblemType.BINARY)
        assert results.metrics["accuracy"] == pytest.approx(1.0)


class TestResultsMulticlassRouting:
    """Results correctly routes to classification metrics for MULTICLASS."""

    def test_classification_metrics_computed(self):
        probs = multiclass_probs()
        targets = np.array([0, 1, 2])
        preds = Predictions(means=probs)
        results = Results(targets=targets, predictions=preds, problem_type=ProblemType.MULTICLASS)
        assert "accuracy" in results.metrics
        assert "auc_roc" in results.metrics

    def test_perfect_multiclass(self):
        probs = multiclass_probs()
        targets = np.array([0, 1, 2])
        preds = Predictions(means=probs)
        results = Results(targets=targets, predictions=preds, problem_type=ProblemType.MULTICLASS)
        assert results.metrics["accuracy"] == pytest.approx(1.0)


class TestResultsLengthAssertion:
    """Shape validation works for both 1D (regression) and 2D (classification) means."""

    def test_2d_means_length_check_passes(self):
        probs = binary_probs(4)
        targets = np.array([0, 1, 0, 1])
        preds = Predictions(means=probs)
        results = Results(targets=targets, predictions=preds, problem_type=ProblemType.BINARY)
        assert len(results.metrics) > 0

    def test_length_mismatch_raises(self):
        probs = binary_probs(4)
        targets = np.array([0, 1, 0])  # wrong length
        preds = Predictions(means=probs)
        with pytest.raises(AssertionError):
            Results(targets=targets, predictions=preds, problem_type=ProblemType.BINARY)
