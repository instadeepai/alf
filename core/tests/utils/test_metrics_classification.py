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

"""Tests for classification metrics."""

import numpy as np
import pytest
from alf_core.utils.metrics.base import classification_metric_registry
from alf_core.utils.metrics.classification import (
    accuracy,
    auc_roc,
    f1,
    precision,
    recall,
)


class TestClassificationMetricRegistry:
    """Tests for the classification metric registry."""

    def test_all_metrics_registered(self):
        """All expected metrics are present in the registry."""
        names = set(classification_metric_registry.get_metrics().keys())
        assert {"accuracy", "f1", "precision", "recall", "auc_roc"}.issubset(names)


class TestAccuracy:
    """Tests for accuracy metric."""

    def test_perfect_binary(self):
        """Accuracy is 1.0 for perfect binary predictions."""
        probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targets = np.array([1, 0, 1])
        result = accuracy(probs, targets)
        assert result == {"accuracy": 1.0}

    def test_partial_binary(self):
        """Accuracy is 0.5 when half the predictions are correct."""
        probs = np.array([[0.1, 0.9], [0.1, 0.9]])
        targets = np.array([1, 0])
        result = accuracy(probs, targets)
        assert result == {"accuracy": 0.5}

    def test_perfect_multiclass(self):
        """Accuracy is 1.0 for perfect multiclass predictions."""
        probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        targets = np.array([0, 1, 2])
        result = accuracy(probs, targets)
        assert result == {"accuracy": 1.0}

    def test_returns_dict(self):
        """Accuracy returns a dict with a float value."""
        probs = np.array([[0.6, 0.4], [0.4, 0.6]])
        targets = np.array([0, 1])
        result = accuracy(probs, targets)
        assert isinstance(result, dict)
        assert "accuracy" in result
        assert isinstance(result["accuracy"], float)


class TestF1:
    """Tests for F1 metric."""

    def test_perfect_binary(self):
        """F1 is 1.0 for perfect binary predictions."""
        probs = np.array([[0.9, 0.1], [0.1, 0.9]])
        targets = np.array([0, 1])
        result = f1(probs, targets)
        assert result["f1"] == pytest.approx(1.0)

    def test_zero_division_does_not_raise(self):
        """zero_division=0 prevents errors when a class is never predicted."""
        probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
        targets = np.array([0, 0, 1])
        result = f1(probs, targets)
        assert "f1" in result
        assert 0.0 <= result["f1"] <= 1.0

    def test_multiclass_macro(self):
        """F1 macro average is 1.0 for perfect multiclass predictions."""
        probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        targets = np.array([0, 1, 2])
        result = f1(probs, targets)
        assert result["f1"] == pytest.approx(1.0)


class TestPrecision:
    """Tests for precision metric."""

    def test_perfect(self):
        """Precision is 1.0 for perfect predictions."""
        probs = np.array([[0.9, 0.1], [0.1, 0.9]])
        targets = np.array([0, 1])
        result = precision(probs, targets)
        assert result["precision"] == pytest.approx(1.0)

    def test_zero_division_does_not_raise(self):
        """zero_division=0 prevents errors when a class is never predicted."""
        probs = np.array([[0.9, 0.1], [0.9, 0.1]])
        targets = np.array([0, 1])
        result = precision(probs, targets)
        assert "precision" in result

    def test_returns_float(self):
        """Precision returns a float value."""
        probs = np.array([[0.6, 0.4], [0.4, 0.6]])
        targets = np.array([0, 1])
        assert isinstance(precision(probs, targets)["precision"], float)


class TestRecall:
    """Tests for recall metric."""

    def test_perfect(self):
        """Recall is 1.0 for perfect predictions."""
        probs = np.array([[0.9, 0.1], [0.1, 0.9]])
        targets = np.array([0, 1])
        result = recall(probs, targets)
        assert result["recall"] == pytest.approx(1.0)

    def test_zero_division_does_not_raise(self):
        """zero_division=0 prevents errors when a class has no true positives."""
        probs = np.array([[0.9, 0.1], [0.9, 0.1]])
        targets = np.array([0, 0])
        result = recall(probs, targets)
        assert "recall" in result

    def test_returns_float(self):
        """Recall returns a float value."""
        probs = np.array([[0.6, 0.4], [0.4, 0.6]])
        targets = np.array([0, 1])
        assert isinstance(recall(probs, targets)["recall"], float)


class TestAucRoc:
    """Tests for AUC-ROC metric."""

    def test_perfect_binary(self):
        """AUC-ROC is 1.0 for perfect binary predictions."""
        probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        targets = np.array([0, 1, 0, 1])
        result = auc_roc(probs, targets)
        assert result["auc_roc"] == pytest.approx(1.0)

    def test_binary_uses_positive_class_probs(self):
        """Binary branch is taken when probs has 2 columns."""
        probs = np.array([[0.7, 0.3], [0.3, 0.7]])
        targets = np.array([0, 1])
        result = auc_roc(probs, targets)
        assert "auc_roc" in result
        assert 0.0 <= result["auc_roc"] <= 1.0

    def test_multiclass_ovr(self):
        """AUC-ROC uses one-vs-rest for multiclass inputs."""
        probs = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
        ])
        targets = np.array([0, 1, 2, 0])
        result = auc_roc(probs, targets)
        assert "auc_roc" in result
        assert 0.0 <= result["auc_roc"] <= 1.0

    def test_returns_float(self):
        """AUC-ROC returns a float value."""
        probs = np.array([[0.6, 0.4], [0.4, 0.6]])
        targets = np.array([0, 1])
        assert isinstance(auc_roc(probs, targets)["auc_roc"], float)

    def test_single_class_targets_returns_empty(self):
        """Fewer than 2 unique classes returns empty dict."""
        probs = np.array([[0.2, 0.8], [0.3, 0.7]])
        targets = np.array([1, 1])
        assert auc_roc(probs, targets) == {}

    def test_one_sample_require_min_returns_empty(self):
        """1 sample triggers require_min_samples guard and returns {}."""
        probs = np.array([[0.4, 0.6]])
        targets = np.array([1])
        assert auc_roc(probs, targets) == {}


class TestClassificationMetricInputValidation:
    """Tests for input validation in the classification_metric decorator."""

    def test_1d_probs_raises(self):
        """1D probability arrays are rejected."""
        probs = np.array([0.1, 0.9])
        targets = np.array([1])
        with pytest.raises(AssertionError, match="n_samples, num_classes"):
            accuracy(probs, targets)

    def test_empty_input_raises(self):
        """Empty inputs are rejected."""
        probs = np.empty((0, 2))
        targets = np.empty(0)
        with pytest.raises(AssertionError, match="Empty input arrays"):
            accuracy(probs, targets)

    def test_batch_size_mismatch_raises(self):
        """Mismatched batch sizes between probs and targets are rejected."""
        probs = np.array([[0.6, 0.4], [0.4, 0.6]])
        targets = np.array([0])
        with pytest.raises(AssertionError, match="probs and targets batch size mismatch"):
            accuracy(probs, targets)
