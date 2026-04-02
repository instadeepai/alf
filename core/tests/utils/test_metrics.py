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

"""Unit tests for classification metrics."""

import numpy as np
import pytest
from alf_core.utils.metrics import (
    accuracy,
    auc_roc,
    classification_metric_registry,
    f1,
    precision,
    recall,
)


class TestClassificationMetricRegistry:
    """Tests for the classification metric registry."""

    def test_all_metrics_registered(self):
        names = set(classification_metric_registry.metrics.keys())
        assert {"accuracy", "f1", "precision", "recall", "auc_roc"}.issubset(names)


class TestAccuracy:
    """Tests for accuracy metric."""

    def test_perfect_binary(self):
        probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targets = np.array([1, 0, 1])
        result = accuracy(probs, targets)
        assert result == {"accuracy": 1.0}

    def test_partial_binary(self):
        probs = np.array([[0.1, 0.9], [0.1, 0.9]])
        targets = np.array([1, 0])
        result = accuracy(probs, targets)
        assert result == {"accuracy": 0.5}

    def test_perfect_multiclass(self):
        probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        targets = np.array([0, 1, 2])
        result = accuracy(probs, targets)
        assert result == {"accuracy": 1.0}

    def test_returns_dict(self):
        probs = np.array([[0.6, 0.4], [0.4, 0.6]])
        targets = np.array([0, 1])
        result = accuracy(probs, targets)
        assert isinstance(result, dict)
        assert "accuracy" in result
        assert isinstance(result["accuracy"], float)


class TestF1:
    """Tests for F1 metric."""

    def test_perfect_binary(self):
        probs = np.array([[0.9, 0.1], [0.1, 0.9]])
        targets = np.array([0, 1])
        result = f1(probs, targets)
        assert result["f1"] == pytest.approx(1.0)

    def test_zero_division_does_not_raise(self):
        # All predicted as class 0 — recall for class 1 is 0, zero_division=0
        probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
        targets = np.array([0, 0, 1])
        result = f1(probs, targets)
        assert "f1" in result
        assert 0.0 <= result["f1"] <= 1.0

    def test_multiclass_macro(self):
        probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        targets = np.array([0, 1, 2])
        result = f1(probs, targets)
        assert result["f1"] == pytest.approx(1.0)


class TestPrecision:
    """Tests for precision metric."""

    def test_perfect(self):
        probs = np.array([[0.9, 0.1], [0.1, 0.9]])
        targets = np.array([0, 1])
        result = precision(probs, targets)
        assert result["precision"] == pytest.approx(1.0)

    def test_zero_division_does_not_raise(self):
        probs = np.array([[0.9, 0.1], [0.9, 0.1]])
        targets = np.array([0, 1])
        result = precision(probs, targets)
        assert "precision" in result

    def test_returns_float(self):
        probs = np.array([[0.6, 0.4], [0.4, 0.6]])
        targets = np.array([0, 1])
        assert isinstance(precision(probs, targets)["precision"], float)


class TestRecall:
    """Tests for recall metric."""

    def test_perfect(self):
        probs = np.array([[0.9, 0.1], [0.1, 0.9]])
        targets = np.array([0, 1])
        result = recall(probs, targets)
        assert result["recall"] == pytest.approx(1.0)

    def test_zero_division_does_not_raise(self):
        probs = np.array([[0.9, 0.1], [0.9, 0.1]])
        targets = np.array([0, 0])
        result = recall(probs, targets)
        assert "recall" in result

    def test_returns_float(self):
        probs = np.array([[0.6, 0.4], [0.4, 0.6]])
        targets = np.array([0, 1])
        assert isinstance(recall(probs, targets)["recall"], float)


class TestAucRoc:
    """Tests for AUC-ROC metric."""

    def test_perfect_binary(self):
        probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        targets = np.array([0, 1, 0, 1])
        result = auc_roc(probs, targets)
        assert result["auc_roc"] == pytest.approx(1.0)

    def test_binary_uses_positive_class_probs(self):
        # Verify binary branch is taken (probs.shape[1] == 2)
        probs = np.array([[0.7, 0.3], [0.3, 0.7]])
        targets = np.array([0, 1])
        result = auc_roc(probs, targets)
        assert "auc_roc" in result
        assert 0.0 <= result["auc_roc"] <= 1.0

    def test_multiclass_ovr(self):
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
        probs = np.array([[0.6, 0.4], [0.4, 0.6]])
        targets = np.array([0, 1])
        assert isinstance(auc_roc(probs, targets)["auc_roc"], float)


class TestClassificationMetricInputValidation:
    """Tests for input validation in the classification_metric decorator."""

    def test_1d_probs_raises(self):
        probs = np.array([0.1, 0.9])
        targets = np.array([1])
        with pytest.raises(AssertionError, match="2D"):
            accuracy(probs, targets)

    def test_empty_input_raises(self):
        probs = np.empty((0, 2))
        targets = np.empty(0)
        with pytest.raises(AssertionError, match="Empty"):
            accuracy(probs, targets)

    def test_batch_size_mismatch_raises(self):
        probs = np.array([[0.6, 0.4], [0.4, 0.6]])
        targets = np.array([0])
        with pytest.raises(AssertionError, match="batch size mismatch"):
            accuracy(probs, targets)
