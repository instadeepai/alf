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
    pearson,
    precision,
    recall,
    require_min_samples,
    spearman,
)


class TestClassificationMetricRegistry:
    """Tests for the classification metric registry."""

    def test_all_metrics_registered(self):
        """Test that all expected metrics are present in the registry."""
        names = set(classification_metric_registry.metrics.keys())
        assert {"accuracy", "f1", "precision", "recall", "auc_roc"}.issubset(names)


class TestAccuracy:
    """Tests for accuracy metric."""

    def test_perfect_binary(self):
        """Test accuracy is 1.0 for perfect binary predictions."""
        probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targets = np.array([1, 0, 1])
        result = accuracy(probs, targets)
        assert result == {"accuracy": 1.0}

    def test_partial_binary(self):
        """Test accuracy is 0.5 when half the predictions are correct."""
        probs = np.array([[0.1, 0.9], [0.1, 0.9]])
        targets = np.array([1, 0])
        result = accuracy(probs, targets)
        assert result == {"accuracy": 0.5}

    def test_perfect_multiclass(self):
        """Test accuracy is 1.0 for perfect multiclass predictions."""
        probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        targets = np.array([0, 1, 2])
        result = accuracy(probs, targets)
        assert result == {"accuracy": 1.0}

    def test_returns_dict(self):
        """Test that accuracy returns a dict with a float value."""
        probs = np.array([[0.6, 0.4], [0.4, 0.6]])
        targets = np.array([0, 1])
        result = accuracy(probs, targets)
        assert isinstance(result, dict)
        assert "accuracy" in result
        assert isinstance(result["accuracy"], float)


class TestF1:
    """Tests for F1 metric."""

    def test_perfect_binary(self):
        """Test F1 is 1.0 for perfect binary predictions."""
        probs = np.array([[0.9, 0.1], [0.1, 0.9]])
        targets = np.array([0, 1])
        result = f1(probs, targets)
        assert result["f1"] == pytest.approx(1.0)

    def test_zero_division_does_not_raise(self):
        """Test that zero_division=0 prevents errors when a class is never predicted."""
        # All predicted as class 0 — recall for class 1 is 0, zero_division=0
        probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
        targets = np.array([0, 0, 1])
        result = f1(probs, targets)
        assert "f1" in result
        assert 0.0 <= result["f1"] <= 1.0

    def test_multiclass_macro(self):
        """Test F1 macro average is 1.0 for perfect multiclass predictions."""
        probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        targets = np.array([0, 1, 2])
        result = f1(probs, targets)
        assert result["f1"] == pytest.approx(1.0)


class TestPrecision:
    """Tests for precision metric."""

    def test_perfect(self):
        """Test precision is 1.0 for perfect predictions."""
        probs = np.array([[0.9, 0.1], [0.1, 0.9]])
        targets = np.array([0, 1])
        result = precision(probs, targets)
        assert result["precision"] == pytest.approx(1.0)

    def test_zero_division_does_not_raise(self):
        """Test that zero_division=0 prevents errors when a class is never predicted."""
        probs = np.array([[0.9, 0.1], [0.9, 0.1]])
        targets = np.array([0, 1])
        result = precision(probs, targets)
        assert "precision" in result

    def test_returns_float(self):
        """Test that precision returns a float value."""
        probs = np.array([[0.6, 0.4], [0.4, 0.6]])
        targets = np.array([0, 1])
        assert isinstance(precision(probs, targets)["precision"], float)


class TestRecall:
    """Tests for recall metric."""

    def test_perfect(self):
        """Test recall is 1.0 for perfect predictions."""
        probs = np.array([[0.9, 0.1], [0.1, 0.9]])
        targets = np.array([0, 1])
        result = recall(probs, targets)
        assert result["recall"] == pytest.approx(1.0)

    def test_zero_division_does_not_raise(self):
        """Test that zero_division=0 prevents errors when a class has no true positives."""
        probs = np.array([[0.9, 0.1], [0.9, 0.1]])
        targets = np.array([0, 0])
        result = recall(probs, targets)
        assert "recall" in result

    def test_returns_float(self):
        """Test that recall returns a float value."""
        probs = np.array([[0.6, 0.4], [0.4, 0.6]])
        targets = np.array([0, 1])
        assert isinstance(recall(probs, targets)["recall"], float)


class TestAucRoc:
    """Tests for AUC-ROC metric."""

    def test_perfect_binary(self):
        """Test AUC-ROC is 1.0 for perfect binary predictions."""
        probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        targets = np.array([0, 1, 0, 1])
        result = auc_roc(probs, targets)
        assert result["auc_roc"] == pytest.approx(1.0)

    def test_binary_uses_positive_class_probs(self):
        """Test that the binary branch is taken when probs has 2 columns."""
        # Verify binary branch is taken (probs.shape[1] == 2)
        probs = np.array([[0.7, 0.3], [0.3, 0.7]])
        targets = np.array([0, 1])
        result = auc_roc(probs, targets)
        assert "auc_roc" in result
        assert 0.0 <= result["auc_roc"] <= 1.0

    def test_multiclass_ovr(self):
        """Test that AUC-ROC uses one-vs-rest for multiclass inputs."""
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
        """Test that AUC-ROC returns a float value."""
        probs = np.array([[0.6, 0.4], [0.4, 0.6]])
        targets = np.array([0, 1])
        assert isinstance(auc_roc(probs, targets)["auc_roc"], float)


class TestClassificationMetricInputValidation:
    """Tests for input validation in the classification_metric decorator."""

    def test_1d_probs_raises(self):
        """Test that 1D probability arrays are rejected."""
        probs = np.array([0.1, 0.9])
        targets = np.array([1])
        with pytest.raises(AssertionError, match="n_samples, num_classes"):
            accuracy(probs, targets)

    def test_empty_input_raises(self):
        """Test that empty inputs are rejected."""
        probs = np.empty((0, 2))
        targets = np.empty(0)
        with pytest.raises(AssertionError, match="Empty input arrays"):
            accuracy(probs, targets)

    def test_batch_size_mismatch_raises(self):
        """Test that mismatched batch sizes between probs and targets are rejected."""
        probs = np.array([[0.6, 0.4], [0.4, 0.6]])
        targets = np.array([0])
        with pytest.raises(AssertionError, match="probs and targets batch size mismatch"):
            accuracy(probs, targets)


class TestRequireMinSamples:
    """Tests for require_min_samples decorator."""

    def test_zero_samples_returns_empty_dict(self):
        """Test that zero samples returns empty dict."""

        @require_min_samples(2)
        def dummy(means, variances, targets):
            return {"value": 1.0}

        result = dummy(np.array([]), None, np.array([]))
        assert result == {}

    def test_one_sample_returns_empty_dict(self):
        """Test that one sample below minimum returns empty dict."""

        @require_min_samples(2)
        def dummy(means, variances, targets):
            return {"value": 1.0}

        result = dummy(np.array([1.0]), None, np.array([1.0]))
        assert result == {}

    def test_two_samples_calls_through(self):
        """Test that exactly minimum samples calls the underlying function."""

        @require_min_samples(2)
        def dummy(means, variances, targets):
            return {"value": 42.0}

        result = dummy(np.array([1.0, 2.0]), None, np.array([1.0, 2.0]))
        assert result == {"value": 42.0}

    def test_three_samples_calls_through(self):
        """Test that more than minimum samples calls the underlying function."""

        @require_min_samples(2)
        def dummy(means, variances, targets):
            return {"value": float(len(means))}

        result = dummy(np.array([1.0, 2.0, 3.0]), None, np.array([1.0, 2.0, 3.0]))
        assert result == {"value": 3.0}


class TestPearsonGuard:
    def test_one_sample_returns_empty_dict(self):
        result = pearson(np.array([1.0]), None, np.array([1.0]))
        assert result == {}

    def test_two_samples_returns_pearson_key(self):
        result = pearson(np.array([1.0, 2.0]), None, np.array([1.0, 2.0]))
        assert "pearson" in result
        assert isinstance(result["pearson"], float)

    def test_three_samples_returns_finite_value(self):
        result = pearson(np.array([1.0, 2.0, 3.0]), None, np.array([1.0, 2.0, 3.0]))
        assert "pearson" in result
        assert np.isfinite(result["pearson"])


class TestSpearmanGuard:
    def test_one_sample_returns_empty_dict(self):
        result = spearman(np.array([1.0]), None, np.array([1.0]))
        assert result == {}

    def test_two_samples_returns_spearman_key(self):
        result = spearman(np.array([1.0, 2.0]), None, np.array([1.0, 2.0]))
        assert "spearman" in result
        assert isinstance(result["spearman"], float)

    def test_three_samples_returns_finite_value(self):
        result = spearman(np.array([1.0, 2.0, 3.0]), None, np.array([1.0, 2.0, 3.0]))
        assert "spearman" in result
        assert np.isfinite(result["spearman"])
