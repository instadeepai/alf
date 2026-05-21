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

"""Tests for shared metric utilities and registry classes."""

import numpy as np
import pytest
from alf_core.utils.metrics.base import (
    ClassificationMetricRegistry,
    RegressionMetricRegistry,
    check_inputs,
    check_variance_validity,
    classification_metric_registry,
    regression_metric_registry,
    require_min_samples,
)


class TestCheckInputs:
    """Tests for check_inputs validator."""

    def test_valid_inputs_pass(self):
        """Valid matching arrays pass without error."""
        check_inputs(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    def test_shape_mismatch_raises(self):
        """Mismatched shapes raise AssertionError."""
        with pytest.raises(AssertionError, match="shape"):
            check_inputs(np.array([1.0, 2.0]), np.array([1.0]))

    def test_empty_arrays_raise(self):
        """Empty arrays raise AssertionError."""
        with pytest.raises(AssertionError, match="empty"):
            check_inputs(np.array([]), np.array([]))

    def test_nan_means_raise(self):
        """NaN in means raises AssertionError."""
        with pytest.raises(AssertionError, match="NaN"):
            check_inputs(np.array([np.nan, 1.0]), np.array([1.0, 2.0]))

    def test_nan_targets_raise(self):
        """NaN in targets raises AssertionError."""
        with pytest.raises(AssertionError, match="NaN"):
            check_inputs(np.array([1.0, 2.0]), np.array([np.nan, 2.0]))


class TestCheckVarianceValidity:
    """Tests for check_variance_validity validator."""

    def test_valid_variances_pass(self):
        """Non-negative variances pass without error."""
        check_variance_validity(np.array([0.1, 0.2]), np.array([1.0, 2.0]))

    def test_none_variances_raise(self):
        """None variances raise AssertionError."""
        with pytest.raises(AssertionError):
            check_variance_validity(None, np.array([1.0, 2.0]))

    def test_negative_variance_raises(self):
        """Negative variance raises AssertionError."""
        with pytest.raises(AssertionError, match="non-negative"):
            check_variance_validity(np.array([-0.1, 0.2]), np.array([1.0, 2.0]))

    def test_length_mismatch_raises(self):
        """Length mismatch raises AssertionError."""
        with pytest.raises(AssertionError, match="variances has"):
            check_variance_validity(np.array([0.1]), np.array([1.0, 2.0]))

    def test_nan_variance_raises(self):
        """NaN variance raises AssertionError with NaN message."""
        with pytest.raises(AssertionError, match="NaN"):
            check_variance_validity(np.array([np.nan, 0.2]), np.array([1.0, 2.0]))


class TestRequireMinSamples:
    """Tests for require_min_samples decorator."""

    def test_zero_samples_returns_empty_dict(self):
        """Zero samples returns empty dict."""

        @require_min_samples(2)
        def dummy(means, variances, targets):
            return {"value": 1.0}

        result = dummy(np.array([]), None, np.array([]))
        assert result == {}

    def test_one_sample_returns_empty_dict(self):
        """One sample below minimum returns empty dict."""

        @require_min_samples(2)
        def dummy(means, variances, targets):
            return {"value": 1.0}

        result = dummy(np.array([1.0]), None, np.array([1.0]))
        assert result == {}

    def test_two_samples_calls_through(self):
        """Exactly minimum samples calls the underlying function."""

        @require_min_samples(2)
        def dummy(means, variances, targets):
            return {"value": 42.0}

        result = dummy(np.array([1.0, 2.0]), None, np.array([1.0, 2.0]))
        assert result == {"value": 42.0}

    def test_three_samples_calls_through(self):
        """More than minimum samples calls the underlying function."""

        @require_min_samples(2)
        def dummy(means, variances, targets):
            return {"value": float(len(means))}

        result = dummy(np.array([1.0, 2.0, 3.0]), None, np.array([1.0, 2.0, 3.0]))
        assert result == {"value": 3.0}


class TestRegressionMetricRegistry:
    """Tests for RegressionMetricRegistry."""

    def test_register_and_get_with_variance(self):
        """Registered variance-required function is returned by get_metrics(True)."""
        reg = RegressionMetricRegistry()
        reg.register("my_var_metric", lambda: None, requires_variance=True)
        assert "my_var_metric" in reg.get_metrics(requires_variance=True)
        assert "my_var_metric" not in reg.get_metrics(requires_variance=False)

    def test_register_and_get_without_variance(self):
        """Registered no-variance function is returned by get_metrics(False)."""
        reg = RegressionMetricRegistry()
        reg.register("my_no_var_metric", lambda: None, requires_variance=False)
        assert "my_no_var_metric" in reg.get_metrics(requires_variance=False)
        assert "my_no_var_metric" not in reg.get_metrics(requires_variance=True)


class TestClassificationMetricRegistry:
    """Tests for ClassificationMetricRegistry."""

    def test_register_and_get(self):
        """Registered function is returned by get_metrics."""
        reg = ClassificationMetricRegistry()
        reg.register("my_cls_metric", lambda: None)
        assert "my_cls_metric" in reg.get_metrics()

    def test_global_registry_has_expected_metrics(self):
        """Global classification registry contains all expected metric names."""
        names = set(classification_metric_registry.get_metrics().keys())
        assert {"accuracy", "f1", "precision", "recall", "auc_roc"}.issubset(names)


class TestGlobalRegistryInstances:
    """Tests that the global instances are accessible from _base."""

    def test_regression_registry_is_regression_metric_registry(self):
        """regression_metric_registry is a RegressionMetricRegistry instance."""
        assert isinstance(regression_metric_registry, RegressionMetricRegistry)

    def test_classification_registry_is_classification_metric_registry(self):
        """classification_metric_registry is a ClassificationMetricRegistry instance."""
        assert isinstance(classification_metric_registry, ClassificationMetricRegistry)
