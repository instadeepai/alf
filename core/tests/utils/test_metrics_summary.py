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

"""Tests for alf_core.utils.metrics.aggregate standalone functions."""

import numpy as np
import pytest
from alf_core.utils.metrics import auc_top_k, compute_aggregate_metrics


class TestAucTopK:
    """Tests for auc_top_k standalone function."""

    def test_perfect_experiment_scores_one(self):
        """An experiment that always achieves best_value scores 1.0."""
        result = auc_top_k(np.array([1.0, 1.0, 1.0, 1.0]), best_value=1.0)
        assert result["auc_top_k"] == pytest.approx(1.0)

    def test_returns_auc_top_k_key(self):
        """auc_top_k returns a dict with key 'auc_top_k'."""
        result = auc_top_k(np.array([0.5, 0.7, 0.8]), best_value=1.0)
        assert "auc_top_k" in result

    def test_value_in_range(self):
        """AUC is in [0, 1] for well-formed inputs."""
        result = auc_top_k(np.array([0.3, 0.5, 0.7, 0.9]), best_value=1.0)
        assert 0.0 <= result["auc_top_k"] <= 1.0

    def test_early_experiment_scores_higher_than_late(self):
        """An experiment that finds hits early scores higher AUC than one that finds them late."""
        early_bloom = auc_top_k(np.array([0.9, 0.85, 0.8, 0.75]), best_value=1.0)
        late_bloom = auc_top_k(np.array([0.1, 0.3, 0.7, 0.9]), best_value=1.0)
        assert early_bloom["auc_top_k"] > late_bloom["auc_top_k"]

    def test_single_round_raises(self):
        """Fewer than 2 rounds raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 rounds"):
            auc_top_k(np.array([0.9]), best_value=1.0)

    def test_zero_best_value_raises(self):
        """best_value of 0 raises ValueError."""
        with pytest.raises(ValueError, match="non-zero"):
            auc_top_k(np.array([0.5, 0.6]), best_value=0.0)

    def test_negative_best_value_raises(self):
        """Negative best_value raises ValueError."""
        with pytest.raises(ValueError, match="non-zero"):
            auc_top_k(np.array([0.5, 0.6]), best_value=-1.0)

    def test_round_values_exceeding_best_value_clamps_to_one(self):
        """When round_values exceed best_value the result is clamped to 1.0 with a warning."""
        with pytest.warns(UserWarning, match="best_value"):
            result = auc_top_k(np.array([1.2, 1.1, 1.05, 1.0]), best_value=1.0)
        assert result["auc_top_k"] == pytest.approx(1.0)

    def test_nan_best_value_raises(self):
        """best_value of NaN raises ValueError (NaN bypasses <= 0.0 guard)."""
        with pytest.raises(ValueError, match="non-zero"):
            auc_top_k(np.array([0.5, 0.6]), best_value=float("nan"))

    def test_negative_round_values_warns_and_clamps_to_zero(self):
        """Negative round_values emit a warning and the result is clamped to 0.0."""
        with pytest.warns(UserWarning, match="negative"):
            result = auc_top_k(np.array([-0.5, -0.3]), best_value=1.0)
        assert result["auc_top_k"] == pytest.approx(0.0)


class TestComputeAggregateMetrics:
    """Tests for the compute_aggregate_metrics entry point."""

    def test_runs_all_aggregate_metrics(self):
        """compute_aggregate_metrics merges the results of every aggregate metric."""
        result = compute_aggregate_metrics(np.array([0.5, 0.7, 0.8]), best_value=1.0)
        assert result == auc_top_k(np.array([0.5, 0.7, 0.8]), best_value=1.0)

    def test_skips_metrics_that_cannot_be_computed(self):
        """Metrics raising ValueError (e.g. too few rounds) are skipped, not raised."""
        assert compute_aggregate_metrics(np.array([0.9]), best_value=1.0) == {}
