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
from alf_core.dataclasses.candidate import Candidate, Modality
from alf_core.dataclasses.predictions import Predictions


class TestPredictionsInitialization:
    """Test cases for Predictions initialization and basic functionality."""

    def test_predictions_with_means_only(self):
        """Test Predictions initialization with only means."""
        means = np.array([1.0, 2.0, 3.0])
        predictions = Predictions(means=means)

        np.testing.assert_array_equal(predictions.means, means)
        assert predictions.variances is None
        assert predictions.empirical_dist is None

    def test_predictions_with_means_and_variances(self):
        """Test Predictions initialization with means and variances."""
        means = np.array([1.0, 2.0, 3.0])
        variances = np.array([0.1, 0.2, 0.3])
        predictions = Predictions(means=means, variances=variances)

        np.testing.assert_array_equal(predictions.means, means)
        np.testing.assert_array_equal(predictions.variances, variances)
        assert predictions.empirical_dist is None

    def test_predictions_with_all_parameters(self):
        """Test Predictions initialization with all parameters."""
        means = np.array([1.0, 2.0, 3.0])
        variances = np.array([0.1, 0.2, 0.3])
        empirical_dist = np.array([[1.1, 1.2], [2.1, 2.2], [3.1, 3.2]])
        predictions = Predictions(means=means, variances=variances, empirical_dist=empirical_dist)

        np.testing.assert_array_equal(predictions.means, means)
        np.testing.assert_array_equal(predictions.variances, variances)
        np.testing.assert_array_equal(predictions.empirical_dist, empirical_dist)

    def test_predictions_with_empty_means(self):
        """Test Predictions initialization with empty means array."""
        means = np.array([])
        with pytest.raises(AssertionError):
            Predictions(means=means)

    def test_predictions_with_large_arrays(self):
        """Test Predictions initialization with large arrays."""
        n_predictions = 1000
        means = np.random.randn(n_predictions)
        variances = np.random.rand(n_predictions)
        empirical_dist = np.random.randn(n_predictions, 5)  # 5 ensemble models

        predictions = Predictions(means=means, variances=variances, empirical_dist=empirical_dist)

        np.testing.assert_array_equal(predictions.means, means)
        np.testing.assert_array_equal(predictions.variances, variances)
        np.testing.assert_array_equal(predictions.empirical_dist, empirical_dist)


class TestPredictionsValidation:
    """Test cases for Predictions validation and error handling."""

    def test_predictions_requires_means(self):
        """Test that Predictions requires means parameter."""
        with pytest.raises(TypeError):
            Predictions()

    def test_predictions_with_mismatched_variances_array_lengths(self):
        """Test Predictions with mismatched array lengths between means and variances."""
        means = np.array([1.0, 2.0, 3.0])
        variances = np.array([0.1, 0.2])  # Different length

        with pytest.raises(AssertionError):
            Predictions(means=means, variances=variances)

    def test_predictions_with_mismatched_empirical_dist_array_lengths(self):
        """Test Predictions with mismatched array lengths between means and
        empirical distribution.
        """
        means = np.array([1.0, 2.0, 3.0])
        variances = np.array([0.1, 0.2, 0.3])
        empirical_dist = np.array([[1.1, 1.2], [2.1, 2.2]])
        with pytest.raises(AssertionError):
            Predictions(means=means, variances=variances, empirical_dist=empirical_dist)


class TestPredictionsToDataFrame:
    """Test cases for Predictions to_dataframe method."""

    def test_to_dataframe_basic(self):
        """Test basic to_dataframe functionality."""
        # Setup test data
        means = np.array([1.0, 2.0, 3.0])
        variances = np.array([0.1, 0.2, 0.3])
        predictions = Predictions(means=means, variances=variances)

        candidates = [
            Candidate(data="seq1", modality=Modality.SEQUENCE),
            Candidate(data="seq2", modality=Modality.SEQUENCE),
            Candidate(data="seq3", modality=Modality.SEQUENCE),
        ]
        targets = np.array([1.1, 2.1, 3.1])

        # Convert to DataFrame
        df = predictions.to_dataframe(candidates, targets)

        # Verify DataFrame structure
        assert len(df) == 3
        assert "sequence" in df.columns
        assert "mean" in df.columns
        assert "variance" in df.columns
        assert "targets" in df.columns

        # Check values
        np.testing.assert_array_equal(df["sequence"].values, ["seq1", "seq2", "seq3"])
        np.testing.assert_array_almost_equal(df["mean"].values, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(df["variance"].values, [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(df["targets"].values, [1.1, 2.1, 3.1])

    def test_to_dataframe_with_empirical_dist(self):
        """Test to_dataframe with empirical distribution."""
        # Setup test data
        means = np.array([1.0, 2.0])
        variances = np.array([0.1, 0.2])
        empirical_dist = np.array([[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]])
        predictions = Predictions(means=means, variances=variances, empirical_dist=empirical_dist)

        candidates = [
            Candidate(data="seq1", modality=Modality.SEQUENCE),
            Candidate(data="seq2", modality=Modality.SEQUENCE),
        ]
        targets = np.array([1.1, 2.1])

        # Convert to DataFrame
        df = predictions.to_dataframe(candidates, targets)

        # Check ensemble prediction columns
        assert "ensemble_pred_0" in df.columns
        assert "ensemble_pred_1" in df.columns
        assert "ensemble_pred_2" in df.columns

        # Check values
        np.testing.assert_array_almost_equal(df["ensemble_pred_0"].values, [1.1, 2.1])
        np.testing.assert_array_almost_equal(df["ensemble_pred_1"].values, [1.2, 2.2])
        np.testing.assert_array_almost_equal(df["ensemble_pred_2"].values, [1.3, 2.3])

    def test_to_dataframe_without_variances(self):
        """Test to_dataframe without variances."""
        # Setup test data
        means = np.array([1.0, 2.0, 3.0])
        predictions = Predictions(means=means)  # No variances

        candidates = [
            Candidate(data="seq1", modality=Modality.SEQUENCE),
            Candidate(data="seq2", modality=Modality.SEQUENCE),
            Candidate(data="seq3", modality=Modality.SEQUENCE),
        ]
        targets = np.array([1.1, 2.1, 3.1])

        # Convert to DataFrame
        df = predictions.to_dataframe(candidates, targets)

        # Check that variances are set to 0
        np.testing.assert_array_equal(df["variance"].values, [0, 0, 0])

    def test_to_dataframe_without_empirical_dist(self):
        """Test to_dataframe without empirical distribution."""
        # Setup test data
        means = np.array([1.0, 2.0])
        variances = np.array([0.1, 0.2])
        predictions = Predictions(means=means, variances=variances)  # No empirical_dist

        candidates = [
            Candidate(data="seq1", modality=Modality.SEQUENCE),
            Candidate(data="seq2", modality=Modality.SEQUENCE),
        ]
        targets = np.array([1.1, 2.1])

        # Convert to DataFrame
        df = predictions.to_dataframe(candidates, targets)

        # Check that no ensemble columns exist
        ensemble_cols = [col for col in df.columns if col.startswith("ensemble_pred_")]
        assert len(ensemble_cols) == 0


class TestPredictionsToDataframeClassification:
    """Predictions.to_dataframe() expands 2D means into prob_class_N columns."""

    def _make_candidates(self, n: int):
        return [Candidate(data=f"SEQ{i}", modality=Modality.SEQUENCE) for i in range(n)]

    def test_binary_columns(self):
        probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]])
        targets = np.array([0.0, 1.0, 0.0, 1.0])
        preds = Predictions(means=probs)
        df = preds.to_dataframe(self._make_candidates(4), targets)
        assert "prob_class_0" in df.columns
        assert "prob_class_1" in df.columns
        assert "mean" not in df.columns
        assert "variance" not in df.columns

    def test_multiclass_columns(self):
        probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]])
        targets = np.array([0.0, 1.0, 2.0])
        preds = Predictions(means=probs)
        df = preds.to_dataframe(self._make_candidates(3), targets)
        assert "prob_class_0" in df.columns
        assert "prob_class_1" in df.columns
        assert "prob_class_2" in df.columns

    def test_regression_columns_unchanged(self):
        means = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.1, 1.9, 3.1])
        preds = Predictions(means=means)
        df = preds.to_dataframe(self._make_candidates(3), targets)
        assert "mean" in df.columns
        assert "variance" in df.columns
        assert "prob_class_0" not in df.columns

    def test_prob_values_correct(self):
        probs = np.array([[0.3, 0.7], [0.8, 0.2]])
        targets = np.array([1.0, 0.0])
        preds = Predictions(means=probs)
        df = preds.to_dataframe(self._make_candidates(2), targets)
        assert df["prob_class_0"].tolist() == pytest.approx([0.3, 0.8])
        assert df["prob_class_1"].tolist() == pytest.approx([0.7, 0.2])
