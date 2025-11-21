import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from alf.core.dataclasses.candidate import Candidate, Modality
from alf.core.dataclasses.predictions import Predictions


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


class TestPredictionsSavePredictions:
    """Test cases for Predictions save method."""

    def test_save_predictions_basic(self):
        """Test basic save functionality."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
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

            # Save predictions
            predictions.save(temp_dir, candidates, targets, "test_predictions.csv")

            # Check if file was created
            file_path = os.path.join(temp_dir, "test_predictions.csv")
            assert os.path.exists(file_path)

            # Load and verify content
            df = pd.read_csv(file_path)
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

    def test_save_predictions_with_empirical_dist(self):
        """Test save_predictions with empirical distribution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup test data
            means = np.array([1.0, 2.0])
            variances = np.array([0.1, 0.2])
            empirical_dist = np.array([[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]])
            predictions = Predictions(
                means=means, variances=variances, empirical_dist=empirical_dist
            )

            candidates = [
                Candidate(data="seq1", modality=Modality.SEQUENCE),
                Candidate(data="seq2", modality=Modality.SEQUENCE),
            ]
            targets = np.array([1.1, 2.1])

            # Save predictions
            predictions.save(temp_dir, candidates, targets, "test_empirical.csv")

            # Load and verify content
            file_path = os.path.join(temp_dir, "test_empirical.csv")
            df = pd.read_csv(file_path)

            # Check ensemble prediction columns
            assert "ensemble_pred_0" in df.columns
            assert "ensemble_pred_1" in df.columns
            assert "ensemble_pred_2" in df.columns

            # Check values
            np.testing.assert_array_almost_equal(df["ensemble_pred_0"].values, [1.1, 2.1])
            np.testing.assert_array_almost_equal(df["ensemble_pred_1"].values, [1.2, 2.2])
            np.testing.assert_array_almost_equal(df["ensemble_pred_2"].values, [1.3, 2.3])

    def test_save_predictions_without_variances(self):
        """Test save_predictions without variances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup test data
            means = np.array([1.0, 2.0, 3.0])
            predictions = Predictions(means=means)  # No variances

            candidates = [
                Candidate(data="seq1", modality=Modality.SEQUENCE),
                Candidate(data="seq2", modality=Modality.SEQUENCE),
                Candidate(data="seq3", modality=Modality.SEQUENCE),
            ]
            targets = np.array([1.1, 2.1, 3.1])

            # Save predictions
            predictions.save(temp_dir, candidates, targets, "test_no_var.csv")

            # Load and verify content
            file_path = os.path.join(temp_dir, "test_no_var.csv")
            df = pd.read_csv(file_path)

            # Check that variances are set to 0
            np.testing.assert_array_equal(df["variance"].values, [0, 0, 0])

    def test_save_predictions_without_empirical_dist(self):
        """Test save_predictions without empirical distribution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup test data
            means = np.array([1.0, 2.0])
            variances = np.array([0.1, 0.2])
            predictions = Predictions(means=means, variances=variances)  # No empirical_dist

            candidates = [
                Candidate(data="seq1", modality=Modality.SEQUENCE),
                Candidate(data="seq2", modality=Modality.SEQUENCE),
            ]
            targets = np.array([1.1, 2.1])

            # Save predictions
            predictions.save(temp_dir, candidates, targets, "test_no_emp.csv")

            # Load and verify content
            file_path = os.path.join(temp_dir, "test_no_emp.csv")
            df = pd.read_csv(file_path)

            # Check that no ensemble columns exist
            ensemble_cols = [col for col in df.columns if col.startswith("ensemble_pred_")]
            assert len(ensemble_cols) == 0
