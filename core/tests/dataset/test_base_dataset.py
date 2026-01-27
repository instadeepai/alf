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

"""Tests for the BaseDataset class."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from alf_core.dataclasses import Candidate, LabelledCandidates
from alf_core.dataclasses.candidate import Modality
from alf_core.dataset.base_dataset import BaseDataset, BaseDatasetConfig
from pydantic import ValidationError


class TestBaseDatasetInitialization:
    """Tests for BaseDataset initialization and configuration."""

    def test_initialization_with_valid_config(self, dummy_dataset_factory):
        """Test initialization with valid split configuration, without candidate pool."""
        dataset = dummy_dataset_factory(
            train_ratio=0.8,
            validation_frac=0.25,
            test_ratio=0.2,
            split_type="random",
            num_samples=100,
        )

        assert dataset.config.name == "dummy"
        assert dataset.modality == Modality.SEQUENCE
        assert dataset.config.seed == 42
        assert dataset.split_ratio == {"train": 0.8, "validation_frac": 0.25, "test": 0.2}
        assert dataset.config.split_type == "random"

    def test_initialization_with_candidate_pool(self, dummy_dataset_factory):
        """Test initialization with candidate pool in config."""
        dataset = dummy_dataset_factory(
            train_ratio=0.6,
            validation_frac=0.3,
            test_ratio=0.2,
            split_type="random",
            max_candidate_pool=20,
            num_samples=100,
        )

        assert dataset.config.max_candidate_pool == 20


class TestBaseDatasetValidation:
    """Tests for BaseDataset configuration validation."""

    def test_validate_config_ratios_sum_exceeds_one(self, dummy_dataset_factory):
        """Test that split ratios summing to more than 1 raises an error."""
        with pytest.raises(ValidationError, match="train_ratio \\+ test_ratio must be <= 1"):
            dummy_dataset_factory(
                train_ratio=0.8,
                validation_frac=0.5,
                test_ratio=0.6,
                split_type="random",
                num_samples=100,
            )

    def test_validate_config_invalid_train_ratio(self, dummy_dataset_factory):
        """Test that invalid train_ratio raises an error."""
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            dummy_dataset_factory(
                train_ratio=1.5,
                validation_frac=0.2,
                test_ratio=0.2,
                split_type="random",
                num_samples=100,
            )

    def test_validate_config_invalid_modality(self):
        """Test that invalid modality raises an error."""
        with pytest.raises(ValidationError, match="Input should be"):
            BaseDatasetConfig(
                name="test",
                modality="invalid_modality",
                seed=42,
                train_ratio=0.6,
                validation_frac=0.2,
                test_ratio=0.2,
                split_type="random",
            )


class TestBaseDatasetSplitting:
    """Tests for dataset splitting functionality."""

    def test_split_dataset_random(self, dummy_dataset_factory):
        """Test random splitting of dataset."""
        dataset = dummy_dataset_factory(
            train_ratio=0.6,
            validation_frac=0.25,
            test_ratio=0.2,
            split_type="random",
            num_samples=100,
        )

        # train+val = 60, of which 25% (15) goes to validation, 75% (45) to train
        assert len(dataset.train_dataset) == 45
        assert len(dataset.validation_dataset) == 15
        assert len(dataset.test_dataset) == 20

    def test_split_dataset_with_candidate_pool(self, dummy_dataset_factory):
        """Test splitting with explicit candidate pool."""
        dataset = dummy_dataset_factory(
            train_ratio=0.5,
            validation_frac=0.2,
            test_ratio=0.25,
            split_type="random",
            max_candidate_pool=25,
            num_samples=100,
        )

        # train+val = 50, of which 20% (10) goes to validation, 80% (40) to train
        assert len(dataset.train_dataset) == 40
        assert len(dataset.validation_dataset) == 10
        assert len(dataset.test_dataset) == 25
        assert len(dataset.candidate_pool) == 25

    def test_split_dataset_low_vs_high(self, dummy_dataset_factory):
        """Test low vs high splitting of dataset."""
        dataset = dummy_dataset_factory(
            train_ratio=0.5,
            validation_frac=0.2,
            test_ratio=0.25,
            split_type="low_vs_high",
            num_samples=100,
        )

        # Check sizes: train+val = 50, of which 20% (10) goes to validation, 80% (40) to train
        assert len(dataset.train_dataset) == 40
        assert len(dataset.validation_dataset) == 10
        assert len(dataset.test_dataset) == 25
        assert len(dataset.candidate_pool) == 25

        # Check that train/val have lower scores than test/pool on average
        train_val_mean = (
            np.mean(dataset.train_dataset.labels) + np.mean(dataset.validation_dataset.labels)
        ) / 2
        test_pool_mean = (
            np.mean(dataset.test_dataset.labels) + np.mean(dataset.candidate_pool.labels)
        ) / 2
        assert test_pool_mean >= train_val_mean

    def test_init_candidate_pool_is_deepcopy(self, dummy_dataset_factory):
        """Test that init_candidate_pool is a deep copy of the original candidate pool."""
        dataset = dummy_dataset_factory(
            train_ratio=0.5,
            validation_frac=0.2,
            test_ratio=0.25,
            split_type="random",
            num_samples=100,
        )

        # Verify init_candidate_pool exists
        assert hasattr(dataset, "init_candidate_pool")
        assert len(dataset.init_candidate_pool) == len(dataset.candidate_pool)

        # Modify current pool
        original_pool_size = len(dataset.candidate_pool)
        dataset.splits["candidate_pool"].labels[0] = 999.0

        # Check that init pool is unaffected
        assert dataset.init_candidate_pool.labels[0] != 999.0
        assert len(dataset.init_candidate_pool) == original_pool_size


class TestBaseDatasetProperties:
    """Tests for BaseDataset properties."""

    def test_train_dataset_property(self, dummy_dataset_factory):
        """Test train_dataset property."""
        dataset = dummy_dataset_factory(num_samples=100)
        train = dataset.train_dataset

        assert isinstance(train, LabelledCandidates)
        assert len(train) == 48

    def test_validation_dataset_property(self, dummy_dataset_factory):
        """Test validation_dataset property."""
        dataset = dummy_dataset_factory(num_samples=100)
        validation = dataset.validation_dataset

        assert isinstance(validation, LabelledCandidates)
        assert len(validation) == 12

    def test_test_dataset_property(self, dummy_dataset_factory):
        """Test test_dataset property."""
        dataset = dummy_dataset_factory(num_samples=100)
        test = dataset.test_dataset

        assert isinstance(test, LabelledCandidates)
        assert len(test) == 20

    def test_candidate_pool_property(self, dummy_dataset_factory):
        """Test candidate_pool property."""
        dataset = dummy_dataset_factory(
            train_ratio=0.6,
            validation_frac=0.25,
            test_ratio=0.2,
            split_type="random",
            num_samples=100,
        )
        pool = dataset.candidate_pool

        assert isinstance(pool, LabelledCandidates)
        # When no candidate_pool specified, it gets the remainder
        # train+val = 60 (45 train, 15 val), test = 20, pool = 100 - 60 - 20 = 20
        assert len(pool) == 20

    def test_property_access_before_split_raises_error(self):
        """Test that accessing properties before splitting raises an assertion error."""

        # Create a custom dataset that doesn't auto-setup
        class NoSetupDataset(BaseDataset):
            def load_dataset(self):
                return LabelledCandidates(
                    candidates=[
                        Candidate(data=f"seq_{i}", modality="sequence") for i in range(100)
                    ],
                    labels=np.random.rand(100),
                )

        config = BaseDatasetConfig(
            name="test",
            modality="sequence",
            seed=42,
            train_ratio=0.6,
            validation_frac=0.25,
            test_ratio=0.2,
            split_type="random",
        )
        dataset = NoSetupDataset(config)

        with pytest.raises(AssertionError, match="Dataset must be split before accessing"):
            _ = dataset.train_dataset


class TestBaseDatasetUpdateSplits:
    """Tests for updating dataset splits with acquired candidates."""

    def test_update_splits_with_acquired_candidates(self, dummy_dataset_factory):
        """Test that update_splits correctly moves candidates from pool to train/val."""
        dataset = dummy_dataset_factory(
            train_ratio=0.5,
            validation_frac=0.2,
            test_ratio=0.25,
            split_type="random",
            num_samples=100,
        )

        # Get initial sizes
        initial_train_size = len(dataset.train_dataset)
        initial_val_size = len(dataset.validation_dataset)
        initial_pool_size = len(dataset.candidate_pool)

        # Acquire some candidates from the pool
        num_acquired = 5
        acquired_candidates = LabelledCandidates(*dataset.candidate_pool[:num_acquired])

        # Update splits
        dataset.update_splits(acquired_candidates)

        # Check that pool size decreased
        assert len(dataset.candidate_pool) == initial_pool_size - num_acquired

        # Check that train + val size increased by num_acquired
        new_train_size = len(dataset.train_dataset)
        new_val_size = len(dataset.validation_dataset)
        assert (new_train_size + new_val_size) == (
            initial_train_size + initial_val_size + num_acquired
        )

    def test_update_splits_respects_validation_ratio(self, dummy_dataset_factory):
        """Test that update_splits maintains the validation ratio."""
        dataset = dummy_dataset_factory(
            train_ratio=0.5,
            validation_frac=0.2,
            test_ratio=0.25,
            split_type="random",
            num_samples=100,
        )

        # Acquire 10 candidates
        acquired_candidates = LabelledCandidates(*dataset.candidate_pool[:10])
        initial_val_size = len(dataset.validation_dataset)

        dataset.update_splits(acquired_candidates)

        # With validation_frac of 0.2, expect 2 of the 10 to go to validation (10 * 0.2 = 2)
        expected_new_val = int(10 * dataset.split_ratio["validation_frac"])
        actual_new_val = len(dataset.validation_dataset) - initial_val_size
        assert actual_new_val == expected_new_val


class TestBaseDatasetQuery:
    """Tests for querying labels for candidates."""

    def test_query_returns_correct_labels(self, dummy_dataset_factory):
        """Test that query returns the correct labels for given candidates."""
        dataset = dummy_dataset_factory(num_samples=100)

        # Get some candidates from train set
        candidates_to_query = dataset.train_dataset.candidates[:5]

        # Query their labels
        result = dataset.query(candidates_to_query)

        assert isinstance(result, LabelledCandidates)
        assert len(result) == 5
        assert np.array_equal(result.candidates, candidates_to_query)

        # Verify labels match the original data
        for i, candidate in enumerate(candidates_to_query):
            expected_idx = dataset._raw_dataset.data.index(candidate.data)
            expected_label = dataset._raw_dataset.labels[expected_idx]
            assert result.labels[i] == expected_label


class TestBaseDatasetSaveSplits:
    """Tests for saving dataset splits."""

    def test_save_splits_creates_files(self, dummy_dataset_factory):
        """Test that save_splits creates the expected CSV files."""
        dataset = dummy_dataset_factory(num_samples=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset.save_splits(tmpdir)

            # Check that files were created
            assert os.path.exists(os.path.join(tmpdir, "data_splits", "train.csv"))
            assert os.path.exists(os.path.join(tmpdir, "data_splits", "validation.csv"))
            assert os.path.exists(os.path.join(tmpdir, "data_splits", "test.csv"))
            assert os.path.exists(os.path.join(tmpdir, "data_splits", "candidate_pool.csv"))

    def test_save_splits_file_contents(self, dummy_dataset_factory):
        """Test that saved files contain the correct data."""
        dataset = dummy_dataset_factory(num_samples=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset.save_splits(tmpdir)

            # Read back the train file and verify
            train_df = pd.read_csv(os.path.join(tmpdir, "data_splits", "train.csv"))

            # Check that number of rows matches
            assert len(train_df) == len(dataset.train_dataset)


class TestBaseDatasetGetMetrics:
    """Tests for getting dataset metrics."""

    def test_get_metrics_returns_correct_keys(self, dummy_dataset_factory):
        """Test that get_metrics returns metrics for all splits."""
        dataset = dummy_dataset_factory(
            train_ratio=0.5,
            validation_frac=0.2,
            test_ratio=0.25,
            split_type="random",
            num_samples=100,
        )

        metrics = dataset.get_metrics()

        # Check that all expected keys are present
        expected_keys = [
            "num_train",
            "train_mean",
            "num_validation",
            "validation_mean",
            "num_test",
            "test_mean",
            "num_candidate_pool",
            "candidate_pool_mean",
        ]
        for key in expected_keys:
            assert key in metrics

    def test_get_metrics_returns_correct_values(self, dummy_dataset_factory):
        """Test that get_metrics returns correct metric values."""
        dataset = dummy_dataset_factory(num_samples=100)

        metrics = dataset.get_metrics()

        # Check counts
        assert metrics["num_train"] == len(dataset.train_dataset)
        assert metrics["num_validation"] == len(dataset.validation_dataset)
        assert metrics["num_test"] == len(dataset.test_dataset)

        # Check means
        assert np.isclose(metrics["train_mean"], np.mean(dataset.train_dataset.labels))
        assert np.isclose(metrics["validation_mean"], np.mean(dataset.validation_dataset.labels))
        assert np.isclose(metrics["test_mean"], np.mean(dataset.test_dataset.labels))


class TestBaseDatasetReproducibility:
    """Tests for reproducibility with different seeds."""

    def test_different_seeds_produce_different_splits(self, dummy_dataset_factory):
        """Test that different seeds produce different random splits."""
        dataset1 = dummy_dataset_factory(seed=42, num_samples=100)
        dataset2 = dummy_dataset_factory(seed=123, num_samples=100)

        # Check that train sets are different
        train1_labels = dataset1.train_dataset.labels
        train2_labels = dataset2.train_dataset.labels

        assert not np.array_equal(train1_labels, train2_labels)

    def test_same_seed_produces_same_splits(self, dummy_dataset_factory):
        """Test that the same seed produces identical splits."""
        dataset1 = dummy_dataset_factory(seed=42, num_samples=100)
        dataset2 = dummy_dataset_factory(seed=42, num_samples=100)

        # Check that train sets are identical
        train1_labels = dataset1.train_dataset.labels
        train2_labels = dataset2.train_dataset.labels

        assert np.array_equal(train1_labels, train2_labels)
