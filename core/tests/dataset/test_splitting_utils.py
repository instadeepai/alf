# Copyright 2026 InstaDeep Ltd. All rights reserved.
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

"""Unit tests for stratified dataset splitting."""

import numpy as np
import pytest
from alf_core.dataclasses import Candidate, LabelledCandidates
from alf_core.dataclasses.candidate import Modality
from alf_core.dataset.base_dataset import BaseDatasetConfig
from alf_core.dataset.splitting_utils import split_dataset, split_stratified
from alf_core.utils.enums import ProblemType
from pydantic import ValidationError


def make_dataset(labels: list[int]) -> LabelledCandidates:
    """Create a LabelledCandidates with dummy sequence candidates for the given labels.

    Returns:
        A LabelledCandidates with one sequence candidate per label.
    """
    candidates = [
        Candidate(data="ACGT"[i % 4] * 4, modality=Modality.SEQUENCE) for i in range(len(labels))
    ]
    return LabelledCandidates(candidates=candidates, labels=np.array(labels, dtype=float))


class TestSplitStratified:
    """Tests for split_stratified function."""

    def test_class_proportions_preserved_in_train(self):
        """Test that class proportions in the training split match the original dataset."""
        # 60 class-0, 40 class-1 → 60/40 split
        labels = [0] * 60 + [1] * 40
        dataset = make_dataset(labels)
        splits = split_stratified(
            dataset,
            train_size=60,
            validation_size=20,
            test_size=10,
            candidate_pool_size=10,
            seed=42,
        )
        train_labels = splits["train"].labels.astype(int)
        n0 = (train_labels == 0).sum()
        # Expect roughly 36 class-0 in train (60% of 60)
        assert abs(n0 / len(train_labels) - 0.6) < 0.05

    def test_all_splits_returned(self):
        """Test that all four split keys are returned."""
        labels = [0] * 50 + [1] * 50
        dataset = make_dataset(labels)
        splits = split_stratified(
            dataset,
            train_size=40,
            validation_size=20,
            test_size=20,
            candidate_pool_size=10,
            seed=0,
        )
        assert set(splits.keys()) == {"train", "validation", "test", "candidate_pool"}

    def test_multiclass_proportions(self):
        """Test that all classes are represented in the training split for multiclass data."""
        # 3 classes, equal counts
        labels = [0] * 30 + [1] * 30 + [2] * 30
        dataset = make_dataset(labels)
        splits = split_stratified(
            dataset, train_size=30, validation_size=15, test_size=15, candidate_pool_size=15, seed=7
        )
        train_labels = splits["train"].labels.astype(int)
        for cls in range(3):
            count = (train_labels == cls).sum()
            assert count > 0, f"Class {cls} missing from train split"

    def test_reproducible_with_same_seed(self):
        """Test that the same seed produces identical splits."""
        labels = [0] * 50 + [1] * 50
        dataset = make_dataset(labels)
        splits_a = split_stratified(dataset, 40, 20, 20, 10, seed=99)
        splits_b = split_stratified(dataset, 40, 20, 20, 10, seed=99)
        np.testing.assert_array_equal(splits_a["train"].labels, splits_b["train"].labels)

    def test_different_seeds_differ(self):
        """Test that different seeds produce different splits."""
        labels = [0] * 50 + [1] * 50
        dataset = make_dataset(labels)
        splits_a = split_stratified(dataset, 40, 20, 20, 10, seed=1)
        splits_b = split_stratified(dataset, 40, 20, 20, 10, seed=2)
        # Candidate data within each class should be shuffled differently per seed
        cands_a = [c.data for c in splits_a["train"].candidates]
        cands_b = [c.data for c in splits_b["train"].candidates]
        assert cands_a != cands_b

    def test_stratified_split_sizes_clamped(self):
        """Test that stratified split sizes are clamped to the requested values."""
        labels = [0] * 3 + [1] * 3 + [2] * 3
        dataset = make_dataset(labels)
        requested = dict(train_size=2, validation_size=2, test_size=2, candidate_pool_size=2)
        splits = split_stratified(dataset, **requested, seed=0)

        assert len(splits["train"]) == requested["train_size"]
        assert len(splits["validation"]) == requested["validation_size"]
        assert len(splits["test"]) == requested["test_size"]
        assert len(splits["candidate_pool"]) == requested["candidate_pool_size"]
        assert sum(len(split) for split in splits.values()) == sum(requested.values())

        # Test that shuffling is working
        assert splits["train"].labels.astype(int).tolist() != [0, 0]


class TestBaseDatasetConfigStratifiedValidator:
    """Tests for the Pydantic validator guarding stratified + REGRESSION."""

    def _base_kwargs(self):
        """Return base kwargs for constructing a BaseDatasetConfig."""
        return dict(
            name="test",
            modality=Modality.SEQUENCE,
            seed=0,
            train_ratio=0.6,
            validation_frac=0.1,
            test_ratio=0.2,
        )

    def test_stratified_with_binary_ok(self):
        """Test that stratified split is accepted with BINARY problem type."""
        config = BaseDatasetConfig(
            **self._base_kwargs(),
            split_type="stratified",
            problem_type=ProblemType.BINARY,
        )
        assert config.split_type == "stratified"

    def test_stratified_with_multiclass_ok(self):
        """Test that stratified split is accepted with MULTICLASS problem type."""
        config = BaseDatasetConfig(
            **self._base_kwargs(),
            split_type="stratified",
            problem_type=ProblemType.MULTICLASS,
        )
        assert config.split_type == "stratified"

    def test_stratified_with_regression_raises(self):
        """Test that stratified split raises a ValidationError with REGRESSION problem type."""
        with pytest.raises(ValidationError, match="stratified"):
            BaseDatasetConfig(
                **self._base_kwargs(),
                split_type="stratified",
                problem_type=ProblemType.REGRESSION,
            )

    def test_random_with_regression_ok(self):
        """Test that random split is accepted with REGRESSION problem type."""
        config = BaseDatasetConfig(
            **self._base_kwargs(), split_type="random", problem_type=ProblemType.REGRESSION
        )
        assert config.problem_type == ProblemType.REGRESSION

    def test_string_coercion_to_problem_type(self):
        """Test that string values are coerced to the corresponding ProblemType enum member."""
        config = BaseDatasetConfig(**self._base_kwargs(), problem_type="regression")
        assert config.problem_type == ProblemType.REGRESSION
        config = BaseDatasetConfig(
            **self._base_kwargs(), split_type="stratified", problem_type="binary"
        )
        assert config.problem_type == ProblemType.BINARY
        config = BaseDatasetConfig(
            **self._base_kwargs(), split_type="stratified", problem_type="multiclass"
        )
        assert config.problem_type == ProblemType.MULTICLASS

    def test_invalid_split_type_raises(self):
        """Test that an unknown split type raises a ValidationError."""
        with pytest.raises(ValidationError):
            BaseDatasetConfig(**self._base_kwargs(), split_type="unknown")


def test_split_dataset_invalid_split_type_string():
    """Test that passing an invalid split type string
    raises a ValueError with an informative message.
    """
    dataset = make_dataset([0, 1, 2])
    with pytest.raises(ValueError, match="Expected one of"):
        split_dataset("not-a-number", dataset, 1, 1, 1, 0, seed=0)


def test_split_dataset_invalid_split_type_type():
    """Test that passing a non-string split type raises a TypeError with an informative message."""
    dataset = make_dataset([0, 1, 2])
    with pytest.raises(ValueError, match="Invalid split type"):
        split_dataset(123, dataset, 1, 1, 1, 0, seed=0)
