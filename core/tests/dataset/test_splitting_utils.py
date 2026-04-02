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

"""Unit tests for stratified dataset splitting."""

import numpy as np
import pytest
from alf_core.dataclasses import Candidate, LabelledCandidates
from alf_core.dataclasses.candidate import Modality
from alf_core.dataset.base_dataset import BaseDatasetConfig
from alf_core.dataset.splitting_utils import split_stratified
from alf_core.enums import ProblemType
from pydantic import ValidationError


def make_dataset(labels: list[int]) -> LabelledCandidates:
    """Helper to create a LabelledCandidates with dummy sequence candidates."""
    candidates = [
        Candidate(data=f"ACGT"[i % 4] * 4, modality=Modality.SEQUENCE)
        for i in range(len(labels))
    ]
    return LabelledCandidates(candidates=candidates, labels=np.array(labels, dtype=float))


class TestSplitStratified:
    """Tests for split_stratified function."""

    def test_class_proportions_preserved_in_train(self):
        # 60 class-0, 40 class-1 → 60/40 split
        labels = [0] * 60 + [1] * 40
        dataset = make_dataset(labels)
        splits = split_stratified(dataset, train_size=60, validation_size=20, test_size=10, candidate_pool_size=10, seed=42)
        train_labels = splits["train"].labels.astype(int)
        n0 = (train_labels == 0).sum()
        n1 = (train_labels == 1).sum()
        # Expect roughly 36 class-0 and 24 class-1 in train (60% vs 40%)
        assert abs(n0 / len(train_labels) - 0.6) < 0.05

    def test_all_splits_returned(self):
        labels = [0] * 50 + [1] * 50
        dataset = make_dataset(labels)
        splits = split_stratified(dataset, train_size=40, validation_size=20, test_size=20, candidate_pool_size=10, seed=0)
        assert set(splits.keys()) == {"train", "validation", "test", "candidate_pool"}

    def test_multiclass_proportions(self):
        # 3 classes, equal counts
        labels = [0] * 30 + [1] * 30 + [2] * 30
        dataset = make_dataset(labels)
        splits = split_stratified(dataset, train_size=30, validation_size=15, test_size=15, candidate_pool_size=15, seed=7)
        train_labels = splits["train"].labels.astype(int)
        for cls in range(3):
            count = (train_labels == cls).sum()
            assert count > 0, f"Class {cls} missing from train split"

    def test_reproducible_with_same_seed(self):
        labels = [0] * 50 + [1] * 50
        dataset = make_dataset(labels)
        splits_a = split_stratified(dataset, 40, 20, 20, 10, seed=99)
        splits_b = split_stratified(dataset, 40, 20, 20, 10, seed=99)
        np.testing.assert_array_equal(splits_a["train"].labels, splits_b["train"].labels)

    def test_different_seeds_differ(self):
        labels = [0] * 50 + [1] * 50
        dataset = make_dataset(labels)
        splits_a = split_stratified(dataset, 40, 20, 20, 10, seed=1)
        splits_b = split_stratified(dataset, 40, 20, 20, 10, seed=2)
        # Candidate data within each class should be shuffled differently per seed
        cands_a = [c.data for c in splits_a["train"].candidates]
        cands_b = [c.data for c in splits_b["train"].candidates]
        assert cands_a != cands_b


class TestBaseDatasetConfigStratifiedValidator:
    """Tests for the Pydantic validator guarding stratified + REGRESSION."""

    def _base_kwargs(self):
        return dict(
            name="test",
            modality=Modality.SEQUENCE,
            seed=0,
            train_ratio=0.6,
            validation_frac=0.1,
            test_ratio=0.2,
        )

    def test_stratified_with_binary_ok(self):
        config = BaseDatasetConfig(
            **self._base_kwargs(),
            split_type="stratified",
            problem_type=ProblemType.BINARY,
        )
        assert config.split_type == "stratified"

    def test_stratified_with_multiclass_ok(self):
        config = BaseDatasetConfig(
            **self._base_kwargs(),
            split_type="stratified",
            problem_type=ProblemType.MULTICLASS,
        )
        assert config.split_type == "stratified"

    def test_stratified_with_regression_raises(self):
        with pytest.raises(ValidationError, match="stratified"):
            BaseDatasetConfig(
                **self._base_kwargs(),
                split_type="stratified",
                problem_type=ProblemType.REGRESSION,
            )

    def test_random_with_regression_ok(self):
        config = BaseDatasetConfig(**self._base_kwargs(), split_type="random", problem_type=ProblemType.REGRESSION)
        assert config.problem_type == ProblemType.REGRESSION

    def test_string_coercion_to_problem_type(self):
        config = BaseDatasetConfig(**self._base_kwargs(), problem_type="regression")
        assert config.problem_type == ProblemType.REGRESSION
        config = BaseDatasetConfig(**self._base_kwargs(), split_type="stratified", problem_type="binary")
        assert config.problem_type == ProblemType.BINARY
        config = BaseDatasetConfig(**self._base_kwargs(), split_type="stratified", problem_type="multiclass")
        assert config.problem_type == ProblemType.MULTICLASS

    def test_invalid_split_type_raises(self):
        with pytest.raises(ValidationError):
            BaseDatasetConfig(**self._base_kwargs(), split_type="unknown")
