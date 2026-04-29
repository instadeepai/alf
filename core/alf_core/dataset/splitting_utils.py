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


from typing import Literal
import numpy as np
from alf_core.dataclasses.labelled_candidates import LabelledCandidates

ALLOWED_SPLIT_TYPES = ("random", "low_vs_high", "stratified")


def split_dataset(
    split_type: Literal["random", "low_vs_high", "stratified"],
    dataset: LabelledCandidates,
    train_size: int,
    validation_size: int,
    test_size: int,
    candidate_pool_size: int,
    seed: int,
) -> dict[str, LabelledCandidates]:
    """Split dataset into train, validation, test, and candidate pool.

    Args:
        split_type: Type of split to perform ("random", "low_vs_high", or "stratified").
        dataset: Dataset to split.
        train_size: Number of samples for training set.
        validation_size: Number of samples for validation set.
        test_size: Number of samples for test set.
        candidate_pool_size: Number of samples for candidate pool.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys "train", "validation", "test", and "candidate_pool".

    Raises:
        TypeError: If split_type is not a string.
        ValueError: If split_type is not one of the supported split types.
    """
    if not isinstance(split_type, str):
        raise TypeError(
            f"split_type must be one of {ALLOWED_SPLIT_TYPES}, got {type(split_type).__name__}"
        )
    if split_type not in ALLOWED_SPLIT_TYPES:
        raise ValueError(
            f"Invalid split type: {split_type!r}. Expected one of {ALLOWED_SPLIT_TYPES}."
        )
    if split_type == "random":
        return split_random(
            dataset, train_size, validation_size, test_size, candidate_pool_size, seed
        )
    elif split_type == "low_vs_high":
        return split_low_vs_high(
            dataset, train_size, validation_size, test_size, candidate_pool_size, seed
        )
    elif split_type == "stratified":
        return split_stratified(
            dataset, train_size, validation_size, test_size, candidate_pool_size, seed
        )
    else:
        raise ValueError(f"Invalid split type: {split_type}")


def split_random(
    dataset: LabelledCandidates,
    train_size: int,
    validation_size: int,
    test_size: int,
    candidate_pool_size: int,
    seed: int,
) -> dict[str, LabelledCandidates]:
    """Split dataset randomly into train, validation, test, and candidate pool.

    Shuffles the dataset and splits it sequentially into the specified sizes.
    Remaining samples go to the candidate pool.

    Args:
        dataset: Dataset to split.
        train_size: Number of samples for training set.
        validation_size: Number of samples for validation set.
        test_size: Number of samples for test set.
        candidate_pool_size: Number of samples for candidate pool.
        seed: Random seed for shuffling.

    Returns:
        Dictionary with keys "train", "validation", "test", and "candidate_pool".
    """
    shuffled_candidates = dataset.shuffle(seed=seed)

    start_idx = 0
    train = LabelledCandidates(*shuffled_candidates[start_idx : start_idx + train_size])
    start_idx += train_size

    validation = LabelledCandidates(*shuffled_candidates[start_idx : start_idx + validation_size])
    start_idx += validation_size

    test = LabelledCandidates(*shuffled_candidates[start_idx : start_idx + test_size])
    start_idx += test_size

    candidate_pool = LabelledCandidates(
        *shuffled_candidates[start_idx : start_idx + candidate_pool_size]
    )

    return {
        "train": train,
        "validation": validation,
        "test": test,
        "candidate_pool": candidate_pool,
    }


def split_low_vs_high(
    dataset: LabelledCandidates,
    train_size: int,
    validation_size: int,
    test_size: int,
    candidate_pool_size: int,
    seed: int,
) -> dict[str, LabelledCandidates]:
    """Split dataset with low-scoring candidates in train/val, high-scoring in test/pool.

    Sorts candidates by label value, assigns low-scoring candidates to train/validation
    and high-scoring candidates to test/candidate_pool. Within each group, candidates
    are randomly shuffled.

    Args:
        dataset: Dataset to split.
        train_size: Number of samples for training set (from low-scoring group).
        validation_size: Number of samples for validation set (from low-scoring group).
        test_size: Number of samples for test set (from high-scoring group).
        candidate_pool_size: Number of samples for candidate pool (from high-scoring group).
        seed: Random seed for shuffling within groups.

    Returns:
        Dictionary with keys "train", "validation", "test", and "candidate_pool".
    """
    # Sort indices by label (lowest to highest)
    sorted_indices = dataset.labels.argsort()

    # Low-scoring candidates go to train/validation, high-scoring to test/pool
    train_plus_validation_size = train_size + validation_size
    low_scoring_indices = sorted_indices[:train_plus_validation_size]
    high_scoring_indices = sorted_indices[train_plus_validation_size:]

    # Randomly shuffle within each group
    shuffled_low = LabelledCandidates(
        *dataset[np.random.RandomState(seed).permutation(low_scoring_indices)]
    )
    shuffled_high = LabelledCandidates(
        *dataset[np.random.RandomState(seed).permutation(high_scoring_indices)]
    )

    return {
        "train": LabelledCandidates(*shuffled_low[:train_size]),
        "validation": LabelledCandidates(*shuffled_low[train_size:]),
        "test": LabelledCandidates(*shuffled_high[:test_size]),
        "candidate_pool": LabelledCandidates(
            *shuffled_high[test_size : test_size + candidate_pool_size]
        ),
    }


def split_stratified(
    dataset: LabelledCandidates,
    train_size: int,
    validation_size: int,
    test_size: int,
    candidate_pool_size: int,
    seed: int,
) -> dict[str, LabelledCandidates]:
    """Split dataset preserving class proportions across all splits.

    For each class in the dataset, samples are proportionally distributed
    across train, validation, test, and candidate pool splits. Within each
    class the samples are randomly shuffled before assignment.

    Args:
        dataset: Dataset to split. Labels must be integer class indices.
        train_size: Total number of samples for training set.
        validation_size: Total number of samples for validation set.
        test_size: Total number of samples for test set.
        candidate_pool_size: Total number of samples for candidate pool.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys "train", "validation", "test", and "candidate_pool".
    """
    labels = dataset.labels.astype(int)
    n = len(labels)
    rng = np.random.RandomState(seed)

    classes = np.unique(labels)
    split_indices: dict[str, list[int]] = {
        "train": [],
        "validation": [],
        "test": [],
        "candidate_pool": [],
    }

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        cls_idx = rng.permutation(cls_idx)
        frac = len(cls_idx) / n

        n_train = round(frac * train_size)
        n_val = round(frac * validation_size)
        n_test = round(frac * test_size)
        n_pool = round(frac * candidate_pool_size)

        i = 0
        split_indices["train"].extend(cls_idx[i : i + n_train].tolist())
        i += n_train
        split_indices["validation"].extend(cls_idx[i : i + n_val].tolist())
        i += n_val
        split_indices["test"].extend(cls_idx[i : i + n_test].tolist())
        i += n_test
        split_indices["candidate_pool"].extend(cls_idx[i : i + n_pool].tolist())

    return {
        key: LabelledCandidates(*dataset[np.array(indices)])
        for key, indices in split_indices.items()
    }
