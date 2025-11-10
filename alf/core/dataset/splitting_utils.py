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

from alf.core.dataclasses.labeled_candidates import LabeledCandidates


def split_dataset(
    split_type: str,
    dataset: LabeledCandidates,
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
) -> dict[str, LabeledCandidates]:
    """Split dataset into train, validation, test, and candidate pool.

    Args:
        split_type: Type of split to perform ("random" or "low_vs_high").
        dataset: Dataset to split.
        train_size: Number of samples for training set.
        validation_size: Number of samples for validation set.
        test_size: Number of samples for test set.
        seed: Random seed for reproducibility.

    Returns:
        dict[str, LabeledCandidates]: Dictionary with keys "train", "validation",
            "test", and "candidate_pool".

    Raises:
        ValueError: If split_type is not "random" or "low_vs_high".
    """
    if split_type == "random":
        return split_random(dataset, train_size, validation_size, test_size, seed)
    elif split_type == "low_vs_high":
        return split_low_vs_high(dataset, train_size, validation_size, test_size, seed)
    else:
        raise ValueError(f"Invalid split type: {split_type}")


def split_random(
    dataset: LabeledCandidates,
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
) -> dict[str, LabeledCandidates]:
    """Split dataset randomly into train, validation, test, and candidate pool.

    Shuffles the dataset and splits it sequentially into the specified sizes.
    Remaining samples go to the candidate pool.

    Args:
        dataset: Dataset to split.
        train_size: Number of samples for training set.
        validation_size: Number of samples for validation set.
        test_size: Number of samples for test set.
        seed: Random seed for shuffling.

    Returns:
        dict[str, LabeledCandidates]: Dictionary with keys "train", "validation",
            "test", and "candidate_pool".
    """
    shuffled_candidates = dataset.shuffle(seed=seed)

    start_idx = 0
    train = shuffled_candidates[start_idx : start_idx + train_size]
    start_idx += train_size

    validation = shuffled_candidates[start_idx : start_idx + validation_size]
    start_idx += validation_size

    test = shuffled_candidates[start_idx : start_idx + test_size]
    start_idx += test_size

    candidate_pool = shuffled_candidates[start_idx:]

    return {
        "train": train,
        "validation": validation,
        "test": test,
        "candidate_pool": candidate_pool,
    }


def split_low_vs_high(
    dataset: LabeledCandidates,
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
) -> dict[str, LabeledCandidates]:
    """Split dataset with low-scoring candidates in train/val, high-scoring in test/pool.

    Sorts candidates by label value, assigns low-scoring candidates to train/validation
    and high-scoring candidates to test/candidate_pool. Within each group, candidates
    are randomly shuffled.

    Args:
        dataset: Dataset to split.
        train_size: Number of samples for training set (from low-scoring group).
        validation_size: Number of samples for validation set (from low-scoring group).
        test_size: Number of samples for test set (from high-scoring group).
        seed: Random seed for shuffling within groups.

    Returns:
        dict[str, LabeledCandidates]: Dictionary with keys "train", "validation",
            "test", and "candidate_pool".
    """
    # Sort indices by label (highest to lowest)
    sorted_indices = dataset.labels.argsort()[::-1]

    # Low-scoring candidates go to train/validation, high-scoring to test/pool
    train_plus_validation_size = train_size + validation_size
    low_scoring_indices = sorted_indices[:train_plus_validation_size]
    high_scoring_indices = sorted_indices[train_plus_validation_size:]

    # Randomly shuffle within each group
    shuffled_low = dataset[np.random.RandomState(seed).permutation(low_scoring_indices)]
    shuffled_high = dataset[
        np.random.RandomState(seed).permutation(high_scoring_indices)
    ]

    return {
        "train": shuffled_low[:train_size],
        "validation": shuffled_low[train_size:],
        "test": shuffled_high[:test_size],
        "candidate_pool": shuffled_high[test_size:],
    }
