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


from typing import Literal, get_args

import numpy as np
from alf_core.dataclasses.labelled_candidates import LabelledCandidates

SplitType = Literal["random", "low_vs_high", "stratified"]
ALLOWED_SPLIT_TYPES = get_args(SplitType)


def split_dataset(
    split_type: SplitType,
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
        ValueError: If split_type is not one of the supported split types.
    """
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

    Raises:
        ValueError: If any split requests more samples than remain after earlier splits.
    """
    labels = dataset.labels.astype(int)
    non_integer_mask = ~np.isclose(dataset.labels, dataset.labels.astype(int).astype(float))
    assert np.allclose(dataset.labels, labels.astype(float)), (
        f"split_type='stratified' requires integer class labels, but {non_integer_mask.sum()} "
        f"labels have fractional parts (e.g. {dataset.labels[non_integer_mask][:3]}). "
        f"Cast your labels to int or use split_type='random'."
    )
    rng = np.random.RandomState(seed)

    classes = np.unique(labels)
    split_names = ["train", "validation", "test", "candidate_pool"]
    split_sizes = [train_size, validation_size, test_size, candidate_pool_size]

    # Ensure that split ratios do not leave out any samples.
    # Shuffle each class's indices once, then track how many have been consumed.
    cls_idxs = {cls: rng.permutation(np.where(labels == cls)[0]) for cls in classes}
    cls_used = {cls: 0 for cls in classes}

    split_indices: dict[str, list[int]] = {name: [] for name in split_names}

    for name, S in zip(split_names, split_sizes):
        # Remaining samples available per class at this point in the loop.
        avail = [len(cls_idxs[cls]) - cls_used[cls] for cls in classes]
        total_avail = sum(avail)

        if S == 0:
            continue
        if total_avail < S:
            raise ValueError(
                f"split_stratified: requested {S} samples for '{name}' "
                f"but only {total_avail} remain."
            )

        # Largest remainder method: allocate exactly S samples proportional to
        # remaining class availability, guaranteeing no rounding waste within the split.
        exact = [a / total_avail * S for a in avail]
        alloc = [int(e) for e in exact]
        remainders = [e - a for e, a in zip(exact, alloc)]
        leftover = S - sum(alloc)
        for i in sorted(range(len(classes)), key=lambda i: -remainders[i])[:leftover]:
            alloc[i] += 1

        for cls, n in zip(classes, alloc):
            start = cls_used[cls]
            split_indices[name].extend(cls_idxs[cls][start : start + n].tolist())
            cls_used[cls] += n

    return {
        key: LabelledCandidates(*dataset[np.array(indices, dtype=int)])
        for key, indices in split_indices.items()
    }
