from alf.core.dataclasses.labeled_candidates import LabeledCandidates
import numpy as np

def split_dataset(split_type: str, dataset: LabeledCandidates, train_size: int, validation_size: int, test_size: int, seed: int) -> dict[str, LabeledCandidates]:
    """Split dataset into train, validation, test, and candidate pool."""
    if split_type == "random":
        return split_random(dataset, train_size, validation_size, test_size, seed)
    elif split_type == "low_vs_high":
        return split_low_vs_high(dataset, train_size, validation_size, test_size, seed)
    else:
        raise ValueError(f"Invalid split type: {split_type}")


def split_random(dataset: LabeledCandidates, train_size: int, validation_size: int, test_size: int, seed: int) -> dict[str, LabeledCandidates]:
    """Split dataset randomly into train, validation, test, and candidate pool."""
    shuffled_candidates = dataset.shuffle(seed=seed)
    
    start_idx = 0
    train = shuffled_candidates[start_idx:start_idx + train_size]
    start_idx += train_size
    
    validation = shuffled_candidates[start_idx:start_idx + validation_size]
    start_idx += validation_size
    
    test = shuffled_candidates[start_idx:start_idx + test_size]
    start_idx += test_size
    
    candidate_pool = shuffled_candidates[start_idx:start_idx + candidate_pool_size]
    
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "candidate_pool": candidate_pool
    }


def split_low_vs_high(dataset: LabeledCandidates, train_size: int, validation_size: int, test_size: int, seed: int) -> dict[str, LabeledCandidates]:
    """Split dataset so train/validation contain low-scoring candidates, test/pool contain high-scoring."""
    # Sort indices by label (highest to lowest)
    sorted_indices = dataset.labels.argsort()[::-1]
    
    # Low-scoring candidates go to train/validation, high-scoring to test/pool
    train_plus_validation_size = train_size + validation_size
    low_scoring_indices = sorted_indices[:train_plus_validation_size]
    high_scoring_indices = sorted_indices[train_plus_validation_size:]
    
    # Randomly shuffle within each group
    shuffled_low = dataset[np.random.RandomState(seed).permutation(low_scoring_indices)]
    shuffled_high = dataset[np.random.RandomState(seed).permutation(high_scoring_indices)]
        
    return {
        "train": shuffled_low[:train_size],
        "validation": shuffled_low[train_size:],
        "test": shuffled_high[:test_size],
        "candidate_pool": shuffled_high[test_size:]
    }
