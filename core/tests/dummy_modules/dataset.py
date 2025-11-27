from typing import Any

import numpy as np
from alf_core.dataclasses import Candidate, LabeledCandidates
from alf_core.dataset.base_dataset import BaseDataset


class DummyDataset(BaseDataset):
    """Dummy dataset that generates random data for testing."""

    def __init__(
        self,
        name: str = "dummy",
        modality: str = "sequence",
        seed: int = 42,
        split_config: dict[str, Any] | None = None,
        num_samples: int = 1000,
    ):
        """Initialize a dummy dataset.

        Args:
            name: Name of the dataset
            modality: Modality of the data (e.g., "sequence")
            seed: Random seed for reproducibility
            split_config: Split configuration dictionary
            num_samples: Number of dummy samples to generate
        """
        # Default split config if none provided
        if split_config is None:
            split_config = {
                "split_ratio": {"train": 0.6, "validation_frac": 0.2, "test": 0.2},
                "split_type": "random",
            }
        super().__init__(name, modality, seed, split_config)
        self.num_samples = num_samples
        self.setup()

    def load_dataset(self) -> LabeledCandidates:
        """Generate dummy dataset with random sequences and labels.

        Returns:
            LabeledCandidates with dummy data
        """
        # Generate dummy sequences (e.g., random strings)
        candidates = []
        labels = []

        for i in range(self.num_samples):
            # Generate a dummy sequence (e.g., random string of length 10)
            dummy_sequence = "".join(self.rng.choice(list("ACGT"), size=10))

            # Generate a dummy label (random float between 0 and 10)
            dummy_label = float(self.rng.uniform(0, 10))

            candidates.append(Candidate(data=dummy_sequence, modality=self.modality))
            labels.append(dummy_label)

        return LabeledCandidates(candidates=candidates, labels=np.array(labels))
