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


from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class Modality(Enum):
    """Enum for different data modalities."""

    SEQUENCE = "sequence"
    IMAGE = "image"
    GRAPH = "graph"
    STRUCTURE = "structure"
    TABULAR = "tabular"
    EMBEDDING = "embedding"


@dataclass(eq=False, unsafe_hash=False)
class Candidate:
    """A candidate is a data point with a modality and features.

    Attributes:
        data: The raw data of the candidate (e.g., sequence string, graph, image).
        modality: The type/modality of the data (e.g., "sequence", "graph", "image").
        features: Optional dictionary of precomputed features for the candidate.
    """

    data: Any
    modality: Modality
    features: dict | None = None

    def __post_init__(self) -> None:
        """Check and convert modality to Modality enum if necessary and
        initialize features to empty dict if None.

        Raises:
            ValueError: If the modality is not a valid Modality enum.
        """
        if not isinstance(self.modality, Modality):
            try:
                self.modality = Modality(self.modality)
            except ValueError:
                raise ValueError(f"Invalid modality: {self.modality}")

        if self.features is None:
            self.features = {}

    def __repr__(self) -> str:
        """Return a string representation of the candidate.

        Returns:
            A string representation showing the candidate's data, modality, and features.
        """
        return f"Candidate(data={self.data}, modality={self.modality}, features={self.features})"

    def stringify(self) -> str:
        """Convert the candidate data to a string representation.

        Returns:
            The string representation of the candidate's data.

        Raises:
            ValueError: If the modality is not "sequence" (other modalities not yet supported).
        """
        if self.modality == Modality.SEQUENCE:
            return self.data
        else:
            # TODO: Implement stringification for other modalities
            # Once implemented, add test cases to test_candidate.py
            raise ValueError(f"Unsupported modality: {self.modality}")

    def _safe_equal(self, a: Any, b: Any) -> bool:
        """Compare two values, handling numpy arrays and nested structures.

        Args:
            a: First value to compare.
            b: Second value to compare.

        Returns:
            True if the values are equal, False otherwise.
        """
        # Handle None cases
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False

        # Handle numpy arrays
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np.array_equal(a, b, equal_nan=True)

        # Handle torch tensors
        if HAS_TORCH and isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return torch.equal(a, b)

        # Handle dict (for features)
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                return False
            return all(self._safe_equal(a[k], b[k]) for k in a.keys())

        # Handle lists/tuples (for nested data)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False
            return all(self._safe_equal(x, y) for x, y in zip(a, b))

        # Default comparison
        try:
            result = a == b
            # Handle case where comparison returns array-like object
            # Convert to boolean if possible
            if hasattr(result, "__len__") and len(result) == 1:
                return bool(result[0])
            elif hasattr(result, "item"):  # For single-element tensors/arrays
                return bool(result.item())
            return bool(result)
        except (ValueError, TypeError, RuntimeError):
            # If comparison fails (e.g., unexpected numpy arrays or incompatible types)
            return False

    def __eq__(self, other: object) -> bool:
        """Compare two Candidate objects for equality.

        Handles numpy arrays in data and features fields correctly.

        Args:
            other: The object to compare with.

        Returns:
            True if the candidates are equal, False otherwise.
        """
        if not isinstance(other, Candidate):
            return False

        return (
            self._safe_equal(self.data, other.data)
            and self.modality == other.modality
            and self._safe_equal(self.features, other.features)
        )

    __hash__ = None  # type: ignore[assignment]
