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


import io
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeAlias, Union

import numpy as np

if TYPE_CHECKING:
    import torch

    DataFrameCompatible: TypeAlias = Union[
        str, int, float, bool, dict, list, tuple, np.ndarray, torch.Tensor
    ]
else:
    DataFrameCompatible: TypeAlias = Union[
        str, int, float, bool, dict, list, tuple, np.ndarray, Any
    ]

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from ase import Atoms as _AseAtoms
    from ase.io import write as _ase_write

    HAS_ASE = True
except ImportError:
    HAS_ASE = False


class Modality(Enum):
    """Enum for different data modalities."""

    SEQUENCE = "sequence"
    GRAPH = "graph"
    STRUCTURE = "structure"
    TABULAR = "tabular"


@dataclass(eq=False, unsafe_hash=False)
class Candidate:
    """A candidate is a data point with a modality and features.

    Attributes:
        data: The raw data of the candidate (e.g., sequence string, graph, image).
        modality: The type/modality of the data (e.g., "sequence", "graph").
        features: Optional dictionary of precomputed features for the candidate.
    """

    data: Any
    modality: Modality | str
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

    def _convert_data_to_npy(self, accepted_description: str) -> np.ndarray:
        """Convert data to a numpy array if it is a torch tensor or numpy array.

        Args:
            accepted_description: Description of accepted types, used in the TypeError message.

        Returns:
            numpy array representation of the data.

        Raises:
            TypeError: If data is neither a numpy array nor a torch tensor.
        """
        if HAS_TORCH and isinstance(self.data, torch.Tensor):
            return self.data.cpu().numpy()
        if isinstance(self.data, np.ndarray):
            return self.data
        raise TypeError(f"{accepted_description} Got: {type(self.data).__name__}")

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
        # Note, if comparison fails, an error will be thrown
        result = a == b
        # Handle case where comparison returns array-like object
        # Convert to boolean if possible
        if hasattr(result, "__len__") and len(result) == 1:
            return bool(result[0])
        elif hasattr(result, "item"):  # For single-element tensors/arrays
            return bool(result.item())
        return bool(result)

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

    def to_serializable(self) -> DataFrameCompatible | None:
        """Convert candidate data to a format suitable for pandas DataFrame storage.

        This method transforms the candidate's data into a format that can be efficiently
        stored in a pandas DataFrame column. The conversion strategy varies by modality:

        - SEQUENCE: Returns stringified data for efficient string storage
        - TABULAR: Validates and returns data (scalar, dict, numpy array, pandas Series,
          list, tuple, or torch tensor). Torch tensors are converted to numpy arrays.
        - STRUCTURE: Converts torch tensors/arrays to numpy arrays; strings are passed
          through as-is to support serialised representations (e.g. JSON-encoded crystal
          structures); ASE Atoms are serialised losslessly to a JSON string
        - GRAPH: Not yet supported (raises NotImplementedError)

        Returns:
            DataFrameCompatible: The candidate data in a DataFrame-compatible format.
                Common types include str, dict, np.ndarray, pd.Series, or torch.Tensor.

        Raises:
            NotImplementedError: If modality is GRAPH (not yet supported).
            ValueError: If modality is unknown or not recognized.
            TypeError: If TABULAR or STRUCTURE modality data is not a supported
                DataFrame-compatible type.

        Examples:
            >>> # Sequence modality
            >>> candidate = Candidate(data="ACDEFG", modality=Modality.SEQUENCE)
            >>> candidate.to_serializable()
            'ACDEFG'

            >>> # Structure modality with JSON-encoded crystal structure
            >>> json_str = '{"lattice": [[3.84, 0, 0], [0, 3.84, 0], [0, 0, 3.84]]}'
            >>> candidate = Candidate(data=json_str, modality=Modality.STRUCTURE)
            >>> candidate.to_serializable()
            '{"lattice": [[3.84, 0, 0], [0, 3.84, 0], [0, 0, 3.84]]}'

            >>> # Tabular modality
            >>> candidate = Candidate(data={"age": 32, "height": 178}, modality=Modality.TABULAR)
            >>> candidate.to_serializable()
            {'age': 32, 'height': 178}
        """
        # Handle None data
        if self.data is None:
            return None
        if self.modality == Modality.SEQUENCE:
            return str(self.data)  # Efficient string storage

        elif self.modality == Modality.TABULAR:
            # Validate and return tabular data
            # Acceptable types: scalar values, dict, numpy arrays, pandas Series, lists/tuples
            if isinstance(self.data, (str, int, float, bool, dict, np.ndarray, list, tuple)):
                return self.data

            # Check for pandas Series (without requiring pandas import)
            if hasattr(self.data, "__class__") and self.data.__class__.__name__ == "Series":
                return self.data

            # Check for torch tensor - convert to numpy
            if HAS_TORCH and isinstance(self.data, torch.Tensor):
                return self.data.cpu().numpy()

            # If we reach here, the type is not supported
            raise TypeError(
                f"TABULAR modality data must be a scalar (str, int, float, bool), "
                f"dict, numpy array, pandas Series, list, or tuple. "
                f"Got: {type(self.data).__name__}"
            )

        elif self.modality == Modality.STRUCTURE:
            # Strings are passed through to support serialised representations
            # (e.g. JSON-encoded crystal structures); arrays/tensors are converted to numpy;
            # ASE Atoms are serialised losslessly to a JSON string.
            if isinstance(self.data, str):
                return self.data
            if isinstance(self.data, np.ndarray):
                return self.data
            if HAS_TORCH and isinstance(self.data, torch.Tensor):
                return self.data.cpu().numpy()
            if HAS_ASE and isinstance(self.data, _AseAtoms):
                buffer = io.StringIO()
                _ase_write(buffer, self.data, format="json")
                return buffer.getvalue()
            raise TypeError(
                "STRUCTURE modality data must be a string, numpy array, torch tensor, or ASE Atoms."
            )

        elif self.modality == Modality.GRAPH:
            raise NotImplementedError("Graph datatype not supported yet")

        raise ValueError(f"Unknown modality: {self.modality}")
