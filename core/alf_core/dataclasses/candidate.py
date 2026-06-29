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


class Modality(Enum):
    """The *kind* of candidate — used to match datasets with compatible models and metrics.

    Modality answers "what kind of thing is this candidate?" so the framework can pair a
    dataset with models that can consume it and pick an appropriate diversity metric. For
    candidates that have a meaningful domain this is that domain (a protein sequence, a
    molecule); for unstructured numeric inputs it is the representation itself (a feature
    vector). It deliberately does **not** describe how the data is *stored*: a protein
    sequence and a SMILES string are both Python ``str`` objects but are different
    modalities (``SEQUENCE`` vs ``MOLECULE``). The storage type is inferred separately from
    ``type(data)`` where it matters (see :meth:`Candidate.to_serializable`).

    Members:
        SEQUENCE: Biological sequences (protein / nucleotide), stored as strings.
        MOLECULE: Small molecules, stored as SMILES strings.
        TABULAR: The domain-agnostic case — fixed-length numeric feature vectors
            (arrays, tensors, scalars, dicts), e.g. for synthetic or pre-featurised
            continuous optimization with no richer domain.
    """

    SEQUENCE = "sequence"
    MOLECULE = "molecule"
    TABULAR = "tabular"


@dataclass(eq=False, unsafe_hash=False)
class Candidate:
    """A candidate is a data point with a modality and features.

    Attributes:
        data: The raw data of the candidate (e.g., a sequence string, a SMILES string,
            a feature vector).
        modality: The data's domain (see :class:`Modality`) — what it represents, e.g.
            ``"sequence"`` or ``"molecule"``. Not its storage type.
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

        Dispatch is based on the **type** of ``data``, not on :attr:`modality`. Modality
        describes the data's domain (see :class:`Modality`); how it is stored — and
        therefore how it is serialised — is determined by its Python type:

        - ``str`` (sequences, SMILES, JSON-encoded payloads): returned unchanged.
        - ``torch.Tensor``: converted to a numpy array for compact storage.
        - numpy array, scalar (``int``/``float``/``bool``), ``dict``, ``list``, ``tuple``,
          pandas ``Series``: returned unchanged.
        - ``None``: returned as ``None``.

        Returns:
            DataFrameCompatible: The candidate data in a DataFrame-compatible format.
                Common types include str, dict, np.ndarray, pd.Series, or torch.Tensor.

        Raises:
            TypeError: If the data type is not DataFrame-compatible.

        Examples:
            >>> # A sequence or SMILES string is stored as-is
            >>> Candidate(data="ACDEFG", modality=Modality.SEQUENCE).to_serializable()
            'ACDEFG'

            >>> Candidate(data="CC(=O)O", modality=Modality.MOLECULE).to_serializable()
            'CC(=O)O'

            >>> # A feature dict is stored as-is
            >>> c = Candidate(data={"age": 32, "height": 178}, modality=Modality.TABULAR)
            >>> c.to_serializable()
            {'age': 32, 'height': 178}
        """
        data = self.data
        if data is None:
            return None
        # Strings (sequences, SMILES, JSON-encoded payloads) are stored as-is.
        if isinstance(data, str):
            return data
        # Torch tensors are converted to numpy arrays for compact storage.
        if HAS_TORCH and isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        # numpy arrays, scalars, and standard containers are stored as-is.
        if isinstance(data, (int, float, bool, dict, np.ndarray, list, tuple)):
            return data
        # pandas Series (checked without importing pandas).
        if data.__class__.__name__ == "Series":
            return data

        raise TypeError(
            f"Cannot serialise candidate data of type {type(data).__name__}. Supported "
            f"types: str, int, float, bool, dict, list, tuple, numpy.ndarray, "
            f"pandas.Series, or torch.Tensor."
        )
