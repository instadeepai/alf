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

"""Sequence processing utilities for ALF model implementations.

This module provides utilities for processing biological sequences (e.g., DNA, RNA, proteins)
in machine learning models. It handles common tasks like alphabet mapping, sequence extraction
from ALF data structures, and one-hot encoding.

Key Functionality:
    - Character-to-index mapping creation for sequence alphabets
    - Sequence extraction from ALF Candidate and LabelledCandidates objects
    - One-hot encoding of sequences with configurable output shape

Note:
    All sequences in a batch must have the same length for one-hot encoding.
    Characters not present in the alphabet will raise a ValueError.
"""

from typing import Literal, Union, overload

import numpy as np
import torch
from alf_core import Candidate, LabelledCandidates
from alf_core.utils.enums import ProblemType
from jaxtyping import Float


def create_char_to_idx_mapping(alphabet: str) -> dict[str, int]:
    """Create a character to index mapping from an alphabet.

    Args:
        alphabet: String containing all characters in the alphabet.

    Returns:
        Dictionary mapping each character to its index.
    """
    return {char: idx for idx, char in enumerate(alphabet)}


def extract_sequences_from_inputs(inputs: Union[LabelledCandidates, list[Candidate]]) -> list[str]:
    """Extract sequences from model inputs.

    Args:
        inputs: Either LabelledCandidates or list of Candidates.

    Returns:
        List of sequences as strings.

    Raises:
        ValueError: If input type is not supported.
    """
    if isinstance(inputs, LabelledCandidates):
        return inputs.data
    elif isinstance(inputs, list) and all(isinstance(c, Candidate) for c in inputs):
        return [c.data for c in inputs]
    else:
        raise ValueError(
            f"Inputs must be a LabelledCandidates or list[Candidate], got {type(inputs).__name__}"
        )


@overload
def one_hot_encode(
    sequences: list[str],
    char_to_idx: dict[str, int],
    alphabet_size: int,
    flatten: Literal[True],
) -> Float[torch.Tensor, "batch_size flat_features"]: ...


@overload
def one_hot_encode(
    sequences: list[str],
    char_to_idx: dict[str, int],
    alphabet_size: int,
    flatten: Literal[False] = ...,
) -> Float[torch.Tensor, "batch_size alphabet_size seq_length"]: ...


def one_hot_encode(
    sequences: list[str],
    char_to_idx: dict[str, int],
    alphabet_size: int,
    flatten: bool = False,
) -> (
    Float[torch.Tensor, "batch_size alphabet_size seq_length"]
    | Float[torch.Tensor, "batch_size flat_features"]
):
    """One-hot encode sequences.

    Args:
        sequences: List of sequences as strings.
        char_to_idx: Mapping from characters to indices.
        alphabet_size: Size of the alphabet.
        flatten: If True, flatten from (batch, alphabet_size, seq_length) to
            (batch, alphabet_size * seq_length). If False, keep 3D shape.

    Returns:
        One-hot encoded tensor. Shape depends on flatten parameter:
        - If flatten=True: (batch_size, alphabet_size * seq_length)
        - If flatten=False: (batch_size, alphabet_size, seq_length)

    Raises:
        ValueError: If sequences contain characters not in char_to_idx.
    """
    batch_size = len(sequences)
    seq_length = len(sequences[0])

    one_hot = torch.zeros(batch_size, alphabet_size, seq_length)

    for i, sequence in enumerate(sequences):
        for j, char in enumerate(sequence):
            if char in char_to_idx:
                one_hot[i, char_to_idx[char], j] = 1.0
            else:
                raise ValueError(
                    f"Character '{char}' not in alphabet. "
                    f"Available characters: {list(char_to_idx.keys())}"
                )

    if flatten:
        one_hot = one_hot.view(batch_size, -1)

    return one_hot


def determine_num_classes(labels: Float, problem_type: ProblemType) -> int:
    """Determine the number of output neurons based on problem type and labels.

    Args:
        labels (Float): Array of labels for the dataset.
        problem_type (ProblemType): Type of problem (REGRESSION, BINARY, MULTICLASS).

    Returns:
        int: Number of output neurons. For REGRESSION and BINARY, returns 1 and 2 respectively.
    """
    if problem_type == ProblemType.BINARY:
        return 2
    if problem_type == ProblemType.MULTICLASS:
        return int(np.max(labels)) + 1
    return 1
