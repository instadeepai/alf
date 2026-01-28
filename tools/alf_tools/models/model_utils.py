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

"""Common utilities for model implementations."""

from __future__ import annotations

from typing import Any, Union

import torch
from alf_core import Candidate, LabelledCandidates
from jaxtyping import Float


def create_char_to_idx_mapping(alphabet: str) -> dict[str, int]:
    """Create a character to index mapping from an alphabet.

    Args:
        alphabet: String containing all characters in the alphabet.

    Returns:
        Dictionary mapping each character to its index.
    """
    return {char: idx for idx, char in enumerate(alphabet)}


def get_device(device: str | None = None) -> torch.device:
    """Get the appropriate torch device.

    Args:
        device: Device specification ('cuda', 'cpu', or None for auto-detect).

    Returns:
        torch.device object.
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def extract_sequences_from_inputs(inputs: Union[LabelledCandidates, list[Candidate]]) -> Any:
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
        raise ValueError("Input must be LabelledCandidates or list of Candidates")


def one_hot_encode(
    sequences: list[str],
    char_to_idx: dict[str, int],
    alphabet_size: int,
    flatten: bool = False,
) -> Float[torch.Tensor, "batch_size ..."]:
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

    # Flatten if requested
    if flatten:
        one_hot = one_hot.view(batch_size, -1)

    return one_hot
