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

from typing import Union

import numpy as np
import torch
from alf_core import Candidate, LabelledCandidates
from alf_core.model.normaliser import (
    InputNormaliser,
    OutputStandardiser,
    fit_input_normaliser,
    fit_output_standardiser,
)
from torch import dtype as TorchDtype


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
        raise ValueError("Input must be LabelledCandidates or list of Candidates")


def one_hot_encode(
    sequences: list[str],
    char_to_idx: dict[str, int],
    alphabet_size: int,
    flatten: bool = False,
) -> torch.Tensor:
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


def transform_data(
    train_x: torch.Tensor,
    labels: np.ndarray,
    normalise_inputs: bool,
    standardise_outputs: bool,
    label_dtype: TorchDtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, InputNormaliser | None, OutputStandardiser | None]:
    """Apply input normalisation and output standardisation to training data.

    Args:
        train_x (torch.Tensor): Training features tensor.
        labels (np.ndarray): Training labels array.
        normalise_inputs (bool): Whether to apply min-max normalisation to input features.
        standardise_outputs (bool): Whether to apply Z-score standardisation to output labels.
        label_dtype (TorchDtype): Data type for the output labels.
        device (torch.device): Device to move tensors to.

    Returns:
        tuple[torch.Tensor, torch.Tensor, InputNormaliser | None, OutputStandardiser | None]:
            Transformed training features, training labels, and the fitted normalisers.
    """
    input_normaliser = None
    output_standardiser = None
    if normalise_inputs:
        train_x_np = np.array(train_x.cpu())
        train_x_np, input_normaliser = fit_input_normaliser(train_x_np)
        train_x = torch.tensor(train_x_np, dtype=torch.float32).to(device)
    else:
        train_x = train_x.to(device)

    if standardise_outputs:
        labels, output_standardiser = fit_output_standardiser(labels)
        train_y = torch.tensor(labels, dtype=label_dtype).to(device)
    else:
        train_y = torch.tensor(labels, dtype=label_dtype).to(device)
    return train_x, train_y, input_normaliser, output_standardiser
