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

"""Tests for sequence utility functions."""

import numpy as np
import pytest
from alf_core import Candidate, LabelledCandidates
from alf_tools.models.utils import (
    create_char_to_idx_mapping,
    extract_sequences_from_inputs,
    one_hot_encode,
)
from alf_tools.utils.constants import PROTEIN_ALPHABET


class TestSequenceUtils:
    """Test suite for sequence utility functions."""

    def test_create_char_to_idx_mapping(self):
        """Test character to index mapping creation."""
        alphabet = "ABC"
        mapping = create_char_to_idx_mapping(alphabet)

        assert len(mapping) == 3
        assert mapping["A"] == 0
        assert mapping["B"] == 1
        assert mapping["C"] == 2

    def test_create_char_to_idx_mapping_protein(self):
        """Test mapping for protein alphabet."""
        mapping = create_char_to_idx_mapping(PROTEIN_ALPHABET)

        assert len(mapping) == 20
        assert all(char in mapping for char in PROTEIN_ALPHABET)

    def test_extract_sequences_from_labeled_candidates(self):
        """Test extracting sequences from LabelledCandidates."""
        sequences = ["AAA", "BBB", "CCC"]
        candidates = [Candidate(data=seq, modality="sequence") for seq in sequences]
        labeled = LabelledCandidates(candidates, np.array([1.0, 2.0, 3.0]))

        extracted = extract_sequences_from_inputs(labeled)

        assert extracted == sequences

    def test_extract_sequences_from_candidate_list(self):
        """Test extracting sequences from list of Candidates."""
        sequences = ["AAA", "BBB", "CCC"]
        candidates = [Candidate(data=seq, modality="sequence") for seq in sequences]

        extracted = extract_sequences_from_inputs(candidates)

        assert extracted == sequences

    def test_extract_sequences_invalid_input(self):
        """Test that invalid input raises ValueError."""
        with pytest.raises(ValueError, match="Inputs must be"):
            extract_sequences_from_inputs("invalid")

        with pytest.raises(ValueError, match="Inputs must be"):
            extract_sequences_from_inputs(["not", "candidates"])

    def test_one_hot_encode_basic(self):
        """Test basic one-hot encoding."""
        sequences = ["ABC", "ACB"]
        alphabet = "ABC"
        char_to_idx = create_char_to_idx_mapping(alphabet)

        encoded = one_hot_encode(sequences, char_to_idx, len(alphabet), flatten=False)

        assert encoded.shape == (2, 3, 3)  # batch=2, alphabet=3, seq_len=3
        # Check first sequence "ABC"
        assert encoded[0, 0, 0] == 1  # A at position 0
        assert encoded[0, 1, 1] == 1  # B at position 1
        assert encoded[0, 2, 2] == 1  # C at position 2

    def test_one_hot_encode_flattened(self):
        """Test one-hot encoding with flattening."""
        sequences = ["ABC"]
        alphabet = "ABC"
        char_to_idx = create_char_to_idx_mapping(alphabet)

        encoded = one_hot_encode(sequences, char_to_idx, len(alphabet), flatten=True)

        assert encoded.shape == (1, 9)  # batch=1, alphabet*seq_len=3*3=9

    def test_one_hot_encode_invalid_character(self):
        """Test that invalid character raises ValueError."""
        sequences = ["ABD"]  # D not in alphabet
        alphabet = "ABC"
        char_to_idx = create_char_to_idx_mapping(alphabet)

        with pytest.raises(ValueError, match="Character 'D' not in alphabet"):
            one_hot_encode(sequences, char_to_idx, len(alphabet), flatten=False)

    def test_one_hot_encode_protein_alphabet(self):
        """Test one-hot encoding with protein alphabet."""
        sequences = ["ACDEFG"]
        char_to_idx = create_char_to_idx_mapping(PROTEIN_ALPHABET)

        encoded = one_hot_encode(
            sequences,
            char_to_idx,
            len(PROTEIN_ALPHABET),
            flatten=False,
        )

        assert encoded.shape == (1, 20, 6)  # batch=1, alphabet=20, seq_len=6
        # Check that exactly one position is hot per sequence position
        for pos in range(6):
            assert encoded[0, :, pos].sum() == 1

    def test_one_hot_encode_batch(self):
        """Test one-hot encoding with batch of sequences."""
        sequences = ["ACE", "DEF", "GHI"]
        char_to_idx = create_char_to_idx_mapping(PROTEIN_ALPHABET)

        encoded = one_hot_encode(
            sequences,
            char_to_idx,
            len(PROTEIN_ALPHABET),
            flatten=False,
        )

        assert encoded.shape == (3, 20, 3)  # batch=3, alphabet=20, seq_len=3
        # Check that each sequence is properly encoded
        for i in range(3):
            for pos in range(3):
                assert encoded[i, :, pos].sum() == 1
