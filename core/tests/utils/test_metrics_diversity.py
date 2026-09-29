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

"""Tests for intra-batch diversity metric."""

import numpy as np
import pytest
from alf_core.dataclasses.candidate import Candidate, Modality
from alf_core.utils.metrics.acquisition_batch import intra_batch_diversity


def _seq(s: str) -> Candidate:
    return Candidate(data=s, modality=Modality.SEQUENCE)


def _emb(arr: np.ndarray) -> Candidate:
    return Candidate(data=arr, modality=Modality.TABULAR)


class TestIntraBatchDiversitySequence:
    """Tests for intra_batch_diversity with SEQUENCE candidates."""

    def test_identical_sequences_zero_diversity(self):
        """Identical sequences give 0.0 diversity."""
        candidates = [_seq("ACGT"), _seq("ACGT"), _seq("ACGT")]
        result = intra_batch_diversity(candidates)
        assert result["intra_batch_diversity"] == pytest.approx(0.0)

    def test_completely_different_sequences_high_diversity(self):
        """Completely different sequences give non-zero diversity."""
        candidates = [_seq("AAAA"), _seq("CCCC"), _seq("TTTT")]
        result = intra_batch_diversity(candidates)
        assert result["intra_batch_diversity"] > 0.0

    def test_value_in_zero_one_range(self):
        """Diversity value is in [0, 1]."""
        candidates = [_seq("ACGT"), _seq("TGCA"), _seq("ATAT")]
        result = intra_batch_diversity(candidates)
        assert 0.0 <= result["intra_batch_diversity"] <= 1.0

    def test_returns_correct_key(self):
        """Returns dict with key 'intra_batch_diversity'."""
        result = intra_batch_diversity([_seq("ABC"), _seq("DEF")])
        assert "intra_batch_diversity" in result

    def test_two_candidates_returns_single_distance(self):
        """Two candidates compute exactly one pairwise distance."""
        result = intra_batch_diversity([_seq("AAAA"), _seq("TTTT")])
        assert isinstance(result["intra_batch_diversity"], float)


class TestIntraBatchDiversityTabular:
    """Tests for intra_batch_diversity with TABULAR candidates."""

    def test_identical_embeddings_zero_cosine_distance(self):
        """Identical embeddings give 0.0 cosine distance."""
        vec = np.array([1.0, 0.0, 0.0])
        candidates = [_emb(vec), _emb(vec)]
        result = intra_batch_diversity(candidates)
        assert result["intra_batch_diversity"] == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_embeddings_max_cosine_distance(self):
        """Orthogonal embeddings give cosine distance of 1.0."""
        candidates = [_emb(np.array([1.0, 0.0])), _emb(np.array([0.0, 1.0]))]
        result = intra_batch_diversity(candidates)
        assert result["intra_batch_diversity"] == pytest.approx(1.0, abs=1e-6)

    def test_value_non_negative(self):
        """Diversity value is non-negative for typical embeddings."""
        rng = np.random.default_rng(0)
        candidates = [_emb(rng.standard_normal(8)) for _ in range(5)]
        result = intra_batch_diversity(candidates)
        assert result["intra_batch_diversity"] >= 0.0


class TestIntraBatchDiversityEdgeCases:
    """Edge case tests for intra_batch_diversity."""

    def test_single_candidate_returns_empty(self):
        """Single candidate returns empty dict (no pairs to compare)."""
        result = intra_batch_diversity([_seq("ACGT")])
        assert result == {}

    def test_empty_list_returns_empty(self):
        """Empty list returns empty dict."""
        result = intra_batch_diversity([])
        assert result == {}

    def test_mixed_modalities_raises(self):
        """Candidates with different modalities raise ValueError."""
        mixed = [_seq("ACGT"), _emb(np.array([1.0, 0.0]))]
        with pytest.raises(ValueError, match="same modality"):
            intra_batch_diversity(mixed)

    def test_molecule_raises_not_implemented(self):
        """MOLECULE candidates are not yet supported and raise NotImplementedError."""
        candidates = [
            Candidate(data="CCO", modality=Modality.MOLECULE),
            Candidate(data="c1ccccc1", modality=Modality.MOLECULE),
        ]
        with pytest.raises(NotImplementedError, match="MOLECULE"):
            intra_batch_diversity(candidates)

    def test_materials_raises_not_implemented(self):
        """MATERIALS candidates are not yet supported and raise NotImplementedError."""
        candidates = [
            Candidate(data="Fe0.62C0.01Mn0.37", modality=Modality.MATERIALS),
            Candidate(data="Fe0.5C0.2Mn0.3", modality=Modality.MATERIALS),
        ]
        with pytest.raises(NotImplementedError, match="MATERIALS"):
            intra_batch_diversity(candidates)

    def test_zero_vector_embedding_raises(self):
        """All-zero embedding raises ValueError (cosine distance undefined)."""
        candidates = [_emb(np.zeros(4)), _emb(np.array([1.0, 0.0, 0.0, 0.0]))]
        with pytest.raises(ValueError, match="all-zero"):
            intra_batch_diversity(candidates)
