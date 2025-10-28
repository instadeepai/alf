import pytest
import numpy as np
import torch
import networkx as nx
import pandas as pd

from core.dataclasses.candidate import Candidate
from core.dataclasses.labeled_candidates import LabeledCandidates


class TestLabeledCandidatesInitialization:
    """Test cases for LabeledCandidates initialization and basic functionality."""

    def test_initialization_with_valid_data(self):
        """Test LabeledCandidates initialization with valid candidates and labels."""
        candidates = [
            Candidate(data="sequence1", modality="sequence"),
            Candidate(data="sequence2", modality="sequence"),
            Candidate(data="sequence3", modality="sequence")
        ]
        labels = np.array([1, 0, 1])
        
        labeled_candidates = LabeledCandidates(candidates=candidates, labels=labels)
        
        assert len(labeled_candidates) == 3
        assert labeled_candidates.candidates == candidates
        np.testing.assert_array_equal(labeled_candidates.labels, labels)

    def test_initialization_with_empty_data(self):
        """Test LabeledCandidates initialization with empty lists."""
        candidates = []
        labels = np.array([])
        
        labeled_candidates = LabeledCandidates(candidates=candidates, labels=labels)
        
        assert len(labeled_candidates) == 0
        assert labeled_candidates.candidates == []
        assert len(labeled_candidates.labels) == 0

    def test_initialization_with_single_item(self):
        """Test LabeledCandidates initialization with single candidate and label."""
        candidates = [Candidate(data="test", modality="test")]
        labels = np.array([1])
        
        labeled_candidates = LabeledCandidates(candidates=candidates, labels=labels)
        
        assert len(labeled_candidates) == 1
        assert labeled_candidates.candidates[0].data == "test"
        assert labeled_candidates.labels[0] == 1

    def test_large_number_of_candidates(self):
        """Test LabeledCandidates with a large number of candidates."""
        n_candidates = 1000
        candidates = [Candidate(data=f"test{i}", modality="test") for i in range(n_candidates)]
        labels = np.random.randint(0, 2, size=n_candidates)
        
        labeled_candidates = LabeledCandidates(candidates=candidates, labels=labels)
        
        assert len(labeled_candidates) == n_candidates
        assert len(labeled_candidates.labels) == n_candidates

    def test_multidimensional_labels(self):
        """Test labels with multidimensional arrays."""
        candidates = [
            Candidate(data="test1", modality="test"),
            Candidate(data="test2", modality="test")
        ]
        labels = np.array([[1, 0], [0, 1]])  # 2D labels
        
        labeled_candidates = LabeledCandidates(candidates=candidates, labels=labels)
        
        assert labeled_candidates.labels.shape == (2, 2)
        np.testing.assert_array_equal(labeled_candidates.labels, np.array([[1, 0], [0, 1]]))


class TestLabeledCandidatesValidation:
    """Test cases for LabeledCandidates validation and error handling."""

    def test_initialization_with_mismatched_lengths_raises_assertion(self):
        """Test that initialization with mismatched candidates and labels lengths raises AssertionError."""
        candidates = [
            Candidate(data="test1", modality="test"),
            Candidate(data="test2", modality="test")
        ]
        labels = np.array([1])  # Only one label for two candidates
        
        with pytest.raises(AssertionError, match="Candidates and labels must have the same length"):
            LabeledCandidates(candidates=candidates, labels=labels)

    def test_initialization_with_empty_candidates_non_empty_labels(self):
        """Test initialization with empty candidates but non-empty labels."""
        candidates = []
        labels = np.array([1, 2, 3])
        
        with pytest.raises(AssertionError, match="Candidates and labels must have the same length"):
            LabeledCandidates(candidates=candidates, labels=labels)

    def test_initialization_with_non_empty_candidates_empty_labels(self):
        """Test initialization with non-empty candidates but empty labels."""
        candidates = [
            Candidate(data="test1", modality="test"),
            Candidate(data="test2", modality="test")
        ]
        labels = np.array([])
        
        with pytest.raises(AssertionError, match="Candidates and labels must have the same length"):
            LabeledCandidates(candidates=candidates, labels=labels)


class TestLabeledCandidatesMethods:
    """Test cases for LabeledCandidates methods."""

    def test_len_method(self):
        """Test __len__ method returns correct length."""
        candidates = [
            Candidate(data="test1", modality="test"),
            Candidate(data="test2", modality="test"),
            Candidate(data="test3", modality="test")
        ]
        labels = np.array([1, 0, 1])
        
        labeled_candidates = LabeledCandidates(candidates=candidates, labels=labels)
        
        assert len(labeled_candidates) == 3

    def test_data_property(self):
        """Test data property returns raw data from candidates."""
        candidates = [
            Candidate(data="sequence1", modality="sequence"),
            Candidate(data="sequence2", modality="sequence"),
            Candidate(data="sequence3", modality="sequence")
        ]
        labels = np.array([1, 0, 1])
        
        labeled_candidates = LabeledCandidates(candidates=candidates, labels=labels)
        
        expected_data = ["sequence1", "sequence2", "sequence3"]
        assert labeled_candidates.data == expected_data


class TestLabeledCandidatesAppend:
    """Test cases for LabeledCandidates append method."""

    def test_append_with_candidates_list(self):
        """Test append method with list of candidates and labels."""
        candidates = [Candidate(data="test1", modality="test")]
        labels = np.array([1])
        labeled_candidates = LabeledCandidates(candidates=candidates, labels=labels)
        
        new_candidates = [
            Candidate(data="test2", modality="test"),
            Candidate(data="test3", modality="test")
        ]
        new_labels = np.array([0, 1])
        
        labeled_candidates.append(new_candidates, new_labels)
        
        assert len(labeled_candidates) == 3
        assert labeled_candidates.candidates[0].data == "test1"
        assert labeled_candidates.candidates[1].data == "test2"
        assert labeled_candidates.candidates[2].data == "test3"
        np.testing.assert_array_equal(labeled_candidates.labels, np.array([1, 0, 1]))

    def test_append_with_labeled_candidates(self):
        """Test append method with another LabeledCandidates object."""
        candidates1 = [Candidate(data="test1", modality="test")]
        labels1 = np.array([1])
        labeled_candidates1 = LabeledCandidates(candidates=candidates1, labels=labels1)
        
        candidates2 = [
            Candidate(data="test2", modality="test"),
            Candidate(data="test3", modality="test")
        ]
        labels2 = np.array([0, 1])
        labeled_candidates2 = LabeledCandidates(candidates=candidates2, labels=labels2)
        
        labeled_candidates1.append(labeled_candidates2)
        
        assert len(labeled_candidates1) == 3
        assert labeled_candidates1.candidates[0].data == "test1"
        assert labeled_candidates1.candidates[1].data == "test2"
        assert labeled_candidates1.candidates[2].data == "test3"
        np.testing.assert_array_equal(labeled_candidates1.labels, np.array([1, 0, 1]))

    def test_append_with_mismatched_lengths_raises_assertion(self):
        """Test that append with mismatched candidates and labels lengths raises AssertionError."""
        candidates = [Candidate(data="test1", modality="test")]
        labels = np.array([1])
        labeled_candidates = LabeledCandidates(candidates=candidates, labels=labels)
        
        new_candidates = [
            Candidate(data="test2", modality="test"),
            Candidate(data="test3", modality="test")
        ]
        new_labels = np.array([0])  # Only one label for two candidates
        
        with pytest.raises(AssertionError, match="Candidates and labels must have the same length"):
            labeled_candidates.append(new_candidates, new_labels)

    def test_append_with_none_labels_raises_assertion(self):
        """Test that append with None labels raises AssertionError."""
        candidates = [Candidate(data="test1", modality="test")]
        labels = np.array([1])
        labeled_candidates = LabeledCandidates(candidates=candidates, labels=labels)
        
        new_candidates = [Candidate(data="test2", modality="test")]
        
        with pytest.raises(AssertionError, match="Candidates and labels must have the same length"):
            labeled_candidates.append(new_candidates, None)

    def test_multiple_append_operations(self):
        """Test multiple consecutive append operations."""
        labeled_candidates = LabeledCandidates(candidates=[], labels=np.array([]))
        
        # First append
        candidates1 = [Candidate(data="test1", modality="test")]
        labels1 = np.array([1])
        labeled_candidates.append(candidates1, labels1)
        
        # Second append
        candidates2 = [Candidate(data="test2", modality="test")]
        labels2 = np.array([0])
        labeled_candidates.append(candidates2, labels2)
        
        # Third append with LabeledCandidates
        candidates3 = [Candidate(data="test3", modality="test")]
        labels3 = np.array([1])
        labeled_candidates3 = LabeledCandidates(candidates=candidates3, labels=labels3)
        labeled_candidates.append(labeled_candidates3)
        
        assert len(labeled_candidates) == 3
        assert labeled_candidates.candidates[0].data == "test1"
        assert labeled_candidates.candidates[1].data == "test2"
        assert labeled_candidates.candidates[2].data == "test3"
        np.testing.assert_array_equal(labeled_candidates.labels, np.array([1, 0, 1]))


@pytest.mark.parametrize("n_candidates", [0, 1, 5, 10, 100])
def test_labeled_candidates_length_consistency(n_candidates):
    """Parametrized test for length consistency with different numbers of candidates."""
    candidates = [Candidate(data=f"test{i}", modality="test") for i in range(n_candidates)]
    labels = np.random.randint(0, 2, size=n_candidates)
    
    labeled_candidates = LabeledCandidates(candidates=candidates, labels=labels)
    
    assert len(labeled_candidates) == n_candidates
    assert len(labeled_candidates.candidates) == n_candidates
    assert len(labeled_candidates.labels) == n_candidates


@pytest.mark.parametrize("modality,data_factory", [
    ("sequence", lambda: "ATCGATCG"),
    ("image", lambda: np.random.rand(32, 32, 3)),
    ("graph", lambda: nx.path_graph(5)),
    ("structure", lambda: np.random.rand(10, 3)),
    ("tabular", lambda: {"feature1": 1, "feature2": 2}),
    ("embedding", lambda: torch.randn(10, 5)),
])
def test_labeled_candidates_modality_consistency(modality, data_factory):
    """Parametrized test for different modalities with consistent data types."""
    data = data_factory()
    candidates = [Candidate(data=data, modality=modality)]
    labels = np.array([1])
    
    labeled_candidates = LabeledCandidates(candidates=candidates, labels=labels)
    
    assert len(labeled_candidates) == 1
    assert labeled_candidates.candidates[0].modality == modality
    assert labeled_candidates.data[0] is not None
