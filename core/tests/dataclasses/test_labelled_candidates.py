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

import networkx as nx
import numpy as np
import pytest
import torch
from alf_core.dataclasses.candidate import Candidate, Modality
from alf_core.dataclasses.labelled_candidates import LabelledCandidates


class TestLabelledCandidatesInitialization:
    """Test cases for LabelledCandidates initialization and basic functionality."""

    def test_initialization_with_valid_data(self):
        """Test LabelledCandidates initialization with valid candidates and labels."""
        candidates = [
            Candidate(data="sequence1", modality=Modality.SEQUENCE),
            Candidate(data="sequence2", modality=Modality.SEQUENCE),
            Candidate(data="sequence3", modality=Modality.SEQUENCE),
        ]
        labels = np.array([1, 0, 1])

        labelled_candidates = LabelledCandidates(candidates=candidates, labels=labels)

        assert len(labelled_candidates) == 3
        assert labelled_candidates.candidates == candidates
        np.testing.assert_array_equal(labelled_candidates.labels, labels)

    def test_initialization_with_empty_data(self):
        """Test LabelledCandidates initialization with empty lists."""
        candidates = []
        labels = np.array([])

        labelled_candidates = LabelledCandidates(candidates=candidates, labels=labels)

        assert len(labelled_candidates) == 0
        assert labelled_candidates.candidates == []
        assert len(labelled_candidates.labels) == 0

    def test_initialization_with_single_item(self):
        """Test LabelledCandidates initialization with single candidate and label."""
        candidates = [Candidate(data="test", modality=Modality.SEQUENCE)]
        labels = np.array([1])

        labelled_candidates = LabelledCandidates(candidates=candidates, labels=labels)

        assert len(labelled_candidates) == 1
        assert labelled_candidates.candidates[0].data == "test"
        assert labelled_candidates.labels[0] == 1

    def test_large_number_of_candidates(self):
        """Test LabelledCandidates with a large number of candidates."""
        n_candidates = 1000
        candidates = [
            Candidate(data=f"test{i}", modality=Modality.SEQUENCE) for i in range(n_candidates)
        ]
        labels = np.random.randint(0, 2, size=n_candidates)

        labelled_candidates = LabelledCandidates(candidates=candidates, labels=labels)

        assert len(labelled_candidates) == n_candidates
        assert len(labelled_candidates.labels) == n_candidates

    def test_multidimensional_labels(self):
        """Test labels with multidimensional arrays."""
        candidates = [
            Candidate(data="test1", modality=Modality.SEQUENCE),
            Candidate(data="test2", modality=Modality.SEQUENCE),
        ]
        labels = np.array([[1, 0], [0, 1]])  # 2D labels

        labelled_candidates = LabelledCandidates(candidates=candidates, labels=labels)

        assert labelled_candidates.labels.shape == (2, 2)
        np.testing.assert_array_equal(labelled_candidates.labels, np.array([[1, 0], [0, 1]]))


class TestLabelledCandidatesValidation:
    """Test cases for LabelledCandidates validation and error handling."""

    def test_initialization_with_mismatched_lengths_raises_assertion(self):
        """Test that initialization with mismatched candidates and labels lengths
        raises AssertionError.
        """
        candidates = [
            Candidate(data="test1", modality=Modality.SEQUENCE),
            Candidate(data="test2", modality=Modality.SEQUENCE),
        ]
        labels = np.array([1])  # Only one label for two candidates

        with pytest.raises(AssertionError, match="Candidates and labels must have the same length"):
            LabelledCandidates(candidates=candidates, labels=labels)

    def test_initialization_with_empty_candidates_non_empty_labels(self):
        """Test initialization with empty candidates but non-empty labels."""
        candidates = []
        labels = np.array([1, 2, 3])

        with pytest.raises(AssertionError, match="Candidates and labels must have the same length"):
            LabelledCandidates(candidates=candidates, labels=labels)

    def test_initialization_with_non_empty_candidates_empty_labels(self):
        """Test initialization with non-empty candidates but empty labels."""
        candidates = [
            Candidate(data="test1", modality=Modality.SEQUENCE),
            Candidate(data="test2", modality=Modality.SEQUENCE),
        ]
        labels = np.array([])

        with pytest.raises(AssertionError, match="Candidates and labels must have the same length"):
            LabelledCandidates(candidates=candidates, labels=labels)


class TestLabelledCandidatesMethods:
    """Test cases for LabelledCandidates methods."""

    def test_len_method(self):
        """Test __len__ method returns correct length."""
        candidates = [
            Candidate(data="test1", modality=Modality.SEQUENCE),
            Candidate(data="test2", modality=Modality.SEQUENCE),
            Candidate(data="test3", modality=Modality.SEQUENCE),
        ]
        labels = np.array([1, 0, 1])

        labelled_candidates = LabelledCandidates(candidates=candidates, labels=labels)

        assert len(labelled_candidates) == 3

    def test_data_property(self):
        """Test data property returns raw data from candidates."""
        candidates = [
            Candidate(data="sequence1", modality=Modality.SEQUENCE),
            Candidate(data="sequence2", modality=Modality.SEQUENCE),
            Candidate(data="sequence3", modality=Modality.SEQUENCE),
        ]
        labels = np.array([1, 0, 1])

        labelled_candidates = LabelledCandidates(candidates=candidates, labels=labels)

        expected_data = ["sequence1", "sequence2", "sequence3"]
        assert labelled_candidates.data == expected_data


class TestLabelledCandidatesGetItem:
    """Tests for __getitem__ behavior."""

    def test_getitem_int_returns_singleton_collection(self):
        """Test that indexing with integer returns a singleton LabelledCandidates."""
        candidates = [
            Candidate(data="a", modality=Modality.SEQUENCE),
            Candidate(data="b", modality=Modality.SEQUENCE),
        ]
        labels = np.array([0, 1])
        lc = LabelledCandidates(candidates=candidates, labels=labels)

        item = lc[1]
        assert isinstance(item, tuple)
        candidates, labels = item
        assert candidates[0].data == "b"
        np.testing.assert_array_equal(labels, np.array([1]))

    def test_getitem_slice_returns_subcollection(self):
        """Test that slicing returns a subcollection of LabelledCandidates."""
        candidates = [
            Candidate(data="a", modality=Modality.SEQUENCE),
            Candidate(data="b", modality=Modality.SEQUENCE),
            Candidate(data="c", modality=Modality.SEQUENCE),
        ]
        labels = np.array([0, 1, 0])
        lc = LabelledCandidates(candidates=candidates, labels=labels)

        sub = lc[1:3]
        candidates, labels = sub
        assert len(candidates) == 2
        assert len(labels) == 2
        assert [c.data for c in candidates] == ["b", "c"]
        np.testing.assert_array_equal(labels, np.array([1, 0]))


class TestLabelledCandidatesValidateShuffleSortRemove:
    """Tests for shuffle, sort, and remove methods."""

    def test_shuffle_deterministic(self):
        """Test that shuffle with same seed produces deterministic results."""
        candidates = [Candidate(data=str(i), modality=Modality.SEQUENCE) for i in range(5)]
        labels = np.array([0, 1, 2, 3, 4])
        lc = LabelledCandidates(candidates=candidates, labels=labels)

        shuffled1 = lc.shuffle(seed=42)
        shuffled2 = lc.shuffle(seed=42)
        assert [c.data for c in shuffled1.candidates] == [c.data for c in shuffled2.candidates]
        np.testing.assert_array_equal(shuffled1.labels, shuffled2.labels)
        # Ensure original is unchanged length-wise and content-wise
        assert [c.data for c in lc.candidates] == [str(i) for i in range(5)]
        np.testing.assert_array_equal(lc.labels, labels)

    def test_sort_ascending_and_descending(self):
        """Test sort method with ascending and descending order."""
        candidates = [
            Candidate(data="x", modality=Modality.SEQUENCE),
            Candidate(data="y", modality=Modality.SEQUENCE),
            Candidate(data="z", modality=Modality.SEQUENCE),
        ]
        labels = np.array([2.0, 1.0, 3.0])
        lc = LabelledCandidates(candidates=candidates, labels=labels)

        asc = lc.sort(ascending=True)
        desc = lc.sort(ascending=False)

        np.testing.assert_array_equal([c.data for c in asc.candidates], ["y", "x", "z"])
        np.testing.assert_array_equal(asc.labels, np.array([1.0, 2.0, 3.0]))

        np.testing.assert_array_equal([c.data for c in desc.candidates], ["z", "x", "y"])
        np.testing.assert_array_equal(desc.labels, np.array([3.0, 2.0, 1.0]))

    def test_remove_with_list_and_with_collection(self):
        """Test remove method with both list and LabelledCandidates input."""
        c1 = Candidate(data="a", modality=Modality.SEQUENCE)
        c2 = Candidate(data="b", modality=Modality.SEQUENCE)
        c3 = Candidate(data="c", modality=Modality.SEQUENCE)
        lc = LabelledCandidates(candidates=[c1, c2, c3], labels=np.array([1, 2, 3]))

        # remove with list
        lc.remove([c2])
        assert [c.data for c in lc.candidates] == ["a", "c"]
        np.testing.assert_array_equal(lc.labels, np.array([1, 3]))

        # remove with LabelledCandidates
        lc2 = LabelledCandidates(candidates=[c1], labels=np.array([1]))
        lc.remove(lc2)
        assert [c.data for c in lc.candidates] == ["c"]
        np.testing.assert_array_equal(lc.labels, np.array([3]))

    def test_remove_ignores_candidates_not_present(self):
        """Test that remove silently ignores candidates not in the collection."""
        c1 = Candidate(data="a", modality=Modality.SEQUENCE)
        c2 = Candidate(data="b", modality=Modality.SEQUENCE)
        lc = LabelledCandidates(candidates=[c1], labels=np.array([1]))

        # Should not raise, just ignore c2
        lc.remove([c2])

        # Collection should be unchanged
        assert [c.data for c in lc.candidates] == ["a"]
        np.testing.assert_array_equal(lc.labels, np.array([1]))


class TestLabelledCandidatesToDataFrame:
    """Tests for to_dataframe ensuring features and stringify are used."""

    def test_to_dataframe_with_features(self):
        """Test to_dataframe method includes candidate features as columns."""
        c1 = Candidate(data="SEQ1", modality=Modality.SEQUENCE, features={"a": 1, "b": "x"})
        c2 = Candidate(data="SEQ2", modality=Modality.SEQUENCE, features={"a": 2})
        lc = LabelledCandidates(candidates=[c1, c2], labels=np.array([0.5, 1.5]))

        df = lc.to_dataframe()
        assert list(df.columns) == ["data", "label", "a", "b"] or list(df.columns) == [
            "data",
            "label",
            "b",
            "a",
        ]
        assert len(df) == 2
        # data should come from stringify (for sequence equals raw string)
        assert df.loc[0, "data"] == "SEQ1"
        assert df.loc[1, "data"] == "SEQ2"
        # features flattened
        assert df.loc[0, "a"] == 1
        assert df.loc[1, "a"] == 2


class TestLabelledCandidatesAppend:
    """Test cases for LabelledCandidates append method."""

    def test_append_with_candidates_list(self):
        """Test append method with list of candidates and labels."""
        candidates = [Candidate(data="test1", modality=Modality.SEQUENCE)]
        labels = np.array([1])
        labelled_candidates = LabelledCandidates(candidates=candidates, labels=labels)

        new_candidates = [
            Candidate(data="test2", modality=Modality.SEQUENCE),
            Candidate(data="test3", modality=Modality.SEQUENCE),
        ]
        new_labels = np.array([0, 1])

        labelled_candidates.append(new_candidates, new_labels)

        assert len(labelled_candidates) == 3
        assert labelled_candidates.candidates[0].data == "test1"
        assert labelled_candidates.candidates[1].data == "test2"
        assert labelled_candidates.candidates[2].data == "test3"
        np.testing.assert_array_equal(labelled_candidates.labels, np.array([1, 0, 1]))

    def test_append_with_labelled_candidates(self):
        """Test append method with another LabelledCandidates object."""
        candidates1 = [Candidate(data="test1", modality=Modality.SEQUENCE)]
        labels1 = np.array([1])
        labelled_candidates1 = LabelledCandidates(candidates=candidates1, labels=labels1)

        candidates2 = [
            Candidate(data="test2", modality=Modality.SEQUENCE),
            Candidate(data="test3", modality=Modality.SEQUENCE),
        ]
        labels2 = np.array([0, 1])
        labelled_candidates2 = LabelledCandidates(candidates=candidates2, labels=labels2)

        labelled_candidates1.append(labelled_candidates2)

        assert len(labelled_candidates1) == 3
        assert labelled_candidates1.candidates[0].data == "test1"
        assert labelled_candidates1.candidates[1].data == "test2"
        assert labelled_candidates1.candidates[2].data == "test3"
        np.testing.assert_array_equal(labelled_candidates1.labels, np.array([1, 0, 1]))

    def test_append_with_mismatched_lengths_raises_assertion(self):
        """Test that append with mismatched candidates and labels lengths raises AssertionError."""
        candidates = [Candidate(data="test1", modality=Modality.SEQUENCE)]
        labels = np.array([1])
        labelled_candidates = LabelledCandidates(candidates=candidates, labels=labels)

        new_candidates = [
            Candidate(data="test2", modality=Modality.SEQUENCE),
            Candidate(data="test3", modality=Modality.SEQUENCE),
        ]
        new_labels = np.array([0])  # Only one label for two candidates

        with pytest.raises(AssertionError, match="Candidates and labels must have the same length"):
            labelled_candidates.append(new_candidates, new_labels)

    def test_append_with_none_labels_raises_assertion(self):
        """Test that append with None labels raises AssertionError."""
        candidates = [Candidate(data="test1", modality=Modality.SEQUENCE)]
        labels = np.array([1])
        labelled_candidates = LabelledCandidates(candidates=candidates, labels=labels)

        new_candidates = [Candidate(data="test2", modality=Modality.SEQUENCE)]

        with pytest.raises(AssertionError, match="Candidates and labels must have the same length"):
            labelled_candidates.append(new_candidates, None)

    def test_multiple_append_operations(self):
        """Test multiple consecutive append operations."""
        labelled_candidates = LabelledCandidates(candidates=[], labels=np.array([]))

        # First append
        candidates1 = [Candidate(data="test1", modality=Modality.SEQUENCE)]
        labels1 = np.array([1])
        labelled_candidates.append(candidates1, labels1)

        # Second append
        candidates2 = [Candidate(data="test2", modality=Modality.SEQUENCE)]
        labels2 = np.array([0])
        labelled_candidates.append(candidates2, labels2)

        # Third append with LabelledCandidates
        candidates3 = [Candidate(data="test3", modality=Modality.SEQUENCE)]
        labels3 = np.array([1])
        labelled_candidates3 = LabelledCandidates(candidates=candidates3, labels=labels3)
        labelled_candidates.append(labelled_candidates3)

        assert len(labelled_candidates) == 3
        assert labelled_candidates.candidates[0].data == "test1"
        assert labelled_candidates.candidates[1].data == "test2"
        assert labelled_candidates.candidates[2].data == "test3"
        np.testing.assert_array_equal(labelled_candidates.labels, np.array([1, 0, 1]))


@pytest.mark.parametrize("n_candidates", [0, 1, 5, 10, 100])
def test_labelled_candidates_length_consistency(n_candidates):
    """Parametrized test for length consistency with different numbers of candidates."""
    candidates = [
        Candidate(data=f"test{i}", modality=Modality.SEQUENCE) for i in range(n_candidates)
    ]
    labels = np.random.randint(0, 2, size=n_candidates)

    labelled_candidates = LabelledCandidates(candidates=candidates, labels=labels)

    assert len(labelled_candidates) == n_candidates
    assert len(labelled_candidates.candidates) == n_candidates
    assert len(labelled_candidates.labels) == n_candidates


@pytest.mark.parametrize(
    "modality,data_factory",
    [
        (Modality.SEQUENCE, lambda: "ATCGATCG"),
        (Modality.IMAGE, lambda: np.random.rand(32, 32, 3)),
        (Modality.GRAPH, lambda: nx.path_graph(5)),
        (Modality.STRUCTURE, lambda: np.random.rand(10, 3)),
        (Modality.TABULAR, lambda: {"feature1": 1, "feature2": 2}),
        (Modality.EMBEDDING, lambda: torch.randn(10, 5)),
    ],
)
def test_labelled_candidates_modality_consistency(modality, data_factory):
    """Parametrized test for different modalities with consistent data types."""
    data = data_factory()
    candidates = [Candidate(data=data, modality=modality)]
    labels = np.array([1])

    labelled_candidates = LabelledCandidates(candidates=candidates, labels=labels)

    assert len(labelled_candidates) == 1
    assert labelled_candidates.candidates[0].modality == modality
    assert labelled_candidates.data[0] is not None
