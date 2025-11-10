import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

from alf.core.dataclasses.candidate import Candidate


class TestCandidateInitialization:
    """Test cases for Candidate initialization and basic functionality."""

    def test_candidate_with_no_features(self):
        """Test Candidate initialization with no features."""
        candidate = Candidate(data="test", modality="test")
        assert candidate.data == "test"
        assert candidate.modality == "test"
        assert candidate.features == {}

    def test_candidate_with_features(self):
        """Test Candidate initialization with features."""
        features = {"feature1": "value1", "feature2": 42}
        candidate = Candidate(data="test", modality="test", features=features)
        assert candidate.data == "test"
        assert candidate.modality == "test"
        assert candidate.features == features

    def test_candidate_with_none_features(self):
        """Test Candidate initialization with None features (should default to empty dict)."""
        candidate = Candidate(data="test", modality="test", features=None)
        assert candidate.data == "test"
        assert candidate.modality == "test"
        assert candidate.features == {}

    def test_candidate_repr(self):
        """Test Candidate string representation."""
        candidate = Candidate(data="test", modality="test", features={"key": "value"})
        repr_str = repr(candidate)
        assert repr_str == "Candidate(data=test, modality=test, features={'key': 'value'})"


class TestCandidateDataModalities:
    """Test cases for different data modalities supported by Candidate."""

    def test_sequence_modality(self):
        """Test Candidate with sequence data (string)."""
        seq_data = "MKTFFVAGLVLLLTICSASG"  # protein sequence
        candidate = Candidate(data=seq_data, modality="sequence")
        assert isinstance(candidate.data, str)
        assert candidate.modality == "sequence"
        assert len(candidate.data) == 20

    def test_image_modality_numpy(self):
        """Test Candidate with image data (NumPy array)."""
        img_data = np.random.rand(3, 64, 64).astype(np.float32)  # RGB image
        candidate = Candidate(data=img_data, modality="image")
        assert isinstance(candidate.data, np.ndarray)
        assert candidate.data.shape == (3, 64, 64)
        assert candidate.data.dtype == np.float32

    def test_image_modality_torch(self):
        """Test Candidate with image data (PyTorch tensor)."""
        img_data = torch.randn(3, 64, 64, dtype=torch.float32)
        candidate = Candidate(data=img_data, modality="image")
        assert isinstance(candidate.data, torch.Tensor)
        assert candidate.data.shape == (3, 64, 64)

    def test_graph_modality(self):
        """Test Candidate with graph data (NetworkX graph)."""
        G = nx.erdos_renyi_graph(n=10, p=0.3)
        candidate = Candidate(data=G, modality="graph")
        assert isinstance(candidate.data, nx.Graph)
        assert candidate.data.number_of_nodes() == 10

    def test_structure_modality(self):
        """Test Candidate with 3D structure data."""
        coords = np.random.rand(50, 3)  # 50 atoms, (x,y,z)
        candidate = Candidate(data=coords, modality="structure")
        assert isinstance(candidate.data, np.ndarray)
        assert candidate.data.shape == (50, 3)

    def test_tabular_modality_pandas(self):
        """Test Candidate with tabular data (pandas Series)."""
        row = pd.Series({"age": 32, "height": 178, "weight": 70})
        candidate = Candidate(data=row, modality="tabular")
        assert isinstance(candidate.data, pd.Series)
        assert candidate.data["age"] == 32

    def test_tabular_modality_dict(self):
        """Test Candidate with tabular data (dictionary)."""
        row = {"age": 32, "height": 178, "weight": 70}
        candidate = Candidate(data=row, modality="tabular")
        assert isinstance(candidate.data, dict)
        assert candidate.data["age"] == 32

    def test_embedding_modality(self):
        """Test Candidate with embedding data (PyTorch tensor)."""
        tensor_data = torch.randn(16, 128)  # sequence embeddings
        candidate = Candidate(data=tensor_data, modality="embedding")
        assert isinstance(candidate.data, torch.Tensor)
        assert candidate.data.shape == (16, 128)


class TestCandidateEdgeCases:
    """Test cases for edge cases and error conditions."""

    def test_empty_string_data(self):
        """Test Candidate with empty string data."""
        candidate = Candidate(data="", modality="sequence")
        assert candidate.data == ""  # noqa: PLC1901
        assert candidate.modality == "sequence"

    def test_none_data(self):
        """Test Candidate with None data."""
        candidate = Candidate(data=None, modality="test")
        assert candidate.data is None
        assert candidate.modality == "test"

    def test_empty_list_data(self):
        """Test Candidate with empty list data."""
        candidate = Candidate(data=[], modality="test")
        assert candidate.data == []
        assert candidate.modality == "test"

    def test_empty_dict_data(self):
        """Test Candidate with empty dictionary data."""
        candidate = Candidate(data={}, modality="test")
        assert candidate.data == {}
        assert candidate.modality == "test"

    def test_empty_features_dict(self):
        """Test Candidate with empty features dictionary."""
        candidate = Candidate(data="test", modality="test", features={})
        assert candidate.features == {}

    def test_nested_features(self):
        """Test Candidate with nested features dictionary."""
        nested_features = {
            "metadata": {"source": "database", "version": "1.0"},
            "stats": {"mean": 0.5, "std": 0.1},
            "flags": [True, False, True],
        }
        candidate = Candidate(data="test", modality="test", features=nested_features)
        assert candidate.features == nested_features
        assert candidate.features["metadata"]["source"] == "database"


class TestCandidateFeatures:
    """Test cases for Candidate features functionality."""

    def test_features_immutability_after_init(self):
        """Test that features can be modified after initialization."""
        candidate = Candidate(data="test", modality="test", features={"key1": "value1"})

        # Modify features
        candidate.features["key2"] = "value2"
        candidate.features["key1"] = "updated_value"

        assert candidate.features["key1"] == "updated_value"
        assert candidate.features["key2"] == "value2"

    def test_features_with_different_types(self):
        """Test Candidate with features containing different data types."""
        features = {
            "string": "text",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "numpy_array": np.array([1, 2, 3]),
            "torch_tensor": torch.tensor([1, 2, 3]),
        }
        candidate = Candidate(data="test", modality="test", features=features)
        assert candidate.features == features


class TestCandidateStringify:
    """Tests for Candidate.stringify method."""

    def test_stringify_sequence(self):
        """Test stringify method with sequence modality."""
        candidate = Candidate(data="ACDEFG", modality="sequence")
        assert candidate.stringify() == "ACDEFG"

    def test_stringify_unsupported_modality_raises(self):
        """Test that stringify raises ValueError for unsupported modality."""
        candidate = Candidate(data=123, modality="numeric")
        with pytest.raises(ValueError, match="Unsupported modality"):
            candidate.stringify()


@pytest.mark.parametrize(
    "modality,data_factory",
    [
        ("sequence", lambda: "ATCGATCG"),
        ("image", lambda: np.random.rand(32, 32, 3)),
        ("graph", lambda: nx.path_graph(5)),
        ("structure", lambda: np.random.rand(10, 3)),
        ("tabular", lambda: {"feature1": 1, "feature2": 2}),
        ("embedding", lambda: torch.randn(10, 5)),
    ],
)
def test_candidate_modality_consistency(modality, data_factory):
    """Parametrized test for different modalities with consistent data types."""
    data = data_factory()
    candidate = Candidate(data=data, modality=modality)

    assert candidate.data is not None
    assert candidate.modality == modality
    assert candidate.features == {}


@pytest.mark.parametrize(
    "features",
    [
        {},
        {"single": "value"},
        {"multiple": "values", "with": "different", "types": 123},
        {"nested": {"inner": "value"}},
    ],
)
def test_candidate_features_parametrized(features):
    """Parametrized test for different feature configurations."""
    candidate = Candidate(data="test", modality="test", features=features)
    assert candidate.features == features
