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

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
from alf_core.dataclasses.candidate import Candidate, Modality


class TestCandidateInitialization:
    """Test cases for Candidate initialization and basic functionality."""

    def test_candidate_with_no_features(self):
        """Test Candidate initialization with no features."""
        candidate = Candidate(data="test", modality=Modality.SEQUENCE)
        assert candidate.data == "test"
        assert candidate.modality == Modality.SEQUENCE
        assert candidate.features == {}

    def test_candidate_with_features(self):
        """Test Candidate initialization with features."""
        features = {"feature1": "value1", "feature2": 42}
        candidate = Candidate(data="test", modality=Modality.SEQUENCE, features=features)
        assert candidate.data == "test"
        assert candidate.modality == Modality.SEQUENCE
        assert candidate.features == features

    def test_candidate_with_none_features(self):
        """Test Candidate initialization with None features (should default to empty dict)."""
        candidate = Candidate(data="test", modality=Modality.SEQUENCE, features=None)
        assert candidate.data == "test"
        assert candidate.modality == Modality.SEQUENCE
        assert candidate.features == {}

    def test_candidate_repr(self):
        """Test Candidate string representation."""
        candidate = Candidate(data="test", modality=Modality.SEQUENCE, features={"key": "value"})
        repr_str = repr(candidate)
        assert (
            repr_str
            == "Candidate(data=test, modality=Modality.SEQUENCE, features={'key': 'value'})"
        )


class TestCandidateDataModalities:
    """Test cases for different data modalities supported by Candidate."""

    def test_sequence_modality(self):
        """Test Candidate with sequence data (string)."""
        seq_data = "MKTFFVAGLVLLLTICSASG"  # protein sequence
        candidate = Candidate(data=seq_data, modality=Modality.SEQUENCE)
        assert isinstance(candidate.data, str)
        assert candidate.modality == Modality.SEQUENCE
        assert len(candidate.data) == 20

    def test_graph_modality(self):
        """Test Candidate with graph data (NetworkX graph)."""
        G = nx.erdos_renyi_graph(n=10, p=0.3)
        candidate = Candidate(data=G, modality=Modality.GRAPH)
        assert isinstance(candidate.data, nx.Graph)
        assert candidate.data.number_of_nodes() == 10

    def test_structure_modality(self):
        """Test Candidate with 3D structure data."""
        coords = np.random.rand(50, 3)  # 50 atoms, (x,y,z)
        candidate = Candidate(data=coords, modality=Modality.STRUCTURE)
        assert isinstance(candidate.data, np.ndarray)
        assert candidate.data.shape == (50, 3)

    def test_tabular_modality_pandas(self):
        """Test Candidate with tabular data (pandas Series)."""
        row = pd.Series({"age": 32, "height": 178, "weight": 70})
        candidate = Candidate(data=row, modality=Modality.TABULAR)
        assert isinstance(candidate.data, pd.Series)
        assert candidate.data["age"] == 32

    def test_tabular_modality_dict(self):
        """Test Candidate with tabular data (dictionary)."""
        row = {"age": 32, "height": 178, "weight": 70}
        candidate = Candidate(data=row, modality=Modality.TABULAR)
        assert isinstance(candidate.data, dict)
        assert candidate.data["age"] == 32


class TestCandidateEdgeCases:
    """Test cases for edge cases and error conditions."""

    def test_empty_string_data(self):
        """Test Candidate with empty string data."""
        candidate = Candidate(data="", modality=Modality.SEQUENCE)
        assert candidate.data == ""  # noqa: PLC1901
        assert candidate.modality == Modality.SEQUENCE

    def test_none_data(self):
        """Test Candidate with None data."""
        candidate = Candidate(data=None, modality=Modality.SEQUENCE)
        assert candidate.data is None
        assert candidate.modality == Modality.SEQUENCE

    def test_empty_list_data(self):
        """Test Candidate with empty list data."""
        candidate = Candidate(data=[], modality=Modality.SEQUENCE)
        assert candidate.data == []
        assert candidate.modality == Modality.SEQUENCE

    def test_empty_features_dict(self):
        """Test Candidate with empty features dictionary."""
        candidate = Candidate(data="test", modality=Modality.SEQUENCE, features={})
        assert candidate.features == {}

    def test_nested_features(self):
        """Test Candidate with nested features dictionary."""
        nested_features = {
            "metadata": {"source": "database", "version": "1.0"},
            "stats": {"mean": 0.5, "std": 0.1},
            "flags": [True, False, True],
        }
        candidate = Candidate(data="test", modality=Modality.SEQUENCE, features=nested_features)
        assert candidate.features == nested_features
        assert candidate.features["metadata"]["source"] == "database"


class TestCandidateFeatures:
    """Test cases for Candidate features functionality."""

    def test_features_immutability_after_init(self):
        """Test that features can be modified after initialization."""
        candidate = Candidate(data="test", modality=Modality.SEQUENCE, features={"key1": "value1"})

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
        candidate = Candidate(data="test", modality=Modality.SEQUENCE, features=features)
        assert candidate.features == features


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
    candidate = Candidate(data="test", modality=Modality.SEQUENCE, features=features)
    assert candidate.features == features


class TestCandidateEquality:
    """Test cases for Candidate equality with numpy arrays and nested structures."""

    def test_equality_basic_same_candidates(self):
        """Test that two candidates with same data are equal."""
        c1 = Candidate(data="ATCG", modality=Modality.SEQUENCE)
        c2 = Candidate(data="ATCG", modality=Modality.SEQUENCE)
        assert c1 == c2

    def test_equality_basic_different_data(self):
        """Test that candidates with different data are not equal."""
        c1 = Candidate(data="ATCG", modality=Modality.SEQUENCE)
        c2 = Candidate(data="GCTA", modality=Modality.SEQUENCE)
        assert c1 != c2

    def test_equality_different_modality(self):
        """Test that candidates with different modalities are not equal."""
        c1 = Candidate(data="test", modality=Modality.SEQUENCE)
        c2 = Candidate(data="test", modality=Modality.TABULAR)
        assert c1 != c2

    def test_equality_with_numpy_array_in_features(self):
        """Test equality when features contain numpy arrays."""
        arr = np.array([1.0, 2.0, 3.0])
        c1 = Candidate(data="ATCG", modality=Modality.SEQUENCE, features={"embedding": arr})
        c2 = Candidate(data="ATCG", modality=Modality.SEQUENCE, features={"embedding": arr.copy()})
        assert c1 == c2

    def test_equality_with_numpy_array_in_features_different_values(self):
        """Test inequality when features contain different numpy arrays."""
        c1 = Candidate(
            data="ATCG",
            modality=Modality.SEQUENCE,
            features={"embedding": np.array([1.0, 2.0, 3.0])},
        )
        c2 = Candidate(
            data="ATCG",
            modality=Modality.SEQUENCE,
            features={"embedding": np.array([1.0, 2.0, 4.0])},
        )
        assert c1 != c2

    def test_equality_with_numpy_array_in_data(self):
        """Test equality when data is a numpy array (TABULAR modality)."""
        arr_data = np.random.rand(3, 64, 64).astype(np.float32)
        c1 = Candidate(data=arr_data, modality=Modality.TABULAR)
        c2 = Candidate(data=arr_data.copy(), modality=Modality.TABULAR)
        assert c1 == c2

    def test_equality_with_numpy_array_in_data_different_values(self):
        """Test inequality when data contains different numpy arrays."""
        c1 = Candidate(data=np.array([[1, 2], [3, 4]]), modality=Modality.STRUCTURE)
        c2 = Candidate(data=np.array([[1, 2], [3, 5]]), modality=Modality.STRUCTURE)
        assert c1 != c2

    def test_equality_with_nested_list_of_numpy_arrays_in_data(self):
        """Test equality with nested list containing numpy arrays."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        c1 = Candidate(data=[arr1, arr2], modality=Modality.TABULAR)
        c2 = Candidate(data=[arr1.copy(), arr2.copy()], modality=Modality.TABULAR)
        assert c1 == c2

    def test_equality_with_nested_dict_of_numpy_arrays_in_features(self):
        """Test equality with nested dict containing numpy arrays in features."""
        features = {"layer1": np.array([1, 2]), "layer2": {"hidden": np.array([3, 4])}}
        c1 = Candidate(data="ATCG", modality=Modality.SEQUENCE, features=features)
        # Create a deep copy of nested features
        features_copy = {
            "layer1": features["layer1"].copy(),
            "layer2": {"hidden": features["layer2"]["hidden"].copy()},
        }
        c2 = Candidate(data="ATCG", modality=Modality.SEQUENCE, features=features_copy)
        assert c1 == c2

    def test_equality_with_mixed_features(self):
        """Test equality with features containing mix of numpy arrays and regular values."""
        features = {"score": 0.95, "embedding": np.array([1.0, 2.0]), "name": "test"}
        c1 = Candidate(data="ATCG", modality=Modality.SEQUENCE, features=features)
        features_copy = {"score": 0.95, "embedding": features["embedding"].copy(), "name": "test"}
        c2 = Candidate(data="ATCG", modality=Modality.SEQUENCE, features=features_copy)
        assert c1 == c2

    def test_equality_different_feature_keys(self):
        """Test inequality when features have different keys."""
        c1 = Candidate(data="ATCG", modality=Modality.SEQUENCE, features={"a": np.array([1])})
        c2 = Candidate(data="ATCG", modality=Modality.SEQUENCE, features={"b": np.array([1])})
        assert c1 != c2

    def test_equality_with_none_features(self):
        """Test equality when features are None (should become empty dict)."""
        c1 = Candidate(data="ATCG", modality=Modality.SEQUENCE, features=None)
        c2 = Candidate(data="ATCG", modality=Modality.SEQUENCE, features={})
        assert c1 == c2

    def test_equality_with_torch_tensors_in_features(self):
        """Test equality with torch tensors in features."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        c1 = Candidate(data="ATCG", modality=Modality.SEQUENCE, features={"tensor": tensor})
        c2 = Candidate(data="ATCG", modality=Modality.SEQUENCE, features={"tensor": tensor.clone()})
        # torch.equal() should be used for exact tensor comparison, but our fallback handles it
        assert c1 == c2

    def test_equality_with_nan_in_numpy_arrays(self):
        """Test equality with NaN values in numpy arrays."""
        arr_with_nan = np.array([1.0, np.nan, 3.0])
        c1 = Candidate(data="ATCG", modality=Modality.SEQUENCE, features={"values": arr_with_nan})
        c2 = Candidate(
            data="ATCG", modality=Modality.SEQUENCE, features={"values": arr_with_nan.copy()}
        )
        # np.array_equal treats NaN as equal to NaN
        assert c1 == c2

    def test_equality_with_different_numpy_dtypes(self):
        """Test equality with numpy arrays of different dtypes but same values."""
        c1 = Candidate(
            data="ATCG",
            modality=Modality.SEQUENCE,
            features={"arr": np.array([1, 2, 3], dtype=np.int32)},
        )
        c2 = Candidate(
            data="ATCG",
            modality=Modality.SEQUENCE,
            features={"arr": np.array([1, 2, 3], dtype=np.int64)},
        )
        # np.array_equal actually considers different dtypes with same values as equal
        assert c1 == c2

    def test_equality_with_non_candidate_object(self):
        """Test that comparing with non-Candidate returns False."""
        c1 = Candidate(data="ATCG", modality=Modality.SEQUENCE)
        assert c1 != "not a candidate"
        assert c1 != 42
        assert c1 is not None

    def test_unhashable(self):
        """Test that Candidate objects are unhashable."""
        c = Candidate(data="ATCG", modality=Modality.SEQUENCE)
        with pytest.raises(TypeError):
            hash(c)

    def test_cannot_use_in_set(self):
        """Test that Candidate objects cannot be added to sets."""
        c1 = Candidate(data="ATCG", modality=Modality.SEQUENCE)
        c2 = Candidate(data="GCTA", modality=Modality.SEQUENCE)
        with pytest.raises(TypeError):
            {c1, c2}

    def test_equality_preserves_in_operator_semantics(self):
        """Test that equality works correctly with 'in' operator in lists."""
        c1 = Candidate(data="ATCG", modality=Modality.SEQUENCE, features={"arr": np.array([1, 2])})
        c2 = Candidate(data="ATCG", modality=Modality.SEQUENCE, features={"arr": np.array([1, 2])})
        c3 = Candidate(data="GCTA", modality=Modality.SEQUENCE)

        candidates_list = [c1, c3]
        # c2 should be found because it's equal to c1
        assert c2 in candidates_list
        # But note: using 'in' checks equality, not identity
        assert c1 in candidates_list


class TestCandidateToSerializable:
    """Test cases for Candidate.to_serializable method."""

    def test_to_serializable_sequence_modality(self):
        """Test to_serializable with sequence modality returns stringified data."""
        seq_data = "MKTFFVAGLVLLLTICSASG"
        candidate = Candidate(data=seq_data, modality=Modality.SEQUENCE)
        result = candidate.to_serializable()

        assert isinstance(result, str)
        assert result == seq_data

    def test_to_serializable_sequence_empty_string(self):
        """Test to_serializable with empty sequence."""
        candidate = Candidate(data="", modality=Modality.SEQUENCE)
        result = candidate.to_serializable()

        assert isinstance(result, str)
        assert result == ""  # noqa: PLC1901

    def test_to_serializable_tabular_dict(self):
        """Test to_serializable with tabular modality (dict) returns raw data."""
        tabular_data = {"age": 32, "height": 178, "weight": 70}
        candidate = Candidate(data=tabular_data, modality=Modality.TABULAR)
        result = candidate.to_serializable()

        assert isinstance(result, dict)
        assert result == tabular_data
        assert result["age"] == 32

    def test_to_serializable_tabular_pandas_series(self):
        """Test to_serializable with tabular modality (pandas Series) returns raw data."""
        tabular_data = pd.Series({"age": 32, "height": 178, "weight": 70})
        candidate = Candidate(data=tabular_data, modality=Modality.TABULAR)
        result = candidate.to_serializable()

        assert isinstance(result, pd.Series)
        assert result.equals(tabular_data)
        assert result["age"] == 32

    def test_to_serializable_tabular_numpy_array(self):
        """Test to_serializable with tabular modality (numpy array) returns raw data."""
        tabular_data = np.array([1, 2, 3, 4, 5])
        candidate = Candidate(data=tabular_data, modality=Modality.TABULAR)
        result = candidate.to_serializable()

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, tabular_data)

    def test_to_serializable_graph_modality(self):
        """Test to_serializable with graph modality raises NotImplementedError."""
        graph_data = nx.erdos_renyi_graph(n=10, p=0.3)
        candidate = Candidate(data=graph_data, modality=Modality.GRAPH)

        with pytest.raises(NotImplementedError, match="Graph datatype not supported yet"):
            candidate.to_serializable()

    def test_to_serializable_structure_modality(self):
        """Test to_serializable with structure modality returns raw structure data."""
        coords = np.random.rand(50, 3)
        candidate = Candidate(data=coords, modality=Modality.STRUCTURE)
        result = candidate.to_serializable()

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, coords)
        assert result.shape == (50, 3)

    def test_to_serializable_structure_string(self):
        """Test to_serializable with structure modality as JSON-encoded string passes through."""
        json_structure = '{"lattice": [[3.84, 0, 0], [0, 3.84, 0], [0, 0, 3.84]], "sites": []}'
        candidate = Candidate(data=json_structure, modality=Modality.STRUCTURE)
        result = candidate.to_serializable()

        assert isinstance(result, str)
        assert result == json_structure

    def test_to_serializable_structure_invalid_type(self):
        """Test to_serializable with structure modality raises TypeError for unsupported types."""
        candidate = Candidate(data={"key": "value"}, modality=Modality.STRUCTURE)
        with pytest.raises(
            TypeError,
            match="STRUCTURE modality data must be a string, numpy array, "
            "torch tensor, or ASE Atoms",
        ):
            candidate.to_serializable()

    def test_to_serializable_structure_ase_atoms_roundtrip(self):
        """Test to_serializable with structure modality serialises ASE Atoms losslessly."""
        ase = pytest.importorskip("ase")
        ase_io = pytest.importorskip("ase.io")
        atoms = ase.Atoms(
            "H2O",
            positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
            cell=[5, 5, 5],
            pbc=True,
        )
        candidate = Candidate(data=atoms, modality=Modality.STRUCTURE)
        result = candidate.to_serializable()

        assert isinstance(result, str)
        restored = ase_io.read(io.StringIO(result), format="json")
        assert list(restored.symbols) == list(atoms.symbols)
        assert np.allclose(restored.get_positions(), atoms.get_positions())
        assert np.allclose(restored.get_cell(), atoms.get_cell())

    def test_to_serializable_preserves_features(self):
        """Test that to_serializable doesn't modify candidate features."""
        features = {"key": "value", "number": 42}
        candidate = Candidate(data="ACDEFG", modality=Modality.SEQUENCE, features=features)
        result = candidate.to_serializable()

        assert result == "ACDEFG"
        assert candidate.features == features

    def test_to_serializable_none_data(self):
        """Test to_serializable with None data for non-sequence modality."""
        candidate = Candidate(data=None, modality=Modality.STRUCTURE)
        result = candidate.to_serializable()

        assert result is None


@pytest.mark.parametrize(
    "modality,data_factory,expected_type",
    [
        (Modality.SEQUENCE, lambda: "ATCGATCG", str),
        (Modality.TABULAR, lambda: {"col1": 1, "col2": 2}, dict),
        (Modality.TABULAR, lambda: np.array([1, 2, 3]), np.ndarray),
        (Modality.STRUCTURE, lambda: np.random.rand(10, 3), np.ndarray),
    ],
)
def test_to_serializable_modality_types(modality, data_factory, expected_type):
    """Parametrized test for to_serializable with different modalities."""
    data = data_factory()
    candidate = Candidate(data=data, modality=modality)
    result = candidate.to_serializable()

    assert isinstance(result, expected_type)

    # For sequence, result should be stringified
    if modality == Modality.SEQUENCE:
        assert result == data
    # For tabular and other modalities, result should be raw data
    elif modality == Modality.TABULAR:
        if isinstance(data, np.ndarray):
            assert np.array_equal(result, data)
        elif isinstance(data, dict):
            assert result == data
    else:
        # For STRUCTURE modality
        if isinstance(data, np.ndarray):
            assert np.array_equal(result, data)
        elif isinstance(data, torch.Tensor):
            # Torch tensors are converted to numpy arrays
            assert isinstance(result, np.ndarray)
            assert np.array_equal(result, data.cpu().numpy())
        else:
            assert result is data
