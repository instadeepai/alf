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

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def test_require_rdkit_raises_informative_error_when_unavailable():
    """_require_rdkit() must raise ImportError with install instructions."""
    import alf_tools.datasets.guacamol as gm
    original = gm._RDKIT_AVAILABLE
    gm._RDKIT_AVAILABLE = False
    try:
        with pytest.raises(ImportError, match="alf_tools\\[benchmarks\\]"):
            gm._require_rdkit()
    finally:
        gm._RDKIT_AVAILABLE = original


from alf_tools.datasets.guacamol import (
    GuacaMolConfig, FILENAME_ALL, FILENAME_TRAIN, FILENAME_VALID, FILENAME_TEST,
)


def _base_config(**overrides) -> GuacaMolConfig:
    defaults = dict(
        name="guacamol",
        modality="sequence",
        seed=42,
        train_ratio=0.6,
        validation_frac=0.1,
        test_ratio=0.2,
        target_property="TPSA",
    )
    defaults.update(overrides)
    return GuacaMolConfig(**defaults)


class TestGuacaMolConfig:
    def test_task_type_auto_set_to_property(self):
        config = _base_config(target_property="TPSA")
        assert config.task_type == "property"

    def test_task_type_auto_set_to_benchmark_task(self):
        config = _base_config(target_property="celecoxib_rediscovery")
        assert config.task_type == "benchmark_task"

    def test_default_max_molecules_is_none(self):
        assert _base_config().max_molecules is None

    def test_default_split_mode_is_random(self):
        assert _base_config().split_mode == "random"

    def test_default_computed_properties_is_none(self):
        assert _base_config().computed_properties is None

    def test_target_property_absent_from_explicit_computed_properties_raises(self):
        with pytest.raises(ValueError, match="target_property"):
            _base_config(target_property="TPSA", computed_properties=["MolLogP", "MolWt"])

    def test_target_property_present_in_computed_properties_is_valid(self):
        config = _base_config(target_property="TPSA", computed_properties=["TPSA", "MolLogP"])
        assert config.computed_properties == ["TPSA", "MolLogP"]

    def test_split_type_synced_with_split_mode_low_vs_high(self):
        config = _base_config(split_mode="low_vs_high")
        assert config.split_type == "low_vs_high"

    def test_split_mode_paper_does_not_alter_split_type(self):
        config = _base_config(split_mode="paper")
        assert config.split_mode == "paper"

    def test_computed_properties_none_always_valid_regardless_of_target(self):
        for prop in ["TPSA", "MolWt", "QED"]:
            config = _base_config(target_property=prop, computed_properties=None)
            assert config.task_type == "property"


class TestComputeProperties:
    def test_returns_all_keys_when_all_properties_requested(self):
        from alf_tools.datasets.guacamol import _compute_properties, ALL_PROPERTIES
        result = _compute_properties("c1ccccc1", list(ALL_PROPERTIES))
        assert set(result.keys()) == ALL_PROPERTIES

    def test_returns_only_requested_subset(self):
        from alf_tools.datasets.guacamol import _compute_properties
        result = _compute_properties("c1ccccc1", ["TPSA", "MolWt"])
        assert set(result.keys()) == {"TPSA", "MolWt"}

    def test_tpsa_of_benzene_is_zero(self):
        from alf_tools.datasets.guacamol import _compute_properties
        result = _compute_properties("c1ccccc1", ["TPSA"])
        assert result["TPSA"] == pytest.approx(0.0, abs=1e-3)

    def test_molwt_of_ethanol(self):
        from alf_tools.datasets.guacamol import _compute_properties
        result = _compute_properties("CCO", ["MolWt"])
        assert result["MolWt"] == pytest.approx(46.069, rel=1e-3)

    def test_all_values_are_float(self):
        from alf_tools.datasets.guacamol import _compute_properties, ALL_PROPERTIES
        result = _compute_properties("CC(=O)O", list(ALL_PROPERTIES))
        for key, val in result.items():
            assert isinstance(val, float), f"{key} value is not a float"


VALID_SMILES_LINES = [
    "c1ccccc1",
    "CC(=O)O",
    "CCO",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "c1ccc2ccccc2c1",
]
INVALID_SMILES_LINE = "NOTASMILES"


def _make_mock_response(smiles_lines: list[str]) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = iter(line.encode() for line in smiles_lines)
    return mock_resp


class TestGuacaMolSingleFileLoad:
    def test_load_returns_correct_number_of_candidates(self, tmp_path):
        config = _base_config(max_molecules=5)
        with patch("alf_tools.datasets.guacamol.DATAPATH", tmp_path), \
             patch("requests.get", return_value=_make_mock_response(VALID_SMILES_LINES)):
            from alf_tools.datasets.guacamol import GuacaMol
            dataset = GuacaMol(config)
        assert len(dataset._raw_dataset) == 5

    def test_labels_are_1d_numpy_array(self, tmp_path):
        config = _base_config(max_molecules=5)
        with patch("alf_tools.datasets.guacamol.DATAPATH", tmp_path), \
             patch("requests.get", return_value=_make_mock_response(VALID_SMILES_LINES)):
            from alf_tools.datasets.guacamol import GuacaMol
            dataset = GuacaMol(config)
        assert isinstance(dataset._raw_dataset.labels, np.ndarray)
        assert dataset._raw_dataset.labels.ndim == 1
        assert len(dataset._raw_dataset.labels) == 5

    def test_candidate_features_contain_computed_properties(self, tmp_path):
        config = _base_config(max_molecules=3, computed_properties=["TPSA", "MolWt"])
        with patch("alf_tools.datasets.guacamol.DATAPATH", tmp_path), \
             patch("requests.get", return_value=_make_mock_response(VALID_SMILES_LINES)):
            from alf_tools.datasets.guacamol import GuacaMol
            dataset = GuacaMol(config)
        for cand in dataset._raw_dataset.candidates:
            assert "TPSA" in cand.features
            assert "MolWt" in cand.features

    def test_invalid_smiles_are_skipped_and_not_in_dataset(self, tmp_path):
        lines_with_invalid = VALID_SMILES_LINES[:3] + [INVALID_SMILES_LINE] + VALID_SMILES_LINES[3:]
        config = _base_config(max_molecules=len(lines_with_invalid))
        with patch("alf_tools.datasets.guacamol.DATAPATH", tmp_path), \
             patch("requests.get", return_value=_make_mock_response(lines_with_invalid)):
            from alf_tools.datasets.guacamol import GuacaMol
            dataset = GuacaMol(config)
        assert len(dataset._raw_dataset) == len(VALID_SMILES_LINES)

    def test_max_molecules_caps_corpus_size(self, tmp_path):
        config = _base_config(max_molecules=3)
        with patch("alf_tools.datasets.guacamol.DATAPATH", tmp_path), \
             patch("requests.get", return_value=_make_mock_response(VALID_SMILES_LINES)):
            from alf_tools.datasets.guacamol import GuacaMol
            dataset = GuacaMol(config)
        assert len(dataset._raw_dataset) <= 3

    def test_benchmark_task_target_raises_not_implemented(self, tmp_path):
        config = _base_config(target_property="celecoxib_rediscovery")
        with patch("alf_tools.datasets.guacamol.DATAPATH", tmp_path):
            from alf_tools.datasets.guacamol import GuacaMol
            with pytest.raises(NotImplementedError):
                GuacaMol(config)

    def test_no_download_if_file_already_cached(self, tmp_path):
        (tmp_path / FILENAME_ALL).write_text("\n".join(VALID_SMILES_LINES[:3]))
        config = _base_config(max_molecules=3)
        with patch("alf_tools.datasets.guacamol.DATAPATH", tmp_path), \
             patch("requests.get") as mock_get:
            from alf_tools.datasets.guacamol import GuacaMol
            GuacaMol(config)
        mock_get.assert_not_called()
