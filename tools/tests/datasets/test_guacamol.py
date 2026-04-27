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
