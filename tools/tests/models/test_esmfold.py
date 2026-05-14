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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from alf_core import Candidate, Predictions
from alf_core.dataclasses import LabelledCandidates
from alf_core.dataclasses.candidate import Modality
from alf_core.oracle.oracle import Oracle

from alf_tools.models.esmfold import ESMFoldConfig, ESMFoldModel

MOCK_PTM = 0.7
MOCK_PLDDT = 60.0  # raw; /100 → 0.6


class TestESMFoldConfig:
    """Tests for ESMFoldConfig defaults and field acceptance."""

    def test_default_model_name(self):
        """Default model_name is facebook/esmfold_v1."""
        assert ESMFoldConfig().model_name == "facebook/esmfold_v1"

    def test_default_device(self):
        """Default device is cpu."""
        assert ESMFoldConfig().device == "cpu"

    def test_default_scoring_metric(self):
        """Default scoring_metric is ptm."""
        assert ESMFoldConfig().scoring_metric == "ptm"

    def test_default_combined_ptm_weight(self):
        """Default combined_ptm_weight is 0.5."""
        assert ESMFoldConfig().combined_ptm_weight == 0.5

    def test_default_batch_size(self):
        """Default batch_size is 1."""
        assert ESMFoldConfig().batch_size == 1

    def test_default_chunk_size_is_none(self):
        """Default chunk_size is None."""
        assert ESMFoldConfig().chunk_size is None

    def test_default_low_memory_is_false(self):
        """Default low_memory is False."""
        assert ESMFoldConfig().low_memory is False

    def test_each_valid_scoring_metric_accepted(self):
        """Each of ptm, mean_plddt, combined is accepted."""
        for metric in ("ptm", "mean_plddt", "combined"):
            assert ESMFoldConfig(scoring_metric=metric).scoring_metric == metric

    def test_combined_ptm_weight_boundary_zero(self):
        """combined_ptm_weight=0.0 is accepted."""
        assert ESMFoldConfig(combined_ptm_weight=0.0).combined_ptm_weight == 0.0

    def test_combined_ptm_weight_boundary_one(self):
        """combined_ptm_weight=1.0 is accepted."""
        assert ESMFoldConfig(combined_ptm_weight=1.0).combined_ptm_weight == 1.0

    def test_chunk_size_positive_accepted(self):
        """chunk_size=64 is accepted."""
        assert ESMFoldConfig(chunk_size=64).chunk_size == 64

    def test_batch_size_greater_than_one_accepted(self):
        """batch_size=4 is accepted."""
        assert ESMFoldConfig(batch_size=4).batch_size == 4

    def test_local_path_accepted_as_model_name(self):
        """model_name can be a local filesystem path."""
        cfg = ESMFoldConfig(model_name="/models/esmfold_v1")
        assert cfg.model_name == "/models/esmfold_v1"
