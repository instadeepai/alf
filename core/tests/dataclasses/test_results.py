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
import torch
from alf_core.dataclasses.predictions import Predictions
from alf_core.dataclasses.results import Results


@pytest.fixture
def predictions():
    """Return a Predictions instance with numpy array means."""
    return Predictions(means=np.array([1.0, 2.0, 3.0]))


class TestResultsNumpyTypeValidation:
    """Test that Results rejects non-numpy arrays for targets."""

    def test_torch_targets_raises_type_error(self, predictions):
        """Test that torch tensor targets raises TypeError."""
        targets = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(TypeError, match="targets must be a numpy array"):
            Results(targets=targets, predictions=predictions)

    def test_numpy_targets_accepted(self, predictions):
        """Test that numpy array targets are accepted."""
        targets = np.array([1.0, 2.0, 3.0])
        results = Results(targets=targets, predictions=predictions)
        assert isinstance(results.targets, np.ndarray)


class TestResultsValidation:
    """Test cases for Results validation."""

    def test_mismatched_lengths_raises(self, predictions):
        """Test that mismatched targets and predictions lengths raises AssertionError."""
        targets = np.array([1.0, 2.0])  # length 2 vs predictions length 3
        with pytest.raises(AssertionError, match="same length"):
            Results(targets=targets, predictions=predictions)

    def test_results_computes_metrics(self, predictions):
        """Test that Results computes metrics on valid inputs."""
        targets = np.array([1.0, 2.0, 3.0])
        results = Results(targets=targets, predictions=predictions)
        assert isinstance(results.metrics, dict)
        assert len(results.metrics) > 0
