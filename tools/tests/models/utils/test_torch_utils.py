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

"""Tests for torch utility functions."""

import torch
from alf_tools.models.utils import get_device


class TestTorchUtils:
    """Test suite for torch utility functions."""

    def test_get_device_auto(self):
        """Test automatic device selection."""
        device = get_device(None)

        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda"]

    def test_get_device_explicit_cpu(self):
        """Test explicit CPU device."""
        device = get_device("cpu")

        assert device.type == "cpu"
