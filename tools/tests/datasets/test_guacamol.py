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
