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

"""Tests for the PyRosetta model. Skipped unless PyRosetta is installed."""

import pytest

pytest.importorskip("pyrosetta")

from alf_tools.models.pyrosetta import PyRosetta  # noqa: E402


def test_mutate_and_relax_rejects_length_mismatch():
    """A sequence whose length differs from the wild-type raises ValueError instead
    of being silently truncated by zip (only point substitutions are supported).
    """
    model = PyRosetta.__new__(PyRosetta)  # bypass __init__ (no PDB / pose needed)
    model.wt_sequence = "ACDEFG"

    with pytest.raises(ValueError, match="length must match"):
        model.mutate_and_relax("ACDE")
