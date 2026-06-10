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

"""Tests for the YAML loader (parses into the shared pydantic configs)."""

from pathlib import Path

import pytest
from alf_benchmark.yaml_loader import load_run_config
from pydantic import ValidationError

_EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "example.yaml"


def test_loads_example_config():
    """The shipped example YAML parses into a valid RunConfig."""
    run = load_run_config(_EXAMPLE)
    assert run.suite_name == "example"
    assert len(run.problems) == 1
    assert run.problems[0].seeds == [0, 1, 2]
    method_names = {m.name for m in run.methods}
    assert method_names == {"cnn_greedy", "cnn_ucb"}
    ucb = next(m for m in run.methods if m.name == "cnn_ucb")
    assert ucb.acquisition.config == {"alpha": 2.0}


def test_non_mapping_top_level_raises(tmp_path):
    """A YAML whose top level is not a mapping raises a clear error."""
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_run_config(path)


def test_invalid_config_raises_validation_error(tmp_path):
    """A structurally valid YAML that violates the schema raises ValidationError."""
    path = tmp_path / "invalid.yaml"
    # design method missing required acquisition/search.
    path.write_text(
        "output_dir: runs\n"
        "problems:\n"
        "  - name: p\n"
        "    dataset: {name: gfp}\n"
        "methods:\n"
        "  - name: m\n"
        "    family: design\n"
        "    surrogate: {name: cnn}\n"
    )
    with pytest.raises(ValidationError):
        load_run_config(path)
