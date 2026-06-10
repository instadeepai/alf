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

"""Tests for the pydantic configuration models."""

import pytest
from alf_benchmark.config import (
    ComponentSpec,
    MethodConfig,
    OracleSpec,
    ProblemConfig,
    RunConfig,
)
from pydantic import ValidationError


def test_valid_design_method_parses():
    """A complete design method config validates successfully."""
    method = MethodConfig(
        name="cnn_ucb",
        family="design",
        surrogate=ComponentSpec(name="cnn"),
        acquisition=ComponentSpec(name="ucb", config={"alpha": 2.0}),
        search=ComponentSpec(name="dataset_search"),
    )
    assert method.acquisition is not None
    assert method.oracle.mode == "offline"


def test_design_method_requires_acquisition_and_search():
    """A design method without acquisition/search is rejected."""
    with pytest.raises(ValidationError):
        MethodConfig(name="bad", family="design", surrogate=ComponentSpec(name="cnn"))


def test_online_oracle_requires_scorer():
    """An online oracle without a scorer is rejected."""
    with pytest.raises(ValidationError):
        OracleSpec(mode="online")


def test_offline_oracle_forbids_scorer():
    """An offline oracle with a scorer is rejected."""
    with pytest.raises(ValidationError):
        OracleSpec(mode="offline", scorer=ComponentSpec(name="esmfold"))


def test_problem_requires_at_least_one_seed():
    """A problem with no seeds is rejected."""
    with pytest.raises(ValidationError):
        ProblemConfig(name="p", dataset=ComponentSpec(name="gfp"), seeds=[])


def test_unknown_field_is_rejected():
    """Unknown keys are rejected (extra='forbid')."""
    with pytest.raises(ValidationError):
        ComponentSpec(name="cnn", typo=True)


def test_run_config_parses():
    """A top-level run config with a problem and method validates."""
    run = RunConfig(
        output_dir="runs",
        problems=[ProblemConfig(name="p", dataset=ComponentSpec(name="gfp"))],
        methods=[
            MethodConfig(
                name="m",
                surrogate=ComponentSpec(name="cnn"),
                acquisition=ComponentSpec(name="greedy"),
                search=ComponentSpec(name="dataset_search"),
            )
        ],
    )
    assert run.output_dir == "runs"
    assert len(run.problems) == 1
    assert len(run.methods) == 1
