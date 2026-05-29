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

import gpytorch
import pytest
from alf_tools.models.utils.config_utils import build_from_target


def test_build_from_target_none_returns_none():
    assert build_from_target(None) is None


def test_build_from_target_lognormal_prior():
    cfg = {
        "_target_": "gpytorch.priors.LogNormalPrior",
        "loc": 0.0,
        "scale": 1.0,
    }
    prior = build_from_target(cfg)
    assert isinstance(prior, gpytorch.priors.LogNormalPrior)


def test_build_from_target_gamma_prior():
    cfg = {
        "_target_": "gpytorch.priors.GammaPrior",
        "concentration": 3.0,
        "rate": 6.0,
    }
    prior = build_from_target(cfg)
    assert isinstance(prior, gpytorch.priors.GammaPrior)


def test_build_from_target_greater_than_constraint():
    cfg = {
        "_target_": "gpytorch.constraints.GreaterThan",
        "lower_bound": 0.01,
    }
    constraint = build_from_target(cfg)
    assert isinstance(constraint, gpytorch.constraints.GreaterThan)


def test_build_from_target_does_not_mutate_input():
    cfg = {
        "_target_": "gpytorch.priors.LogNormalPrior",
        "loc": 0.0,
        "scale": 1.0,
    }
    original_keys = set(cfg.keys())
    build_from_target(cfg)
    assert set(cfg.keys()) == original_keys  # _target_ not popped from original


def test_build_from_target_invalid_target_raises():
    cfg = {"_target_": "gpytorch.priors.NonExistentPrior"}
    with pytest.raises(AttributeError):
        build_from_target(cfg)
