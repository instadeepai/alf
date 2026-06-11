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
    """Tests that build_from_target returns None when given None as input."""
    assert build_from_target(None) is None


def test_build_from_target_lognormal_prior():
    """Tests that build_from_target can construct a LogNormalPrior from a
    valid configuration. This verifies that the function correctly resolves
    the class and initializes it with the provided parameters.
    """
    cfg = {
        "_target_": "gpytorch.priors.LogNormalPrior",
        "loc": 0.0,
        "scale": 1.0,
    }
    prior = build_from_target(cfg)
    assert isinstance(prior, gpytorch.priors.LogNormalPrior)


def test_build_from_target_gamma_prior():
    """Tests that build_from_target can construct a GammaPrior from a valid
    configuration. This verifies that the function correctly resolves the
    class and initializes it with the provided parameters.
    """
    cfg = {
        "_target_": "gpytorch.priors.GammaPrior",
        "concentration": 3.0,
        "rate": 6.0,
    }
    prior = build_from_target(cfg)
    assert isinstance(prior, gpytorch.priors.GammaPrior)


def test_build_from_target_greater_than_constraint():
    """Tests that build_from_target can construct a GreaterThan constraint
    from a valid configuration. This verifies that the function correctly
    resolves the class and initializes it with the provided parameters.
    """
    cfg = {
        "_target_": "gpytorch.constraints.GreaterThan",
        "lower_bound": 0.01,
    }
    constraint = build_from_target(cfg)
    assert isinstance(constraint, gpytorch.constraints.GreaterThan)


def test_build_from_target_does_not_mutate_input():
    """Tests that the input configuration dictionary is not mutated by
    build_from_target. This ensures that the original configuration can be
    reused or inspected after building.
    """
    cfg = {
        "_target_": "gpytorch.priors.LogNormalPrior",
        "loc": 0.0,
        "scale": 1.0,
    }
    original_keys = set(cfg.keys())
    build_from_target(cfg)
    assert set(cfg.keys()) == original_keys  # _target_ not popped from original


def test_build_from_target_invalid_target_raises():
    """Tests that an invalid _target_ value raises an AttributeError."""
    cfg = {"_target_": "gpytorch.priors.NonExistentPrior"}
    with pytest.raises(AttributeError):
        build_from_target(cfg)


def test_build_from_target_blocks_disallowed_module():
    """Tests that a _target_ outside the allowed modules raises ValueError."""
    cfg = {"_target_": "os.path.join"}
    with pytest.raises(ValueError, match="not in the allowed module list"):
        build_from_target(cfg)


@pytest.mark.parametrize("target", ["os.system", "builtins.eval"])
def test_build_from_target_blocks_dangerous_targets(target):
    """Tests that dangerous targets outside the allowed modules raise ValueError."""
    with pytest.raises(ValueError, match="not in the allowed module list"):
        build_from_target({"_target_": target})


def test_build_from_target_blocks_non_class_attribute():
    """Tests that an allowed-module attribute that is not a class raises ValueError."""
    # `gpytorch.priors.smoothed_box_prior` is a submodule, not a class
    cfg = {"_target_": "gpytorch.priors.smoothed_box_prior"}
    with pytest.raises(ValueError, match="does not resolve to a subclass"):
        build_from_target(cfg)


def test_build_from_target_blocks_private_name():
    """Tests that a private (leading underscore) class name raises ValueError."""
    cfg = {"_target_": "gpytorch.priors._private_thing"}
    with pytest.raises(ValueError, match="private attribute"):
        build_from_target(cfg)


def test_build_from_target_prior_rejected_when_constraint_expected():
    """Tests that a prior target fails validation against a constraint base."""
    cfg = {"_target_": "gpytorch.priors.LogNormalPrior", "loc": 0.0, "scale": 1.0}
    with pytest.raises(ValueError, match="does not resolve to a subclass"):
        build_from_target(cfg, expected_base=gpytorch.constraints.Interval)


def test_build_from_target_constraint_rejected_when_prior_expected():
    """Tests that a constraint target fails validation against a prior base."""
    cfg = {"_target_": "gpytorch.constraints.GreaterThan", "lower_bound": 0.01}
    with pytest.raises(ValueError, match="does not resolve to a subclass"):
        build_from_target(cfg, expected_base=gpytorch.priors.Prior)


def test_build_from_target_with_explicit_expected_base():
    """Tests that valid targets instantiate when expected_base matches."""
    prior = build_from_target(
        {"_target_": "gpytorch.priors.LogNormalPrior", "loc": 0.0, "scale": 1.0},
        expected_base=gpytorch.priors.Prior,
    )
    assert isinstance(prior, gpytorch.priors.LogNormalPrior)

    constraint = build_from_target(
        {"_target_": "gpytorch.constraints.GreaterThan", "lower_bound": 0.01},
        expected_base=gpytorch.constraints.Interval,
    )
    assert isinstance(constraint, gpytorch.constraints.GreaterThan)


def test_build_from_target_bad_kwargs_raise_value_error():
    """Tests that constructor kwargs rejected by the target raise ValueError."""
    cfg = {"_target_": "gpytorch.priors.LogNormalPrior", "bogus_kwarg": 1.0}
    with pytest.raises(ValueError, match="gpytorch.priors.LogNormalPrior"):
        build_from_target(cfg)
