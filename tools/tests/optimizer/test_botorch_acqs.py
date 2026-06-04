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

"""Tests for BoTorch acquisition function wrappers.

Fixtures are shared from tools/tests/conftest.py:
- mock_alf_model_with_variances
- botorch_gp_model
- test_tensor_2d
- test_candidates_2d
"""

import numpy as np
import pytest
import torch
from alf_core import LabelledCandidates
from alf_tools.optimizer.acquisition_functions.botorch_acquisition_function import (
    ACQUISITION_REGISTRY,
    BotorchAcquisitionConfig,
    BotorchAcquisitionFunction,
)


# =============================================================================
# BotorchAcquisitionConfig — name validation
# =============================================================================


def test_botorch_acquisition_config_accepts_valid_name():
    """BotorchAcquisitionConfig does not raise for a registered acquisition name."""
    cfg = BotorchAcquisitionConfig(name="expected_improvement", kwargs={"best_f": 0.0})
    assert cfg.name == "expected_improvement"
    assert cfg.kwargs == {"best_f": 0.0}


def test_botorch_acquisition_config_rejects_unknown_name():
    """BotorchAcquisitionConfig raises ValueError for an unregistered name."""
    with pytest.raises(ValueError, match="Unknown acquisition function"):
        BotorchAcquisitionConfig(name="nonexistent_acq")


# =============================================================================
# BotorchAcquisitionConfig — kwargs validation
# =============================================================================


def test_botorch_acquisition_config_rejects_missing_best_f():
    """Missing best_f for expected_improvement raises ValueError."""
    with pytest.raises(ValueError, match="Missing required kwargs"):
        BotorchAcquisitionConfig(name="expected_improvement", kwargs={})


def test_botorch_acquisition_config_rejects_missing_beta():
    """Missing beta for upper_confidence_bound raises ValueError."""
    with pytest.raises(ValueError, match="Missing required kwargs"):
        BotorchAcquisitionConfig(name="upper_confidence_bound", kwargs={})


def test_botorch_acquisition_config_rejects_missing_x_baseline():
    """Missing X_baseline for log_noisy_expected_improvement raises ValueError."""
    with pytest.raises(ValueError, match="Missing required kwargs"):
        BotorchAcquisitionConfig(name="log_noisy_expected_improvement", kwargs={})


def test_botorch_acquisition_config_accepts_all_required_kwargs():
    """All registered acquisition functions accept their required kwargs without error."""
    valid_kwargs = {
        "expected_improvement": {"best_f": 0.0},
        "upper_confidence_bound": {"beta": 2.0},
        "probability_of_improvement": {"best_f": 0.0},
        "log_noisy_expected_improvement": {"X_baseline": torch.rand(3, 2)},
    }
    for name in ACQUISITION_REGISTRY:
        cfg = BotorchAcquisitionConfig(name=name, kwargs=valid_kwargs[name])
        assert cfg.name == name


# =============================================================================
# BotorchAcquisitionFunction — helpers
# =============================================================================


class _MockSurrogate:
    """Minimal surrogate stub that exposes a .model attribute."""

    def __init__(self, model):
        self.model = model


class _MockState:
    """Minimal state stub with surrogate."""

    def __init__(self, model):
        self.surrogate = _MockSurrogate(model)


# =============================================================================
# BotorchAcquisitionFunction tests
# =============================================================================


def test_botorch_acquisition_function_returns_labelled_candidates(
    botorch_gp_model, test_candidates_2d
):
    """BotorchAcquisitionFunction returns LabelledCandidates with one score per candidate."""
    cfg = BotorchAcquisitionConfig(name="expected_improvement", kwargs={"best_f": 0.0})
    acq_fn = BotorchAcquisitionFunction(cfg)
    state = _MockState(botorch_gp_model)

    result = acq_fn(test_candidates_2d, state)

    assert isinstance(result, LabelledCandidates)
    assert len(result.labels) == len(test_candidates_2d)


def test_botorch_acquisition_function_scores_are_finite(botorch_gp_model, test_candidates_2d):
    """BotorchAcquisitionFunction produces finite scores."""
    cfg = BotorchAcquisitionConfig(name="expected_improvement", kwargs={"best_f": 0.0})
    acq_fn = BotorchAcquisitionFunction(cfg)
    state = _MockState(botorch_gp_model)

    result = acq_fn(test_candidates_2d, state)

    assert np.all(np.isfinite(result.labels))


def test_botorch_acquisition_function_wraps_alf_model(
    mock_alf_model_with_variances, test_candidates_2d
):
    """BotorchAcquisitionFunction adapts an ALF BaseModel via BoTorchModelAdapter."""
    cfg = BotorchAcquisitionConfig(name="upper_confidence_bound", kwargs={"beta": 2.0})
    acq_fn = BotorchAcquisitionFunction(cfg)
    state = _MockState(mock_alf_model_with_variances)

    result = acq_fn(test_candidates_2d, state)

    assert len(result.labels) == len(test_candidates_2d)
    assert np.all(np.isfinite(result.labels))
