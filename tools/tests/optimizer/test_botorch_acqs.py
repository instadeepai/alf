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

Covers the `acquisition` decorator, `_AcquisitionCallable`, and the two
concrete factories `expected_improvement` and `upper_confidence_bound`.

Fixtures are shared from tools/tests/conftest.py:
- mock_alf_model_with_variances
- botorch_gp_model
- test_tensor_2d
"""

import numpy as np
import pytest
import torch
from alf_core import LabelledCandidates, Predictions
from alf_tools.optimizer.acquisition_functions.botorch_acquisition_function import (
    BotorchAcquisitionConfig,
    BotorchAcquisitionFunction,
    _AcquisitionCallable,  # noqa: PLC2701
    acquisition,
    expected_improvement,
    upper_confidence_bound,
)
from alf_tools.optimizer.acquisition_functions.utils.botorch_model_adapter import (
    BoTorchModelAdapter,
)
from botorch.acquisition.analytic import (
    ExpectedImprovement as _BotorchEI,
)
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from omegaconf import OmegaConf

# =============================================================================
# _AcquisitionCallable attribute tests
# =============================================================================


def test_acquisition_callable_stores_name_and_kwargs(botorch_gp_model):
    """_AcquisitionCallable stores name and kwargs for config serialisation."""
    acq = _BotorchEI(model=botorch_gp_model, best_f=0.42)
    callable_ = _AcquisitionCallable(acq, name="expected_improvement", kwargs={"best_f": 0.42})

    assert callable_.name == "expected_improvement"
    assert callable_.kwargs == {"best_f": 0.42}


def test_acquisition_callable_call_returns_predictions(botorch_gp_model, test_tensor_2d):
    """__call__ returns a Predictions object."""
    acq = _BotorchEI(model=botorch_gp_model, best_f=0.0)
    callable_ = _AcquisitionCallable(acq, name="expected_improvement", kwargs={"best_f": 0.0})

    result = callable_(test_tensor_2d)

    assert isinstance(result, Predictions)


def test_acquisition_callable_scores_shape_matches_candidates(botorch_gp_model, test_tensor_2d):
    """The scores in Predictions.means have shape (n_candidates,)."""
    n = test_tensor_2d.shape[0]
    acq = _BotorchEI(model=botorch_gp_model, best_f=0.0)
    callable_ = _AcquisitionCallable(acq, name="expected_improvement", kwargs={"best_f": 0.0})

    result = callable_(test_tensor_2d)

    assert result.means.shape == (n,)


def test_acquisition_callable_scores_are_finite(botorch_gp_model, test_tensor_2d):
    """All acquisition scores must be finite (no NaN or inf)."""
    acq = _BotorchEI(model=botorch_gp_model, best_f=0.0)
    callable_ = _AcquisitionCallable(acq, name="expected_improvement", kwargs={"best_f": 0.0})

    result = callable_(test_tensor_2d)

    assert np.all(np.isfinite(result.means))


def test_acquisition_callable_ei_scores_nonnegative(botorch_gp_model, test_tensor_2d):
    """Expected Improvement scores are always >= 0."""
    acq = _BotorchEI(model=botorch_gp_model, best_f=0.0)
    callable_ = _AcquisitionCallable(acq, name="expected_improvement", kwargs={"best_f": 0.0})

    result = callable_(test_tensor_2d)

    assert np.all(result.means >= 0.0)


# =============================================================================
# acquisition decorator — model routing
# =============================================================================


def test_acquisition_decorator_passes_botorch_model_directly():
    """When the model is already a BoTorch Model, it is passed through without wrapping."""
    captured_models = []

    @acquisition
    def _test_acq(model, dummy: float = 1.0):
        captured_models.append(model)
        return _BotorchEI(model=model, best_f=0.0)

    torch.manual_seed(0)
    gp = SingleTaskGP(torch.rand(5, 2, dtype=torch.float32), torch.rand(5, 1, dtype=torch.float32))
    _test_acq(gp, dummy=1.0)

    assert captured_models[0] is gp


def test_acquisition_decorator_wraps_alf_model(mock_alf_model_with_variances):
    """When passed an ALF BaseModel, the decorator wraps it in BoTorchModelAdapter."""
    captured_models = []

    @acquisition
    def _test_acq(model, dummy: float = 1.0):
        captured_models.append(model)
        return _BotorchEI(model=model, best_f=0.0)

    _test_acq(mock_alf_model_with_variances, dummy=1.0)

    assert isinstance(captured_models[0], BoTorchModelAdapter)
    assert isinstance(captured_models[0], Model)


def test_acquisition_decorator_adapter_not_rewrapped():
    """A BoTorchModelAdapter (itself a Model) is not double-wrapped."""
    captured_models = []

    @acquisition
    def _test_acq(model, dummy: float = 1.0):
        captured_models.append(model)
        return _BotorchEI(model=model, best_f=0.0)

    torch.manual_seed(0)
    gp = SingleTaskGP(torch.rand(5, 2, dtype=torch.float32), torch.rand(5, 1, dtype=torch.float32))
    adapter = BoTorchModelAdapter(gp)
    _test_acq(adapter, dummy=1.0)

    assert captured_models[0] is adapter


# =============================================================================
# acquisition decorator — kwargs capture
# =============================================================================


def test_acquisition_decorator_captures_explicit_kwargs(botorch_gp_model):
    """Kwargs passed by name are stored on the returned callable."""
    acq = expected_improvement(botorch_gp_model, best_f=0.42, maximize=True)

    assert acq.kwargs["best_f"] == 0.42
    assert acq.kwargs["maximize"] is True


def test_acquisition_decorator_captures_default_kwargs(botorch_gp_model):
    """Default parameter values are also stored when not explicitly passed."""
    acq = expected_improvement(botorch_gp_model, best_f=0.5)

    # maximize has a default of True
    assert "maximize" in acq.kwargs
    assert acq.kwargs["maximize"] is True


def test_acquisition_decorator_captures_positional_args_as_kwargs(botorch_gp_model):
    """Positional args are stored under their parameter names."""
    acq = upper_confidence_bound(botorch_gp_model, 3.0)

    assert "beta" in acq.kwargs
    assert acq.kwargs["beta"] == 3.0


# =============================================================================
# expected_improvement factory
# =============================================================================


def test_expected_improvement_name(botorch_gp_model):
    """expected_improvement callable has the correct name attribute."""
    acq = expected_improvement(botorch_gp_model, best_f=0.0)

    assert acq.name == "expected_improvement"


def test_expected_improvement_returns_callable(botorch_gp_model):
    """expected_improvement returns an _AcquisitionCallable."""
    acq = expected_improvement(botorch_gp_model, best_f=0.0)

    assert isinstance(acq, _AcquisitionCallable)


def test_expected_improvement_scores_with_botorch_model(botorch_gp_model, test_tensor_2d):
    """EI produces finite, non-negative scores for a native BoTorch GP model."""
    acq = expected_improvement(botorch_gp_model, best_f=0.0)

    result = acq(test_tensor_2d)

    assert isinstance(result, Predictions)
    assert result.means.shape == (test_tensor_2d.shape[0],)
    assert np.all(np.isfinite(result.means))
    assert np.all(result.means >= 0.0)


def test_expected_improvement_scores_with_alf_model(mock_alf_model_with_variances, test_tensor_2d):
    """EI works transparently with an ALF BaseModel."""
    acq = expected_improvement(mock_alf_model_with_variances, best_f=0.0)

    result = acq(test_tensor_2d)

    assert isinstance(result, Predictions)
    assert result.means.shape == (test_tensor_2d.shape[0],)
    assert np.all(np.isfinite(result.means))
    assert np.all(result.means >= 0.0)


# =============================================================================
# upper_confidence_bound factory
# =============================================================================


def test_upper_confidence_bound_name(botorch_gp_model):
    """upper_confidence_bound callable has the correct name attribute."""
    acq = upper_confidence_bound(botorch_gp_model)

    assert acq.name == "upper_confidence_bound"


def test_upper_confidence_bound_returns_callable(botorch_gp_model):
    """upper_confidence_bound returns an _AcquisitionCallable."""
    acq = upper_confidence_bound(botorch_gp_model)

    assert isinstance(acq, _AcquisitionCallable)


def test_upper_confidence_bound_scores_with_botorch_model(botorch_gp_model, test_tensor_2d):
    """UCB produces finite scores for a native BoTorch GP model."""
    acq = upper_confidence_bound(botorch_gp_model, beta=2.0)

    result = acq(test_tensor_2d)

    assert isinstance(result, Predictions)
    assert result.means.shape == (test_tensor_2d.shape[0],)
    assert np.all(np.isfinite(result.means))


def test_upper_confidence_bound_scores_with_alf_model(
    mock_alf_model_with_variances, test_tensor_2d
):
    """UCB works transparently with an ALF BaseModel."""
    acq = upper_confidence_bound(mock_alf_model_with_variances, beta=2.0)

    result = acq(test_tensor_2d)

    assert isinstance(result, Predictions)
    assert result.means.shape == (test_tensor_2d.shape[0],)
    assert np.all(np.isfinite(result.means))


def test_upper_confidence_bound_higher_beta_increases_scores(botorch_gp_model, test_tensor_2d):
    """Higher beta leads to equal or higher UCB scores (more exploration)."""
    result_low = upper_confidence_bound(botorch_gp_model, beta=0.5)(test_tensor_2d)
    result_high = upper_confidence_bound(botorch_gp_model, beta=10.0)(test_tensor_2d)

    assert np.all(result_high.means >= result_low.means - 1e-6)


def test_upper_confidence_bound_default_beta_stored(botorch_gp_model):
    """Default beta=2.0 is stored in kwargs when not explicitly provided."""
    acq = upper_confidence_bound(botorch_gp_model)

    assert acq.kwargs.get("beta") == 2.0


# =============================================================================
# BotorchAcquisitionConfig validation
# =============================================================================


def test_botorch_acquisition_config_accepts_valid_target():
    """BotorchAcquisitionConfig does not raise for a valid botorch.acquisition target."""
    cfg = BotorchAcquisitionConfig({
        "_target_": "botorch.acquisition.analytic.ExpectedImprovement",
        "best_f": 0.0,
    })
    assert cfg.cfg["_target_"] == "botorch.acquisition.analytic.ExpectedImprovement"


def test_botorch_acquisition_config_rejects_non_botorch_target():
    """BotorchAcquisitionConfig raises ValueError for a target outside botorch.acquisition."""
    with pytest.raises(ValueError, match="botorch.acquisition"):
        BotorchAcquisitionConfig({"_target_": "torch.nn.Linear"})


def test_botorch_acquisition_config_rejects_missing_target():
    """BotorchAcquisitionConfig raises ValueError when _target_ is absent."""
    with pytest.raises(ValueError, match="botorch.acquisition"):
        BotorchAcquisitionConfig({"best_f": 0.0})


def test_botorch_acquisition_config_accepts_dictconfig():
    """BotorchAcquisitionConfig accepts an omegaconf DictConfig."""
    cfg_dict = OmegaConf.create({
        "_target_": "botorch.acquisition.analytic.UpperConfidenceBound",
        "beta": 2.0,
    })
    cfg = BotorchAcquisitionConfig(cfg_dict)
    assert cfg.cfg["_target_"] == "botorch.acquisition.analytic.UpperConfidenceBound"


# =============================================================================
# BotorchAcquisitionFunction — helpers
# =============================================================================


class _MockSurrogate:
    """Minimal surrogate stub that exposes a .model attribute."""

    def __init__(self, model):
        self.model = model


class _MockState:
    """Minimal state stub with surrogate and acq_batch_size."""

    def __init__(self, model):
        self.surrogate = _MockSurrogate(model)


# =============================================================================
# BotorchAcquisitionFunction tests
# =============================================================================


def test_botorch_acquisition_function_returns_labelled_candidates(
    botorch_gp_model, test_candidates_2d
):
    """BotorchAcquisitionFunction returns LabelledCandidates with one score per candidate."""
    cfg = BotorchAcquisitionConfig({
        "_target_": "botorch.acquisition.analytic.ExpectedImprovement",
        "best_f": 0.0,
    })
    acq_fn = BotorchAcquisitionFunction(cfg)
    state = _MockState(botorch_gp_model)

    result = acq_fn(test_candidates_2d, state)

    assert isinstance(result, LabelledCandidates)
    assert len(result.labels) == len(test_candidates_2d)


def test_botorch_acquisition_function_scores_are_finite(botorch_gp_model, test_candidates_2d):
    """BotorchAcquisitionFunction produces finite scores."""
    cfg = BotorchAcquisitionConfig({
        "_target_": "botorch.acquisition.analytic.ExpectedImprovement",
        "best_f": 0.0,
    })
    acq_fn = BotorchAcquisitionFunction(cfg)
    state = _MockState(botorch_gp_model)

    result = acq_fn(test_candidates_2d, state)

    assert np.all(np.isfinite(result.labels))


def test_botorch_acquisition_function_wraps_alf_model(
    mock_alf_model_with_variances, test_candidates_2d
):
    """BotorchAcquisitionFunction adapts an ALF BaseModel via BoTorchModelAdapter."""
    cfg = BotorchAcquisitionConfig({
        "_target_": "botorch.acquisition.analytic.UpperConfidenceBound",
        "beta": 2.0,
    })
    acq_fn = BotorchAcquisitionFunction(cfg)
    state = _MockState(mock_alf_model_with_variances)

    result = acq_fn(test_candidates_2d, state)

    assert len(result.labels) == len(test_candidates_2d)
    assert np.all(np.isfinite(result.labels))
