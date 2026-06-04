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

"""BoTorch acquisition functions adapted for ALF and native BoTorch models.

Provides the :func:`acquisition` decorator, four ready-to-use factories
(:func:`expected_improvement`, :func:`upper_confidence_bound`,
:func:`probability_of_improvement`, :func:`log_noisy_expected_improvement`),
the :data:`ACQUISITION_REGISTRY` mapping names to those factories, and the
:class:`BotorchAcquisitionFunction` class.  All factories accept either a native
BoTorch ``Model`` or an ALF ``BaseModel`` — the decorator inserts a
:class:`~alf_tools.optimizer.acquisition_functions.utils.botorch_model_adapter.BoTorchModelAdapter`
automatically when needed.

Usage::
    # Config-driven usage
    cfg = BotorchAcquisitionConfig(name="expected_improvement", kwargs={"best_f": 0.5})
    acq_fn = BotorchAcquisitionFunction(cfg)
    labelled = acq_fn(candidates, state)

    # Low-level functional usage
    acq = expected_improvement(surrogate_model, best_f=0.5)
    scores = acq(candidates_tensor)   # returns Predictions

    acq = upper_confidence_bound(surrogate_model, beta=2.0)
    acq = probability_of_improvement(surrogate_model, best_f=0.5)
    acq = log_noisy_expected_improvement(surrogate_model, X_baseline=X_train)
"""

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from alf_core import AcquisitionFunction as AlfAcquisitionFunction
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions, State
from alf_tools.optimizer.acquisition_functions.utils.botorch_model_adapter import (
    BoTorchModelAdapter,
)
from alf_tools.utils.botorch_utils import candidates_to_tensor
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.models.model import Model as BotorchModel


class _AcquisitionCallable:
    """Wraps a BoTorch acquisition function with a tensor-in, Predictions-out interface.

    Not intended to be instantiated directly — use the decorated acquisition
    factories such as :func:`expected_improvement` and
    :func:`upper_confidence_bound`.

    Args:
        acq_fn: BoTorch acquisition function to evaluate.
        name: Name of the acquisition factory (e.g. `"expected_improvement"`).
        kwargs: Parameters passed to the factory, excluding the model.
            Note: some factories (e.g. `log_noisy_expected_improvement`) accept
            tensors, which are not JSON/YAML-serializable.
    """

    def __init__(self, acq_fn: AcquisitionFunction, name: str, kwargs: dict[str, Any]):
        self._acq = acq_fn
        self.name = name
        self.kwargs = kwargs

    def __call__(self, candidates: torch.Tensor) -> Predictions:
        """Evaluate acquisition scores on candidate points.

        Args:
            candidates: Tensor of shape `(n, d)`.

        Returns:
            :class:`~alf_core.Predictions` with acquisition scores as `means`.
        """
        X = candidates.unsqueeze(1) if candidates.dim() == 2 else candidates  # (n, 1, d) — BoTorch analytic fns expect q-batch dim
        with torch.no_grad():
            scores = self._acq(X)
        return Predictions(means=scores.cpu().numpy())


def acquisition(fn):
    """Decorator that makes a BoTorch acquisition factory model-agnostic.

    The decorated function receives a `botorch.models.model.Model` regardless
    of whether the caller passed a native BoTorch model or an ALF `BaseModel`;
    the decorator inserts a :class:`BoTorchModelAdapter` when needed.

    The returned :class:`_AcquisitionCallable` exposes `name` and `kwargs`
    for config serialisation.

    Example::

        @acquisition
        def expected_improvement(model, best_f: float, maximize: bool = True):
            return ExpectedImprovement(model=model, best_f=best_f, maximize=maximize)

        acq = expected_improvement(my_model, best_f=0.5)
        acq.name     # "expected_improvement"
        acq.kwargs   # {"best_f": 0.5, "maximize": True}

    Returns:
        Decorated function that accepts any model type and returns an
        :class:`_AcquisitionCallable`.
    """

    @functools.wraps(fn)
    def wrapper(model: BaseModel | BotorchModel, *args, **kwargs):
        adapted = model if isinstance(model, BotorchModel) else BoTorchModelAdapter(model)
        botorch_acq = fn(adapted, *args, **kwargs)

        # Bind positional and keyword args to parameter names for kwargs storage.
        # Pass `model` (not `adapted`) so inspect.signature sees the original value.
        sig = inspect.signature(fn)
        first_param = next(iter(sig.parameters))
        bound = sig.bind(model, *args, **kwargs)
        bound.apply_defaults()
        captured_kwargs = {k: v for k, v in bound.arguments.items() if k != first_param}

        return _AcquisitionCallable(botorch_acq, name=fn.__name__, kwargs=captured_kwargs)

    return wrapper


@acquisition
def expected_improvement(
    model: BotorchModel, best_f: float, maximize: bool = True
) -> AcquisitionFunction:
    """Expected Improvement acquisition function.

    Args:
        model: ALF `BaseModel` or native BoTorch `Model`.
        best_f: Best observed function value so far.
        maximize: If `True` (default), optimise for the maximum.

    Returns:
        :class:`_AcquisitionCallable` wrapping BoTorch `ExpectedImprovement`.
    """
    return ExpectedImprovement(model=model, best_f=best_f, maximize=maximize)


@acquisition
def upper_confidence_bound(model: BotorchModel, beta: float = 2.0) -> AcquisitionFunction:
    """Upper Confidence Bound acquisition function.

    Args:
        model: ALF `BaseModel` or native BoTorch `Model`.
        beta: Exploration-exploitation trade-off (default: `2.0`).  Higher
            values favour exploration.

    Returns:
        :class:`_AcquisitionCallable` wrapping BoTorch `UpperConfidenceBound`.
    """
    return UpperConfidenceBound(model=model, beta=beta)


@acquisition
def probability_of_improvement(model: BotorchModel, best_f: float, maximize: bool = True):
    """Probability of Improvement over best_f.

    Args:
        model: ALF `BaseModel` or native BoTorch `Model`.
        best_f: Best observed function value so far.
        maximize: If `True` (default), optimise for the maximum.

    Returns:
        :class:`_AcquisitionCallable` wrapping BoTorch `ProbabilityOfImprovement`.
    """
    return ProbabilityOfImprovement(model=model, best_f=best_f, maximize=maximize)


@acquisition
def log_noisy_expected_improvement(
    model: BotorchModel,
    X_baseline: torch.Tensor,
    prune_baseline: bool = True,
):
    """Log q-Noisy Expected Improvement; robust to observation noise.

    Args:
        model: ALF `BaseModel` or native BoTorch `Model`.
        X_baseline: Baseline points for noisy improvement estimation.
        prune_baseline: If `True` (default), prune the baseline.

    Returns:
        :class:`_AcquisitionCallable` wrapping BoTorch
        `qLogNoisyExpectedImprovement`.
    """
    return qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=X_baseline,
        prune_baseline=prune_baseline,
    )


ACQUISITION_REGISTRY: dict[str, Callable[..., _AcquisitionCallable]] = {
    "expected_improvement": expected_improvement,
    "upper_confidence_bound": upper_confidence_bound,
    "probability_of_improvement": probability_of_improvement,
    "log_noisy_expected_improvement": log_noisy_expected_improvement,
}
"""Maps acquisition function names to their factory callables.

Keys correspond to the `name` field of :class:`BotorchAcquisitionConfig`.
"""


@dataclass
class BotorchAcquisitionConfig:
    """Config for a registered ALF BoTorch acquisition function.

    Args:
        name: Name of the acquisition function.  Must be a key in
            :data:`ACQUISITION_REGISTRY` (one of ``"expected_improvement"``,
            ``"upper_confidence_bound"``, ``"probability_of_improvement"``,
            ``"log_noisy_expected_improvement"``).
        kwargs: Keyword arguments forwarded to the acquisition factory
            (everything except `model`).

    Raises:
        ValueError: If `name` is not a key in :data:`ACQUISITION_REGISTRY`.
    """

    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that `name` refers to a registered acquisition function.

        Raises:
            ValueError: If `name` is not found in :data:`ACQUISITION_REGISTRY`.
        """
        if self.name not in ACQUISITION_REGISTRY:
            raise ValueError(
                f"Unknown acquisition function: {self.name!r}. "
                f"Must be one of {sorted(ACQUISITION_REGISTRY)}"
            )


class BotorchAcquisitionFunction(AlfAcquisitionFunction):
    """ALF :class:`~alf_core.AcquisitionFunction` backed by a registered BoTorch acquisition.

    The acquisition function is looked up by name in :data:`ACQUISITION_REGISTRY`
    and instantiated on each call with `model` injected from `state.surrogate.model`.
    If the surrogate model is an ALF `BaseModel` it is adapted via
    :class:`~alf_tools.optimizer.acquisition_functions.utils.botorch_model_adapter.BoTorchModelAdapter`
    before being passed to the BoTorch acquisition.

    Args:
        cfg: Config specifying the acquisition function name and its keyword
            arguments.  `model` must *not* appear in `cfg.kwargs` — it is
            always injected from `state.surrogate.model` at call time.
    """

    def __init__(self, cfg: BotorchAcquisitionConfig) -> None:
        """Initialise with a validated BoTorch acquisition config.

        Args:
            cfg: Validated config for the BoTorch acquisition function.
        """
        self._cfg = cfg

    def __call__(
        self,
        search_candidates: list[Candidate],
        state: State,
    ) -> LabelledCandidates:
        """Compute acquisition scores for candidate points.

        Args:
            search_candidates: Unlabelled candidates to score.
            state: Task state; `state.surrogate.model` is used as the model.

        Returns:
            :class:`~alf_core.LabelledCandidates` with acquisition scores as labels.
        """
        model = state.surrogate.model
        device = getattr(model, "device", None)
        X = candidates_to_tensor(search_candidates, device=device)

        acq_fn = ACQUISITION_REGISTRY[self._cfg.name](model, **self._cfg.kwargs)
        predictions = acq_fn(X)

        return LabelledCandidates(candidates=search_candidates, labels=predictions.means)
