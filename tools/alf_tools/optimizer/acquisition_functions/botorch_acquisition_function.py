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

Provides :data:`ACQUISITION_REGISTRY` mapping names to BoTorch acquisition
function classes and :class:`BotorchAcquisitionFunction`.  The class accepts
either a native BoTorch ``Model`` or an ALF ``BaseModel`` — a
:class:`~alf_tools.optimizer.acquisition_functions.utils.botorch_model_adapter.BoTorchModelAdapter`
is inserted automatically when needed.

Usage::
    cfg = BotorchAcquisitionConfig(name="expected_improvement", kwargs={"best_f": 0.5})
    acq_fn = BotorchAcquisitionFunction(cfg)
    labelled = acq_fn(candidates, state)
"""

import inspect
from dataclasses import dataclass, field
from typing import Any

import torch
from alf_core import AcquisitionFunction as AlfAcquisitionFunction
from alf_core import Candidate, LabelledCandidates, State
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

ACQUISITION_REGISTRY: dict[str, type[AcquisitionFunction]] = {
    "expected_improvement": ExpectedImprovement,
    "upper_confidence_bound": UpperConfidenceBound,
    "probability_of_improvement": ProbabilityOfImprovement,
    "log_noisy_expected_improvement": qLogNoisyExpectedImprovement,
}
"""Maps acquisition function names to their BoTorch acquisition classes.

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
        kwargs: Keyword arguments forwarded to the acquisition constructor
            (everything except ``model``).

    Raises:
        ValueError: If ``name`` is not a key in :data:`ACQUISITION_REGISTRY`.
        ValueError: If any required kwargs for the named acquisition class are missing.
    """

    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate name and required kwargs.

        Raises:
            ValueError: If ``name`` is not found in :data:`ACQUISITION_REGISTRY`.
            ValueError: If required constructor kwargs for the named class are missing.
        """
        if self.name not in ACQUISITION_REGISTRY:
            raise ValueError(
                f"Unknown acquisition function: {self.name!r}. "
                f"Must be one of {sorted(ACQUISITION_REGISTRY)}"
            )
        acq_cls = ACQUISITION_REGISTRY[self.name]
        sig = inspect.signature(acq_cls.__init__)
        missing = [
            param_name
            for param_name, param in sig.parameters.items()
            if param_name not in ("self", "model")
            and param.default is inspect.Parameter.empty
            and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
            and param_name not in self.kwargs
        ]
        if missing:
            raise ValueError(
                f"Missing required kwargs for {self.name!r}: {missing}. "
                f"Provided: {sorted(self.kwargs)}"
            )


class BotorchAcquisitionFunction(AlfAcquisitionFunction):
    """ALF :class:`~alf_core.AcquisitionFunction` backed by a registered BoTorch acquisition.

    The acquisition function is looked up by name in :data:`ACQUISITION_REGISTRY`
    and instantiated on each call with ``model`` injected from ``state.surrogate.model``.
    If the surrogate model is an ALF ``BaseModel`` it is adapted via
    :class:`~alf_tools.optimizer.acquisition_functions.utils.botorch_model_adapter.BoTorchModelAdapter`
    before being passed to the BoTorch acquisition.

    Args:
        cfg: Config specifying the acquisition function name and its keyword
            arguments.  ``model`` must *not* appear in ``cfg.kwargs`` — it is
            always injected from ``state.surrogate.model`` at call time.
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
            state: Task state; ``state.surrogate.model`` is used as the model.

        Returns:
            :class:`~alf_core.LabelledCandidates` with acquisition scores as labels.
        """
        model = state.surrogate.model
        device = getattr(model, "device", None)
        X = candidates_to_tensor(search_candidates, device=device)

        adapted = model if isinstance(model, BotorchModel) else BoTorchModelAdapter(model)
        acq_cls = ACQUISITION_REGISTRY[self._cfg.name]
        botorch_acq = acq_cls(model=adapted, **self._cfg.kwargs)

        X_batched = X.unsqueeze(1) if X.dim() == 2 else X
        with torch.no_grad():
            scores = botorch_acq(X_batched)

        return LabelledCandidates(candidates=search_candidates, labels=scores.cpu().numpy())
