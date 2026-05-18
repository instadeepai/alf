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


from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

import numpy as np
from alf_core.dataclasses import Candidate, LabelledCandidates, Predictions

if TYPE_CHECKING:
    from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics


@dataclass
class BaseTrainConfig:
    """Base configuration shared by all model training configs.

    Args:
        learning_rate: Learning rate for the optimizer.
        log_frequency: How often (in epochs/iterations) to log training metrics.
        normalise_inputs: Whether to apply min-max normalisation to input
            features before training. Defaults to False.
        standardise_outputs: Whether to apply Z-score standardisation to
            outputs before training. Defaults to False.
        label_dtype: dtype for label tensors during training. None means each
            model uses its own default (e.g. float32 for regression, long for
            classification). Override to force a specific dtype.
    """

    learning_rate: float = 1e-3
    log_frequency: int = 10
    normalise_inputs: bool = False
    standardise_outputs: bool = False
    label_dtype: object | None = None


class BaseModel(abc.ABC):
    """Base class for all models.
    Several components of the framework can be treated as models, such as the surrogate, oracle,
    and the generator defined as model based search.
    """

    @abc.abstractmethod
    def featurise(self, inputs: list[Candidate]) -> Any:
        """Convert inputs into feature representations.

        Args:
            inputs: List of Candidate objects to featurize.

        Returns:
            Feature representation of the inputs (format depends on implementation).
        """
        pass

    @abc.abstractmethod
    def train(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates,
    ) -> None:
        """Train the model on the provided training and validation data.

        Args:
            train_data: Labeled candidates for training.
            val_data: Labeled candidates for validation.
        """
        pass

    @abc.abstractmethod
    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Predict scores for the given candidate points.

        Args:
            candidate_points: List of Candidate objects to make predictions for.

        Returns:
            Predictions object containing means and optionally variances
            and empirical distributions.
        """
        pass

    @abc.abstractmethod
    def sample(self, condition: Any | None = None) -> list[Candidate]:
        """Sample candidate points from the model.

        Args:
            condition: Optional conditioning information for sampling.

        Returns:
            List of sampled candidate points.
        """
        pass

    def get_epoch_metrics(self) -> list[SurrogateEpochMetrics]:
        """Return per-epoch training metrics from the most recent train() call.

        Returns:
            List of SurrogateEpochMetrics, one per epoch trained. Returns an empty list
            by default; subclasses that record per-epoch metrics should override.
        """
        return []

    def get_training_summary_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Get summary metrics from the most recent training run.

        Returns:
            Dictionary of metric names to values. Returns empty dict by default;
            subclasses should override to provide metrics.
        """
        return {}

    def cleanup(self) -> None:
        """Clean up temporary files and resources.

        Deletes any temporary files, checkpoints, or other resources that aren't
        being persisted. Subclasses should override to implement cleanup logic.
        """
        pass
