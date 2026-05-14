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


from typing import Union

import numpy as np
from alf_core.dataclasses import Candidate, LabelledCandidates, Predictions
from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics
from alf_core.model.base_model import BaseModel
from alf_core.normalizer.normalizer import IdentityNormalizer, Normalizer


class Surrogate:
    """Surrogate model fine-tuned during the active learning process.

    Wraps a BaseModel and an optional Normalizer. Labels are normalised
    before training and predictions are inverse-transformed before returning.
    """

    def __init__(self, model: BaseModel, normalizer: Normalizer | None = None):
        """Initialize the Surrogate.

        Args:
            model: The BaseModel instance to use as the surrogate model.
            normalizer: Optional normalizer for target labels. Defaults to
                IdentityNormalizer (no-op).
        """
        self.model = model
        self.normalizer: Normalizer = normalizer if normalizer is not None else IdentityNormalizer()

    def _normalise(self, data: LabelledCandidates) -> LabelledCandidates:
        """Return a new LabelledCandidates with transformed labels.

        Does not mutate the input.

        Args:
            data: Original labelled candidates.

        Returns:
            New LabelledCandidates with normalised labels.
        """
        return LabelledCandidates(
            candidates=data.candidates,
            labels=self.normalizer.transform(data.labels),
        )

    def fit(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates,
    ) -> list[SurrogateEpochMetrics]:
        """Fit the surrogate model on training and validation data.

        Fits the normalizer on training labels, then trains the model on
        normalised data.

        Args:
            train_data: Labeled candidates for training.
            val_data: Labeled candidates for validation.

        Returns:
            List of SurrogateEpochMetrics, one per epoch trained.
        """
        self.normalizer.fit(train_data.labels)
        self.model.train(self._normalise(train_data), self._normalise(val_data))
        return self.model.get_epoch_metrics()

    def predict(self, candidates: list[Candidate]) -> Predictions:
        """Predict scores for the given candidates.

        Predictions from the model are inverse-transformed back to the
        original label space before returning.

        Args:
            candidates: List of Candidate objects to predict for.

        Returns:
            Predictions in the original (un-normalised) label space.
        """
        raw = self.model.predict(candidates)
        means = self.normalizer.inverse_transform(raw.means)
        variances = (
            self.normalizer.inverse_transform_variance(raw.variances)
            if raw.variances is not None
            else None
        )
        return Predictions(means=means, variances=variances)

    def get_training_summary_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Get summary metrics from the most recent training run.

        Returns:
            Dictionary of metric names to values from the underlying model.
        """
        return self.model.get_training_summary_metrics()
