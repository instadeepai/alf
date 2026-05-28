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


from typing import Any, Union

import torch

import numpy as np
from alf_core.dataclasses import Candidate, LabelledCandidates, Predictions
from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics
from alf_core.dataset.base_dataset import BaseDataset
from alf_core.model.base_model import BaseModel


class Surrogate:
    """Surrogate model is fine-tuned during the active learning process on the acquired candidates.
    Any BaseModel child class can be used as a surrogate model.
    """

    def __init__(self, model: BaseModel):
        """Initialize the Surrogate with a model.

        Args:
            model: The BaseModel instance to use as the surrogate model.
        """
        self.model = model

    def setup(self, dataset: BaseDataset) -> None:
        """Configure the surrogate model for the given dataset.

        Args:
            dataset: The dataset this surrogate will be trained on.
        """
        self.model.setup(dataset)

    def fit(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates,
    ) -> list[SurrogateEpochMetrics]:
        """Fit the surrogate model on training and validation data.

        Args:
            train_data: Labeled candidates for training.
            val_data: Labeled candidates for validation.

        Returns:
            List of SurrogateEpochMetrics, one per epoch trained. Empty if the
            underlying model does not track per-epoch metrics.
        """
        self.model.train(train_data, val_data)
        return self.model.get_epoch_metrics()

    def predict(self, candidates: list[Candidate]) -> Predictions:
        """Predict scores for the given candidates.

        Args:
            candidates: List of Candidate objects to make predictions for.

        Returns:
            Predictions object containing means and optionally variances
            and empirical distributions.
        """
        return self.model.predict(candidates)

    def featurise(self, inputs: LabelledCandidates | list[Candidate]) -> np.ndarray | torch.Tensor:
        """Featurise the given inputs using the surrogate model's featurisation method.

        Args:
            inputs: List of Candidate objects or a LabelledCandidates instance
                to featurise.

        Returns:
            Feature representation from the underlying model, typically
            an np.ndarray or torch.Tensor.
        """
        return self.model.featurise(inputs)

    def get_training_summary_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Get summary metrics from the most recent training run.

        Returns:
            Dictionary of metric names to values from the underlying model's training.
        """
        return self.model.get_training_summary_metrics()
