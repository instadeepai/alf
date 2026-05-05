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


from dataclasses import dataclass
from typing import Union

import numpy as np
from alf_core.dataclasses.predictions import Predictions
from alf_core.enums import ProblemType
from alf_core.utils.metrics import classification_metric_registry, metric_registry


@dataclass
class Results:
    """Computes metrics based on the predictions and targets.

    Attributes:
        targets: A numpy array of ground truth target values.
        predictions: A Predictions object containing model predictions.
        problem_type: Type of problem determining which metrics are computed.
    """

    targets: np.ndarray
    predictions: Predictions
    problem_type: ProblemType

    def __post_init__(self) -> None:
        """Validate inputs and compute metrics.

        Raises:
            AssertionError: If targets and predictions have different lengths.
        """
        assert self.targets.shape[0] == self.predictions.means.shape[0], (
            f"Targets and predictions must have the same length (number of samples), "
            f"got targets.shape={self.targets.shape} and predictions.means.shape={self.predictions.means.shape}"
        )
        self.metrics = self.compute_metrics()

    def compute_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Compute evaluation metrics based on predictions and targets.

        For regression, routes to the variance-aware regression registry.
        For classification (binary or multiclass), routes to the classification
        metric registry using the probability array stored in ``predictions.means``.

        Returns:
            A dictionary of metric names to their computed values.
        """
        metrics: dict[str, Union[float, int, np.number]] = {}

        if self.problem_type == ProblemType.REGRESSION:
            metrics_dict = (
                metric_registry.get_metrics_not_requiring_variance()
                if self.predictions.variances is None
                else metric_registry.get_metrics_requiring_variance()
            )
            for _, metric_fn in metrics_dict.items():
                metrics.update(
                    metric_fn(self.predictions.means, self.predictions.variances, self.targets)
                )
        else:
            for _, metric_fn in classification_metric_registry.metrics.items():
                metrics.update(metric_fn(self.predictions.means, self.targets))

        return metrics
