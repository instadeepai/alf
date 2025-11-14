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

import matplotlib.pyplot as plt
import numpy as np

from alf.core.dataclasses.predictions import Predictions
from alf.core.utils.metrics import metric_registry
from alf.core.utils.plots import plot_registry


@dataclass
class Results:
    """Computes metrics and figures based on the predictions and targets.

    Attributes:
        targets: A numpy array of ground truth target values.
        predictions: A Predictions object containing model predictions.
    """

    targets: np.ndarray
    predictions: Predictions

    def __post_init__(self) -> None:
        """Validate inputs and compute metrics and figures."""
        assert len(self.targets) == len(self.predictions.means), (
            "Targets and predictions must have the same length"
        )
        self.metrics = self.compute_metrics()
        self.figures = self.compute_figures()

    def compute_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Compute evaluation metrics based on predictions and targets.

        Returns:
            dict[str, Union[float, int, np.number]]: A dictionary of metric names
                to their computed values. The metrics depend on whether variances
                are available.
        """
        metrics: dict[str, Union[float, int, np.number]] = {}
        metrics_dict = (
            metric_registry.get_metrics_not_requiring_variance()
            if self.predictions.variances is None
            else metric_registry.get_metrics_requiring_variance()
        )
        for _, metric_fn in metrics_dict.items():
            metrics.update(
                metric_fn(self.predictions.means, self.predictions.variances, self.targets)
            )

        return metrics

    def compute_figures(self) -> dict[str, plt.Figure]:
        """Compute visualization figures based on predictions and targets.

        Returns:
            dict[str, plt.Figure]: A dictionary of figure names to matplotlib
                Figure objects. The figures depend on whether variances are available.
        """
        plots: dict[str, plt.Figure] = {}
        plots_dict = (
            plot_registry.get_plots_not_requiring_variance()
            if self.predictions.variances is None
            else plot_registry.get_plots_requiring_variance()
        )
        for _, plot_fn in plots_dict.items():
            plots.update(plot_fn(self.predictions.means, self.predictions.variances, self.targets))

        return plots
