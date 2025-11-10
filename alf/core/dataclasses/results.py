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
from typing import Dict, Union

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
        predictions: A Predictions object containing model predictions (optional).
        means: A numpy array of mean predictions (optional).
        variances: A numpy array of prediction variances (optional).
        metrics: A dictionary of computed metrics (computed in __post_init__).
        figures: A dictionary of matplotlib figures (computed in __post_init__).

    Two instantiation options:
    1. Results(predictions=my_predictions, targets=my_targets)
    2. Results(means=my_means, variances=my_variances, targets=my_targets)

    Either predictions OR (means, variances) must be set, not both.
    """

    targets: np.ndarray
    predictions: Predictions | None = None
    means: np.ndarray | None = None
    variances: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate inputs and compute metrics and figures.

        Ensures exactly one of the two instantiation options is used, then
        computes metrics and figures based on the predictions and targets.

        Raises:
            AssertionError: If both predictions and (means, variances) are set,
                or if neither is set.
        """
        # Ensure exactly one of the two instantiation options is used
        has_predictions = self.predictions is not None
        has_means = self.means is not None
        assert has_predictions ^ has_means, (
            "Either predictions OR (means, variances) must be set, not both"
        )

        # If predictions are not set, create predictions object
        if not has_predictions:
            self.predictions = Predictions(means=self.means, variances=self.variances)

        self.metrics = self.compute_metrics()
        self.figures = self.compute_figures()

    def compute_metrics(self) -> Dict[str, Union[float, int, np.number]]:
        """Compute evaluation metrics based on predictions and targets.

        Returns:
            Dict[str, Union[float, int, np.number]]: A dictionary of metric names
                to their computed values. The metrics depend on whether variances
                are available.

        Raises:
            ValueError: If predictions is not set.
        """
        if self.predictions is None:
            raise ValueError("Predictions must be set")

        metrics: Dict[str, Union[float, int, np.number]] = {}
        metrics_dict = (
            metric_registry.get_metrics_not_requiring_variance()
            if self.variances is None
            else metric_registry.get_metrics_requiring_variance()
        )
        for _, metric_fn in metrics_dict.items():
            metrics.update(
                metric_fn(self.predictions.means, self.predictions.variances, self.targets)
            )

        return metrics

    def compute_figures(self) -> Dict[str, plt.Figure]:
        """Compute visualization figures based on predictions and targets.

        Returns:
            Dict[str, plt.Figure]: A dictionary of figure names to matplotlib
                Figure objects. The figures depend on whether variances are available.

        Raises:
            ValueError: If predictions is not set.
        """
        if self.predictions is None:
            raise ValueError("Predictions must be set")

        plots: Dict[str, plt.Figure] = {}
        plots_dict = (
            plot_registry.get_plots_not_requiring_variance()
            if self.variances is None
            else plot_registry.get_plots_requiring_variance()
        )
        for _, plot_fn in plots_dict.items():
            plots.update(plot_fn(self.predictions.means, self.predictions.variances, self.targets))

        return plots
