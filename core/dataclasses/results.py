from dataclasses import dataclass, field
import numpy as np
from core.dataclasses.predictions import Predictions
from core.utils.metrics import metric_registry
from core.utils.plots import plot_registry
from typing import Dict, Union
import matplotlib.pyplot as plt


@dataclass
class Results:
    """Computes metrics and figures based on the predictions and targets.
    
    Attributes:
        predictions: A Predictions object containing model predictions.
        means: A numpy array of mean predictions.
        variances: A numpy array of prediction variances (optional).
        targets: A numpy array of targets.

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
        """Post init is used to check the means and predictions, and compute the metrics and figures."""
        # Ensure exactly one of the two instantiation options is used
        has_predictions = self.predictions is not None
        has_means = self.means is not None
        assert has_predictions ^ has_means, "Either predictions OR (means, variances) must be set, not both"
        
        # If means/variances were provided, create predictions object
        if has_means:
            self.predictions = Predictions(means=self.means, variances=self.variances)

        self.metrics = self.compute_metrics()
        self.figures = self.compute_figures()

    def compute_metrics(self) -> Dict[str, Union[float, int, np.number]]:
        """Compute the metrics."""

        metrics: Dict[str, Union[float, int, np.number]] = {}
        metrics_dict = metric_registry.get_metrics_not_requiring_variance() if self.variances is None else metric_registry.get_metrics_requiring_variance()
        for _, metric_fn in metrics_dict.items():
            metrics.update(metric_fn(self.predictions.means, self.predictions.variances, self.targets))
        
        return metrics

    def compute_figures(self) -> None:
        """Compute the figures."""

        plots: Dict[str, plt.Figure] = {}
        plots_dict = plot_registry.get_plots_not_requiring_variance() if self.variances is None else plot_registry.get_plots_requiring_variance()
        for _, plot_fn in plots_dict.items():
            plots.update(plot_fn(self.predictions.means, self.predictions.variances, self.targets))

        return plots