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

from functools import wraps
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, pearsonr, spearmanr

from alf.core.utils.metrics import check_inputs, check_variance_validity

DEFAULT_FIGURE_SIZE = (6, 6)


class PlotRegistry:
    """Simple registry for plots with variance requirements."""

    def __init__(self):
        self.plots: dict[str, Callable] = {}
        self.variance_required: dict[str, bool] = {}

    def register(self, name: str, plot_fn: Callable, requires_variance: bool = False):
        """Register a plot function."""
        self.plots[name] = plot_fn
        self.variance_required[name] = requires_variance

    def get_plots_requiring_variance(self) -> dict[str, Callable]:
        """Get plots that require variance."""
        return {
            name: fn for name, fn in self.plots.items() if self.variance_required[name]
        }

    def get_plots_not_requiring_variance(self) -> dict[str, Callable]:
        """Get plots that don't require variance."""
        return {
            name: fn
            for name, fn in self.plots.items()
            if not self.variance_required[name]
        }


# Create the global registry instance
plot_registry = PlotRegistry()


def requires_variance(plot_fn: Callable) -> Callable:
    """Decorator to mark a plot as requiring variance.
    This automatically registers the plot in the global registry and applies input validation.
    """

    @wraps(plot_fn)
    def wrapper(
        means: np.ndarray,
        variances: np.ndarray | None,
        targets: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        check_inputs(means, targets)
        check_variance_validity(variances, targets)
        return plot_fn(means, variances, targets, *args, **kwargs)

    # Register the plot with variance requirement
    plot_registry.register(plot_fn.__name__, wrapper, requires_variance=True)
    return wrapper


def no_variance_required(plot_fn: Callable) -> Callable:
    """Decorator to mark a plot as not requiring variance.
    This automatically registers the plot in the global registry and applies input validation.
    """

    @wraps(plot_fn)
    def wrapper(
        means: np.ndarray,
        variances: np.ndarray | None,
        targets: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        check_inputs(means, targets)
        return plot_fn(means, variances, targets, *args, **kwargs)

    # Register the plot without variance requirement
    plot_registry.register(plot_fn.__name__, wrapper, requires_variance=False)
    return wrapper


@requires_variance
def create_ece_plot(
    means: np.ndarray,
    variances: np.ndarray,
    targets: np.ndarray,
    n_grid_points: int = 100,
) -> dict[str, plt.Figure]:
    """Create ECE plot
    -- requires uncertainty --
    For alpha in grid from 0 -> 1
    Find alpha% confidence intervals for all predictions
    Count % of targets which fall within the confidence intervals
    ECE = Compute area between x=y and the curve
    (We want to minimize this metric)
    Plot the counts for different values of alpha
    with comparison to perfect calibration x=y

    Args:
        means: Array of shape (b,). Predicted means
        variances: Array of shape (b,). Predicted variances
        targets: Array of shape (b,). True labels
        n_grid_points: Integer for the coarseness of the grid

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    grid = np.linspace(0, 1, n_grid_points)
    perc = np.zeros(n_grid_points)
    for i, cdf_cutoff in enumerate(grid):
        num_stds = norm.ppf(1 - ((1 - cdf_cutoff) / 2))
        perc[i] = (
            (targets >= means - num_stds * np.sqrt(variances))
            & (targets <= means + num_stds * np.sqrt(variances))
        ).mean()

    # get ece scalar
    ece = np.sum(np.abs(perc - grid)) * (1 / n_grid_points)

    # create figure
    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
    ax.scatter(grid, perc, label="Observed coverage")
    ax.plot([0, 1], [0, 1], linestyle="--", color="red", label="Perfect calibration")
    ax.set_xlabel("Confidence Level")
    ax.set_ylabel("Observed coverage")
    ax.set_title(f"ECE Plot (ECE={ece:.3f})")

    ticks = np.arange(0, 1, 0.2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    # stop automatic plotting of the diagram
    plt.close(fig)
    return {"ece": fig}


@no_variance_required
def create_predictions_plot(
    means: np.ndarray,
    variances: np.ndarray | None,
    targets: np.ndarray,
    confidence_level: float = 0.95,
) -> dict[str, plt.Figure]:
    """Create predictions plot
    -- optionally uses uncertainty --
    Creates a simple scatter plot of predictions against targets
    (Optional: if one has variances then also add confidence_level% confidence intervals)


    Args:
        means: Array of shape (b,). Predicted means
        variances: Array of shape (b,). Predicted variances, optionally None
        targets: Array of shape (b,). True labels
        confidence_level: float.

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if variances is not None:
        has_variance = True
    else:
        has_variance = False

    # compute mse
    mse_val = np.mean((means - targets) ** 2)
    spearman_val = spearmanr(means, targets)[0]
    pearson_val = pearsonr(means, targets)[0]

    # create figure
    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)

    y_min = min(means.min(), targets.min())
    y_max = max(means.max(), targets.max())

    margin = 0.05 * (y_max - y_min)
    y_min -= margin
    y_max += margin

    ax.set_xlim(y_min, y_max)
    ax.set_ylim(y_min, y_max)

    if has_variance:
        # compute z-score for CI
        z = norm.ppf(1 - (1 - confidence_level) / 2)
        yerr = z * np.sqrt(variances)  # type: ignore[arg-type]
        ax.errorbar(
            means,
            targets,
            yerr=yerr,
            fmt="o",
            alpha=0.7,
            ecolor="C0",
            capsize=2,
            label=f"{int(confidence_level * 100)}% CI",
        )
    else:
        ax.scatter(means, targets, alpha=0.7, color="C0", label="Predictions")

    # perfect prediction
    ax.plot(
        [y_min, y_max],
        [y_min, y_max],
        linestyle="--",
        color="red",
        label="Perfect prediction",
    )

    ax.set_xlabel("Predicted mean")
    ax.set_ylabel("Target")
    ax.set_title(
        f"Predictions vs Targets \n (MSE={mse_val:.2f}, "
        f"pearson={pearson_val:.2f}, spearman={spearman_val:.2f})"
    )

    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    # stop automatic plotting of the diagram
    plt.close(fig)
    return {"predictions_scatter": fig}


@no_variance_required
def create_ranks_plot(
    means: np.ndarray,
    _: np.ndarray | None,
    targets: np.ndarray,
) -> dict[str, plt.Figure]:
    """Create predicted ranks plot
    Creates a simple scatter plot of predicted rank against true rank

    Args:
        means: Array of shape (b,). Predicted means
        _: Array of shape (b,), optionally None. Unused.
        targets: Array of shape (b,). True labels

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # compute mse
    mse_val = np.mean((means - targets) ** 2)
    spearman_val = spearmanr(means, targets)[0]
    pearson_val = pearsonr(means, targets)[0]

    pred_ranks = means.argsort()[::-1].argsort()
    true_ranks = targets.argsort()[::-1].argsort() + 1

    # create figure
    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)

    y_min = 0
    y_max = len(means)

    ax.scatter(pred_ranks, true_ranks, alpha=0.7, color="C0", label="Predictions")

    # perfect prediction
    ax.plot(
        [y_min, y_max],
        [y_min, y_max],
        linestyle="--",
        color="red",
        label="Perfect ranks",
    )

    ax.set_xlabel("Predicted mean")
    ax.set_ylabel("Target")
    ax.set_title(
        f"Predicted ranks vs Targets \n (MSE={mse_val:.2f}, "
        f"pearson={pearson_val:.2f}, spearman={spearman_val:.2f})"
    )

    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    # stop automatic plotting of the diagram
    plt.close(fig)
    return {"ranks_scatter": fig}


@requires_variance
def create_variance_histogram(
    _: np.ndarray,
    variances: np.ndarray,
    targets: np.ndarray,
) -> dict[str, plt.Figure]:
    """Create a histogram of predicted variances.

    Args:
        _: unused
        variances: Array of shape (b,). Predicted variances
        targets: Array of shape (b,). True labels

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # create figure
    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
    ax.hist(variances, bins=30, color="C0", density=True, alpha=0.7)
    ax.set_xlabel("Predicted variance")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of predicted variances")
    ax.grid(True)
    fig.tight_layout()

    # stop automatic plotting of the diagram
    plt.close(fig)
    return {"variance_histogram": fig}
