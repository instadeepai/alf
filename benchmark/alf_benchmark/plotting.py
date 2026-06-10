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

"""Active-learning curve and leaderboard plots.

The mission-aligned headline figure is performance *as a function of acquisition
round* (best-found, regret, recall, calibration) with confidence bands across
seeds. Curves guard missing metrics: a metric only appears for the methods that
produced it (e.g. calibration needs an uncertainty model; regret/recall need a
``DatasetSearch`` method).
"""

import logging

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from alf_benchmark.results import BenchmarkResults

logger = logging.getLogger("alf-benchmark")


def _resolve_problem(results: BenchmarkResults, problem: str | None) -> str:
    """Resolve which problem to plot.

    Args:
        results: The results to inspect.
        problem: Explicit problem id, or None to use the sole problem present.

    Returns:
        The resolved problem id.

    Raises:
        ValueError: If results are empty, or ``problem`` is None but several exist.
    """
    problems = sorted(results.long["problem"].unique()) if not results.long.empty else []
    if not problems:
        raise ValueError("No results to plot.")
    if problem is not None:
        if problem not in problems:
            raise ValueError(f"Unknown problem '{problem}'. Available: {', '.join(problems)}")
        return problem
    if len(problems) > 1:
        raise ValueError(f"Multiple problems present; specify one of: {', '.join(problems)}")
    return problems[0]


def plot_curves(
    results: BenchmarkResults,
    metric: str,
    problem: str | None = None,
    ci: bool = True,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot a metric versus acquisition round, one line per method, with CI bands.

    Args:
        results: Results to plot.
        metric: Metric column to plot (e.g. ``"derived/best_found"``).
        problem: Problem id; optional when only one problem is present.
        ci: Whether to shade the bootstrap confidence interval.
        ax: Existing axes to draw on; a new figure is created if omitted.

    Returns:
        The matplotlib figure.

    Raises:
        ValueError: If the metric is absent for the resolved problem.
    """
    problem = _resolve_problem(results, problem)
    aggregated = results.aggregate()
    data = aggregated[(aggregated["problem"] == problem) & (aggregated["metric"] == metric)]
    if data.empty:
        available = sorted(aggregated[aggregated["problem"] == problem]["metric"].unique())
        raise ValueError(
            f"Metric '{metric}' not found for problem '{problem}'. "
            f"Available: {', '.join(available)}"
        )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure  # type: ignore[assignment]  # SubFigure not expected here

    for method in sorted(data["method"].unique()):
        method_data = data[data["method"] == method].sort_values("round")
        line = ax.plot(method_data["round"], method_data["mean"], marker="o", label=method)[0]
        if ci:
            ax.fill_between(
                method_data["round"],
                method_data["ci_low"],
                method_data["ci_high"],
                alpha=0.2,
                color=line.get_color(),
            )

    ax.set_xlabel("acquisition round")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs round — {problem}")
    ax.legend()
    return fig


def plot_leaderboard(
    results: BenchmarkResults,
    metric: str | None = None,
    problem: str | None = None,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot final-round method means with confidence-interval error bars.

    Args:
        results: Results to plot.
        metric: Metric to rank on; defaults to the problem's primary metric.
        problem: Problem id; optional when only one problem is present.
        ax: Existing axes to draw on; a new figure is created if omitted.

    Returns:
        The matplotlib figure.

    Raises:
        ValueError: If there is nothing to plot for the resolved problem.
    """
    problem = _resolve_problem(results, problem)
    board = results.leaderboard(metric=metric)
    board = board[board["problem"] == problem].sort_values("rank")
    if board.empty:
        raise ValueError(f"No leaderboard entries for problem '{problem}'.")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure  # type: ignore[assignment]  # SubFigure not expected here

    methods = board["method"].tolist()
    means = board["mean"].to_numpy()
    lower = means - board["ci_low"].to_numpy()
    upper = board["ci_high"].to_numpy() - means
    ax.bar(methods, means, yerr=[lower, upper], capsize=4)
    ax.set_ylabel(board["metric"].iloc[0])
    ax.set_title(f"Leaderboard — {problem}")
    ax.tick_params(axis="x", rotation=45)
    return fig
