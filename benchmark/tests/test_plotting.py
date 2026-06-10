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

"""Smoke tests for the plotting helpers (headless Agg backend)."""

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from alf_benchmark.plotting import plot_curves, plot_leaderboard  # noqa: E402
from alf_benchmark.results import BenchmarkResults  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402 - backend must be set before pyplot import


def _results() -> BenchmarkResults:
    """Build small in-memory results for two methods.

    Returns:
        Results with a ``derived/best_found`` curve over three rounds.
    """
    rows = []
    for method, base in (("m_a", 5.0), ("m_b", 3.0)):
        for seed in (0, 1):
            for rnd in (0, 1, 2):
                rows.append({
                    "problem": "prob",
                    "method": method,
                    "seed": seed,
                    "round": rnd,
                    "metric": "derived/best_found",
                    "value": base + rnd + 0.1 * seed,
                })
    return BenchmarkResults(pd.DataFrame(rows), {"prob": "derived/best_found"})


def test_plot_curves_returns_figure():
    """plot_curves returns a populated figure with one line per method."""
    fig = plot_curves(_results(), metric="derived/best_found")
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].lines) == 2


def test_plot_curves_unknown_metric_raises():
    """Plotting a missing metric raises with the available metrics listed."""
    with pytest.raises(ValueError, match="not found"):
        plot_curves(_results(), metric="optimizer/regret")


def test_plot_leaderboard_returns_figure():
    """plot_leaderboard returns a bar-chart figure."""
    fig = plot_leaderboard(_results())
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].patches) == 2  # one bar per method
