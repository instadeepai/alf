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

"""Load replications, aggregate with bootstrap CIs, and rank methods.

A benchmark number without seed variance and significance is not credible, so
this module turns the raw per-replication ``metrics.csv`` files into a tidy
long-form table and derives mean ± bootstrap confidence intervals, paired
significance tests, and a leaderboard. Best-found-so-far is *derived* here (a
running max) rather than logged; simple regret is logged directly as
``optimizer/regret``.
"""

import dataclasses
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from alf_benchmark.schema import read_manifest

logger = logging.getLogger("alf-benchmark")

_LONG_COLUMNS = ["problem", "method", "seed", "round", "metric", "value"]
_ID_COLUMNS = ["problem_id", "method_id", "seed", "round"]


def _bootstrap_ci(
    values: np.ndarray, n_boot: int, ci: float, rng: np.random.RandomState
) -> tuple[float, float, float]:
    """Compute the mean and a bootstrap confidence interval.

    Args:
        values: Sample values (NaNs are dropped).
        n_boot: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95).
        rng: Seeded RNG, so the interval is reproducible.

    Returns:
        ``(mean, ci_low, ci_high)``; all NaN if there are no values, and a
        degenerate interval (``low == high == mean``) for a single value.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean, mean
    indices = rng.randint(0, values.size, size=(n_boot, values.size))
    boot_means = values[indices].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return mean, float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1 - alpha))


def _lower_is_better(metric: str) -> bool:
    """Heuristic for the optimisation direction of a metric.

    This is a name-based heuristic; metrics whose direction is not encoded in
    their name (e.g. ``coverage``, which is best near its nominal level rather
    than monotone) should be ranked by passing an explicit metric whose
    direction is known, not relied upon here.

    Args:
        metric: Metric column name.

    Returns:
        True if lower values are better (regret, loss, error, mse, ece, width).
    """
    lowered = metric.lower()
    return any(token in lowered for token in ("regret", "loss", "error", "mse", "ece", "width"))


@dataclasses.dataclass
class BenchmarkResults:
    """Tidy, long-form results loaded from a run directory.

    Attributes:
        long: Long-form table with columns
            ``[problem, method, seed, round, metric, value]``.
        primary_metrics: Mapping of problem id to its primary metric.
    """

    long: pd.DataFrame
    primary_metrics: dict[str, str]

    @staticmethod
    def _load_replication(metrics_path: Path) -> pd.DataFrame:
        """Load one replication's metrics and add derived per-round columns.

        Args:
            metrics_path: Path to a replication's ``metrics.csv``.

        Returns:
            The wide metrics frame, sorted by round, with ``derived/best_found``
            (the running max of the per-round acquired batch maxima) added where
            its input exists.

        Note:
            ``optimizer/regret`` is already a *simple* (best-found-so-far) regret:
            ``compute_regret`` measures it against the accumulated training set, so
            it is monotonically non-increasing and can be plotted directly as the
            simple-regret-vs-round curve. *Cumulative* regret (a sum of per-round
            *instantaneous* regrets) is not derivable from the current schema --
            it needs the per-round best acquired label relative to the fixed pool
            optimum, which is not logged -- so it is deliberately not produced here.
        """
        frame = pd.read_csv(metrics_path)
        if "round" in frame.columns:
            frame = frame.sort_values("round").reset_index(drop=True)
        if "acquired_candidates/round_max" in frame.columns:
            frame["derived/best_found"] = frame["acquired_candidates/round_max"].cummax()
        return frame

    @staticmethod
    def _to_long(wide: pd.DataFrame) -> pd.DataFrame:
        """Melt a wide metrics frame into tidy long form, dropping missing values.

        Args:
            wide: Concatenated wide metrics frames (ragged columns allowed).

        Returns:
            Long-form frame with columns ``[problem, method, seed, round, metric, value]``.
        """
        metric_cols = [col for col in wide.columns if col not in _ID_COLUMNS]
        long = wide.melt(
            id_vars=_ID_COLUMNS, value_vars=metric_cols, var_name="metric", value_name="value"
        )
        long = long.rename(columns={"problem_id": "problem", "method_id": "method"})
        return long.dropna(subset=["value"]).reset_index(drop=True)

    @classmethod
    def from_dir(cls, output_dir: str | Path) -> "BenchmarkResults":
        """Load all completed replications under a run directory.

        Args:
            output_dir: Root directory written by :class:`BenchmarkRunner`.

        Returns:
            The loaded results (empty long frame if nothing completed).
        """
        output_dir = Path(output_dir)
        frames: list[pd.DataFrame] = []
        primary_metrics: dict[str, str] = {}
        for manifest_path in sorted(output_dir.glob("*/*/seed=*/manifest.json")):
            rep_dir = manifest_path.parent
            manifest = read_manifest(rep_dir)
            if manifest is None or manifest.status != "completed":
                continue
            primary_metrics.setdefault(manifest.problem_id, manifest.primary_metric)
            metrics_path = rep_dir / "metrics.csv"
            if metrics_path.exists():
                frames.append(cls._load_replication(metrics_path))
        if not frames:
            return cls(pd.DataFrame(columns=_LONG_COLUMNS), primary_metrics)
        wide = pd.concat(frames, ignore_index=True)
        return cls(cls._to_long(wide), primary_metrics)

    def metrics(self) -> list[str]:
        """List the metric names present in the results.

        Returns:
            Sorted metric names.
        """
        if self.long.empty:
            return []
        return sorted(self.long["metric"].unique())

    def aggregate(self, n_boot: int = 1000, ci: float = 0.95, seed: int = 0) -> pd.DataFrame:
        """Aggregate across seeds into mean ± bootstrap CI per round.

        Args:
            n_boot: Number of bootstrap resamples.
            ci: Confidence level.
            seed: Seed for the bootstrap RNG (makes CIs reproducible).

        Returns:
            Frame with columns
            ``[problem, method, round, metric, mean, ci_low, ci_high, n_seeds]``.
        """
        if self.long.empty:
            return pd.DataFrame(
                columns=[
                    "problem",
                    "method",
                    "round",
                    "metric",
                    "mean",
                    "ci_low",
                    "ci_high",
                    "n_seeds",
                ]
            )
        rng = np.random.RandomState(seed)
        rows = []
        keys = ["problem", "method", "round", "metric"]
        for (problem, method, rnd, metric), group in self.long.groupby(keys, sort=True):
            mean, low, high = _bootstrap_ci(group["value"].to_numpy(), n_boot, ci, rng)
            rows.append({
                "problem": problem,
                "method": method,
                "round": rnd,
                "metric": metric,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "n_seeds": int(group["seed"].nunique()),
            })
        return pd.DataFrame(rows)

    def _metric_for(self, problem: str, metric: str | None) -> str:
        """Resolve the metric to use for a problem.

        Args:
            problem: Problem id.
            metric: Explicit metric, or None to use the problem's primary metric.

        Returns:
            The resolved metric name.
        """
        return metric or self.primary_metrics.get(problem, "optimizer/regret")

    def leaderboard(
        self, metric: str | None = None, n_boot: int = 1000, ci: float = 0.95, seed: int = 0
    ) -> pd.DataFrame:
        """Rank methods per problem by their final-round score.

        Args:
            metric: Metric to rank on; defaults to each problem's primary metric.
            n_boot: Number of bootstrap resamples.
            ci: Confidence level.
            seed: Seed for the bootstrap RNG.

        Returns:
            Frame with columns
            ``[problem, method, metric, round, mean, ci_low, ci_high, n_seeds, rank]``,
            ranked best-first (direction inferred from the metric name).
        """
        if self.long.empty:
            return pd.DataFrame(
                columns=[
                    "problem",
                    "method",
                    "metric",
                    "round",
                    "mean",
                    "ci_low",
                    "ci_high",
                    "n_seeds",
                    "rank",
                ]
            )
        rng = np.random.RandomState(seed)
        rows = []
        for problem in sorted(self.long["problem"].unique()):
            metric_name = self._metric_for(problem, metric)
            subset = self.long[
                (self.long["problem"] == problem) & (self.long["metric"] == metric_name)
            ]
            if subset.empty:
                logger.warning("No '%s' values for problem '%s'; skipping.", metric_name, problem)
                continue
            method_rows = []
            # Rank each method at its OWN final round, so a method that ran fewer
            # rounds is still scored (rather than vanishing at a global max round).
            for method in sorted(subset["method"].unique()):
                method_subset = subset[subset["method"] == method]
                final_round = int(method_subset["round"].max())
                values = method_subset[method_subset["round"] == final_round]["value"].to_numpy()
                mean, low, high = _bootstrap_ci(values, n_boot, ci, rng)
                method_rows.append({
                    "problem": problem,
                    "method": method,
                    "metric": metric_name,
                    "round": final_round,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n_seeds": len(values),
                })
            method_rows.sort(key=lambda row: row["mean"], reverse=not _lower_is_better(metric_name))
            for rank, row in enumerate(method_rows, start=1):
                row["rank"] = rank
                rows.append(row)
        return pd.DataFrame(rows)

    def significance(self, metric: str | None = None) -> pd.DataFrame:
        """Paired t-tests between methods on the final-round metric, per problem.

        Methods are compared per seed (paired), so the same seeds must be present
        for both methods in a pair. Reported p-values are **raw**: no
        multiple-comparison correction is applied (the caller should adjust when
        comparing many pairs), and the paired t-test assumes approximately normal
        paired differences, which is weak at the small seed counts typical here --
        treat results from few seeds as indicative.

        Args:
            metric: Metric to test; defaults to each problem's primary metric.

        Returns:
            Frame with columns
            ``[problem, metric, method_a, method_b, statistic, p_value, n_pairs]``.
        """
        if self.long.empty:
            return pd.DataFrame(
                columns=[
                    "problem",
                    "metric",
                    "method_a",
                    "method_b",
                    "statistic",
                    "p_value",
                    "n_pairs",
                ]
            )
        rows = []
        for problem in sorted(self.long["problem"].unique()):
            metric_name = self._metric_for(problem, metric)
            subset = self.long[
                (self.long["problem"] == problem) & (self.long["metric"] == metric_name)
            ]
            if subset.empty:
                continue
            final_round = int(subset["round"].max())
            final = subset[subset["round"] == final_round]
            pivot = final.pivot_table(index="seed", columns="method", values="value")
            methods = sorted(pivot.columns)
            for method_a, method_b in combinations(methods, 2):
                paired = pivot[[method_a, method_b]].dropna()
                if len(paired) < 2:
                    continue
                result = stats.ttest_rel(paired[method_a], paired[method_b])
                rows.append({
                    "problem": problem,
                    "metric": metric_name,
                    "method_a": method_a,
                    "method_b": method_b,
                    "statistic": float(result.statistic),
                    "p_value": float(result.pvalue),
                    "n_pairs": len(paired),
                })
        return pd.DataFrame(rows)
