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

"""Tests for loading, aggregation, derivation, leaderboard, and significance."""

import numpy as np
import pandas as pd
from alf_benchmark.results import BenchmarkResults
from alf_benchmark.schema import Manifest, write_manifest


def _write_replication(root, problem, method, seed, round_max, regret):
    """Write a synthetic replication (metrics.csv + completed manifest).

    Args:
        root: Run-directory root.
        problem: Problem id.
        method: Method id.
        seed: Replication seed.
        round_max: Per-round acquired max values (NaN for the initial round).
        regret: Per-round optimizer regret values (NaN for the initial round).
    """
    rep_dir = root / problem / method / f"seed={seed}"
    rep_dir.mkdir(parents=True)
    frame = pd.DataFrame({
        "round": list(range(len(regret))),
        "acquired_candidates/round_max": round_max,
        "optimizer/regret": regret,
        "problem_id": problem,
        "method_id": method,
        "seed": seed,
    })
    frame.to_csv(rep_dir / "metrics.csv", index=False)
    write_manifest(
        rep_dir,
        Manifest(
            suite_name="s",
            suite_version="0.1.0",
            problem_id=problem,
            problem_version="0.1.0",
            method_id=method,
            method_version="0.1.0",
            seed=seed,
            family="design",
            primary_metric="optimizer/regret",
            status="completed",
        ),
    )


def _example_run(root):
    """Populate a run directory with two methods over two seeds.

    Args:
        root: Run-directory root.
    """
    _write_replication(root, "prob", "m_a", 0, [np.nan, 5.0, 7.0], [np.nan, 2.0, 1.0])
    _write_replication(root, "prob", "m_a", 1, [np.nan, 4.0, 6.0], [np.nan, 2.0, 2.0])
    _write_replication(root, "prob", "m_b", 0, [np.nan, 3.0, 4.0], [np.nan, 3.0, 3.0])
    _write_replication(root, "prob", "m_b", 1, [np.nan, 2.0, 3.0], [np.nan, 1.0, 1.0])


def test_from_dir_loads_only_completed(tmp_path):
    """Loading yields tidy long-form rows with NaNs dropped."""
    _example_run(tmp_path)
    results = BenchmarkResults.from_dir(tmp_path)
    assert not results.long.empty
    assert set(results.long.columns) == {"problem", "method", "seed", "round", "metric", "value"}
    assert results.primary_metrics == {"prob": "optimizer/regret"}
    # round-0 NaNs for regret are dropped.
    assert results.long["value"].notna().all()


def test_derived_best_found_is_running_max(tmp_path):
    """derived/best_found is the running max of per-round acquired batch maxima."""
    _example_run(tmp_path)
    results = BenchmarkResults.from_dir(tmp_path)
    best = results.long[
        (results.long["method"] == "m_a")
        & (results.long["seed"] == 0)
        & (results.long["round"] == 2)
        & (results.long["metric"] == "derived/best_found")
    ]
    assert best["value"].iloc[0] == 7.0  # cummax of [nan, 5, 7]
    # Cumulative regret is intentionally NOT derived (not recoverable from the schema).
    assert "derived/cumulative_regret" not in set(results.long["metric"])


def test_aggregate_mean_across_seeds(tmp_path):
    """Aggregation averages a metric across seeds for a given round."""
    _example_run(tmp_path)
    agg = BenchmarkResults.from_dir(tmp_path).aggregate(n_boot=200, seed=0)
    row = agg[
        (agg["problem"] == "prob")
        & (agg["method"] == "m_a")
        & (agg["round"] == 2)
        & (agg["metric"] == "optimizer/regret")
    ]
    assert np.isclose(row["mean"].iloc[0], 1.5)  # mean(1.0, 2.0)
    assert int(row["n_seeds"].iloc[0]) == 2
    assert row["ci_low"].iloc[0] <= 1.5 <= row["ci_high"].iloc[0]


def test_leaderboard_ranks_by_final_regret_lower_is_better(tmp_path):
    """Leaderboard ranks methods best-first; lower regret wins."""
    _example_run(tmp_path)
    board = BenchmarkResults.from_dir(tmp_path).leaderboard(n_boot=200)
    ranked = board.sort_values("rank")
    # final-round mean regret: m_a = 1.5, m_b = 2.0 -> m_a ranks first.
    assert list(ranked["method"]) == ["m_a", "m_b"]
    assert ranked.iloc[0]["rank"] == 1


def test_significance_returns_paired_test_rows(tmp_path):
    """Significance produces one paired-test row per method pair."""
    _example_run(tmp_path)
    sig = BenchmarkResults.from_dir(tmp_path).significance()
    assert len(sig) == 1
    row = sig.iloc[0]
    assert {row["method_a"], row["method_b"]} == {"m_a", "m_b"}
    assert 0.0 <= row["p_value"] <= 1.0
    assert row["n_pairs"] == 2


def _regret_results(values: list[float]) -> BenchmarkResults:
    """Build single-method, single-round results with one value per seed.

    Args:
        values: One regret value per seed.

    Returns:
        Results carrying the values as ``optimizer/regret`` at round 1.
    """
    rows = [
        {
            "problem": "p",
            "method": "m",
            "seed": i,
            "round": 1,
            "metric": "optimizer/regret",
            "value": v,
        }
        for i, v in enumerate(values)
    ]
    return BenchmarkResults(pd.DataFrame(rows), {"p": "optimizer/regret"})


def test_aggregate_ci_shrinks_with_more_seeds():
    """The bootstrap CI narrows as the number of seeds grows (plan acceptance)."""
    width_small = _regret_results([0.0, 2.0]).aggregate(n_boot=2000, seed=0)
    width_large = _regret_results([0.0, 2.0] * 8).aggregate(n_boot=2000, seed=0)
    small = (width_small["ci_high"] - width_small["ci_low"]).iloc[0]
    large = (width_large["ci_high"] - width_large["ci_low"]).iloc[0]
    assert large < small


def test_aggregate_is_reproducible():
    """Aggregation with the same seed produces identical CIs across calls."""
    results = _regret_results([0.0, 1.0, 2.0, 3.0, 4.0])
    pd.testing.assert_frame_equal(results.aggregate(seed=0), results.aggregate(seed=0))


def test_empty_dir_is_handled(tmp_path):
    """An empty run directory yields empty, well-formed results."""
    results = BenchmarkResults.from_dir(tmp_path)
    assert results.long.empty
    assert results.aggregate().empty
    assert results.leaderboard().empty
