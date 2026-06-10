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

"""Integration tests: a tiny end-to-end sweep over synthetic components."""

import pandas as pd
import pytest
from alf_benchmark.config import ComponentSpec, MethodConfig, ProblemConfig, TaskConfig
from alf_benchmark.method import BenchmarkMethod
from alf_benchmark.problem import BenchmarkSuite
from alf_benchmark.runner import BenchmarkRunner
from alf_benchmark.schema import read_manifest


def _problem_config() -> ProblemConfig:
    """Build a synthetic problem config (seed injected by the runner).

    Returns:
        A problem config over the dummy dataset with two seeds.
    """
    return ProblemConfig(
        name="dummy_problem",
        dataset=ComponentSpec(
            name="dummy",
            config={
                "name": "dummy",
                "modality": "sequence",
                # Pool must exceed the top_n=100 used by DatasetSearch recall.
                "train_ratio": 0.4,
                "validation_frac": 0.2,
                "test_ratio": 0.2,
                "problem_type": "regression",
                "num_samples": 400,
            },
        ),
        seeds=[0, 1],
        primary_metric="optimizer/regret",
    )


def _method_config(name: str, acq_seed: int) -> MethodConfig:
    """Build a synthetic design-method config.

    Args:
        name: Method identifier.
        acq_seed: Seed for the dummy acquisition function.

    Returns:
        A design method config using the dummy components.
    """
    return MethodConfig(
        name=name,
        family="design",
        surrogate=ComponentSpec(name="dummy"),
        acquisition=ComponentSpec(name="dummy_acq", config={"seed": acq_seed}),
        search=ComponentSpec(name="dataset_search"),
        task=TaskConfig(num_acq_rounds=2, acq_batch_size=10),
    )


@pytest.fixture
def methods(registry):
    """Build two distinct dummy methods.

    Args:
        registry: Registry wired to the synthetic components.

    Returns:
        A list of two benchmark methods.
    """
    return [
        BenchmarkMethod(_method_config("m_a", acq_seed=1), registry=registry),
        BenchmarkMethod(_method_config("m_b", acq_seed=2), registry=registry),
    ]


@pytest.fixture
def suite(registry):
    """Build a one-problem suite over the synthetic dataset.

    Args:
        registry: Registry wired to the synthetic components.

    Returns:
        A benchmark suite with a single problem.
    """
    return BenchmarkSuite.from_configs(
        "test_suite", "0.1.0", [_problem_config()], registry=registry
    )


def test_sweep_produces_per_replication_outputs(registry, suite, methods, tmp_path):
    """A 2-method x 1-problem x 2-seed sweep writes metrics + manifests."""
    runner = BenchmarkRunner(registry=registry)
    manifests = runner.run(suite, methods, tmp_path)

    assert len(manifests) == 4  # 2 methods x 2 seeds
    assert all(m.status == "completed" for m in manifests)

    for method_id in ("m_a", "m_b"):
        for seed in (0, 1):
            rep_dir = tmp_path / "dummy_problem" / method_id / f"seed={seed}"
            assert (rep_dir / "metrics.csv").exists()
            assert (rep_dir / "manifest.json").exists()

            frame = pd.read_csv(rep_dir / "metrics.csv")
            # initial train round + 2 acquisition rounds = 3 rows
            assert list(frame["round"]) == [0, 1, 2]
            assert set(frame["problem_id"]) == {"dummy_problem"}
            assert set(frame["method_id"]) == {method_id}
            assert set(frame["seed"]) == {seed}
            assert "optimizer/regret" in frame.columns


def test_resume_skips_completed_replications(registry, suite, methods, tmp_path):
    """A second run reuses completed manifests and leaves outputs untouched."""
    runner = BenchmarkRunner(registry=registry)
    runner.run(suite, methods, tmp_path)

    metrics_path = tmp_path / "dummy_problem" / "m_a" / "seed=0" / "metrics.csv"
    first_mtime = metrics_path.stat().st_mtime_ns
    first_rows = len(pd.read_csv(metrics_path))

    manifests = runner.run(suite, methods, tmp_path)

    assert all(m.status == "completed" for m in manifests)
    # Skipped: file not rewritten, no duplicated rows.
    assert metrics_path.stat().st_mtime_ns == first_mtime
    assert len(pd.read_csv(metrics_path)) == first_rows


def test_failed_replication_is_recorded_not_raised(registry, suite, tmp_path):
    """A replication referencing an unknown component fails gracefully."""
    bad_method = BenchmarkMethod(
        MethodConfig(
            name="broken",
            family="design",
            surrogate=ComponentSpec(name="does_not_exist"),
            acquisition=ComponentSpec(name="dummy_acq"),
            search=ComponentSpec(name="dataset_search"),
            task=TaskConfig(num_acq_rounds=1, acq_batch_size=10),
        ),
        registry=registry,
    )
    runner = BenchmarkRunner(registry=registry)
    manifests = runner.run(suite, [bad_method], tmp_path)

    assert len(manifests) == 2  # one per seed
    assert all(m.status == "failed" for m in manifests)
    assert all(m.error is not None for m in manifests)
    # Manifest is still written so the failure is inspectable; a later run will retry it.
    assert read_manifest(tmp_path / "dummy_problem" / "broken" / "seed=0") is not None


def test_retry_after_failure_does_not_leak_rows(registry, suite, tmp_path):
    """Re-running a previously failed cell clears stale outputs (no row leakage)."""
    rep_dir = tmp_path / "dummy_problem" / "m_a" / "seed=0"
    runner = BenchmarkRunner(registry=registry)

    # First attempt fails mid-run: round 2 exhausts the candidate pool (batch=100,
    # pool=160), after the initial + round-1 rows have already been written.
    failing = BenchmarkMethod(
        MethodConfig(
            name="m_a",
            family="design",
            surrogate=ComponentSpec(name="dummy"),
            acquisition=ComponentSpec(name="dummy_acq"),
            search=ComponentSpec(name="dataset_search"),
            task=TaskConfig(num_acq_rounds=2, acq_batch_size=100),
        ),
        registry=registry,
    )
    runner.run(suite, [failing], tmp_path)
    assert read_manifest(rep_dir).status == "failed"
    assert (rep_dir / "metrics.csv").exists()  # a partial metrics file was left behind

    # Retry with a working config of the same id: stale rows must be cleared, not appended.
    fixed = BenchmarkMethod(_method_config("m_a", acq_seed=1), registry=registry)
    runner.run(suite, [fixed], tmp_path)

    manifest = read_manifest(rep_dir)
    assert manifest.status == "completed"
    frame = pd.read_csv(rep_dir / "metrics.csv")
    assert list(frame["round"]) == [0, 1, 2]
    assert manifest.num_metric_rows == 3
