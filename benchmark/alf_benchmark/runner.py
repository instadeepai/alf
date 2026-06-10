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

"""The sweep orchestrator.

``BenchmarkRunner`` expands ``(problem × method × seed)`` into replications.
Each replication is one unchanged ``task.run()``: the runner builds components
from the registry, seeds the RNGs, binds the oracle by mode, points a
``FileStateLogger`` at a per-replication directory, and records a manifest.

The runner is resilient (one failure never aborts the sweep) and resumable
(replications with a ``completed`` manifest are skipped).
"""

import logging
import shutil
import time
from pathlib import Path

from alf_core.utils.state_logger import FileStateLogger

from alf_benchmark.method import BenchmarkMethod
from alf_benchmark.problem import BenchmarkProblem, BenchmarkSuite
from alf_benchmark.registry import Registry, default_registry
from alf_benchmark.schema import (
    Manifest,
    augment_metrics_csv,
    collect_package_versions,
    get_git_sha,
    read_manifest,
    write_manifest,
)
from alf_benchmark.seeding import seed_everything

logger = logging.getLogger("alf-benchmark")


class BenchmarkRunner:
    """Runs a suite of problems against a list of methods over their seeds."""

    def __init__(self, registry: Registry | None = None, deterministic: bool = False) -> None:
        """Initialise the runner.

        Args:
            registry: Registry used to build components (defaults to shared).
            deterministic: Request deterministic kernels per replication.
        """
        self._registry = registry if registry is not None else default_registry()
        self.deterministic = deterministic

    def run(
        self,
        suite: BenchmarkSuite,
        methods: list[BenchmarkMethod],
        output_dir: str | Path,
    ) -> list[Manifest]:
        """Run every ``(problem, method, seed)`` replication in the suite.

        Args:
            suite: The suite of problems to run.
            methods: The methods to evaluate against every problem.
            output_dir: Root directory for per-replication outputs.

        Returns:
            The manifest for every replication, in sweep order.
        """
        output_dir = Path(output_dir)
        manifests: list[Manifest] = []
        for problem in suite.problems:
            for method in methods:
                for seed in problem.seeds:
                    manifests.append(
                        self._run_replication(suite, problem, method, seed, output_dir)
                    )
        return manifests

    def replication_dir(
        self, output_dir: str | Path, problem_id: str, method_id: str, seed: int
    ) -> Path:
        """Return the output directory for a single replication.

        Args:
            output_dir: Root output directory.
            problem_id: Problem identifier.
            method_id: Method identifier.
            seed: Replication seed.

        Returns:
            ``<output_dir>/<problem_id>/<method_id>/seed=<seed>``.
        """
        return Path(output_dir) / problem_id / method_id / f"seed={seed}"

    def _run_replication(
        self,
        suite: BenchmarkSuite,
        problem: BenchmarkProblem,
        method: BenchmarkMethod,
        seed: int,
        output_dir: Path,
    ) -> Manifest:
        """Run (or skip) one replication and write its manifest.

        Args:
            suite: The suite being run.
            problem: The problem for this replication.
            method: The method for this replication.
            seed: The replication seed.
            output_dir: Root output directory.

        Returns:
            The manifest for the replication (the existing one if skipped).
        """
        rep_dir = self.replication_dir(output_dir, problem.name, method.name, seed)
        existing = read_manifest(rep_dir)
        if existing is not None and existing.status == "completed":
            logger.info("Skipping completed replication: %s", rep_dir)
            return existing

        # Clear any partial outputs from a previous failed attempt: FileStateLogger
        # appends to metrics.csv, so a stale file would leak rows into the retry.
        if rep_dir.exists():
            shutil.rmtree(rep_dir)

        seed_everything(seed, self.deterministic)
        started = time.perf_counter()
        manifest_fields = {
            "suite_name": suite.name,
            "suite_version": suite.version,
            "problem_id": problem.name,
            "problem_version": problem.version,
            "method_id": method.name,
            "method_version": method.version,
            "seed": seed,
            "family": method.family,
            "primary_metric": problem.primary_metric,
            "problem_config": problem.config.model_dump(mode="json"),
            "method_config": method.config.model_dump(mode="json"),
            "package_versions": collect_package_versions(),
            "git_sha": get_git_sha(),
        }

        try:
            dataset = problem.build_dataset(seed)
            dataset_fingerprint = {k: float(v) for k, v in dataset.get_metrics().items()}
            surrogate = method.build_surrogate()
            optimizer = method.build_optimizer()
            oracle = method.build_oracle(dataset)
            task = method.build_task()
            state = task.setup(dataset=dataset, surrogate=surrogate)
            file_logger = FileStateLogger(output_path=rep_dir)
            task.run(
                state=state,
                state_loggers=[file_logger],
                optimizer=optimizer,
                oracle=oracle,
            )
            num_rows = augment_metrics_csv(rep_dir, problem.name, method.name, seed)
            manifest = Manifest(
                status="completed",
                num_metric_rows=num_rows,
                wall_clock_seconds=time.perf_counter() - started,
                dataset_fingerprint=dataset_fingerprint,
                **manifest_fields,
            )
        except Exception as exc:
            logger.exception("Replication failed: %s", rep_dir)
            manifest = Manifest(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                wall_clock_seconds=time.perf_counter() - started,
                **manifest_fields,
            )

        write_manifest(rep_dir, manifest)
        return manifest
