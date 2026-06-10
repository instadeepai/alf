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

"""Result schema: the per-replication manifest and metrics-CSV augmentation.

Each replication writes ``metrics.csv`` (from ``FileStateLogger``) plus a
``manifest.json`` reproducibility record. A stable schema is what lets reference
results stay comparable across ALF versions, and what the resumable runner reads
to skip already-completed cells.
"""

import logging
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger("alf-benchmark")

MANIFEST_FILENAME = "manifest.json"
METRICS_FILENAME = "metrics.csv"

# Packages whose versions are recorded for reproducibility.
_TRACKED_PACKAGES = ("alf_core", "alf_tools", "alf_benchmark", "torch", "numpy", "pandas")


class Manifest(BaseModel):
    """Reproducibility record for a single replication.

    Attributes:
        schema_version: Version of this manifest schema.
        suite_name: Name of the suite the replication belongs to.
        suite_version: Version of the suite.
        problem_id: Problem identifier.
        problem_version: Problem version.
        method_id: Method identifier.
        method_version: Method version.
        seed: Replication seed.
        family: Experiment family (e.g. ``"design"``).
        primary_metric: Metric column defining success for the problem.
        status: ``"completed"`` or ``"failed"``.
        error: Error string if the replication failed, else ``None``.
        num_metric_rows: Number of rows written to ``metrics.csv``.
        wall_clock_seconds: Wall-clock duration of the replication.
        git_sha: Git commit SHA of the working tree, if available.
        python_version: Python version string.
        platform: Platform identifier string.
        package_versions: Versions of tracked packages.
        dataset_fingerprint: Split sizes and label means of the loaded dataset,
            identifying the exact data used (``None`` if the dataset failed to load).
        problem_config: Resolved problem configuration.
        method_config: Resolved method configuration.
    """

    schema_version: str = "1"
    suite_name: str
    suite_version: str
    problem_id: str
    problem_version: str
    method_id: str
    method_version: str
    seed: int
    family: str
    primary_metric: str
    status: Literal["completed", "failed"]
    error: str | None = None
    num_metric_rows: int | None = None
    wall_clock_seconds: float | None = None
    git_sha: str | None = None
    python_version: str = Field(default_factory=platform.python_version)
    platform: str = Field(default_factory=platform.platform)
    package_versions: dict[str, str] = Field(default_factory=dict)
    dataset_fingerprint: dict[str, float] | None = None
    problem_config: dict = Field(default_factory=dict)
    method_config: dict = Field(default_factory=dict)


def collect_package_versions() -> dict[str, str]:
    """Collect versions of the tracked packages.

    Returns:
        Mapping of package name to version string; packages that are not
        installed are omitted.
    """
    versions: dict[str, str] = {}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions


def get_git_sha() -> str | None:
    """Return the current git commit SHA, or ``None`` if unavailable.

    Returns:
        The commit SHA string, or ``None`` outside a git checkout.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def write_manifest(directory: str | Path, manifest: Manifest) -> Path:
    """Write a manifest to ``manifest.json`` in a replication directory.

    Args:
        directory: Replication directory (created if missing).
        manifest: The manifest to serialise.

    Returns:
        Path to the written ``manifest.json``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_FILENAME
    path.write_text(manifest.model_dump_json(indent=2))
    return path


def read_manifest(directory: str | Path) -> Manifest | None:
    """Read a manifest from a replication directory if present.

    Args:
        directory: Replication directory to read from.

    Returns:
        The parsed :class:`Manifest`, or ``None`` if no manifest exists.
    """
    path = Path(directory) / MANIFEST_FILENAME
    if not path.exists():
        return None
    return Manifest.model_validate_json(path.read_text())


def augment_metrics_csv(directory: str | Path, problem_id: str, method_id: str, seed: int) -> int:
    """Append identifier columns to a replication's ``metrics.csv``.

    Adds ``round`` (row order, 0-based), ``problem_id``, ``method_id`` and
    ``seed`` columns so the file is self-describing for aggregation.

    Args:
        directory: Replication directory containing ``metrics.csv``.
        problem_id: Problem identifier to stamp on every row.
        method_id: Method identifier to stamp on every row.
        seed: Replication seed to stamp on every row.

    Returns:
        The number of metric rows, or ``0`` if no ``metrics.csv`` was written.
    """
    path = Path(directory) / METRICS_FILENAME
    if not path.exists():
        return 0
    frame = pd.read_csv(path)
    frame.insert(0, "round", range(len(frame)))
    frame["problem_id"] = problem_id
    frame["method_id"] = method_id
    frame["seed"] = seed
    frame.to_csv(path, index=False)
    return len(frame)
