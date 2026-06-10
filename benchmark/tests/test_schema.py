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

"""Tests for the manifest and metrics-CSV augmentation."""

import pandas as pd
from alf_benchmark.schema import (
    Manifest,
    augment_metrics_csv,
    read_manifest,
    write_manifest,
)


def _make_manifest(**overrides) -> Manifest:
    """Build a minimal completed manifest for tests.

    Args:
        **overrides: Fields to override on the manifest.

    Returns:
        A constructed manifest.
    """
    fields = {
        "suite_name": "suite",
        "suite_version": "0.1.0",
        "problem_id": "p",
        "problem_version": "0.1.0",
        "method_id": "m",
        "method_version": "0.1.0",
        "seed": 0,
        "family": "design",
        "primary_metric": "optimizer/regret",
        "status": "completed",
    }
    fields.update(overrides)
    return Manifest(**fields)


def test_manifest_round_trip(tmp_path):
    """A manifest written to disk reads back equal."""
    manifest = _make_manifest(num_metric_rows=3, package_versions={"alf_core": "0.1.0"})
    write_manifest(tmp_path, manifest)
    loaded = read_manifest(tmp_path)
    assert loaded is not None
    assert loaded == manifest


def test_read_manifest_missing_returns_none(tmp_path):
    """Reading a directory without a manifest returns None."""
    assert read_manifest(tmp_path) is None


def test_augment_metrics_csv_adds_identifier_columns(tmp_path):
    """Augmentation adds round/problem_id/method_id/seed columns."""
    pd.DataFrame({"optimizer/regret": [0.5, 0.3, 0.1]}).to_csv(
        tmp_path / "metrics.csv", index=False
    )
    num_rows = augment_metrics_csv(tmp_path, "p", "m", 7)
    assert num_rows == 3
    frame = pd.read_csv(tmp_path / "metrics.csv")
    assert list(frame["round"]) == [0, 1, 2]
    assert set(frame["problem_id"]) == {"p"}
    assert set(frame["method_id"]) == {"m"}
    assert set(frame["seed"]) == {7}


def test_augment_metrics_csv_missing_file_returns_zero(tmp_path):
    """Augmenting when no metrics.csv exists returns zero rows."""
    assert augment_metrics_csv(tmp_path, "p", "m", 0) == 0
