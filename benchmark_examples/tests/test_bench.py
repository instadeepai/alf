# Copyright 2026 InstaDeep Ltd. All rights reserved.
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

"""Tests for bench.py helper functions."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from bench import _read_metrics  # noqa: E402


def test_read_metrics_uses_round_column(tmp_path: Path) -> None:
    """`_read_metrics` must read the `round` column directly rather than reconstruct it."""
    csv = tmp_path / "metrics.csv"
    csv.write_text("round,tell_time\n0,1.0\n1,2.0\n")
    df = _read_metrics(csv, init_best=0.5)
    assert list(df["round"]) == [0, 1]


def test_read_metrics_adds_best_found_so_far(tmp_path: Path) -> None:
    """`best_found_so_far` must be computed even when `round` column is already present."""
    csv = tmp_path / "metrics.csv"
    csv.write_text("round,acquired_candidates/round_max\n0,\n1,3.5\n")
    df = _read_metrics(csv, init_best=2.0)
    assert df["best_found_so_far"].iloc[0] == pytest.approx(2.0)
    assert df["best_found_so_far"].iloc[1] == pytest.approx(3.5)
