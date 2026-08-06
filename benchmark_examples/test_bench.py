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

"""Regression tests for `build_dataset` in the benchmark_examples taster scripts.

`bench.py`'s "flip" dataset must use a FLIP split whose `train` partition is large
enough to support the default acquisition budgets used by the benchmark_examples
scripts (`--num-rounds 5 --batch-size 50`, i.e. 250 candidates). GB1's "one_vs_rest"
split only tags 28 rows as `train` (a deliberate extrapolation split), which starves
the candidate pool after `train_ratio` is applied and trips `_check_budget`. See
`flip_split="low_vs_high"` in `build_dataset`.
"""

import bench
import pandas as pd
from alf_tools.datasets.flip import FLIP

# Default budget used by benchmarking_acquisition_functions.py / benchmarking_surrogates.py.
_DEFAULT_NUM_ROUNDS = 5
_DEFAULT_BATCH_SIZE = 50
_DEFAULT_BUDGET = _DEFAULT_NUM_ROUNDS * _DEFAULT_BATCH_SIZE


def _make_gb1_low_vs_high_mock_df() -> pd.DataFrame:
    """Synthetic DataFrame matching real GB1 `low_vs_high` split proportions.

    Real counts (from FLIP's splits.zip): 5089 train rows, 3644 test rows. We use a
    smaller but proportionally similar mock to keep the test fast while still
    exercising the same code path as `build_dataset`.

    Returns:
        DataFrame with columns: sequence, target, set, validation.
    """
    rows = []
    for i in range(509):
        rows.append({
            "sequence": f"TRAINSEQ{i:05d}",
            "target": float(i),
            "set": "train",
            "validation": False,
        })
    for i in range(364):
        rows.append({
            "sequence": f"TESTSEQQ{i:05d}",
            "target": float(1000 + i),
            "set": "test",
            "validation": False,
        })
    return pd.DataFrame(rows)


def test_flip_dataset_uses_low_vs_high_split():
    """`build_dataset("flip", ...)` must not regress to the degenerate `one_vs_rest` split."""
    dataset = bench.build_dataset("flip", seed=0)
    assert dataset.config.flip_dataset == "gb1"
    assert dataset.config.flip_split == "low_vs_high"


def test_flip_candidate_pool_supports_default_budget(monkeypatch):
    """The GB1 candidate pool must be large enough for the scripts' default budget.

    Reproduces (at reduced scale) the crash described in `bench.py`'s `_check_budget`:
    with the old `one_vs_rest` split, `(1 - train_ratio) * 28 candidates` is far smaller
    than `num_rounds * batch_size`, so `_check_budget` raises `SystemExit`. With
    `low_vs_high`'s much larger `train` partition, the pool comfortably clears the
    default budget.
    """
    mock_df = _make_gb1_low_vs_high_mock_df()
    monkeypatch.setattr(FLIP, "_load_split_dataframe", lambda self: mock_df)

    dataset = bench.build_dataset("flip", seed=0)
    dataset.setup()

    # Must not raise SystemExit for the scripts' default acquisition budget.
    bench._check_budget(dataset, num_rounds=_DEFAULT_NUM_ROUNDS, batch_size=_DEFAULT_BATCH_SIZE)
    assert len(dataset.candidate_pool) > _DEFAULT_BUDGET
