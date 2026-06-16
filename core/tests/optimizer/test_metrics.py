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

"""Unit tests for optimizer recall/regret metrics."""

import numpy as np
import pytest
from alf_core.dataclasses import Candidate, LabelledCandidates, Modality
from alf_core.utils.metrics.acquisition_batch import compute_recall, compute_regret


def _labelled(labels: list[float]) -> LabelledCandidates:
    """Build a LabelledCandidates from a list of labels."""
    candidates = [Candidate(data=f"seq{i}", modality=Modality.SEQUENCE) for i in range(len(labels))]
    return LabelledCandidates(candidates=candidates, labels=np.array(labels, dtype=float))


class TestComputeRecall:
    """Tests for compute_recall, including small-pool edge cases."""

    def test_top_n_larger_than_pool_is_clamped_to_pool_size(self):
        """top_n exceeding the pool size is clamped instead of raising IndexError.

        The key keeps the nominal top_n name, but the recall is computed over the
        whole pool: the threshold is the worst label (1.0) and the denominator is 5.
        """
        pool = _labelled([1.0, 2.0, 3.0, 4.0, 5.0])
        acquired = _labelled([5.0, 4.0])
        result = compute_recall(pool, acquired, top_percentile=0.4, top_n=100)
        # top 40% -> 2 candidates (>= 4.0); both acquired -> 2/2 = 1.0
        assert result["optimizer/top_40pc_recall"] == pytest.approx(1.0)
        # top_n clamped to 5 -> threshold 1.0; 2 acquired >= 1.0 -> 2/5
        assert result["optimizer/top_100_recall"] == pytest.approx(0.4)

    def test_small_percentile_uses_correct_threshold(self):
        """A percentile that rounds the rank count to zero is clamped to 1, so the
        threshold is the pool's best label (not, via a negative index, its worst).

        Here int(5 * 0.1) == 0. Buggy behaviour indexed [-1] (worst label 1.0) and
        divided by 0 (-> inf, clipped to 1.0), spuriously reporting full recall. The
        fix uses the top candidate (label 5.0) as the threshold, so acquiring only a
        mid-ranked candidate yields zero recall.
        """
        pool = _labelled([1.0, 2.0, 3.0, 4.0, 5.0])  # int(5 * 0.1) == 0
        acquired = _labelled([3.0])
        result = compute_recall(pool, acquired, top_percentile=0.1, top_n=3)
        assert result["optimizer/top_10pc_recall"] == pytest.approx(0.0)

    def test_recall_values_for_known_pool(self):
        """Recall values match a hand-computed example."""
        pool = _labelled([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        acquired = _labelled([10.0, 9.0])
        result = compute_recall(pool, acquired, top_percentile=0.2, top_n=3)
        # top 20% -> 2 candidates (>= 9.0); both acquired -> 2/2 = 1.0
        assert result["optimizer/top_20pc_recall"] == pytest.approx(1.0)
        # top 3 -> threshold 8.0; 2 of the acquired are >= 8.0 -> 2/3
        assert result["optimizer/top_3_recall"] == pytest.approx(2.0 / 3.0)


class TestComputeRegret:
    """Tests for compute_regret."""

    def test_regret_is_best_pool_minus_best_acquired(self):
        """Regret is the gap between the pool's best and the best acquired label."""
        pool = _labelled([1.0, 2.0, 3.0, 10.0])
        acquired = _labelled([2.0, 7.0])
        result = compute_regret(pool, acquired)
        assert result["optimizer/regret"] == pytest.approx(3.0)
