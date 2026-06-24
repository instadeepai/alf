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

"""Unit tests for DatasetSearch metric wiring."""

import numpy as np
import pytest
from alf_core.dataclasses import Candidate, LabelledCandidates, Modality, State
from alf_core.optimizer.search import DatasetSearch


def _labelled(labels: list[float]) -> LabelledCandidates:
    """Build a LabelledCandidates from a list of labels."""
    candidates = [Candidate(data=f"seq{i}", modality=Modality.SEQUENCE) for i in range(len(labels))]
    return LabelledCandidates(candidates=candidates, labels=np.array(labels, dtype=float))


def _state(dummy_dataset, dummy_surrogate, pool, train, history) -> State:
    """Build a State with controlled pool, train split, and acquisition history."""
    dummy_dataset.init_candidate_pool = pool
    dummy_dataset.splits["train"] = train
    return State(dataset=dummy_dataset, surrogate=dummy_surrogate, history=history)


class TestDatasetSearchRegret:
    """Regret wiring in DatasetSearch.get_metrics uses acquired-only labels."""

    def test_regret_excludes_seed_and_stays_non_negative(self, dummy_dataset, dummy_surrogate):
        """A seed label exceeding the pool best does not drive regret negative.

        Regret is reconstructed from `state.history` (acquired-only), so the seed
        label of 99.0 in the train split is ignored.
        """
        pool = _labelled([1.0, 5.0, 10.0])
        train = _labelled([99.0, 3.0])  # seed (99.0) exceeds the pool best
        history = [_labelled([3.0]), _labelled([7.0])]  # acquired during the loop
        state = _state(dummy_dataset, dummy_surrogate, pool, train, history)

        metrics = DatasetSearch().get_metrics(state)

        # best pool (10.0) - best acquired (7.0) = 3.0, independent of the seed's 99.0
        assert metrics["optimizer/regret"] == pytest.approx(3.0)
        assert metrics["optimizer/regret"] >= 0.0

    def test_regret_reaches_zero_when_pool_optimum_acquired(self, dummy_dataset, dummy_surrogate):
        """Regret decays to zero once the pool optimum is among the acquired candidates."""
        pool = _labelled([1.0, 5.0, 10.0])
        train = _labelled([2.0])
        history = [_labelled([5.0]), _labelled([10.0])]  # pool optimum acquired
        state = _state(dummy_dataset, dummy_surrogate, pool, train, history)

        metrics = DatasetSearch().get_metrics(state)

        assert metrics["optimizer/regret"] == pytest.approx(0.0)


class TestDatasetSearchRecall:
    """Recall wiring in DatasetSearch.get_metrics uses acquired-only labels."""

    def test_recall_excludes_seed(self, dummy_dataset, dummy_surrogate):
        """A top-pool label held only by the seed does not count towards recall.

        The pool best (10.0) sits in the train split (seed) but was never
        acquired, so top-percentile recall is zero.
        """
        pool = _labelled([float(i) for i in range(1, 11)])  # 1.0 .. 10.0
        train = _labelled([10.0])  # seed holds the pool optimum
        history = [_labelled([5.0])]  # only a mid-ranked candidate acquired
        state = _state(dummy_dataset, dummy_surrogate, pool, train, history)

        metrics = DatasetSearch().get_metrics(state)

        # top 10% -> 1 candidate (>= 10.0); the seed's 10.0 must not count -> 0/1
        assert metrics["optimizer/top_10pc_recall"] == pytest.approx(0.0)


class TestDatasetSearchMetricGuards:
    """Behaviour before any acquisition has occurred."""

    def test_metrics_omitted_before_any_acquisition(self, dummy_dataset, dummy_surrogate):
        """No recall or regret is reported until at least one candidate is acquired."""
        pool = _labelled([1.0, 5.0, 10.0])
        train = _labelled([2.0])
        state = _state(dummy_dataset, dummy_surrogate, pool, train, [])

        metrics = DatasetSearch().get_metrics(state)

        assert metrics == {}
