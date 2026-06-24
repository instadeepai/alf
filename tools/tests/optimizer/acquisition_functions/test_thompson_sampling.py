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

"""Tests for the Thompson Sampling acquisition function (ensemble and GP paths)."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from alf_core import Candidate, Modality
from alf_core.dataclasses.predictions import Predictions
from alf_core.dataclasses.round_metrics import RoundMetrics
from alf_core.dataclasses.state import State
from alf_core.dataset.base_dataset import BaseDataset
from alf_core.surrogate.surrogate import Surrogate
from alf_tools.optimizer.acquisition_functions.thompson_sampling import ThompsonSampling


def _make_state(predictions: Predictions, round_num: int = 1) -> State:
    """Build a State wired to a stub surrogate for acquisition testing.

    Args:
        predictions: Predictions the stub surrogate returns from `predict`.
        round_num: Round number recorded in `round_metrics`.

    Returns:
        A State whose surrogate yields `predictions`.
    """
    surrogate = MagicMock(spec=Surrogate)
    surrogate.predict.return_value = predictions
    state = State(dataset=MagicMock(spec=BaseDataset), surrogate=surrogate)
    state.round_metrics = RoundMetrics(round=round_num)
    return state


def _candidates(n: int) -> list[Candidate]:
    """Build `n` trivial sequence candidates.

    Args:
        n: Number of candidates.

    Returns:
        A list of `n` Candidate objects.
    """
    return [Candidate(data="A" * (i + 1), modality=Modality.SEQUENCE) for i in range(n)]


def test_gp_path_preserves_direction() -> None:
    """With zero variance the GP samples equal the means, so higher mean wins."""
    means = np.array([0.0, 10.0, 20.0])
    predictions = Predictions(means=means, variances=np.zeros_like(means))
    acquisition = ThompsonSampling(seed=0)

    result = acquisition(_candidates(3), _make_state(predictions))

    # Zero variance -> samples collapse to the means, so direction is preserved.
    np.testing.assert_allclose(result.labels, means)
    assert int(np.argmax(result.labels)) == 2


def test_gp_path_is_reproducible_for_same_seed_and_round() -> None:
    """Identical (seed, round) draws the same posterior sample vector."""
    predictions = Predictions(means=np.zeros(5), variances=np.ones(5))
    candidates = _candidates(5)

    first = ThompsonSampling(seed=7)(candidates, _make_state(predictions, 2))
    second = ThompsonSampling(seed=7)(candidates, _make_state(predictions, 2))

    np.testing.assert_array_equal(first.labels, second.labels)


def test_gp_path_decorrelates_across_seeds() -> None:
    """Different configured seeds at the same round draw different samples.

    This is the regression guard for the bug where the GP path seeded only on
    the round, so every seed replication shared one standard-normal vector.
    """
    predictions = Predictions(means=np.zeros(5), variances=np.ones(5))
    candidates = _candidates(5)

    seed_a = ThompsonSampling(seed=1)(candidates, _make_state(predictions, 3))
    seed_b = ThompsonSampling(seed=2)(candidates, _make_state(predictions, 3))

    assert not np.allclose(seed_a.labels, seed_b.labels)


def test_gp_path_decorrelates_across_rounds() -> None:
    """The same seed draws different samples on different rounds."""
    predictions = Predictions(means=np.zeros(5), variances=np.ones(5))
    candidates = _candidates(5)
    acquisition = ThompsonSampling(seed=4)

    round_1 = acquisition(candidates, _make_state(predictions, 1))
    round_2 = acquisition(candidates, _make_state(predictions, 2))

    assert not np.allclose(round_1.labels, round_2.labels)


def test_gp_path_unseeded_is_nondeterministic() -> None:
    """The default (unseeded) config draws from fresh entropy each call."""
    predictions = Predictions(means=np.zeros(5), variances=np.ones(5))
    candidates = _candidates(5)
    acquisition = ThompsonSampling()  # default: seed=None

    first = acquisition(candidates, _make_state(predictions, 1))
    second = acquisition(candidates, _make_state(predictions, 1))

    assert not np.allclose(first.labels, second.labels)


def test_ensemble_path_ranks_higher_predictions_higher() -> None:
    """The empirical_dist path scores consistently-higher candidates highest."""
    # Rows = candidates, columns = ensemble members; candidate 2 dominates.
    empirical_dist = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    predictions = Predictions(means=empirical_dist.mean(axis=1), empirical_dist=empirical_dist)

    result = ThompsonSampling()(_candidates(3), _make_state(predictions))

    assert int(np.argmax(result.labels)) == 2
    assert result.labels[2] >= result.labels[0]


def test_raises_when_no_distribution_or_variance() -> None:
    """Means-only predictions cannot be Thompson sampled."""
    predictions = Predictions(means=np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="empirical_dist"):
        ThompsonSampling()(_candidates(2), _make_state(predictions))


def test_negative_seed_rejected_at_construction() -> None:
    """A negative seed fails fast with a clear message, not a deferred numpy error."""
    with pytest.raises(ValueError, match="non-negative"):
        ThompsonSampling(seed=-1)
