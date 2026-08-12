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

"""Tests for the Random acquisition function (the mandatory floor baseline)."""

from unittest.mock import MagicMock

import numpy as np
from alf_core import Candidate, Modality
from alf_core.dataclasses.predictions import Predictions
from alf_core.dataclasses.round_metrics import RoundMetrics
from alf_core.dataclasses.state import State
from alf_core.dataset.base_dataset import BaseDataset
from alf_core.surrogate.surrogate import Surrogate
from alf_tools.optimizer.acquisition_functions.random_acquisition import RandomAcquisition


def _make_state(round_num: int = 1) -> State:
    """Build a State wired to a stub surrogate for acquisition testing.

    Args:
        round_num: Round number recorded in `round_metrics`.

    Returns:
        A State with a mock surrogate (never consulted by RandomAcquisition).
    """
    surrogate = MagicMock(spec=Surrogate)
    surrogate.predict.return_value = Predictions(means=np.zeros(5))
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


def test_ignores_surrogate() -> None:
    """RandomAcquisition never calls the surrogate's predict method."""
    state = _make_state()
    acquisition = RandomAcquisition(seed=0)

    acquisition(_candidates(5), state)

    state.surrogate.predict.assert_not_called()


def test_reproducible_for_same_seed_and_round() -> None:
    """Identical (seed, round) draws the same random score vector."""
    candidates = _candidates(5)

    first = RandomAcquisition(seed=7)(candidates, _make_state(2))
    second = RandomAcquisition(seed=7)(candidates, _make_state(2))

    np.testing.assert_array_equal(first.labels, second.labels)


def test_decorrelates_across_seeds() -> None:
    """Different configured seeds at the same round draw different scores."""
    candidates = _candidates(5)

    seed_a = RandomAcquisition(seed=1)(candidates, _make_state(3))
    seed_b = RandomAcquisition(seed=2)(candidates, _make_state(3))

    assert not np.allclose(seed_a.labels, seed_b.labels)


def test_decorrelates_across_rounds() -> None:
    """The same seed draws different scores on different rounds."""
    candidates = _candidates(5)
    acquisition = RandomAcquisition(seed=4)

    round_1 = acquisition(candidates, _make_state(1))
    round_2 = acquisition(candidates, _make_state(2))

    assert not np.allclose(round_1.labels, round_2.labels)


def test_scores_are_uniform_in_unit_interval() -> None:
    """Random scores land in [0, 1), matching `np.random.Generator.random`."""
    candidates = _candidates(20)

    result = RandomAcquisition(seed=0)(candidates, _make_state())

    assert len(result.labels) == 20
    assert np.all(result.labels >= 0.0)
    assert np.all(result.labels < 1.0)
