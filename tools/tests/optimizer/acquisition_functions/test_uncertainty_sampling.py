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

"""Tests for the UncertaintySampling (maximum-variance) acquisition function."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from alf_core import Candidate, Modality
from alf_core.dataclasses.predictions import Predictions
from alf_core.dataclasses.round_metrics import RoundMetrics
from alf_core.dataclasses.state import State
from alf_core.dataset.base_dataset import BaseDataset
from alf_core.surrogate.surrogate import Surrogate
from alf_tools.optimizer.acquisition_functions.uncertainty_sampling import UncertaintySampling


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


def test_scores_are_the_variances() -> None:
    """The acquisition value of each candidate is its predictive variance."""
    variances = np.array([0.1, 5.0, 2.0])
    predictions = Predictions(means=np.zeros_like(variances), variances=variances)

    result = UncertaintySampling()(_candidates(3), _make_state(predictions))

    np.testing.assert_allclose(result.labels, variances)


def test_argmax_is_the_most_uncertain_candidate() -> None:
    """The highest-variance candidate scores highest, regardless of the mean."""
    variances = np.array([0.1, 5.0, 2.0])
    # Means are anti-correlated with variance to confirm the mean is ignored.
    means = np.array([100.0, 0.0, 50.0])
    predictions = Predictions(means=means, variances=variances)

    result = UncertaintySampling()(_candidates(3), _make_state(predictions))

    assert int(np.argmax(result.labels)) == 1


def test_uses_ensemble_variances() -> None:
    """An ensemble surrogate (means + variances + empirical_dist) is scored on variances."""
    empirical_dist = np.array([[0.0, 0.2], [1.0, 5.0], [2.0, 2.1]])
    variances = empirical_dist.var(axis=1)
    predictions = Predictions(
        means=empirical_dist.mean(axis=1), variances=variances, empirical_dist=empirical_dist
    )

    result = UncertaintySampling()(_candidates(3), _make_state(predictions))

    np.testing.assert_allclose(result.labels, variances)


def test_raises_when_no_variances() -> None:
    """Means-only predictions cannot be scored by UncertaintySampling."""
    predictions = Predictions(means=np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="variances"):
        UncertaintySampling()(_candidates(2), _make_state(predictions))
