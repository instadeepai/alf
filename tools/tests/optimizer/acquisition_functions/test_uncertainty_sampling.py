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

"""Tests for the uncertainty sampling acquisition function."""

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


def _make_state(predictions: Predictions) -> State:
    """Build a State wired to a stub surrogate returning fixed predictions.

    Args:
        predictions: Predictions the stub surrogate returns from `predict`.

    Returns:
        A State whose surrogate returns `predictions`.
    """
    surrogate = MagicMock(spec=Surrogate)
    surrogate.predict.return_value = predictions
    state = State(dataset=MagicMock(spec=BaseDataset), surrogate=surrogate)
    state.round_metrics = RoundMetrics(round=1)
    return state


def _candidates(n: int) -> list[Candidate]:
    """Build `n` trivial sequence candidates.

    Args:
        n: Number of candidates.

    Returns:
        A list of `n` Candidate objects.
    """
    return [Candidate(data="A" * (i + 1), modality=Modality.SEQUENCE) for i in range(n)]


def test_scores_are_standard_deviations() -> None:
    """Acquisition values are the square root of the reported variances."""
    variances = np.array([0.04, 0.25, 1.0])
    state = _make_state(Predictions(means=np.array([5.0, 1.0, 3.0]), variances=variances))

    result = UncertaintySampling()(_candidates(3), state)

    np.testing.assert_allclose(result.labels, [0.2, 0.5, 1.0])


def test_ignores_the_predicted_mean() -> None:
    """Two candidate sets differing only in means score identically."""
    variances = np.array([0.1, 0.9, 0.5])
    candidates = _candidates(3)

    low_means = UncertaintySampling()(
        candidates, _make_state(Predictions(means=np.zeros(3), variances=variances))
    )
    high_means = UncertaintySampling()(
        candidates, _make_state(Predictions(means=np.full(3, 100.0), variances=variances))
    )

    np.testing.assert_array_equal(low_means.labels, high_means.labels)


def test_top_candidate_is_the_most_uncertain() -> None:
    """The highest-variance candidate wins, regardless of its mean."""
    candidates = _candidates(3)
    state = _make_state(
        Predictions(means=np.array([10.0, 0.0, 5.0]), variances=np.array([0.1, 4.0, 1.0]))
    )

    result = UncertaintySampling()(candidates, state)

    assert result.get_top_k(1).candidates == [candidates[1]]


def test_falls_back_to_ensemble_disagreement() -> None:
    """Without `variances`, the population std across ensemble members is used."""
    # Member predictions [0, 2, 4] have population std sqrt(8/3); hardcoded rather than
    # recomputed so a ddof regression (sample std = 2.0) fails the test.
    empirical_dist = np.array([[1.0, 1.0, 1.0], [0.0, 2.0, 4.0]])
    state = _make_state(Predictions(means=np.array([1.0, 2.0]), empirical_dist=empirical_dist))

    result = UncertaintySampling()(_candidates(2), state)

    np.testing.assert_allclose(result.labels, [0.0, np.sqrt(8.0 / 3.0)])


def test_single_member_ensemble_reports_no_uncertainty() -> None:
    """A one-member ensemble has no disagreement, so every score is zero."""
    state = _make_state(
        Predictions(means=np.array([1.0, 2.0]), empirical_dist=np.array([[1.0], [2.0]]))
    )

    result = UncertaintySampling()(_candidates(2), state)

    np.testing.assert_array_equal(result.labels, [0.0, 0.0])


def test_prefers_variances_over_empirical_dist() -> None:
    """When both are present, the reported `variances` take precedence."""
    state = _make_state(
        Predictions(
            means=np.array([0.0, 0.0]),
            variances=np.array([9.0, 16.0]),
            empirical_dist=np.array([[0.0, 0.0], [0.0, 0.0]]),
        )
    )

    result = UncertaintySampling()(_candidates(2), state)

    np.testing.assert_allclose(result.labels, [3.0, 4.0])


def test_clips_negative_variances() -> None:
    """Tiny negative variances from numerical error score 0 rather than NaN."""
    state = _make_state(Predictions(means=np.zeros(2), variances=np.array([-1e-12, 0.25])))

    result = UncertaintySampling()(_candidates(2), state)

    assert not np.isnan(result.labels).any()
    np.testing.assert_allclose(result.labels, [0.0, 0.5])


def test_raises_without_any_uncertainty() -> None:
    """A mean-only surrogate cannot support uncertainty sampling."""
    state = _make_state(Predictions(means=np.array([1.0, 2.0])))

    with pytest.raises(ValueError, match="Cannot perform uncertainty sampling"):
        UncertaintySampling()(_candidates(2), state)
