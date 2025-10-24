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
import functools
from typing import Callable

import numpy as np
from scipy.stats import norm

from core.dataclasses import Predictions

EMPTY_ARRAY: np.ndarray = np.array([])    


# N.B. this is nonsense with preferential model: best f is not the value we want
def expected_improvement(
    predictions: Predictions, best_f: np.ndarray
) -> np.ndarray:
    assert (
        best_f is not EMPTY_ARRAY
    ), "expected improvement requires best_f, ensure train dataset is provided"
    if predictions.empirical_dist is not None:
        return np.mean(np.maximum(predictions.empirical_dist - best_f, 0), -1)
    elif predictions.variances is not None:
        mu = predictions.means
        sigma = np.sqrt(predictions.variances)

        sigma = np.clip(sigma, 1e-9, None)

        z = (mu - best_f) / sigma
        ei = (mu - best_f) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(ei, 0.0)
    else:
        raise ValueError(
            "Expected either `empirical_dist` or `variances` in predictions, "
            "but neither was found. Cannot compute expected improvement."
        )


def greedy(predictions: Predictions, _: np.ndarray) -> np.ndarray:\
    return predictions.means


def ucb(alpha: float, predictions: Predictions, _: np.ndarray) -> np.ndarray:
    if predictions.variances is not None:
        sigma = np.sqrt(predictions.variances)
        mu = predictions.means
        return mu + alpha * sigma
    else:
        raise ValueError(
            "Expected `variances` in predictions, but was not found. Cannot compute UCB."
        )


def thompson_sampling(predictions: Predictions, _: np.ndarray) -> np.ndarray:
    """A generalisation of Thompson Sampling to the case where batch size > num posterior samples.

    For each point, we find its maximum rank under any ensemble member,
    when predictions are sorted in ascending order.
    (Higher predictions correspond to higher ranks)
    We return the maximum rank for each candidate as an acquisition value, so that higher
    is better.
    """
    if predictions.empirical_dist is not None:
        samples = predictions.empirical_dist
        ranks = samples.argsort(axis=0).argsort(axis=0) + 1
        return ranks.max(-1)
    elif predictions.variances is not None:
        # NOTE: This needs to be implemented for GP
        raise NotImplementedError
    else:
        raise ValueError(
            "Expected either `empirical_dist` or `variances` in predictions, "
            "but neither was found. Cannot perform Thomson Sampling."
        )


npt_ucb = functools.partial(ucb, 0.1)
ucb1 = functools.partial(ucb, 0.1)
ucb3 = functools.partial(ucb, 0.3)
ucb5 = functools.partial(ucb, 0.5)
ucb10 = functools.partial(ucb, 1.0)

ACQ_DICT: dict[str, Callable] = {
    "expected_improvement": expected_improvement,
    "greedy": greedy,
    "thompson_sampling": thompson_sampling,
    "npt_ucb": npt_ucb,
    "ucb1": ucb1,
    "ucb3": ucb3,
    "ucb5": ucb5,
    "ucb10": ucb10,
}