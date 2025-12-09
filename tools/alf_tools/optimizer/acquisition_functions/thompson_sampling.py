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

import numpy as np
from alf_core import AcquisitionFunction, Predictions, TaskState


class ThompsonSampling(AcquisitionFunction):
    """Thompson Sampling acquisition function.

    A generalisation of Thompson Sampling to the case where batch size > num posterior samples.

    For each point, we find its maximum rank under any ensemble member,
    when predictions are sorted in ascending order.
    (Higher predictions correspond to higher ranks)
    We return the maximum rank for each candidate as an acquisition value, so that higher
    is better.
    """

    def _get_acquisition_values(self, predictions: Predictions, state: TaskState) -> np.ndarray:
        """Computes acquisition values for candidates based on surrogate predictions.

        Args:
            predictions: The predictions from the surrogate model.
            state: The task state containing the dataset and surrogate model.

        Returns:
            The acquisition values for the candidates.

        Raises:
            ValueError: If `empirical_dist` or `variances` is not found in predictions.
            NotImplementedError: If `variances` is not found in predictions.
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
