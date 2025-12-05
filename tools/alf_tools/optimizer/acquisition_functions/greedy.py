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


class Greedy(AcquisitionFunction):
    """Greedy acquisition function."""

    def _get_acquisition_values(self, predictions: Predictions, state: TaskState) -> np.ndarray:
        """Computes acquisition values for candidates based on surrogate predictions.

        Args:
            predictions: The predictions from the surrogate model.
            state: The task state containing the dataset and surrogate model.

        Returns:
            np.ndarray: The acquisition values for the candidates.
        """
        return predictions.means
