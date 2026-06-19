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

"""Search functions for BoTorch-based continuous optimization.

This module provides search functions designed to work with BoTorch acquisition
functions that can directly optimize candidates in continuous spaces.
"""

from alf_core import Candidate, State
from alf_core.optimizer.search import BaseSearch


class ContinuousSearch(BaseSearch):
    """Search function for continuous optimization with BoTorch.

    This search function returns an empty list of candidates, signaling to
    BoTorch acquisition functions that they should perform continuous optimization
    (via optimize_acqf) rather than scoring a discrete pool.

    This is typically used with:
    - BoTorchSyntheticDataset (continuous test functions)
    - Any continuous search space where gradient-based optimization is desired

    Example:
        >>> from alf_tools.optimizer.search import ContinuousSearch
        >>> from alf_tools.optimizer.acquisition_functions import BoTorchAcquisition
        >>> from alf_core import Optimizer
        >>>
        >>> optimizer = Optimizer(
        ...     acquisition_fn=BoTorchAcquisition(
        ...         acquisition_type="qEI",
        ...         bounds=[[0, 1], [0, 1]]
        ...     ),
        ...     search_fn=ContinuousSearch()
        ... )
    """

    def __call__(self, task_state: State, **kwargs) -> list[Candidate]:
        """Return empty list to signal continuous optimization mode.

        Args:
            task_state: Current task state (not used).
            **kwargs: Additional keyword arguments (not used).

        Returns:
            Empty list of candidates, signaling the acquisition function
            should generate candidates via optimization.
        """
        return []

    def get_metrics(self, task_state: State) -> dict[str, float]:
        """Return empty metrics dict.

        Args:
            task_state: Current task state.

        Returns:
            Empty dictionary (no search-specific metrics for continuous optimization).
        """
        return {}
