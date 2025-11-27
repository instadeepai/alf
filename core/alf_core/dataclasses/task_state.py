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

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from alf_core.dataclasses import LabeledCandidates

if TYPE_CHECKING:
    from alf_core.dataclasses import Predictions
    from alf_core.dataset.base_dataset import BaseDataset
    from alf_core.surrogate.surrogate import Surrogate


@dataclass
class TaskState:
    """Tracks the state of a task.

    Attributes:
        dataset: The dataset containing train/validation/test splits and candidate pool.
        surrogate: The surrogate model used for predictions.
        round: Current round number in the active learning loop.
        acq_batch_size: Number of candidates to acquire per round.
        history: List of LabeledCandidates acquired in each round.
        round_metrics: Dictionary of metrics computed for the current round.
    """

    dataset: "BaseDataset"
    surrogate: "Surrogate"
    round: int = 0
    acq_batch_size: int = 0
    history: list = field(default_factory=list)
    round_metrics: dict[str, Any] = field(default_factory=dict)
    round_predictions: "Predictions" | None = None

    def update(self, acquired_candidates: LabeledCandidates) -> None:
        """Update the task state with newly acquired candidates.

        Adds the acquired candidates to history and updates the dataset splits.
        Also increments the round counter.

        Args:
            acquired_candidates: The newly acquired candidates with their labels.
        """
        self.history.append(copy.copy(acquired_candidates))
        self.dataset.update_splits(acquired_candidates)
        self.round += 1

    def check_termination(self):
        """Check if the task should be terminated early.

        Termination occurs when the remaining candidate pool is smaller than
        the acquisition batch size.

        Raises:
            AssertionError: If acquisition batch size is larger than remaining candidate pool.
        """
        assert len(self.dataset.candidate_pool) > self.acq_batch_size, (
            f"The acquisition batch size ({self.acq_batch_size}) is larger than the remaining \
              candidate pool ({len(self.dataset.candidate_pool)}), breaking ..."
        )
