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

import copy
from dataclasses import dataclass, field

from alf_core.dataclasses import LabelledCandidates, Predictions
from alf_core.dataclasses.round_metrics import RoundMetrics
from alf_core.dataset.base_dataset import BaseDataset
from alf_core.enums import ProblemType
from alf_core.surrogate.surrogate import Surrogate


@dataclass
class State:
    """Tracks the state of a task.

    Attributes:
        dataset: The dataset containing train/validation/test splits and candidate pool.
        surrogate: The surrogate model used for predictions.
        round: Current round number in the active learning loop.
        acq_batch_size: Number of candidates to acquire per round.
        history: List of LabelledCandidates acquired in each round.
        round_metrics: RoundMetrics instance holding scalar metrics and per-epoch
            training history for the current round.
        round_predictions: Predictions on the test set for the current round.
    """

    dataset: BaseDataset
    surrogate: Surrogate
    round: int = 0
    acq_batch_size: int = 0
    history: list = field(default_factory=list)
    round_metrics: RoundMetrics = field(default_factory=lambda: RoundMetrics(round=0))
    round_predictions: Predictions | None = None

    @property
    def problem_type(self) -> "ProblemType":
        """The problem type, as declared in the dataset configuration.

        Returns:
            ProblemType enum value from the dataset config.
        """
        return self.dataset.config.problem_type

    def update(self, acquired_candidates: LabelledCandidates) -> None:
        """Update the state with newly acquired candidates.

        Adds the acquired candidates to history and updates the dataset splits.
        Also increments the round counter.

        Args:
            acquired_candidates: The newly acquired candidates with their labels.
        """
        self.history.append(copy.copy(acquired_candidates))
        self.dataset.update_splits(acquired_candidates)
        self.round += 1
