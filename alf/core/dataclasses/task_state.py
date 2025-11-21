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
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from alf.core.dataclasses import LabeledCandidates
from alf.core.utils.io import input_handler

log = logging.getLogger("alf-core")

if TYPE_CHECKING:
    from alf.core.dataset.base_dataset import BaseDataset
    from alf.core.surrogate.surrogate import Surrogate


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

    def print_metrics(self, round_name: int | str) -> None:
        """Print the current round's metrics to the logger.

        Formats and logs all metrics for the current round.

        Args:
            round_name: Name or number identifying the current round.
        """
        metrics_list: list = []
        for key, value in self.round_metrics.items():
            # Skip the "round" key explicitly
            if key != "round":
                metrics_list.append(f"{key}: {value:.3f}")
        metrics_str = "\t".join(metrics_list)
        log.info(f"Round {round_name}:\t{metrics_str}")  # noqa: E231

    def save_metrics(
        self,
        output_dir: str,
    ) -> None:
        """Save the current round's metrics to a CSV file.

        Appends the metrics to metrics.csv in the output directory. If the file
        already exists, the new metrics are appended to it.

        Args:
            output_dir: Directory path where the metrics CSV file will be saved.
        """
        metrics_df = pd.DataFrame.from_records([self.round_metrics])

        if input_handler.isfile(os.path.join(output_dir, "metrics.csv")):
            saved_df = input_handler.read_csv(os.path.join(output_dir, "metrics.csv"))
            metrics_df = pd.concat([saved_df, metrics_df])

        input_handler.save_csv(os.path.join(output_dir, "metrics.csv"), metrics_df)

    def save_history(self, output_dir: str) -> None:
        """Save the acquisition history to CSV files.

        Saves each round's acquired candidates to a separate CSV file named
        acq_round_{round_number}.csv.

        Args:
            output_dir: Directory path where the history CSV files will be saved.
        """
        for acq_round, acq_points in enumerate(self.history):
            input_handler.save_csv(
                os.path.join(output_dir, f"acq_round_{acq_round}.csv"),
                acq_points.to_dataframe(),
            )

    def save(self, save_path: str | None, _verbose: bool = False) -> None:
        """Save both metrics and acquisition history to the output directory.

        Args:
            save_path: Directory path where files will be saved. If None, nothing is saved.
            _verbose: If True, log a message when saving.
        """
        if _verbose:
            log.info(f"Saving metrics and history to {save_path}")

        if save_path:
            self.save_metrics(save_path)
            self.save_history(save_path)

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
