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

import abc
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from alf.core.dataclasses import Candidate, LabeledCandidates, Predictions, TaskState

logger = logging.getLogger("alf-core")


class TaskStateLogger(abc.ABC):
    """Abstract base class for task_state_loggers to write task state information."""

    @abc.abstractmethod
    def log(self, state: TaskState, round_name: str | None = None) -> None:
        """Log data from the task state to the logger destination.

        Args:
            state: TaskState object to log
            round_name: Name of the current round in the task, depending on the task type
                e.g. "initial_train_round", "supervised evaluation", "zero-shot evaluation",
                or the round number for design tasks.
        """
        pass


class TerminalTaskStateLogger(TaskStateLogger):
    """Logger that outputs metrics to the terminal."""

    def log(
        self,
        state: TaskState,
        round_name: str | None = None,
    ) -> None:
        """Log metrics in the task state to the terminal.

        Args:
            state: TaskState object with metrics to log
            round_name: Name of the current round in the task, depending on the task type
                e.g. "initial_train_round", "supervised evaluation", "zero-shot evaluation",
                or the round number for design tasks.
        """
        if round_name is None:
            round_name = str(state.round)
        metrics = [f"{key}: {value:.3f}" for key, value in state.round_metrics.items()]
        message = "\n".join(metrics)
        logger.info("Round %s:\n%s", round_name, message)


class FileTaskStateLogger(TaskStateLogger):
    """Logger that saves certain components of the state to a file.
    This includes the metrics, the acquisition batch, the data splits, and the predictions.
    """

    def __init__(self, output_path: str | os.PathLike):
        """Initialize FileTaskStateLogger.

        Args:
            output_path: Path to the directory to save the state information to
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        logger.info("Initializing FileTaskStateLogger at %s", self.output_path)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to file.

        Args:
            metrics: Dictionary of metrics to log
        """
        metrics_df = pd.DataFrame.from_records([metrics])
        if (self.output_path / "metrics.csv").exists():
            saved_df = pd.read_csv(self.output_path / "metrics.csv")
            metrics_df = pd.concat([saved_df, metrics_df])
        metrics_df.to_csv(self.output_path / "metrics.csv", index=False)

    def log_acquisition_batch(self, acq_batch: LabeledCandidates, acq_round: int) -> None:
        """Log the acquisition batch to file.

        Args:
            acq_batch: the current batch of acquired candidates
            acq_round: the current round
        """
        acq_batch.to_dataframe().to_csv(
            self.output_path / f"acq_round_{acq_round}.csv", index=False
        )

    def log_predictions(
        self,
        predictions: Predictions,
        candidates: list[Candidate],
        targets: np.ndarray,
        round_name: str,
    ) -> None:
        """Log predictions to file.

        Args:
            predictions: Predictions object to log
            candidates: List of Candidate objects corresponding to the predictions
            targets: Ground truth scores corresponding to the predictions
            round_name: Name of the current round in the task, depending on the task type
                e.g. "initial_train_round", "supervised evaluation", "zero-shot evaluation",
                or the round number for design tasks.
        """
        round_name = round_name.lower().replace(" ", "_")
        predictions_df = predictions.to_dataframe(candidates, targets)
        predictions_df.to_csv(self.output_path / f"{round_name}_predictions.csv", index=False)

    def log(self, state: TaskState, round_name: str | None = None) -> None:
        """Log data from the task state to the logger destination.

        Args:
            state: TaskState object to log
            round_name: Name of the current round in the task, depending on the task type
                e.g. "initial_train_round", "supervised evaluation", "zero-shot evaluation",
                or the round number for design tasks.
        """
        if round_name is None:
            round_name = "round_" + str(state.round)

        self.log_metrics(state.round_metrics)

        if state.round_predictions is not None:
            self.log_predictions(
                state.round_predictions,
                state.dataset.test_dataset.candidates,
                state.dataset.test_dataset.labels,
                round_name,
            )

        if state.history:
            self.log_acquisition_batch(state.history[-1], state.round)
