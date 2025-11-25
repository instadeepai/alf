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
from typing import Any

import numpy as np
import pandas as pd

from alf.core.dataclasses import Candidate, LabeledCandidates, Predictions, TaskState

log = logging.getLogger("alf-core")


class Logger(abc.ABC):
    """Abstract base class for loggers that can write data and artifacts."""

    @abc.abstractmethod
    def write(self, state: TaskState, round_name: str | None = None) -> None:
        """Write data from the task state to the logger destination.

        Args:
            state: TaskState object to write
            round_name: Name of the round to write
        """
        pass

    def close(self) -> None:
        """Close the logger and clean up resources."""
        pass


class MetricsLogger(Logger):
    """Logger that outputs metrics to the console."""

    def write(
        self,
        state: TaskState,
        round_name: str | None = None,
    ) -> None:
        """Write metrics to terminal.

        Args:
            state: TaskState object with metrics to write
            round_name: Name of the round to write
        """
        if round_name is None:
            round_name = str(state.round)
        metrics = [f"{key}: {value:.3f}" for key, value in state.round_metrics.items()]
        message = "\n".join(metrics)
        log.info(f"Round {round_name}:\n{message}")


class FileLogger(Logger):
    """Logger that saves certain components of the state to a file.
    This includes the metrics, the acquisition batch, the data splits, and the predictions.
    """

    def __init__(self, file_path: str):
        """Initialize FileLogger.

        Args:
            file_path: Path to the file to write metrics to
        """
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        log.info(f"Initializing FileLogger at {file_path}")

    def write_metrics(self, metrics: dict[str, float]) -> None:
        """Write metrics to file.

        Args:
            metrics: Dictionary of metrics to log
        """
        metrics_df = pd.DataFrame.from_records([metrics])
        if os.path.exists(os.path.join(self.file_path, "metrics.csv")):
            saved_df = pd.read_csv(os.path.join(self.file_path, "metrics.csv"))
            metrics_df = pd.concat([saved_df, metrics_df])
        metrics_df.to_csv(os.path.join(self.file_path, "metrics.csv"), index=False)

    def write_acquisition_batch(self, acq_batch: LabeledCandidates, acq_round: int) -> None:
        """Write the acquisition batch to file.

        Args:
            acq_batch: the current batch of acquired candidates
            acq_round: the current round
        """
        acq_batch.to_dataframe().to_csv(
            os.path.join(self.file_path, f"acq_round_{acq_round}.csv"), index=False
        )

    def write_data_splits(self, data_splits: dict[str, Any]) -> None:
        """Write data splits to file.

        Args:
            data_splits: Dictionary of data splits to log
        """
        os.makedirs(os.path.join(self.file_path, "data_splits"), exist_ok=True)
        for key, data_split in data_splits.items():
            data_split.to_dataframe().to_csv(
                os.path.join(self.file_path, f"data_splits/{key}.csv"), index=False
            )

    def write_predictions(
        self,
        predictions: Predictions,
        candidates: list[Candidate],
        targets: np.ndarray,
        round_name: str,
    ) -> None:
        """Write predictions to file.

        Args:
            predictions: Predictions object to log
            candidates: List of Candidate objects corresponding to the predictions
            targets: Ground truth scores corresponding to the predictions
            round_name: Name of the round to write
        """
        round_name = round_name.lower().replace(" ", "_")
        predictions_df = predictions.to_dataframe(candidates, targets)
        predictions_df.to_csv(
            os.path.join(self.file_path, f"{round_name}_predictions.csv"), index=False
        )

    def write(self, state: TaskState, round_name: str | None = None) -> None:
        """Write data from the task state to the logger destination.

        Args:
            state: TaskState object to write
            round_name: Name of the round to write
        """
        if round_name is None:
            round_name = "round_" + str(state.round)

        self.write_metrics(state.round_metrics)

        if state.round_predictions is not None:
            self.write_predictions(
                state.round_predictions,
                state.dataset.test_dataset.candidates,
                state.dataset.test_dataset.labels,
                round_name,
            )

        if state.history:
            self.write_acquisition_batch(state.history[-1], state.round)
