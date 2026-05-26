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
from typing import Callable

import numpy as np
import pandas as pd
from alf_core.dataclasses import (
    Candidate,
    LabelledCandidates,
    Predictions,
    State,
    SurrogateEpochMetrics,
)
from alf_core.utils.enums import ProblemType

logger = logging.getLogger("alf-core")


class StateLogger(abc.ABC):
    """Abstract base class for state loggers.

    Subclasses must implement :meth:`log`. They may also override
    :meth:`_log_training_history` to capture per-epoch training metrics.

    Per-epoch metrics are available as `state.round_metrics.training_history`,
    a `list[SurrogateEpochMetrics]` populated by the surrogate's
    `get_epoch_metrics()` after each training round.  `SurrogateEpochMetrics`
    carries standard CNN fields (`train_loss`, `val_loss`,
    `train_spearman`, `val_spearman`, `train_mse`, `val_mse`) plus an
    `extra` dict for model-specific metrics.  The list is empty when the
    surrogate does not override `get_epoch_metrics()`.
    """

    @abc.abstractmethod
    def log(self, state: State, round_name: str | None = None) -> None:
        """Log data from the task state to the logger destination.

        `state.round_metrics.training_history` contains a
        `list[SurrogateEpochMetrics]` with per-epoch training metrics for this round.
        Override :meth:`_log_training_history` to capture them.

        Args:
            state: State object to log
            round_name: Name of the current round in the task, depending on the task type
                e.g. "initial_train_round", "supervised evaluation", "zero-shot evaluation",
                or the round number for design tasks.
        """
        pass

    def _log_training_history(
        self, training_history: list[SurrogateEpochMetrics], round_num: int | None = None
    ) -> None:
        """Optionally log per-epoch training metrics for the current round.

        This is a no-op by default. Override in a subclass to capture
        `SurrogateEpochMetrics` from `state.round_metrics.training_history`.

        `SurrogateEpochMetrics` fields: `epoch`, `train_loss`, `val_loss`,
        `train_spearman`, `val_spearman`, `train_mse`, `val_mse`, and
        `extra` (a dict for model-specific metrics). Use
        `epoch_metrics.to_metrics_dict()` to get a flat `dict[str, float]`
        with `None` fields omitted.

        Args:
            training_history: List of SurrogateEpochMetrics from
                state.round_metrics.training_history. May be empty if the
                surrogate does not override get_epoch_metrics().
            round_num: The round number. Available for subclasses that need it
                (e.g. to name output files); can be ignored otherwise.
        """


class TerminalStateLogger(StateLogger):
    """Logger that outputs metrics to the terminal."""

    def _log_training_history(
        self, training_history: list[SurrogateEpochMetrics], round_num: int | None = None
    ) -> None:
        """Log per-epoch metrics for a single round to the terminal. training_history is
        logged at DEBUG level to avoid cluttering the terminal output, which is typically
        reserved for round-level metrics.

        Args:
            training_history: List of SurrogateEpochMetrics from
                state.round_metrics.training_history.
            round_num: Unused; present for signature compatibility with the base class.
        """
        for epoch_metrics in training_history:
            metrics = [f"{k}: {v:.3f}" for k, v in epoch_metrics.to_metrics_dict().items()]
            logger.debug("  Epoch %d: %s", epoch_metrics.epoch, ", ".join(metrics))

    def log(
        self,
        state: State,
        round_name: str | None = None,
    ) -> None:
        """Log metrics in the task state to the terminal.

        Args:
            state: State object with metrics to log
            round_name: Name of the current round in the task, depending on the task type
                e.g. "initial_train_round", "supervised evaluation", "zero-shot evaluation",
                or the round number for design tasks.
        """
        if round_name is None:
            round_name = str(state.round_metrics.round)
        metrics = [f"{key}: {value:.3f}" for key, value in state.round_metrics.metrics.items()]
        message = "\n".join(metrics)
        logger.info(f"Round {round_name}:\n{message}")
        logger.debug(f"Training history for round {round_name}:")
        self._log_training_history(state.round_metrics.training_history)


class FileStateLogger(StateLogger):
    """Logger that saves certain components of the state to a file.
    This includes the metrics, the acquisition batch, the data splits, and the predictions.
    """

    def __init__(
        self, output_path: str | os.PathLike, upload_function: Callable[[Path], None] | None = None
    ):
        """Initialize FileStateLogger.

        Args:
            output_path: Path to the directory to save the state information to
            upload_function: Function to upload the state information to a remote location
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.upload_function = upload_function
        logger.info("Initializing FileStateLogger at %s", self.output_path)

    def _log_training_history(
        self, training_history: list[SurrogateEpochMetrics], round_num: int | None = None
    ) -> None:
        """Write per-epoch metrics for a single round to training_history/round_N.csv.

        Creates the `training_history/` subdirectory on first use. Each round
        gets its own file so column schemas never conflict across rounds.

        Args:
            training_history: List of SurrogateEpochMetrics from
                state.round_metrics.training_history.
            round_num: The round number, used as the filename suffix.

        Raises:
            ValueError: If round_num is None, since it's needed to name the output file.
        """
        if not training_history:
            return
        if round_num is None:
            raise ValueError(
                "round_num must be provided to log training history in FileStateLogger"
            )
        training_history_dir = self.output_path / "training_history"
        training_history_dir.mkdir(exist_ok=True)
        rows = [em.to_metrics_dict() for em in training_history]
        pd.DataFrame.from_records(rows).to_csv(
            training_history_dir / f"round_{round_num}.csv", index=False
        )

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to file.

        Args:
            metrics: Dictionary of metrics to log
        """
        metrics_df = pd.DataFrame.from_records([metrics])
        if (self.output_path / "metrics.csv").exists():
            saved_df = pd.read_csv(self.output_path / "metrics.csv")
            metrics_df = pd.concat([saved_df, metrics_df])
        metrics_df.to_csv(self.output_path / "metrics.csv", index=False)

    def _log_acquisition_batch(self, acq_batch: LabelledCandidates, acq_round: int) -> None:
        """Log the acquisition batch to file.

        Args:
            acq_batch: the current batch of acquired candidates
            acq_round: the current round
        """
        acq_batch.to_dataframe().to_csv(
            self.output_path / f"acq_round_{acq_round}.csv", index=False
        )

    def _log_predictions(
        self,
        predictions: Predictions,
        candidates: list[Candidate],
        targets: np.ndarray,
        round_name: str,
        problem_type: ProblemType = ProblemType.REGRESSION,
    ) -> None:
        """Log predictions to file.

        Args:
            predictions: Predictions object to log
            candidates: List of Candidate objects corresponding to the predictions
            targets: Ground truth scores corresponding to the predictions
            round_name: Name of the current round in the task, depending on the task type
                e.g. "initial_train_round", "supervised evaluation", "zero-shot evaluation",
                or the round number for design tasks.
            problem_type: ProblemType to determine how predictions are formatted.
        """
        round_name = round_name.lower().replace(" ", "_")
        predictions_df = predictions.to_dataframe(candidates, targets, problem_type=problem_type)
        predictions_df.to_csv(self.output_path / f"{round_name}_predictions.csv", index=False)

    def log(self, state: State, round_name: str | None = None) -> None:
        """Log data from the task state to the logger destination.

        Args:
            state: State object to log
            round_name: Name of the current round in the task, depending on the task type
                e.g. "initial_train_round", "supervised evaluation", "zero-shot evaluation",
                or the round number for design tasks.
        """
        if round_name is None:
            round_name = "round_" + str(state.round_metrics.round)

        self._log_metrics(state.round_metrics.metrics)
        self._log_training_history(state.round_metrics.training_history, state.round_metrics.round)

        if state.round_predictions is not None:
            self._log_predictions(
                state.round_predictions,
                state.dataset.test_dataset.candidates,
                state.dataset.test_dataset.labels,
                round_name,
                problem_type=state.problem_type,
            )

        if state.history:
            self._log_acquisition_batch(state.history[-1], state.round)

        if self.upload_function is not None:
            self.upload_function(self.output_path)
