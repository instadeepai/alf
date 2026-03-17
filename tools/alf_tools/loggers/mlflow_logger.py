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

from pathlib import Path

import mlflow
from alf_core.dataclasses import State
from alf_core.utils.state_logger import StateLogger


class MLflowLogger(StateLogger):
    """Logs round-level and per-epoch training metrics to MLflow.

    Uses two distinct namespaces to keep step axes independent:
    - ``training/<metric>`` at ``step=epoch`` for per-epoch training curves.
    - ``round/<metric>`` at ``step=round`` for per-round evaluation metrics.

    Starts and manages its own MLflow run on construction.
    """

    def __init__(self, experiment_name: str, output_path: str | None = None) -> None:
        """Initialize the MLflowLogger.

        Args:
            experiment_name: Name of the MLflow experiment to log to. Created
                automatically if it does not exist.
            output_path: Optional local directory for storing MLflow logs. A
                ``mlflow.db`` SQLite file will be created inside this directory.
                If None, uses the default MLflow tracking URI.
        """
        if output_path:
            db_path = Path(output_path).resolve() / "mlflow.db"
            mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run()

    def __del__(self) -> None:
        """End the active MLflow run when this logger is garbage-collected."""
        if self._run and mlflow.active_run():
            mlflow.end_run()

    def log(self, state: State, round_name: str | None = None) -> None:
        """Log training history and round metrics to MLflow.

        Args:
            state: Task state containing ``round_metrics`` with ``training_history``
                and scalar ``metrics``.
            round_name: Unused by MLflow (step axes provide the x-axis).
        """
        self._log_training_history_per_round(state)
        self._log_round_metrics(state)

    def _log_round_metrics(self, state: State) -> None:
        """Write each scalar in round_metrics.metrics as an MLflow metric.

        Tag: ``round/<key>``, step: ``state.round_metrics.round``.
        """
        step = state.round_metrics.round
        mlflow.log_metrics(
            {f"round/{k}": v for k, v in state.round_metrics.metrics.items()},
            step=step,
        )

    def _log_training_history_per_round(self, state: State) -> None:
        """Write per-epoch metrics from training_history to MLflow.

        Tag: ``round/<round>/training_history/<key>``, step: ``epoch_metrics.epoch``.
        Each round's training history is stored under its own namespace so that
        data from different rounds never overwrites each other.
        """
        for em in state.round_metrics.training_history:
            mlflow.log_metrics(
                {
                    f"round/{state.round_metrics.round}/training_history/{k}": v
                    for k, v in em.to_metrics_dict().items()
                },
                step=em.epoch,
            )
