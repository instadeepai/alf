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

import logging
from typing import Any

import numpy as np
from alf_core.dataclasses import State
from alf_core.dataclasses.round_metrics import RoundMetrics
from alf_core.optimizer.optimizer import Optimizer
from alf_core.oracle.oracle import Oracle
from alf_core.tasks.base_task import BaseTask
from alf_core.utils.enums import ProblemType
from alf_core.utils.metrics.aggregate import compute_aggregate_metrics
from alf_core.utils.metrics.regression import top_k_mean
from alf_core.utils.state_logger import StateLogger

logger = logging.getLogger("alf-core")


class DesignTask(BaseTask):
    """Multi-round design task for iteratively optimising the surrogate model.
    Performs multiple rounds of candidate acquisition, evaluation, and model training.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the design task.

        Args:
            **kwargs: Additional arguments passed to BaseTask (acq_batch_size,
                num_acq_rounds, save_round_predictions).
        """
        super().__init__(task_type="Design", **kwargs)

    def run_initial_train_round(self, state: State, state_loggers: list[StateLogger]) -> State:
        """Run the initial train round on the train and validation sets.

        Args:
            state: Task state with dataset and surrogate.
            state_loggers: List of StateLogger for recording the state.

        Returns:
            Updated state with surrogate fine-tuned on the train and validation sets.
        """
        logger.info("Running initial round of surrogate model fine-tuning on the train dataset ...")
        # Construct RoundMetrics before fit() so state is always typed, even on failure
        state.round_metrics = RoundMetrics(round=0)
        state.metrics_history.append(state.round_metrics)
        epoch_metrics = state.surrogate.fit(
            train_data=state.dataset.train_dataset,
            val_data=state.dataset.validation_dataset,
        )
        state.round_metrics.training_history = epoch_metrics
        state = self.evaluate(state=state)
        for state_logger in state_loggers:
            state_logger.log(state, round_name="initial_train_round")
        return state

    def run(  # type: ignore[override]
        self,
        state: State,
        state_loggers: list[StateLogger],
        optimizer: Optimizer,
        oracle: Oracle,
    ) -> None:
        """Run the multi-round design task.

        Executes multiple rounds of active learning:
        1. Optimizer proposes candidates (ask)
        2. Oracle evaluates candidates
        3. Surrogate model is updated with new data (tell)
        4. Model is evaluated on test set
        5. Metrics are logged

        The loop continues for num_acq_rounds or until termination conditions are met.

        After all rounds complete, runs all aggregate metrics (see
        `alf_core.utils.metrics.aggregate`) over a per-round sample-efficiency
        curve and logs the result under the round name `experiment_summary`.
        The summary is skipped when no aggregate metric could be computed.

        Args:
            state: Initial task state with dataset and surrogate.
            state_loggers: List of StateLogger for recording the state.
            optimizer: Optimizer for candidate acquisition.
            oracle: Oracle for evaluating candidate labels.
        """
        logger.info("Multi-round Design Task: %d Rounds", self.num_acq_rounds)

        # If the train data is provided, run an initial round of fine-tuning the surrogate
        # model on the training dataset.
        if len(state.dataset.train_dataset) > 0:
            state = self.run_initial_train_round(state, state_loggers)

        for round_i in range(1, self.num_acq_rounds + 1):
            state.round_metrics = RoundMetrics(round=round_i)
            state.metrics_history.append(state.round_metrics)
            acquired_candidates, state = optimizer.ask(state)
            labelled_candidates, state = oracle.evaluate(acquired_candidates, state)
            state.update(labelled_candidates)  # increments state.round to round_i + 1
            state = optimizer.tell(state=state)  # populates round_metrics.training_history
            state = self.evaluate(state=state)
            for state_logger in state_loggers:
                state_logger.log(state)

        is_regression = state.problem_type == ProblemType.REGRESSION
        best_value = float(state.dataset.raw_dataset.labels.max()) if is_regression else 1.0
        experiment_metrics = compute_aggregate_metrics(
            np.array(self._sample_efficiency_curve(state)), best_value
        )
        if experiment_metrics:
            state.round_metrics = RoundMetrics(
                round=self.num_acq_rounds, metrics=experiment_metrics
            )
            state.round_predictions = None
            for state_logger in state_loggers:
                state_logger.log(state, round_name="experiment_summary")

        return

    def _sample_efficiency_curve(self, state: State) -> list[float]:
        """Build the per-round sample-efficiency curve fed to the aggregate metrics.

        For regression the curve is the top-k mean of all candidates acquired
        up to each round (it should rise as good candidates accumulate); for
        classification it is the per-round test-set accuracy.

        Args:
            state: Final task state after all acquisition rounds.

        Returns:
            One value per acquisition round that produced a valid value.
        """
        if state.problem_type == ProblemType.REGRESSION:
            curve = []
            for round_i in range(1, len(state.history) + 1):
                labels = np.concatenate([c.labels for c in state.history[:round_i]])
                # top_k_mean returns a single dynamically-keyed entry (e.g.
                # "top_10_mean"); take its value for the curve.
                curve.append(float(next(iter(top_k_mean(labels, None, labels).values()))))
            return curve
        return [
            float(metrics.metrics["surrogate/test_accuracy"])
            for metrics in state.metrics_history
            if metrics.round > 0 and "surrogate/test_accuracy" in metrics.metrics
        ]
