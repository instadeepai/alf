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

from alf_core.dataclasses import TaskState
from alf_core.optimizer.optimizer import Optimizer
from alf_core.oracle.oracle import Oracle
from alf_core.tasks.base_task import BaseTask
from alf_core.utils.task_state_logger import TaskStateLogger

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

    def run_initial_train_round(
        self, state: TaskState, state_loggers: list[TaskStateLogger]
    ) -> TaskState:
        """Run the initial train round on the train and validation sets.

        Args:
            state: Task state with dataset and surrogate.
            state_loggers: List of TaskStateLogger for recording the state.

        Returns:
            Updated state with surrogate fine-tuned on the train and validation sets.
        """
        logger.info("Running initial round of surrogate model fine-tuning on the train dataset ...")
        state.surrogate.fit(
            train_data=state.dataset.train_dataset,
            val_data=state.dataset.validation_dataset,
        )
        state.round_metrics = {"round": 0}
        self.evaluate(state=state)
        for state_logger in state_loggers:
            state_logger.log(state, round_name="initial_train_round")
        return state

    def run(  # type: ignore[override]
        self,
        state: TaskState,
        task_state_loggers: list[TaskStateLogger],
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

        Args:
            state: Initial task state with dataset and surrogate.
            task_state_loggers: List of TaskStateLogger for recording the state.
            optimizer: Optimizer for candidate acquisition.
            oracle: Oracle for evaluating candidate labels.
        """
        logger.info("Multi-round Design Task: %d Rounds", self.num_acq_rounds)

        # If the train data is provided, run an initial round of fine-tuning the surrogate
        # model on the training dataset.
        if len(state.dataset.train_dataset) > 0:
            state = self.run_initial_train_round(state, task_state_loggers)

        for round_i in range(1, self.num_acq_rounds + 1):
            state.round_metrics = {"round": round_i}
            acquired_candidates, state = optimizer.ask(state)
            labeled_candidates, state = oracle.evaluate(acquired_candidates, state)
            state.update(labeled_candidates)
            state = optimizer.tell(state=state)

            state = self.evaluate(state=state)
            for task_state_logger in task_state_loggers:
                task_state_logger.log(state)
            state.check_termination()

        return
