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
import time
from typing import Any

from alf_core.dataclasses import State
from alf_core.tasks.base_task import BaseTask
from alf_core.utils.state_logger import StateLogger

logger = logging.getLogger("alf-core")


class SupervisedTask(BaseTask):
    """Supervised learning task for training and evaluating models on fixed splits.
    Trains the surrogate model on the training set and evaluates it on the test set.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the supervised task.

        Args:
            **kwargs: Additional arguments passed to BaseTask. Note that
                save_round_predictions is automatically set to True.
        """
        super().__init__(task_type="Supervised", save_round_predictions=True, **kwargs)

    def run(  # type: ignore[override]
        self,
        state: State,
        state_loggers: list[StateLogger],
    ) -> None:
        """Run the supervised learning task.

        Trains the surrogate model on the training and validation sets, then evaluates
        it on the test set. Saves predictions and logs metrics.

        Args:
            state: Task state with dataset and surrogate model.
            state_loggers: List of StateLogger for recording the state.
        """
        logger.info("Running supervised task ...")

        t0 = time.perf_counter()
        state.surrogate.fit(
            train_data=state.dataset.train_dataset,
            val_data=state.dataset.validation_dataset,
        )
        t1 = time.perf_counter()
        state.round_metrics = {"tell_time": t1 - t0}

        state = self.evaluate(state=state)
        for state_logger in state_loggers:
            state_logger.log(state, round_name="supervised evaluation")

        return
