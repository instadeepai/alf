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

from alf.core.dataclasses import TaskState
from alf.core.tasks.base_task import BaseTask
from alf.core.utils.task_state_logger import TaskStateLogger

logger = logging.getLogger("alf-core")


class ZeroShotTask(BaseTask):
    """Zero-shot evaluation task for pre-trained models.

    Evaluates a pre-trained surrogate model on the test set without any training
    or fine-tuning.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the zero-shot task.

        Args:
            **kwargs: Additional arguments passed to BaseTask (acq_batch_size,
                num_acq_rounds, save_round_predictions).
        """
        super().__init__(task_type="ZeroShot", **kwargs)

    def run(  # type: ignore[override]
        self,
        state: TaskState,
        loggers: list[TaskStateLogger],
    ) -> None:
        """Run the zero-shot evaluation task.

        Evaluates the pre-trained surrogate model on the test set without any training.
        Saves predictions and logs metrics.

        Args:
            state: Task state with dataset and pre-trained surrogate model.
            loggers: List of loggers for recording the state.
        """
        logger.info(
            f"Zero-shot evaluation on test data with {len(state.dataset.test_dataset)} sequences"
        )

        state = self.evaluate(state=state)

        logger.info(
            "Note, the zero-shot predictions do not currently exclude any randomly initialised "
            "layers."
        )

        for alf_logger in loggers:
            alf_logger.log(state, round_name="zero-shot evaluation")
