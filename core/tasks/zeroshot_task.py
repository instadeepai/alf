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
from typing import Any, Optional

from core.utils.logger import Logger
from core.tasks.base_task import BaseTask
from core.dataclasses.task_state import TaskState

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]")
log = logging.getLogger("rich")


class ZeroShotTask(BaseTask):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(task_type="ZeroShot", **kwargs)

    def run(
        self,
        state: TaskState,
        logger: Logger,
        save_path: Optional[str] = None,
    ) -> None:
        """This is the multi-round design task."""

        log.info(
            f"Zero-shot evaluation on test data with {len(state.dataset.test_dataset)} sequences"
        )

        # Evaluate Surrogate
        state = self.evaluate(
            state=state,
            round_i="zero-shot evaluation",
            save_path=save_path,
            filename="zero_shot_predictions.csv",
        )

        log.info(
            "Note, the zero-shot predictions do not currently exclude any randomly initialised "
            "layers."
        )

        logger.write(state.round_metrics, timestep=0)

        return