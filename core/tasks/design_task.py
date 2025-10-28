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
from core.optimizer.optimizer import Optimizer
from core.oracle.oracle import Oracle

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]")
log = logging.getLogger("rich")


class DesignTask(BaseTask):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(task_type="Design", **kwargs)

    def run(
        self,
        state: TaskState,
        logger: Logger,
        optimizer: Optimizer,
        oracle: Oracle,
        save_path: Optional[str] = None,
    ) -> None:
        """This is the multi-round design task."""

        log.info(f"Multi-round Design Task: {state.num_acq_rounds} Rounds")

        if save_path:
            state.dataset.save_splits(save_path, _verbose=True)

        for round_i in range(state.num_acq_rounds + 1):
            state.round_metrics = {"round": round_i}
            acquired_candidates, state =  optimizer.ask(state)
            labeled_candidates, state = oracle.evaluate(acquired_candidates, state)
            state.update(labeled_candidates)
            state = optimizer.tell(state=state, logger=logger)

            state = self.evaluate(
                state=state,
                round_i=round_i,
                save_path=save_path,
                filename=f"round_{round_i}_predictions.csv",
            )
            logger.write(state.round_metrics, timestep=round_i)

            if state.should_terminate():
                break
        
        return
