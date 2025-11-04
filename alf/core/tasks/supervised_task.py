import logging
import time
from typing import Any, Optional

from alf.core.dataclasses.task_state import TaskState
from alf.core.oracle.oracle import Oracle
from alf.core.tasks.base_task import BaseTask
from alf.core.utils.logger import Logger

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]")
log = logging.getLogger("rich")


class SupervisedTask(BaseTask):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(task_type="Supervised", save_round_predictions=True, **kwargs)

    def run(  # type: ignore[override]
        self,
        state: TaskState,
        logger: Logger,
        oracle: Oracle,
        save_path: Optional[str] = None,
    ) -> None:
        """Run the supervised task."""
        log.info("Running supervised task ...")

        if save_path:
            state.dataset.save_splits(save_path, _verbose=True)

        t0 = time.perf_counter()
        state.surrogate.fit(
            train_data=state.dataset.train_dataset,
            val_data=state.dataset.validation_dataset,
            logger=logger,
        )
        t1 = time.perf_counter()
        state.round_metrics = {"tell_time": t1 - t0}

        state = self.evaluate(
            state=state,
            round_name="supervised evaluation",
            save_path=save_path,
            filename="supervised_predictions.csv",
        )

        logger.write(state.round_metrics, timestep=0)
        return
