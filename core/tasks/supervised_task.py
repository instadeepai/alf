from core.tasks.base_task import BaseTask
from core.dataclasses.task_state import TaskState
from core.datasets.base_dataset import BaseDataset
from core.surrogate.surrogate import Surrogate
from core.optimizer.optimizer import Optimizer
from core.oracle.oracle import Oracle
from core.utils.logger import Logger
import logging
import time
from typing import Any, Optional

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]")
log = logging.getLogger("rich")

class SupervisedTask(BaseTask):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(task_type="Supervised", save_round_predictions=True, **kwargs)

    def run(
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
        state.surrogate.fit(train_data=state.dataset.train_dataset, val_data=state.dataset.validation_dataset, logger=logger)
        t1 = time.perf_counter()
        state.round_metrics = {"tell_time": t1 - t0}

        # Evaluate Surrogate
        state = self.evaluate(
            state=state,
            round_i="supervised evaluation",
            save_path=save_path,
            filename="supervised_predictions.csv",
        )

        logger.write(state.round_metrics, timestep=0)
        return 