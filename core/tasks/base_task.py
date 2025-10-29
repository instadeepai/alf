import abc
from typing import Optional
from core.dataclasses import Results
from core.dataclasses.task_state import TaskState
import logging
from core.dataset.base_dataset import BaseDataset
from core.surrogate.surrogate import Surrogate

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]")
log = logging.getLogger("rich")


class BaseTask(abc.ABC):
    """Base class for all tasks."""

    def __init__(self, task_type: str, acq_batch_size: int = 0, num_acq_rounds: int = 0, save_round_predictions: bool = False) -> None:
        self.task_type = task_type
        self.acq_batch_size = acq_batch_size
        self.num_acq_rounds = num_acq_rounds
        self.save_round_predictions = save_round_predictions

    def setup(self, dataset: BaseDataset, surrogate: Surrogate) -> TaskState:
        """Setup the task."""
        return TaskState(
            dataset=dataset,
            surrogate=surrogate,
            acq_batch_size=self.acq_batch_size,
            num_acq_rounds=self.num_acq_rounds,
            save_round_predictions=self.save_round_predictions,
        )

    @abc.abstractmethod
    def run(self) -> None:
        """Run the task."""
        pass

    def evaluate(
        self,  
        state: TaskState, 
        round_name: int | str, 
        save_path: Optional[str], 
        filename: str,
    ) -> TaskState:
        """Evaluate the surrogate model and return the updated state."""

        if len(state.dataset.test_dataset) > 0 and state.surrogate:
            predictions = state.surrogate.predict(state.dataset.test_dataset.candidates)
            results = Results(predictions=predictions, targets=state.dataset.test_dataset.labels)

            if self.save_round_predictions and save_path:
                assert filename, "Filename must be provided to save predictions"
                predictions.save(
                    save_path,
                    state.dataset.test_dataset.candidates,
                    state.dataset.test_dataset.labels,
                    filename=filename,
                )

            state.round_metrics.update({f"plots/test/{key}": value for key, value in results.figures.items()})
            state.round_metrics.update({f"surrogate/test_{key}": value for key, value in results.metrics.items()})

        dataset_metrics = state.dataset.get_metrics()
        state.round_metrics.update(
            {f"dataset/{k}": v for k, v in dataset_metrics.items()}
        )

        state.print_metrics(round_name)
        state.save(save_path, _verbose=True)
        return state
