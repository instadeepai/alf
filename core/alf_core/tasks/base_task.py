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


import abc
from typing import Any

from alf_core.dataclasses import Results, State
from alf_core.dataset.base_dataset import BaseDataset
from alf_core.surrogate.surrogate import Surrogate


class BaseTask(abc.ABC):
    """Base class for all tasks.

    Provides common functionality for different types of experiments/tasks,
    including setup, evaluation, and saving predictions.
    """

    def __init__(
        self,
        task_type: str,
        acq_batch_size: int = 0,
        num_acq_rounds: int = 0,
        save_round_predictions: bool = False,
    ) -> None:
        """Initialize the base task.

        Args:
            task_type: Type identifier for the task ("Design", "Supervised", "ZeroShot").
            acq_batch_size: Number of candidates to acquire per round. Defaults to 0.
            num_acq_rounds: Total number of acquisition rounds. Defaults to 0.
            save_round_predictions: Whether to save predictions at each round.
                Defaults to False.
        """
        self.task_type = task_type
        self.acq_batch_size = acq_batch_size
        self.num_acq_rounds = num_acq_rounds
        self.save_round_predictions = save_round_predictions

    def setup(self, dataset: BaseDataset, surrogate: Surrogate) -> State:
        """Setup the task with dataset and surrogate model.

        Args:
            dataset: The dataset containing train/validation/test splits.
            surrogate: The surrogate model to use for predictions.

        Returns:
            Initialized task state with the provided dataset and surrogate.
        """
        return State(
            dataset=dataset,
            surrogate=surrogate,
            acq_batch_size=self.acq_batch_size,
        )

    @abc.abstractmethod
    def run(self, **kwargs: Any) -> None:
        """Run the task.

        This method must be implemented by subclasses to define the specific
        task execution logic.

        Args:
            **kwargs: Task-specific keyword arguments.
        """
        pass

    def evaluate(
        self,
        state: State,
    ) -> State:
        """Evaluate the surrogate model on the test dataset and return the updated state.

        Computes predictions, metrics on the test set. Optionally saves
        predictions if configured. Updates the task state with computed metrics.

        Args:
            state: Current task state containing dataset and surrogate.

        Returns:
            Updated task state with evaluation metrics.

        Raises:
            AssertionError: If save_round_predictions is True but filename is empty.
        """
        if len(state.dataset.test_dataset) > 0:
            predictions = state.surrogate.predict(state.dataset.test_dataset.candidates)
            results = Results(predictions=predictions, targets=state.dataset.test_dataset.labels)

            state.round_predictions = predictions
            state.round_metrics.metrics.update({
                f"surrogate/test_{key}": value for key, value in results.metrics.items()
            })

        dataset_metrics = state.dataset.get_metrics()
        state.round_metrics.metrics.update({f"dataset/{k}": v for k, v in dataset_metrics.items()})
        return state
