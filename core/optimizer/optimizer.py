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

import time
from typing import List, Optional, Tuple, Dict


from core.dataclasses import (
    Candidate,
    LabeledCandidates,
    TaskState,
)
from core.utils.logger import Logger
from core.optimizer.acquisition import Acquisition
from core.optimizer.search import BaseSearch


def select_top_k(candidates: LabeledCandidates, k: int) -> List[Candidate]:
    """Select the top k candidates from the list."""
    top_k_indices = candidates.labels.argsort()[::-1][:k]
    top_k_candidates = [candidates.candidates[i] for i in top_k_indices]
    top_k_labels = candidates.labels[top_k_indices]
    return LabeledCandidates(top_k_candidates, top_k_labels)


class Optimizer:
    """High level ask/tell routine"""

    def __init__(
        self,
        acquisition: Acquisition,
        search: BaseSearch,
    ) -> None:
        """Initialize the optimizer."""
        self.acq_fn = acquisition
        self.search_fn = search

    def ask(
        self,
        state: TaskState,
    ) -> Tuple[List[Candidate], TaskState]:
        """Ask the optimizer to propose the next batch of candidates through the search and acquisition functions."""

        t0 = time.perf_counter()

        # During the first round, we use the training dataset as the acquired candidates.
        # During the subsequent rounds, we use the search and acquisition functions to acquire candidates.
        if state.step_count == 0 and len(state.dataset.train_dataset) > 0:
            acquired_candidates = state.dataset.train_dataset.candidates
        else:
            search_candidates = self.search_fn(state)
            acquisition_candidates = self.acq_fn(search_candidates, best_f=state.dataset.train_dataset.labels.max())
            acquired_candidates = select_top_k(acquisition_candidates, state.acq_batch_size).candidates

        t1 = time.perf_counter()
        state.round_metrics.update({"ask_time": t1 - t0})
        return acquired_candidates, state

    def tell(
        self,
        state: TaskState,
        logger: Optional[Logger] = None,
    ) -> TaskState:
        """
        Train the surrogate model on the new train/val datasets.

        Args:
            state: the optimizer state
            dataset: the dataset
            surrogate: the surrogate model
            logger: the logger
        """
        t0 = time.perf_counter()
        if state.surrogate:
            state.surrogate.fit(
                train_data=state.dataset.train_dataset, 
                val_data=state.dataset.validation_dataset, 
                logger=logger
            )
        t1 = time.perf_counter()

        state.round_metrics.update({"tell_time": t1 - t0})
        state.round_metrics.update(self.get_metrics(state))

        return state

    def get_metrics(
        self,
        state: TaskState,
    ) -> Dict[str, float]:
        """Update the state with the metrics from the acquired candidates, surrogate, search, and acquisition functions."""

        acquired_candidates = state.history[-1]
        metrics = {
             # Metrics on the acquired candidates during the current round
            "acquired_candidates/round_mean": acquired_candidates.labels.mean(),
            "acquired_candidates/round_max": acquired_candidates.labels.max(),
            "acquired_candidates/round_min": acquired_candidates.labels.min(),
        }

        if state.surrogate:
            metrics.update(state.surrogate.get_training_summary_metrics())

        metrics.update(self.search_fn.get_metrics(state))
        metrics.update(self.acq_fn.get_metrics(state))

        return metrics
