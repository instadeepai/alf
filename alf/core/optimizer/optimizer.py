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
from typing import Dict, Tuple

from alf.core.dataclasses import (
    Candidate,
    LabeledCandidates,
    TaskState,
)
from alf.core.optimizer.acquisition_function import AcquisitionFunction
from alf.core.optimizer.search import BaseSearch
from alf.core.utils.logger import Logger


def select_top_k(candidates: LabeledCandidates, k: int) -> LabeledCandidates:
    """Select the top k candidates based on their label values.

    Args:
        candidates: LabeledCandidates to select from.
        k: Number of top candidates to select.

    Returns:
        LabeledCandidates: New LabeledCandidates object containing the top k candidates
            sorted by label values (highest first).
    """
    top_k_indices = candidates.labels.argsort()[::-1][:k]
    top_k_candidates = [candidates.candidates[i] for i in top_k_indices]
    top_k_labels = candidates.labels[top_k_indices]
    return LabeledCandidates(top_k_candidates, top_k_labels)


class Optimizer:
    """Implements the ask/tell interface for active learning.

    Coordinates search and acquisition functions to propose candidates and update
    the surrogate model iteratively.
    """

    def __init__(
        self,
        acquisition_fn: AcquisitionFunction,
        search_fn: BaseSearch,
    ) -> None:
        """Initialize the optimizer.

        Args:
            acquisition_fn: Acquisition function for scoring candidates.
            search_fn: Search function for generating candidate pools.
        """
        self.acquisition_fn = acquisition_fn
        self.search_fn = search_fn

    def ask(
        self,
        state: TaskState,
    ) -> Tuple[list[Candidate], TaskState]:
        """Propose the next batch of candidates to evaluate.

        Uses the search function to generate a candidate pool, then the acquisition
        function to score and select the top candidates. In the first round, uses
        the training dataset if available.

        Args:
            state: Current task state.

        Returns:
            Tuple[list[Candidate], TaskState]: A tuple containing:
                - list[Candidate]: The proposed candidates to evaluate
                - TaskState: Updated state with ask_time metric
        """
        t0 = time.perf_counter()

        # During the first round, we use the training dataset as the acquired candidates.
        # During the subsequent rounds, we use the search and acquisition functions to acquire candidates.
        if state.round == 0 and len(state.dataset.train_dataset) > 0:
            acquired_candidates = state.dataset.train_dataset.candidates
        else:
            search_candidates = self.search_fn(state)
            acquisition_candidates = self.acquisition_fn(search_candidates, state)
            acquired_candidates = select_top_k(
                acquisition_candidates, state.acq_batch_size
            ).candidates

        t1 = time.perf_counter()
        state.round_metrics.update({"ask_time": t1 - t0})
        return acquired_candidates, state

    def tell(
        self,
        state: TaskState,
        logger: Logger | None = None,
    ) -> TaskState:
        """Update the surrogate model with newly acquired data.

        Trains the surrogate model on the updated training and validation datasets,
        then computes and updates metrics.

        Args:
            state: Current task state with updated dataset.
            logger: Optional logger for recording training metrics.

        Returns:
            TaskState: Updated state with tell_time and optimizer metrics.
        """
        t0 = time.perf_counter()
        if state.surrogate:
            state.surrogate.fit(
                train_data=state.dataset.train_dataset,
                val_data=state.dataset.validation_dataset,
                logger=logger,
            )
        t1 = time.perf_counter()

        state.round_metrics.update({"tell_time": t1 - t0})
        state.round_metrics.update(self.get_metrics(state))

        return state

    def get_metrics(
        self,
        state: TaskState,
    ) -> Dict[str, float]:
        """Collect metrics from acquired candidates, surrogate, and search functions.

        Args:
            state: Current task state.

        Returns:
            Dict[str, float]: Dictionary of metric names to values, including:
                - Metrics on acquired candidates (mean, max, min)
                - Surrogate training metrics (if available)
                - Search function metrics
        """
        acquired_candidates = state.history[-1]
        metrics = {
            # Metrics on the acquired candidates during the current round
            "acquired_candidates/round_mean": acquired_candidates.labels.mean(),
            "acquired_candidates/round_max": acquired_candidates.labels.max(),
            "acquired_candidates/round_min": acquired_candidates.labels.min(),
        }

        if state.surrogate:
            surrogate_metrics = state.surrogate.get_training_summary_metrics()
            metrics.update({f"surrogate/{k}": v for k, v in surrogate_metrics.items()})

        metrics.update(self.search_fn.get_metrics(state))

        return metrics
