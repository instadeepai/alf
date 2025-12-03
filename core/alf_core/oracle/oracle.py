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
from typing import Union

import numpy as np
from alf_core.dataclasses import Candidate, LabeledCandidates, TaskState
from alf_core.dataset.base_dataset import BaseDataset
from alf_core.model.base_model import BaseModel


class Oracle:
    """Oracle model is used to evaluate new candidates proposed by the search/optimiser process.
    For offline optimization tasks, the oracle is the dataset.
    For online optimization tasks, the oracle is a model.
    """

    def __init__(self, scorer: BaseModel | BaseDataset) -> None:
        """Initialize the oracle with a model or dataset.

        Args:
            scorer: Either a BaseModel (for online evaluation) or BaseDataset
                (for offline evaluation from a dataset).
        """
        self.scorer: BaseModel | BaseDataset = scorer

    def evaluate(
        self, candidates: list[Candidate], state: TaskState
    ) -> tuple[LabeledCandidates, TaskState]:
        """Evaluate candidates and return their labels.

        Args:
            candidates: List of Candidate objects to evaluate.
            state: Current task state (updated with evaluation time).

        Returns:
            tuple[LabeledCandidates, TaskState]: A tuple containing:
                - LabeledCandidates: Candidates paired with their evaluated labels
                - TaskState: Updated state with oracle_time metric
        """
        t0 = time.perf_counter()
        if isinstance(self.scorer, BaseDataset):
            evaluated_candidates = self.scorer.query(candidates)
        else:
            evaluated_candidates = LabeledCandidates(
                candidates=candidates, labels=self.scorer.predict(candidates).means
            )
        t1 = time.perf_counter()
        state.round_metrics.update({"oracle_time": t1 - t0})
        return evaluated_candidates, state

    def get_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Get metrics from the underlying scorer if available.

        Returns:
            dict[str, Union[float, int, np.number]]: Dictionary of metric names to values.
                Returns empty dict if the scorer doesn't provide metrics.
        """
        if hasattr(self.scorer, "get_metrics"):
            return self.scorer.get_metrics()
        return {}
