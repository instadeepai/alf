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

"""Unit tests for the State dataclass."""

import numpy as np
from alf_core.dataclasses import Candidate, LabelledCandidates, State
from alf_core.dataclasses.candidate import Modality
from alf_core.dataset.base_dataset import BaseDataset, BaseDatasetConfig
from alf_core.utils.enums import ProblemType


def _make_minimal_dataset(problem_type: ProblemType) -> BaseDataset:
    """Create a minimal dataset with appropriate candidates for the given problem type.

    Returns:
        A set-up BaseDataset instance configured with the given problem type.
    """
    if problem_type == ProblemType.BINARY:
        candidates = [
            Candidate(data="ACGT", modality=Modality.SEQUENCE),
            Candidate(data="TGCA", modality=Modality.SEQUENCE),
        ]
        labels = np.array([0.0, 1.0])
    elif problem_type == ProblemType.MULTICLASS:
        candidates = [
            Candidate(data="ACGT", modality=Modality.SEQUENCE),
            Candidate(data="TGCA", modality=Modality.SEQUENCE),
            Candidate(data="AAAA", modality=Modality.SEQUENCE),
        ]
        labels = np.array([0.0, 1.0, 2.0])
    else:
        candidates = [Candidate(data="ACGT", modality=Modality.SEQUENCE)]
        labels = np.array([0.0])

    class MinimalDataset(BaseDataset):
        def load_dataset(self):
            """Load a minimal dataset.

            Returns:
                A LabelledCandidates with sequence candidates.
            """
            return LabelledCandidates(
                candidates=candidates,
                labels=labels,
            )

    config = BaseDatasetConfig(
        name="test",
        modality=Modality.SEQUENCE,
        seed=0,
        train_ratio=0.5,
        validation_frac=0.0,
        test_ratio=0.5,
        problem_type=problem_type,
    )
    dataset = MinimalDataset(config)
    dataset.setup()
    return dataset


class TestStateProblemTypeProperty:
    """State.problem_type delegates to dataset.config.problem_type."""

    def test_binary_problem_type(self, dummy_surrogate):
        """Test that State.problem_type returns BINARY when dataset is configured as such."""
        dataset = _make_minimal_dataset(ProblemType.BINARY)
        state = State(dataset=dataset, surrogate=dummy_surrogate)
        assert state.problem_type == ProblemType.BINARY

    def test_regression_problem_type(self, dummy_surrogate):
        """Test that State.problem_type returns REGRESSION when dataset is configured as such."""
        dataset = _make_minimal_dataset(ProblemType.REGRESSION)
        state = State(dataset=dataset, surrogate=dummy_surrogate)
        assert state.problem_type == ProblemType.REGRESSION

    def test_multiclass_problem_type(self, dummy_surrogate):
        """Test that State.problem_type returns MULTICLASS when dataset is configured as such."""
        dataset = _make_minimal_dataset(ProblemType.MULTICLASS)
        state = State(dataset=dataset, surrogate=dummy_surrogate)
        assert state.problem_type == ProblemType.MULTICLASS
