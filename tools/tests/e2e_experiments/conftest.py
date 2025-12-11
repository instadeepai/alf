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

from typing import Any, List, Union

import numpy as np
import pytest
from alf_core import BaseModel, Candidate, LabeledCandidates, Predictions


@pytest.fixture
def random_model():
    """Fixture to create a random model for testing.

    Returns:
        A random model.
    """
    return RandomModel()


class RandomModel(BaseModel):
    """Random model that generates random predictions for testing."""

    def __init__(self, name: str = "random_model", seed: int = 42):
        """Initialize the random model.

        Args:
            name: The name of the model.
            seed: The random seed.
        """
        self.name = name
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def featurise(self, inputs: Union[LabeledCandidates, List[Candidate]]) -> Any:
        """Featurise the input data, this is no-op for this model.

        Args:
            inputs: The input data to featurise.

        Returns:
            The featurised input data.
        """
        pass

    def train(
        self,
        train_data: LabeledCandidates,
        val_data: LabeledCandidates,
    ) -> None:
        """Train the model, this is no-op for this model.

        Args:
            train_data: The training data.
            val_data: The validation data.
        """
        pass

    def predict(self, candidate_points: List[Candidate]) -> Predictions:
        """Predict the labels for the candidate points.

        Args:
            candidate_points: The candidate points to predict the labels for.

        Returns:
            The predictions for the candidate points.
        """
        labels = self.rng.randn(len(candidate_points))
        return Predictions(means=labels)

    def sample(self, *args: Any, **kwargs: Any) -> List[Candidate]:
        """Sample candidate points from the model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: Sampling is not implemented for this model.
        """
        raise NotImplementedError("Sampling is not implemented for this model.")
