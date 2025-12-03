from typing import Any, List, Union

import numpy as np
from alf_core import BaseModel, Candidate, LabeledCandidates, Predictions


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
            Any: The featurised input data.
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
            Predictions: The predictions for the candidate points.
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

    def get_training_summary_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Get the training summary metrics.

        Returns:
            dict[str, Union[float, int, np.number]]: The training summary metrics.
        """
        return {}
