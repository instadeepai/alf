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

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Union

import numpy as np
import torch.optim as optim
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions
from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics
from torch.utils.data import DataLoader

from alf_tools.models.utils import get_device

if TYPE_CHECKING:
    from chemprop.models import MPNN

logger = logging.getLogger("alf-tools")


@dataclass
class ChempropModelConfig:
    """Architecture hyperparameters for the Chemprop MPNN.

    Args:
        hidden_size: Dimension of message-passing hidden layers.
        depth: Number of message-passing steps.
        ffn_num_layers: Number of feed-forward layers after aggregation.
        dropout: Dropout rate applied in the FFN.
        aggregation: Graph-level aggregation function.
    """

    hidden_size: int = 300
    depth: int = 3
    ffn_num_layers: int = 2
    dropout: float = 0.0
    aggregation: Literal["mean", "sum", "norm"] = "mean"


@dataclass
class ChempropTrainConfig:
    """Training hyperparameters for the Chemprop MPNN.

    Args:
        learning_rate: Optimizer learning rate.
        batch_size: Number of molecules per batch.
        num_epochs: Training epochs.
        optimizer: Optimizer type.
        weight_init: Weight initialisation strategy; None uses Chemprop's default.
        seed: Random seed for reproducibility; None means no seeding.
    """

    learning_rate: float = 1e-3
    batch_size: int = 50
    num_epochs: int = 50
    optimizer: Literal["adam", "sgd", "adamw"] = "adam"
    weight_init: Literal["default", "xavier_uniform", "kaiming_normal"] | None = None
    seed: int | None = None


def _get_aggregations() -> dict[str, type]:
    """Return mapping of aggregation name to Chemprop aggregation class.

    Imports are deferred to avoid loading chemprop at module level.

    Returns:
        Dictionary mapping aggregation name strings to aggregation classes.
    """
    from chemprop.nn import MeanAggregation, NormAggregation, SumAggregation  # noqa: PLC0415

    return {
        "mean": MeanAggregation,
        "sum": SumAggregation,
        "norm": NormAggregation,
    }


# Callable alias for deferred aggregation lookup in Task 3.
# Use: _AGGREGATIONS()[cfg.aggregation]() to avoid top-level chemprop import.
_AGGREGATIONS = _get_aggregations  # noqa: N816


_OPTIMIZERS: dict[str, type] = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adamw": optim.AdamW,
}


class ChempropModel(BaseModel):
    """Mean-only MPNN surrogate model using Chemprop v2.x as backbone.

    Accepts SMILES strings via Candidate.data and returns scalar fitness predictions.
    Model is initialised lazily on the first train() call.
    """

    def __init__(
        self,
        name: str = "chemprop_model",
        model_config: ChempropModelConfig | None = None,
        train_config: ChempropTrainConfig | None = None,
        device: str | None = None,
    ) -> None:
        """Initialise ChempropModel.

        Args:
            name: Name identifier for this model.
            model_config: Architecture hyperparameters.
            train_config: Training hyperparameters.
            device: Device to run on ('cpu', 'cuda', or None for auto-detect).
        """
        self.name = name
        self.model_config = model_config or ChempropModelConfig()
        self.train_config = train_config or ChempropTrainConfig()
        self.device = get_device(device)
        self._model: "MPNN | None" = None
        self._epoch_metrics: list[SurrogateEpochMetrics] = []
        self._training_metrics: dict[str, Union[float, int, np.number]] = {}

    def featurise(self, inputs: Union[LabelledCandidates, list[Candidate]]) -> list[str]:
        """Extract SMILES strings from inputs.

        Args:
            inputs: LabelledCandidates or list of Candidate objects whose .data
                fields hold SMILES strings.

        Returns:
            List of SMILES strings.
        """
        if isinstance(inputs, LabelledCandidates):
            return [str(c.data) for c in inputs.candidates]
        return [str(c.data) for c in inputs]

    def _build_dataloader(
        self,
        smiles: list[str],
        labels: np.ndarray | None = None,
        shuffle: bool = False,
    ) -> DataLoader:
        """Build a Chemprop DataLoader from SMILES and optional labels.

        Args:
            smiles: SMILES strings, one per molecule.
            labels: Scalar labels; None for inference.
            shuffle: Whether to shuffle.

        Returns:
            Chemprop DataLoader.
        """
        raise NotImplementedError  # completed in Task 3

    def _init_model(self) -> None:
        """Instantiate the Chemprop MPNN and move it to device.

        Applies weight initialisation if configured by train_config.weight_init.
        Use _AGGREGATIONS()[cfg.aggregation]() to retrieve the aggregation class.
        """
        raise NotImplementedError  # completed in Task 3

    def train(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates | None = None,
    ) -> None:
        """Train the MPNN with MSE loss.

        Args:
            train_data: Training molecules and labels.
            val_data: Optional validation molecules and labels.
        """
        pass  # completed in Task 3

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Predict fitness means for a list of candidates.

        Args:
            candidate_points: Candidates to predict.

        Raises:
            RuntimeError: If called before train().
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        raise NotImplementedError  # completed in Task 3

    def sample(self, condition: object | None = None) -> list[Candidate]:
        """Not implemented for mean-only MPNN."""
        raise NotImplementedError("Sampling is not implemented for ChempropModel.")

    def get_epoch_metrics(self) -> list[SurrogateEpochMetrics]:
        """Return per-epoch metrics from the most recent train() call.

        Returns:
            List of SurrogateEpochMetrics, one per epoch.
        """
        return self._epoch_metrics

    def get_training_summary_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Return summary metrics from the most recent train() call.

        Returns:
            Dictionary of final train/val losses and Spearman correlations.
        """
        return self._training_metrics
