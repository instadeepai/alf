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
from typing import Any, Literal, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions, Results
from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics
from torch.utils.data import DataLoader

from alf_tools.models.utils import get_device

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
        self._model: nn.Module | None = None
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
        from chemprop.data import (  # noqa: PLC0415
            MoleculeDatapoint,
            MoleculeDataset,
            build_dataloader,
        )

        if labels is not None:
            datapoints = [
                MoleculeDatapoint.from_smi(smi, y=np.array([label]))
                for smi, label in zip(smiles, labels)
            ]
        else:
            datapoints = [MoleculeDatapoint.from_smi(smi) for smi in smiles]

        dataset = MoleculeDataset(datapoints)
        return build_dataloader(dataset, batch_size=self.train_config.batch_size, shuffle=shuffle)

    def _init_model(self) -> None:
        """Instantiate the Chemprop MPNN and move it to device.

        Builds a plain nn.Module wrapping BondMessagePassing, aggregation, and
        RegressionFFN without depending on lightning.LightningModule.
        Applies weight initialisation if configured by train_config.weight_init.
        Use _AGGREGATIONS()[cfg.aggregation]() to retrieve the aggregation class.
        """
        from chemprop.nn import BondMessagePassing  # noqa: PLC0415
        from chemprop.nn.predictors import RegressionFFN  # noqa: PLC0415

        cfg = self.model_config
        mp = BondMessagePassing(d_h=cfg.hidden_size, depth=cfg.depth)
        agg = _AGGREGATIONS()[cfg.aggregation]()
        ffn = RegressionFFN(
            input_dim=mp.output_dim, n_layers=cfg.ffn_num_layers, dropout=cfg.dropout
        )

        class _MPNNWrapper(nn.Module):
            """Lightweight nn.Module combining message passing, aggregation, and FFN."""

            def __init__(
                self, message_passing: nn.Module, agg: nn.Module, predictor: nn.Module
            ) -> None:
                """Initialise the MPNN wrapper.

                Args:
                    message_passing: Chemprop message passing module.
                    agg: Graph-level aggregation module.
                    predictor: Feed-forward prediction head.
                """
                super().__init__()
                self.message_passing = message_passing
                self.agg = agg
                self.predictor = predictor

            def forward(self, bmg: Any, V_d: Any = None, X_d: Any = None) -> Any:
                """Run message passing, aggregate, and predict.

                Args:
                    bmg: Batched molecular graph.
                    V_d: Optional node-level descriptors.
                    X_d: Optional molecule-level descriptors.

                Returns:
                    Predictions tensor.
                """
                H_v = self.message_passing(bmg, V_d)
                H = self.agg(H_v, bmg.batch)
                return self.predictor(H)

        self._model = _MPNNWrapper(mp, agg, ffn).to(self.device)

        if self.train_config.weight_init == "xavier_uniform":
            for m in self._model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        elif self.train_config.weight_init == "kaiming_normal":
            for m in self._model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def train(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates | None = None,
    ) -> None:
        """Train the MPNN with MSE loss.

        Lazy-initialises the model on the first call. Tracks per-epoch train loss
        and train Spearman correlation. val_data is accepted but ignored (added in Task 4).

        Args:
            train_data: Training molecules and labels.
            val_data: Optional validation molecules and labels (ignored in this task).
        """
        self._epoch_metrics = []

        if self._model is None:
            self._init_model()

        model = self._model
        assert model is not None

        train_smiles = self.featurise(train_data)
        train_loader = self._build_dataloader(train_smiles, train_data.labels, shuffle=True)

        optimizer = _OPTIMIZERS[self.train_config.optimizer](
            model.parameters(),
            lr=self.train_config.learning_rate,
        )
        criterion = nn.MSELoss()

        for epoch in range(self.train_config.num_epochs):
            model.train()
            train_preds_all: list[np.ndarray] = []
            train_targets_all: list[np.ndarray] = []
            train_losses: list[float] = []

            for batch in train_loader:
                b: Any = batch
                optimizer.zero_grad()
                preds = model(b.bmg, b.V_d, b.X_d).squeeze(-1)
                targets_1d = b.Y.squeeze(-1).to(self.device)
                loss = criterion(preds, targets_1d)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                train_preds_all.append(preds.detach().cpu().numpy())
                train_targets_all.append(targets_1d.detach().cpu().numpy())

            avg_train_loss = float(np.mean(train_losses))
            train_preds_np = np.concatenate(train_preds_all)
            train_targets_np = np.concatenate(train_targets_all)

            additional: dict[str, float] = {}
            if len(train_preds_np) >= 2:
                train_results = Results(
                    targets=train_targets_np,
                    predictions=Predictions(means=train_preds_np),
                )
                spearman = train_results.metrics.get("spearman")
                if spearman is not None:
                    additional["train_spearman"] = float(spearman)

            self._epoch_metrics.append(
                SurrogateEpochMetrics(
                    epoch=epoch,
                    train_loss=avg_train_loss,
                    additional_metrics=additional,
                )
            )

        last = self._epoch_metrics[-1]
        self._training_metrics = {"final_train_loss": last.train_loss}
        if "train_spearman" in last.additional_metrics:
            self._training_metrics["final_train_spearman"] = last.additional_metrics[
                "train_spearman"
            ]

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Predict fitness means for a list of candidates.

        Args:
            candidate_points: Candidates to predict.

        Returns:
            Predictions with means only, no variances.

        Raises:
            RuntimeError: If called before train().
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        model = self._model
        smiles = self.featurise(candidate_points)
        loader = self._build_dataloader(smiles, labels=None, shuffle=False)

        model.eval()
        preds_all: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                b: Any = batch
                preds = model(b.bmg, b.V_d, b.X_d).squeeze(-1)
                preds_all.append(preds.cpu().numpy())

        return Predictions(means=np.concatenate(preds_all))

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
