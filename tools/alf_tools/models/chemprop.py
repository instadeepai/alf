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
from typing import Literal

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions, Results
from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics
from alf_tools.models.utils import get_device
from torch.utils.data import DataLoader

from chemprop.nn import MeanAggregation, NormAggregation, SumAggregation, BondMessagePassing
from chemprop.nn.predictors import RegressionFFN  
from chemprop.data import (
    BatchMolGraph,
    MoleculeDatapoint,
    MoleculeDataset,
    build_dataloader,
)
        
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
    weight_init: Literal["default", "xavier_uniform",
                         "kaiming_normal"] | None = None
    seed: int | None = None


def _get_aggregations() -> dict[str, type]:
    """Return mapping of aggregation name to Chemprop aggregation class.

    Returns:
        Dictionary mapping aggregation name strings to aggregation classes.
    """

    return {
        "mean": MeanAggregation,
        "sum": SumAggregation,
        "norm": NormAggregation,
    }


_OPTIMIZERS: dict[str, type] = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adamw": optim.AdamW,
}


class _MPNNWrapper(nn.Module):
    """Lightweight nn.Module combining message passing, aggregation, and FFN."""

    def __init__(self, message_passing: nn.Module, agg: nn.Module, predictor: nn.Module) -> None:
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

    def forward(
        self,
        bmg: BatchMolGraph,
        V_d: torch.Tensor | None = None,
        X_d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run message passing, aggregate, and predict.

        Args:
            bmg: Batched molecular graph (already on the correct device).
            V_d: Optional node-level descriptors.
            X_d: Optional molecule-level descriptors.

        Returns:
            Predictions tensor of shape (batch_size, 1).
        """
        H_v = self.message_passing(bmg, V_d)
        H = self.agg(H_v, bmg.batch)
        return self.predictor(H)


class ChempropModel(BaseModel):
    """Mean-only MPNN surrogate model using Chemprop v2.x as backbone.

    Accepts SMILES strings via Candidate.data and returns scalar fitness predictions.
    The network is initialised lazily on the first train() call. Subsequent train()
    calls fine-tune from the current weights rather than resetting them.
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
        self.training_metrics: dict[str, float] = {}

    def featurise(self, inputs: LabelledCandidates | list[Candidate]) -> list[str]:
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
        """
        cfg = self.model_config
        mp = BondMessagePassing(d_h=cfg.hidden_size, depth=cfg.depth)
        agg = _get_aggregations()[cfg.aggregation]()
        ffn = RegressionFFN(
            input_dim=mp.output_dim, n_layers=cfg.ffn_num_layers, dropout=cfg.dropout
        )

        self._model = _MPNNWrapper(mp, agg, ffn).to(self.device)
        total_params = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "ChempropModel initialised: hidden=%d depth=%d ffn_layers=%d params=%d device=%s",
            cfg.hidden_size,
            cfg.depth,
            cfg.ffn_num_layers,
            total_params,
            self.device,
        )

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

    def _train_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> tuple[float, dict[str, float]]:
        """Run one training pass over a DataLoader with gradient updates.

        Args:
            loader: DataLoader to iterate over.
            criterion: Loss function.
            optimizer: Optimizer for gradient updates.

        Returns:
            Tuple of (average loss weighted by batch size, metrics dict).
        """
        if self._model is None:
            raise RuntimeError("_train_epoch called before model is initialised")
        model = self._model
        n = len(loader.dataset)  # type: ignore[arg-type]
        all_preds = torch.empty(n, device="cpu")
        all_targets = torch.empty(n, device="cpu")
        total_loss = 0.0
        idx = 0

        model.train(True)
        for batch in loader:
            # BatchMolGraph.to() is in-place with no return value; unlike Tensor.to() which creates
            # a new tensor and must be assigned — assigning bmg.to() would leave bmg on the old device
            batch.bmg.to(self.device)
            V_d = batch.V_d.to(self.device) if batch.V_d is not None else None
            X_d = batch.X_d.to(self.device) if batch.X_d is not None else None
            targets = batch.Y.squeeze(-1).to(self.device)
            preds = model(batch.bmg, V_d, X_d).squeeze(-1)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            b = targets.shape[0]
            total_loss += loss.item() * b
            all_preds[idx: idx + b] = preds.detach().cpu()
            all_targets[idx: idx + b] = targets.detach().cpu()
            idx += b

        all_preds = all_preds[:idx]
        all_targets = all_targets[:idx]
        avg_loss = total_loss / idx if idx > 0 else 0.0
        preds_np = all_preds.numpy()
        targets_np = all_targets.numpy()
        if len(preds_np) >= 2:
            raw = Results(targets=targets_np,
                          predictions=Predictions(means=preds_np)).metrics
            metrics = {k: float(v) for k in ("spearman", "mse")
                       if (v := raw.get(k)) is not None}
        else:
            metrics = {"mse": float(np.mean((preds_np - targets_np) ** 2))}
        return avg_loss, metrics

    def _eval_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, dict[str, float]]:
        """Run one evaluation pass over a DataLoader without gradient updates.

        Args:
            loader: DataLoader to iterate over.
            criterion: Loss function.

        Returns:
            Tuple of (average loss weighted by batch size, metrics dict).
        """
        if self._model is None:
            raise RuntimeError("_eval_epoch called before model is initialised")
        model = self._model
        n = len(loader.dataset)  # type: ignore[arg-type]
        all_preds = torch.empty(n, device="cpu")
        all_targets = torch.empty(n, device="cpu")
        total_loss = 0.0
        idx = 0

        model.train(False)
        with torch.no_grad():
            for batch in loader:
                # BatchMolGraph.to() is in-place with no return value; unlike Tensor.to() which creates
                # a new tensor and must be assigned — assigning bmg.to() would leave bmg on the old device
                batch.bmg.to(self.device)
                V_d = batch.V_d.to(self.device) if batch.V_d is not None else None
                X_d = batch.X_d.to(self.device) if batch.X_d is not None else None
                targets = batch.Y.squeeze(-1).to(self.device)
                preds = model(batch.bmg, V_d, X_d).squeeze(-1)
                loss = criterion(preds, targets)

                b = targets.shape[0]
                total_loss += loss.item() * b
                all_preds[idx: idx + b] = preds.detach().cpu()
                all_targets[idx: idx + b] = targets.detach().cpu()
                idx += b

        all_preds = all_preds[:idx]
        all_targets = all_targets[:idx]
        avg_loss = total_loss / idx if idx > 0 else 0.0
        preds_np = all_preds.numpy()
        targets_np = all_targets.numpy()
        if len(preds_np) >= 2:
            raw = Results(targets=targets_np,
                          predictions=Predictions(means=preds_np)).metrics
            metrics = {k: float(v) for k in ("spearman", "mse")
                       if (v := raw.get(k)) is not None}
        else:
            metrics = {"mse": float(np.mean((preds_np - targets_np) ** 2))}
        return avg_loss, metrics

    def _record_epoch_metrics(
        self,
        epoch: int,
        avg_train_loss: float,
        train_metrics: dict[str, float],
        avg_val_loss: float | None = None,
        val_metrics: dict[str, float] | None = None,
    ) -> None:
        """Assemble and append a SurrogateEpochMetrics for one epoch.

        Args:
            epoch: Zero-based epoch index.
            avg_train_loss: Average training loss for the epoch.
            train_metrics: Metrics dict from the training pass.
            avg_val_loss: Average validation loss, or ``None`` if no validation set.
            val_metrics: Metrics dict from the validation pass, or ``None``.
        """
        additional: dict[str, float] = {}
        for key in ("spearman", "mse"):
            if (v := train_metrics.get(key)) is not None:
                additional[f"train_{key}"] = v
        if val_metrics is not None:
            for key in ("spearman", "mse"):
                if (v := val_metrics.get(key)) is not None:
                    additional[f"val_{key}"] = v
        self._epoch_metrics.append(
            SurrogateEpochMetrics(
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                additional_metrics=additional,
            )
        )

    def train(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates | None = None,
    ) -> None:
        """Train the MPNN with MSE loss.

        Lazy-initialises the network on the first call. Subsequent calls fine-tune
        from the existing weights rather than resetting them. Tracks per-epoch train
        loss, train MSE, train Spearman, and (when val_data is provided) val loss,
        val MSE, and val Spearman.

        Args:
            train_data: Training molecules and labels.
            val_data: Optional validation molecules and labels.
        """
        self._epoch_metrics = []
        # TODO: also reset training_metrics = {} here; should be symmetric with _epoch_metrics
        # reset across all model implementations (CNNModel, GPModel) so callers never read
        # stale metrics after a failed train() call
        logger.info("Training ChempropModel on %d samples", len(train_data))

        if self.train_config.seed is not None:

            random.seed(self.train_config.seed)
            np.random.seed(self.train_config.seed)
            torch.manual_seed(self.train_config.seed)
            torch.cuda.manual_seed_all(self.train_config.seed)

        if self._model is None:
            self._init_model()
            
        train_loader = self._build_dataloader(
            self.featurise(train_data), train_data.labels, shuffle=True
        )
        val_loader = (
            self._build_dataloader(self.featurise(
                val_data), val_data.labels, shuffle=False)
            if val_data is not None and len(val_data) > 0
            else None
        )
        optimizer = _OPTIMIZERS[self.train_config.optimizer](
            self._model.parameters(), lr=self.train_config.learning_rate
        )
        criterion = nn.MSELoss()

        if self.train_config.num_epochs == 0:
            logger.warning("num_epochs=0; skipping training loop.")
            return

        for epoch in range(self.train_config.num_epochs):
            avg_train_loss, train_metrics = self._train_epoch(
                train_loader, criterion, optimizer)
            if val_loader is not None:
                avg_val_loss, val_metrics = self._eval_epoch(
                    val_loader, criterion)
                self._record_epoch_metrics(
                    epoch, avg_train_loss, train_metrics, avg_val_loss, val_metrics
                )
            else:
                self._record_epoch_metrics(
                    epoch, avg_train_loss, train_metrics)

        self.training_metrics = {"final_train_loss": avg_train_loss}
        self.training_metrics.update(
            {f"final_train_{k}": v for k, v in train_metrics.items()})
        if val_loader is not None:
            self.training_metrics["final_val_loss"] = avg_val_loss
            self.training_metrics.update(
                {f"final_val_{k}": v for k, v in val_metrics.items()})

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
        n = len(smiles)
        all_preds = torch.empty(n, device="cpu")
        idx = 0

        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch.bmg.to(self.device)
                V_d = batch.V_d.to(
                    self.device) if batch.V_d is not None else None
                X_d = batch.X_d.to(
                    self.device) if batch.X_d is not None else None
                preds = model(batch.bmg, V_d, X_d).squeeze(-1)
                b = preds.shape[0]
                all_preds[idx: idx + b] = preds.cpu()
                idx += b

        all_preds = all_preds[:idx]
        return Predictions(means=all_preds.numpy())

    def sample(self, condition: object | None = None) -> list[Candidate]:
        """Not implemented for mean-only MPNN."""
        raise NotImplementedError(
            "Sampling is not implemented for ChempropModel.")

    def get_epoch_metrics(self) -> list[SurrogateEpochMetrics]:
        """Return per-epoch metrics from the most recent train() call.

        Returns:
            List of SurrogateEpochMetrics, one per epoch.
        """
        return self._epoch_metrics

    def get_training_summary_metrics(self) -> dict[str, float]:
        """Return summary metrics from the most recent train() call.

        Returns:
            Dictionary of final train/val losses, MSE, and Spearman correlations.
        """
        return self.training_metrics
