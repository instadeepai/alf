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
from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions, Results
from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, EsmModel

from alf_tools.models.utils import extract_sequences_from_inputs, get_device

logger = logging.getLogger("alf-tools")


@dataclass
class ESM2ModelConfig:
    """Configuration for ESM-2 model architecture.

    Args:
        esm_model_id: HuggingFace model ID for the ESM-2 encoder.
        embedding_dim: Output dimension of the ESM-2 encoder.
        hidden_dim: Hidden dimension of the regression head MLP.
        num_hidden_layers: Number of hidden layers in the regression head.
        dropout: Dropout rate for MC Dropout uncertainty estimation.
        num_mc_samples: Number of forward passes for MC Dropout prediction.
    """

    esm_model_id: str = "facebook/esm2_t30_150M_UR50D"
    embedding_dim: int = 640
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    dropout: float = 0.1
    num_mc_samples: int = 30


@dataclass
class ESM2TrainConfig:
    """Configuration for ESM-2 head training.

    Args:
        learning_rate: Learning rate for the Adam optimizer.
        batch_size: Batch size for training and embedding computation.
        num_epochs: Number of epochs to train for.
        log_frequency: Frequency of logging training metrics.
    """

    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 50
    log_frequency: int = 10


class ESM2RegressionHead(nn.Module):
    """MLP regression head with MC Dropout for uncertainty estimation.

    Architecture: embedding_dim → [hidden_dim → ReLU → Dropout] x num_hidden_layers → 1
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        dropout: float,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = embedding_dim
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Embeddings of shape (batch_size, embedding_dim).

        Returns:
            Scalar predictions of shape (batch_size,).
        """
        return self.net(x).squeeze(-1)


class ESM2Embedder:
    """Frozen ESM-2 encoder with in-memory sequence embedding cache.

    Owns the tokenizer, encoder weights, and embedding cache. The encoder is
    frozen at construction and never receives gradients.
    """

    def __init__(self, model_id: str, batch_size: int, device: torch.device) -> None:
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.encoder = EsmModel.from_pretrained(model_id)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.to(device)
        self.encoder.eval()
        self._cache: dict[str, np.ndarray] = {}

    def get_embeddings(self, sequences: list[str]) -> torch.Tensor:
        """Compute or retrieve cached mean-pooled ESM-2 embeddings.

        Uncached sequences are run through the encoder in batches, mean-pooled
        over non-padding token positions, and stored in the cache.

        Args:
            sequences: Protein sequences to embed.

        Returns:
            Float32 tensor of shape (len(sequences), embedding_dim).
        """
        uncached = [seq for seq in sequences if seq not in self._cache]

        if uncached:
            for i in range(0, len(uncached), self.batch_size):
                batch_seqs = uncached[i : i + self.batch_size]
                tokens = self.tokenizer(batch_seqs, return_tensors="pt", padding=True)
                input_ids = tokens["input_ids"].to(self.device)
                attention_mask = tokens["attention_mask"].to(self.device)

                with torch.no_grad():
                    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

                last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_embeddings = (last_hidden * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask  # (batch, hidden)

                for seq, emb in zip(batch_seqs, mean_embeddings):
                    self._cache[seq] = emb.cpu().numpy()

        embeddings = np.stack([self._cache[seq] for seq in sequences])
        return torch.tensor(embeddings, dtype=torch.float32)


class ESM2DropoutModel(BaseModel):
    """ESM-2 frozen encoder with trainable MC Dropout regression head.

    Uses a frozen ESM-2 150M encoder to embed protein sequences and a
    trainable MLP regression head with MC Dropout for uncertainty quantification.
    Embeddings are cached in memory since the encoder is frozen.
    """

    def __init__(
        self,
        name: str = "esm2_dropout_model",
        model_config: ESM2ModelConfig | None = None,
        train_config: ESM2TrainConfig | None = None,
        device: str | None = None,
    ):
        """Initialize the ESM2DropoutModel.

        Args:
            name: Name of the surrogate model.
            model_config: Configuration for model architecture.
            train_config: Configuration for training.
            device: Device to use ('cuda', 'cpu', or None for auto-detect).
        """
        self.model_config = model_config or ESM2ModelConfig()
        self.train_config = train_config or ESM2TrainConfig()
        self.device = get_device(device)

        self._embedder = ESM2Embedder(
            model_id=self.model_config.esm_model_id,
            batch_size=self.train_config.batch_size,
            device=self.device,
        )

        self.head: ESM2RegressionHead | None = None
        self.training_metrics: dict[str, Union[float, int, np.number]] = {}
        self._epoch_metrics: list[SurrogateEpochMetrics] = []

    def featurise(self, inputs: Union[LabelledCandidates, list[Candidate]]) -> torch.Tensor:
        """Embed sequences using the frozen ESM-2 encoder.

        Args:
            inputs: LabelledCandidates or list of Candidates with sequence modality.

        Returns:
            Float32 tensor of shape (N, embedding_dim).
        """
        sequences = extract_sequences_from_inputs(inputs)
        return self._embedder.get_embeddings(sequences)

    def _prepare_data_loader(self, data: LabelledCandidates, shuffle: bool = False) -> DataLoader:
        x = self.featurise(data).to(self.device)
        y = torch.tensor(data.labels, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=self.train_config.batch_size, shuffle=shuffle)

    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> dict:
        if len(predictions) >= 2:
            metrics = Results(
                predictions=Predictions(means=predictions, variances=None), targets=targets
            ).metrics
        else:
            metrics = {
                "mse": nn.MSELoss()(
                    torch.tensor(predictions), torch.tensor(targets)
                ).item()
            }
        return metrics

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> tuple[float, dict]:
        if self.head is None:
            raise ValueError("Head must be initialized before training")
        self.head.train()
        train_losses = []
        train_predictions_all = []
        train_targets_all = []

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = self.head(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_predictions_all.append(predictions.detach().cpu().numpy())
            train_targets_all.append(batch_y.detach().cpu().numpy())

        avg_train_loss = float(np.mean(train_losses))
        train_preds = np.concatenate(train_predictions_all)
        train_targets = np.concatenate(train_targets_all)

        train_metrics = self._calculate_metrics(train_preds, train_targets)
        return avg_train_loss, train_metrics

    def _validate_epoch(
        self, val_loader: DataLoader, criterion: nn.Module
    ) -> tuple[float, dict]:
        if self.head is None:
            raise ValueError("Head must be initialized before validation")
        self.head.eval()
        val_losses = []
        val_predictions_all = []
        val_targets_all = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                predictions = self.head(batch_x)
                loss = criterion(predictions, batch_y)
                val_losses.append(loss.item())
                val_predictions_all.append(predictions.cpu().numpy())
                val_targets_all.append(batch_y.cpu().numpy())

        avg_val_loss = float(np.mean(val_losses))
        val_preds = np.concatenate(val_predictions_all)
        val_targets = np.concatenate(val_targets_all)

        val_metrics = self._calculate_metrics(val_preds, val_targets)
        return avg_val_loss, val_metrics

    def _record_epoch_metrics(
        self,
        epoch: int,
        avg_train_loss: float,
        train_metrics: dict,
        avg_val_loss: float | None = None,
        val_metrics: dict | None = None,
    ) -> None:
        additional: dict[str, float] = {}
        if (v := train_metrics.get("spearman")) is not None:
            additional["train_spearman"] = float(v)
        if (v := train_metrics.get("mse")) is not None:
            additional["train_mse"] = float(v)
        if val_metrics is not None:
            if (v := val_metrics.get("spearman")) is not None:
                additional["val_spearman"] = float(v)
            if (v := val_metrics.get("mse")) is not None:
                additional["val_mse"] = float(v)
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
        """Train the regression head on pre-computed ESM-2 embeddings.

        The encoder is never updated. Embeddings are cached after the first call.
        Each call reinitialises the regression head from scratch.

        Args:
            train_data: Training data with sequence candidates and labels.
            val_data: Optional validation data.
        """
        self._epoch_metrics = []
        logger.info(f"Training ESM-2 head with {len(train_data)} samples")

        self.head = ESM2RegressionHead(
            embedding_dim=self.model_config.embedding_dim,
            hidden_dim=self.model_config.hidden_dim,
            num_hidden_layers=self.model_config.num_hidden_layers,
            dropout=self.model_config.dropout,
        ).to(self.device)
        total_params = sum(p.numel() for p in self.head.parameters())
        logger.info(f"ESM-2 regression head initialized with {total_params:,} parameters")

        train_loader = self._prepare_data_loader(train_data, shuffle=True)
        val_loader = None
        if val_data is not None and len(val_data) > 0:
            val_loader = self._prepare_data_loader(val_data, shuffle=False)

        optimizer = optim.Adam(self.head.parameters(), lr=self.train_config.learning_rate)
        criterion = nn.MSELoss()

        avg_train_loss = 0.0
        train_metrics: dict = {}
        avg_val_loss: float | None = None
        val_metrics: dict | None = None

        for epoch in range(self.train_config.num_epochs):
            avg_train_loss, train_metrics = self._train_epoch(train_loader, optimizer, criterion)

            if val_loader is not None:
                avg_val_loss, val_metrics = self._validate_epoch(val_loader, criterion)
                self._record_epoch_metrics(
                    epoch, avg_train_loss, train_metrics, avg_val_loss, val_metrics
                )
            else:
                self._record_epoch_metrics(epoch, avg_train_loss, train_metrics)

        self.training_metrics = {"final_train_loss": avg_train_loss}
        self.training_metrics.update({f"final_train_{k}": v for k, v in train_metrics.items()})

        if val_loader is not None and avg_val_loss is not None and val_metrics is not None:
            self.training_metrics["final_val_loss"] = avg_val_loss
            self.training_metrics.update({f"final_val_{k}": v for k, v in val_metrics.items()})

    def predict(self, candidate_points: list[Candidate], with_uncertainty: bool = True) -> Predictions:
        """Make predictions using MC Dropout.

        Runs num_mc_samples forward passes through the head (in train mode,
        so dropout is active) to estimate predictive uncertainty.

        Args:
            candidate_points: Candidates to predict fitness for.
            with_uncertainty: Whether to compute predictive uncertainty.

        Returns:
            Predictions with means, variances, and empirical_dist of shape
            (N_candidates, num_mc_samples).

        Raises:
            RuntimeError: If the model has not been trained.
        """
        num_mc_samples = self.model_config.num_mc_samples if with_uncertainty else 1
        if self.head is None:
            raise RuntimeError("Model not trained. Call train() first.")

        x = self.featurise(candidate_points).to(self.device)

        if with_uncertainty:
            self.head.train()  # Keeps dropout active for MC sampling
        else:
            self.head.eval()

        samples = []
        with torch.no_grad():
            for _ in range(num_mc_samples):
                preds = self.head(x)
                samples.append(preds.cpu().numpy())

        samples_array = np.stack(samples, axis=1)  # (N, num_mc_samples)

        return Predictions(
            means=samples_array.mean(axis=1),
            variances=samples_array.var(axis=1) if with_uncertainty else None,
            empirical_dist=samples_array,
        )

    def sample(self, *args: Any, **kwargs: Any) -> list[Candidate]:
        """Not implemented for this model."""
        raise NotImplementedError("Sampling is not implemented for this model.")

    def get_epoch_metrics(self) -> list[SurrogateEpochMetrics]:
        """Return per-epoch metrics from the most recent train() call."""
        return self._epoch_metrics

    def get_training_summary_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Return training summary metrics."""
        return self.training_metrics
