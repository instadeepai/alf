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
from alf_core import Candidate, LabelledCandidates, Predictions, Results
from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics
from alf_core.model.base_model import BaseModel, BaseTrainConfig
from alf_core.model.normaliser import (
    InputNormaliser,
    OutputStandardiser,
)
from torch.utils.data import DataLoader, TensorDataset

from alf_tools.models.utils import (
    create_char_to_idx_mapping,
    get_device,
    one_hot_encode,
    transform_data,
)
from alf_tools.utils.constants import PROTEIN_ALPHABET

logger = logging.getLogger("alf-tools")


@dataclass
class CNNModelConfig:
    """Configuration for CNN model architecture.

    Args:
        num_filters: Number of filters in convolutional layers.
        kernel_size: Size of convolutional kernels.
        num_conv_layers: Number of convolutional layers.
        fc_hidden_dim: Dimension of fully connected hidden layers.
        dropout: Dropout rate.
    """

    num_filters: int = 128
    kernel_size: int = 3
    num_conv_layers: int = 3
    fc_hidden_dim: int = 256
    dropout: float = 0.3


@dataclass
class CNNTrainConfig(BaseTrainConfig):
    """Configuration for CNN training.

    Args:
        batch_size: Batch size for training.
        num_epochs: Number of epochs to train for.
        learning_rate: Inherited from BaseTrainConfig. Default: 1e-3.
        log_frequency: Inherited from BaseTrainConfig. Default: 10.
        label_dtype: Inherited from BaseTrainConfig. None uses the model
            default (float32 for CNN regression). Override to force a dtype.
    """

    batch_size: int = 32
    num_epochs: int = 50


class SequenceCNN(nn.Module):
    """Simple 1D CNN for sequence regression.

    Architecture:
    - One-hot encoding → Conv1D layers → Fully connected → Scalar output
    """

    def __init__(
        self,
        seq_length: int,
        alphabet_size: int = 20,
        num_filters: int = 128,
        kernel_size: int = 3,
        num_conv_layers: int = 3,
        fc_hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        """Initialize the SequenceCNN model.

        Args:
            seq_length: Length of input sequences.
            alphabet_size: Size of the sequence alphabet (default: 20).
            num_filters: Number of filters in convolutional layers.
            kernel_size: Size of convolutional kernels.
            num_conv_layers: Number of convolutional layers.
            fc_hidden_dim: Dimension of fully connected hidden layers.
            dropout: Dropout probability.
        """
        super().__init__()

        # Convolutional layers
        conv_layers = []
        in_channels = alphabet_size
        current_length = seq_length

        for _ in range(num_conv_layers):
            conv_layers.extend([
                nn.Conv1d(in_channels, num_filters, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout),
            ])
            in_channels = num_filters
            current_length = current_length // 2

        self.conv_block = nn.Sequential(*conv_layers)

        # Fully connected layers
        flattened_size = num_filters * max(1, current_length)
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: One-hot encoded sequences (batch_size, alphabet_size, seq_length).

        Returns:
            Fitness predictions (batch_size,).
        """
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x.squeeze(-1)


class CNNModel(BaseModel):
    """Minimal CNN model for sequence fitness prediction.

    One-hot encodes sequences, trains a simple 1D CNN with MSE loss.
    """

    _default_label_dtype: torch.dtype = torch.float32

    def __init__(
        self,
        name: str = "cnn_model",
        model_config: CNNModelConfig | None = None,
        train_config: CNNTrainConfig | None = None,
        alphabet: str = PROTEIN_ALPHABET,
        device: str | None = None,
    ):
        """Initialize the CNNModel.

        Args:
            name: Name of the surrogate model.
            model_config: Configuration for model architecture.
            train_config: Configuration for training.
            alphabet: Sequence alphabet to use.
            device: Device to use for training ('cuda', 'cpu', or None for auto-detect).
        """
        # Use defaults if configs not provided
        self.model_config = model_config or CNNModelConfig()
        self.train_config = train_config or CNNTrainConfig()

        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.char_to_idx = create_char_to_idx_mapping(alphabet)

        # Device setup
        self.device = get_device(device)

        # Model initialized on first fit
        self.model: SequenceCNN | None = None
        self.seq_length: int | None = None

        # Input normaliser — fitted on each train() call, applied at predict() time
        self._input_normaliser: InputNormaliser | None = None
        self._output_standardiser: OutputStandardiser | None = None

        # Track metrics
        self.training_metrics: dict[str, Union[float, int, np.number]] = {}
        self._epoch_metrics: list[SurrogateEpochMetrics] = []

    def _one_hot_encode(self, sequences: list[str]) -> torch.Tensor:
        """One-hot encode sequences.

        Args:
            sequences: List of sequences as strings.

        Returns:
            One-hot encoded tensor of shape (batch_size, alphabet_size, seq_length).
        """
        return one_hot_encode(
            sequences=sequences,
            char_to_idx=self.char_to_idx,
            alphabet_size=self.alphabet_size,
            flatten=False,
        )

    def featurise(self, inputs: Union[LabelledCandidates, list[Candidate]]) -> torch.Tensor:
        """Convert inputs to one-hot encoded tensors.

        Args:
            inputs: Either LabelledCandidates or list of Candidates to featurise.

        Returns:
            A one-hot encoded tensor of shape (batch_size, alphabet_size, seq_length).

        Raises:
            ValueError: If the input is not LabelledCandidates or list of Candidates.
        """
        if isinstance(inputs, LabelledCandidates):
            sequences = inputs.data
        elif isinstance(inputs, list) and all(isinstance(c, Candidate) for c in inputs):
            sequences = [c.data for c in inputs]
        else:
            raise ValueError("Input must be LabelledCandidates or list of Candidates")

        return self._one_hot_encode(sequences)

    def _build_data_loaders(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates | None,
    ) -> tuple[DataLoader, DataLoader | None]:
        """Fit normalisers on training data and build DataLoaders for train and val.

        Normalisers are fitted exclusively on training data. Val data is transformed
        using train statistics to avoid data leakage.

        Args:
            train_data: Training data.
            val_data: Optional validation data.

        Returns:
            Tuple of (train_loader, val_loader). val_loader is None if val_data is None.
        """
        label_dtype = self.train_config.label_dtype or self._default_label_dtype
        train_x, train_y, self._input_normaliser, self._output_standardiser = transform_data(
            self.featurise(train_data),
            train_data.labels,
            self.train_config.normalise_inputs,
            self.train_config.standardise_outputs,
            label_dtype,
            self.device,
        )

        train_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = None
        if val_data is not None and len(val_data) > 0:
            val_x_np = np.array(self.featurise(val_data).cpu())
            if self._input_normaliser is not None:
                val_x_np = self._input_normaliser.transform(val_x_np)
            val_y_np = val_data.labels
            if self._output_standardiser is not None:
                val_y_np = self._output_standardiser.transform(val_y_np)
            val_x = torch.tensor(val_x_np, dtype=torch.float32).to(self.device)
            val_y = torch.tensor(val_y_np, dtype=label_dtype).to(self.device)
            val_loader = DataLoader(
                TensorDataset(val_x, val_y),
                batch_size=self.train_config.batch_size,
                shuffle=False,
                num_workers=0,
            )

        return train_loader, val_loader

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> tuple[float, dict]:
        """Train for one epoch.

        Args:
            train_loader: DataLoader for training data.
            optimizer: Optimizer for training.
            criterion: Loss criterion.

        Returns:
            Tuple of (average_loss, metrics_dict).

        Raises:
            ValueError: If the model is not initialized.
        """
        if self.model is None:
            raise ValueError("Model must be initialized before training")
        self.model.train()
        train_losses = []
        train_predictions_all = []
        train_targets_all = []

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_predictions_all.append(predictions.detach().cpu().numpy())
            train_targets_all.append(batch_y.detach().cpu().numpy())

        avg_train_loss = np.mean(train_losses)
        train_preds = np.concatenate(train_predictions_all)
        train_targets = np.concatenate(train_targets_all)

        # Use Predictions to compute all metrics
        train_predictions_obj = Predictions(means=train_preds, variances=None)
        # Handle case where dataset is too small for correlation metrics
        if len(train_preds) >= 2:
            train_metrics = Results(
                predictions=train_predictions_obj, targets=train_targets
            ).metrics
        else:
            # Only compute MSE for very small datasets
            mse = np.mean((train_preds - train_targets) ** 2)
            train_metrics = {"mse": mse}

        return avg_train_loss, train_metrics

    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> tuple[float, dict]:
        """Validate for one epoch.

        Args:
            val_loader: DataLoader for validation data.
            criterion: Loss criterion.

        Returns:
            Tuple of (average_loss, metrics_dict).

        Raises:
            ValueError: If the model is not initialized.
        """
        if self.model is None:
            raise ValueError("Model must be initialized before validation")

        self.model.eval()
        val_losses = []
        val_predictions_all = []
        val_targets_all = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                predictions = self.model(batch_x)
                loss = criterion(predictions, batch_y)
                val_losses.append(loss.item())
                val_predictions_all.append(predictions.cpu().numpy())
                val_targets_all.append(batch_y.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        val_preds = np.concatenate(val_predictions_all)
        val_targets = np.concatenate(val_targets_all)

        # Use Predictions to compute all metrics
        val_predictions_obj = Predictions(means=val_preds, variances=None)
        # Handle case where dataset is too small for correlation metrics
        if len(val_preds) >= 2:
            val_metrics = Results(predictions=val_predictions_obj, targets=val_targets).metrics
        else:
            # Only compute MSE for very small datasets
            mse = np.mean((val_preds - val_targets) ** 2)
            val_metrics = {"mse": mse}

        return avg_val_loss, val_metrics

    def _record_epoch_metrics(
        self,
        epoch: int,
        avg_train_loss: float,
        train_metrics: dict,
        avg_val_loss: float | None = None,
        val_metrics: dict | None = None,
    ) -> None:
        """Record epoch metrics and log at the configured frequency.

        Args:
            epoch: Current epoch.
            avg_train_loss: Average training loss.
            train_metrics: Dictionary of training metrics.
            avg_val_loss: Average validation loss.
            val_metrics: Dictionary of validation metrics.
        """
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
        epoch_metrics = SurrogateEpochMetrics(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            additional_metrics=additional,
        )
        self._epoch_metrics.append(epoch_metrics)

    def train(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates | None = None,
    ) -> None:
        """Train the CNN model.

        Args:
            train_data: Training data containing sequences and oracle values.
            val_data: Optional validation data.
        """
        self._epoch_metrics = []
        logger.info(f"Training CNN with {len(train_data)} samples")

        # Initialize model on first call
        if self.model is None:
            self.seq_length = len(train_data.data[0])
            self.model = SequenceCNN(
                seq_length=self.seq_length,
                alphabet_size=self.alphabet_size,
                num_filters=self.model_config.num_filters,
                kernel_size=self.model_config.kernel_size,
                num_conv_layers=self.model_config.num_conv_layers,
                fc_hidden_dim=self.model_config.fc_hidden_dim,
                dropout=self.model_config.dropout,
            ).to(self.device)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"CNN initialized with {total_params:,} parameters")

        # Featurize, fit normalisers on train, and build DataLoaders
        train_loader, val_loader = self._build_data_loaders(train_data, val_data)

        # Setup training
        optimizer = optim.Adam(self.model.parameters(), lr=self.train_config.learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        for epoch in range(self.train_config.num_epochs):
            # Train
            avg_train_loss, train_metrics = self._train_epoch(train_loader, optimizer, criterion)

            # Record metrics
            if val_loader is not None:
                avg_val_loss, val_metrics = self._validate_epoch(val_loader, criterion)
                self._record_epoch_metrics(
                    epoch,
                    avg_train_loss,
                    train_metrics,
                    avg_val_loss,
                    val_metrics,
                )
            else:
                self._record_epoch_metrics(
                    epoch,
                    avg_train_loss,
                    train_metrics,
                )

            # Log at configured frequency
            if epoch % self.train_config.log_frequency == 0:
                logger.info(
                    f"Epoch {epoch}/{self.train_config.num_epochs}"
                    f" - train_loss={avg_train_loss:.4f}"
                )

        # Store final metrics
        self.training_metrics = {
            "final_train_loss": avg_train_loss,
        }
        # Add all final train metrics
        self.training_metrics.update({f"final_train_{k}": v for k, v in train_metrics.items()})

        if val_loader is not None:
            self.training_metrics["final_val_loss"] = avg_val_loss
            # Add all final validation metrics
            self.training_metrics.update({f"final_val_{k}": v for k, v in val_metrics.items()})

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Make predictions for candidates.

        Args:
            candidate_points: List of candidates to predict fitness for.

        Returns:
            Predictions containing predicted fitness values.

        Raises:
            RuntimeError: If the model is not trained.
            ValueError: If the input is not a list of candidates.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        self.model.eval()
        x_np = np.array(self.featurise(candidate_points).cpu())

        if self._input_normaliser is not None:
            x_np = self._input_normaliser.transform(x_np)
        x = torch.tensor(x_np, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            means = self.model(x).cpu().numpy()

        if self._output_standardiser is not None:
            means, _ = self._output_standardiser.inverse_transform(means)

        return Predictions(means=means)

    def sample(self, *args: Any, **kwargs: Any) -> list[Candidate]:
        """Sample candidate points from the model."""
        raise NotImplementedError("Sampling is not implemented for this model.")

    def get_epoch_metrics(self) -> list[SurrogateEpochMetrics]:
        """Return per-epoch metrics from the most recent train() call.

        Returns:
            List of SurrogateEpochMetrics, one per epoch trained.
        """
        return self._epoch_metrics

    def get_training_summary_metrics(
        self,
    ) -> dict[str, Union[float, int, np.number]]:
        """Return training metrics.

        Returns:
            Dictionary of training metrics including losses and Spearman correlations.
        """
        return self.training_metrics
