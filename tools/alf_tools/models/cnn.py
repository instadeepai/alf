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
from alf_core.dataset.base_dataset import BaseDataset
from alf_core.model.base_model import BaseModel, BaseTrainConfig
from alf_core.model.normaliser import (
    InputNormaliser,
    InputStandardiser,
    OutputStandardiser,
)
from alf_core.utils.enums import ProblemType
from jaxtyping import Float
from torch.utils.data import DataLoader, TensorDataset

from alf_tools.models.utils import (
    create_char_to_idx_mapping,
    get_device,
    one_hot_encode,
    transform_data,
)
from alf_tools.utils.constants import PROTEIN_ALPHABET

logger = logging.getLogger("alf-tools")


def _apply_activation(logits: torch.Tensor, problem_type: ProblemType) -> torch.Tensor:
    """Apply output activation for the given problem type.

    Args:
        logits: Raw model output tensor of shape (batch, 1) for BINARY or (batch, k) for MULTICLASS.
        problem_type: Determines the activation. BINARY → sigmoid + complement pair (n, 2);
            MULTICLASS → softmax (n, k); REGRESSION → identity (no-op).

    Returns:
        Activated tensor. BINARY returns (n, 2) [neg_prob, pos_prob];
        others pass through shape unchanged.
    """
    if problem_type == ProblemType.BINARY:
        pos_prob = torch.sigmoid(logits)
        return torch.stack([1.0 - pos_prob, pos_prob], dim=-1)
    if problem_type == ProblemType.MULTICLASS:
        return torch.softmax(logits, dim=-1)
    return logits


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
        normalise_inputs_strategy: Inherited from BaseTrainConfig. Default: None
            (no input normalisation). Z-score standardisation is a poor fit for
            the one-hot sequence inputs CNNModel uses, so it is left disabled by
            default; set explicitly to opt in for continuous-feature inputs.
        standardise_outputs: Inherited from BaseTrainConfig. Default: False.
        label_dtype: Inherited from BaseTrainConfig. None uses the model
            default (float32 for CNN regression). Override to force a dtype.
    """

    batch_size: int = 32
    num_epochs: int = 50
    # normalise_inputs_strategy is intentionally left at the inherited None default:
    # z-score standardisation distorts one-hot sequence inputs (constant positions
    # become dead channels at train time but active at test time). For non-one-hot,
    # continuous-feature inputs, "zscore" is the recommended choice — set it explicitly.


class SequenceCNN(nn.Module):
    """Simple 1D CNN for sequence prediction.

    Architecture:
    - One-hot encoding → Conv1D layers → Fully connected → output
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
        output_neurons: int = 1,
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
            output_neurons: Number of output neurons. Use 1 for regression and
                binary classification; use num_classes for multiclass.
        """
        super().__init__()
        self._output_neurons = output_neurons

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
            nn.Linear(fc_hidden_dim // 2, output_neurons),
        )

    def forward(
        self, x: Float[torch.Tensor, "batch_size alphabet_size seq_length"]
    ) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: One-hot encoded sequences (batch_size, alphabet_size, seq_length).

        Returns:
            Logits of shape (batch_size,) when output_neurons==1, else
            (batch_size, output_neurons).
        """
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        if self._output_neurons == 1:
            return x.squeeze(-1)
        return x


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
            model_config: Configuration for model architecture and problem type.
            name: Name of the surrogate model.
            train_config: Configuration for training.
            alphabet: Sequence alphabet to use.
            device: Device to use for training ('cuda', 'cpu', or None for auto-detect).
        """
        self.name = name
        self.model_config = model_config or CNNModelConfig()
        self.train_config = train_config or CNNTrainConfig()

        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.char_to_idx = create_char_to_idx_mapping(alphabet)

        self.device = get_device(device)

        # Pytorch model, initialized on first train() call
        self.model: SequenceCNN | None = None
        self.seq_length: int | None = None

        # Input normaliser — fitted on each train() call, applied at predict() time
        self._input_normaliser: InputNormaliser | InputStandardiser | None = None
        self._output_standardiser: OutputStandardiser | None = None

        # Track metrics
        self.training_metrics: dict[str, Union[float, int, np.number]] = {}
        self._epoch_metrics: list[SurrogateEpochMetrics] = []

    def setup(self, dataset: BaseDataset) -> None:
        """Configure the model from the full dataset before training begins.

        Determines the number of output neurons from the complete label space and
        validates consistency with the declared problem type.

        Args:
            dataset: The dataset this model will be trained on.

        Raises:
            ValueError: If the model's problem_type disagrees with the dataset's.
        """
        super().setup(dataset)
        self._num_output_neurons = self.output_dim

    def _one_hot_encode(
        self, sequences: list[str]
    ) -> Float[torch.Tensor, "batch_size alphabet_size seq_length"]:
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

    def featurise(
        self, inputs: Union[LabelledCandidates, list[Candidate]]
    ) -> Float[torch.Tensor, "batch_size alphabet_size seq_length"]:
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
            raise ValueError(
                f"Inputs must be a LabelledCandidates or list[Candidate], "
                f"got {type(inputs).__name__}"
            )

        return self._one_hot_encode(sequences)

    def _prepare_train_data(
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

        Raises:
            ValueError: If standardise_outputs=True is set for a non-regression problem.

        Returns:
            Tuple of (train_loader, val_loader). val_loader is None if val_data is None.
        """
        if self.train_config.label_dtype is not None:
            label_dtype = self.train_config.label_dtype
        elif getattr(self, "problem_type", None) == ProblemType.MULTICLASS:
            label_dtype = torch.long
        else:
            label_dtype = self._default_label_dtype
        problem_type = getattr(self, "problem_type", None)
        if self.train_config.standardise_outputs and problem_type != ProblemType.REGRESSION:
            raise ValueError(
                f"standardise_outputs=True is not supported for {problem_type} — "
                "standardisation only applies to regression targets."
            )
        train_x, train_y, self._input_normaliser, self._output_standardiser = transform_data(
            self.featurise(train_data),
            train_data.labels,
            self.train_config.normalise_inputs_strategy,
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
            val_x_tensor = self.featurise(val_data)
            if self._input_normaliser is not None:
                val_x_np = self._input_normaliser.transform(np.array(val_x_tensor.cpu()))
                val_x = torch.tensor(val_x_np, dtype=train_x.dtype).to(self.device)
            else:
                val_x = val_x_tensor.to(dtype=train_x.dtype).to(self.device)
            val_y_np = val_data.labels
            if self._output_standardiser is not None:
                val_y_np = self._output_standardiser.transform(val_y_np)
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
            RuntimeError: If the model is not initialized.
        """
        if self.model is None:
            raise RuntimeError(
                "CNN model has not been initialized — call model.train() before _train_epoch()"
            )
        self.model.train()
        train_losses = []
        train_predictions_all = []
        train_targets_all = []

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_predictions_all.append(
                _apply_activation(logits, self.problem_type).detach().cpu().numpy()
            )
            train_targets_all.append(batch_y.detach().cpu().numpy())

        avg_train_loss = float(np.mean(train_losses))
        train_preds = np.concatenate(train_predictions_all)
        train_targets = np.concatenate(train_targets_all)

        if self._output_standardiser is not None:
            train_preds, _ = self._output_standardiser.inverse_transform(train_preds)
            train_targets, _ = self._output_standardiser.inverse_transform(train_targets)

        train_metrics = Results(
            predictions=Predictions(means=train_preds, variances=None),
            targets=train_targets,
            problem_type=self.problem_type,
        ).metrics

        return avg_train_loss, train_metrics

    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> tuple[float, dict]:
        """Validate for one epoch.

        Args:
            val_loader: DataLoader for validation data.
            criterion: Loss criterion.

        Returns:
            Tuple of (average_loss, metrics_dict).

        Raises:
            RuntimeError: If the model is not initialized.
        """
        if self.model is None:
            raise RuntimeError(
                "CNN model has not been initialized — call model.train() before _validate_epoch()"
            )

        self.model.eval()
        val_losses = []
        val_predictions_all = []
        val_targets_all = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                val_losses.append(loss.item())
                val_predictions_all.append(
                    _apply_activation(logits, self.problem_type).cpu().numpy()
                )
                val_targets_all.append(batch_y.cpu().numpy())

        avg_val_loss = float(np.mean(val_losses))
        val_preds = np.concatenate(val_predictions_all)
        val_targets = np.concatenate(val_targets_all)

        if self._output_standardiser is not None:
            val_preds, _ = self._output_standardiser.inverse_transform(val_preds)
            val_targets, _ = self._output_standardiser.inverse_transform(val_targets)

        val_metrics = Results(
            predictions=Predictions(means=val_preds, variances=None),
            targets=val_targets,
            problem_type=self.problem_type,
        ).metrics

        return avg_val_loss, val_metrics

    def _record_epoch_metrics(
        self,
        epoch: int,
        avg_train_loss: float,
        train_metrics: dict,
        avg_val_loss: float | None = None,
        val_metrics: dict[str, float] | None = None,
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
        for k, v in train_metrics.items():
            additional[f"train_{k}"] = float(v)
        if val_metrics is not None:
            for k, v in val_metrics.items():
                additional[f"val_{k}"] = float(v)
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

        Raises:
            ValueError: If the model's problem_type disagrees with the dataset's.
            RuntimeError: If the model is not properly initialized.
        """
        if not hasattr(self, "output_dim"):
            raise ValueError("CNNModel.setup(dataset) must be called before train()")
        if self.problem_type == ProblemType.BINARY:
            invalid_labels = train_data.labels[~np.isin(train_data.labels, [0, 1])]
            if len(invalid_labels) > 0:
                raise ValueError(
                    f"BINARY classification requires labels in {{0, 1}}, "
                    f"got invalid values: {np.unique(invalid_labels).tolist()}"
                )
        logger.info(
            f"Training CNN with {len(train_data)} samples (problem_type={self.problem_type})"
        )
        self._epoch_metrics = []

        # Initialize model on first call; raise if output shape changes
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
                output_neurons=self.output_dim,
            ).to(self.device)
        elif self.model._output_neurons != self.output_dim:
            raise RuntimeError(
                f"CNNModel output neuron count changed from {self.model._output_neurons} "
                f"to {self.output_dim} between train() calls. "
                "Call setup() again with the correct dataset before retraining."
            )

        assert self.model is not None  # guaranteed by the block above
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"CNN initialized with {total_params:,} parameters")

        # Featurize, fit normalisers on train, and build DataLoaders
        train_loader, val_loader = self._prepare_train_data(train_data, val_data)

        # Setup training
        optimizer = optim.Adam(self.model.parameters(), lr=self.train_config.learning_rate)
        if self.problem_type == ProblemType.REGRESSION:
            criterion: nn.Module = nn.MSELoss()
        elif self.problem_type == ProblemType.BINARY:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        # Training loop
        avg_train_loss: float = 0.0
        train_metrics: dict[str, float] = {}
        avg_val_loss = None
        val_metrics: dict[str, float] = {}
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

        For regression, returns raw scalar predictions.
        For binary classification, returns class probabilities of shape
        `(n_samples, 2)` via sigmoid on the single output neuron.
        For multiclass, returns softmax probabilities of shape
        `(n_samples, num_classes)`.

        Args:
            candidate_points: List of candidates to predict for.

        Returns:
            Predictions containing means of shape (n_samples,) for regression
            or (n_samples, num_classes) for classification.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        self.model.eval()
        x = self.featurise(candidate_points)

        if self._input_normaliser is not None:
            x_np = x.cpu().numpy()
            x_np = self._input_normaliser.transform(x_np)
            x = torch.tensor(x_np, dtype=x.dtype).to(self.device)
        else:
            x = x.to(self.device)

        with torch.no_grad():
            logits = self.model(x)

        outputs = _apply_activation(logits, self.problem_type).cpu().numpy()  # (n, num_classes)

        if self._output_standardiser is not None:
            outputs, _ = self._output_standardiser.inverse_transform(outputs)

        return Predictions(means=outputs)

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
