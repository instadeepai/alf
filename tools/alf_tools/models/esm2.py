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
from typing import Any, Literal

import numpy as np
import torch
import torch.optim as optim
from alf_core import BaseModel, BaseTrainConfig, Candidate, LabelledCandidates, Predictions
from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics
from torch.utils.data import DataLoader, TensorDataset

try:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

from alf_tools.models.utils import get_device

logger = logging.getLogger("alf-tools")


@dataclass
class ESM2ModelConfig:
    """Configuration for ESM-2 model architecture.

    Args:
        model_id: HuggingFace model identifier, e.g. 'facebook/esm2_t6_8M_UR50D'.
        pooling: Strategy for reducing per-token hidden states to a sequence embedding.
        repr_layer: Transformer layer index to extract embeddings from. -1 = final layer.
        max_length: Maximum tokenisation length. Defaults to the tokeniser's model_max_length.
    """

    model_id: str
    pooling: Literal["mean", "cls", "last_hidden_state"] = "mean"
    repr_layer: int = -1
    max_length: int | None = None


@dataclass
class ESM2TrainConfig(BaseTrainConfig):
    """Configuration for ESM-2 training.

    Args:
        freeze_backbone: When True and loss_type='log_likelihood', train() is a no-op.
            Must be True when loss_type='mlp_head' (backbone is always frozen in that mode).
        learning_rate: Learning rate for the optimizer.
        optimizer_type: Which optimizer to use ('adam' or 'adamw').
        batch_size: Batch size for training.
        batch_size_inference: Batch size for predict() and embed(). None defaults to batch_size.
        num_epochs: Number of epochs to train for.
        log_frequency: Record epoch metrics every N epochs.
        max_grad_norm: Maximum norm for gradient clipping. None disables clipping.
        loss_type: Training scheme. 'log_likelihood' masks ALL non-special tokens and
            computes cross-entropy over all of them (self-supervised). 'mlp_head' freezes
            the ESM-2 backbone and trains a linear head on top of sequence embeddings
            using the labels provided to train().
        output_dim: Output dimension of the MLP head. 1 for regression; N for N-class
            classification. Only used when loss_type='mlp_head'.
        mlp_loss: Loss function for MLP head training. 'mse' for regression;
            'cross_entropy' for classification (expects integer class labels).
            Only used when loss_type='mlp_head'.
    """

    freeze_backbone: bool = True
    learning_rate: float = 1e-4
    optimizer_type: Literal["adam", "adamw"] = "adamw"
    batch_size: int = 8
    batch_size_inference: int | None = None
    num_epochs: int = 10
    log_frequency: int = 1
    max_grad_norm: float | None = None
    loss_type: Literal["log_likelihood", "mlp_head"] = "log_likelihood"
    output_dim: int = 1
    mlp_loss: Literal["mse", "cross_entropy"] = "mse"

    def __post_init__(self) -> None:
        """Post-initialization checks for ESM2TrainConfig.

        Raises:
            ValueError: If num_epochs < 1.
            ValueError: If optimizer_type is not 'adam' or 'adamw'.
            ValueError: If loss_type is not 'log_likelihood' or 'mlp_head'.
            ValueError: If loss_type='mlp_head' and freeze_backbone=False.
            ValueError: If mlp_loss is not 'mse' or 'cross_entropy'.
        """
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {self.num_epochs}")
        if self.optimizer_type not in ("adam", "adamw"):
            raise ValueError(
                f"optimizer_type must be 'adam' or 'adamw', got {self.optimizer_type!r}"
            )
        if self.loss_type not in ("log_likelihood", "mlp_head"):
            raise ValueError(
                f"loss_type must be 'log_likelihood' or 'mlp_head', got {self.loss_type!r}"
            )
        if self.loss_type == "mlp_head" and not self.freeze_backbone:
            raise ValueError(
                "mlp_head mode requires freeze_backbone=True. "
                "The ESM-2 backbone is always frozen when training an MLP head."
            )
        if self.mlp_loss not in ("mse", "cross_entropy"):
            raise ValueError(f"mlp_loss must be 'mse' or 'cross_entropy', got {self.mlp_loss!r}")


class ESM2Model(BaseModel):
    """ESM-2 protein language model wrapper.

    Loads a pre-trained ESM-2 checkpoint from HuggingFace and exposes it as a
    BaseModel. predict() returns per-sequence pseudo-log-likelihood scores;
    embed() returns per-sequence embeddings. Optionally fine-tunes the backbone
    with log-likelihood masking, or trains a frozen-backbone linear head for
    regression or classification.
    """

    def __init__(
        self,
        name: str,
        model_config: ESM2ModelConfig,
        train_config: ESM2TrainConfig | None = None,
        device: str | None = None,
    ):
        """Initialize the ESM2Model.

        Args:
            name: Name of the surrogate model.
            model_config: Configuration for the ESM-2 model architecture.
            train_config: Configuration for fine-tuning. Defaults to ESM2TrainConfig().
            device: Device to run on ('cuda', 'cpu', or None for auto-detect).

        Raises:
            ImportError: If transformers package is not available.
        """
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "The 'transformers' package is required for ESM2Model. "
                "Install it with: pip install transformers"
            )
        self.name = name
        self.model_config = model_config
        self.train_config = train_config or ESM2TrainConfig()
        if self.train_config.batch_size_inference is None:
            self.train_config.batch_size_inference = self.train_config.batch_size
        self._validate_model_config()
        self.device = get_device(device)

        self.tokeniser = AutoTokenizer.from_pretrained(self.model_config.model_id)
        self.esm_model = AutoModelForMaskedLM.from_pretrained(self.model_config.model_id)
        self.esm_model.to(self.device)
        _raw_max = self.model_config.max_length or self.tokeniser.model_max_length
        _arch_limit = self.esm_model.config.max_position_embeddings
        if _raw_max > 10_000:
            logger.info(
                f"Tokeniser model_max_length={_raw_max} looks like a sentinel value; "
                f"clamping to architectural limit {_arch_limit}."
            )
            _raw_max = _arch_limit
        self.max_length = _raw_max
        self._validate_esm_config()

        self._head: torch.nn.Linear | None = None
        if self.train_config.loss_type == "mlp_head":
            hidden_dim = self.esm_model.config.hidden_size
            self._head = torch.nn.Linear(hidden_dim, self.train_config.output_dim)
            self._head.to(self.device)
            for param in self.esm_model.parameters():
                param.requires_grad = False

        total_params = sum(p.numel() for p in self.esm_model.parameters())
        logger.info(f"ESM-2 loaded: {self.model_config.model_id} ({total_params:,} parameters)")

        self._epoch_metrics: list[SurrogateEpochMetrics] = []
        self.training_metrics: dict[str, float | int | np.number] = {}

    def _validate_esm_config(self) -> None:
        num_layers = self.esm_model.config.num_hidden_layers + 1  # +1 for embedding
        valid_range = range(-num_layers, num_layers)
        if self.model_config.repr_layer not in valid_range:
            raise ValueError(
                f"repr_layer={self.model_config.repr_layer} is out of range for "
                f"{self.model_config.model_id} which has {num_layers} hidden states "
                f"(valid: {-num_layers} to {num_layers - 1})"
            )

    def _validate_model_config(self) -> None:
        bsi = self.train_config.batch_size_inference or self.train_config.batch_size
        if self.model_config.pooling == "last_hidden_state" and (
            self.train_config.batch_size > 1 or bsi > 1
        ):
            raise ValueError(
                "pooling='last_hidden_state' requires batch_size=1 and batch_size_inference=1. "
                "Each sequence has a different length, so per-sequence hidden-state tensors "
                "have incompatible shapes along the sequence dimension and cannot be "
                "concatenated across mini-batches. Set both to 1 or "
                "use pooling='mean' or pooling='cls' instead."
            )

    def featurise(self, inputs: LabelledCandidates | list[Candidate]) -> dict[str, torch.Tensor]:
        """Tokenize sequences into input tensors for the ESM-2 model.

        Args:
            inputs: Either LabelledCandidates or a list of Candidates to featurise.

        Returns:
            Dictionary with keys 'input_ids' and 'attention_mask' as tensors.

        Raises:
            ValueError: If the input is not LabelledCandidates or list of Candidates.
        """
        if isinstance(inputs, LabelledCandidates):
            sequences = inputs.data
        elif isinstance(inputs, list) and all(isinstance(c, Candidate) for c in inputs):
            sequences = [c.data for c in inputs]
        else:
            raise ValueError("Input must be LabelledCandidates or list of Candidates")

        for seq in sequences:
            if not isinstance(seq, str):
                raise ValueError(
                    f"Expected string sequences, got {type(seq).__name__!r}. "
                    "Ensure Candidate.data contains amino acid sequence strings."
                )

        encoding = self.tokeniser(
            sequences,
            max_length=self.max_length,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if self.tokeniser.unk_token_id is not None:
            if encoding["input_ids"].eq(self.tokeniser.unk_token_id).any():
                logger.warning(
                    "Input sequences contain unknown tokens (UNK). Non-standard amino acid "
                    "characters will be excluded from masking and scoring. Check your sequences."
                )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Compute pseudo-log-likelihood scores for the given candidates.

        Masks all non-special tokens, forward-passes through the model, and
        averages the per-token log-probabilities at the true token identities.
        Higher values indicate sequences the model considers more probable.

        Args:
            candidate_points: List of candidates to score. Must be non-empty.

        Returns:
            Predictions whose means are per-sequence pseudo-log-likelihoods,
            shape (n_candidates,). variances is always None.

        Raises:
            ValueError: If candidate_points is empty.

        Note:
            All sequences are tokenized in one pass before batching the forward pass.
            ``batch_size_inference`` controls only the model forward pass. For very
            large candidate lists, consider chunking externally.
        """
        if not candidate_points:
            raise ValueError("candidate_points must be non-empty")

        batch = self.featurise(candidate_points)
        all_input_ids = batch["input_ids"]
        all_attention_mask = batch["attention_mask"]

        log_likelihoods: list[float] = []
        batch_size = self.train_config.batch_size_inference
        assert batch_size is not None

        self.esm_model.eval()
        with torch.no_grad():
            for start in range(0, len(candidate_points), batch_size):
                input_ids = all_input_ids[start : start + batch_size].to(self.device)
                attention_mask = all_attention_mask[start : start + batch_size].to(self.device)

                masked_ids, labels = self._compute_log_likelihood_labels(input_ids)

                outputs = self.esm_model(
                    input_ids=masked_ids,
                    attention_mask=attention_mask,
                )

                log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
                labeled = labels != -100

                safe_labels = labels.clone()
                safe_labels[~labeled] = 0

                token_log_probs = log_probs.gather(2, safe_labels.unsqueeze(2)).squeeze(2)
                token_log_probs = token_log_probs * labeled.float()
                labeled_counts = labeled.float().sum(dim=1)
                if (labeled_counts == 0).any():
                    raise ValueError(
                        "One or more sequences have no scoreable positions (all special tokens "
                        "after masking). Ensure each sequence contains at least one amino acid "
                        "residue, or increase max_length to avoid full truncation."
                    )
                seq_lls = token_log_probs.sum(dim=1) / labeled_counts

                log_likelihoods.extend(seq_lls.cpu().tolist())

        return Predictions(means=np.array(log_likelihoods, dtype=np.float32))

    def embed(self, candidate_points: list[Candidate]) -> np.ndarray:
        """Compute sequence embeddings using the configured pooling strategy.

        Args:
            candidate_points: List of candidates to embed.

        Returns:
            Numpy array of shape (n_candidates, hidden_dim) for mean or cls pooling,
            or (n_candidates, seq_len, hidden_dim) for last_hidden_state pooling.
            Returns shape (0, hidden_dim) if candidate_points is empty.

        Note:
            All sequences are tokenized in one pass before batching the forward pass.
            ``batch_size_inference`` controls only the model forward pass. For very
            large candidate lists, consider chunking externally.
        """
        if not candidate_points:
            return np.empty((0, self.esm_model.config.hidden_size), dtype=np.float32)

        batch = self.featurise(candidate_points)
        all_input_ids = batch["input_ids"]
        all_attention_mask = batch["attention_mask"]

        all_embeddings: list[torch.Tensor] = []
        batch_size = self.train_config.batch_size_inference
        assert batch_size is not None

        self.esm_model.eval()
        with torch.no_grad():
            for start in range(0, len(candidate_points), batch_size):
                input_ids = all_input_ids[start : start + batch_size].to(self.device)
                attention_mask = all_attention_mask[start : start + batch_size].to(self.device)

                outputs = self.esm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                hidden_state = outputs.hidden_states[self.model_config.repr_layer]

                if self.model_config.pooling == "mean":
                    mask = attention_mask.unsqueeze(-1).float()
                    embeddings = (hidden_state * mask).sum(1) / mask.sum(1)
                elif self.model_config.pooling == "cls":
                    embeddings = hidden_state[:, 0, :]
                else:  # last_hidden_state
                    embeddings = hidden_state

                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0).numpy()

    def _prepare_data_loader(self, data: LabelledCandidates, shuffle: bool = False) -> DataLoader:
        """Create a DataLoader for training or validation.

        Args:
            data: LabelledCandidates containing sequences and (for mlp_head mode) labels.
            shuffle: Whether to shuffle the dataset.

        Returns:
            DataLoader yielding (input_ids, attention_mask) pairs in log_likelihood mode,
            or (input_ids, attention_mask, targets) triples in mlp_head mode.
        """
        batch = self.featurise(data)
        if self.train_config.loss_type == "mlp_head":
            targets = torch.tensor(data.labels, dtype=torch.float32)
            dataset = TensorDataset(batch["input_ids"], batch["attention_mask"], targets)
        else:
            dataset = TensorDataset(batch["input_ids"], batch["attention_mask"])
        return DataLoader(dataset, batch_size=self.train_config.batch_size, shuffle=shuffle)

    def _special_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        special_ids = {
            self.tokeniser.cls_token_id,
            self.tokeniser.eos_token_id,
            self.tokeniser.pad_token_id,
            self.tokeniser.unk_token_id,
        } - {None}
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for sid in special_ids:
            mask |= input_ids.eq(sid)
        return mask

    def _compute_log_likelihood_labels(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare inputs for log-likelihood training.

        All non-special positions are replaced with [MASK] in the input and
        labeled with the original token ID. Special positions (CLS, EOS, PAD)
        receive label -100 so the loss ignores them.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).

        Raises:
            ValueError: If the tokeniser does not have a mask token.

        Returns:
            Tuple of (masked_input_ids, labels), both of shape (batch, seq_len).
        """
        special_tokens_mask = self._special_tokens_mask(input_ids)

        labels = input_ids.clone()
        labels[special_tokens_mask] = -100

        if self.tokeniser.mask_token_id is None:
            raise ValueError(
                "Tokeniser has no mask token. Cannot perform log-likelihood masking. "
                "Ensure the tokeniser is initialised with a [MASK] token."
            )
        masked_input_ids = input_ids.clone()
        masked_input_ids[~special_tokens_mask] = self.tokeniser.mask_token_id

        return masked_input_ids, labels

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
    ) -> tuple[float, dict[str, float]]:
        """Train for one epoch.

        Args:
            train_loader: DataLoader for training data.
            optimizer: Optimizer for training.

        Returns:
            Tuple of (average_loss, metrics_dict) where metrics_dict contains
            perplexity and token_accuracy over all labeled positions in the epoch.
            When loss_type='log_likelihood', also includes log_likelihood = -avg_loss.

        Raises:
            RuntimeError: If training loss becomes NaN or infinite.
            ValueError: If the DataLoader produces no batches.
        """
        self.esm_model.train()
        epoch_losses: list[float] = []
        correct_tokens = 0
        total_tokens = 0

        for raw_ids, raw_mask in train_loader:
            batch_ids = raw_ids.to(self.device)
            batch_mask = raw_mask.to(self.device)

            masked_ids, labels = self._compute_log_likelihood_labels(batch_ids)

            optimizer.zero_grad()
            outputs = self.esm_model(
                input_ids=masked_ids,
                attention_mask=batch_mask,
                labels=labels,
            )
            if not torch.isfinite(outputs.loss):
                raise RuntimeError(
                    f"Training loss is {outputs.loss.item():.6g} at epoch batch. "
                    "Check for degenerate sequences or reduce the learning rate."
                )
            outputs.loss.backward()
            if self.train_config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.esm_model.parameters(), self.train_config.max_grad_norm
                )
            optimizer.step()
            epoch_losses.append(outputs.loss.item())

            labeled_positions = labels != -100
            preds = outputs.logits[labeled_positions].detach().argmax(dim=-1)
            correct_tokens += (preds == labels[labeled_positions]).sum().item()
            total_tokens += labeled_positions.sum().item()

        if not epoch_losses:
            raise ValueError(
                "Training DataLoader produced no batches. Ensure train_data is non-empty "
                "and batch_size does not exceed the number of training sequences."
            )
        avg_train_loss = float(np.mean(epoch_losses))
        token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        train_metrics: dict[str, float] = {
            "perplexity": float(np.exp(avg_train_loss)),
            "token_accuracy": token_accuracy,
        }
        if self.train_config.loss_type == "log_likelihood":
            train_metrics["log_likelihood"] = -avg_train_loss
        return avg_train_loss, train_metrics

    def _validate_epoch(self, val_loader: DataLoader) -> tuple[float, dict[str, float]]:
        """Validate for one epoch.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            Tuple of (average_loss, metrics_dict) where metrics_dict contains
            perplexity and token_accuracy over all labeled positions.
            When loss_type='log_likelihood', also includes log_likelihood = -avg_loss.
        """
        self.esm_model.eval()
        val_losses: list[float] = []
        correct_tokens = 0
        total_tokens = 0

        with torch.no_grad():
            for raw_ids, raw_mask in val_loader:
                batch_ids = raw_ids.to(self.device)
                batch_mask = raw_mask.to(self.device)

                masked_ids, labels = self._compute_log_likelihood_labels(batch_ids)

                outputs = self.esm_model(
                    input_ids=masked_ids,
                    attention_mask=batch_mask,
                    labels=labels,
                )
                val_losses.append(outputs.loss.item())

                labeled_positions = labels != -100
                preds = outputs.logits[labeled_positions].argmax(dim=-1)
                correct_tokens += (preds == labels[labeled_positions]).sum().item()
                total_tokens += labeled_positions.sum().item()

        avg_val_loss = float(np.mean(val_losses))
        token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        val_metrics: dict[str, float] = {
            "perplexity": float(np.exp(avg_val_loss)),
            "token_accuracy": token_accuracy,
        }
        if self.train_config.loss_type == "log_likelihood":
            val_metrics["log_likelihood"] = -avg_val_loss
        return avg_val_loss, val_metrics

    def _record_epoch_metrics(
        self,
        epoch: int,
        avg_train_loss: float,
        train_metrics: dict[str, float],
        avg_val_loss: float | None = None,
        val_metrics: dict[str, float] | None = None,
    ) -> None:
        """Record epoch metrics and log at the configured frequency.

        Args:
            epoch: Current epoch index.
            avg_train_loss: Average training loss for the epoch.
            train_metrics: Dictionary of training metrics (perplexity, token_accuracy,
                and optionally log_likelihood when loss_type='log_likelihood').
            avg_val_loss: Average validation loss for the epoch.
            val_metrics: Dictionary of validation metrics (same keys as train_metrics).
        """
        is_last_epoch = epoch == self.train_config.num_epochs - 1
        if (epoch + 1) % self.train_config.log_frequency == 0 or is_last_epoch:
            additional: dict[str, float] = {}
            if (v := train_metrics.get("perplexity")) is not None:
                additional["train_perplexity"] = float(v)
            if (v := train_metrics.get("token_accuracy")) is not None:
                additional["train_token_accuracy"] = float(v)
            if (v := train_metrics.get("log_likelihood")) is not None:
                additional["train_log_likelihood"] = float(v)
            if val_metrics is not None:
                if (v := val_metrics.get("perplexity")) is not None:
                    additional["val_perplexity"] = float(v)
                if (v := val_metrics.get("token_accuracy")) is not None:
                    additional["val_token_accuracy"] = float(v)
                if (v := val_metrics.get("log_likelihood")) is not None:
                    additional["val_log_likelihood"] = float(v)
            epoch_metric = SurrogateEpochMetrics(
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                additional_metrics=additional,
            )
            self._epoch_metrics.append(epoch_metric)

            log_message = (
                f"Epoch {epoch + 1}/{self.train_config.num_epochs}: train_loss={avg_train_loss:.4f}"
            )
            if avg_val_loss is not None:
                log_message += f", val_loss={avg_val_loss:.4f}"
            logger.info(log_message)

    def train(
        self, train_data: LabelledCandidates, val_data: LabelledCandidates | None = None
    ) -> None:
        """Fine-tune the ESM-2 backbone using the configured training objective.

        Raises:
            AssertionError: Unreachable — if optimizer_type bypasses __post_init__ validation.

        Args:
            train_data: Training data containing sequences.
            val_data: Optional validation data for monitoring training loss.
        """
        self._epoch_metrics = []
        self.training_metrics = {}

        if self.train_config.freeze_backbone:
            return

        logger.info(
            f"Fine-tuning ESM-2 ({self.model_config.model_id}) with {len(train_data)} sequences"
        )

        # Prepare data loaders
        train_loader = self._prepare_data_loader(train_data, shuffle=True)
        val_loader = None
        if val_data is not None and len(val_data) > 0:
            val_loader = self._prepare_data_loader(val_data, shuffle=False)

        # Setup training
        if self.train_config.optimizer_type == "adamw":
            optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                self.esm_model.parameters(), lr=self.train_config.learning_rate
            )
        elif self.train_config.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.esm_model.parameters(), lr=self.train_config.learning_rate
            )
        else:
            raise AssertionError(
                f"Unreachable: optimizer_type={self.train_config.optimizer_type!r} "
                "should have been caught by ESM2TrainConfig.__post_init__"
            )

        avg_train_loss = 0.0
        avg_val_loss: float | None = None
        train_metrics: dict[str, float] = {}
        val_metrics: dict[str, float] = {}

        for epoch in range(self.train_config.num_epochs):
            # Train
            avg_train_loss, train_metrics = self._train_epoch(train_loader, optimizer)

            # Validate
            if val_loader is not None:
                avg_val_loss, val_metrics = self._validate_epoch(val_loader)
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

        self.training_metrics = {"final_train_loss": avg_train_loss}
        self.training_metrics.update({f"final_train_{k}": v for k, v in train_metrics.items()})
        if avg_val_loss is not None:
            self.training_metrics["final_val_loss"] = avg_val_loss
            self.training_metrics.update({f"final_val_{k}": v for k, v in val_metrics.items()})

    def sample(self, *args: Any, **kwargs: Any) -> list[Candidate]:
        """Not implemented for ESM-2.

        Raises:
            NotImplementedError: Always, as sampling is not supported.
        """
        raise NotImplementedError("Sampling is not implemented for this model.")

    def get_epoch_metrics(self) -> list[SurrogateEpochMetrics]:
        """Return per-epoch metrics from the most recent train() call.

        Returns:
            List of SurrogateEpochMetrics, one per logged epoch.
        """
        return self._epoch_metrics

    def get_training_summary_metrics(self) -> dict[str, float | int | np.number]:
        """Return summary metrics from the most recent train() call.

        Returns:
            Dictionary of training metrics, e.g. final train and validation losses.
        """
        return self.training_metrics
