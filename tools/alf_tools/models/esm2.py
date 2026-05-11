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
import torch.optim as optim
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions
from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from alf_tools.models.utils import get_device

logger = logging.getLogger("alf-tools")


@dataclass
class ESM2ModelConfig:
    """Configuration for ESM-2 model architecture.

    Args:
        model_id: HuggingFace model identifier, e.g. 'facebook/esm2_t6_8M_UR50D'.
        pooling: Strategy for reducing per-token hidden states to a sequence embedding.
        repr_layer: Transformer layer index to extract embeddings from. -1 = final layer.
    """

    model_id: str
    pooling: Literal["mean", "cls", "last_hidden_state"] = "mean"
    repr_layer: int = -1


@dataclass
class ESM2TrainConfig:
    """Configuration for ESM-2 training.

    Args:
        freeze_backbone: When True, train() is a no-op (pure embedding extractor).
        learning_rate: Learning rate for the optimizer.
        optimizer_type: Which optimizer to use ('adam' or 'adamw').
        batch_size: Batch size for training.
        num_epochs: Number of epochs to train for.
        mask_probability: Fraction of non-special tokens to randomly mask (MLM).
        mask_splitting: Tuple of (p_mask, p_random, p_unchanged) probabilities
            for masked token replacement.
        log_frequency: Record epoch metrics every N epochs.
        loss_type: Training objective. 'mlm' masks a random fraction of tokens
            (controlled by mask_probability and mask_splitting) and computes
            cross-entropy over those positions. 'log_likelihood' masks ALL
            non-special tokens with [MASK] and computes cross-entropy over
            all of them, approximating the pseudo-log-likelihood of the sequence.
    """

    freeze_backbone: bool = True
    learning_rate: float = 1e-4
    optimizer_type: Literal["adam", "adamw"] = "adamw"
    batch_size: int = 8
    num_epochs: int = 10
    mask_probability: float = 0.15
    mask_splitting: tuple[float, float, float] = (0.8, 0.1, 0.1)  # mask / random / unchanged
    log_frequency: int = 1
    loss_type: Literal["mlm", "log_likelihood"] = "mlm"


class ESM2Model(BaseModel):
    """ESM-2 protein language model wrapper.

    Loads a pre-trained ESM-2 checkpoint from HuggingFace and exposes it as a
    BaseModel. predict() returns per-sequence embeddings. Optionally fine-tunes
    the backbone with masked language modelling (MLM).
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
        """
        self.name = name
        self.model_config = model_config
        self.train_config = train_config or ESM2TrainConfig()
        self.device = get_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_id)
        self.esm_model = AutoModelForMaskedLM.from_pretrained(model_config.model_id)
        self.esm_model.to(self.device)

        total_params = sum(p.numel() for p in self.esm_model.parameters())
        logger.info(f"ESM-2 loaded: {model_config.model_id} ({total_params:,} parameters)")

        self._epoch_metrics: list[SurrogateEpochMetrics] = []
        self.training_metrics: dict[str, Union[float, int, np.number]] = {}

    def featurise(
        self, inputs: Union[LabelledCandidates, list[Candidate]]
    ) -> dict[str, torch.Tensor]:
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

        encoding = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Compute sequence embeddings for the given candidates.

        Args:
            candidate_points: List of candidates to generate embeddings for.

        Returns:
            Predictions whose means are per-sequence embeddings as a numpy array.
        """
        batch = self.featurise(candidate_points)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        self.esm_model.eval()
        with torch.no_grad():
            outputs = self.esm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden_state = outputs.hidden_states[
            self.model_config.repr_layer
        ]  # (batch, seq_len, hidden_dim)

        if self.model_config.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            embeddings = (hidden_state * mask).sum(1) / mask.sum(1)  # (batch, hidden_dim)
        elif self.model_config.pooling == "cls":
            embeddings = hidden_state[:, 0, :]  # (batch, hidden_dim)
        else:  # last_hidden_state
            embeddings = hidden_state  # (batch, seq_len, hidden_dim)

        return Predictions(means=embeddings.cpu().numpy())

    def _prepare_data_loader(self, data: LabelledCandidates, shuffle: bool = False) -> DataLoader:
        batch = self.featurise(data)
        dataset = TensorDataset(batch["input_ids"], batch["attention_mask"])
        return DataLoader(dataset, batch_size=self.train_config.batch_size, shuffle=shuffle)

    def _mask_tokens(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random token masking for MLM.

        Non-masked positions in labels are set to -100 so CrossEntropyLoss ignores them.
        Special tokens (cls, eos, pad) are never masked.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).

        Raises:
            ValueError: If the tokenizer does not have a mask token.

        Returns:
            Tuple of (masked_input_ids, labels), both of shape (batch, seq_len).
        """
        labels = input_ids.clone()

        special_ids = {
            self.tokenizer.cls_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        } - {None}

        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for sid in special_ids:
            special_tokens_mask |= input_ids.eq(sid)

        # eligible[i, j] is True when position (i, j) may be masked
        eligible = ~special_tokens_mask  # (batch, seq_len)

        # Sample masked positions
        prob_matrix = torch.full(
            input_ids.shape, self.train_config.mask_probability, device=self.device
        )
        prob_matrix.masked_fill_(special_tokens_mask, 0.0)
        masked = torch.bernoulli(prob_matrix).bool()

        # Guarantee at least one token is masked per row so CrossEntropyLoss is never NaN.
        # Pick a random eligible position to avoid systematic positional bias.
        rows_with_no_mask = ~masked.any(dim=1)
        if rows_with_no_mask.any():
            # eligible_float: ineligible positions get 0 weight so they are never picked
            eligible_float = eligible[rows_with_no_mask].float()  # (n_empty, seq_len)

            if eligible_float.sum(dim=1).eq(0).any():
                # Every token in this row is a special token — cannot mask anything.
                # Log a warning; the row's labels will be all -100 (loss contribution = 0).
                logger.warning(
                    "One or more sequences consist entirely of special tokens. "
                    "These rows will contribute zero loss. Check your data pipeline."
                )
                # Zero-weight rows would cause multinomial to raise; fall back to no-op.
                eligible_float = eligible_float.clamp(min=0)  # already 0, kept for clarity
                has_eligible = eligible_float.sum(dim=1) > 0  # (n_empty,)
                if has_eligible.any():
                    picks = torch.multinomial(eligible_float[has_eligible], num_samples=1).squeeze(
                        1
                    )  # (n_has_eligible,)
                    target_rows = rows_with_no_mask.nonzero(as_tuple=True)[0][has_eligible]
                    masked[target_rows, picks] = True
            else:
                picks = torch.multinomial(eligible_float, num_samples=1).squeeze(1)
                target_rows = rows_with_no_mask.nonzero(as_tuple=True)[0]
                masked[target_rows, picks] = True

        labels[~masked] = -100

        masked_input_ids = input_ids.clone()
        if self.tokenizer.mask_token_id is None:
            raise ValueError(
                "Tokenizer has no mask token. Cannot perform MLM masking. "
                "Ensure the tokenizer is initialised with a [MASK] token."
            )
        # masked_input_ids[masked] = self.tokenizer.mask_token_id

        # Apply 80 / 10 / 10 (or specified) replacement split
        masked_indices = masked.nonzero(as_tuple=False)  # (n_masked, 2)

        n_masked = masked_indices.shape[0]
        p_mask, p_random, p_unchanged = self.train_config.mask_splitting
        if n_masked > 0:
            split = torch.rand(n_masked, device=self.device)

            # X %: replace with [MASK]
            replace_with_mask = split < p_mask
            if replace_with_mask.any():
                idx = masked_indices[replace_with_mask]
                masked_input_ids[idx[:, 0], idx[:, 1]] = self.tokenizer.mask_token_id

            # Y %: replace with a uniformly random vocabulary token
            replace_with_random = (split >= p_mask) & (split < (p_mask + p_random))
            if replace_with_random.any():
                idx = masked_indices[replace_with_random]
                random_ids = torch.randint(
                    low=0,
                    high=self.tokenizer.vocab_size,
                    size=(idx.shape[0],),
                    device=self.device,
                )
                masked_input_ids[idx[:, 0], idx[:, 1]] = random_ids

            # Z %: leave unchanged — no write needed, masked_input_ids is already a
            # copy of input_ids. Documented explicitly to make the split complete.

        return masked_input_ids, labels

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
            ValueError: If the tokenizer does not have a mask token.

        Returns:
            Tuple of (masked_input_ids, labels), both of shape (batch, seq_len).
        """
        special_ids = {
            self.tokenizer.cls_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        } - {None}

        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for sid in special_ids:
            special_tokens_mask |= input_ids.eq(sid)

        labels = input_ids.clone()
        labels[special_tokens_mask] = -100

        if self.tokenizer.mask_token_id is None:
            raise ValueError(
                "Tokenizer has no mask token. Cannot perform log-likelihood masking. "
                "Ensure the tokenizer is initialised with a [MASK] token."
            )
        masked_input_ids = input_ids.clone()
        masked_input_ids[~special_tokens_mask] = self.tokenizer.mask_token_id

        return masked_input_ids, labels

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
    ) -> tuple[float, dict]:
        """Train for one epoch.

        Args:
            train_loader: DataLoader for training data.
            optimizer: Optimizer for training.

        Returns:
            Tuple of (average_loss, metrics_dict) where metrics_dict contains
            perplexity and token_accuracy over all labeled positions in the epoch.
            When loss_type='log_likelihood', also includes log_likelihood = -avg_loss.
        """
        self.esm_model.train()
        epoch_losses: list[float] = []
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for raw_ids, raw_mask in train_loader:
            batch_ids = raw_ids.to(self.device)
            batch_mask = raw_mask.to(self.device)

            if self.train_config.loss_type == "mlm":
                masked_ids, labels = self._mask_tokens(batch_ids)
            else:
                masked_ids, labels = self._compute_log_likelihood_labels(batch_ids)

            optimizer.zero_grad()
            outputs = self.esm_model(
                input_ids=masked_ids,
                attention_mask=batch_mask,
                labels=labels,
            )
            outputs.loss.backward()
            optimizer.step()
            epoch_losses.append(outputs.loss.item())

            labeled_positions = labels != -100
            all_logits.append(outputs.logits[labeled_positions].detach().cpu())
            all_labels.append(labels[labeled_positions].detach().cpu())

        avg_train_loss = float(np.mean(epoch_losses))
        logits = torch.cat(all_logits, dim=0)  # (N, vocab_size)
        targets = torch.cat(all_labels, dim=0)  # (N,)
        token_accuracy = (logits.argmax(dim=-1) == targets).float().mean().item()
        train_metrics: dict[str, float] = {
            "perplexity": float(np.exp(avg_train_loss)),
            "token_accuracy": token_accuracy,
        }
        if self.train_config.loss_type == "log_likelihood":
            train_metrics["log_likelihood"] = -avg_train_loss
        return avg_train_loss, train_metrics

    def _validate_epoch(self, val_loader: DataLoader) -> tuple[float, dict]:
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
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for raw_ids, raw_mask in val_loader:
                batch_ids = raw_ids.to(self.device)
                batch_mask = raw_mask.to(self.device)

                if self.train_config.loss_type == "mlm":
                    masked_ids, labels = self._mask_tokens(batch_ids)
                else:
                    masked_ids, labels = self._compute_log_likelihood_labels(batch_ids)

                outputs = self.esm_model(
                    input_ids=masked_ids,
                    attention_mask=batch_mask,
                    labels=labels,
                )
                val_losses.append(outputs.loss.item())

                labeled_positions = labels != -100
                all_logits.append(outputs.logits[labeled_positions].cpu())
                all_labels.append(labels[labeled_positions].cpu())

        avg_val_loss = float(np.mean(val_losses))
        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_labels, dim=0)
        token_accuracy = (logits.argmax(dim=-1) == targets).float().mean().item()
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
        if (epoch + 1) % self.train_config.log_frequency == 0:
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
        else:
            optimizer = torch.optim.Adam(
                self.esm_model.parameters(), lr=self.train_config.learning_rate
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

    def get_training_summary_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Return summary metrics from the most recent train() call.

        Returns:
            Dictionary of training metrics, e.g. final train and validation losses.
        """
        return self.training_metrics
