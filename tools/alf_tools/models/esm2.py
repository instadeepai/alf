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
from typing import Any, Iterator

import numpy as np
import torch
import torch.optim as optim
from alf_core import (
    BaseModel,
    Candidate,
    LabelledCandidates,
    Predictions,
    SurrogateEpochMetrics,
)
from torch.utils.data import DataLoader, TensorDataset

try:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

from alf_tools.models.esm_utils.config import ESM2ModelConfig, ESM2TrainConfig
from alf_tools.models.esm_utils.loss import compute_supervised_loss, mask_tokens
from alf_tools.models.esm_utils.scoring_function import compute_pll
from alf_tools.models.utils import get_device

logger = logging.getLogger("alf-tools")


class ESM2Model(BaseModel):
    """ESM-2 protein language model wrapper.

    Loads a pre-trained ESM-2 checkpoint from HuggingFace and exposes it as a
    BaseModel. predict() returns per-sequence masked-marginal scores
    (mode='likelihoods') or passes embeddings through a trainable linear head
    (mode='linear_head'). embed() returns per-sequence embeddings.
    """

    _PLL_BATCH_THRESHOLD = 128  # Use batching for shorter sequences

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
            ValueError: If the tokeniser has no mask token.
        """
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "The 'transformers' package is required for ESM2Model. "
                "Install it with: pip install transformers"
            )
        self.name = name
        self.model_config = model_config
        self.train_config = train_config or ESM2TrainConfig()
        self._batch_size_inference = (
            self.train_config.batch_size_inference
            if self.train_config.batch_size_inference is not None
            else self.train_config.batch_size
        )
        self._validate_model_config()
        self.device = get_device(device)

        self.tokeniser = AutoTokenizer.from_pretrained(self.model_config.model_id)
        if self.tokeniser.mask_token_id is None:
            raise ValueError(
                "Tokeniser has no mask token. Cannot perform masked-marginal scoring. "
                "Ensure the tokeniser is initialised with a [MASK] token."
            )
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
        if self.train_config.mode == "linear_head":
            torch.manual_seed(self.model_config.seed)
            hidden_dim = self.esm_model.config.hidden_size
            self._head = torch.nn.Linear(hidden_dim, self.train_config.output_dim)
            self._head.to(self.device)
            if self.train_config.freeze_backbone:
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
        if self.model_config.pooling == "last_hidden_state" and (
            self.train_config.batch_size > 1 or self._batch_size_inference > 1
        ):
            raise ValueError(
                "pooling='last_hidden_state' requires batch_size=1. "
                "Each sequence has a different length, so per-sequence hidden-state tensors "
                "have incompatible shapes along the sequence dimension and cannot be "
                "concatenated across mini-batches. Set batch_size=1 (and batch_size_inference=1 "
                "or None) or use pooling='mean' or pooling='cls' instead."
            )
        if (
            self.model_config.pooling == "last_hidden_state"
            and self.train_config.mode == "linear_head"
        ):
            raise ValueError(
                "pooling='last_hidden_state' is not supported with mode='linear_head'. "
                "The linear head requires a fixed-size embedding. "
                "Use pooling='mean' or pooling='cls' instead."
            )

    def _require_head(self) -> torch.nn.Linear:
        """Return the linear head.

        Raises:
            RuntimeError: If the head has not been initialised.
        """
        if self._head is None:
            raise RuntimeError("_head is None; model was not configured with mode='linear_head'")
        return self._head

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

        effective_max = self.max_length - 2  # account for CLS and EOS tokens
        if any(len(seq) > effective_max for seq in sequences):
            logger.warning(
                "One or more sequences exceed max_length=%d (after reserving 2 positions for "
                "CLS/EOS tokens). They will be silently truncated, which may affect "
                "log-likelihood scores. Increase ESM2ModelConfig.max_length to avoid this.",
                self.max_length,
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
        """Compute predictions for the given candidates.

        When mode='likelihoods': computes pseudo-log-likelihood (PLL) by masking one
        residue at a time and recording log P(token_i | all other tokens). Returns the
        mean PLL over residue positions per sequence (higher = more probable). Sequences
        with ≤_PLL_BATCH_THRESHOLD residues are scored in a single batched forward pass;
        longer sequences are scored position-by-position to bound memory usage.
        When mode='linear_head': embeds sequences through the backbone (frozen or
        fine-tuned) and passes them through the linear head. Returns regression values
        (output_dim=1) or
        argmax class indices (output_dim>1).

        Args:
            candidate_points: List of candidates to score. Must be non-empty.

        Returns:
            Predictions whose means are shape (n_candidates,). variances is always None.

        Raises:
            ValueError: If candidate_points is empty.
            ValueError: If any sequence has no scoreable residue positions.
            RuntimeError: If mode='linear_head' but the head is uninitialised
                (should not happen if __init__ ran without error).

        Note:
            All sequences are tokenised in one pass before batching. For very large
            candidate lists, consider calling predict() on smaller chunks externally.
        """
        if not candidate_points:
            raise ValueError("candidate_points must be non-empty")

        batch = self.featurise(candidate_points)
        all_input_ids = batch["input_ids"]
        all_attention_mask = batch["attention_mask"]

        self.esm_model.eval()

        if self.train_config.mode == "linear_head":
            head = self._require_head()
            head.eval()
            all_preds: list[torch.Tensor] = []
            with torch.no_grad():
                for embeddings in self._iter_embedding_batches(all_input_ids, all_attention_mask):
                    head_out = head(embeddings)
                    if self.train_config.output_dim == 1:
                        preds = head_out.squeeze(-1)
                    else:
                        preds = head_out.argmax(dim=-1).float()
                    all_preds.append(preds.cpu())
            return Predictions(means=torch.cat(all_preds, dim=0).numpy().astype(np.float32))
        else:
            return compute_pll(
                self.esm_model,
                self.tokeniser,
                self.device,
                all_input_ids,
                all_attention_mask,
                self._PLL_BATCH_THRESHOLD,
            )

    def embed(self, candidate_points: list[Candidate]) -> np.ndarray:
        """Compute sequence embeddings using the configured pooling strategy.

        Args:
            candidate_points: List of candidates to embed.

        Returns:
            Numpy array of shape (n_candidates, hidden_dim) for mean or cls pooling,
            or (n_candidates, seq_len, hidden_dim) for last_hidden_state pooling.
            Returns shape (0, hidden_dim) for mean/cls pooling, or (0, 0, hidden_dim) for
            last_hidden_state pooling, if candidate_points is empty.

        Note:
            All sequences are tokenized in one pass before batching the forward pass.
            `batch_size_inference` controls the embed() forward pass batch size but has
            no effect on zero-shot PLL predict(). For very large candidate lists, consider
            chunking externally.

            For last_hidden_state pooling, the returned array has shape
            (n_candidates, max_padded_seq_len, hidden_dim). Positions beyond each
            sequence's EOS token are padding and have non-zero values. Use the
            attention_mask from featurise() to identify valid positions.
        """
        if not candidate_points:
            hidden_dim = self.esm_model.config.hidden_size
            if self.model_config.pooling == "last_hidden_state":
                return np.empty((0, 0, hidden_dim), dtype=np.float32)
            return np.empty((0, hidden_dim), dtype=np.float32)

        batch = self.featurise(candidate_points)
        all_input_ids = batch["input_ids"]
        all_attention_mask = batch["attention_mask"]

        all_embeddings: list[torch.Tensor] = []
        self.esm_model.eval()
        with torch.no_grad():
            for embeddings in self._iter_embedding_batches(all_input_ids, all_attention_mask):
                all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)

    def _prepare_data_loader(
        self,
        data: LabelledCandidates,
        shuffle: bool = False,
        generator: torch.Generator | None = None,
    ) -> DataLoader:
        """Create a DataLoader for training or validation.

        For mode='linear_head', the whole dataset is tokenised eagerly into a single
        tensor (labels are required). For mode='likelihoods' (MLM), sequences are
        tokenised lazily per batch via `_collate_mlm`, so memory scales with batch size
        rather than corpus size and each batch is padded only to its own longest sequence.

        Args:
            data: LabelledCandidates containing sequences and (for
                mode='linear_head') labels.
            shuffle: Whether to shuffle the dataset.
            generator: Optional torch.Generator driving the shuffle order. Passing a seeded
                generator (MLM mode) makes the epoch ordering reproducible; None uses the
                global RNG.

        Returns:
            DataLoader yielding (input_ids, attention_mask) pairs when mode='likelihoods',
            or (input_ids, attention_mask, targets) triples when mode='linear_head'.
        """
        if self.train_config.mode == "linear_head":
            batch = self.featurise(data)
            targets = torch.tensor(data.labels, dtype=torch.float32)
            if self.train_config.loss_fn == "cross_entropy":
                labels_arr = np.asarray(data.labels)
                if not np.all(labels_arr == labels_arr.astype(int)):
                    logger.warning(
                        "loss_fn='cross_entropy' expects integer class labels. "
                        "Non-integer values will be truncated (e.g., 2.7 → 2). "
                        "Pass integer labels or switch to loss_fn='mse' for regression."
                    )
            dataset = TensorDataset(batch["input_ids"], batch["attention_mask"], targets)
            return DataLoader(dataset, batch_size=self.train_config.batch_size, shuffle=shuffle)

        effective_max = self.max_length - 2  # account for CLS and EOS tokens
        if any(len(seq) > effective_max for seq in data.data):
            logger.warning(
                "One or more sequences exceed max_length=%d (after reserving 2 positions for "
                "CLS/EOS tokens). They will be silently truncated during MLM fine-tuning. "
                "Increase ESM2ModelConfig.max_length to avoid this.",
                self.max_length,
            )
        return DataLoader(
            list(data.data),
            batch_size=self.train_config.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_mlm,
            generator=generator,
        )

    def _collate_mlm(self, sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise one batch of raw sequences for MLM, padding to the batch's longest sequence.

        Args:
            sequences: Amino-acid sequence strings for a single batch.

        Returns:
            Tuple of (input_ids, attention_mask) tensors of shape (batch, batch_max_len).
        """
        encoding = self.tokeniser(
            sequences,
            max_length=self.max_length,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return encoding["input_ids"], encoding["attention_mask"]

    def _embed_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run the frozen backbone and pool hidden states for one batch.

        Args:
            input_ids: Tensor of shape (batch, seq_len) on self.device.
            attention_mask: Tensor of shape (batch, seq_len) on self.device.

        Returns:
            Embeddings of shape (batch, hidden_dim) for mean/cls pooling,
            or (batch, seq_len, hidden_dim) for last_hidden_state.
        """
        # output_hidden_states=True returns all N layer states; HuggingFace has no per-layer API.
        outputs = self.esm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_state = outputs.hidden_states[self.model_config.repr_layer]
        return self._pool_hidden_state(hidden_state, attention_mask)

    def _iter_embedding_batches(
        self, all_input_ids: torch.Tensor, all_attention_mask: torch.Tensor
    ) -> Iterator[torch.Tensor]:
        """Yield per-batch embeddings (on self.device) using batch_size_inference."""
        batch_size = self._batch_size_inference
        for start in range(0, all_input_ids.size(0), batch_size):
            input_ids = all_input_ids[start : start + batch_size].to(self.device)
            attention_mask = all_attention_mask[start : start + batch_size].to(self.device)
            yield self._embed_batch(input_ids, attention_mask)

    def _pool_hidden_state(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply the configured pooling strategy to reduce hidden states to sequence embeddings.

        Args:
            hidden_state: Tensor of shape (batch, seq_len, hidden_dim).
            attention_mask: Tensor of shape (batch, seq_len) with 1 for non-padding tokens.
                Note: attention_mask is 1 for CLS, EOS, and amino-acid tokens alike, so
                'mean' pooling includes CLS and EOS token representations in the average.

        Returns:
            Embeddings of shape (batch, hidden_dim) for mean/cls pooling,
            or (batch, seq_len, hidden_dim) for last_hidden_state.
        """
        if self.model_config.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_state * mask).sum(1) / mask.sum(1)
        elif self.model_config.pooling == "cls":
            return hidden_state[:, 0, :]
        else:  # last_hidden_state
            return hidden_state

    def _mask_tokens(
        self, input_ids: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply BERT-style random token masking for MLM.

        Thin wrapper around `esm_utils.loss.mask_tokens` using this model's tokeniser and
        configured `mask_probability` / `mask_splitting`.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            generator: Optional torch.Generator for reproducible masking; None uses the
                global RNG.

        Returns:
            Tuple of (masked_input_ids, labels), both of shape (batch, seq_len).

        Raises:
            ValueError: If the tokeniser does not have a mask token.
        """
        return mask_tokens(
            input_ids,
            self.tokeniser,
            self.train_config.mask_probability,
            self.train_config.mask_splitting,
            generator,
        )

    def _train_epoch_mlm(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        generator: torch.Generator | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Train the full ESM-2 backbone for one epoch via masked language modelling.

        Args:
            train_loader: DataLoader yielding (input_ids, attention_mask) pairs.
            optimizer: Optimizer over esm_model.parameters().
            generator: Optional torch.Generator driving the per-batch masking. Persisting a
                single generator across epochs makes a whole training run reproducible from a
                seed while still drawing fresh masks each epoch; None uses the global RNG.

        Returns:
            Tuple of (avg_loss, {"perplexity": float, "token_accuracy": float}). avg_loss is
            the token-weighted mean cross-entropy over masked positions.

        Raises:
            RuntimeError: If the loss becomes NaN or infinite.
            ValueError: If the DataLoader yields no maskable positions.
        """
        self.esm_model.train()
        total_loss = 0.0
        n_correct = 0
        n_total = 0

        for input_ids, attention_mask in train_loader:
            batch_ids = input_ids.to(self.device)
            batch_mask = attention_mask.to(self.device)

            optimizer.zero_grad()
            masked_ids, labels = self._mask_tokens(batch_ids, generator=generator)
            logits = self.esm_model(
                input_ids=masked_ids, attention_mask=batch_mask
            ).logits  # (B, L, vocab)

            vocab_size = logits.shape[-1]
            flat_logits = logits.view(-1, vocab_size)
            flat_labels = labels.view(-1)
            loss = torch.nn.functional.cross_entropy(flat_logits, flat_labels, ignore_index=-100)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"MLM training loss is {loss.item():.6g}. This can happen when a batch has "
                    "no maskable residue positions (all special tokens) or from an unstable "
                    "learning rate. Check your data or reduce the learning rate."
                )

            loss.backward()
            if self.train_config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.esm_model.parameters(), self.train_config.max_grad_norm
                )
            optimizer.step()

            active = flat_labels != -100
            n_active = int(active.sum().item())
            total_loss += loss.item() * n_active
            preds = flat_logits[active].argmax(dim=-1)
            n_correct += int((preds == flat_labels[active]).sum().item())
            n_total += n_active

        if n_total == 0:
            raise ValueError(
                "MLM training produced no maskable positions. Ensure train_data is non-empty "
                "and contains residue (non-special) tokens."
            )

        avg_loss = total_loss / n_total
        token_accuracy = n_correct / n_total
        self.esm_model.eval()
        return avg_loss, {"perplexity": float(np.exp(avg_loss)), "token_accuracy": token_accuracy}

    def _validate_epoch_mlm(self, val_loader: DataLoader) -> tuple[float, dict[str, float]]:
        """Validate the MLM model for one epoch.

        Args:
            val_loader: DataLoader yielding (input_ids, attention_mask) pairs.

        Returns:
            Tuple of (avg_loss, {"perplexity": float, "token_accuracy": float}). avg_loss is
            the token-weighted mean cross-entropy over masked positions.

        Raises:
            ValueError: If the DataLoader yields no maskable positions.

        Note:
            Validation masking is seeded from `ESM2ModelConfig.seed` so the same positions
            are masked every epoch, making val metrics comparable across epochs rather than
            fluctuating with the random masking draw.
        """
        self.esm_model.eval()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.model_config.seed)
        total_loss = 0.0
        n_correct = 0
        n_total = 0

        with torch.no_grad():
            for input_ids, attention_mask in val_loader:
                batch_ids = input_ids.to(self.device)
                batch_mask = attention_mask.to(self.device)

                masked_ids, labels = self._mask_tokens(batch_ids, generator=generator)
                logits = self.esm_model(input_ids=masked_ids, attention_mask=batch_mask).logits

                vocab_size = logits.shape[-1]
                flat_logits = logits.view(-1, vocab_size)
                flat_labels = labels.view(-1)
                loss = torch.nn.functional.cross_entropy(
                    flat_logits, flat_labels, ignore_index=-100
                )

                active = flat_labels != -100
                n_active = int(active.sum().item())
                total_loss += loss.item() * n_active
                preds = flat_logits[active].argmax(dim=-1)
                n_correct += int((preds == flat_labels[active]).sum().item())
                n_total += n_active

        if n_total == 0:
            raise ValueError(
                "MLM validation produced no maskable positions. Ensure val_data is non-empty "
                "and contains residue (non-special) tokens."
            )

        avg_loss = total_loss / n_total
        token_accuracy = n_correct / n_total
        return avg_loss, {"perplexity": float(np.exp(avg_loss)), "token_accuracy": token_accuracy}

    def _train_epoch_linear_head(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
    ) -> tuple[float, dict[str, float]]:
        """Train the linear head for one epoch.

        When freeze_backbone=False the backbone is fine-tuned jointly with the head:
        embeddings are computed with gradients enabled and the backbone runs in train()
        mode. When frozen, embeddings are computed under torch.no_grad() with the backbone
        in eval() mode and only the head is updated.

        Args:
            train_loader: DataLoader yielding (input_ids, attention_mask, targets).
            optimizer: Optimizer for the head (and, when unfrozen, the backbone).

        Returns:
            Tuple of (average_loss, empty metrics_dict).

        Raises:
            RuntimeError: If the loss becomes NaN or infinite.
            ValueError: If the DataLoader produces no batches.
        """
        head = self._require_head()
        freeze = self.train_config.freeze_backbone
        if freeze:
            self.esm_model.eval()
        else:
            self.esm_model.train()
        head.train()
        epoch_losses: list[float] = []

        for input_ids, attention_mask, targets in train_loader:
            batch_ids = input_ids.to(self.device)
            batch_mask = attention_mask.to(self.device)
            batch_targets = targets.to(self.device)

            if freeze:
                with torch.no_grad():
                    embeddings = self._embed_batch(batch_ids, batch_mask)
            else:
                embeddings = self._embed_batch(batch_ids, batch_mask)

            optimizer.zero_grad()
            preds = head(embeddings)

            loss = compute_supervised_loss(preds, batch_targets, self.train_config.loss_fn)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Linear head training loss is {loss.item():.6g}. "
                    "Check labels, reduce learning rate, or inspect embeddings."
                )
            loss.backward()
            if self.train_config.max_grad_norm is not None:
                clip_params = list(head.parameters())
                if not freeze:
                    clip_params += list(self.esm_model.parameters())
                torch.nn.utils.clip_grad_norm_(clip_params, self.train_config.max_grad_norm)
            optimizer.step()
            epoch_losses.append(loss.item())

        if not epoch_losses:
            raise ValueError(
                "Linear training DataLoader produced no batches. Ensure train_data is non-empty."
            )
        if not freeze:
            self.esm_model.eval()
        avg_loss = float(np.mean(epoch_losses))
        return avg_loss, {}

    def _validate_epoch_linear_head(self, val_loader: DataLoader) -> tuple[float, dict[str, float]]:
        """Validate the linear head for one epoch.

        Args:
            val_loader: DataLoader yielding (input_ids, attention_mask, targets).

        Returns:
            Tuple of (average_loss, empty metrics_dict).

        Raises:
            RuntimeError: If the model was not configured with mode='linear_head'.
            ValueError: If the DataLoader produces no batches.
        """
        head = self._require_head()
        self.esm_model.eval()
        head.eval()
        val_losses: list[float] = []

        with torch.no_grad():
            for input_ids, attention_mask, targets in val_loader:
                batch_ids = input_ids.to(self.device)
                batch_mask = attention_mask.to(self.device)
                batch_targets = targets.to(self.device)

                embeddings = self._embed_batch(batch_ids, batch_mask)
                preds = head(embeddings)
                loss = compute_supervised_loss(preds, batch_targets, self.train_config.loss_fn)
                val_losses.append(loss.item())

        if not val_losses:
            raise ValueError(
                "Linear validation DataLoader produced no batches. Ensure val_data is non-empty."
            )
        avg_loss = float(np.mean(val_losses))
        return avg_loss, {}

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
            train_metrics: Dictionary of training metrics.
            avg_val_loss: Average validation loss for the epoch.
            val_metrics: Dictionary of validation metrics.
        """
        is_last_epoch = epoch == self.train_config.num_epochs - 1
        if (epoch + 1) % self.train_config.log_frequency == 0 or is_last_epoch:
            additional: dict[str, float] = {}
            # MLM fine-tuning populates perplexity / token_accuracy; in linear head mode
            # train_metrics / val_metrics are always {}.
            if (v := train_metrics.get("perplexity")) is not None:
                additional["train_perplexity"] = float(v)
            if (v := train_metrics.get("token_accuracy")) is not None:
                additional["train_token_accuracy"] = float(v)
            if val_metrics is not None:
                if (v := val_metrics.get("perplexity")) is not None:
                    additional["val_perplexity"] = float(v)
                if (v := val_metrics.get("token_accuracy")) is not None:
                    additional["val_token_accuracy"] = float(v)
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
        """Fine-tune ESM-2 using the configured mode and loss function.

        When mode='likelihoods' and freeze_backbone=True, train() raises
        NotImplementedError (zero-shot PLL only; nothing to train). When
        mode='likelihoods' and freeze_backbone=False, fine-tunes the full
        ESM-2 backbone via masked language modelling. When mode='linear_head',
        trains the linear head (with frozen backbone) or fine-tunes the backbone
        jointly with the head when freeze_backbone=False.

        Args:
            train_data: Training data containing sequences and (for linear_head) labels.
                MLM fine-tuning ignores labels, but LabelledCandidates still requires them —
                pass placeholder values.
            val_data: Optional validation data for monitoring training loss.

        Raises:
            NotImplementedError: If mode='likelihoods' and freeze_backbone=True.
            ValueError: If mode='linear_head' and loss_fn is None.
            AssertionError: If optimizer_type is invalid (unreachable if __post_init__ ran).
            RuntimeError: If mode='linear_head' but head is uninitialised.
        """
        if self.train_config.mode == "likelihoods" and self.train_config.freeze_backbone:
            raise NotImplementedError(
                "train() is not available when mode='likelihoods' and "
                "freeze_backbone=True. The backbone is frozen and there is no linear head "
                "to train. Set freeze_backbone=False and loss_fn='mlm' to enable MLM "
                "fine-tuning, or call predict() directly for zero-shot PLL scoring."
            )
        if self.train_config.mode == "linear_head" and self.train_config.loss_fn is None:
            raise ValueError(
                "loss_fn must be set for mode='linear_head' training. "
                "Choose 'mse' for regression or 'cross_entropy' for classification."
            )

        self._epoch_metrics = []
        self.training_metrics = {}

        logger.info(
            f"Fine-tuning ESM-2 ({self.model_config.model_id}) with {len(train_data)} sequences"
        )

        # Seed both the MLM shuffle order and the per-batch masking from model_config.seed so
        # a whole fine-tuning run is reproducible, while still drawing a fresh mask each epoch
        # as the generators advance. Two generators are used because the DataLoader sampler
        # requires a CPU generator whereas masking runs on self.device. Unused for linear_head.
        mlm_generator: torch.Generator | None = None
        shuffle_generator: torch.Generator | None = None
        if self.train_config.mode == "likelihoods":
            mlm_generator = torch.Generator(device=self.device)
            mlm_generator.manual_seed(self.model_config.seed)
            shuffle_generator = torch.Generator()
            shuffle_generator.manual_seed(self.model_config.seed)

        train_loader = self._prepare_data_loader(
            train_data, shuffle=True, generator=shuffle_generator
        )
        val_loader = None
        if val_data is not None and len(val_data) > 0:
            val_loader = self._prepare_data_loader(val_data, shuffle=False)

        if self.train_config.mode == "linear_head":
            # Head trains at learning_rate; when unfrozen, the backbone is fine-tuned in a
            # separate param group at backbone_learning_rate (typically lower).
            params: Any = [
                {"params": self._require_head().parameters()},
            ]
            if not self.train_config.freeze_backbone:
                params.append({
                    "params": self.esm_model.parameters(),
                    "lr": self.train_config.backbone_learning_rate,
                })
        else:
            params = self.esm_model.parameters()

        if self.train_config.optimizer_type == "adamw":
            optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                params, lr=self.train_config.learning_rate
            )
        elif self.train_config.optimizer_type == "adam":
            optimizer = torch.optim.Adam(params, lr=self.train_config.learning_rate)
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
            if self.train_config.mode == "linear_head":
                avg_train_loss, train_metrics = self._train_epoch_linear_head(
                    train_loader, optimizer
                )
                if val_loader is not None:
                    avg_val_loss, val_metrics = self._validate_epoch_linear_head(val_loader)
            else:  # likelihoods, freeze_backbone=False
                avg_train_loss, train_metrics = self._train_epoch_mlm(
                    train_loader, optimizer, generator=mlm_generator
                )
                if val_loader is not None:
                    avg_val_loss, val_metrics = self._validate_epoch_mlm(val_loader)

            if val_loader is not None:
                self._record_epoch_metrics(
                    epoch, avg_train_loss, train_metrics, avg_val_loss, val_metrics
                )
            else:
                self._record_epoch_metrics(epoch, avg_train_loss, train_metrics)

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
