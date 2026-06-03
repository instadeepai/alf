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
from typing import Any, Iterator, Literal

import numpy as np
import torch
import torch.optim as optim
from alf_core import (
    BaseModel,
    BaseTrainConfig,
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

from alf_tools.models.utils import get_device

logger = logging.getLogger("alf-tools")


@dataclass
class ESM2ModelConfig:
    """Configuration for ESM-2 model architecture.

    Args:
        model_id: HuggingFace model identifier, e.g. 'facebook/esm2_t6_8M_UR50D'.
        pooling: Strategy for reducing per-token hidden states to a sequence embedding.
            'mean' averages over all non-padding positions (CLS and EOS included).
            'cls' uses only the first [CLS] token representation.
            'last_hidden_state' returns the full (seq_len, hidden_dim) tensor per sequence.
        repr_layer: Transformer layer index to extract embeddings from. -1 = final layer.
        max_length: Maximum tokenisation length. Defaults to the tokeniser's model_max_length.
    """

    model_id: str
    pooling: Literal["mean", "cls", "last_hidden_state"] = "mean"
    repr_layer: int = -1
    max_length: int | None = None

    def __post_init__(self) -> None:
        """Validate ESM2ModelConfig fields.

        Raises:
            ValueError: If pooling is not a recognised strategy.
        """
        if self.pooling not in ("mean", "cls", "last_hidden_state"):
            raise ValueError(
                f"pooling must be 'mean', 'cls', or 'last_hidden_state', got {self.pooling!r}"
            )


@dataclass
class ESM2TrainConfig(BaseTrainConfig):
    """Configuration for ESM-2 training.

    Args:
        freeze_backbone: Must be True. Unfrozen backbone training is not yet supported.
        learning_rate: Learning rate for the optimizer.
        optimizer_type: Which optimizer to use ('adam' or 'adamw').
        batch_size: Batch size for training.
        batch_size_inference: Batch size for embed() and linear-head predict(). Has no effect on
            zero-shot PLL scoring (scoring_function='pll'); use smaller call-site batches instead.
            None defaults to batch_size.
        num_epochs: Number of epochs to train for.
        log_frequency: Record epoch metrics every N epochs.
        max_grad_norm: Maximum norm for gradient clipping. None disables clipping.
        scoring_function: Scoring function to use. 'linear_head' (default) freezes the backbone
            and trains a linear head via loss_fn. 'pll' skips the head; predict() returns
            per-sequence masked-marginal scores and train() raises NotImplementedError.
        loss_fn: Loss function for linear head training. 'mse' for regression;
            'cross_entropy' for classification. Cross-entropy expects integer class labels in
            [0, output_dim); float labels are truncated with a warning. Only used when
            scoring_function='linear_head'.
        output_dim: Output dimension of the linear head. 1 for regression; N for N-class
            classification. Only used when scoring_function='linear_head'.
    """

    freeze_backbone: bool = True
    learning_rate: float = 1e-4
    optimizer_type: Literal["adam", "adamw"] = "adamw"
    batch_size: int = 8
    batch_size_inference: int | None = None
    num_epochs: int = 10
    log_frequency: int = 1
    max_grad_norm: float | None = None
    scoring_function: Literal["linear_head", "pll"] = "linear_head"
    loss_fn: Literal["mse", "cross_entropy"] = "mse"
    output_dim: int = 1

    def __post_init__(self) -> None:
        """Post-initialization checks for ESM2TrainConfig.

        Raises:
            NotImplementedError: If freeze_backbone=False.
            ValueError: If num_epochs < 1.
            ValueError: If optimizer_type is not 'adam' or 'adamw'.
            ValueError: If loss_fn is not 'mse' or 'cross_entropy'.
            ValueError: If scoring_function is not 'linear_head' or 'pll'.
        """
        if not self.freeze_backbone:
            raise NotImplementedError(
                "freeze_backbone=False is not yet supported. Set freeze_backbone=True."
            )
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {self.num_epochs}")
        if self.optimizer_type not in ("adam", "adamw"):
            raise ValueError(
                f"optimizer_type must be 'adam' or 'adamw', got {self.optimizer_type!r}"
            )
        if self.loss_fn not in ("mse", "cross_entropy"):
            raise ValueError(f"loss_fn must be 'mse' or 'cross_entropy', got {self.loss_fn!r}")
        if self.scoring_function not in ("linear_head", "pll"):
            raise ValueError(
                f"scoring_function must be 'linear_head' or 'pll', got {self.scoring_function!r}"
            )


class ESM2Model(BaseModel):
    """ESM-2 protein language model wrapper.

    Loads a pre-trained ESM-2 checkpoint from HuggingFace and exposes it as a
    BaseModel. predict() returns per-sequence masked-marginal scores
    (scoring_function='pll') or passes embeddings through a trainable linear head
    (scoring_function='linear_head'). embed() returns per-sequence embeddings.
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
        if self.train_config.scoring_function == "linear_head":
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
        if self.model_config.pooling == "last_hidden_state" and (
            self.train_config.batch_size > 1 or self._batch_size_inference > 1
        ):
            raise ValueError(
                "pooling='last_hidden_state' requires batch_size=1 and batch_size_inference=1. "
                "Each sequence has a different length, so per-sequence hidden-state tensors "
                "have incompatible shapes along the sequence dimension and cannot be "
                "concatenated across mini-batches. Set both to 1 or "
                "use pooling='mean' or pooling='cls' instead."
            )
        if (
            self.model_config.pooling == "last_hidden_state"
            and self.train_config.scoring_function == "linear_head"
        ):
            raise ValueError(
                "pooling='last_hidden_state' is not supported with scoring_function='linear_head'. "
                "The linear head requires a fixed-size embedding. "
                "Use pooling='mean' or pooling='cls' instead."
            )

    def _require_head(self) -> torch.nn.Linear:
        """Return the linear head.

        Raises:
            RuntimeError: If the head has not been initialised.
        """
        if self._head is None:
            raise RuntimeError(
                "_head is None; model was not configured with scoring_function='linear_head'"
            )
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

        When scoring_function='pll': computes pseudo-log-likelihood (PLL) by masking one
        residue at a time and recording log P(token_i | all other tokens). Returns the
        mean PLL over residue positions per sequence (higher = more probable). Sequences
        with ≤_PLL_BATCH_THRESHOLD residues are scored in a single batched forward pass;
        longer sequences are scored position-by-position to bound memory usage.
        When scoring_function='linear_head': embeds sequences through the frozen backbone and passes
        them through the linear head. Returns regression values (output_dim=1) or
        argmax class indices (output_dim>1).

        Args:
            candidate_points: List of candidates to score. Must be non-empty.

        Returns:
            Predictions whose means are shape (n_candidates,). variances is always None.

        Raises:
            ValueError: If candidate_points is empty.
            ValueError: If any sequence has no scoreable residue positions.
            RuntimeError: If scoring_function='linear_head' but the head is uninitialised
                (should not happen if __init__ ran without error).
        """
        if not candidate_points:
            raise ValueError("candidate_points must be non-empty")

        batch = self.featurise(candidate_points)
        all_input_ids = batch["input_ids"]
        all_attention_mask = batch["attention_mask"]

        self.esm_model.eval()

        if self.train_config.scoring_function == "linear_head":
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
            return self._predict_zeroshot(all_input_ids, all_attention_mask, len(candidate_points))

    def _predict_zeroshot(
        self, all_input_ids: torch.Tensor, all_attention_mask: torch.Tensor, n_candidates: int
    ) -> Predictions:
        """Compute zero-shot pseudo-log-likelihood (PLL) scores for sequences.

        This method scores each sequence by masking one residue at a time and
        accumulating the log-probability the model assigns to the correct
        residue at that position. The per-sequence score is the mean
        (average) log-probability across all non-special (i.e. amino-acid)
        positions. Two execution modes are used to trade off memory and
        speed:

        - If the number of scoreable residues is ≤ _PLL_BATCH_THRESHOLD, residues
            are masked in a single batched forward pass (one masked position per
            batch row) to leverage GPU parallelism. This creates a forward pass of
            shape (n_residues × padded_seq_len), which can spike GPU memory for
            sequences near the threshold on large models.
        - For longer sequences, positions are masked and scored one-at-a-time
            to avoid excessive memory usage.

        Args:
            all_input_ids: Tensor of shape (n_candidates, seq_len) with token IDs.
            all_attention_mask: Tensor of shape (n_candidates, seq_len) with 1
                for non-padding tokens.
            n_candidates: Number of candidate sequences (batch size).

        Raises:
            ValueError: If any sequence has no scoreable residue positions
                (e.g. all special tokens).

        Returns:
            Predictions: means is a float32 numpy array of per-sequence PLL
                scores (average log-likelihood per residue).
        """
        self.esm_model.eval()

        # PLL: mask one residue at a time, scored per sequence
        log_likelihoods: list[float] = []
        _pll_oom_warned = False
        with torch.no_grad():
            for i in range(n_candidates):
                input_ids_i = all_input_ids[i].unsqueeze(0).to(self.device)  # (1, L)
                attention_mask_i = all_attention_mask[i].unsqueeze(0).to(self.device)

                special_mask = self._special_tokens_mask(input_ids_i[0])
                residue_positions = (~special_mask).nonzero(as_tuple=True)[0].tolist()

                if not residue_positions:
                    raise ValueError(
                        "One or more sequences have no scoreable positions (all special tokens). "
                        "Ensure each sequence contains at least one amino acid residue, "
                        "or increase max_length to avoid full truncation."
                    )

                if len(residue_positions) <= self._PLL_BATCH_THRESHOLD:
                    n = len(residue_positions)
                    seq_len = input_ids_i.shape[1]
                    if not _pll_oom_warned and n * seq_len > 50_000:
                        logger.warning(
                            "Zero-shot PLL batched forward pass: %d residues × %d padded tokens "
                            "= %d tokens. This may cause OOM on memory-constrained devices. "
                            "Reduce ESM2ModelConfig.max_length or call predict() on "
                            "smaller batches.",
                            n,
                            seq_len,
                            n * seq_len,
                        )
                        _pll_oom_warned = True
                    batch_input = input_ids_i.expand(n, -1).clone()  # (N, L)
                    for row, pos in enumerate(residue_positions):
                        batch_input[row, pos] = self.tokeniser.mask_token_id
                    logits = self.esm_model(
                        input_ids=batch_input,
                        attention_mask=attention_mask_i.expand(n, -1),
                    ).logits  # (N, L, vocab_size)
                    ll = sum(
                        torch.nn.functional.log_softmax(logits[row, pos], dim=-1)[
                            input_ids_i[0, pos]
                        ].item()
                        for row, pos in enumerate(residue_positions)
                    )
                else:
                    ll = 0.0
                    for pos in residue_positions:
                        masked_input = input_ids_i.clone()
                        masked_input[0, pos] = self.tokeniser.mask_token_id
                        logits = self.esm_model(
                            input_ids=masked_input,
                            attention_mask=attention_mask_i,
                        ).logits  # (1, L, vocab_size)
                        ll += torch.nn.functional.log_softmax(logits[0, pos], dim=-1)[
                            input_ids_i[0, pos]
                        ].item()

                log_likelihoods.append(ll / len(residue_positions))

        return Predictions(means=np.array(log_likelihoods, dtype=np.float32))

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

    def _prepare_data_loader(self, data: LabelledCandidates, shuffle: bool = False) -> DataLoader:
        """Create a DataLoader for training or validation.

        Args:
            data: LabelledCandidates containing sequences and (for
                scoring_function='linear_head') labels.
            shuffle: Whether to shuffle the dataset.

        Returns:
            DataLoader yielding (input_ids, attention_mask) pairs when scoring_function='pll',
            or (input_ids, attention_mask, targets) triples when scoring_function='linear_head'.
        """
        batch = self.featurise(data)
        if self.train_config.scoring_function == "linear_head":
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
        else:
            dataset = TensorDataset(batch["input_ids"], batch["attention_mask"])
        return DataLoader(dataset, batch_size=self.train_config.batch_size, shuffle=shuffle)

    def _special_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return a boolean tensor True at positions occupied by CLS, EOS, PAD, or UNK tokens."""
        special_ids = {
            self.tokeniser.cls_token_id,
            self.tokeniser.eos_token_id,
            self.tokeniser.pad_token_id,
            self.tokeniser.unk_token_id,
        } - {None}
        special_id_tensor = torch.tensor(
            list(special_ids), dtype=input_ids.dtype, device=input_ids.device
        )
        return torch.isin(input_ids, special_id_tensor)

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

    def _compute_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss between predictions and targets using the configured loss function.

        Returns:
            Scalar loss tensor.
        """
        if self.train_config.loss_fn == "mse":
            return torch.nn.functional.mse_loss(preds.squeeze(-1), targets)
        else:
            return torch.nn.functional.cross_entropy(preds, targets.long())

    def _train_epoch_linear_head(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
    ) -> tuple[float, dict[str, float]]:
        """Train the linear head for one epoch with frozen backbone.

        Args:
            train_loader: DataLoader yielding (input_ids, attention_mask, targets).
            optimizer: Optimizer for the head parameters.

        Returns:
            Tuple of (average_loss, empty metrics_dict).

        Raises:
            RuntimeError: If the loss becomes NaN or infinite.
            ValueError: If the DataLoader produces no batches.
        """
        head = self._require_head()
        self.esm_model.eval()
        head.train()
        epoch_losses: list[float] = []

        for input_ids, attention_mask, targets in train_loader:
            batch_ids = input_ids.to(self.device)
            batch_mask = attention_mask.to(self.device)
            batch_targets = targets.to(self.device)

            with torch.no_grad():
                embeddings = self._embed_batch(batch_ids, batch_mask)

            optimizer.zero_grad()
            preds = head(embeddings)

            loss = self._compute_loss(preds, batch_targets)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Linear head training loss is {loss.item():.6g}. "
                    "Check labels, reduce learning rate, or inspect embeddings."
                )
            loss.backward()
            if self.train_config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(head.parameters(), self.train_config.max_grad_norm)
            optimizer.step()
            epoch_losses.append(loss.item())

        if not epoch_losses:
            raise ValueError(
                "Linear training DataLoader produced no batches. Ensure train_data is non-empty."
            )
        avg_loss = float(np.mean(epoch_losses))
        return avg_loss, {}

    def _validate_epoch_linear_head(self, val_loader: DataLoader) -> tuple[float, dict[str, float]]:
        """Validate the linear head for one epoch.

        Args:
            val_loader: DataLoader yielding (input_ids, attention_mask, targets).

        Returns:
            Tuple of (average_loss, empty metrics_dict).

        Raises:
            RuntimeError: If the model was not configured with scoring_function='linear_head'.
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
                loss = self._compute_loss(preds, batch_targets)
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
            # These keys are reserved for future training modes (e.g. full MLM fine-tuning).
            # In linear head mode, train_metrics / val_metrics are always {}.
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
        """Fine-tune the linear head using the configured loss function.

        When scoring_function='pll', train() raises NotImplementedError (predict uses
        masked-marginal scoring). When scoring_function='linear_head', trains the linear head
        on top of frozen embeddings.

        Args:
            train_data: Training data containing sequences and labels.
            val_data: Optional validation data for monitoring training loss.

        Raises:
            NotImplementedError: If scoring_function='pll'.
            AssertionError: If optimizer_type is invalid (unreachable if __post_init__ ran).
            RuntimeError: If scoring_function='linear_head' but head is uninitialised.
        """
        if self.train_config.scoring_function == "pll":
            raise NotImplementedError(
                "train() requires scoring_function='linear_head'. "
                "Set scoring_function='linear_head' in ESM2TrainConfig to enable "
                "training a linear head, "
                "or use predict() for masked-marginal scoring without training."
            )

        self._epoch_metrics = []
        self.training_metrics = {}

        logger.info(
            f"Fine-tuning ESM-2 ({self.model_config.model_id}) with {len(train_data)} sequences"
        )

        self._require_head()

        train_loader = self._prepare_data_loader(train_data, shuffle=True)
        val_loader = None
        if val_data is not None and len(val_data) > 0:
            val_loader = self._prepare_data_loader(val_data, shuffle=False)

        if self.train_config.optimizer_type == "adamw":
            optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                self._require_head().parameters(), lr=self.train_config.learning_rate
            )
        elif self.train_config.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self._require_head().parameters(), lr=self.train_config.learning_rate
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
            avg_train_loss, train_metrics = self._train_epoch_linear_head(train_loader, optimizer)
            if val_loader is not None:
                avg_val_loss, val_metrics = self._validate_epoch_linear_head(val_loader)
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
