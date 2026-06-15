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
import warnings
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
        seed: Random seed for reproducible linear head initialisation.
    """

    model_id: str
    pooling: Literal["mean", "cls", "last_hidden_state"] = "mean"
    repr_layer: int = -1
    max_length: int | None = None
    seed: int = 42

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

    Set `mode` first — it determines the architecture and which other fields are active:

    - 'linear_head': Frozen backbone + trainable linear head (supervised).
      Active fields: loss_fn ('mse'/'cross_entropy'), output_dim.
    - 'likelihoods': Base ESM-2 backbone, no linear head.
      freeze_backbone=True  -> zero-shot PLL only; train() unavailable.
      freeze_backbone=False -> MLM fine-tuning; loss_fn must be 'mlm'.
      Active fields (when unfrozen): loss_fn ('mlm'), mask_probability, mask_splitting.

    Args:
        mode: Architecture mode. 'linear_head' adds a trainable head on top of a frozen
            backbone. 'likelihoods' uses the base backbone directly; predict() always
            returns PLL scores.
        freeze_backbone: Whether to freeze ESM-2 backbone parameters.
            For 'linear_head': must be True (unfrozen not yet supported).
            For 'likelihoods': True = zero-shot PLL only; False = MLM fine-tuning.
        loss_fn: Training objective. None = no training intended. 'mse' and 'cross_entropy'
            are only for mode='linear_head'. 'mlm' is only for mode='likelihoods' +
            freeze_backbone=False.
        output_dim: Output dimension of the linear head. Only for mode='linear_head'.
        mask_probability: Fraction of eligible tokens to mask per sequence.
            Only active when mode='likelihoods' + freeze_backbone=False.
        mask_splitting: (p_mask, p_random, p_unchanged) 3-way replacement probabilities.
            Must sum to 1.0. Only active when mode='likelihoods' + freeze_backbone=False.
        learning_rate: Learning rate for the optimizer.
        optimizer_type: Which optimizer to use ('adam' or 'adamw').
        batch_size: Batch size for training.
        batch_size_inference: Batch size for embed() and linear-head predict().
            None defaults to batch_size.
        num_epochs: Number of epochs to train for.
        log_frequency: Record epoch metrics every N epochs.
        max_grad_norm: Maximum norm for gradient clipping. None disables clipping.
    """

    # ── mode ──────────────────────────────────────────────────────────
    mode: Literal["linear_head", "likelihoods"] = "linear_head"

    # ── backbone freezing ─────────────────────────────────────────────
    # linear_head:       True only (False not yet supported)
    # likelihoods:  True = zero-shot PLL; False = MLM fine-tuning
    freeze_backbone: bool = True

    # ── training objective ────────────────────────────────────────────
    # None:                   no training (likelihoods + freeze_backbone=True)
    # 'mse', 'cross_entropy': linear_head only
    # 'mlm':                  likelihoods + freeze_backbone=False only
    loss_fn: Literal["mse", "cross_entropy", "mlm"] | None = None

    # ── linear_head only ──────────────────────────────────────────────
    output_dim: int = 1

    # ── likelihoods + freeze_backbone=False only ─────────────────
    mask_probability: float = 0.15
    mask_splitting: tuple[float, float, float] = (0.8, 0.1, 0.1)

    # ── shared ────────────────────────────────────────────────────────
    learning_rate: float = 1e-4
    optimizer_type: Literal["adam", "adamw"] = "adamw"
    batch_size: int = 8
    batch_size_inference: int | None = None
    num_epochs: int = 10
    log_frequency: int = 1
    max_grad_norm: float | None = None

    def __post_init__(self) -> None:
        """Validate ESM2TrainConfig fields.

        Raises:
            ValueError: For invalid field combinations or out-of-range values.
            NotImplementedError: If freeze_backbone=False with mode='linear_head'.
        """
        if self.mode not in ("linear_head", "likelihoods"):
            raise ValueError(f"mode must be 'linear_head' or 'likelihoods', got {self.mode!r}")
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {self.num_epochs}")
        if self.optimizer_type not in ("adam", "adamw"):
            raise ValueError(
                f"optimizer_type must be 'adam' or 'adamw', got {self.optimizer_type!r}"
            )

        if self.mode == "linear_head":
            if not self.freeze_backbone:
                raise NotImplementedError(
                    "freeze_backbone=False is not yet supported for mode='linear_head'. "
                    "Set freeze_backbone=True."
                )
            if self.loss_fn == "mlm":
                raise ValueError(
                    "loss_fn='mlm' is only valid for mode='likelihoods' with "
                    "freeze_backbone=False."
                )

        else:  # likelihoods
            if self.freeze_backbone:
                if self.loss_fn is not None:
                    raise ValueError(
                        "loss_fn cannot be set when freeze_backbone=True in "
                        "mode='likelihoods'; the backbone is frozen and no training "
                        "will occur."
                    )
                if self.mask_probability != 0.15:
                    warnings.warn(
                        "mask_probability has no effect: freeze_backbone=True means no "
                        "training will occur.",
                        UserWarning,
                        stacklevel=3,
                    )
                if self.mask_splitting != (0.8, 0.1, 0.1):
                    warnings.warn(
                        "mask_splitting has no effect: freeze_backbone=True means no "
                        "training will occur.",
                        UserWarning,
                        stacklevel=3,
                    )
            else:
                if self.loss_fn != "mlm":
                    raise ValueError(
                        "ESM-2 base model can only be trained with an MLM loss; other losses "
                        "are not implemented. Set loss_fn='mlm'."
                    )
                if not (0.0 < self.mask_probability < 1.0):
                    raise ValueError(
                        f"mask_probability must be in (0, 1), got {self.mask_probability}"
                    )
                if any(p < 0.0 for p in self.mask_splitting):
                    raise ValueError(
                        f"mask_splitting probabilities must be non-negative, got "
                        f"{self.mask_splitting}"
                    )
                p_sum = sum(self.mask_splitting)
                if abs(p_sum - 1.0) > 1e-6:
                    raise ValueError(
                        f"mask_splitting must sum to 1.0, got {p_sum:.6f} "
                        f"(values: {self.mask_splitting})"
                    )


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
        When mode='linear_head': embeds sequences through the frozen backbone and passes
        them through the linear head. Returns regression values (output_dim=1) or
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
            return self._compute_pll(all_input_ids, all_attention_mask)

    def _compute_pll(
        self, all_input_ids: torch.Tensor, all_attention_mask: torch.Tensor
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

        Returns:
            Predictions: means is a float32 numpy array of per-sequence PLL
                scores (average log-likelihood per residue).

        Raises:
            ValueError: If any sequence has no scoreable residue positions
                (e.g. all special tokens).
        """
        n_candidates = all_input_ids.shape[0]

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

        For mode='linear_head', the whole dataset is tokenised eagerly into a single
        tensor (labels are required). For mode='likelihoods' (MLM), sequences are
        tokenised lazily per batch via `_collate_mlm`, so memory scales with batch size
        rather than corpus size and each batch is padded only to its own longest sequence.

        Args:
            data: LabelledCandidates containing sequences and (for
                mode='linear_head') labels.
            shuffle: Whether to shuffle the dataset.

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

        return DataLoader(
            list(data.data),
            batch_size=self.train_config.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_mlm,
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

        Raises:
            AssertionError: If `loss_fn` is not 'mse' or 'cross_entropy' (should be unreachable
                given `ESM2TrainConfig.__post_init__` validation).
        """
        if self.train_config.loss_fn == "mse":
            return torch.nn.functional.mse_loss(preds.squeeze(-1), targets)
        elif self.train_config.loss_fn == "cross_entropy":
            return torch.nn.functional.cross_entropy(preds, targets.long())
        else:
            raise AssertionError(
                f"Unreachable: loss_fn={self.train_config.loss_fn!r} "
                "should have been caught by ESM2TrainConfig.__post_init__"
            )

    def _mask_tokens(
        self, input_ids: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random token masking for MLM.

        Non-masked positions in labels are set to -100 so CrossEntropyLoss ignores them.
        Special tokens (cls, eos, pad, unk) are never masked.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            generator: Optional torch.Generator for reproducible masking. Pass a seeded
                generator (e.g. during validation) to keep the masking fixed across calls;
                None uses the global RNG.

        Returns:
            Tuple of (masked_input_ids, labels), both of shape (batch, seq_len).

        Raises:
            ValueError: If the tokeniser does not have a mask token.
        """
        if self.tokeniser.mask_token_id is None:
            raise ValueError(
                "Tokeniser has no mask token. Cannot perform MLM masking. "
                "Ensure the tokeniser is initialised with a [MASK] token."
            )

        labels = input_ids.clone()

        special_tokens_mask = self._special_tokens_mask(input_ids)
        eligible = ~special_tokens_mask

        prob_matrix = torch.full(
            input_ids.shape, self.train_config.mask_probability, device=input_ids.device
        )
        prob_matrix.masked_fill_(special_tokens_mask, 0.0)
        masked = torch.bernoulli(prob_matrix, generator=generator).bool()

        rows_with_no_mask = ~masked.any(dim=1)
        if rows_with_no_mask.any():
            eligible_float = eligible[rows_with_no_mask].float()
            if eligible_float.sum(dim=1).eq(0).any():
                logger.warning(
                    "One or more sequences consist entirely of special tokens. "
                    "These rows will contribute zero loss. Check your data pipeline."
                )
                has_eligible = eligible_float.sum(dim=1) > 0
                if has_eligible.any():
                    picks = torch.multinomial(
                        eligible_float[has_eligible], num_samples=1, generator=generator
                    ).squeeze(1)
                    target_rows = rows_with_no_mask.nonzero(as_tuple=True)[0][has_eligible]
                    masked[target_rows, picks] = True
            else:
                picks = torch.multinomial(
                    eligible_float, num_samples=1, generator=generator
                ).squeeze(1)
                target_rows = rows_with_no_mask.nonzero(as_tuple=True)[0]
                masked[target_rows, picks] = True

        labels[~masked] = -100

        masked_input_ids = input_ids.clone()

        masked_indices = masked.nonzero(as_tuple=False)
        n_masked = masked_indices.shape[0]
        p_mask, p_random, _ = self.train_config.mask_splitting
        if n_masked > 0:
            split = torch.rand(n_masked, device=input_ids.device, generator=generator)

            replace_with_mask = split < p_mask
            if replace_with_mask.any():
                idx = masked_indices[replace_with_mask]
                masked_input_ids[idx[:, 0], idx[:, 1]] = self.tokeniser.mask_token_id

            replace_with_random = (split >= p_mask) & (split < (p_mask + p_random))
            if replace_with_random.any():
                idx = masked_indices[replace_with_random]
                # Samples from full vocab including special tokens, matching vanilla BERT.
                random_ids = torch.randint(
                    low=0,
                    high=self.tokeniser.vocab_size,
                    size=(idx.shape[0],),
                    device=input_ids.device,
                    generator=generator,
                )
                masked_input_ids[idx[:, 0], idx[:, 1]] = random_ids

        return masked_input_ids, labels

    def _train_epoch_mlm(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
    ) -> tuple[float, dict[str, float]]:
        """Train the full ESM-2 backbone for one epoch via masked language modelling.

        Args:
            train_loader: DataLoader yielding (input_ids, attention_mask) pairs.
            optimizer: Optimizer over esm_model.parameters().

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
            masked_ids, labels = self._mask_tokens(batch_ids)
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
            # MLM fine-tuning populates perplexity / token_accuracy; in linear head mode
            # train_metrics / val_metrics are always {}.
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
        """Fine-tune ESM-2 using the configured mode and loss function.

        When mode='likelihoods' and freeze_backbone=True, train() raises
        NotImplementedError (zero-shot PLL only; nothing to train). When
        mode='likelihoods' and freeze_backbone=False, fine-tunes the full
        ESM-2 backbone via masked language modelling. When mode='linear_head',
        trains the linear head on top of frozen embeddings.

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

        train_loader = self._prepare_data_loader(train_data, shuffle=True)
        val_loader = None
        if val_data is not None and len(val_data) > 0:
            val_loader = self._prepare_data_loader(val_data, shuffle=False)

        if self.train_config.mode == "linear_head":
            params = self._require_head().parameters()
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
                avg_train_loss, train_metrics = self._train_epoch_mlm(train_loader, optimizer)
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
