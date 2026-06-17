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

import warnings
from dataclasses import dataclass
from typing import Literal

from alf_core import BaseTrainConfig


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

    - 'linear_head': trainable linear head on top of pooled embeddings (supervised).
      freeze_backbone=True  -> backbone frozen; only the head is trained.
      freeze_backbone=False -> backbone fine-tuned jointly with the head, using
                               backbone_learning_rate for the backbone parameters.
      Active fields: loss_fn ('mse'/'cross_entropy'), output_dim, and (when unfrozen)
      backbone_learning_rate.
    - 'likelihoods': Base ESM-2 backbone, no linear head.
      freeze_backbone=True  -> zero-shot PLL only; train() unavailable.
      freeze_backbone=False -> MLM fine-tuning; loss_fn must be 'mlm'.
      Active fields (when unfrozen): loss_fn ('mlm'), mask_probability, mask_splitting.

    Args:
        mode: Architecture mode. 'linear_head' adds a trainable head on top of the
            backbone. 'likelihoods' uses the base backbone directly; predict() always
            returns PLL scores.
        freeze_backbone: Whether to freeze ESM-2 backbone parameters.
            For 'linear_head': True = train head only; False = fine-tune backbone + head.
            For 'likelihoods': True = zero-shot PLL only; False = MLM fine-tuning.
        loss_fn: Training objective. None = no training intended. 'mse' and 'cross_entropy'
            are only for mode='linear_head'. 'mlm' is only for mode='likelihoods' +
            freeze_backbone=False.
        output_dim: Output dimension of the linear head. Only for mode='linear_head'.
        backbone_learning_rate: Learning rate applied to the ESM-2 backbone parameters
            when mode='linear_head' + freeze_backbone=False. The linear head still uses
            learning_rate. Has no effect when the backbone is frozen.
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
    # linear_head:  True = train head only; False = fine-tune backbone + head
    # likelihoods:  True = zero-shot PLL; False = MLM fine-tuning
    freeze_backbone: bool = True

    # ── training objective ────────────────────────────────────────────
    # None:                   no training (likelihoods + freeze_backbone=True)
    # 'mse', 'cross_entropy': linear_head only
    # 'mlm':                  likelihoods + freeze_backbone=False only
    loss_fn: Literal["mse", "cross_entropy", "mlm"] | None = None

    # ── linear_head only ──────────────────────────────────────────────
    output_dim: int = 1
    # backbone_learning_rate is used only when freeze_backbone=False (fine-tunes backbone).
    backbone_learning_rate: float = 1e-5

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
            if self.loss_fn == "mlm":
                raise ValueError(
                    "loss_fn='mlm' is only valid for mode='likelihoods' with freeze_backbone=False."
                )
            if self.freeze_backbone and self.backbone_learning_rate != 1e-5:
                warnings.warn(
                    "backbone_learning_rate has no effect when freeze_backbone=True; "
                    "the backbone is frozen and only the linear head is trained.",
                    UserWarning,
                    stacklevel=3,
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
