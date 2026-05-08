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
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions
from torch.utils.data import DataLoader, TensorDataset
from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics
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
        log_frequency: Record epoch metrics every N epochs.
    """

    freeze_backbone: bool = True
    learning_rate: float = 1e-4
    optimizer_type: Literal["adam", "adamw"] = "adamw"
    batch_size: int = 8
    num_epochs: int = 10
    mask_probability: float = 0.15
    log_frequency: int = 1


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
        """Apply random token masking for MLM. Returns (masked_input_ids, labels).

        Non-masked positions in labels are set to -100 so CrossEntropyLoss ignores them.
        Special tokens (cls, eos, pad) are never masked.
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

        prob_matrix = torch.full(input_ids.shape, self.train_config.mask_probability)
        prob_matrix.masked_fill_(special_tokens_mask, 0.0)

        masked = torch.bernoulli(prob_matrix).bool()

        # Guarantee at least one token is masked per row so CrossEntropyLoss is never NaN.
        # Pick a random eligible position to avoid systematic positional bias.
        rows_with_no_mask = ~masked.any(dim=1)
        if rows_with_no_mask.any():
            eligible = ~special_tokens_mask  # (batch, seq_len)
            for row_idx in rows_with_no_mask.nonzero(as_tuple=True)[0]:
                eligible_positions = eligible[row_idx].nonzero(as_tuple=True)[0]
                if len(eligible_positions) > 0:
                    pick = torch.randint(len(eligible_positions), (1,)).item()
                    masked[row_idx, eligible_positions[pick]] = True

        labels[~masked] = -100

        masked_input_ids = input_ids.clone()
        masked_input_ids[masked] = self.tokenizer.mask_token_id

        return masked_input_ids, labels

    def train(
        self, train_data: LabelledCandidates, val_data: LabelledCandidates | None = None
    ) -> None:
        self._epoch_metrics = []
        self.training_metrics = {}

        if self.train_config.freeze_backbone:
            return

        logger.info(f"Fine-tuning ESM-2 ({self.model_config.model_id}) with {len(train_data)} sequences")

        if self.train_config.optimizer_type == "adamw":
            optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                self.esm_model.parameters(), lr=self.train_config.learning_rate
            )
        else:
            optimizer = torch.optim.Adam(
                self.esm_model.parameters(), lr=self.train_config.learning_rate
            )

        train_loader = self._prepare_data_loader(train_data, shuffle=True)
        val_loader = (
            self._prepare_data_loader(val_data, shuffle=False)
            if val_data is not None and len(val_data) > 0
            else None
        )

        avg_train_loss = 0.0
        avg_val_loss: float | None = None

        for epoch in range(self.train_config.num_epochs):
            self.esm_model.train()
            epoch_losses: list[float] = []

            for batch_ids, batch_mask in train_loader:
                batch_ids = batch_ids.to(self.device)
                batch_mask = batch_mask.to(self.device)
                masked_ids, labels = self._mask_tokens(batch_ids)

                optimizer.zero_grad()
                outputs = self.esm_model(
                    input_ids=masked_ids,
                    attention_mask=batch_mask,
                    labels=labels,
                )
                outputs.loss.backward()
                optimizer.step()
                epoch_losses.append(outputs.loss.item())

            avg_train_loss = float(np.mean(epoch_losses))

            if val_loader is not None:
                self.esm_model.eval()
                val_losses: list[float] = []
                with torch.no_grad():
                    for batch_ids, batch_mask in val_loader:
                        batch_ids = batch_ids.to(self.device)
                        batch_mask = batch_mask.to(self.device)
                        masked_ids, labels = self._mask_tokens(batch_ids)
                        outputs = self.esm_model(
                            input_ids=masked_ids,
                            attention_mask=batch_mask,
                            labels=labels,
                        )
                        val_losses.append(outputs.loss.item())
                avg_val_loss = float(np.mean(val_losses))

            if (epoch + 1) % self.train_config.log_frequency == 0:
                self._epoch_metrics.append(
                    SurrogateEpochMetrics(
                        epoch=epoch,
                        train_loss=avg_train_loss,
                        val_loss=avg_val_loss,
                    )
                )
                logger.info(f"Epoch {epoch + 1}/{self.train_config.num_epochs}: train_loss={avg_train_loss:.4f}")

        self.training_metrics["final_train_loss"] = avg_train_loss
        if avg_val_loss is not None:
            self.training_metrics["final_val_loss"] = avg_val_loss

    def sample(self, *args: Any, **kwargs: Any) -> list[Candidate]:
        raise NotImplementedError("Sampling is not implemented for this model.")

    def get_epoch_metrics(self) -> list[SurrogateEpochMetrics]:
        return self._epoch_metrics

    def get_training_summary_metrics(self) -> dict[str, Union[float, int, np.number]]:
        return self.training_metrics
