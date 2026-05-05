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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from alf_core import Candidate, LabelledCandidates
from alf_tools.models.esm2 import ESM2DropoutModel, ESM2ModelConfig, ESM2TrainConfig

_HIDDEN_SIZE = 32  # Matches MockEsmEncoder; passed as embedding_dim in fixtures


class _MockEsmOutput:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class _MockEsmEncoder(nn.Module):
    """Lightweight stand-in for EsmModel that avoids downloading weights."""

    def __init__(self, hidden_size: int = _HIDDEN_SIZE):
        super().__init__()
        # One real parameter so requires_grad checks are meaningful
        self._sentinel = nn.Parameter(torch.ones(hidden_size), requires_grad=True)
        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **_):
        b, s = input_ids.shape
        # Non-zero constant embeddings — real variation comes from the head's dropout
        last_hidden_state = torch.ones(b, s, self.hidden_size, device=input_ids.device)
        return _MockEsmOutput(last_hidden_state)


class _MockTokenizer:
    """Lightweight stand-in for AutoTokenizer that avoids downloading vocab files."""

    def __call__(self, sequences: list[str], return_tensors: str = "pt", padding: bool = True, **_):
        batch_size = len(sequences)
        max_len = max(len(s) for s in sequences) + 2  # +2 for CLS / EOS tokens
        return {
            "input_ids": torch.ones(batch_size, max_len, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, max_len, dtype=torch.long),
        }


@pytest.fixture
def mock_esm():
    """Patch AutoTokenizer and EsmModel so no weights are downloaded."""
    mock_encoder = _MockEsmEncoder()
    mock_tokenizer = _MockTokenizer()
    with (
        patch("alf_tools.models.esm2.AutoTokenizer") as mock_tok_cls,
        patch("alf_tools.models.esm2.EsmModel") as mock_model_cls,
    ):
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_encoder
        yield mock_encoder, mock_tokenizer


@pytest.fixture
def esm2_model(mock_esm):
    """Small ESM-2 model wired to the mock encoder/tokenizer."""
    model_config = ESM2ModelConfig(
        embedding_dim=_HIDDEN_SIZE,
        hidden_dim=16,
        num_hidden_layers=1,
        dropout=0.3,
        num_mc_samples=15,
    )
    train_config = ESM2TrainConfig(batch_size=4, num_epochs=2)
    return ESM2DropoutModel(
        name="test_esm2",
        model_config=model_config,
        train_config=train_config,
        device="cpu",
    )


@pytest.fixture
def train_data():
    sequences = ["ACDE", "FGHI", "KLMN", "PQRS", "TVWY", "ACDE", "FGHI", "KLMN", "PQRS", "TVWY"]
    candidates = [Candidate(data=seq, modality="sequence") for seq in sequences]
    labels = np.random.randn(len(sequences)).astype(np.float32) + 1.0
    return LabelledCandidates(candidates, labels)


class TestESM2Predictions:
    def test_predict_returns_variance(self, esm2_model, train_data):
        esm2_model.train(train_data)
        candidates = [Candidate(data="ACDE", modality="sequence")]
        predictions = esm2_model.predict(candidates)

        assert predictions.variances is not None
        assert predictions.variances.shape == predictions.means.shape

    def test_predict_returns_empirical_dist(self, esm2_model, train_data):
        esm2_model.train(train_data)
        candidates = [
            Candidate(data="ACDE", modality="sequence"),
            Candidate(data="FGHI", modality="sequence"),
        ]
        predictions = esm2_model.predict(candidates)

        assert predictions.empirical_dist is not None
        assert predictions.empirical_dist.shape == (
            len(candidates),
            esm2_model.model_config.num_mc_samples,
        )

    def test_uncertainty_nonzero(self, esm2_model, train_data):
        esm2_model.train(train_data)
        candidates = [Candidate(data="ACDE", modality="sequence")] * 5
        predictions = esm2_model.predict(candidates)

        # Dropout is active so MC samples must vary — at least some variance is non-zero
        assert np.any(predictions.variances > 0)


class TestESM2EmbeddingCache:
    def test_embedding_cache_populated(self, esm2_model, train_data):
        esm2_model.train(train_data)
        training_sequences = {c.data for c in train_data.candidates}
        for seq in training_sequences:
            assert seq in esm2_model._embedding_cache

    def test_no_new_embeddings_on_predict(self, esm2_model, train_data):
        esm2_model.train(train_data)

        # Replace the encoder with a mock — any call here is a cache miss
        spy = MagicMock()
        esm2_model.encoder = spy

        esm2_model.predict(train_data.candidates)

        spy.assert_not_called()


class TestESM2FrozenEncoder:
    def test_frozen_encoder(self, esm2_model):
        for param in esm2_model.encoder.parameters():
            assert not param.requires_grad, (
                f"Encoder parameter {param.shape} should have requires_grad=False"
            )
