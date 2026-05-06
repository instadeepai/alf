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
    """A lightweight stand-in for the output of EsmModel that provides a last_hidden_state
    tensor with the expected shape and dtype.
    """

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
    """Patch AutoTokenizer and EsmModel so no weights are downloaded.

    Yields:
        Tuple[_MockEsmEncoder, _MockTokenizer]: Generator for the mock ESM encoder and tokenizer.
    """
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
    """Small ESM-2 model wired to the mock encoder/tokenizer.

    Args:
        mock_esm (MockEsmEncoder, MockTokenizer): The mock ESM encoder and tokenizer.

    Returns:
        ESM2DropoutModel: _description_
    """
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
    """Generate a small set of labelled candidates for training and testing.

    Returns:
        LabelledCandidates: A set of sequences (4-letter str) and labels (random floats).
    """
    sequences = ["ACDE", "FGHI", "KLMN", "PQRS", "TVWY", "ACDE", "FGHI", "KLMN", "PQRS", "TVWY"]
    candidates = [Candidate(data=seq, modality="sequence") for seq in sequences]
    labels = np.random.randn(len(sequences)).astype(np.float32) + 1.0
    return LabelledCandidates(candidates, labels)


class TestESM2Predictions:
    """Test that the predict method returns Predictions with correct
    variance and empirical distribution shapes.
    """

    def test_predict_returns_variance(self, esm2_model, train_data):
        """Test that predict returns per-candidate variance with matching shape.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
            train_data (LabelledCandidates): Labelled training sequences and targets.
        """
        esm2_model.train(train_data)
        candidates = [Candidate(data="ACDE", modality="sequence")]
        predictions = esm2_model.predict(candidates)

        assert predictions.variances is not None
        assert predictions.variances.shape == predictions.means.shape

    def test_predict_returns_empirical_dist(self, esm2_model, train_data):
        """Test that predict returns an empirical distribution of shape (N, num_mc_samples).

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
            train_data (LabelledCandidates): Labelled training sequences and targets.
        """
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
        """Test that MC Dropout produces non-zero variance for at least one candidate.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
            train_data (LabelledCandidates): Labelled training sequences and targets.
        """
        esm2_model.train(train_data)
        candidates = [Candidate(data="ACDE", modality="sequence")] * 5
        predictions = esm2_model.predict(candidates)

        # Dropout is active so MC samples must vary — at least some variance is non-zero
        assert np.any(predictions.variances > 0)


class TestESM2EmbeddingCache:
    """Test that the embedding cache is populated after training and
    prevents redundant encoding on predict.
    """

    def test_embedding_cache_populated(self, esm2_model, train_data):
        """Test that all training sequences are cached after train() is called.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
            train_data (LabelledCandidates): Labelled training sequences and targets.
        """
        esm2_model.train(train_data)
        training_sequences = {c.data for c in train_data.candidates}
        for seq in training_sequences:
            assert seq in esm2_model._embedder._cache

    def test_no_new_embeddings_on_predict(self, esm2_model, train_data):
        """Test that predict does not re-encode sequences already in the cache.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
            train_data (LabelledCandidates): Labelled training sequences and targets.
        """
        esm2_model.train(train_data)

        # Replace the encoder with a mock — any call here is a cache miss
        spy = MagicMock()
        esm2_model._embedder.encoder = spy

        esm2_model.predict(train_data.candidates)

        spy.assert_not_called()


class TestESM2FrozenEncoder:
    """Test that the ESM-2 encoder is frozen (requires_grad=False)
    to prevent updates during training.
    """

    def test_frozen_encoder(self, esm2_model):
        """Test that all encoder parameters have requires_grad=False.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
        """
        for param in esm2_model._embedder.encoder.parameters():
            assert not param.requires_grad, (
                f"Encoder parameter {param.shape} should have requires_grad=False"
            )


class TestESM2HeadReinitialisation:
    """Test that the regression head is reinitialized with fresh weights on each call to train()."""

    def test_head_reinitialised_on_retrain(self, esm2_model, train_data):
        """Test that calling train() a second time creates a fresh regression head.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
            train_data (LabelledCandidates): Labelled training sequences and targets.
        """
        esm2_model.train(train_data)
        first_head_id = id(esm2_model.head)
        esm2_model.train(train_data)
        assert id(esm2_model.head) != first_head_id


class TestESM2EmbeddingDimension:
    """Test that the embedding dimension of the ESM-2 model matches the
    expected size from the config and mock encoder.
    """

    def test_embedding_dimension_matches_config(self, esm2_model):
        """Test that the embedding width equals the model config's embedding_dim.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
        """
        sequences = ["ACDE", "FGHI"]
        embeddings = esm2_model._embedder.get_embeddings(sequences)
        assert embeddings.shape[-1] == esm2_model.model_config.embedding_dim

    def test_embedding_dimension_matches_hidden_size(self, esm2_model):
        """Test that the embedding width matches the mock encoder's hidden size.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
        """
        embeddings = esm2_model._embedder.get_embeddings(["ACDE"])
        assert embeddings.shape[-1] == _HIDDEN_SIZE


class TestESM2FrozenModelRunsSequences:
    """Test that the frozen ESM-2 model produces embeddings correctly."""

    def test_frozen_model_produces_embeddings(self, esm2_model):
        """Test that the frozen encoder returns a float32 tensor of the correct shape.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
        """
        sequences = ["ACDE", "FGHI", "KLMN"]
        embeddings = esm2_model._embedder.get_embeddings(sequences)
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.dtype == torch.float32
        assert embeddings.shape == (len(sequences), _HIDDEN_SIZE)

    def test_frozen_model_encoder_not_updated_by_embedding_call(self, esm2_model):
        """Test that encoder weights are unchanged after computing embeddings.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
        """
        before = [p.clone() for p in esm2_model._embedder.encoder.parameters()]
        esm2_model._embedder.get_embeddings(["ACDE", "FGHI"])
        after = list(esm2_model._embedder.encoder.parameters())
        for b, a in zip(before, after):
            assert torch.equal(b, a)


class TestESM2EmbeddingsRetrieval:
    """Test the get_embeddings method of the embedder to ensure it
    returns correct embeddings for input sequences.
    """

    def test_get_embeddings_returns_one_row_per_sequence(self, esm2_model):
        """Test that get_embeddings returns exactly one embedding row per input sequence.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
        """
        sequences = ["ACDE", "FGHI", "KLMN", "PQRS"]
        embeddings = esm2_model._embedder.get_embeddings(sequences)
        assert embeddings.shape[0] == len(sequences)

    def test_get_embeddings_returns_float32(self, esm2_model):
        """Test that get_embeddings returns a float32 tensor.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
        """
        embeddings = esm2_model._embedder.get_embeddings(["ACDE"])
        assert embeddings.dtype == torch.float32

    def test_get_embeddings_is_consistent_across_calls(self, esm2_model):
        """Test that get_embeddings returns identical values for
        the same sequence on repeated calls.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
        """
        seq = ["ACDE"]
        first = esm2_model._embedder.get_embeddings(seq)
        second = esm2_model._embedder.get_embeddings(seq)
        assert torch.allclose(first, second)


class TestESM2Featurisation:
    """Test the featurisation method that converts candidates to model input tensors."""

    def test_featurise_from_list_of_candidates(self, esm2_model):
        """Test that featurise returns a tensor of correct shape from a list of Candidates.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
        """
        candidates = [
            Candidate(data="ACDE", modality="sequence"),
            Candidate(data="FGHI", modality="sequence"),
        ]
        embeddings = esm2_model.featurise(candidates)
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (len(candidates), _HIDDEN_SIZE)

    def test_featurise_from_labelled_candidates(self, esm2_model, train_data):
        """Test that featurise returns a tensor of correct shape from LabelledCandidates.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
            train_data (LabelledCandidates): Labelled training sequences and targets.
        """
        embeddings = esm2_model.featurise(train_data)
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (len(train_data), _HIDDEN_SIZE)

    def test_featurise_returns_float32(self, esm2_model, train_data):
        """Test that featurise returns a float32 tensor.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
            train_data (LabelledCandidates): Labelled training sequences and targets.
        """
        embeddings = esm2_model.featurise(train_data)
        assert embeddings.dtype == torch.float32

    def test_featurise_embedding_dim_matches_model_config(self, esm2_model, train_data):
        """Test that the last dimension of featurise output equals embedding_dim in the config.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
            train_data (LabelledCandidates): Labelled training sequences and targets.
        """
        embeddings = esm2_model.featurise(train_data)
        assert embeddings.shape[-1] == esm2_model.model_config.embedding_dim


class TestESM2DropoutActivations:
    """Test that Dropout layers in the regression head are active during training
    and zero out activations as expected.
    """

    def test_dropout_deactivates_expected_fraction_of_activations(self, esm2_model, train_data):
        """Test that Dropout zeros approximately `dropout_rate` of non-zero activations per pass.

        Uses forward pre/post hooks to compare activations entering and leaving each Dropout
        layer. Only elements that were non-zero before dropout are counted, to exclude zeros
        already introduced by the preceding ReLU.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
            train_data (LabelledCandidates): Labelled training sequences and targets.
        """
        esm2_model.train(train_data)
        dropout_rate = esm2_model.model_config.dropout

        # Capture the input and output of each Dropout layer separately so we can
        # count only activations that were non-zero *before* dropout (ReLU may
        # already produce zeros before Dropout is reached).
        pre_dropout: list[torch.Tensor] = []
        post_dropout: list[torch.Tensor] = []

        def _pre_hook(_module, args):
            pre_dropout.append(args[0].detach().clone())

        def _post_hook(_module, _input, output: torch.Tensor) -> None:
            post_dropout.append(output.detach())

        pre_handles, post_handles = [], []
        for m in esm2_model.head.net.modules():
            if isinstance(m, nn.Dropout):
                pre_handles.append(m.register_forward_pre_hook(_pre_hook))
                post_handles.append(m.register_forward_hook(_post_hook))

        esm2_model.head.train()
        x = torch.ones(64, _HIDDEN_SIZE)
        new_zero_fractions: list[float] = []
        for _ in range(50):
            pre_dropout.clear()
            post_dropout.clear()
            with torch.no_grad():
                esm2_model.head(x)
            for pre, post in zip(pre_dropout, post_dropout):
                non_zero_mask = pre != 0
                n_non_zero = non_zero_mask.sum().item()
                if n_non_zero > 0:
                    newly_zeroed = (non_zero_mask & (post == 0)).sum().item()
                    new_zero_fractions.append(newly_zeroed / n_non_zero)

        for h in pre_handles + post_handles:
            h.remove()

        avg_fraction = float(np.mean(new_zero_fractions))
        assert abs(avg_fraction - dropout_rate) < 0.05, (
            f"Expected ~{dropout_rate:.2f} of non-zero activations dropped, got {avg_fraction:.3f}"
        )

    def test_dropout_produces_different_outputs_across_mc_samples(self, esm2_model, train_data):
        """Test that repeated forward passes in train mode produce different outputs due to dropout.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
            train_data (LabelledCandidates): Labelled training sequences and targets.
        """
        esm2_model.train(train_data)
        esm2_model.head.train()
        x = torch.ones(1, _HIDDEN_SIZE)
        outputs = [esm2_model.head(x).item() for _ in range(20)]
        # With dropout active, not all outputs should be identical
        assert len(set(outputs)) > 1


class TestESM2TrainingSummaryMetrics:
    """Test the summary metrics generated for ESM-2 training."""

    def test_summary_metrics_contains_final_train_loss(self, esm2_model, train_data):
        """Test that get_training_summary_metrics includes 'final_train_loss' after training.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
            train_data (LabelledCandidates): Labelled training sequences and targets.
        """
        esm2_model.train(train_data)
        metrics = esm2_model.get_training_summary_metrics()
        assert "final_train_loss" in metrics

    def test_summary_metrics_contains_val_loss_when_val_data_provided(self, esm2_model, train_data):
        """Test that get_training_summary_metrics includes 'final_val_loss' when val data is given.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
            train_data (LabelledCandidates): Labelled training sequences and targets.
        """
        val_data = train_data  # reuse training set as validation for simplicity
        esm2_model.train(train_data, val_data=val_data)
        metrics = esm2_model.get_training_summary_metrics()
        assert "final_val_loss" in metrics

    def test_summary_metrics_omits_val_loss_without_val_data(self, esm2_model, train_data):
        """Test that get_training_summary_metrics omits 'final_val_loss' when no val data is given.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
            train_data (LabelledCandidates): Labelled training sequences and targets.
        """
        esm2_model.train(train_data)
        metrics = esm2_model.get_training_summary_metrics()
        assert "final_val_loss" not in metrics

    def test_epoch_metrics_count_equals_num_epochs(self, esm2_model, train_data):
        """Test that get_epoch_metrics returns exactly one entry per training epoch.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
            train_data (LabelledCandidates): Labelled training sequences and targets.
        """
        esm2_model.train(train_data)
        assert len(esm2_model.get_epoch_metrics()) == esm2_model.train_config.num_epochs

    def test_epoch_metrics_to_dict_contains_epoch_and_train_loss(self, esm2_model, train_data):
        """Test that to_metrics_dict on an epoch entry contains 'epoch' and 'train_loss' keys.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
            train_data (LabelledCandidates): Labelled training sequences and targets.
        """
        esm2_model.train(train_data)
        epoch_dict = esm2_model.get_epoch_metrics()[0].to_metrics_dict()
        assert "epoch" in epoch_dict
        assert "train_loss" in epoch_dict

    def test_epoch_metrics_val_loss_present_when_val_data_provided(self, esm2_model, train_data):
        """Test that val_loss is set on every epoch entry when validation data is provided.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
            train_data (LabelledCandidates): Labelled training sequences and targets.
        """
        esm2_model.train(train_data, val_data=train_data)
        for em in esm2_model.get_epoch_metrics():
            assert em.val_loss is not None

    def test_epoch_metrics_val_loss_absent_without_val_data(self, esm2_model, train_data):
        """Test that val_loss is None on every epoch entry when no validation data is provided.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
            train_data (LabelledCandidates): Labelled training sequences and targets.
        """
        esm2_model.train(train_data)
        for em in esm2_model.get_epoch_metrics():
            assert em.val_loss is None


class TestESM2RecordEpochMetrics:
    """Test the internal method _record_epoch_metrics to
    ensure it correctly stores epoch metrics.
    """

    def test_records_epoch_index(self, esm2_model):
        """Test that _record_epoch_metrics stores the correct epoch index.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
        """
        esm2_model._record_epoch_metrics(7, 0.5, {"mse": 0.5})
        assert esm2_model._epoch_metrics[-1].epoch == 7

    def test_records_train_loss(self, esm2_model):
        """Test that _record_epoch_metrics stores the correct training loss.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
        """
        esm2_model._record_epoch_metrics(0, 1.23, {})
        assert esm2_model._epoch_metrics[-1].train_loss == pytest.approx(1.23)

    def test_records_train_mse_in_additional_metrics(self, esm2_model):
        """Test that train MSE is stored under 'train_mse' in additional_metrics.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
        """
        esm2_model._record_epoch_metrics(0, 0.5, {"mse": 0.42})
        assert esm2_model._epoch_metrics[-1].additional_metrics["train_mse"] == pytest.approx(0.42)

    def test_records_train_spearman_in_additional_metrics(self, esm2_model):
        """Test that train Spearman is stored under 'train_spearman' in additional_metrics.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
        """
        esm2_model._record_epoch_metrics(0, 0.5, {"spearman": 0.88})
        assert esm2_model._epoch_metrics[-1].additional_metrics["train_spearman"] == pytest.approx(
            0.88
        )

    def test_records_val_loss(self, esm2_model):
        """Test that _record_epoch_metrics stores the correct validation loss.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
        """
        esm2_model._record_epoch_metrics(0, 0.5, {}, avg_val_loss=0.77, val_metrics={})
        assert esm2_model._epoch_metrics[-1].val_loss == pytest.approx(0.77)

    def test_records_val_mse_in_additional_metrics(self, esm2_model):
        """Test that validation MSE is stored under 'val_mse' in additional_metrics.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
        """
        esm2_model._record_epoch_metrics(0, 0.5, {}, avg_val_loss=0.3, val_metrics={"mse": 0.11})
        assert esm2_model._epoch_metrics[-1].additional_metrics["val_mse"] == pytest.approx(0.11)

    def test_records_val_spearman_in_additional_metrics(self, esm2_model):
        """Test that validation Spearman is stored under 'val_spearman' in additional_metrics.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
        """
        esm2_model._record_epoch_metrics(
            0, 0.5, {}, avg_val_loss=0.3, val_metrics={"spearman": 0.75}
        )
        assert esm2_model._epoch_metrics[-1].additional_metrics["val_spearman"] == pytest.approx(
            0.75
        )

    def test_omits_val_metrics_when_none(self, esm2_model):
        """Test that val_mse and val_spearman are absent from additional_metrics without val data.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
        """
        esm2_model._record_epoch_metrics(0, 0.5, {})
        additional = esm2_model._epoch_metrics[-1].additional_metrics
        assert "val_mse" not in additional
        assert "val_spearman" not in additional

    def test_appends_on_each_call(self, esm2_model):
        """Test that epoch metrics are being appended correctly.

        Args:
            esm2_model (ESM2DropoutModel): The ESM2DropoutModel instance to test.
        """
        esm2_model._record_epoch_metrics(0, 1.0, {})
        esm2_model._record_epoch_metrics(1, 0.5, {})
        assert len(esm2_model._epoch_metrics) == 2
        assert esm2_model._epoch_metrics[0].epoch == 0
        assert esm2_model._epoch_metrics[1].epoch == 1
