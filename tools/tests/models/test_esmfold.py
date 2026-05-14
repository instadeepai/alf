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
from alf_core import Candidate, Predictions
from alf_core.dataclasses import LabelledCandidates
from alf_core.dataclasses.candidate import Modality
from alf_core.oracle.oracle import Oracle

from alf_tools.models.esmfold import ESMFoldConfig, ESMFoldModel

MOCK_PTM = 0.7
MOCK_PLDDT = 60.0  # raw; /100 → 0.6


class TestESMFoldConfig:
    """Tests for ESMFoldConfig defaults and field acceptance."""

    def test_default_model_name(self):
        """Default model_name is facebook/esmfold_v1."""
        assert ESMFoldConfig().model_name == "facebook/esmfold_v1"

    def test_default_device(self):
        """Default device is cpu."""
        assert ESMFoldConfig().device == "cpu"

    def test_default_scoring_metric(self):
        """Default scoring_metric is ptm."""
        assert ESMFoldConfig().scoring_metric == "ptm"

    def test_default_combined_ptm_weight(self):
        """Default combined_ptm_weight is 0.5."""
        assert ESMFoldConfig().combined_ptm_weight == 0.5

    def test_default_batch_size(self):
        """Default batch_size is 1."""
        assert ESMFoldConfig().batch_size == 1

    def test_default_chunk_size_is_none(self):
        """Default chunk_size is None."""
        assert ESMFoldConfig().chunk_size is None

    def test_default_low_memory_is_false(self):
        """Default low_memory is False."""
        assert ESMFoldConfig().low_memory is False

    def test_each_valid_scoring_metric_accepted(self):
        """Each of ptm, mean_plddt, combined is accepted."""
        for metric in ("ptm", "mean_plddt", "combined"):
            assert ESMFoldConfig(scoring_metric=metric).scoring_metric == metric

    def test_combined_ptm_weight_boundary_zero(self):
        """combined_ptm_weight=0.0 is accepted."""
        assert ESMFoldConfig(combined_ptm_weight=0.0).combined_ptm_weight == 0.0

    def test_combined_ptm_weight_boundary_one(self):
        """combined_ptm_weight=1.0 is accepted."""
        assert ESMFoldConfig(combined_ptm_weight=1.0).combined_ptm_weight == 1.0

    def test_chunk_size_positive_accepted(self):
        """chunk_size=64 is accepted."""
        assert ESMFoldConfig(chunk_size=64).chunk_size == 64

    def test_batch_size_greater_than_one_accepted(self):
        """batch_size=4 is accepted."""
        assert ESMFoldConfig(batch_size=4).batch_size == 4

    def test_local_path_accepted_as_model_name(self):
        """model_name can be a local filesystem path."""
        cfg = ESMFoldConfig(model_name="/models/esmfold_v1")
        assert cfg.model_name == "/models/esmfold_v1"


# ─── Shared mock fixture ────────────────────────────────────────────────────


@pytest.fixture
def mock_components():
    """Patch HuggingFace ESMFold components. Yields (mock_model, mock_tok, mock_cls, mock_tok_cls).

    Forward pass returns ptm=0.7, plddt=60.0 per residue (scaled mean pLDDT = 0.6).
    """

    def _tok_call(seqs, return_tensors="pt", padding=True):
        n = len(seqs)
        seq_len = max(len(s) for s in seqs) if seqs else 1
        return {
            "input_ids": torch.ones(n, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(n, seq_len, dtype=torch.long),
        }

    def _model_call(**tokens):
        n = tokens["input_ids"].shape[0]
        seq_len = tokens["input_ids"].shape[1]
        out = MagicMock()
        out.ptm = torch.tensor([MOCK_PTM] * n, dtype=torch.float32)
        out.plddt = torch.full((n, seq_len), MOCK_PLDDT, dtype=torch.float32)
        return out

    with (
        patch("alf_tools.models.esmfold.EsmForProteinFolding") as mock_cls,
        patch("alf_tools.models.esmfold.AutoTokenizer") as mock_tok_cls,
    ):
        mock_tok = MagicMock()
        mock_tok.side_effect = _tok_call
        mock_tok_cls.from_pretrained.return_value = mock_tok

        mock_mdl = MagicMock()
        mock_mdl.to.return_value = mock_mdl
        mock_mdl.eval.return_value = mock_mdl
        mock_mdl.esm = MagicMock()
        mock_mdl.esm.encoder = MagicMock()
        mock_mdl.side_effect = _model_call
        mock_cls.from_pretrained.return_value = mock_mdl

        yield mock_mdl, mock_tok, mock_cls, mock_tok_cls


@pytest.fixture
def default_model(mock_components):
    """ESMFoldModel with default ESMFoldConfig and mocked HuggingFace components."""
    return ESMFoldModel(ESMFoldConfig())


@pytest.fixture
def protein_candidates():
    """Five short valid protein sequence candidates."""
    return [
        Candidate(data="ACDEF", modality="sequence"),
        Candidate(data="GHIKL", modality="sequence"),
        Candidate(data="MNPQR", modality="sequence"),
        Candidate(data="STVWY", modality="sequence"),
        Candidate(data="ACGHI", modality="sequence"),
    ]


# ─── Tests ──────────────────────────────────────────────────────────────────


class TestESMFoldModelInit:
    """Tests for ESMFoldModel.__init__ with mocked HuggingFace components."""

    def test_tokenizer_loaded_from_model_name(self, mock_components):
        """Tokenizer is loaded using the configured model_name."""
        _, _, _, mock_tok_cls = mock_components
        ESMFoldModel(ESMFoldConfig(model_name="facebook/esmfold_v1"))
        mock_tok_cls.from_pretrained.assert_called_once_with("facebook/esmfold_v1")

    def test_model_loaded_from_model_name(self, mock_components):
        """EsmForProteinFolding is loaded using the configured model_name."""
        _, _, mock_cls, _ = mock_components
        ESMFoldModel(ESMFoldConfig(model_name="facebook/esmfold_v1"))
        mock_cls.from_pretrained.assert_called_once_with(
            "facebook/esmfold_v1", low_cpu_mem_usage=False
        )

    def test_model_moved_to_configured_device(self, mock_components):
        """Model tensor is moved to the configured device."""
        mock_mdl, _, _, _ = mock_components
        ESMFoldModel(ESMFoldConfig(device="cpu"))
        mock_mdl.to.assert_called_once_with(torch.device("cpu"))

    def test_model_set_to_eval_mode(self, mock_components):
        """Model is placed in eval() mode after loading."""
        mock_mdl, _, _, _ = mock_components
        ESMFoldModel(ESMFoldConfig())
        mock_mdl.eval.assert_called_once()

    def test_chunk_size_applied_to_encoder(self, mock_components):
        """set_chunk_size() is called on the encoder when chunk_size is configured."""
        mock_mdl, _, _, _ = mock_components
        ESMFoldModel(ESMFoldConfig(chunk_size=64))
        mock_mdl.esm.encoder.set_chunk_size.assert_called_once_with(64)

    def test_chunk_size_not_applied_when_none(self, mock_components):
        """set_chunk_size() is not called when chunk_size is None."""
        mock_mdl, _, _, _ = mock_components
        ESMFoldModel(ESMFoldConfig(chunk_size=None))
        mock_mdl.esm.encoder.set_chunk_size.assert_not_called()

    def test_low_memory_passed_as_low_cpu_mem_usage(self, mock_components):
        """low_memory=True sets low_cpu_mem_usage=True in from_pretrained."""
        _, _, mock_cls, _ = mock_components
        ESMFoldModel(ESMFoldConfig(low_memory=True))
        mock_cls.from_pretrained.assert_called_once_with(
            "facebook/esmfold_v1", low_cpu_mem_usage=True
        )

    def test_invalid_combined_ptm_weight_raises(self, mock_components):
        """combined_ptm_weight > 1 raises ValueError before model loading."""
        with pytest.raises(ValueError, match="combined_ptm_weight"):
            ESMFoldModel(ESMFoldConfig(combined_ptm_weight=1.5))

    def test_negative_combined_ptm_weight_raises(self, mock_components):
        """combined_ptm_weight < 0 raises ValueError."""
        with pytest.raises(ValueError, match="combined_ptm_weight"):
            ESMFoldModel(ESMFoldConfig(combined_ptm_weight=-0.1))

    def test_zero_chunk_size_raises(self, mock_components):
        """chunk_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size"):
            ESMFoldModel(ESMFoldConfig(chunk_size=0))

    def test_negative_chunk_size_raises(self, mock_components):
        """chunk_size=-1 raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size"):
            ESMFoldModel(ESMFoldConfig(chunk_size=-1))

    def test_zero_batch_size_raises(self, mock_components):
        """batch_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="batch_size"):
            ESMFoldModel(ESMFoldConfig(batch_size=0))

    def test_not_implemented_methods_raise(self, default_model, protein_candidates):
        """featurise, train, sample, and get_training_summary_metrics raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            default_model.featurise(protein_candidates)
        with pytest.raises(NotImplementedError):
            default_model.train(MagicMock(), MagicMock())
        with pytest.raises(NotImplementedError):
            default_model.sample()
        with pytest.raises(NotImplementedError):
            default_model.get_training_summary_metrics()


class TestESMFoldModelPredict:
    """Tests for ESMFoldModel.predict() output shapes, metric selection, and types."""

    def test_single_sequence_returns_shape_1(self, mock_components, default_model):
        """Single candidate -> Predictions with means shape (1,)."""
        cand = Candidate(data="ACDE", modality="sequence")
        result = default_model.predict([cand])
        assert result.means.shape == (1,)

    def test_batch_5_returns_shape_5(self, mock_components, protein_candidates, default_model):
        """5 candidates -> Predictions with means shape (5,)."""
        result = default_model.predict(protein_candidates)
        assert result.means.shape == (5,)

    def test_returns_predictions_instance(self, mock_components, default_model):
        """predict() returns a Predictions object."""
        cand = Candidate(data="ACDE", modality="sequence")
        result = default_model.predict([cand])
        assert isinstance(result, Predictions)

    def test_means_is_numpy_array(self, mock_components, default_model):
        """means is np.ndarray."""
        cand = Candidate(data="ACDE", modality="sequence")
        result = default_model.predict([cand])
        assert isinstance(result.means, np.ndarray)

    def test_means_are_finite(self, mock_components, protein_candidates, default_model):
        """All means are finite floats."""
        result = default_model.predict(protein_candidates)
        assert np.all(np.isfinite(result.means))

    def test_metric_ptm_values_in_range(self, mock_components, protein_candidates):
        """scoring_metric='ptm' -> values in [0, 1]."""
        mock_mdl, mock_tok, mock_cls, mock_tok_cls = mock_components
        config = ESMFoldConfig(scoring_metric="ptm")
        model = ESMFoldModel(config)
        result = model.predict(protein_candidates)
        assert np.all(result.means >= 0.0)
        assert np.all(result.means <= 1.0)

    def test_metric_mean_plddt_values_in_range(self, mock_components, protein_candidates):
        """scoring_metric='mean_plddt' -> values in [0, 1] after rescaling."""
        mock_mdl, mock_tok, mock_cls, mock_tok_cls = mock_components
        config = ESMFoldConfig(scoring_metric="mean_plddt")
        model = ESMFoldModel(config)
        result = model.predict(protein_candidates)
        assert np.all(result.means >= 0.0)
        assert np.all(result.means <= 1.0)

    def test_metric_combined_values_in_range(self, mock_components, protein_candidates):
        """scoring_metric='combined' -> values in [0, 1]."""
        mock_mdl, mock_tok, mock_cls, mock_tok_cls = mock_components
        config = ESMFoldConfig(scoring_metric="combined")
        model = ESMFoldModel(config)
        result = model.predict(protein_candidates)
        assert np.all(result.means >= 0.0)
        assert np.all(result.means <= 1.0)

    def test_combined_weight_0_equals_mean_plddt(self, mock_components, protein_candidates):
        """combined_ptm_weight=0.0 -> means equal to mean_plddt/100."""
        mock_mdl, mock_tok, mock_cls, mock_tok_cls = mock_components
        config_combined = ESMFoldConfig(scoring_metric="combined", combined_ptm_weight=0.0)
        model_combined = ESMFoldModel(config_combined)
        result_combined = model_combined.predict(protein_candidates)

        mock_mdl, mock_tok, mock_cls, mock_tok_cls = mock_components
        config_plddt = ESMFoldConfig(scoring_metric="mean_plddt")
        model_plddt = ESMFoldModel(config_plddt)
        result_plddt = model_plddt.predict(protein_candidates)

        np.testing.assert_array_almost_equal(result_combined.means, result_plddt.means)

    def test_combined_weight_1_equals_ptm(self, mock_components, protein_candidates):
        """combined_ptm_weight=1.0 -> means equal to ptm."""
        mock_mdl, mock_tok, mock_cls, mock_tok_cls = mock_components
        config_combined = ESMFoldConfig(scoring_metric="combined", combined_ptm_weight=1.0)
        model_combined = ESMFoldModel(config_combined)
        result_combined = model_combined.predict(protein_candidates)

        mock_mdl, mock_tok, mock_cls, mock_tok_cls = mock_components
        config_ptm = ESMFoldConfig(scoring_metric="ptm")
        model_ptm = ESMFoldModel(config_ptm)
        result_ptm = model_ptm.predict(protein_candidates)

        np.testing.assert_array_almost_equal(result_combined.means, result_ptm.means)

    def test_ptm_output_matches_mock_value(self, mock_components, default_model):
        """ptm scores match the mocked value (0.7)."""
        cand = Candidate(data="ACDE", modality="sequence")
        result = default_model.predict([cand])
        np.testing.assert_array_almost_equal(result.means, [MOCK_PTM])

    def test_mean_plddt_output_matches_mock_value(self, mock_components, protein_candidates):
        """mean_plddt scores match the mocked value (60.0/100 = 0.6)."""
        mock_mdl, mock_tok, mock_cls, mock_tok_cls = mock_components
        config = ESMFoldConfig(scoring_metric="mean_plddt")
        model = ESMFoldModel(config)
        result = model.predict(protein_candidates)
        expected = MOCK_PLDDT / 100.0
        np.testing.assert_array_almost_equal(result.means, [expected] * len(protein_candidates))
