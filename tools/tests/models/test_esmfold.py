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

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from alf_core import Candidate, Predictions
from alf_core.dataclasses import LabelledCandidates, State
from alf_core.dataset.base_dataset import BaseDataset, BaseDatasetConfig
from alf_core.model.base_model import BaseModel
from alf_core.oracle.oracle import Oracle
from alf_core.surrogate.surrogate import Surrogate
from alf_core.utils.enums import ProblemType
from alf_tools.models.esmfold import ESMFoldModel, ESMFoldModelConfig

MOCK_PTM = 0.7
MOCK_PLDDT = 0.6  # EsmForProteinFolding returns plddt already in [0, 1]


class TestESMFoldModelConfig:
    """Tests for ESMFoldModelConfig defaults and field acceptance."""

    def test_default_model_name(self):
        """Default model_name is facebook/esmfold_v1."""
        assert ESMFoldModelConfig().model_name == "facebook/esmfold_v1"

    def test_default_device(self):
        """Default device is cpu."""
        assert ESMFoldModelConfig().device == "cpu"

    def test_default_scoring_metric(self):
        """Default scoring_metric is ptm."""
        assert ESMFoldModelConfig().scoring_metric == "ptm"

    def test_default_combined_ptm_weight(self):
        """Default combined_ptm_weight is 0.5."""
        assert ESMFoldModelConfig().combined_ptm_weight == 0.5

    def test_default_batch_size(self):
        """Default batch_size is 1."""
        assert ESMFoldModelConfig().batch_size == 1

    def test_default_chunk_size_is_none(self):
        """Default chunk_size is None."""
        assert ESMFoldModelConfig().chunk_size is None

    def test_default_low_memory_is_false(self):
        """Default low_memory is False."""
        assert ESMFoldModelConfig().low_memory is False

    def test_each_valid_scoring_metric_accepted(self):
        """Each of ptm, mean_plddt, combined is accepted."""
        for metric in ("ptm", "mean_plddt", "combined"):
            assert ESMFoldModelConfig(scoring_metric=metric).scoring_metric == metric

    def test_combined_ptm_weight_boundary_zero(self):
        """combined_ptm_weight=0.0 is accepted."""
        assert ESMFoldModelConfig(combined_ptm_weight=0.0).combined_ptm_weight == 0.0

    def test_combined_ptm_weight_boundary_one(self):
        """combined_ptm_weight=1.0 is accepted."""
        assert ESMFoldModelConfig(combined_ptm_weight=1.0).combined_ptm_weight == 1.0

    def test_chunk_size_positive_accepted(self):
        """chunk_size=64 is accepted."""
        assert ESMFoldModelConfig(chunk_size=64).chunk_size == 64

    def test_batch_size_greater_than_one_accepted(self):
        """batch_size=4 is accepted."""
        assert ESMFoldModelConfig(batch_size=4).batch_size == 4

    def test_local_path_accepted_as_model_name(self):
        """model_name can be a local filesystem path."""
        cfg = ESMFoldModelConfig(model_name="/models/esmfold_v1")
        assert cfg.model_name == "/models/esmfold_v1"


# ─── Shared mock fixture ────────────────────────────────────────────────────


@pytest.fixture
def mock_components():
    """Patch HuggingFace ESMFold components.

    Forward pass returns ptm=0.7, plddt=0.6 per residue (mean pLDDT = 0.6).

    Yields:
        tuple: (mock_model, mock_tok, mock_cls, mock_tok_cls).
    """

    def _tok_call(seqs, return_tensors="pt", padding=True, add_special_tokens=False):
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
        out.ptm = torch.tensor(MOCK_PTM, dtype=torch.float32)  # 0-dim scalar
        out.plddt = torch.full((n, seq_len, 37), MOCK_PLDDT, dtype=torch.float32)  # (B, L, 37)
        out.atom37_atom_exists = torch.ones(n, seq_len, 37, dtype=torch.bool)  # all atoms exist
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
    """ESMFoldModel with default ESMFoldModelConfig and mocked HuggingFace components.

    Returns:
        ESMFoldModel: model instance with mocked HuggingFace components.
    """
    return ESMFoldModel(ESMFoldModelConfig())


@pytest.fixture
def protein_candidates():
    """Five short valid protein sequence candidates.

    Returns:
        list[Candidate]: list of five short amino acid sequence candidates.
    """
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
        ESMFoldModel(ESMFoldModelConfig(model_name="facebook/esmfold_v1"))
        mock_tok_cls.from_pretrained.assert_called_once_with("facebook/esmfold_v1")

    def test_model_loaded_from_model_name(self, mock_components):
        """EsmForProteinFolding is loaded using the configured model_name."""
        _, _, mock_cls, _ = mock_components
        ESMFoldModel(ESMFoldModelConfig(model_name="facebook/esmfold_v1"))
        mock_cls.from_pretrained.assert_called_once_with(
            "facebook/esmfold_v1", low_cpu_mem_usage=False
        )

    def test_model_moved_to_configured_device(self, mock_components):
        """Model tensor is moved to the configured device."""
        mock_mdl, _, _, _ = mock_components
        ESMFoldModel(ESMFoldModelConfig(device="cpu"))
        mock_mdl.to.assert_called_once_with(torch.device("cpu"))

    def test_model_set_to_eval_mode(self, mock_components):
        """Model is placed in eval() mode after loading."""
        mock_mdl, _, _, _ = mock_components
        ESMFoldModel(ESMFoldModelConfig())
        mock_mdl.eval.assert_called_once()

    def test_chunk_size_applied_to_encoder(self, mock_components):
        """set_chunk_size() is called on the encoder when chunk_size is configured."""
        mock_mdl, _, _, _ = mock_components
        ESMFoldModel(ESMFoldModelConfig(chunk_size=64))
        mock_mdl.esm.encoder.set_chunk_size.assert_called_once_with(64)

    def test_chunk_size_not_applied_when_none(self, mock_components):
        """set_chunk_size() is not called when chunk_size is None."""
        mock_mdl, _, _, _ = mock_components
        ESMFoldModel(ESMFoldModelConfig(chunk_size=None))
        mock_mdl.esm.encoder.set_chunk_size.assert_not_called()

    def test_low_memory_passed_as_low_cpu_mem_usage(self, mock_components):
        """low_memory=True sets low_cpu_mem_usage=True in from_pretrained."""
        _, _, mock_cls, _ = mock_components
        ESMFoldModel(ESMFoldModelConfig(low_memory=True))
        mock_cls.from_pretrained.assert_called_once_with(
            "facebook/esmfold_v1", low_cpu_mem_usage=True
        )

    def test_invalid_combined_ptm_weight_raises(self, mock_components):
        """combined_ptm_weight > 1 raises ValueError before model loading."""
        with pytest.raises(ValueError, match="combined_ptm_weight"):
            ESMFoldModel(ESMFoldModelConfig(combined_ptm_weight=1.5))

    def test_negative_combined_ptm_weight_raises(self, mock_components):
        """combined_ptm_weight < 0 raises ValueError."""
        with pytest.raises(ValueError, match="combined_ptm_weight"):
            ESMFoldModel(ESMFoldModelConfig(combined_ptm_weight=-0.1))

    def test_zero_chunk_size_raises(self, mock_components):
        """chunk_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size"):
            ESMFoldModel(ESMFoldModelConfig(chunk_size=0))

    def test_negative_chunk_size_raises(self, mock_components):
        """chunk_size=-1 raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size"):
            ESMFoldModel(ESMFoldModelConfig(chunk_size=-1))

    def test_zero_batch_size_raises(self, mock_components):
        """batch_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="batch_size"):
            ESMFoldModel(ESMFoldModelConfig(batch_size=0))

    def test_batch_size_gt_1_with_ptm_raises(self, mock_components):
        """batch_size > 1 with scoring_metric='ptm' raises ValueError."""
        with pytest.raises(ValueError, match="batch_size > 1 is not supported"):
            ESMFoldModel(ESMFoldModelConfig(batch_size=2, scoring_metric="ptm"))

    def test_batch_size_gt_1_with_combined_raises(self, mock_components):
        """batch_size > 1 with scoring_metric='combined' raises ValueError."""
        with pytest.raises(ValueError, match="batch_size > 1 is not supported"):
            ESMFoldModel(ESMFoldModelConfig(batch_size=2, scoring_metric="combined"))

    def test_esm_backbone_converted_to_float32_on_cpu(self, mock_components):
        """On CPU, model.esm.float() is called to fix fp16 emulation artifacts."""
        mock_mdl, _, _, _ = mock_components
        ESMFoldModel(ESMFoldModelConfig(device="cpu"))
        mock_mdl.esm.float.assert_called_once()

    def test_esm_backbone_not_converted_on_cuda(self, mock_components):
        """On non-CPU devices, model.esm.float() is not called."""
        mock_mdl, _, _, _ = mock_components
        ESMFoldModel(ESMFoldModelConfig(device="cuda"))
        mock_mdl.esm.float.assert_not_called()

    def test_featurise_raises_not_implemented(self, default_model, protein_candidates):
        """featurise() raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            default_model.featurise(protein_candidates)

    def test_train_raises_not_implemented(self, default_model):
        """train() raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            default_model.train(MagicMock(), MagicMock())

    def test_sample_raises_not_implemented(self, default_model):
        """sample() raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            default_model.sample()

    def test_get_training_summary_metrics_returns_empty_dict(self, default_model):
        """get_training_summary_metrics() returns empty dict (inference-only model)."""
        assert default_model.get_training_summary_metrics() == {}


class TestESMFoldModelPredict:
    """Tests for ESMFoldModel.predict() output shapes, metric selection, and types."""

    def test_single_sequence_returns_shape_1(self, mock_components, default_model):
        """Single candidate -> Predictions with means shape (1,)."""
        cand = Candidate(data="ACDE", modality="sequence")
        result = default_model.predict([cand])
        assert result.means.shape == (1,)
        assert result.means.dtype in (np.float32, np.float64)

    def test_batch_5_returns_shape_5(self, mock_components, protein_candidates, default_model):
        """5 candidates -> Predictions with means shape (5,), all equal to mock ptm value."""
        result = default_model.predict(protein_candidates)
        assert result.means.shape == (5,)
        assert result.means.dtype in (np.float32, np.float64)
        np.testing.assert_array_almost_equal(result.means, np.full(5, MOCK_PTM))

    def test_returns_predictions_instance(self, mock_components, default_model):
        """predict() returns a Predictions object."""
        cand = Candidate(data="ACDE", modality="sequence")
        result = default_model.predict([cand])
        assert isinstance(result, Predictions)

    def test_means_is_numpy_array(self, mock_components, default_model):
        """Means is np.ndarray."""
        cand = Candidate(data="ACDE", modality="sequence")
        result = default_model.predict([cand])
        assert isinstance(result.means, np.ndarray)

    def test_means_are_finite(self, mock_components, protein_candidates, default_model):
        """All means are finite floats."""
        result = default_model.predict(protein_candidates)
        assert np.all(np.isfinite(result.means))

    @pytest.mark.parametrize("metric", ["ptm", "mean_plddt", "combined"])
    def test_metric_values_in_range(self, mock_components, protein_candidates, metric):
        """All scoring metrics produce values in [0, 1]."""
        config = ESMFoldModelConfig(scoring_metric=metric)
        model = ESMFoldModel(config)
        result = model.predict(protein_candidates)
        assert np.all(result.means >= 0.0)
        assert np.all(result.means <= 1.0)

    def test_combined_weight_0_equals_mean_plddt(self, mock_components, protein_candidates):
        """combined_ptm_weight=0.0 -> means equal to mean_plddt."""
        config_combined = ESMFoldModelConfig(scoring_metric="combined", combined_ptm_weight=0.0)
        model_combined = ESMFoldModel(config_combined)
        result_combined = model_combined.predict(protein_candidates)

        config_plddt = ESMFoldModelConfig(scoring_metric="mean_plddt")
        model_plddt = ESMFoldModel(config_plddt)
        result_plddt = model_plddt.predict(protein_candidates)

        np.testing.assert_array_almost_equal(result_combined.means, result_plddt.means)

    def test_combined_weight_1_equals_ptm(self, mock_components, protein_candidates):
        """combined_ptm_weight=1.0 -> means equal to ptm."""
        config_combined = ESMFoldModelConfig(scoring_metric="combined", combined_ptm_weight=1.0)
        model_combined = ESMFoldModel(config_combined)
        result_combined = model_combined.predict(protein_candidates)

        config_ptm = ESMFoldModelConfig(scoring_metric="ptm")
        model_ptm = ESMFoldModel(config_ptm)
        result_ptm = model_ptm.predict(protein_candidates)

        np.testing.assert_array_almost_equal(result_combined.means, result_ptm.means)

    def test_combined_weight_intermediate_interpolates(self, mock_components):
        """combined_ptm_weight=0.3 -> means = 0.3*ptm + 0.7*plddt."""
        w = 0.3
        config = ESMFoldModelConfig(scoring_metric="combined", combined_ptm_weight=w)
        model = ESMFoldModel(config)
        cand = Candidate(data="ACDE", modality="sequence")
        result = model.predict([cand])
        expected = w * MOCK_PTM + (1.0 - w) * MOCK_PLDDT
        np.testing.assert_almost_equal(result.means[0], expected, decimal=6)

    def test_ptm_output_matches_mock_value(self, mock_components, default_model):
        """Ptm scores match the mocked value (0.7)."""
        cand = Candidate(data="ACDE", modality="sequence")
        result = default_model.predict([cand])
        np.testing.assert_array_almost_equal(result.means, [MOCK_PTM])

    def test_mean_plddt_output_matches_mock_value(self, mock_components, protein_candidates):
        """mean_plddt scores match the mocked value (0.6)."""
        config = ESMFoldModelConfig(scoring_metric="mean_plddt")
        model = ESMFoldModel(config)
        result = model.predict(protein_candidates)
        expected = MOCK_PLDDT
        np.testing.assert_array_almost_equal(result.means, [expected] * len(protein_candidates))


class TestESMFoldModelEdgeCases:
    """Tests for ESMFoldModel.predict() error handling on invalid inputs."""

    def test_predict_empty_list_raises(self, mock_components, default_model):
        """predict([]) raises ValueError."""
        with pytest.raises(ValueError, match="No candidates provided"):
            default_model.predict([])

    def test_predict_empty_sequence_raises(self, mock_components, default_model):
        """Candidate with data='' raises ValueError naming the index."""
        cand = Candidate(data="", modality="sequence")
        with pytest.raises(ValueError, match="index 0"):
            default_model.predict([cand])

    def test_predict_valid_aa_char_X_accepted(self, mock_components, default_model):
        """Sequence containing 'X' (IUPAC unknown residue) is accepted by ESMFold."""
        cand = Candidate(data="ACDEFX", modality="sequence")
        result = default_model.predict([cand])  # should not raise
        assert result.means.shape == (1,)

    def test_predict_valid_iupac_ambiguity_codes_accepted(self, mock_components, default_model):
        """Sequences containing IUPAC ambiguity codes B, Z, U, O are accepted."""
        for char in "BZUO":
            cand = Candidate(data=f"ACDE{char}", modality="sequence")
            result = default_model.predict([cand])  # should not raise
            assert result.means.shape == (1,)

    def test_predict_invalid_aa_digit_raises(self, mock_components, default_model):
        """Sequence containing a digit raises ValueError."""
        cand = Candidate(data="ACD3E", modality="sequence")
        with pytest.raises(ValueError, match="3"):
            default_model.predict([cand])

    def test_predict_invalid_aa_whitespace_raises(self, mock_components, default_model):
        """Sequence containing whitespace raises ValueError."""
        cand = Candidate(data="ACD E", modality="sequence")
        with pytest.raises(ValueError, match="invalid amino acid character"):
            default_model.predict([cand])

    def test_predict_wrong_modality_raises(self, mock_components, default_model):
        """Candidate with modality != SEQUENCE raises ValueError."""
        cand = Candidate(data="ACDE", modality="graph")
        with pytest.raises(ValueError, match="Modality.SEQUENCE"):
            default_model.predict([cand])

    def test_predict_error_message_names_index(self, mock_components, default_model):
        """Error for invalid AA names the sequence index."""
        candidates = [
            Candidate(data="ACDE", modality="sequence"),
            Candidate(data="GHIK", modality="sequence"),
            Candidate(data="MNP2R", modality="sequence"),
        ]
        with pytest.raises(ValueError, match="index 2"):
            default_model.predict(candidates)


class TestESMFoldBatching:
    """Tests for ESMFoldModel batching behavior."""

    def test_7_seqs_batch3_calls_3_times(self, mock_components):
        """7 sequences with batch_size=3 -> exactly 3 forward-pass calls (ceil(7/3)=3)."""
        mock_mdl, _, _, _ = mock_components
        config = ESMFoldModelConfig(batch_size=3, scoring_metric="mean_plddt")
        model = ESMFoldModel(config)
        seqs = [Candidate(data="ACDE", modality="sequence")] * 7
        model.predict(seqs)
        assert mock_mdl.call_count == 3

    def test_7_seqs_batch3_returns_7_results(self, mock_components):
        """7 sequences with batch_size=3 -> output has 7 elements."""
        config = ESMFoldModelConfig(batch_size=3, scoring_metric="mean_plddt")
        model = ESMFoldModel(config)
        seqs = [Candidate(data="ACDE", modality="sequence")] * 7
        result = model.predict(seqs)
        assert result.means.shape == (7,)

    def test_partial_last_batch_handled(self, mock_components):
        """The final batch of 1 (in 7 seqs with batch_size=3) is handled correctly."""
        config = ESMFoldModelConfig(batch_size=3, scoring_metric="mean_plddt")
        model = ESMFoldModel(config)
        seqs = [Candidate(data="ACDE", modality="sequence")] * 7
        result = model.predict(seqs)
        assert np.all(np.isfinite(result.means))
        np.testing.assert_array_almost_equal(result.means, np.full(7, MOCK_PLDDT))

    def test_batch_size_1_calls_n_times(self, mock_components):
        """batch_size=1 -> N forward calls for N sequences."""
        mock_mdl, _, _, _ = mock_components
        config = ESMFoldModelConfig(batch_size=1)
        model = ESMFoldModel(config)
        seqs = [Candidate(data="ACDE", modality="sequence")] * 5
        model.predict(seqs)
        assert mock_mdl.call_count == 5

    def test_batch_size_n_calls_1_time(self, mock_components):
        """batch_size >= N -> 1 forward call for N sequences."""
        mock_mdl, _, _, _ = mock_components
        config = ESMFoldModelConfig(batch_size=10, scoring_metric="mean_plddt")
        model = ESMFoldModel(config)
        seqs = [Candidate(data="ACDE", modality="sequence")] * 5
        model.predict(seqs)
        assert mock_mdl.call_count == 1


class TestESMFoldSequenceLengths:
    """Tests for single-residue, long, and mixed-length sequences."""

    def test_single_residue_sequence(self, mock_components):
        """Single 'A' residue -> valid Predictions with shape (1,)."""
        config = ESMFoldModelConfig()
        model = ESMFoldModel(config)
        cand = Candidate(data="A", modality="sequence")
        result = model.predict([cand])
        assert result.means.shape == (1,)
        assert np.all(np.isfinite(result.means))

    def test_long_sequence_with_chunk_size(self, mock_components):
        """1024-AA sequence with chunk_size=64 -> set_chunk_size called, prediction completes."""
        mock_mdl, _, _, _ = mock_components
        config = ESMFoldModelConfig(chunk_size=64)
        model = ESMFoldModel(config)
        mock_mdl.esm.encoder.set_chunk_size.assert_called_once_with(64)
        cand = Candidate(data="A" * 1024, modality="sequence")
        result = model.predict([cand])
        assert result.means.shape == (1,)
        assert np.all(np.isfinite(result.means))

    def test_mixed_length_batch(self, mock_components):
        """Batch with 5-AA and 50-AA sequences -> both results valid and independent."""
        config = ESMFoldModelConfig()
        model = ESMFoldModel(config)
        cands = [
            Candidate(data="ACDEF", modality="sequence"),
            Candidate(data="A" * 50, modality="sequence"),
        ]
        result = model.predict(cands)
        assert result.means.shape == (2,)
        assert np.all(np.isfinite(result.means))

    def test_plddt_aggregation_uses_correct_axes(self):
        """Plddt mean over (L, 37) axes is verified with a non-uniform tensor."""

        # Use a custom fixture that gives different plddt values across residues
        # so the mean is only correct if (B, L, 37) is reduced correctly.
        def _tok_call(seqs, return_tensors="pt", padding=True, add_special_tokens=False):
            n, seq_len = len(seqs), max(len(s) for s in seqs)
            return {
                "input_ids": torch.ones(n, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(n, seq_len, dtype=torch.long),
            }

        with (
            patch("alf_tools.models.esmfold.EsmForProteinFolding") as mock_cls,
            patch("alf_tools.models.esmfold.AutoTokenizer") as mock_tok_cls,
        ):
            mock_tok = MagicMock()
            mock_tok.side_effect = _tok_call
            mock_tok_cls.from_pretrained.return_value = mock_tok

            def _model_call(**tokens):
                n = tokens["input_ids"].shape[0]
                seq_len = tokens["input_ids"].shape[1]
                out = MagicMock()
                out.ptm = torch.tensor(0.5, dtype=torch.float32)
                # plddt varies across residue dim: position 0 = 0.8, rest = 0.4
                plddt = torch.full((n, seq_len, 37), 0.4, dtype=torch.float32)
                plddt[:, 0, :] = 0.8
                out.plddt = plddt
                out.atom37_atom_exists = torch.ones(n, seq_len, 37, dtype=torch.bool)
                return out

            mock_mdl = MagicMock()
            mock_mdl.to.return_value = mock_mdl
            mock_mdl.side_effect = _model_call
            mock_cls.from_pretrained.return_value = mock_mdl

            config = ESMFoldModelConfig(scoring_metric="mean_plddt")
            model = ESMFoldModel(config)
            cand = Candidate(data="ACG", modality="sequence")  # 3 residues
            result = model.predict([cand])

        # expected: (0.8 + 0.4 + 0.4) / 3 ≈ 0.5333
        expected = (0.8 + 0.4 * 2) / 3.0
        np.testing.assert_almost_equal(result.means[0], expected, decimal=5)


class TestESMFoldPLDDTMasking:
    """Tests that pLDDT aggregation correctly excludes padding positions."""

    def test_plddt_masked_mean_excludes_padding(self):
        """Padding zeros in attention_mask are excluded from pLDDT mean (batch_size=2)."""

        def _tok_call(seqs, return_tensors="pt", padding=True, add_special_tokens=False):
            # Two sequences: lengths 2 and 3, padded to length 3
            return {
                "input_ids": torch.ones(2, 3, dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long),
            }

        with (
            patch("alf_tools.models.esmfold.EsmForProteinFolding") as mock_cls,
            patch("alf_tools.models.esmfold.AutoTokenizer") as mock_tok_cls,
        ):
            mock_tok = MagicMock()
            mock_tok.side_effect = _tok_call
            mock_tok_cls.from_pretrained.return_value = mock_tok

            def _model_call(**tokens):
                out = MagicMock()
                out.ptm = torch.tensor(0.5, dtype=torch.float32)
                out.plddt = torch.full((2, 3, 37), 0.4, dtype=torch.float32)
                out.atom37_atom_exists = torch.ones(2, 3, 37, dtype=torch.bool)
                return out

            mock_mdl = MagicMock()
            mock_mdl.to.return_value = mock_mdl
            mock_mdl.side_effect = _model_call
            mock_mdl.esm = MagicMock()
            mock_cls.from_pretrained.return_value = mock_mdl

            # batch_size=2 so both sequences are processed in one forward pass
            model = ESMFoldModel(ESMFoldModelConfig(scoring_metric="mean_plddt", batch_size=2))
            cands = [
                Candidate(data="AC", modality="sequence"),
                Candidate(data="ACG", modality="sequence"),
            ]
            result = model.predict(cands)

        # seq 0: 2 real residues, plddt 0.4 -> masked mean = 0.4
        # seq 1: 3 real residues, plddt 0.4 -> masked mean = 0.4
        np.testing.assert_almost_equal(result.means[0], 0.4, decimal=5)
        np.testing.assert_almost_equal(result.means[1], 0.4, decimal=5)

    def test_plddt_masked_mean_differs_from_unmasked_when_padding_present(self):
        """Confirms masking changes the result when padding values would inflate the mean."""

        def _tok_call(seqs, return_tensors="pt", padding=True, add_special_tokens=False):
            # seq 0: length 2, padded to 3; padding position gets plddt=1.0 from model
            return {
                "input_ids": torch.ones(1, 3, dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
            }

        with (
            patch("alf_tools.models.esmfold.EsmForProteinFolding") as mock_cls,
            patch("alf_tools.models.esmfold.AutoTokenizer") as mock_tok_cls,
        ):
            mock_tok = MagicMock()
            mock_tok.side_effect = _tok_call
            mock_tok_cls.from_pretrained.return_value = mock_tok

            def _model_call(**tokens):
                out = MagicMock()
                out.ptm = torch.tensor(0.5, dtype=torch.float32)
                plddt = torch.zeros(1, 3, 37, dtype=torch.float32)
                plddt[:, :2, :] = 0.6  # real residues
                plddt[:, 2, :] = 1.0  # padding position — should be excluded
                out.plddt = plddt
                out.atom37_atom_exists = torch.ones(1, 3, 37, dtype=torch.bool)
                return out

            mock_mdl = MagicMock()
            mock_mdl.to.return_value = mock_mdl
            mock_mdl.side_effect = _model_call
            mock_mdl.esm = MagicMock()
            mock_cls.from_pretrained.return_value = mock_mdl

            model = ESMFoldModel(ESMFoldModelConfig(scoring_metric="mean_plddt"))
            cand = Candidate(data="AC", modality="sequence")
            result = model.predict([cand])

        # Masked mean: (0.6 * 2 residues) / 2 = 0.6 (not 0.733 which would be unmasked mean)
        np.testing.assert_almost_equal(result.means[0], 0.6, decimal=5)


class TestESMFoldDuplicates:
    """Duplicate sequences are each processed independently."""

    def test_two_identical_sequences(self, mock_components):
        """Two identical sequences -> each gets its own output at separate indices."""
        config = ESMFoldModelConfig()
        model = ESMFoldModel(config)
        cands = [
            Candidate(data="ACDE", modality="sequence"),
            Candidate(data="ACDE", modality="sequence"),
        ]
        result = model.predict(cands)
        assert result.means.shape == (2,)
        # Both values equal (same mock output), but stored at separate indices
        assert result.means[0] == result.means[1]

    def test_5_sequences_with_3_duplicates(self, mock_components):
        """5-sequence batch with 3 duplicates -> all 5 results returned."""
        config = ESMFoldModelConfig()
        model = ESMFoldModel(config)
        cands = [
            Candidate(data="ACDE", modality="sequence"),
            Candidate(data="FGHI", modality="sequence"),
            Candidate(data="ACDE", modality="sequence"),
            Candidate(data="ACDE", modality="sequence"),
            Candidate(data="MNPQ", modality="sequence"),
        ]
        result = model.predict(cands)
        assert result.means.shape == (5,)
        assert np.all(np.isfinite(result.means))


class TestESMFoldCleanup:
    """Tests for ESMFoldModel.cleanup()."""

    def test_cleanup_moves_model_to_cpu(self, mock_components):
        """After cleanup(), model.to('cpu') was called."""
        mock_mdl, _, _, _ = mock_components
        model = ESMFoldModel(ESMFoldModelConfig())
        mock_mdl.to.reset_mock()
        model.cleanup()
        mock_mdl.to.assert_called_once_with("cpu")

    def test_cleanup_calls_cuda_empty_cache_when_cuda_available(self, mock_components):
        """torch.cuda.empty_cache() called once when CUDA is available."""
        model = ESMFoldModel(ESMFoldModelConfig())
        with (
            patch("alf_tools.models.esmfold.torch.cuda.is_available", return_value=True),
            patch("alf_tools.models.esmfold.torch.cuda.empty_cache") as mock_cache,
        ):
            model.cleanup()
        assert mock_cache.call_count == 1

    def test_cleanup_does_not_call_empty_cache_when_cuda_unavailable(self, mock_components):
        """torch.cuda.empty_cache() is not called when CUDA is unavailable."""
        model = ESMFoldModel(ESMFoldModelConfig())
        with (
            patch("alf_tools.models.esmfold.torch.cuda.is_available", return_value=False),
            patch("alf_tools.models.esmfold.torch.cuda.empty_cache") as mock_cache,
        ):
            model.cleanup()
        mock_cache.assert_not_called()

    def test_cleanup_updates_device_to_cpu(self, mock_components):
        """After cleanup(), self.device is torch.device('cpu')."""
        model = ESMFoldModel(ESMFoldModelConfig())
        with patch("alf_tools.models.esmfold.torch.cuda.is_available", return_value=False):
            model.cleanup()
        assert model.device == torch.device("cpu")

    def test_cleanup_converts_full_model_to_float32(self, mock_components):
        """After cleanup(), model.float() is called to restore fp32 for all submodules."""
        mock_mdl, _, _, _ = mock_components
        model = ESMFoldModel(ESMFoldModelConfig())
        mock_mdl.float.reset_mock()  # clear any prior calls
        with patch("alf_tools.models.esmfold.torch.cuda.is_available", return_value=False):
            model.cleanup()
        mock_mdl.float.assert_called_once()

    def test_predict_after_cleanup_raises_runtime_error(self, mock_components):
        """predict() called after cleanup() raises RuntimeError."""
        model = ESMFoldModel(ESMFoldModelConfig())
        with patch("alf_tools.models.esmfold.torch.cuda.is_available", return_value=False):
            model.cleanup()
        with pytest.raises(RuntimeError, match="cleanup"):
            model.predict([Candidate(data="ACDE", modality="sequence")])


class TestESMFoldAllZeroAttentionMask:
    """Tests that an all-zero attention mask triggers RuntimeError."""

    def test_all_zero_attention_mask_raises_runtime_error(self):
        """All-zeros attention_mask raises RuntimeError naming the batch offset."""

        def _tok_call(seqs, return_tensors="pt", padding=True, add_special_tokens=False):
            n = len(seqs)
            seq_len = max(len(s) for s in seqs) if seqs else 1
            return {
                "input_ids": torch.ones(n, seq_len, dtype=torch.long),
                "attention_mask": torch.zeros(n, seq_len, dtype=torch.long),  # all padding
            }

        with (
            patch("alf_tools.models.esmfold.EsmForProteinFolding") as mock_cls,
            patch("alf_tools.models.esmfold.AutoTokenizer") as mock_tok_cls,
        ):
            mock_tok = MagicMock()
            mock_tok.side_effect = _tok_call
            mock_tok_cls.from_pretrained.return_value = mock_tok

            def _model_call(**tokens):
                n = tokens["input_ids"].shape[0]
                seq_len = tokens["input_ids"].shape[1]
                out = MagicMock()
                out.ptm = torch.tensor(0.5, dtype=torch.float32)
                out.plddt = torch.full((n, seq_len, 37), 0.6, dtype=torch.float32)
                out.atom37_atom_exists = torch.ones(n, seq_len, 37, dtype=torch.bool)
                return out

            mock_mdl = MagicMock()
            mock_mdl.to.return_value = mock_mdl
            mock_mdl.side_effect = _model_call
            mock_mdl.esm = MagicMock()
            mock_cls.from_pretrained.return_value = mock_mdl

            model = ESMFoldModel(ESMFoldModelConfig(scoring_metric="mean_plddt"))
            with pytest.raises(RuntimeError, match="all-padding"):
                model.predict([Candidate(data="ACDE", modality="sequence")])


class _StubModel(BaseModel):
    """Minimal BaseModel stub for State construction in Oracle tests."""

    def predict(self, candidates):
        """Return zero predictions."""
        return Predictions(means=np.zeros(len(candidates)))

    def featurise(self, inputs):
        """No-op featurisation."""

    def train(self, train_data, val_data):
        """No-op training."""

    def sample(self, condition=None):
        """Not implemented."""
        raise NotImplementedError

    def get_training_summary_metrics(self):
        """Return empty training metrics."""
        return {}


class _StubDataset(BaseDataset):
    """Minimal BaseDataset stub for State construction in Oracle tests."""

    def load_dataset(self) -> LabelledCandidates:
        """Return a trivial empty dataset."""
        return LabelledCandidates(candidates=[], labels=np.array([]))


@pytest.fixture
def esmfold_state():
    """Minimal State instance for Oracle integration tests.

    Returns:
        A State with stub dataset and surrogate instances.
    """
    config = BaseDatasetConfig(
        name="stub",
        modality="sequence",
        seed=0,
        train_ratio=0.6,
        validation_frac=0.2,
        test_ratio=0.2,
        problem_type=ProblemType.REGRESSION,
    )
    dataset = _StubDataset(config)
    surrogate = Surrogate(model=_StubModel())
    return State(dataset=dataset, surrogate=surrogate)


class TestESMFoldOracle:
    """Integration of ESMFoldModel with the Oracle wrapper."""

    def test_oracle_evaluate_returns_labelled_candidates(
        self, mock_components, protein_candidates, esmfold_state
    ):
        """Oracle.evaluate() returns LabelledCandidates with correct length."""
        model = ESMFoldModel(ESMFoldModelConfig())
        oracle = Oracle(scorer=model)
        result, _ = oracle.evaluate(protein_candidates, esmfold_state)
        assert isinstance(result, LabelledCandidates)
        assert len(result.candidates) == len(protein_candidates)

    def test_oracle_evaluate_records_oracle_time(
        self, mock_components, protein_candidates, esmfold_state
    ):
        """oracle_time is recorded in state.round_metrics after evaluate()."""
        model = ESMFoldModel(ESMFoldModelConfig())
        oracle = Oracle(scorer=model)
        _, new_state = oracle.evaluate(protein_candidates, esmfold_state)
        assert "oracle_time" in new_state.round_metrics.metrics


@pytest.fixture(scope="session")
def _golden_esmfold():
    """Load facebook/esmfold_v1 once per pytest session for golden-sequence tests.

    Loading the ~2.6 GB checkpoint is the dominant cost; sharing it across all three
    tests in TestESMFoldGoldenSequences avoids repeating that cost.  Each test sets
    config.scoring_metric before calling predict(), which is safe because integration
    tests run serially.

    Yields:
        ESMFoldModel: loaded model instance shared across the session.
    """
    torch.set_num_threads(os.cpu_count() or 1)
    model = ESMFoldModel(ESMFoldModelConfig(device="cpu", scoring_metric="ptm"))
    yield model
    model.cleanup()


@pytest.mark.integration
class TestESMFoldGoldenSequences:
    """Integration tests with real ESMFold model and known reference sequences.

    pTM and pLDDT values are deterministic for a given model and sequence — they
    do not systematically differ between CPU float32 and GPU float16.  All thresholds
    here are calibrated against CPU float32 inference with add_special_tokens=False
    (the correct ESMFold tokeniser usage per the HuggingFace docs).

    Note on pTM for short peptides: pTM measures global fold confidence for the whole
    chain.  For peptides shorter than ~30 residues ESMFold reliably returns pTM < 0.1
    regardless of secondary-structure propensity, so pTM is not a meaningful benchmark
    at that length.  Use mean_pLDDT (per-residue confidence) to distinguish well-
    structured regions for short sequences.
    """

    @pytest.fixture(autouse=True)
    def _restore_scoring_metric(self, _golden_esmfold):
        """Reset scoring_metric to its default after each test.

        The session-scoped fixture loads the 2.6 GB checkpoint once and shares it
        across all tests. Each test mutates config.scoring_metric before calling
        predict(). This autouse fixture restores the original value after each test
        so test-execution order cannot cause a wrong metric to bleed into the next test.

        Yields:
            None.
        """
        original = _golden_esmfold.config.scoring_metric
        yield
        _golden_esmfold.config.scoring_metric = original

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires GPU for reasonable runtime; skip on CPU-only environments.",
    )
    def test_high_plddt_helical_peptide(self, _golden_esmfold):
        """Poly-Ala helical peptide → mean pLDDT ≥ 0.7: short but locally well-structured."""
        # AAAAAKAAAAKAAAAK — helical propensity peptide; ESMFold gives mean_pLDDT ≈ 0.80 on CPU.
        # pTM is not tested: for sequences under ~30 residues pTM < 0.1 regardless of
        # secondary-structure propensity (see class docstring).
        _golden_esmfold.config.scoring_metric = "mean_plddt"
        cand = Candidate(data="AAAAAKAAAAKAAAAK", modality="sequence")
        result = _golden_esmfold.predict([cand])
        assert result.means[0] >= 0.7, f"Expected mean_plddt >= 0.7, got {result.means[0]}"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires GPU for reasonable runtime; skip on CPU-only environments.",
    )
    def test_low_ptm_disordered_peptide(self, _golden_esmfold):
        """Gly-Ser repeat → pTM in [0.0, 0.1]: intrinsically disordered, low global confidence."""
        # GSGSGSGSGS — canonical disordered linker; ESMFold gives pTM ≈ 0.028 on CPU
        _golden_esmfold.config.scoring_metric = "ptm"
        cand = Candidate(data="GSGSGSGSGS", modality="sequence")
        result = _golden_esmfold.predict([cand])
        assert 0.0 <= result.means[0] <= 0.1, f"Expected ptm in [0.0, 0.1], got {result.means[0]}"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires GPU for reasonable runtime; skip on CPU-only environments.",
    )
    def test_gfp_fragment_high_plddt(self, _golden_esmfold):
        """GFP first 50 AA → mean pLDDT ≥ 0.65: known structured region."""
        # GFP (PDB 1EMA), first 50 residues; ESMFold gives mean_pLDDT ≈ 0.71 on CPU
        _golden_esmfold.config.scoring_metric = "mean_plddt"
        GFP_50 = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTT"
        cand = Candidate(data=GFP_50, modality="sequence")
        result = _golden_esmfold.predict([cand])
        assert result.means[0] >= 0.65, f"Expected mean_plddt >= 0.65, got {result.means[0]}"
