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
from alf_core.dataclasses import LabelledCandidates, State
from alf_core.dataclasses.candidate import Modality
from alf_core.dataset.base_dataset import BaseDataset, BaseDatasetConfig
from alf_core.model.base_model import BaseModel
from alf_core.oracle.oracle import Oracle
from alf_core.surrogate.surrogate import Surrogate

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
        assert result.means.dtype in (np.float32, np.float64)

    def test_batch_5_returns_shape_5(self, mock_components, protein_candidates, default_model):
        """5 candidates -> Predictions with means shape (5,)."""
        result = default_model.predict(protein_candidates)
        assert result.means.shape == (5,)
        assert result.means.dtype in (np.float32, np.float64)

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

    @pytest.mark.parametrize("metric", ["ptm", "mean_plddt", "combined"])
    def test_metric_values_in_range(self, mock_components, protein_candidates, metric):
        """All scoring metrics produce values in [0, 1]."""
        config = ESMFoldConfig(scoring_metric=metric)
        model = ESMFoldModel(config)
        result = model.predict(protein_candidates)
        assert np.all(result.means >= 0.0)
        assert np.all(result.means <= 1.0)

    def test_combined_weight_0_equals_mean_plddt(self, mock_components, protein_candidates):
        """combined_ptm_weight=0.0 -> means equal to mean_plddt/100."""
        config_combined = ESMFoldConfig(scoring_metric="combined", combined_ptm_weight=0.0)
        model_combined = ESMFoldModel(config_combined)
        result_combined = model_combined.predict(protein_candidates)

        config_plddt = ESMFoldConfig(scoring_metric="mean_plddt")
        model_plddt = ESMFoldModel(config_plddt)
        result_plddt = model_plddt.predict(protein_candidates)

        np.testing.assert_array_almost_equal(result_combined.means, result_plddt.means)

    def test_combined_weight_1_equals_ptm(self, mock_components, protein_candidates):
        """combined_ptm_weight=1.0 -> means equal to ptm."""
        config_combined = ESMFoldConfig(scoring_metric="combined", combined_ptm_weight=1.0)
        model_combined = ESMFoldModel(config_combined)
        result_combined = model_combined.predict(protein_candidates)

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
        config = ESMFoldConfig(scoring_metric="mean_plddt")
        model = ESMFoldModel(config)
        result = model.predict(protein_candidates)
        expected = MOCK_PLDDT / 100.0
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

    def test_predict_invalid_aa_char_X_raises(self, mock_components, default_model):
        """Sequence containing 'X' (not a standard AA) raises ValueError."""
        cand = Candidate(data="ACDEFX", modality="sequence")
        with pytest.raises(ValueError, match="X"):
            default_model.predict([cand])

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

    def test_predict_combined_weight_0_equals_mean_plddt(self, mock_components, protein_candidates):
        """combined_ptm_weight=0.0 -> means equal to mean_plddt/100 exactly."""
        config_combined = ESMFoldConfig(scoring_metric="combined", combined_ptm_weight=0.0)
        model_combined = ESMFoldModel(config_combined)
        result_combined = model_combined.predict(protein_candidates)

        config_plddt = ESMFoldConfig(scoring_metric="mean_plddt")
        model_plddt = ESMFoldModel(config_plddt)
        result_plddt = model_plddt.predict(protein_candidates)

        np.testing.assert_array_almost_equal(result_combined.means, result_plddt.means)

    def test_predict_combined_weight_1_equals_ptm(self, mock_components, protein_candidates):
        """combined_ptm_weight=1.0 -> means equal to ptm exactly."""
        config_combined = ESMFoldConfig(scoring_metric="combined", combined_ptm_weight=1.0)
        model_combined = ESMFoldModel(config_combined)
        result_combined = model_combined.predict(protein_candidates)

        config_ptm = ESMFoldConfig(scoring_metric="ptm")
        model_ptm = ESMFoldModel(config_ptm)
        result_ptm = model_ptm.predict(protein_candidates)

        np.testing.assert_array_almost_equal(result_combined.means, result_ptm.means)


class TestESMFoldBatching:
    """Tests for ESMFoldModel batching behavior."""

    def test_7_seqs_batch3_calls_3_times(self, mock_components):
        """7 sequences with batch_size=3 -> exactly 3 forward-pass calls (ceil(7/3)=3)."""
        mock_mdl, _, _, _ = mock_components
        config = ESMFoldConfig(batch_size=3)
        model = ESMFoldModel(config)
        seqs = [Candidate(data="ACDE", modality="sequence")] * 7
        model.predict(seqs)
        assert mock_mdl.call_count == 3

    def test_7_seqs_batch3_returns_7_results(self, mock_components):
        """7 sequences with batch_size=3 -> output has 7 elements."""
        _, _, _, _ = mock_components
        config = ESMFoldConfig(batch_size=3)
        model = ESMFoldModel(config)
        seqs = [Candidate(data="ACDE", modality="sequence")] * 7
        result = model.predict(seqs)
        assert result.means.shape == (7,)

    def test_partial_last_batch_handled(self, mock_components):
        """The final batch of 1 (in 7 seqs with batch_size=3) is handled correctly."""
        _, _, _, _ = mock_components
        config = ESMFoldConfig(batch_size=3)
        model = ESMFoldModel(config)
        seqs = [Candidate(data="ACDE", modality="sequence")] * 7
        result = model.predict(seqs)
        assert np.all(np.isfinite(result.means))

    def test_batch_size_1_calls_n_times(self, mock_components):
        """batch_size=1 -> N forward calls for N sequences."""
        mock_mdl, _, _, _ = mock_components
        config = ESMFoldConfig(batch_size=1)
        model = ESMFoldModel(config)
        seqs = [Candidate(data="ACDE", modality="sequence")] * 5
        model.predict(seqs)
        assert mock_mdl.call_count == 5

    def test_batch_size_n_calls_1_time(self, mock_components):
        """batch_size >= N -> 1 forward call for N sequences."""
        mock_mdl, _, _, _ = mock_components
        config = ESMFoldConfig(batch_size=10)
        model = ESMFoldModel(config)
        seqs = [Candidate(data="ACDE", modality="sequence")] * 5
        model.predict(seqs)
        assert mock_mdl.call_count == 1


class TestESMFoldSequenceLengths:
    """Tests for single-residue, long, and mixed-length sequences."""

    def test_single_residue_sequence(self, mock_components):
        """Single 'A' residue -> valid Predictions with shape (1,)."""
        config = ESMFoldConfig()
        model = ESMFoldModel(config)
        cand = Candidate(data="A", modality="sequence")
        result = model.predict([cand])
        assert result.means.shape == (1,)
        assert np.all(np.isfinite(result.means))

    def test_long_sequence_with_chunk_size(self, mock_components):
        """1024-AA sequence with chunk_size=64 -> set_chunk_size called, prediction completes."""
        mock_mdl, _, _, _ = mock_components
        config = ESMFoldConfig(chunk_size=64)
        model = ESMFoldModel(config)
        mock_mdl.esm.encoder.set_chunk_size.assert_called_once_with(64)
        cand = Candidate(data="A" * 1024, modality="sequence")
        result = model.predict([cand])
        assert result.means.shape == (1,)
        assert np.all(np.isfinite(result.means))

    def test_mixed_length_batch(self, mock_components):
        """Batch with 5-AA and 50-AA sequences -> both results valid and independent."""
        config = ESMFoldConfig()
        model = ESMFoldModel(config)
        cands = [
            Candidate(data="ACDEF", modality="sequence"),
            Candidate(data="A" * 50, modality="sequence"),
        ]
        result = model.predict(cands)
        assert result.means.shape == (2,)
        assert np.all(np.isfinite(result.means))


class TestESMFoldDuplicates:
    """Duplicate sequences are each processed independently."""

    def test_two_identical_sequences(self, mock_components):
        """Two identical sequences -> each gets its own output at separate indices."""
        config = ESMFoldConfig()
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
        config = ESMFoldConfig()
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
        model = ESMFoldModel(ESMFoldConfig())
        mock_mdl.to.reset_mock()
        model.cleanup()
        mock_mdl.to.assert_called_once_with("cpu")

    def test_cleanup_calls_cuda_empty_cache_when_cuda_available(self, mock_components):
        """torch.cuda.empty_cache() called once when CUDA is available."""
        model = ESMFoldModel(ESMFoldConfig())
        with (
            patch("alf_tools.models.esmfold.torch.cuda.is_available", return_value=True),
            patch("alf_tools.models.esmfold.torch.cuda.empty_cache") as mock_cache,
        ):
            model.cleanup()
        assert mock_cache.call_count == 1


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
    )
    dataset = _StubDataset(config)
    surrogate = Surrogate(model=_StubModel())
    return State(dataset=dataset, surrogate=surrogate)


class TestESMFoldOracle:
    """Integration of ESMFoldModel with the Oracle wrapper."""

    def test_oracle_evaluate_returns_labelled_candidates(self, mock_components, protein_candidates, esmfold_state):
        """Oracle.evaluate() returns LabelledCandidates with correct length."""
        model = ESMFoldModel(ESMFoldConfig())
        oracle = Oracle(scorer=model)
        result, _ = oracle.evaluate(protein_candidates, esmfold_state)
        assert isinstance(result, LabelledCandidates)
        assert len(result.candidates) == len(protein_candidates)

    def test_oracle_evaluate_records_oracle_time(self, mock_components, protein_candidates, esmfold_state):
        """oracle_time is recorded in state.round_metrics after evaluate()."""
        model = ESMFoldModel(ESMFoldConfig())
        oracle = Oracle(scorer=model)
        _, new_state = oracle.evaluate(protein_candidates, esmfold_state)
        assert "oracle_time" in new_state.round_metrics.metrics


@pytest.mark.integration
class TestESMFoldGoldenSequences:
    """Integration tests with real ESMFold model and known reference sequences."""

    def test_high_ptm_helical_peptide(self):
        """Short alpha-helical peptide → pTM in [0.6, 1.0]."""
        # AAAAAKAAAAKAAAAK — poly-Ala helical-like peptide; known to fold
        config = ESMFoldConfig(device="cpu", scoring_metric="ptm")
        model = ESMFoldModel(config)
        cand = Candidate(data="AAAAAKAAAAKAAAAK", modality="sequence")
        result = model.predict([cand])
        assert 0.6 <= result.means[0] <= 1.0, f"Expected ptm in [0.6, 1.0], got {result.means[0]}"

    def test_low_ptm_disordered_peptide(self):
        """Short disordered sequence → pTM in [0.0, 0.4]."""
        # GSGSGSGSGS — Gly-Ser repeats, intrinsically disordered
        config = ESMFoldConfig(device="cpu", scoring_metric="ptm")
        model = ESMFoldModel(config)
        cand = Candidate(data="GSGSGSGSGS", modality="sequence")
        result = model.predict([cand])
        assert 0.0 <= result.means[0] <= 0.4, f"Expected ptm in [0.0, 0.4], got {result.means[0]}"

    def test_gfp_fragment_high_plddt(self):
        """GFP first 50 AA → mean pLDDT/100 in [0.7, 1.0]."""
        # GFP (PDB 1EMA), first 50 residues
        GFP_50 = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTT"
        config = ESMFoldConfig(device="cpu", scoring_metric="mean_plddt")
        model = ESMFoldModel(config)
        cand = Candidate(data=GFP_50, modality="sequence")
        result = model.predict([cand])
        assert 0.7 <= result.means[0] <= 1.0, f"Expected mean_plddt/100 in [0.7, 1.0], got {result.means[0]}"
