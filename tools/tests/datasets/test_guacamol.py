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

import inspect
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from alf_core import Candidate, Modality
from alf_tools.datasets.guacamol import (
    ALL_PROPERTIES,
    DATAPATH,
    FILENAME_ALL,
    FILENAME_TEST,
    FILENAME_TRAIN,
    FILENAME_VALID,
    GuacaMol,
    GuacaMolConfig,
    _cache_path,  # noqa: PLC2701
    _compute_properties,  # noqa: PLC2701
    _download_file,  # noqa: PLC2701
    _load_smiles_file,  # noqa: PLC2701
    download_guacamol,
)

pytestmark = [pytest.mark.guacamol, pytest.mark.rdkit]

FIXTURES = Path(__file__).parent.parent / "fixtures" / "guacamol"
VALID_FIXTURE = FIXTURES / "valid.smiles"
INVALID_FIXTURE = FIXTURES / "invalid.smiles"
EMPTY_FIXTURE = FIXTURES / "empty.smiles"
LARGE_FIXTURE = FIXTURES / "large.smiles"


def _base_config(**overrides) -> GuacaMolConfig:
    defaults = dict(
        name="guacamol",
        modality="sequence",
        seed=42,
        train_ratio=0.6,
        validation_frac=0.1,
        test_ratio=0.2,
        target_property="TPSA",
    )
    defaults.update(overrides)
    return GuacaMolConfig(**defaults)


class TestGuacaMolConfig:
    """Unit tests for GuacaMolConfig validation and auto-derived fields."""

    def test_task_type_auto_set_to_property(self):
        """task_type is 'property' when target is a physicochemical property."""
        config = _base_config(target_property="TPSA")
        assert config.task_type == "property"

    def test_task_type_auto_set_to_benchmark_task(self):
        """task_type is 'benchmark_task' for goal-directed benchmark targets."""
        config = _base_config(target_property="celecoxib_rediscovery")
        assert config.task_type == "benchmark_task"

    def test_default_max_molecules_is_none(self):
        """max_molecules defaults to None (no corpus cap)."""
        assert _base_config().max_molecules is None

    def test_default_split_mode_is_random(self):
        """split_mode defaults to 'random'."""
        assert _base_config().split_mode == "random"

    def test_default_computed_properties_is_none(self):
        """computed_properties defaults to None (all 10 properties computed)."""
        assert _base_config().computed_properties is None

    def test_target_property_absent_from_explicit_computed_properties_raises(self):
        """Raises ValueError when target_property is not in computed_properties."""
        with pytest.raises(ValueError, match="target_property"):
            _base_config(target_property="TPSA", computed_properties=["MolLogP", "MolWt"])

    def test_target_property_present_in_computed_properties_is_valid(self):
        """computed_properties list is accepted when it includes the target_property."""
        config = _base_config(target_property="TPSA", computed_properties=["TPSA", "MolLogP"])
        assert config.computed_properties == ["TPSA", "MolLogP"]

    def test_split_type_synced_with_split_mode_low_vs_high(self):
        """split_type is synced to 'low_vs_high' when split_mode is 'low_vs_high'."""
        config = _base_config(split_mode="low_vs_high")
        assert config.split_type == "low_vs_high"

    def test_split_mode_paper_does_not_alter_split_type(self):
        """split_mode 'paper' is preserved and does not overwrite split_type."""
        config = _base_config(split_mode="paper")
        assert config.split_mode == "paper"

    def test_computed_properties_none_always_valid_regardless_of_target(self):
        """computed_properties=None is valid for any target_property."""
        for prop in ["TPSA", "MolWt", "QED"]:
            config = _base_config(target_property=prop, computed_properties=None)
            assert config.task_type == "property"

    def test_data_dir_defaults_to_datapath(self):
        """data_dir defaults to the package-level DATAPATH constant."""
        config = _base_config()
        assert config.data_dir == DATAPATH

    def test_data_dir_can_be_overridden(self, tmp_path):
        """data_dir accepts a custom Path and stores it."""
        config = _base_config(data_dir=tmp_path)
        assert config.data_dir == tmp_path


class TestComputeProperties:
    """Unit tests for the _compute_properties RDKit descriptor helper."""

    def test_returns_all_keys_when_all_properties_requested(self):
        """Returns a dict with all 10 GuacaMol property keys."""
        result = _compute_properties("c1ccccc1", list(ALL_PROPERTIES))
        assert set(result.keys()) == ALL_PROPERTIES

    def test_returns_only_requested_subset(self):
        """Returns only the keys that were explicitly requested."""
        result = _compute_properties("c1ccccc1", ["TPSA", "MolWt"])
        assert set(result.keys()) == {"TPSA", "MolWt"}

    def test_tpsa_of_benzene_is_zero(self):
        """Benzene TPSA is 0.0 (no polar surface area)."""
        result = _compute_properties("c1ccccc1", ["TPSA"])
        assert result["TPSA"] == pytest.approx(0.0, abs=1e-3)

    def test_molwt_of_ethanol(self):
        """Ethanol molecular weight matches the known value."""
        result = _compute_properties("CCO", ["MolWt"])
        assert result["MolWt"] == pytest.approx(46.069, rel=1e-3)

    def test_all_values_are_float(self):
        """All returned property values are Python floats."""
        result = _compute_properties("CC(=O)O", list(ALL_PROPERTIES))
        for key, val in result.items():
            assert isinstance(val, float), f"{key} value is not a float"

    def test_bertzct_of_benzene_is_positive(self):
        """Benzene BertzCT is positive (ring increases graph complexity)."""
        result = _compute_properties("c1ccccc1", ["BertzCT"])
        assert float(result["BertzCT"]) > 0.0

    def test_bertzct_increases_with_complexity(self):
        """Naphthalene (two rings) has higher BertzCT than benzene (one ring)."""
        benzene = float(_compute_properties("c1ccccc1", ["BertzCT"])["BertzCT"])
        naphthalene = float(_compute_properties("c1ccc2ccccc2c1", ["BertzCT"])["BertzCT"])
        assert naphthalene > benzene

    def test_num_h_acceptors_of_ethanol(self):
        """Ethanol has one H-bond acceptor (the oxygen)."""
        result = _compute_properties("CCO", ["NumHAcceptors"])
        assert result["NumHAcceptors"] == pytest.approx(1.0)

    def test_num_h_donors_of_ethanol(self):
        """Ethanol has one H-bond donor (the O-H group)."""
        result = _compute_properties("CCO", ["NumHDonors"])
        assert result["NumHDonors"] == pytest.approx(1.0)

    def test_num_rotatable_bonds_of_butane(self):
        """Butane (CCCC) has one rotatable bond (the central C-C; terminal methyls excluded)."""
        result = _compute_properties("CCCC", ["NumRotatableBonds"])
        assert result["NumRotatableBonds"] == pytest.approx(1.0)

    def test_num_rotatable_bonds_of_ethane_is_zero(self):
        """Ethane (CC) has no rotatable bonds (both carbons are terminal)."""
        result = _compute_properties("CC", ["NumRotatableBonds"])
        assert result["NumRotatableBonds"] == pytest.approx(0.0)

    def test_num_aliphatic_rings_of_cyclohexane(self):
        """Cyclohexane has one aliphatic ring."""
        result = _compute_properties("C1CCCCC1", ["NumAliphaticRings"])
        assert result["NumAliphaticRings"] == pytest.approx(1.0)

    def test_num_aliphatic_rings_of_benzene_is_zero(self):
        """Benzene has no aliphatic rings (it is fully aromatic)."""
        result = _compute_properties("c1ccccc1", ["NumAliphaticRings"])
        assert result["NumAliphaticRings"] == pytest.approx(0.0)

    def test_num_aromatic_rings_of_benzene(self):
        """Benzene has exactly one aromatic ring."""
        result = _compute_properties("c1ccccc1", ["NumAromaticRings"])
        assert result["NumAromaticRings"] == pytest.approx(1.0)

    def test_num_aromatic_rings_of_naphthalene(self):
        """Naphthalene has two aromatic rings."""
        result = _compute_properties("c1ccc2ccccc2c1", ["NumAromaticRings"])
        assert result["NumAromaticRings"] == pytest.approx(2.0)

    def test_mollogp_of_ethanol_is_negative(self):
        """Ethanol MolLogP is negative (hydrophilic molecule)."""
        result = _compute_properties("CCO", ["MolLogP"])
        assert float(result["MolLogP"]) < 0.0

    def test_qed_is_between_zero_and_one(self):
        """QED is always in [0, 1] for any valid molecule."""
        result = _compute_properties("CC(=O)Nc1ccc(O)cc1", ["QED"])
        assert 0.0 <= float(result["QED"]) <= 1.0


VALID_SMILES_LINES = [
    "c1ccccc1",
    "CC(=O)O",
    "CCO",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "c1ccc2ccccc2c1",
]
INVALID_SMILES_LINE = "NOTASMILES"


def _make_mock_response(smiles_lines: list[str]) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.side_effect = lambda **kw: iter(line.encode() for line in smiles_lines)
    return mock_resp


class TestGuacaMolSingleFileLoad:
    """Tests for GuacaMol dataset loading from a single combined corpus file."""

    def test_load_returns_correct_number_of_candidates(self, tmp_path):
        """Dataset contains exactly max_molecules candidates after loading."""
        config = _base_config(data_dir=tmp_path, max_molecules=5)
        with patch("requests.get", return_value=_make_mock_response(VALID_SMILES_LINES)):
            dataset = GuacaMol(config)
        assert len(dataset._raw_dataset) == 5

    def test_labels_are_1d_numpy_array(self, tmp_path):
        """Labels are stored as a 1D numpy array of length max_molecules."""
        config = _base_config(data_dir=tmp_path, max_molecules=5)
        with patch("requests.get", return_value=_make_mock_response(VALID_SMILES_LINES)):
            dataset = GuacaMol(config)
        assert isinstance(dataset._raw_dataset.labels, np.ndarray)
        assert dataset._raw_dataset.labels.ndim == 1
        assert len(dataset._raw_dataset.labels) == 5

    def test_candidate_features_contain_computed_properties(self, tmp_path):
        """Each candidate's features dict includes all requested computed properties."""
        config = _base_config(
            data_dir=tmp_path, max_molecules=3, computed_properties=["TPSA", "MolWt"]
        )
        with patch("requests.get", return_value=_make_mock_response(VALID_SMILES_LINES)):
            dataset = GuacaMol(config)
        assert dataset._raw_dataset is not None
        for cand in dataset._raw_dataset.candidates:
            if cand.features is not None:
                assert "TPSA" in cand.features
                assert "MolWt" in cand.features

    def test_invalid_smiles_are_skipped_and_not_in_dataset(self, tmp_path):
        """Invalid SMILES strings are silently skipped and excluded from candidates."""
        lines_with_invalid = VALID_SMILES_LINES[:3] + [INVALID_SMILES_LINE] + VALID_SMILES_LINES[3:]
        config = _base_config(data_dir=tmp_path, max_molecules=len(lines_with_invalid))
        with patch("requests.get", return_value=_make_mock_response(lines_with_invalid)):
            dataset = GuacaMol(config)
        assert dataset._raw_dataset is not None
        assert len(dataset._raw_dataset) == len(VALID_SMILES_LINES)

    def test_max_molecules_caps_corpus_size(self, tmp_path):
        """Dataset size does not exceed max_molecules."""
        config = _base_config(data_dir=tmp_path, max_molecules=3)
        with patch("requests.get", return_value=_make_mock_response(VALID_SMILES_LINES)):
            dataset = GuacaMol(config)
        assert dataset._raw_dataset is not None
        assert len(dataset._raw_dataset) <= 3

    def test_benchmark_task_target_raises_not_implemented(self, tmp_path):
        """NotImplementedError is raised when target_property is a benchmark task."""
        config = _base_config(data_dir=tmp_path, target_property="celecoxib_rediscovery")
        with pytest.raises(NotImplementedError):
            GuacaMol(config)

    def test_no_download_if_file_already_cached(self, tmp_path):
        """requests.get is not called when the corpus file already exists on disk."""
        (tmp_path / FILENAME_ALL).write_text("\n".join(VALID_SMILES_LINES))
        config = _base_config(data_dir=tmp_path)  # max_molecules=None → uses FILENAME_ALL directly
        with patch("requests.get") as mock_get:
            GuacaMol(config)
        mock_get.assert_not_called()


TRAIN_SMILES = ["c1ccccc1", "CC(=O)O", "CCO"]
PAPER_VALID_SMILES = ["CC(C)Cc1ccc(cc1)C(C)C(=O)O"]
PAPER_TEST_SMILES = ["c1ccc2ccccc2c1", "c1ccncc1"]


def _paper_config(**overrides) -> GuacaMolConfig:
    defaults = dict(
        name="guacamol",
        modality="sequence",
        seed=42,
        train_ratio=0.6,
        validation_frac=0.1,
        test_ratio=0.2,
        target_property="TPSA",
        split_mode="paper",
    )
    defaults.update(overrides)
    return GuacaMolConfig(**defaults)


def _write_paper_files(tmp_path):
    (tmp_path / FILENAME_TRAIN).write_text("\n".join(TRAIN_SMILES))
    (tmp_path / FILENAME_VALID).write_text("\n".join(PAPER_VALID_SMILES))
    (tmp_path / FILENAME_TEST).write_text("\n".join(PAPER_TEST_SMILES))


class TestGuacaMolPaperSplits:
    """Tests for GuacaMol paper-split mode using original figshare file boundaries."""

    def test_train_split_contains_train_file_smiles(self, tmp_path):
        """train_dataset candidates exactly match the train split file contents."""
        _write_paper_files(tmp_path)
        dataset = GuacaMol(_paper_config(data_dir=tmp_path))
        assert {c.data for c in dataset.train_dataset.candidates} == set(TRAIN_SMILES)

    def test_validation_split_contains_valid_file_smiles(self, tmp_path):
        """validation_dataset candidates exactly match the valid split file contents."""
        _write_paper_files(tmp_path)
        dataset = GuacaMol(_paper_config(data_dir=tmp_path))
        assert {c.data for c in dataset.validation_dataset.candidates} == set(PAPER_VALID_SMILES)

    def test_test_split_contains_test_file_smiles(self, tmp_path):
        """test_dataset candidates exactly match the test split file contents."""
        _write_paper_files(tmp_path)
        dataset = GuacaMol(_paper_config(data_dir=tmp_path))
        assert {c.data for c in dataset.test_dataset.candidates} == set(PAPER_TEST_SMILES)

    def test_candidate_pool_is_empty_for_paper_splits(self, tmp_path):
        """candidate_pool is empty when using paper splits (no residual pool)."""
        _write_paper_files(tmp_path)
        dataset = GuacaMol(_paper_config(data_dir=tmp_path))
        assert len(dataset.candidate_pool) == 0

    def test_paper_split_candidates_have_no_split_feature_tag(self, tmp_path):
        """Candidate features must NOT contain a 'split' key for paper-split mode."""
        _write_paper_files(tmp_path)
        dataset = GuacaMol(_paper_config(data_dir=tmp_path))
        all_candidates = (
            dataset.train_dataset.candidates
            + dataset.validation_dataset.candidates
            + dataset.test_dataset.candidates
        )
        for cand in all_candidates:
            assert "split" not in (cand.features or {})


class TestGuacaMolQuery:
    """Tests for GuacaMol.query(), including in-corpus lookup and on-the-fly RDKit labelling."""

    def _loaded_dataset(self, tmp_path):
        config = _base_config(data_dir=tmp_path, max_molecules=5)
        with patch("requests.get", return_value=_make_mock_response(VALID_SMILES_LINES)):
            return GuacaMol(config)

    def test_query_known_candidate_returns_precomputed_label(self, tmp_path):
        """Querying a corpus candidate returns its precomputed label without recomputing."""
        dataset = self._loaded_dataset(tmp_path)
        known = dataset._raw_dataset.candidates[0]
        result = dataset.query([known])
        assert len(result) == 1
        assert result.labels[0] == pytest.approx(dataset._raw_dataset.labels[0], rel=1e-9)

    def test_query_novel_smiles_computes_label_via_rdkit(self, tmp_path):
        """Querying a novel SMILES string computes its label on the fly via RDKit."""
        dataset = self._loaded_dataset(tmp_path)
        novel = Candidate(data="c1ccncc1", modality=Modality.SEQUENCE)  # pyridine, not in corpus
        result = dataset.query([novel])
        assert len(result) == 1
        expected = _compute_properties("c1ccncc1", ["TPSA"])["TPSA"]
        assert result.labels[0] == pytest.approx(expected, rel=1e-6)

    def test_query_mixed_known_and_novel_returns_both(self, tmp_path):
        """Querying a mix of corpus and novel candidates returns labels for all."""
        dataset = self._loaded_dataset(tmp_path)
        known = dataset._raw_dataset.candidates[0]
        novel = Candidate(data="c1ccncc1", modality=Modality.SEQUENCE)
        result = dataset.query([known, novel])
        assert len(result) == 2
        assert result.labels.ndim == 1

    def test_query_returns_caller_candidate_not_corpus_object(self, tmp_path):
        """query() returns the caller's Candidate object, not the stored corpus object."""
        dataset = self._loaded_dataset(tmp_path)
        # Build a candidate whose SMILES is in the corpus but with extra metadata.
        corpus_smiles = dataset._raw_dataset.candidates[0].data
        caller_candidate = Candidate(
            data=corpus_smiles,
            modality=Modality.SEQUENCE,
            features={"custom_tag": 999.0},
        )
        result = dataset.query([caller_candidate])
        assert result.candidates[0] is caller_candidate
        assert result.candidates[0].features is not None
        assert result.candidates[0].features.get("custom_tag") == 999.0

    def test_query_non_canonical_smiles_hits_corpus(self, tmp_path):
        """Querying with a non-canonical SMILES for a corpus molecule returns the corpus label."""
        dataset = self._loaded_dataset(tmp_path)
        # "CCO" (ethanol, canonical) is in VALID_SMILES_LINES; "OCC" is the same molecule
        # written differently.  Both should resolve to the same corpus label.
        canonical_candidate = Candidate(data="CCO", modality=Modality.SEQUENCE)
        non_canonical_candidate = Candidate(data="OCC", modality=Modality.SEQUENCE)
        result_canonical = dataset.query([canonical_candidate])
        result_non_canonical = dataset.query([non_canonical_candidate])
        assert result_canonical.labels[0] == pytest.approx(result_non_canonical.labels[0], rel=1e-9)


# ---------------------------------------------------------------------------
# Fixture-based tests — cover gaps identified in audit
# ---------------------------------------------------------------------------


class TestLoadSmilesFile:
    """Unit tests for the _load_smiles_file helper."""

    def test_returns_non_empty_lines_only(self):
        """All returned strings are non-empty after stripping whitespace."""
        result = _load_smiles_file(VALID_FIXTURE)
        assert all(line.strip() for line in result)

    def test_empty_file_returns_empty_list(self):
        """An empty file returns an empty list."""
        result = _load_smiles_file(EMPTY_FIXTURE)
        assert result == []

    def test_valid_fixture_count_matches_non_blank_lines(self):
        """Returned list length equals the number of non-blank lines in the file."""
        expected = sum(1 for ln in VALID_FIXTURE.read_text().splitlines() if ln.strip())
        assert len(_load_smiles_file(VALID_FIXTURE)) == expected


class TestDownloadFile:
    """Unit tests for _download_file error path."""

    def test_non_200_status_raises_file_not_found(self, tmp_path):
        """FileNotFoundError is raised when the server returns a non-200 status."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(FileNotFoundError, match="404"):
                _download_file("https://example.com/fake.smiles", tmp_path / "out.smiles", None)

    def test_successful_download_writes_file(self, tmp_path):
        """A successful download writes all lines to the destination file."""
        lines = [b"c1ccccc1", b"CCO", b"CC(=O)O"]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)
        out = tmp_path / "out.smiles"
        with patch("requests.get", return_value=mock_resp):
            _download_file("https://example.com/fake.smiles", out, None)
        assert out.exists()
        assert out.read_text().count("\n") == 3

    def test_max_lines_cap_limits_written_lines(self, tmp_path):
        """max_lines truncates the downloaded file to at most that many lines."""
        lines = [b"c1ccccc1", b"CCO", b"CC(=O)O", b"c1ccncc1", b"NCCc1ccc(O)c(O)c1"]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)
        out_base = tmp_path / "out.smiles"
        with patch("requests.get", return_value=mock_resp):
            result = _download_file("https://example.com/fake.smiles", out_base, 2)
        written = [ln for ln in result.read_text().splitlines() if ln.strip()]
        assert len(written) == 2

    def test_smaller_max_lines_cache_does_not_block_larger_request(self, tmp_path):
        """A cached file from max_lines=2 does not prevent a fresh download for max_lines=5."""
        out_base = tmp_path / "out.smiles"
        _cache_path(out_base, 2).write_bytes(b"c1ccccc1\nCCO\n")

        fresh_lines = [b"c1ccccc1", b"CCO", b"CC(=O)O", b"c1ccncc1", b"NCCc1ccc(O)c(O)c1"]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(fresh_lines)

        with patch("requests.get", return_value=mock_resp):
            result = _download_file("https://example.com/fake.smiles", out_base, max_lines=5)

        written = [ln for ln in result.read_text().splitlines() if ln.strip()]
        assert len(written) == 5

    def test_cache_not_re_downloaded_when_sufficient(self, tmp_path):
        """If the exact max_lines-encoded cache file exists, download is skipped."""
        out_base = tmp_path / "out.smiles"
        _cache_path(out_base, 2).write_bytes(b"c1ccccc1\nCCO\n")

        with patch("requests.get") as mock_get:
            _download_file("https://example.com/fake.smiles", out_base, max_lines=2)
            mock_get.assert_not_called()

    def test_max_lines_encoded_in_returned_filename(self, tmp_path):
        """When max_lines is set, the returned path encodes the count in its filename."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter([b"c1ccccc1", b"CCO"])
        out_base = tmp_path / "out.smiles"
        with patch("requests.get", return_value=mock_resp):
            result = _download_file("https://example.com/fake.smiles", out_base, max_lines=2)
        assert result == _cache_path(out_base, 2)
        assert result.exists()
        assert not out_base.exists()

    def test_no_max_lines_uses_original_filename(self, tmp_path):
        """When max_lines is None the returned path is the original filepath."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter([b"c1ccccc1"])
        out = tmp_path / "out.smiles"
        with patch("requests.get", return_value=mock_resp):
            result = _download_file("https://example.com/fake.smiles", out, max_lines=None)
        assert result == out

    def test_different_max_lines_use_separate_cache_files(self, tmp_path):
        """max_lines=2 and max_lines=3 each create a distinct file; neither shadows the other."""

        def make_mock(lines):
            m = MagicMock()
            m.status_code = 200
            m.iter_lines.return_value = iter(ln.encode() for ln in lines)
            return m

        out_base = tmp_path / "out.smiles"
        with patch("requests.get", return_value=make_mock(["c1ccccc1", "CCO"])):
            path2 = _download_file("https://example.com/fake.smiles", out_base, max_lines=2)
        with patch("requests.get", return_value=make_mock(["c1ccccc1", "CCO", "CC(=O)O"])):
            path3 = _download_file("https://example.com/fake.smiles", out_base, max_lines=3)

        assert path2 != path3
        assert path2.exists()
        assert path3.exists()

    def test_partial_download_leaves_no_stale_filepath(self, tmp_path):
        """A mid-stream network failure leaves no file at the destination path."""

        def failing_iter(**kw):
            yield b"c1ccccc1"
            raise OSError("connection reset")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.side_effect = failing_iter
        out = tmp_path / "out.smiles"
        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(OSError):
                _download_file("https://example.com/fake.smiles", out)
        assert not out.exists()


class TestGuacaMolWithFixtures:
    """Integration tests using synthetic fixtures — no network calls."""

    def _config(self, tmp_path, **overrides):
        defaults = dict(
            name="guacamol",
            modality="sequence",
            seed=42,
            train_ratio=0.6,
            validation_frac=0.1,
            test_ratio=0.2,
            target_property="TPSA",
            data_dir=tmp_path,
        )
        defaults.update(overrides)
        return GuacaMolConfig(**defaults)

    def test_load_from_valid_fixture_returns_candidates(self, tmp_path):
        """Loading from a valid SMILES fixture produces at least one candidate."""
        shutil.copy(VALID_FIXTURE, tmp_path / FILENAME_ALL)
        config = self._config(tmp_path)
        dataset = GuacaMol(config)
        assert dataset._raw_dataset is not None
        assert len(dataset._raw_dataset) > 0

    def test_candidate_data_field_is_smiles_string(self, tmp_path):
        """Each candidate's data field is a non-empty SMILES string."""
        shutil.copy(VALID_FIXTURE, tmp_path / FILENAME_ALL)
        config = self._config(tmp_path)
        dataset = GuacaMol(config)
        assert dataset._raw_dataset is not None
        candidate = dataset._raw_dataset.candidates[0]
        assert isinstance(candidate.data, str)
        assert len(candidate.data) > 0

    def test_computed_properties_none_stores_all_ten_in_features(self, tmp_path):
        """computed_properties=None stores all 10 GuacaMol properties in candidate features."""
        shutil.copy(VALID_FIXTURE, tmp_path / FILENAME_ALL)
        config = self._config(tmp_path, computed_properties=None)
        dataset = GuacaMol(config)
        assert dataset._raw_dataset is not None
        for cand in dataset._raw_dataset.candidates:
            assert ALL_PROPERTIES.issubset(set(cand.features.keys()))

    def test_empty_smiles_file_produces_empty_dataset(self, tmp_path):
        """An empty fixture file results in a dataset with zero candidates."""
        shutil.copy(EMPTY_FIXTURE, tmp_path / FILENAME_ALL)
        config = self._config(tmp_path)
        dataset = GuacaMol(config)
        assert dataset._raw_dataset is not None
        assert len(dataset._raw_dataset) == 0

    def test_all_invalid_smiles_skipped_leaves_empty_dataset(self, tmp_path):
        """A fixture containing only invalid SMILES results in an empty dataset."""
        shutil.copy(INVALID_FIXTURE, tmp_path / FILENAME_ALL)
        config = self._config(tmp_path)
        dataset = GuacaMol(config)
        assert dataset._raw_dataset is not None
        assert len(dataset._raw_dataset) == 0

    @pytest.mark.integration
    def test_large_fixture_loads_without_crash(self, tmp_path):
        """A 1000-SMILES fixture loads successfully without errors."""
        shutil.copy(LARGE_FIXTURE, tmp_path / FILENAME_ALL)
        config = self._config(tmp_path)
        dataset = GuacaMol(config)
        assert dataset._raw_dataset is not None
        assert len(dataset._raw_dataset) > 0

    def test_query_benchmark_task_raises_not_implemented(self, tmp_path):
        """query() raises NotImplementedError when task_type is 'benchmark_task'."""
        shutil.copy(VALID_FIXTURE, tmp_path / FILENAME_ALL)
        config = self._config(tmp_path)
        dataset = GuacaMol(config)
        assert dataset._raw_dataset is not None
        cand = dataset._raw_dataset.candidates[0]
        # Benchmark tasks also raise in load_dataset(), preventing normal construction.
        # Use model_copy to reach the guard in query() independently.
        dataset.config = dataset.config.model_copy(update={"task_type": "benchmark_task"})
        with pytest.raises(NotImplementedError):
            dataset.query([cand])

    def test_paper_splits_cache_miss_triggers_download(self, tmp_path):
        """When paper split files are absent, requests.get is called once per split file."""
        smiles = ["c1ccccc1", "CCO", "CC(=O)O"]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # Use side_effect so each call to iter_lines() gets a fresh iterator; return_value
        # would share a single exhausted iterator across all three file downloads.
        mock_resp.iter_lines.side_effect = lambda **kw: iter(ln.encode() for ln in smiles)
        config = self._config(tmp_path, split_mode="paper")
        with patch("requests.get", return_value=mock_resp) as mock_get:
            dataset = GuacaMol(config)
        assert mock_get.call_count == 3
        # Each split file produced real candidates (iterator was not exhausted).
        assert len(dataset.train_dataset.candidates) > 0
        assert len(dataset.validation_dataset.candidates) > 0
        assert len(dataset.test_dataset.candidates) > 0

    def test_paper_splits_max_molecules_caps_each_split(self, tmp_path):
        """max_molecules caps the candidate count in each paper split independently."""
        for filename in [FILENAME_TRAIN, FILENAME_VALID, FILENAME_TEST]:
            shutil.copy(LARGE_FIXTURE, _cache_path(tmp_path / filename, 10))
        config = self._config(tmp_path, split_mode="paper", max_molecules=10)
        dataset = GuacaMol(config)
        train_count = len(dataset.train_dataset.candidates)
        valid_count = len(dataset.validation_dataset.candidates)
        test_count = len(dataset.test_dataset.candidates)
        assert train_count <= 10
        assert valid_count <= 10
        assert test_count <= 10


class TestGuacaMolEdgeCases:
    """Edge-case tests for truncation, invalid SMILES in query, and featurise paths."""

    def _config(self, tmp_path, **overrides):
        defaults = dict(
            name="guacamol",
            modality="sequence",
            seed=42,
            train_ratio=0.6,
            validation_frac=0.1,
            test_ratio=0.2,
            target_property="TPSA",
            data_dir=tmp_path,
        )
        defaults.update(overrides)
        return GuacaMolConfig(**defaults)

    def test_max_molecules_zero_produces_empty_dataset(self, tmp_path):
        """max_molecules=0 should yield an empty dataset without raising."""
        shutil.copy(VALID_FIXTURE, _cache_path(tmp_path / FILENAME_ALL, 0))
        config = self._config(tmp_path, max_molecules=0)
        dataset = GuacaMol(config)
        total = (
            len(dataset.train_dataset)
            + len(dataset.validation_dataset)
            + len(dataset.test_dataset)
            + len(dataset.candidate_pool)
        )
        assert total == 0

    def test_query_invalid_novel_smiles_raises_value_error(self, tmp_path):
        """query() must raise ValueError for a novel candidate with an unparseable SMILES."""
        shutil.copy(VALID_FIXTURE, tmp_path / FILENAME_ALL)
        config = self._config(tmp_path, target_property="MolWt")
        dataset = GuacaMol(config)
        bad = Candidate(data="not_a_smiles!!!", modality="sequence")
        with pytest.raises(ValueError, match="invalid SMILES"):
            dataset.query([bad])

    def test_computed_properties_stored_as_candidate_features(self, tmp_path):
        """Each Candidate must carry exactly the requested computed_properties as features."""
        shutil.copy(VALID_FIXTURE, tmp_path / FILENAME_ALL)
        config = self._config(
            tmp_path,
            target_property="MolWt",
            computed_properties=["MolWt", "MolLogP"],
        )
        dataset = GuacaMol(config)
        all_splits = [
            dataset.train_dataset,
            dataset.validation_dataset,
            dataset.test_dataset,
            dataset.candidate_pool,
        ]
        for split in all_splits:
            for candidate in split.candidates:
                assert candidate.features is not None
                assert "MolWt" in candidate.features
                assert "MolLogP" in candidate.features
                assert isinstance(candidate.features["MolWt"], float)
                assert isinstance(candidate.features["MolLogP"], float)


class TestDownloadGuacaMol:
    """Tests for the download_guacamol public function."""

    def _mock_response(self, lines: list[str] | None = None) -> MagicMock:
        lines = lines or ["c1ccccc1", "CCO"]
        mock = MagicMock()
        mock.status_code = 200
        mock.iter_lines.side_effect = lambda **kw: iter(ln.encode() for ln in lines)
        return mock

    def test_downloads_all_four_files(self, tmp_path):
        """download_guacamol writes all four .smiles files to the target directory."""
        with patch("requests.get", return_value=self._mock_response()):
            download_guacamol(data_dir=tmp_path)
        assert (tmp_path / FILENAME_ALL).exists()
        assert (tmp_path / FILENAME_TRAIN).exists()
        assert (tmp_path / FILENAME_VALID).exists()
        assert (tmp_path / FILENAME_TEST).exists()

    def test_makes_exactly_four_network_requests(self, tmp_path):
        """download_guacamol calls requests.get exactly once per file."""
        with patch("requests.get", return_value=self._mock_response()) as mock_get:
            download_guacamol(data_dir=tmp_path)
        assert mock_get.call_count == 4

    def test_files_written_inside_data_dir_not_datapath(self, tmp_path):
        """Files appear in the provided data_dir, not in the package DATAPATH."""
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        with patch("requests.get", return_value=self._mock_response()):
            download_guacamol(data_dir=other_dir)
        for filename in [FILENAME_ALL, FILENAME_TRAIN, FILENAME_VALID, FILENAME_TEST]:
            assert (other_dir / filename).exists()
            assert not (tmp_path / filename).exists()

    def test_max_lines_caps_each_downloaded_file(self, tmp_path):
        """max_lines is forwarded and each file is truncated to at most that many lines."""
        many_lines = ["c1ccccc1"] * 10
        with patch("requests.get", return_value=self._mock_response(many_lines)):
            download_guacamol(data_dir=tmp_path, max_lines=3)
        for filename in [FILENAME_ALL, FILENAME_TRAIN, FILENAME_VALID, FILENAME_TEST]:
            cache_file = _cache_path(tmp_path / filename, 3)
            lines = [ln for ln in cache_file.read_text().splitlines() if ln.strip()]
            assert 1 <= len(lines) <= 3

    def test_default_data_dir_is_datapath(self):
        """download_guacamol's default data_dir parameter equals the module DATAPATH constant."""
        sig = inspect.signature(download_guacamol)
        assert sig.parameters["data_dir"].default == DATAPATH

    def test_creates_data_dir_if_missing(self, tmp_path):
        """download_guacamol creates the target directory if it does not exist."""
        new_dir = tmp_path / "nonexistent" / "nested"
        with patch("requests.get", return_value=self._mock_response()):
            download_guacamol(data_dir=new_dir)
        assert new_dir.exists()
        assert (new_dir / FILENAME_ALL).exists()
