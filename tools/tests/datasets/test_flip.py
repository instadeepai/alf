# Copyright 2026 InstaDeep Ltd. All rights reserved.
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

import io
import zipfile
from math import floor
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from alf_core import Modality
from alf_tools.datasets.flip import FLIP, FLIP_SPLITS, FLIPConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_splits_zip(csv_name: str, df: pd.DataFrame) -> bytes:
    """Build an in-memory zip file containing a single CSV.

    Returns:
        Raw bytes of the zip archive.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(csv_name, df.to_csv(index=False))
    return buf.getvalue()


def _make_mock_df(n_train: int = 10, n_test: int = 5) -> pd.DataFrame:
    """Create a synthetic FLIP-format DataFrame with known values.

    Train rows get labels 1.0 .. n_train; test rows get labels 100.0 .. 100+n_test-1.
    The 'validation' flag is set True on the last 2 train rows to verify it is
    stored as a feature but does not affect splitting.

    Returns:
        DataFrame with columns: sequence, target, set, validation.
    """
    rows = []
    for i in range(n_train):
        rows.append({
            "sequence": f"TRAINSEQ{i:04d}",
            "target": float(i + 1),
            "set": "train",
            "validation": i >= n_train - 2,  # last 2 rows flagged True
        })
    for i in range(n_test):
        rows.append({
            "sequence": f"TESTSEQQ{i:04d}",
            "target": float(100 + i),
            "set": "test",
            "validation": False,
        })
    return pd.DataFrame(rows)


def _make_flip_config(**overrides) -> FLIPConfig:
    defaults = dict(
        name="flip_test",
        modality="sequence",
        seed=42,
        flip_dataset="gb1",
        flip_split="one_vs_rest",
        train_ratio=0.5,
        validation_frac=0.2,
        test_ratio=1.0,
        problem_type="regression",
    )
    defaults.update(overrides)
    return FLIPConfig(**defaults)


def _make_flip_instance(df: pd.DataFrame, **config_overrides) -> FLIP:
    """Create a FLIP instance with a mocked _load_split_dataframe.

    Returns:
        Initialised FLIP dataset backed by the provided DataFrame.
    """
    config = _make_flip_config(**config_overrides)
    with patch.object(FLIP, "_load_split_dataframe", return_value=df):
        instance = FLIP(config)
    return instance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_df():
    """Standard synthetic DataFrame: 10 train + 5 test rows.

    Returns:
        DataFrame with columns: sequence, target, set, validation.
    """
    return _make_mock_df()


@pytest.fixture
def flip_dataset(mock_df):
    """FLIP instance backed by the standard mock DataFrame.

    Config: train_ratio=0.5, validation_frac=0.2, test_ratio=1.0, seed=42.
    With 10 flip_train rows:
      train_plus_val = floor(10 * 0.5) = 5
      validation     = floor(5 * 0.2)  = 1
      train          = 5 - 1           = 4
      candidate_pool = 10 - 5          = 5
      test           = floor(5 * 1.0)  = 5

    Returns:
        Initialised FLIP dataset backed by the mock DataFrame.
    """
    return _make_flip_instance(mock_df)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestFLIPConfig:
    """Tests for FLIPConfig validation."""

    def test_invalid_split_raises(self):
        """Invalid split name for the chosen dataset should raise at config construction."""
        # "des_mut" is a valid aav split but not valid for gb1 — caught by validate_config
        with pytest.raises(Exception, match="not a valid active split"):
            _make_flip_config(flip_split="des_mut")

    def test_valid_splits_accepted(self):
        """All documented active splits should be accepted without error."""
        for dataset, splits in FLIP_SPLITS.items():
            for split in splits:
                config = _make_flip_config(flip_dataset=dataset, flip_split=split)
                with patch.object(FLIP, "_load_split_dataframe", return_value=_make_mock_df()):
                    instance = FLIP(config)
                assert instance.config.flip_dataset == dataset
                assert instance.config.flip_split == split

    def test_invalid_dataset_raises(self):
        """An unrecognised dataset name should be rejected by Pydantic."""
        with pytest.raises(Exception):
            _make_flip_config(flip_dataset="unknown_dataset")


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestFLIPInitialisation:
    """Tests for post-construction state of a FLIP instance."""

    def test_config_stored(self, flip_dataset):
        """Dataset and split names should be accessible via config."""
        assert flip_dataset.config.name == "flip_test"
        assert flip_dataset.config.flip_dataset == "gb1"
        assert flip_dataset.config.flip_split == "one_vs_rest"

    def test_modality(self, flip_dataset):
        """Modality should match the value given in config."""
        assert flip_dataset.modality == Modality.SEQUENCE

    def test_seed_stored(self, flip_dataset):
        """Seed should be preserved for reproducibility."""
        assert flip_dataset.config.seed == 42

    def test_repr(self, flip_dataset):
        """Repr should include dataset and split name for quick identification."""
        r = repr(flip_dataset)
        assert "gb1" in r
        assert "one_vs_rest" in r


# ---------------------------------------------------------------------------
# Split partitioning sizes
# ---------------------------------------------------------------------------


class TestFLIPSplitSizes:
    """Mock data: 10 flip_train + 5 flip_test.
    Default config: train_ratio=0.5, validation_frac=0.2, test_ratio=1.0.

    Expected sizes:
      train_plus_val = floor(10 * 0.5) = 5
      validation     = floor(5 * 0.2)  = 1
      train          = 4
      candidate_pool = 5
      test           = floor(5 * 1.0)  = 5
    """

    def test_train_size(self, flip_dataset):
        """Train size equals floor(10*0.5) - floor(5*0.2) = 4."""
        assert len(flip_dataset.train_dataset) == 4

    def test_validation_size(self, flip_dataset):
        """Validation size equals floor(floor(10*0.5) * 0.2) = 1."""
        assert len(flip_dataset.validation_dataset) == 1

    def test_test_size(self, flip_dataset):
        """Test size equals floor(5 * 1.0) = 5."""
        assert len(flip_dataset.test_dataset) == 5

    def test_candidate_pool_size(self, flip_dataset):
        """Candidate pool equals remaining flip_train after train+val = 5."""
        assert len(flip_dataset.candidate_pool) == 5

    def test_zero_train_ratio_empties_train(self, mock_df):
        """train_ratio=0 should produce empty train/validation; all flip_train to candidate_pool."""
        instance = _make_flip_instance(mock_df, train_ratio=0.0, validation_frac=0.0)
        assert len(instance.train_dataset) == 0
        assert len(instance.validation_dataset) == 0
        assert len(instance.candidate_pool) == 10

    def test_max_candidate_pool_respected(self, mock_df):
        """max_candidate_pool should cap the candidate pool size."""
        instance = _make_flip_instance(mock_df, max_candidate_pool=3)
        assert len(instance.candidate_pool) == 3

    def test_max_candidate_pool_larger_than_remaining(self, mock_df):
        """max_candidate_pool > remaining rows should not expand the pool."""
        instance = _make_flip_instance(mock_df, max_candidate_pool=100)
        assert len(instance.candidate_pool) == 5  # only 5 rows remain after train+val

    def test_partial_test_ratio(self, mock_df):
        """test_ratio < 1.0 should sample a subset of flip_test rows."""
        instance = _make_flip_instance(mock_df, test_ratio=0.6)
        assert len(instance.test_dataset) == floor(5 * 0.6)

    def test_total_raw_dataset_size(self, flip_dataset, mock_df):
        """Raw dataset should contain all rows (train + test)."""
        assert len(flip_dataset._raw_dataset) == len(mock_df)


# ---------------------------------------------------------------------------
# Partitioning correctness (no cross-contamination)
# ---------------------------------------------------------------------------


class TestFLIPSplitCorrectness:
    """Tests that splits are mutually exclusive and draw from the correct FLIP pools."""

    def test_candidate_pool_contains_no_test_sequences(self, flip_dataset):
        """No sequence should appear in both candidate pool and test."""
        pool_seqs = {c.data for c in flip_dataset.candidate_pool.candidates}
        test_seqs = {c.data for c in flip_dataset.test_dataset.candidates}
        assert pool_seqs.isdisjoint(test_seqs)

    def test_validation_contains_no_test_sequences(self, flip_dataset):
        """No sequence should appear in both validation and test."""
        val_seqs = {c.data for c in flip_dataset.validation_dataset.candidates}
        test_seqs = {c.data for c in flip_dataset.test_dataset.candidates}
        assert val_seqs.isdisjoint(test_seqs)

    def test_train_contains_no_test_sequences(self, flip_dataset):
        """No sequence should appear in both train and test."""
        train_seqs = {c.data for c in flip_dataset.train_dataset.candidates}
        test_seqs = {c.data for c in flip_dataset.test_dataset.candidates}
        assert train_seqs.isdisjoint(test_seqs)

    def test_candidate_pool_is_subset_of_flip_train(self, flip_dataset):
        """Candidate pool sequences should all come from the FLIP train set."""
        pool_seqs = {c.data for c in flip_dataset.candidate_pool.candidates}
        flip_train_seqs = {
            c.data
            for c in flip_dataset._raw_dataset.candidates
            if c.features and c.features["set"] == "train"
        }
        assert pool_seqs.issubset(flip_train_seqs)

    def test_test_sequences_are_subset_of_flip_test(self, flip_dataset):
        """Test sequences should all come from the FLIP test set."""
        test_seqs = {c.data for c in flip_dataset.test_dataset.candidates}
        flip_test_seqs = {
            c.data
            for c in flip_dataset._raw_dataset.candidates
            if c.features and c.features["set"] == "test"
        }
        assert test_seqs.issubset(flip_test_seqs)

    def test_candidate_features_set_column(self, flip_dataset):
        """Every candidate should carry the 'set' feature."""
        for candidate in flip_dataset._raw_dataset.candidates:
            assert candidate.features is not None
            assert "set" in candidate.features
            assert candidate.features["set"] in ("train", "test")

    def test_candidate_features_validation_column(self, flip_dataset):
        """Every candidate should carry the boolean 'validation' feature."""
        for candidate in flip_dataset._raw_dataset.candidates:
            assert candidate.features is not None
            assert "validation" in candidate.features
            assert isinstance(candidate.features["validation"], bool)

    def test_flip_validation_flag_not_used_for_splitting(self, mock_df):
        """Candidates with FLIP validation=True should appear in train/validation/candidate_pool,
        not be excluded from the pool.
        """
        # With train_ratio=0, all flip_train (including validation=True rows) go to candidate_pool
        instance = _make_flip_instance(mock_df, train_ratio=0.0, validation_frac=0.0)
        # All 10 flip_train rows (including the 2 with validation=True) are in candidate_pool
        assert len(instance.candidate_pool) == 10


# ---------------------------------------------------------------------------
# Label statistics
# ---------------------------------------------------------------------------


class TestFLIPLabelStatistics:
    """Use train_ratio=0, validation_frac=0, test_ratio=1.0 for predictable label pools.

    All 10 flip_train rows (labels 1..10) → candidate_pool; mean = 5.5.
    All 5 flip_test rows (labels 100..104) → test; mean = 102.0.
    """

    @pytest.fixture
    def full_pool_instance(self, mock_df):
        """FLIP instance with train_ratio=0 so all flip_train goes to candidate_pool.

        Returns:
            Initialised FLIP dataset with predictable label pools.
        """
        return _make_flip_instance(mock_df, train_ratio=0.0, validation_frac=0.0, test_ratio=1.0)

    def test_candidate_pool_label_mean(self, full_pool_instance):
        """Candidate pool labels 1..10 should have mean 5.5."""
        assert np.mean(full_pool_instance.candidate_pool.labels) == pytest.approx(5.5)

    def test_test_label_mean(self, full_pool_instance):
        """Test labels 100..104 should have mean 102.0."""
        assert np.mean(full_pool_instance.test_dataset.labels) == pytest.approx(102.0)

    def test_labels_are_numpy_arrays(self, flip_dataset):
        """All split label arrays should be numpy ndarrays."""
        assert isinstance(flip_dataset.train_dataset.labels, np.ndarray)
        assert isinstance(flip_dataset.validation_dataset.labels, np.ndarray)
        assert isinstance(flip_dataset.test_dataset.labels, np.ndarray)
        assert isinstance(flip_dataset.candidate_pool.labels, np.ndarray)

    def test_non_empty_splits(self, flip_dataset):
        """All splits should be non-empty with default ratios."""
        assert len(flip_dataset.train_dataset) > 0
        assert len(flip_dataset.validation_dataset) > 0
        assert len(flip_dataset.test_dataset) > 0
        assert len(flip_dataset.candidate_pool) > 0


# ---------------------------------------------------------------------------
# Download and zip-parsing logic
# ---------------------------------------------------------------------------


class TestFLIPDownload:
    """Tests for download caching and zip-parsing behaviour."""

    def test_downloads_when_zip_missing(self, tmp_path):
        """Should call requests.get when splits.zip is not cached."""
        config = _make_flip_config()
        mock_df = _make_mock_df()
        zip_bytes = _make_splits_zip("one_vs_rest.csv", mock_df)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.content = zip_bytes
        mock_response.iter_content.return_value = [zip_bytes]

        data_path = tmp_path / "data" / "FLIP" / "gb1"
        data_path.mkdir(parents=True)

        with (
            patch("alf_tools.datasets.flip.DATAPATH", tmp_path / "data"),
            patch("alf_tools.datasets.flip.requests.get", return_value=mock_response) as mock_get,
        ):
            FLIP(config)

        mock_get.assert_called_once()
        assert "gb1" in mock_get.call_args[0][0]

    def test_no_download_when_zip_exists(self, tmp_path):
        """Should not call requests.get when splits.zip is already cached."""
        config = _make_flip_config()
        mock_df = _make_mock_df()
        zip_bytes = _make_splits_zip("one_vs_rest.csv", mock_df)

        zip_dir = tmp_path / "data" / "FLIP" / "gb1"
        zip_dir.mkdir(parents=True)
        (zip_dir / "splits.zip").write_bytes(zip_bytes)

        with (
            patch("alf_tools.datasets.flip.DATAPATH", tmp_path / "data"),
            patch("alf_tools.datasets.flip.requests.get") as mock_get,
        ):
            FLIP(config)

        mock_get.assert_not_called()

    def test_missing_csv_in_zip_raises(self, tmp_path):
        """FileNotFoundError when the requested split CSV is absent from the zip."""
        config = _make_flip_config()
        mock_df = _make_mock_df()
        # zip contains a different split than requested
        zip_bytes = _make_splits_zip("two_vs_rest.csv", mock_df)

        zip_dir = tmp_path / "data" / "FLIP" / "gb1"
        zip_dir.mkdir(parents=True)
        (zip_dir / "splits.zip").write_bytes(zip_bytes)

        with (
            patch("alf_tools.datasets.flip.DATAPATH", tmp_path / "data"),
            pytest.raises(FileNotFoundError, match="one_vs_rest.csv"),
        ):
            FLIP(config)

    def test_nested_zip_layout_resolved(self, tmp_path):
        """CSV nested in a subdirectory inside the zip should still be found."""
        config = _make_flip_config()
        mock_df = _make_mock_df()
        zip_bytes = _make_splits_zip("gb1/one_vs_rest.csv", mock_df)

        zip_dir = tmp_path / "data" / "FLIP" / "gb1"
        zip_dir.mkdir(parents=True)
        (zip_dir / "splits.zip").write_bytes(zip_bytes)

        with patch("alf_tools.datasets.flip.DATAPATH", tmp_path / "data"):
            instance = FLIP(config)

        assert len(instance.candidate_pool) == 5  # floor(10 * 0.5) = 5 go to train+val, 5 remain
