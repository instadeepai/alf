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

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matbench", reason="matbench not installed; install alf_tools[matbench]")
pytest.importorskip("pymatgen", reason="pymatgen not installed; install alf_tools[matbench]")

from alf_core import Modality, ProblemType  # noqa: E402
from alf_tools.datasets.matbench import Matbench, MatbenchConfig  # noqa: E402
from monty.json import MontyDecoder  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from pymatgen.core import Composition, Lattice, Structure  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _composition(n_fe: int) -> Composition:
    """Build a distinct, valid pymatgen Composition.

    Returns:
        A Composition with a varying Fe count so each call is distinct.
    """
    return Composition({"Fe": n_fe + 1, "O": 1})


def _structure(scale: float) -> Structure:
    """Build a distinct, valid pymatgen Structure.

    Returns:
        A simple two-site cubic Structure with a varying lattice constant.
    """
    lattice = Lattice.cubic(3.0 + scale)
    return Structure(lattice, ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _make_fake_fold_task(
    input_type: str = "composition",
    task_type: str = "regression",
    n_train: int = 10,
    n_test: int = 4,
) -> SimpleNamespace:
    """Build a fake MatbenchTask exposing only the fold-mode API surface used by Matbench.

    Train targets are 0..n_train-1 (regression) or alternate True/False (classification).
    Test targets are 100..100+n_test-1 (regression) or alternate True/False (classification).

    Returns:
        A SimpleNamespace standing in for a real MatbenchTask.
    """
    make_value = _composition if input_type == "composition" else _structure

    def _targets(n: int, offset: int) -> list:
        if task_type == "classification":
            return [bool(i % 2) for i in range(n)]
        return [float(offset + i) for i in range(n)]

    train_inputs = pd.Series([make_value(i) for i in range(n_train)])
    train_targets = pd.Series(_targets(n_train, 0))
    test_inputs = pd.Series([make_value(100 + i) for i in range(n_test)])
    test_targets = pd.Series(_targets(n_test, 100))

    def get_train_and_val_data(fold_number, as_type="tuple"):
        return train_inputs, train_targets

    def get_test_data(fold_number, as_type="tuple", include_target=False):
        return test_inputs, test_targets

    return SimpleNamespace(
        metadata=SimpleNamespace(input_type=input_type, target="target", task_type=task_type),
        folds=[0, 1, 2, 3, 4],
        get_train_and_val_data=get_train_and_val_data,
        get_test_data=get_test_data,
    )


def _make_fake_merged_task(
    input_type: str = "composition",
    task_type: str = "regression",
    n_per_fold: int = 2,
    n_folds: int = 5,
) -> SimpleNamespace:
    """Build a fake MatbenchTask exposing only the merged-mode API surface used by Matbench.

    Returns:
        A SimpleNamespace standing in for a real MatbenchTask, with `.df` populated
        and `.validation` fold membership assigned so each row belongs to exactly
        one fold's test set.
    """
    make_value = _composition if input_type == "composition" else _structure

    n_total = n_per_fold * n_folds
    ids = [f"mb-id-{i:03d}" for i in range(n_total)]
    if task_type == "classification":
        targets = [bool(i % 2) for i in range(n_total)]
    else:
        targets = [float(i) for i in range(n_total)]
    df = pd.DataFrame(
        {input_type: [make_value(i) for i in range(n_total)], "target": targets},
        index=ids,
    )

    folds_map = {f: f"fold_{f}" for f in range(n_folds)}
    validation = {}
    for f in range(n_folds):
        test_ids = ids[f * n_per_fold : (f + 1) * n_per_fold]
        train_ids = [i for i in ids if i not in test_ids]
        validation[folds_map[f]] = SimpleNamespace(train=train_ids, test=test_ids)

    return SimpleNamespace(
        metadata=SimpleNamespace(input_type=input_type, target="target", task_type=task_type),
        folds=list(range(n_folds)),
        folds_map=folds_map,
        validation=validation,
        df=df,
    )


def _make_matbench_config(**overrides) -> MatbenchConfig:
    defaults = dict(
        name="matbench_test",
        task_name="matbench_steels",
        fold_number=0,
        seed=42,
        train_ratio=0.5,
        validation_frac=0.2,
        test_ratio=0.2,
    )
    defaults.update(overrides)
    return MatbenchConfig(**defaults)


def _make_matbench_instance(fake_task: SimpleNamespace, **config_overrides) -> Matbench:
    """Create a Matbench instance with a mocked _load_task.

    Returns:
        Initialised Matbench dataset backed by the provided fake task.
    """
    config = _make_matbench_config(**config_overrides)
    with patch.object(Matbench, "_load_task", return_value=fake_task):
        instance = Matbench(config)
    return instance


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestMatbenchConfig:
    """Tests for MatbenchConfig validation."""

    def test_unknown_task_raises(self):
        """An unrecognised task_name should be rejected with a helpful error."""
        with pytest.raises(ValidationError, match="Unknown Matbench task"):
            _make_matbench_config(task_name="matbench_bogus")

    def test_invalid_fold_number_raises(self):
        """fold_number outside 0-4 should be rejected."""
        with pytest.raises(ValidationError, match="fold_number"):
            _make_matbench_config(fold_number=5)

    def test_negative_fold_number_raises(self):
        """Negative fold_number should be rejected."""
        with pytest.raises(ValidationError, match="fold_number"):
            _make_matbench_config(fold_number=-1)

    def test_fold_number_none_is_valid(self):
        """fold_number=None (merged mode) should be accepted."""
        config = _make_matbench_config(fold_number=None)
        assert config.fold_number is None

    def test_modality_defaults_to_tabular(self):
        """Modality should default to TABULAR regardless of task input type."""
        config = _make_matbench_config(task_name="matbench_mp_e_form")  # structure task
        assert config.modality == Modality.TABULAR
        config = _make_matbench_config(task_name="matbench_steels")  # composition task
        assert config.modality == Modality.TABULAR

    def test_problem_type_auto_set_regression(self):
        """Regression tasks should auto-set problem_type to REGRESSION."""
        config = _make_matbench_config(task_name="matbench_steels")
        assert config.problem_type == ProblemType.REGRESSION

    def test_problem_type_auto_set_classification(self):
        """Classification tasks should auto-set problem_type to BINARY."""
        config = _make_matbench_config(task_name="matbench_glass")
        assert config.problem_type == ProblemType.BINARY

    def test_ratio_sum_exceeding_one_is_allowed(self):
        """train_ratio + test_ratio > 1 should not raise (ratios apply to separate pools)."""
        config = _make_matbench_config(train_ratio=0.9, test_ratio=0.9)
        assert config.train_ratio == 0.9
        assert config.test_ratio == 0.9


# ---------------------------------------------------------------------------
# Fold mode
# ---------------------------------------------------------------------------


class TestMatbenchFoldMode:
    """Tests for fold_number-based (predefined split) loading and splitting."""

    def test_split_sizes(self):
        """train_ratio=0.5, validation_frac=0.2 on 10 matbench-train rows.

        train_plus_val = floor(10*0.5) = 5; validation = floor(5*0.2) = 1; train = 4;
        candidate_pool = 5; test = all 4 matbench-test rows (test_ratio ignored).
        """
        instance = _make_matbench_instance(_make_fake_fold_task())
        assert len(instance.train_dataset) == 4
        assert len(instance.validation_dataset) == 1
        assert len(instance.candidate_pool) == 5
        assert len(instance.test_dataset) == 4

    def test_test_ratio_ignored(self):
        """test_ratio should have no effect in fold mode — full matbench test set is used."""
        instance = _make_matbench_instance(_make_fake_fold_task(), test_ratio=0.01)
        assert len(instance.test_dataset) == 4

    def test_max_candidate_pool_respected(self):
        """max_candidate_pool should cap the candidate pool size."""
        instance = _make_matbench_instance(_make_fake_fold_task(), max_candidate_pool=2)
        assert len(instance.candidate_pool) == 2

    def test_no_leakage_between_train_and_test(self):
        """No candidate should appear in both a train-derived split and test."""
        instance = _make_matbench_instance(_make_fake_fold_task())
        train_data = {c.data for c in instance.train_dataset.candidates}
        pool_data = {c.data for c in instance.candidate_pool.candidates}
        val_data = {c.data for c in instance.validation_dataset.candidates}
        test_data = {c.data for c in instance.test_dataset.candidates}
        assert train_data.isdisjoint(test_data)
        assert pool_data.isdisjoint(test_data)
        assert val_data.isdisjoint(test_data)

    def test_fold_id_stored_on_all_candidates(self):
        """Every candidate should carry fold_id equal to the configured fold_number."""
        instance = _make_matbench_instance(_make_fake_fold_task(), fold_number=2)
        for split in ("train_dataset", "validation_dataset", "test_dataset", "candidate_pool"):
            for candidate in getattr(instance, split).candidates:
                assert candidate.features["fold_id"] == 2

    def test_candidates_are_tabular_modality(self):
        """All candidates should carry Modality.TABULAR regardless of input_type."""
        for input_type in ("composition", "structure"):
            instance = _make_matbench_instance(_make_fake_fold_task(input_type=input_type))
            for candidate in instance._raw_dataset.candidates:
                assert candidate.modality == Modality.TABULAR

    def test_composition_serialised_as_json_string(self):
        """Composition inputs should be stored as JSON strings, round-trippable via MontyDecoder."""
        instance = _make_matbench_instance(_make_fake_fold_task(input_type="composition"))
        candidate = instance._raw_dataset.candidates[0]
        assert isinstance(candidate.data, str)
        restored = json.loads(candidate.data, cls=MontyDecoder)
        assert isinstance(restored, Composition)

    def test_structure_serialised_as_json_string(self):
        """Structure inputs should be stored as JSON strings, round-trippable via MontyDecoder."""
        instance = _make_matbench_instance(_make_fake_fold_task(input_type="structure"))
        candidate = instance._raw_dataset.candidates[0]
        assert isinstance(candidate.data, str)
        restored = json.loads(candidate.data, cls=MontyDecoder)
        assert isinstance(restored, Structure)

    def test_classification_labels_cast_to_float(self):
        """Boolean classification labels should be cast to float 0.0/1.0."""
        instance = _make_matbench_instance(
            _make_fake_fold_task(task_type="classification"), fold_number=0
        )
        assert instance._raw_dataset.labels.dtype != bool
        assert set(np.unique(instance._raw_dataset.labels)) <= {0.0, 1.0}

    def test_classification_num_classes_is_two(self):
        """BINARY problem_type with two label classes should report num_classes=2."""
        instance = _make_matbench_instance(
            _make_fake_fold_task(task_type="classification"),
            task_name="matbench_glass",
            fold_number=0,
        )
        assert instance.num_classes == 2

    def test_regression_labels_are_float(self):
        """Regression targets should remain float and match expected pool/test means."""
        instance = _make_matbench_instance(_make_fake_fold_task())
        # train_ratio=0 puts all matbench-train (labels 0..9) into candidate_pool
        full_pool_instance = _make_matbench_instance(
            _make_fake_fold_task(), train_ratio=0.0, validation_frac=0.0
        )
        assert np.mean(full_pool_instance.candidate_pool.labels) == pytest.approx(4.5)
        assert np.mean(instance.test_dataset.labels) == pytest.approx(101.5)


# ---------------------------------------------------------------------------
# Merged mode (fold_number=None)
# ---------------------------------------------------------------------------


class TestMatbenchMergedMode:
    """Tests for fold_number=None (all folds merged, ratio-based splitting)."""

    def test_all_rows_present(self):
        """Raw dataset should contain all rows across all folds."""
        instance = _make_matbench_instance(
            _make_fake_merged_task(n_per_fold=2, n_folds=5), fold_number=None
        )
        assert len(instance._raw_dataset) == 10

    def test_fold_id_covers_all_folds(self):
        """fold_id feature should take every value in {0,...,4} across candidates."""
        instance = _make_matbench_instance(
            _make_fake_merged_task(n_per_fold=2, n_folds=5), fold_number=None
        )
        fold_ids = {c.features["fold_id"] for c in instance._raw_dataset.candidates}
        assert fold_ids == {0, 1, 2, 3, 4}

    def test_ratio_based_split_applies(self):
        """Standard train/validation/test/candidate_pool ratios should apply."""
        instance = _make_matbench_instance(
            _make_fake_merged_task(n_per_fold=2, n_folds=5),
            fold_number=None,
            train_ratio=0.5,
            validation_frac=0.2,
            test_ratio=0.3,
        )
        # dataset_size=10: train_plus_val=floor(10*0.5)=5, val=floor(5*0.2)=1, train=4
        # test=floor(10*0.3)=3, candidate_pool=10-5-3=2
        assert len(instance.train_dataset) == 4
        assert len(instance.validation_dataset) == 1
        assert len(instance.test_dataset) == 3
        assert len(instance.candidate_pool) == 2

    def test_candidates_are_tabular_modality(self):
        """Merged-mode candidates should also carry Modality.TABULAR."""
        instance = _make_matbench_instance(
            _make_fake_merged_task(input_type="structure", n_per_fold=2, n_folds=5),
            fold_number=None,
        )
        for candidate in instance._raw_dataset.candidates:
            assert candidate.modality == Modality.TABULAR

    def test_classification_labels_cast_to_float(self):
        """Merged-mode boolean labels should be cast to float 0.0/1.0."""
        instance = _make_matbench_instance(
            _make_fake_merged_task(task_type="classification", n_per_fold=2, n_folds=5),
            task_name="matbench_glass",
            fold_number=None,
        )
        assert instance._raw_dataset.labels.dtype != bool
        assert set(np.unique(instance._raw_dataset.labels)) <= {0.0, 1.0}


# ---------------------------------------------------------------------------
# Initialisation / repr
# ---------------------------------------------------------------------------


class TestMatbenchInitialisation:
    """Tests for post-construction state of a Matbench instance."""

    def test_config_stored(self):
        """Dataset name and task_name should be accessible via config."""
        instance = _make_matbench_instance(_make_fake_fold_task())
        assert instance.config.name == "matbench_test"
        assert instance.config.task_name == "matbench_steels"

    def test_modality(self):
        """Modality should be TABULAR."""
        instance = _make_matbench_instance(_make_fake_fold_task())
        assert instance.modality == Modality.TABULAR

    def test_repr(self):
        """Repr should include task_name and fold_number for quick identification."""
        instance = _make_matbench_instance(_make_fake_fold_task(), fold_number=3)
        r = repr(instance)
        assert "matbench_steels" in r
        assert "3" in r
