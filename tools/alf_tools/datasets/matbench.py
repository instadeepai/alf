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

import copy
import logging
from math import floor
from typing import Any, Literal, Self

import numpy as np
from alf_core import BaseDataset, BaseDatasetConfig, Candidate, LabelledCandidates, ProblemType
from alf_core.dataclasses.candidate import Modality
from pydantic import model_validator

logger = logging.getLogger("alf-tools")


class MatbenchConfig(BaseDatasetConfig):
    """Configuration for Matbench benchmark datasets.

    Both composition and structure task inputs are stored under `Modality.TABULAR` —
    Matbench has no data that needs any other modality, so `modality` is fixed and
    should not be overridden. `problem_type` is likewise auto-set from `task_name`
    (`REGRESSION` for Matbench regression tasks, `BINARY` for Matbench classification
    tasks) and should not be set explicitly.

    When `fold_number` is set (0-4), the predefined Matbench train/test split for that
    fold is used: `train_ratio` controls what fraction of the Matbench train set forms
    the initial labelled training set (the remainder becomes `candidate_pool`, same as
    FLIP); `test_ratio` and `split_type` are ignored and the full Matbench test set for
    that fold is used directly. Matbench's predefined folds use a fixed internal seed
    (`18012019`) that cannot be overridden.

    When `fold_number` is `None`, all 5 folds are merged into a single dataset and
    split using the standard ratio-based `train_ratio`/`validation_frac`/`test_ratio`/
    `split_type` — this loses Matbench's benchmark integrity guarantees (results are
    no longer directly comparable to published Matbench leaderboard scores). The fold
    each candidate originally belonged to (0-4) is recorded in `features["fold_id"]`
    regardless of mode, for traceability.

    Attributes:
        task_name: Name of the Matbench task, e.g. `"matbench_steels"`.
        fold_number: `0`-`4` to use a single predefined Matbench fold; `None` to merge
            all 5 folds and split by ratio instead.

    Example::

        # Fold mode
        config = MatbenchConfig(
            name="matbench_mp_e_form_fold0",
            task_name="matbench_mp_e_form",
            fold_number=0,
            seed=42,
            train_ratio=0.1,
            validation_frac=0.1,
            test_ratio=0.2,  # ignored in fold mode
        )

        # Merged mode
        config = MatbenchConfig(
            name="matbench_mp_e_form_merged",
            task_name="matbench_mp_e_form",
            fold_number=None,
            seed=42,
            train_ratio=0.1,
            validation_frac=0.1,
            test_ratio=0.2,
        )
    """

    task_name: str
    fold_number: int | None = None
    split_type: Literal["random", "low_vs_high"] = "random"
    modality: Modality = Modality.TABULAR
    problem_type: ProblemType = ProblemType.REGRESSION  # overwritten in validate_config

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """Override base class validator.

        Looks up `task_name` in Matbench's own task metadata (raising a clear error
        for unknown tasks), validates `fold_number`, and auto-sets `problem_type` from
        the task's Matbench problem type. The base class's `train_ratio + test_ratio
        <= 1` check is intentionally not carried over: in fold mode the two ratios
        apply to separate pools (Matbench train vs. Matbench test), so their sum is
        allowed to exceed 1 — the same reasoning FLIPConfig uses.

        Returns:
            The validated configuration instance.

        Raises:
            ValueError: If `task_name` is not a recognised Matbench task, or if
                `fold_number` is not `None` and not in `0-4`.
        """
        from matbench.metadata import mbv01_metadata  # noqa: PLC0415

        if self.task_name not in mbv01_metadata:
            raise ValueError(
                f"Unknown Matbench task '{self.task_name}'. Valid tasks: "
                f"{sorted(mbv01_metadata.keys())}"
            )
        if self.fold_number is not None and not (0 <= self.fold_number <= 4):
            raise ValueError(f"fold_number must be None or in 0-4, got {self.fold_number}")

        task_type = mbv01_metadata[self.task_name].task_type
        self.problem_type = (
            ProblemType.REGRESSION if task_type == "regression" else ProblemType.BINARY
        )
        return self


class Matbench(BaseDataset):
    """Matbench benchmark dataset class.

    Matbench provides 13 materials-property prediction tasks, each with predefined
    5-fold cross-validation splits, covering both composition-based and
    structure-based inputs. Composition and structure inputs (pymatgen `Composition`
    and `Structure` objects respectively) are both MSONable, so both are serialised
    identically via `.to_json()` into a JSON string stored in `Candidate.data`; no
    `alf_core` changes are needed. `pymatgen`/`matbench` are lazily imported so that
    `alf_tools` can be used without the `matbench` extras installed.
    """

    config: MatbenchConfig  # narrows the inherited BaseDatasetConfig type

    def __init__(self, config: MatbenchConfig):
        """Initialise Matbench dataset."""
        super().__init__(config)
        self.setup()

    def __repr__(self) -> str:
        """Return a string representation identifying task and fold.

        Returns:
            A string representation of the dataset.
        """
        return (
            f"Matbench(name={self.config.name}, modality={self.modality}, "
            f"seed={self.config.seed}, "
            f"task_name={self.config.task_name}, "
            f"fold_number={self.config.fold_number})"
        )

    def _load_task(self) -> Any:
        """Load (downloading/caching as needed) the MatbenchTask for this config.

        Returns:
            The loaded MatbenchTask for `self.config.task_name`.
        """
        from matbench.bench import MatbenchBenchmark  # noqa: PLC0415

        benchmark = MatbenchBenchmark(autoload=False, subset=[self.config.task_name])
        task = next(iter(benchmark.tasks))
        task.load()
        return task

    def load_dataset(self) -> LabelledCandidates:
        """Load Matbench data via the Matbench API.

        In fold mode, only the configured fold's predefined train/test rows are
        loaded (each candidate tagged with a `matbench_split` feature of "train" or
        "test" for use by `_split_dataset`). In merged mode, all rows across all 5
        folds are loaded. In both modes, every candidate's `features["fold_id"]`
        records which Matbench fold it belongs to.

        Returns:
            LabelledCandidates with JSON-string composition/structure data and
            float labels.
        """
        task = self._load_task()
        is_classification = task.metadata.task_type == "classification"

        candidates: list[Candidate] = []
        labels: list[float] = []

        def _add(value: Any, target: Any, fold_id: int, extra_features: dict) -> None:
            # Composition inputs are plain chemical-formula strings (e.g. "Fe0.62C0.01..."),
            # used as-is; structure inputs are pymatgen Structure objects, serialised via
            # their MSONable .to_json() into an equivalent JSON string.
            data = value if isinstance(value, str) else value.to_json()
            candidates.append(
                Candidate(
                    data=data,
                    modality=Modality.TABULAR,
                    features={"fold_id": fold_id, **extra_features},
                )
            )
            labels.append(float(target) if is_classification else target)

        if self.config.fold_number is not None:
            fold = self.config.fold_number
            train_inputs, train_targets = task.get_train_and_val_data(fold, as_type="tuple")
            test_inputs, test_targets = task.get_test_data(
                fold, as_type="tuple", include_target=True
            )
            for value, target in zip(train_inputs, train_targets):
                _add(value, target, fold, {"matbench_split": "train"})
            for value, target in zip(test_inputs, test_targets):
                _add(value, target, fold, {"matbench_split": "test"})
        else:
            fold_id_by_index: dict[Any, int] = {}
            for fold in task.folds:
                fold_key = task.folds_map[fold]
                for row_id in task.validation[fold_key].test:
                    fold_id_by_index[row_id] = fold
            for row_id, row in task.df.iterrows():
                value = row[task.metadata.input_type]
                target = row[task.metadata.target]
                _add(value, target, fold_id_by_index[row_id], {})

        return LabelledCandidates(candidates=candidates, labels=np.array(labels))

    def _split_dataset(self) -> dict[str, LabelledCandidates]:
        """Split the raw dataset, dispatching on fold mode.

        Returns:
            Dictionary with keys "train", "validation", "test", and "candidate_pool".
        """
        if self.config.fold_number is not None:
            return self._split_fold_mode()
        return super()._split_dataset()

    def _split_fold_mode(self) -> dict[str, LabelledCandidates]:
        """Split using the predefined Matbench train/test pools for the configured fold.

        The Matbench train pool is shuffled, then partitioned using `train_ratio` and
        `validation_frac`; the remainder becomes the candidate pool (capped at
        `max_candidate_pool`). The full Matbench test pool is used directly as "test"
        (`test_ratio` is ignored — Matbench's predefined test set must be used as-is
        for benchmark-comparable results).

        Returns:
            Dictionary with keys "train", "validation", "test", and "candidate_pool".

        Raises:
            RuntimeError: If dataset has not been loaded yet.
        """
        if self._raw_dataset is None:
            raise RuntimeError(
                "Dataset must be loaded before splitting; "
                "_raw_dataset is None — call dataset.setup() (or load_dataset()) before splitting"
            )

        matbench_train = LabelledCandidates(candidates=[], labels=np.array([]))
        matbench_test = LabelledCandidates(candidates=[], labels=np.array([]))
        for candidate, label in self._raw_dataset:
            if candidate.features and candidate.features["matbench_split"] == "train":
                matbench_train.append([candidate], np.array([label]))
            else:
                matbench_test.append([candidate], np.array([label]))

        matbench_train = matbench_train.shuffle(self.config.seed)

        train_plus_val_size = floor(len(matbench_train) * self.split_ratio["train"])
        validation_size = floor(train_plus_val_size * self.split_ratio["validation_frac"])
        train_size = train_plus_val_size - validation_size

        train = LabelledCandidates(
            candidates=matbench_train.candidates[:train_size],
            labels=matbench_train.labels[:train_size],
        )
        validation = LabelledCandidates(
            candidates=matbench_train.candidates[train_size:train_plus_val_size],
            labels=matbench_train.labels[train_size:train_plus_val_size],
        )

        pool_candidates = matbench_train.candidates[train_plus_val_size:]
        pool_labels = matbench_train.labels[train_plus_val_size:]
        if self.config.max_candidate_pool is not None:
            pool_candidates = pool_candidates[: self.config.max_candidate_pool]
            pool_labels = pool_labels[: self.config.max_candidate_pool]
        candidate_pool = LabelledCandidates(candidates=pool_candidates, labels=pool_labels)

        logger.debug(
            "Split sizes — train: %d, validation: %d, candidate_pool: %d, test: %d",
            len(train),
            len(validation),
            len(candidate_pool),
            len(matbench_test),
        )

        self.init_candidate_pool = copy.deepcopy(candidate_pool)

        return {
            "train": train,
            "validation": validation,
            "test": matbench_test,
            "candidate_pool": candidate_pool,
        }
