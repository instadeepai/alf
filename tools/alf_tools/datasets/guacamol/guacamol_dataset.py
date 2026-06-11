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

import copy
import logging
from pathlib import Path
from typing import Callable, Literal, cast, get_args

import numpy as np
from alf_core import (
    BaseDataset,
    BaseDatasetConfig,
    Candidate,
    LabelledCandidates,
    ProblemType,
)
from pydantic import Field, computed_field, model_validator
from rdkit import Chem

from .guacamol_scoring import get_task_scorer
from .guacamol_utils import (
    ALL_PROPERTIES,
    DATAPATH,
    GUACAMOL_FILES,
    PROPERTY_FNS,
    GuacaMolPropertyName,
    GuacaMolTaskName,
    _compute_properties,
    _download_file,
    _label_smiles,
    _label_smiles_benchmark,
    _load_smiles_file,
    _mol_from_smiles,
)

logger = logging.getLogger("alf-tools")


class GuacaMolConfig(BaseDatasetConfig):
    """Configuration for GuacaMol dataset.

    Attributes:
        target_property: Property or task name used as labels in LabelledCandidates.
        task_type: Always auto-derived from target_property in the model validator.
            Any value supplied at construction is silently overwritten. Do not set.
        computed_properties: RDKit properties computed and stored in Candidate.features.
            None defaults to computing only [target_property]. Pass
            `list(_ALL_PROPERTIES_ORDERED)` to compute all 10. Only applies when
            task_type == "property".
        max_molecules: Cap on SMILES lines written to disk and loaded per file. None = full corpus.
        split_mode: "random" and "low_vs_high" use BaseDataset splitting on the combined
            corpus file. "paper" uses the original train/valid/test figshare file boundaries.
            Replaces BaseDatasetConfig.split_type — do not set split_type directly.
        data_dir: Directory where SMILES files are cached. Defaults to the package data dir.
    """

    problem_type: ProblemType = ProblemType.REGRESSION
    target_property: GuacaMolPropertyName | GuacaMolTaskName
    computed_properties: list[GuacaMolPropertyName] | None = None
    max_molecules: int | None = Field(default=None, ge=1)
    split_mode: Literal["random", "low_vs_high", "stratified", "paper"] = "random"
    data_dir: Path = DATAPATH

    @computed_field  # type: ignore[prop-decorator]
    @property
    def task_type(self) -> Literal["property", "benchmark_task"]:
        """Discriminates the two modes of use in the GuacaMol benchmark corpus.

        `"property"` — the label is a continuous RDKit physicochemical value (e.g.
        `MolLogP`, `TPSA`, `QED`) computed molecule-by-molecule via
        `PROPERTY_FNS`.  Molecules are loaded from the corpus and queried by
        canonical SMILES lookup; novel SMILES not in the corpus are scored on the fly.
        Each `Candidate` carries the requested properties in its `features` dict.

        `"benchmark_task"` — the label is a score in [0, 1] produced by one of the
        19 goal-directed scoring functions from Brown et al. (2019).  Scores combine
        Tanimoto fingerprint similarity to reference drug molecules, multi-property
        optimisation objectives (TPSA, logP, ring counts, …), pharmacophoric matching,
        or molecular formula isomer matching — each designed to capture a realistic
        drug-design challenge.  There is no corpus lookup; every SMILES is re-scored
        by the task function.  `Candidate.features` is always empty for benchmark
        tasks.
        """
        return "property" if self.target_property in ALL_PROPERTIES else "benchmark_task"

    @model_validator(mode="after")
    def _validate_and_sync(self) -> "GuacaMolConfig":
        if (
            self.target_property in ALL_PROPERTIES
            and self.computed_properties is not None
            and self.target_property not in self.computed_properties
        ):
            raise ValueError(
                f"target_property '{self.target_property}' must be present in "
                "computed_properties when computed_properties is explicitly set."
            )
        if self.split_mode != "paper":
            # split_mode values "random", "low_vs_high", "stratified" map 1-to-1 to SplitType.
            from alf_core.dataset.splitting_utils import SplitType  # noqa: PLC0415

            if self.split_mode not in get_args(SplitType):
                raise ValueError(
                    f"split_mode {self.split_mode!r} is not a valid SplitType value. "
                    f"Valid values: {get_args(SplitType)}"
                )
            self.split_type = cast(SplitType, self.split_mode)
        return self


class GuacaMol(BaseDataset):
    """GuacaMol dataset for physicochemical property prediction on drug-like molecules.

    Wraps the GuacaMol benchmark corpus (1.6 M SMILES from ChEMBL) and computes
    RDKit physicochemical properties (e.g. MolLogP, TPSA, QED) as regression targets.

    Three split modes are supported:
    - `"random"` / `"low_vs_high"`: BaseDataset splitting on the combined corpus.
    - `"paper"`: uses the original train/valid/test file boundaries from the
      GuacaMol paper (Brown et al., 2019), allowing direct comparison with published results.

    Novel SMILES not present in the corpus can be queried on-the-fly via :meth:`query`;
    their properties are computed directly with RDKit.
    """

    config: GuacaMolConfig  # narrows BaseDataset.config for static analysis

    def __init__(self, config: GuacaMolConfig) -> None:
        """Initialize the GuacaMol dataset.

        Args:
            config: Configuration for the GuacaMol dataset.
        """
        self._paper_splits: dict[str, LabelledCandidates] | None = None
        self._smiles_index: dict[str, float] = {}
        self._prop_matrix: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self._prop_cols: list[str] = []
        super().__init__(config)
        self.setup()

    def setup(self) -> None:
        """Set up the dataset and rebuild the SMILES lookup index."""
        super().setup()
        self._smiles_index = (
            {
                c.data: float(label)
                for c, label in zip(self._raw_dataset.candidates, self._raw_dataset.labels)
            }
            if self._raw_dataset is not None
            else {}
        )

    def __repr__(self) -> str:
        """Return a string representation identifying dataset and target."""
        return (
            f"GuacaMol(name={self.config.name}, modality={self.modality}, "
            f"seed={self.config.seed}, "
            f"target_property={self.config.target_property}, "
            f"split_mode={self.config.split_mode})"
        )

    def load_dataset(self) -> LabelledCandidates:
        """Load GuacaMol SMILES and compute labels via RDKit.

        Returns:
            LabelledCandidates with SMILES candidates and 1D labels.

        Raises:
            FileNotFoundError: If the corpus cannot be downloaded.
        """
        if self.config.task_type == "benchmark_task":
            return self._load_benchmark_task()
        if self.config.split_mode == "paper":
            return self._load_paper_splits()
        return self._load_single_file()

    def _load_single_file(self) -> LabelledCandidates:
        """Download (if absent) and label the combined corpus file.

        Returns:
            LabelledCandidates built from the combined corpus.
        """
        entry_info_all = GUACAMOL_FILES["ALL"]
        filepath = _download_file(
            entry_info_all["url"],
            self.config.data_dir / entry_info_all["name"],
            self.config.max_molecules,
            sha256=entry_info_all.get("sha256"),
        )
        smiles_list = _load_smiles_file(filepath)
        if self.config.max_molecules is not None:
            smiles_list = smiles_list[: self.config.max_molecules]
        target = cast(GuacaMolPropertyName, self.config.target_property)
        properties = list(self.config.computed_properties or [target])
        lc, prop_matrix = _label_smiles(smiles_list, properties, target, self.modality)
        self._prop_matrix = prop_matrix
        self._prop_cols = properties
        return lc

    def _load_paper_splits_with(
        self, label_fn: Callable[[list[str]], LabelledCandidates]
    ) -> LabelledCandidates:
        """Download (if absent) train/valid/test files and label all candidates.

        Stores the three splits in `self._paper_splits` keyed by
        `"train"`, `"validation"`, and `"test"`. Returns a combined
        LabelledCandidates for use as `_raw_dataset`.

        Args:
            label_fn: Callable mapping a list of SMILES to LabelledCandidates.

        Returns:
            Combined LabelledCandidates across all three paper splits.
        """
        split_files = {k: v for k, v in GUACAMOL_FILES.items() if k != "ALL"}
        tag_to_key = {"TRAIN": "train", "VALID": "validation", "TEST": "test"}
        if self.config.max_molecules is not None:
            logger.warning(
                "max_molecules=%d is applied per split file in paper mode — "
                "total molecules may reach %d × 3.",
                self.config.max_molecules,
                self.config.max_molecules,
            )
        self._paper_splits = {}
        all_candidates: list[Candidate] = []
        all_labels: list[float] = []
        for tag, entry_info in split_files.items():
            filepath = _download_file(
                entry_info["url"],
                self.config.data_dir / entry_info["name"],
                self.config.max_molecules,
                sha256=entry_info.get("sha256"),
            )
            smiles_list = _load_smiles_file(filepath)
            if self.config.max_molecules is not None:
                smiles_list = smiles_list[: self.config.max_molecules]
            split_lc = label_fn(smiles_list)
            logger.debug(
                "Paper split '%s': %d SMILES → %d valid candidates",
                tag,
                len(smiles_list),
                len(split_lc.candidates),
            )
            self._paper_splits[tag_to_key[tag]] = split_lc
            all_candidates.extend(split_lc.candidates)
            all_labels.extend(split_lc.labels.tolist())
        return LabelledCandidates(
            candidates=all_candidates, labels=np.array(all_labels, dtype=float)
        )

    def _load_paper_splits(self) -> LabelledCandidates:
        """Download (if absent) train/valid/test files and label all candidates.

        Stores the three splits in `self._paper_splits` keyed by
        `"train"`, `"validation"`, and `"test"`. Returns a combined
        LabelledCandidates (without any split tag in features) for use as
        `_raw_dataset` — this powers the SMILES lookup index in :meth:`query`.

        Returns:
            Combined LabelledCandidates across all three paper splits.
        """
        target = cast(GuacaMolPropertyName, self.config.target_property)
        properties = list(self.config.computed_properties or [target])
        return self._load_paper_splits_with(
            lambda smiles_list: _label_smiles(smiles_list, properties, target, self.modality)
        )

    def _load_benchmark_task(self) -> LabelledCandidates:
        """Load corpus and score each valid SMILES using the benchmark task scorer.

        Returns:
            LabelledCandidates: Scored candidates from the corpus.
        """
        scorer = get_task_scorer(cast(GuacaMolTaskName, self.config.target_property))
        if self.config.split_mode == "paper":
            return self._load_paper_splits_benchmark(scorer)
        return self._load_single_file_benchmark(scorer)

    def _load_single_file_benchmark(self, scorer: Callable[[str], float]) -> LabelledCandidates:
        """Download (if absent) and score the combined corpus file.

        Returns:
            LabelledCandidates: Scored candidates from the corpus.
        """
        entry_info_all = GUACAMOL_FILES["ALL"]
        filepath = _download_file(
            entry_info_all["url"],
            self.config.data_dir / entry_info_all["name"],
            self.config.max_molecules,
            sha256=entry_info_all.get("sha256"),
        )
        smiles_list = _load_smiles_file(filepath)
        if self.config.max_molecules is not None:
            smiles_list = smiles_list[: self.config.max_molecules]
        return _label_smiles_benchmark(smiles_list, scorer, self.modality)

    def _load_paper_splits_benchmark(self, scorer: Callable[[str], float]) -> LabelledCandidates:
        """Download (if absent) train/valid/test files and score all candidates.

        Stores the three splits in `self._paper_splits`.

        Returns:
            LabelledCandidates: All scored candidates across train/valid/test splits.
        """
        return self._load_paper_splits_with(
            lambda smiles_list: _label_smiles_benchmark(smiles_list, scorer, self.modality)
        )

    def _query_benchmark(self, candidates: list[Candidate]) -> LabelledCandidates:
        """Score candidates using the benchmark task scorer.

        Args:
            candidates: Candidates to score. Each must have a parseable SMILES in `.data`.

        Returns:
            LabelledCandidates with scores in [0, 1].

        Raises:
            ValueError: If a candidate's SMILES string is invalid.
        """
        scorer = get_task_scorer(cast(GuacaMolTaskName, self.config.target_property))
        result_labels: list[float] = []
        for candidate in candidates:
            if _mol_from_smiles(candidate.data) is None:
                raise ValueError(f"Cannot compute label for invalid SMILES: {candidate.data!r}")
            result_labels.append(scorer(candidate.data))
        return LabelledCandidates(
            candidates=candidates,
            labels=np.array(result_labels, dtype=float),
        )

    def query(self, candidates: list[Candidate]) -> LabelledCandidates:
        """Return labels for candidates, computing via RDKit for SMILES not in the corpus.

        Args:
            candidates: Candidates to label. May include SMILES not present in _raw_dataset.

        Returns:
            LabelledCandidates with 1D labels of shape (N,). The returned candidates are
            always the caller's input objects — corpus lookup provides the label only.

        Raises:
            ValueError: If a novel candidate's SMILES string is invalid.
            RuntimeError: If the dataset is not loaded before querying.
        """
        if self.config.task_type == "benchmark_task":
            return self._query_benchmark(candidates)
        if self._raw_dataset is None:
            raise RuntimeError("Dataset must be loaded before querying")  # pragma: no cover

        result_candidates: list[Candidate] = []
        result_labels: list[float] = []

        for candidate in candidates:
            mol = _mol_from_smiles(candidate.data)
            key = Chem.MolToSmiles(mol) if mol is not None else candidate.data
            if key in self._smiles_index:
                result_labels.append(self._smiles_index[key])
                result_candidates.append(candidate)
            else:
                if mol is None:
                    raise ValueError(f"Cannot compute label for invalid SMILES: {candidate.data!r}")
                label_val = PROPERTY_FNS[self.config.target_property](mol)
                out_candidate = candidate
                if not candidate.features:
                    props_to_compute: list[GuacaMolPropertyName] = list(
                        self.config.computed_properties
                        or [cast(GuacaMolPropertyName, self.config.target_property)]
                    )
                    out_candidate = Candidate(
                        data=candidate.data,
                        modality=candidate.modality,
                        features=dict(_compute_properties(candidate.data, props_to_compute) or {}),
                    )
                result_labels.append(label_val)
                result_candidates.append(out_candidate)

        return LabelledCandidates(
            candidates=result_candidates,
            labels=np.array(result_labels, dtype=float),
        )

    def _split_dataset(self) -> dict[str, LabelledCandidates]:
        """Split by pre-built paper splits when split_mode is 'paper'; else use base class.

        Raises:
            RuntimeError: If the raw dataset is None, indicating it was not initialized properly.

        Returns:
            dict[str, LabelledCandidates]: Dict with keys "train", "validation", "test",
            and "candidate_pool".
        """
        if self.config.split_mode != "paper":
            return super()._split_dataset()

        if self._paper_splits is None:
            raise RuntimeError("Dataset must be loaded before splitting")  # pragma: no cover

        splits = {k: copy.deepcopy(v) for k, v in self._paper_splits.items()}
        splits["candidate_pool"] = LabelledCandidates(
            candidates=[], labels=np.array([], dtype=float)
        )
        self.init_candidate_pool = copy.deepcopy(splits["candidate_pool"])
        return splits
