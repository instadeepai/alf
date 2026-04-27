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
from typing import Literal, get_args

import numpy as np
import requests
from alf_core import BaseDataset, Candidate, LabelledCandidates
from alf_core.dataset.base_dataset import BaseDatasetConfig
from pydantic import model_validator

logger = logging.getLogger("alf-tools")

DATAPATH = Path(__file__).parent / "data"
FILENAME_ALL = "guacamol_v1_all.smiles"
FILENAME_TRAIN = "guacamol_v1_train.smiles"
FILENAME_VALID = "guacamol_v1_valid.smiles"
FILENAME_TEST = "guacamol_v1_test.smiles"

URL_ALL = "https://figshare.com/ndownloader/files/13612745"
URL_TRAIN = "https://figshare.com/ndownloader/files/13612760"
URL_VALID = "https://figshare.com/ndownloader/files/13612766"
URL_TEST = "https://figshare.com/ndownloader/files/13612757"

GuacaMolPropertyName = Literal[
    "BertzCT", "MolLogP", "MolWt", "TPSA",
    "NumHAcceptors", "NumHDonors", "NumRotatableBonds",
    "NumAliphaticRings", "NumAromaticRings", "QED",
]
GuacaMolTaskName = Literal[
    "celecoxib_rediscovery", "troglitazone_rediscovery", "thiothixene_rediscovery",
    "aripiprazole_similarity", "albuterol_similarity", "mestranol_similarity",
    "camphor_menthol_median", "tadalafil_sildenafil_median",
    "fexofenadine_mpo", "osimertinib_mpo", "ranolazine_mpo",
    "perindopril_mpo", "amlodipine_mpo", "sitagliptin_mpo", "zaleplon_mpo",
    "c7h8n2o2_isomer", "c9h10n2o2pf2cl_isomer",
    "aripiprazole_scaffold_hop", "aripiprazole_decorator_hop",
]

ALL_PROPERTIES: frozenset[str] = frozenset(get_args(GuacaMolPropertyName))
ALL_TASKS: frozenset[str] = frozenset(get_args(GuacaMolTaskName))

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, GraphDescriptors, rdMolDescriptors
    from rdkit.Chem import QED as RDKitQED
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False


def _require_rdkit() -> None:
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for the GuacaMol dataset. "
            "Install it with: pip install 'alf_tools[benchmarks]'"
        )


class GuacaMolConfig(BaseDatasetConfig):
    """Configuration for GuacaMol dataset.

    Attributes:
        target_property: Property or task name used as labels in LabelledCandidates.
        task_type: Auto-derived from target_property — "property" or "benchmark_task".
            Never set directly.
        computed_properties: RDKit properties computed and stored in Candidate.features.
            None means all 10 GuacaMol properties. Only applies when task_type == "property".
        max_molecules: Cap on SMILES lines written to disk and loaded per file. None = full corpus.
        split_mode: "random" and "low_vs_high" use BaseDataset splitting on the combined
            corpus file. "paper" uses the original train/valid/test figshare file boundaries.
            Replaces BaseDatasetConfig.split_type — do not set split_type directly.
    """

    target_property: GuacaMolPropertyName | GuacaMolTaskName
    task_type: Literal["property", "benchmark_task"] = "property"
    computed_properties: list[GuacaMolPropertyName] | None = None
    max_molecules: int | None = None
    split_mode: Literal["random", "low_vs_high", "paper"] = "random"

    @model_validator(mode="after")
    def _validate_and_sync(self) -> "GuacaMolConfig":
        self.task_type = (
            "property" if self.target_property in ALL_PROPERTIES else "benchmark_task"
        )
        if (
            self.computed_properties is not None
            and self.target_property not in self.computed_properties
        ):
            raise ValueError(
                f"target_property '{self.target_property}' must be present in "
                "computed_properties when computed_properties is explicitly set."
            )
        if self.split_mode != "paper":
            self.split_type = self.split_mode  # type: ignore[assignment]
        return self


def _compute_properties(smiles: str, properties: list[str]) -> dict[str, float]:
    """Compute RDKit physicochemical properties for a SMILES string.

    Args:
        smiles: A valid SMILES string (caller must ensure mol parses correctly).
        properties: List of property names from GuacaMolPropertyName to compute.

    Returns:
        Dict mapping each property name to its computed float value.
    """
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    _property_fns: dict[str, object] = {
        "BertzCT": lambda m: float(GraphDescriptors.BertzCT(m)),
        "MolLogP": lambda m: float(Descriptors.MolLogP(m)),
        "MolWt": lambda m: float(Descriptors.MolWt(m)),
        "TPSA": lambda m: float(Descriptors.TPSA(m)),
        "NumHAcceptors": lambda m: float(Descriptors.NumHAcceptors(m)),
        "NumHDonors": lambda m: float(Descriptors.NumHDonors(m)),
        "NumRotatableBonds": lambda m: float(Descriptors.NumRotatableBonds(m)),
        "NumAliphaticRings": lambda m: float(rdMolDescriptors.CalcNumAliphaticRings(m)),
        "NumAromaticRings": lambda m: float(rdMolDescriptors.CalcNumAromaticRings(m)),
        "QED": lambda m: float(RDKitQED.qed(m)),
    }
    return {name: _property_fns[name](mol) for name in properties}  # type: ignore[operator]


def _download_file(url: str, filepath: Path, max_lines: int | None) -> None:
    """Stream a text file from url, writing up to max_lines lines to filepath."""
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise FileNotFoundError(
            f"Failed to download GuacaMol file from {url}. "
            f"Status code: {response.status_code}"
        )
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for i, line in enumerate(response.iter_lines()):
            if max_lines is not None and i >= max_lines:
                break
            f.write(line.decode("utf-8") + "\n")
    logger.info(f"Downloaded GuacaMol file to {filepath}.")


def _load_smiles_file(filepath: Path) -> list[str]:
    """Read non-empty SMILES strings from a file, one per line."""
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip()]


def _label_smiles(
    smiles_list: list[str],
    properties: list[str],
    target_property: str,
    modality: object,
    split_tag: str | None = None,
) -> LabelledCandidates:
    """Parse SMILES, compute properties, build LabelledCandidates.

    Invalid SMILES are skipped with a warning and excluded from the result.

    Args:
        smiles_list: Raw SMILES strings to process.
        properties: Property names to compute via RDKit.
        target_property: The property name whose value becomes the label.
        modality: Modality to assign to each Candidate.
        split_tag: If provided, stored as features["split"] on each Candidate.

    Returns:
        LabelledCandidates with 1D labels of shape (N,).
    """
    _require_rdkit()
    candidates = []
    labels = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Skipping invalid SMILES: {smiles!r}")
            continue
        features = _compute_properties(smiles, properties)
        if split_tag is not None:
            features["split"] = split_tag
        candidates.append(Candidate(data=smiles, modality=modality, features=features))
        labels.append(features[target_property])
    return LabelledCandidates(candidates=candidates, labels=np.array(labels, dtype=float))


class GuacaMol(BaseDataset):
    """GuacaMol dataset class."""

    def __init__(self, config: GuacaMolConfig) -> None:
        """Initialize the GuacaMol dataset.

        Args:
            config: Configuration for the GuacaMol dataset.
        """
        super().__init__(config)
        self.setup()

    def load_dataset(self) -> LabelledCandidates:
        """Load GuacaMol SMILES and compute physicochemical property labels via RDKit.

        Returns:
            LabelledCandidates with SMILES candidates and 1D property labels.

        Raises:
            NotImplementedError: If target_property is a benchmark task.
            FileNotFoundError: If the corpus cannot be downloaded.
        """
        if self.config.task_type == "benchmark_task":
            raise NotImplementedError(
                f"GuacaMol goal-directed task '{self.config.target_property}' is not yet "
                "implemented. Only physicochemical properties are currently supported."
            )
        if self.config.split_mode == "paper":
            return self._load_paper_splits()
        return self._load_single_file()

    def _load_single_file(self) -> LabelledCandidates:
        """Download (if absent) and label the combined corpus file."""
        filepath = DATAPATH / FILENAME_ALL
        if not filepath.exists():
            _download_file(URL_ALL, filepath, self.config.max_molecules)
        smiles_list = _load_smiles_file(filepath)
        if self.config.max_molecules is not None:
            smiles_list = smiles_list[: self.config.max_molecules]
        properties = list(self.config.computed_properties or ALL_PROPERTIES)
        return _label_smiles(
            smiles_list, properties, self.config.target_property, self.modality
        )

    def _load_paper_splits(self) -> LabelledCandidates:
        """Download (if absent) train/valid/test files and label all candidates.

        Each candidate is tagged with a "split" key in features ("train", "valid", "test").
        The three split corpuses are combined into a single LabelledCandidates for storage
        as _raw_dataset; _split_dataset() partitions them back by the tag.
        """
        split_files = [
            (FILENAME_TRAIN, URL_TRAIN, "train"),
            (FILENAME_VALID, URL_VALID, "valid"),
            (FILENAME_TEST, URL_TEST, "test"),
        ]
        properties = list(self.config.computed_properties or ALL_PROPERTIES)
        all_candidates: list[Candidate] = []
        all_labels: list[float] = []
        for filename, url, tag in split_files:
            filepath = DATAPATH / filename
            if not filepath.exists():
                _download_file(url, filepath, self.config.max_molecules)
            smiles_list = _load_smiles_file(filepath)
            if self.config.max_molecules is not None:
                smiles_list = smiles_list[: self.config.max_molecules]
            split_lc = _label_smiles(
                smiles_list, properties, self.config.target_property, self.modality, tag
            )
            all_candidates.extend(split_lc.candidates)
            all_labels.extend(split_lc.labels.tolist())
        return LabelledCandidates(
            candidates=all_candidates, labels=np.array(all_labels, dtype=float)
        )

    def _split_dataset(self) -> dict[str, LabelledCandidates]:
        """Split by paper file tags when split_mode is 'paper'; else use base class."""
        if self.config.split_mode != "paper":
            return super()._split_dataset()

        assert self._raw_dataset is not None, "Dataset must be loaded before splitting"
        tag_to_key = {"train": "train", "valid": "validation", "test": "test"}
        buckets: dict[str, tuple[list[Candidate], list[float]]] = {
            "train": ([], []),
            "validation": ([], []),
            "test": ([], []),
        }
        for candidate, label in self._raw_dataset:
            key = tag_to_key[candidate.features["split"]]
            buckets[key][0].append(candidate)
            buckets[key][1].append(float(label))
        splits = {
            key: LabelledCandidates(
                candidates=cands, labels=np.array(lbls, dtype=float)
            )
            for key, (cands, lbls) in buckets.items()
        }
        splits["candidate_pool"] = LabelledCandidates(
            candidates=[], labels=np.array([], dtype=float)
        )
        self.init_candidate_pool = copy.deepcopy(splits["candidate_pool"])
        return splits
