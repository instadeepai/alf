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
from typing import TYPE_CHECKING, Callable, Literal, Optional, get_args

import numpy as np
import requests
from alf_core import BaseDataset, Candidate, LabelledCandidates
from alf_core.dataset.base_dataset import BaseDatasetConfig
from alf_core.enums import ProblemType
from pydantic import model_validator

if TYPE_CHECKING:
    from rdkit import Chem
    from rdkit.Chem import QED as RDKitQED
    from rdkit.Chem import Descriptors, GraphDescriptors, rdMolDescriptors

logger = logging.getLogger("alf-tools")

DATAPATH = Path.home() / ".cache" / "alf"

# All 4 GuacaMol files via Figshare public API
# Source: https://api.figshare.com/v2/articles/{id}
GUACAMOL_FILES = {
    "TRAIN": {
        "name": "guacamol_v1_train.smiles",
        "url": "https://ndownloader.figshare.com/files/13612760",
        "md5": "05ad85d871958a05c02ab51a4fde8530",
        "size": 61_841_218,
    },
    "VALID": {
        "name": "guacamol_v1_valid.smiles",
        "url": "https://ndownloader.figshare.com/files/13612766",
        "md5": "e53db4bff7dc4784123ae6df72e3b1f0",
        "size": 3_859_125,
    },
    "TEST": {
        "name": "guacamol_v1_test.smiles",
        "url": "https://ndownloader.figshare.com/files/13612757",
        "md5": "677b757ccec4809febd83850b43e1616",
        "size": 11_590_126,
    },
    "ALL": {
        "name": "guacamol_v1_all.smiles",
        "url": "https://ndownloader.figshare.com/files/13612745",
        "md5": "7d45bc95c33c10cb96ef5e78c38ac0b6",
        "size": 77_290_469,
    },
}


GuacaMolPropertyName = Literal[
    "BertzCT",
    "MolLogP",
    "MolWt",
    "TPSA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "NumAliphaticRings",
    "NumAromaticRings",
    "QED",
]
GuacaMolTaskName = Literal[
    "celecoxib_rediscovery",
    "troglitazone_rediscovery",
    "thiothixene_rediscovery",
    "aripiprazole_similarity",
    "albuterol_similarity",
    "mestranol_similarity",
    "camphor_menthol_median",
    "tadalafil_sildenafil_median",
    "fexofenadine_mpo",
    "osimertinib_mpo",
    "ranolazine_mpo",
    "perindopril_mpo",
    "amlodipine_mpo",
    "sitagliptin_mpo",
    "zaleplon_mpo",
    "c7h8n2o2_isomer",
    "c9h10n2o2pf2cl_isomer",
    "aripiprazole_scaffold_hop",
    "aripiprazole_decorator_hop",
]

ALL_PROPERTIES: frozenset[str] = frozenset(get_args(GuacaMolPropertyName))
ALL_TASKS: frozenset[str] = frozenset(get_args(GuacaMolTaskName))

FILENAME_TRAIN: str = GUACAMOL_FILES["TRAIN"]["name"]
FILENAME_VALID: str = GUACAMOL_FILES["VALID"]["name"]
FILENAME_TEST: str = GUACAMOL_FILES["TEST"]["name"]
FILENAME_ALL: str = GUACAMOL_FILES["ALL"]["name"]

try:
    from rdkit import Chem  # type: ignore[no-redef]
    from rdkit.Chem import QED as RDKitQED  # type: ignore[no-redef]
    from rdkit.Chem import Descriptors, GraphDescriptors, rdMolDescriptors  # type: ignore[no-redef]

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
        data_dir: Directory where SMILES files are cached. Defaults to the package data dir.
    """

    problem_type: ProblemType = ProblemType.REGRESSION
    target_property: GuacaMolPropertyName | GuacaMolTaskName
    task_type: Literal["property", "benchmark_task"] = "property"
    computed_properties: list[GuacaMolPropertyName] | None = None
    max_molecules: int | None = None
    split_mode: Literal["random", "low_vs_high", "paper"] = "random"
    data_dir: Path = DATAPATH

    @model_validator(mode="after")
    def _validate_and_sync(self) -> "GuacaMolConfig":
        self.task_type = "property" if self.target_property in ALL_PROPERTIES else "benchmark_task"
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
    _property_fns: dict[str, Callable[..., float]] = {
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


def _download_file(url: str, filepath: Path, max_lines: Optional[int] = None) -> Path:
    """Stream a text file from url to filepath, optionally truncating to max_lines lines.

    Skips the download if filepath already exists. Creates parent directories as needed.

    Args:
        url: HTTPS URL to stream from.
        filepath: Destination file path (not a directory).
        max_lines: If set, stop writing after exactly this many lines.

    Returns:
        The resolved filepath.

    Raises:
        FileNotFoundError: If the server returns a non-200 status code.
    """
    if filepath.exists():
        logger.info("  ✓ %s already exists, skipping.", filepath)
        return filepath

    logger.info(
        "  ↓ Downloading %s%s...",
        filepath.name,
        f" (first {max_lines} lines)" if max_lines is not None else "",
    )
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # allow_redirects=True is the default — requests follows the 302 → S3 automatically
    resp = requests.get(url, stream=True, timeout=60)
    if resp.status_code != 200:
        raise FileNotFoundError(f"Failed to download from {url}. Status code: {resp.status_code}")
    with open(filepath, "wb") as f:
        for idx, raw_line in enumerate(resp.iter_lines()):
            f.write(raw_line + b"\n")
            if max_lines is not None and idx + 1 >= max_lines:
                break
    logger.info("  ✓ %s written to %s.", filepath.name, filepath.parent)
    return filepath


def _load_smiles_file(filepath: Path) -> list[str]:
    """Read non-empty SMILES strings from a file, one per line.

    Args:
        filepath: Path to a newline-delimited SMILES file.

    Returns:
        List of stripped, non-empty SMILES strings.
    """
    with open(filepath, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def download_guacamol(data_dir: Path = DATAPATH, max_lines: int | None = None) -> None:
    """Download all four GuacaMol SMILES files to data_dir.

    Args:
        data_dir: Destination directory. Defaults to the package data directory.
        max_lines: If set, each file is truncated to at most this many lines.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    for file_info in GUACAMOL_FILES.values():
        _download_file(file_info["url"], data_dir / file_info["name"], max_lines)


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
            logger.warning("Skipping invalid SMILES: %r", smiles)
            continue
        features = _compute_properties(smiles, properties)
        if split_tag is not None:
            features["split"] = split_tag
        candidates.append(Candidate(data=smiles, modality=modality, features=features))
        labels.append(features[target_property])
    return LabelledCandidates(candidates=candidates, labels=np.array(labels, dtype=float))


class GuacaMol(BaseDataset):
    """GuacaMol dataset for physicochemical property prediction on drug-like molecules.

    Wraps the GuacaMol benchmark corpus (1.6 M SMILES from ChEMBL) and computes
    RDKit physicochemical properties (e.g. MolLogP, TPSA, QED) as regression targets.

    Three split modes are supported:
    - ``"random"`` / ``"low_vs_high"``: BaseDataset splitting on the combined corpus.
    - ``"paper"``: uses the original train/valid/test file boundaries from the
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
        super().__init__(config)
        self.setup()

    def __repr__(self) -> str:
        """Return a string representation identifying dataset and target."""
        return (
            f"GuacaMol(name={self.config.name}, modality={self.modality}, "
            f"seed={self.config.seed}, "
            f"target_property={self.config.target_property}, "
            f"split_mode={self.config.split_mode})"
        )

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
        """Download (if absent) and label the combined corpus file.

        Returns:
            LabelledCandidates built from the combined corpus.
        """
        entry_info_all = GUACAMOL_FILES["ALL"]
        filepath = _download_file(
            entry_info_all["url"],
            self.config.data_dir / entry_info_all["name"],
            self.config.max_molecules,
        )
        smiles_list = _load_smiles_file(filepath)
        if self.config.max_molecules is not None:
            smiles_list = smiles_list[: self.config.max_molecules]
        properties = list(self.config.computed_properties or ALL_PROPERTIES)
        return _label_smiles(smiles_list, properties, self.config.target_property, self.modality)

    def _load_paper_splits(self) -> LabelledCandidates:
        """Download (if absent) train/valid/test files and label all candidates.

        Each candidate is tagged with a "split" key in features ("train", "valid", "test").
        The three split corpuses are combined into a single LabelledCandidates for storage
        as _raw_dataset; _split_dataset() partitions them back by the tag.

        Returns:
            Combined LabelledCandidates with split tags stored in each candidate's features.
        """
        split_files = {k: GUACAMOL_FILES[k] for k in ("TRAIN", "VALID", "TEST")}
        properties = list(self.config.computed_properties or ALL_PROPERTIES)
        all_candidates: list[Candidate] = []
        all_labels: list[float] = []
        for tag, entry_info in split_files.items():
            filepath = _download_file(
                entry_info["url"],
                self.config.data_dir / entry_info["name"],
                self.config.max_molecules,
            )
            smiles_list = _load_smiles_file(filepath)
            if self.config.max_molecules is not None:
                smiles_list = smiles_list[: self.config.max_molecules]
            split_lc = _label_smiles(
                smiles_list, properties, self.config.target_property, self.modality, tag.lower()
            )
            logger.debug(
                "Paper split '%s': %d SMILES → %d valid candidates",
                tag,
                len(smiles_list),
                len(split_lc.candidates),
            )
            all_candidates.extend(split_lc.candidates)
            all_labels.extend(split_lc.labels.tolist())
        return LabelledCandidates(
            candidates=all_candidates, labels=np.array(all_labels, dtype=float)
        )

    def query(self, candidates: list[Candidate]) -> LabelledCandidates:
        """Return labels for candidates, computing via RDKit for SMILES not in the corpus.

        Args:
            candidates: Candidates to label. May include SMILES not present in _raw_dataset.

        Returns:
            LabelledCandidates with 1D labels of shape (N,).

        Raises:
            NotImplementedError: If task_type is "benchmark_task".
            ValueError: If a novel candidate's SMILES string is invalid.
        """
        if self.config.task_type == "benchmark_task":
            raise NotImplementedError(
                f"Online query for task '{self.config.target_property}' is not yet implemented."
            )
        if self._raw_dataset is None:
            raise RuntimeError("Dataset must be loaded before querying")  # pragma: no cover

        known_smiles_index = {c.data: i for i, c in enumerate(self._raw_dataset.candidates)}
        result_candidates: list[Candidate] = []
        result_labels: list[float] = []

        for candidate in candidates:
            if candidate.data in known_smiles_index:
                idx = known_smiles_index[candidate.data]
                result_labels.append(float(self._raw_dataset.labels[idx]))
            else:
                _require_rdkit()
                mol = Chem.MolFromSmiles(candidate.data)
                if mol is None:
                    raise ValueError(f"Cannot compute label for invalid SMILES: {candidate.data!r}")
                label = _compute_properties(candidate.data, [self.config.target_property])[
                    self.config.target_property
                ]
                result_labels.append(label)
            result_candidates.append(candidate)

        return LabelledCandidates(
            candidates=result_candidates,
            labels=np.array(result_labels, dtype=float),
        )

    def _split_dataset(self) -> dict[str, LabelledCandidates]:
        """Split by paper file tags when split_mode is 'paper'; else use base class.

        Returns:
            Dict with keys "train", "validation", "test", and "candidate_pool".
        """
        if self.config.split_mode != "paper":
            return super()._split_dataset()

        if self._raw_dataset is None:
            raise RuntimeError("Dataset must be loaded before splitting")  # pragma: no cover
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
            key: LabelledCandidates(candidates=cands, labels=np.array(lbls, dtype=float))
            for key, (cands, lbls) in buckets.items()
        }
        splits["candidate_pool"] = LabelledCandidates(
            candidates=[], labels=np.array([], dtype=float)
        )
        self.init_candidate_pool = copy.deepcopy(splits["candidate_pool"])
        return splits
