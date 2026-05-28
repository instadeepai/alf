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
from typing import Callable, Final, Literal, TypedDict, get_args

import numpy as np
import requests
from alf_core import BaseDataset, Candidate, LabelledCandidates
from alf_core.dataset.base_dataset import BaseDatasetConfig
from alf_core.utils.enums import ProblemType
from pydantic import model_validator
from rdkit import Chem
from rdkit.Chem import QED as RDKitQED
from rdkit.Chem import Descriptors, GraphDescriptors, rdMolDescriptors

logger = logging.getLogger("alf-tools")

DATAPATH = Path.home() / ".cache" / "alf"

# All 4 GuacaMol files via Figshare public API
# Source: https://api.figshare.com/v2/articles/{id}
FILENAME_TRAIN: str = "guacamol_v1_train.smiles"
FILENAME_VALID: str = "guacamol_v1_valid.smiles"
FILENAME_TEST: str = "guacamol_v1_test.smiles"
FILENAME_ALL: str = "guacamol_v1_all.smiles"

GuacaMolSplitName = Literal["TRAIN", "VALID", "TEST", "ALL"]

PROPERTY_FNS: dict[str, Callable[..., float]] = {
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


class GuacaMolFileInfo(TypedDict):
    """File metadata for a single GuacaMol download.

    Attributes:
        name: Local filename (e.g. ``guacamol_v1_train.smiles``).
        url: HTTPS download URL.
    """

    name: str
    url: str


GUACAMOL_FILES: Final[dict[GuacaMolSplitName, GuacaMolFileInfo]] = {
    "TRAIN": {
        "name": FILENAME_TRAIN,
        "url": "https://ndownloader.figshare.com/files/13612760",
    },
    "VALID": {
        "name": FILENAME_VALID,
        "url": "https://ndownloader.figshare.com/files/13612766",
    },
    "TEST": {
        "name": FILENAME_TEST,
        "url": "https://ndownloader.figshare.com/files/13612757",
    },
    "ALL": {
        "name": FILENAME_ALL,
        "url": "https://ndownloader.figshare.com/files/13612745",
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
        smiles: A SMILES string to compute properties for.
        properties: List of property names from GuacaMolPropertyName to compute.

    Raises:
        ValueError: If the SMILES string is invalid and cannot be parsed by RDKit.

    Returns:
        Dict mapping each property name to its computed float value.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot compute label for invalid SMILES: {smiles!r}")
    return {name: PROPERTY_FNS[name](mol) for name in properties}  # type: ignore[operator]  # mypy cannot narrow str subscript to Literal key type


def _download_file(url: str, filepath: Path, max_lines: int | None = None) -> Path:
    """Stream a text file from url to filepath, optionally truncating to max_lines lines.

    Writes to a ``.tmp`` file first and renames on success to prevent partial downloads
    from appearing valid on the next call. Skips the download if the target filepath
    already exists and contains at least max_lines non-empty lines (or max_lines is None).
    If the cached file has fewer lines than max_lines, it is deleted and re-downloaded.

    Args:
        url: HTTPS URL to stream from.
        filepath: Destination file path (not a directory).
        max_lines: If set, stop writing after exactly this many lines. An existing file
            with fewer than max_lines lines is treated as stale and re-downloaded.

    Returns:
        The resolved filepath.

    Raises:
        OSError: If a network error occurs while connecting or streaming.
        FileNotFoundError: If the server returns a non-200 status code.
    """
    if filepath.exists():
        if max_lines is None:
            logger.info("  ✓ %s already exists, skipping.", filepath)
            return filepath
        with open(filepath, encoding="utf-8") as f:
            cached_count = sum(1 for line in f if line.strip())
        if cached_count >= max_lines:
            logger.info("  ✓ %s already exists with sufficient lines, skipping.", filepath)
            return filepath
        filepath.unlink()

    logger.info(
        "  ↓ Downloading %s%s...",
        filepath.name,
        f" (first {max_lines} lines)" if max_lines is not None else "",
    )
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_suffix(".tmp")
    # allow_redirects=True is the default — requests follows the 302 → S3 automatically
    try:
        resp = requests.get(url, stream=True, timeout=60)
    except requests.RequestException as exc:
        raise OSError(f"Network error downloading {filepath.name} from {url}") from exc
    if resp.status_code != 200:
        raise FileNotFoundError(f"Failed to download from {url}. Status code: {resp.status_code}")
    with open(tmp_path, "wb") as f:
        for idx, raw_line in enumerate(resp.iter_lines()):
            f.write(raw_line + b"\n")
            if max_lines is not None and idx + 1 >= max_lines:
                break
    tmp_path.rename(filepath)
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


def _canonical_smiles(smiles: str) -> str:
    """Return the RDKit canonical form of a SMILES string, or the original if invalid.

    Used to normalise lookup keys so that structurally identical molecules with different
    SMILES representations resolve to the same index entry in :meth:`GuacaMol.query`.

    Args:
        smiles: Input SMILES string.

    Returns:
        Canonical SMILES string, or the original string if RDKit cannot parse it.
    """
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else smiles


def _label_smiles(
    smiles_list: list[str],
    properties: list[str],
    target_property: str,
    modality: object,
) -> LabelledCandidates:
    """Parse SMILES, compute properties, build LabelledCandidates.

    Invalid SMILES are skipped with a warning and excluded from the result.

    Args:
        smiles_list: Raw SMILES strings to process.
        properties: Property names to compute via RDKit.
        target_property: The property name whose value becomes the label.
        modality: Modality to assign to each Candidate.

    Returns:
        LabelledCandidates with 1D labels of shape (N,).
    """
    candidates = []
    labels = []
    for smiles in smiles_list:
        try:
            props = _compute_properties(smiles, properties)
        except ValueError:
            logger.warning("Skipping invalid SMILES: %r", smiles)
            continue
        candidates.append(Candidate(data=smiles, modality=modality, features=dict(props)))
        labels.append(props[target_property])
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
        self._paper_splits: dict[str, LabelledCandidates] | None = None
        super().__init__(config)
        self.setup()
        self._smiles_index: dict[str, int] = (
            {_canonical_smiles(c.data): i for i, c in enumerate(self._raw_dataset.candidates)}
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

        Stores the three splits in ``self._paper_splits`` keyed by
        ``"train"``, ``"validation"``, and ``"test"``. Returns a combined
        LabelledCandidates (without any split tag in features) for use as
        ``_raw_dataset`` — this powers the SMILES lookup index in :meth:`query`.

        Returns:
            Combined LabelledCandidates across all three paper splits.
        """
        split_files = {k: v for k, v in GUACAMOL_FILES.items() if k != "ALL"}
        tag_to_key = {"TRAIN": "train", "VALID": "validation", "TEST": "test"}
        properties = list(self.config.computed_properties or ALL_PROPERTIES)
        self._paper_splits = {}
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
                smiles_list, properties, self.config.target_property, self.modality
            )
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

    def query(self, candidates: list[Candidate]) -> LabelledCandidates:
        """Return labels for candidates, computing via RDKit for SMILES not in the corpus.

        Args:
            candidates: Candidates to label. May include SMILES not present in _raw_dataset.

        Returns:
            LabelledCandidates with 1D labels of shape (N,). The returned candidates are
            always the caller's input objects — corpus lookup provides the label only.

        Raises:
            NotImplementedError: If task_type is "benchmark_task".
            ValueError: If a novel candidate's SMILES string is invalid.
            RuntimeError: If the dataset is not loaded before querying.
        """
        if self.config.task_type == "benchmark_task":
            raise NotImplementedError(
                f"Online query for task '{self.config.target_property}' is not yet implemented."
            )
        if self._raw_dataset is None:
            raise RuntimeError("Dataset must be loaded before querying")  # pragma: no cover

        result_candidates: list[Candidate] = []
        result_labels: list[float] = []

        for candidate in candidates:
            key = _canonical_smiles(candidate.data)
            if key in self._smiles_index:
                idx = self._smiles_index[key]
                result_labels.append(float(self._raw_dataset.labels[idx]))
                result_candidates.append(candidate)
            else:
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

        # Shallow copy: values still alias self._paper_splits entries.
        # _paper_splits is not re-read after setup(), so in-place mutations via
        # update_splits() do not cause bugs, but future callers should be aware.
        splits = dict(self._paper_splits)
        splits["candidate_pool"] = LabelledCandidates(
            candidates=[], labels=np.array([], dtype=float)
        )
        self.init_candidate_pool = copy.deepcopy(splits["candidate_pool"])
        return splits
