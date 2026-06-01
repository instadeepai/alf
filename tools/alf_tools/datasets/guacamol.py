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
import hashlib
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Final, Literal, NotRequired, TypedDict, cast, get_args

import numpy as np
import requests
from alf_core import BaseDataset, Candidate, LabelledCandidates
from alf_core.dataclasses.candidate import Modality
from alf_core.dataset.base_dataset import BaseDatasetConfig
from alf_core.utils.enums import ProblemType
from pydantic import Field, computed_field, model_validator
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
        sha256: Expected SHA-256 hex digest. When present, the downloaded file is
            verified against this value. Set to None or omit to skip verification.
    """

    name: str
    url: str
    sha256: NotRequired[str | None]


GUACAMOL_FILES: Final[dict[Literal["TRAIN", "VALID", "TEST", "ALL"], GuacaMolFileInfo]] = {
    "TRAIN": {
        "name": FILENAME_TRAIN,
        "url": "https://ndownloader.figshare.com/files/13612760",
        "sha256": None,  # SHA-256 not published by Figshare; Figshare MD5: 05ad85d871958a05c02ab51a4fde8530
    },
    "VALID": {
        "name": FILENAME_VALID,
        "url": "https://ndownloader.figshare.com/files/13612766",
        "sha256": None,  # SHA-256 not published by Figshare; Figshare MD5: e53db4bff7dc4784123ae6df72e3b1f0
    },
    "TEST": {
        "name": FILENAME_TEST,
        "url": "https://ndownloader.figshare.com/files/13612757",
        "sha256": None,  # SHA-256 not published by Figshare; Figshare MD5: 677b757ccec4809febd83850b43e1616
    },
    "ALL": {
        "name": FILENAME_ALL,
        "url": "https://ndownloader.figshare.com/files/13612745",
        "sha256": None,  # SHA-256 not published by Figshare; Figshare MD5: 7d45bc95c33c10cb96ef5e78c38ac0b6
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

ALL_PROPERTIES: frozenset[GuacaMolPropertyName] = frozenset(get_args(GuacaMolPropertyName))
# Stable ordered tuple — use ``computed_properties=list(_ALL_PROPERTIES_ORDERED)`` when
# deterministic iteration over all 10 properties is required.
_ALL_PROPERTIES_ORDERED: tuple[GuacaMolPropertyName, ...] = get_args(GuacaMolPropertyName)


class GuacaMolConfig(BaseDatasetConfig):
    """Configuration for GuacaMol dataset.

    Attributes:
        target_property: Property or task name used as labels in LabelledCandidates.
        task_type: Always auto-derived from target_property in the model validator.
            Any value supplied at construction is silently overwritten. Do not set.
        computed_properties: RDKit properties computed and stored in Candidate.features.
            None defaults to computing only [target_property]. Pass
            ``list(_ALL_PROPERTIES_ORDERED)`` to compute all 10. Only applies when
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

    @computed_field
    @property
    def task_type(self) -> Literal["property", "benchmark_task"]:
        """Derived from target_property; 'property' for RDKit properties, 'benchmark_task' for goal-directed tasks."""
        return "property" if self.target_property in ALL_PROPERTIES else "benchmark_task"

    @model_validator(mode="after")
    def _validate_and_sync(self) -> "GuacaMolConfig":
        if (
            self.computed_properties is not None
            and self.target_property not in self.computed_properties
        ):
            raise ValueError(
                f"target_property '{self.target_property}' must be present in "
                "computed_properties when computed_properties is explicitly set."
            )
        if self.split_mode != "paper":
            # split_mode values "random" and "low_vs_high" are valid SplitType values;
            # cast is safe here since BaseDatasetConfig.split_type accepts them.
            from alf_core.dataset.splitting_utils import SplitType  # noqa: PLC0415

            assert self.split_mode in get_args(SplitType), (
                f"split_mode {self.split_mode!r} is not a valid SplitType value. "
                f"Valid values: {get_args(SplitType)}"
            )
            self.split_type = cast(SplitType, self.split_mode)
        return self


def _compute_properties(
    smiles: str, properties: list[GuacaMolPropertyName]
) -> dict[str, float] | None:
    """Compute RDKit physicochemical properties for a SMILES string.

    Args:
        smiles: A SMILES string to compute properties for.
        properties: List of property names from GuacaMolPropertyName to compute.

    Returns:
        Dict mapping each property name to its computed float value, or None if
        the SMILES string is invalid and cannot be parsed by RDKit.
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    return {name: PROPERTY_FNS[name](mol) for name in properties}


def _cache_path(base: Path, max_lines: int | None) -> Path:
    """Return the cache filepath for a given base path and optional max_lines cap.

    Args:
        base: Base file path (e.g. ``~/.cache/alf/guacamol_v1_all.smiles``).
        max_lines: Line cap. When set, the count is embedded in the filename so that
            different caps never share the same cached file.

    Returns:
        ``base`` unchanged when max_lines is None, otherwise
        ``base.parent / f"{base.stem}_{max_lines}lines{base.suffix}"``.
    """
    if max_lines is None:
        return base
    return base.parent / f"{base.stem}_{max_lines}lines{base.suffix}"


def _download_file(
    url: str,
    filepath: Path,
    max_lines: int | None = None,
    sha256: str | None = None,
) -> Path:
    """Stream a text file from url to filepath, optionally truncating to max_lines lines.

    When max_lines is given the line count is embedded in the filename via
    :func:`_cache_path` (e.g. ``guacamol_v1_all_1000lines.smiles``) so that
    different caps never collide in the cache.  Writes to a ``.tmp`` file first
    and renames on success to prevent partial downloads from appearing valid.

    Args:
        url: HTTPS URL to stream from.
        filepath: Base destination path. The actual path may differ when max_lines
            is set — always use the returned value.
        max_lines: If set, stop writing after exactly this many lines.
        sha256: Expected SHA-256 hex digest. When provided (and max_lines is None),
            the written file is verified against this digest. Omit or pass None to
            skip verification.

    Returns:
        The resolved filepath (may differ from the input when max_lines is set).

    Raises:
        OSError: If a network error occurs while connecting or streaming.
        FileNotFoundError: If the server returns a non-200 status code.
        ValueError: If sha256 is provided and the downloaded file does not match.
    """
    filepath = _cache_path(filepath, max_lines)

    if filepath.exists():
        logger.info("  ✓ %s already exists, skipping.", filepath)
        return filepath

    logger.info(
        "  ↓ Downloading %s%s...",
        filepath.name,
        f" (first {max_lines} lines)" if max_lines is not None else "",
    )
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Clean up stale .tmp files from previously killed processes (SIGKILL cannot run except blocks)
    _stale_threshold = 3600  # 1 hour
    for _stale in filepath.parent.glob(f"{filepath.stem}.*.tmp"):
        try:
            if time.time() - _stale.stat().st_mtime > _stale_threshold:
                _stale.unlink(missing_ok=True)
        except OSError:
            pass
    tmp_path = filepath.with_name(f"{filepath.stem}.{os.getpid()}.tmp")
    # allow_redirects=True is the default — requests follows the 302 → S3 automatically
    try:
        resp = requests.get(url, stream=True, timeout=(10, 120))
    except requests.RequestException as exc:
        raise OSError(f"Network error downloading {filepath.name} from {url}") from exc
    if resp.status_code == 404:
        raise FileNotFoundError(f"File not found at {url}. Status code: 404")
    if resp.status_code != 200:
        raise OSError(f"Failed to download from {url}. Status code: {resp.status_code}")

    try:
        with open(tmp_path, "wb") as f:
            for idx, raw_line in enumerate(resp.iter_lines()):
                f.write(raw_line + b"\n")
                if max_lines is not None and idx + 1 >= max_lines:
                    break
        tmp_path.rename(filepath)
    except requests.RequestException as exc:
        tmp_path.unlink(missing_ok=True)
        raise OSError(f"Network error streaming {filepath.name} from {url}") from exc
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    if sha256 is None:
        logger.warning(
            "No SHA-256 checksum configured for %s — integrity not verified.", filepath.name
        )
    else:
        digest = hashlib.sha256(filepath.read_bytes()).hexdigest()
        if digest != sha256:
            filepath.unlink(missing_ok=True)
            raise ValueError(
                f"SHA-256 mismatch for {filepath.name}: expected {sha256!r}, got {digest!r}"
            )
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
        _download_file(file_info["url"], data_dir / file_info["name"], max_lines, sha256=file_info.get("sha256"))


@lru_cache(maxsize=65536)
def _mol_from_smiles(smiles: str) -> "Chem.Mol | None":
    """Return the RDKit Mol for smiles, or None if invalid. Result is cached per unique string."""
    return Chem.MolFromSmiles(smiles)


def _canonical_smiles(smiles: str) -> str:
    """Return the RDKit canonical form of a SMILES string, or the original if invalid.

    Used to normalise lookup keys so that structurally identical molecules with different
    SMILES representations resolve to the same index entry in :meth:`GuacaMol.query`.

    Args:
        smiles: Input SMILES string.

    Returns:
        Canonical SMILES string, or the original string if RDKit cannot parse it.
    """
    mol = _mol_from_smiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else smiles


def _label_smiles(
    smiles_list: list[str],
    properties: list[GuacaMolPropertyName],
    target_property: GuacaMolPropertyName,
    modality: Modality | str,
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
        mol = _mol_from_smiles(smiles)
        props = _compute_properties(smiles, properties)
        if mol is None or props is None:
            logger.warning("Skipping invalid SMILES: %r", smiles)
            continue
        canonical = Chem.MolToSmiles(mol)
        candidates.append(Candidate(data=canonical, modality=modality, features=dict(props)))
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
        self._smiles_index: dict[str, float] = {}
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
            sha256=entry_info_all.get("sha256"),
        )
        smiles_list = _load_smiles_file(filepath)
        if self.config.max_molecules is not None:
            smiles_list = smiles_list[: self.config.max_molecules]
        target = cast(GuacaMolPropertyName, self.config.target_property)
        properties = list(self.config.computed_properties or [target])
        return _label_smiles(smiles_list, properties, target, self.modality)

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
        target = cast(GuacaMolPropertyName, self.config.target_property)
        properties = list(self.config.computed_properties or [target])
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
            mol = _mol_from_smiles(candidate.data)
            key = Chem.MolToSmiles(mol) if mol is not None else candidate.data
            if key in self._smiles_index:
                result_labels.append(self._smiles_index[key])
                result_candidates.append(candidate)
            else:
                if mol is None:
                    raise ValueError(f"Cannot compute label for invalid SMILES: {candidate.data!r}")
                label_val = PROPERTY_FNS[self.config.target_property](mol)
                # Populate features if not already set
                if not candidate.features:
                    props_to_compute: list[GuacaMolPropertyName] = list(
                        self.config.computed_properties
                        or [cast(GuacaMolPropertyName, self.config.target_property)]
                    )
                    candidate = Candidate(
                        data=candidate.data,
                        modality=candidate.modality,
                        features=dict(_compute_properties(candidate.data, props_to_compute) or {}),
                    )
                result_labels.append(label_val)
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

        splits = {k: copy.deepcopy(v) for k, v in self._paper_splits.items()}
        splits["candidate_pool"] = LabelledCandidates(
            candidates=[], labels=np.array([], dtype=float)
        )
        self.init_candidate_pool = copy.deepcopy(splits["candidate_pool"])
        return splits
