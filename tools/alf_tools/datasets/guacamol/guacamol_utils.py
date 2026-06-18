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

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Callable, Final, Literal, NotRequired, TypedDict, get_args

import numpy as np
import requests
from alf_core import Candidate, LabelledCandidates, Modality
from rdkit import Chem
from rdkit.Chem import QED as RDKitQED
from rdkit.Chem import Descriptors, GraphDescriptors, rdMolDescriptors

from .guacamol_scoring import _mol_from_smiles

logger = logging.getLogger("alf-tools")

DATAPATH = Path(__file__).parent / "data"

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
        name: Local filename (e.g. `guacamol_v1_train.smiles`).
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
        "sha256": None,  # no SHA-256 from Figshare; MD5: 05ad85d871958a05c02ab51a4fde8530
    },
    "VALID": {
        "name": FILENAME_VALID,
        "url": "https://ndownloader.figshare.com/files/13612766",
        "sha256": None,  # no SHA-256 from Figshare; MD5: e53db4bff7dc4784123ae6df72e3b1f0
    },
    "TEST": {
        "name": FILENAME_TEST,
        "url": "https://ndownloader.figshare.com/files/13612757",
        "sha256": None,  # no SHA-256 from Figshare; MD5: 677b757ccec4809febd83850b43e1616
    },
    "ALL": {
        "name": FILENAME_ALL,
        "url": "https://ndownloader.figshare.com/files/13612745",
        "sha256": None,  # no SHA-256 from Figshare; MD5: 7d45bc95c33c10cb96ef5e78c38ac0b6
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
# Stable ordered tuple — use `computed_properties=list(_ALL_PROPERTIES_ORDERED)` when
# deterministic iteration over all 10 properties is required.
_ALL_PROPERTIES_ORDERED: tuple[GuacaMolPropertyName, ...] = get_args(GuacaMolPropertyName)


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
        base: Base file path (e.g. `~/.cache/alf/guacamol_v1_all.smiles`).
        max_lines: Line cap. When set, the count is embedded in the filename so that
            different caps never share the same cached file.

    Returns:
        `base` unchanged when max_lines is None, otherwise
        `base.parent / f"{base.stem}_{max_lines}lines{base.suffix}"`.
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
    :func:`_cache_path` (e.g. `guacamol_v1_all_1000lines.smiles`) so that
    different caps never collide in the cache.  Writes to a `.tmp` file first
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
        logger.info(
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
        _download_file(
            file_info["url"],
            data_dir / file_info["name"],
            max_lines,
            sha256=file_info.get("sha256"),
        )


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
) -> tuple[LabelledCandidates, np.ndarray]:
    """Parse SMILES, compute properties, build LabelledCandidates and property matrix.

    Invalid SMILES are skipped with a warning and excluded from both outputs.
    `Candidate.features` is always empty (`{}`); property values are returned
    in the second element as a 2D array of shape `(N, len(properties))`.

    Args:
        smiles_list: Raw SMILES strings to process.
        properties: Property names to compute via RDKit (determines matrix columns).
        target_property: The property name whose value becomes the label.
        modality: Modality to assign to each Candidate.

    Returns:
        Tuple of (LabelledCandidates with 1D labels, property matrix of shape (N, P)).
    """
    p = len(properties)
    prop_matrix = np.empty((len(smiles_list), p), dtype=np.float64)
    candidates: list[Candidate] = []
    labels: list[float] = []
    row = 0
    for smiles in smiles_list:
        mol = _mol_from_smiles(smiles)
        props = _compute_properties(smiles, properties)
        if mol is None or props is None:
            logger.warning("Skipping invalid SMILES: %r", smiles)
            continue
        canonical = Chem.MolToSmiles(mol)
        prop_matrix[row] = [props[name] for name in properties]
        candidates.append(Candidate(data=canonical, modality=modality, features={}))
        labels.append(props[target_property])
        row += 1
    return (
        LabelledCandidates(candidates=candidates, labels=np.array(labels, dtype=float)),
        prop_matrix[:row],
    )


def _label_smiles_benchmark(
    smiles_list: list[str],
    scorer: Callable[[str], float],
    modality: "Modality | str",
) -> LabelledCandidates:
    """Score SMILES using a benchmark task scorer, skipping invalid SMILES.

    Args:
        smiles_list: Raw SMILES strings to process.
        scorer: Benchmark task scoring function; returns a float in [0, 1].
        modality: Modality to assign to each Candidate.

    Returns:
        LabelledCandidates with 1D labels of shape (N,). Candidates have empty features.
    """
    candidates = []
    labels = []
    for smiles in smiles_list:
        mol = _mol_from_smiles(smiles)
        if mol is None:
            logger.warning("Skipping invalid SMILES: %r", smiles)
            continue
        canonical = Chem.MolToSmiles(mol)
        candidates.append(Candidate(data=canonical, modality=modality, features={}))
        labels.append(scorer(smiles))
    return LabelledCandidates(candidates=candidates, labels=np.array(labels, dtype=float))
