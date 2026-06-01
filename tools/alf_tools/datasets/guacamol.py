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
import math
import os
import re
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
from rdkit import Chem, DataStructs
from rdkit.Chem import QED as RDKitQED
from rdkit.Chem import AllChem, Descriptors, GraphDescriptors, rdMolDescriptors

logger = logging.getLogger("alf-tools")

# ---------------------------------------------------------------------------
# Score modifiers for benchmark task scoring
# ---------------------------------------------------------------------------


def clipped_score(x: float, upper: float = 1.0) -> float:
    """Linear ramp [0, upper] → [0, 1], clipped to 1.0 above upper.

    Returns:
        Score in [0, 1].
    """
    return min(1.0, x / upper)


def gaussian_score(x: float, mu: float, sigma: float) -> float:
    """Gaussian bell: 1.0 at x==mu, decaying symmetrically with sigma.

    Returns:
        Score in (0, 1].
    """
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def max_gaussian_score(x: float, mu: float, sigma: float) -> float:
    """Half-Gaussian: 1.0 for x >= mu, Gaussian fall-off below mu.

    Returns:
        Score in (0, 1].
    """
    return 1.0 if x >= mu else gaussian_score(x, mu, sigma)


def min_gaussian_score(x: float, mu: float, sigma: float) -> float:
    """Half-Gaussian: 1.0 for x <= mu, Gaussian fall-off above mu.

    Returns:
        Score in (0, 1].
    """
    return 1.0 if x <= mu else gaussian_score(x, mu, sigma)


def geometric_mean(scores: list[float]) -> float:
    """Geometric mean of scores; returns 0.0 for empty list.

    Returns:
        Geometric mean in [0, 1] for scores in [0, 1].
    """
    if not scores:
        return 0.0
    return math.prod(scores) ** (1.0 / len(scores))


def arithmetic_mean(scores: list[float]) -> float:
    """Arithmetic mean of scores; returns 0.0 for empty list.

    Returns:
        Arithmetic mean of the input scores.
    """
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def task_type(self) -> Literal["property", "benchmark_task"]:
        """'property' for RDKit properties, 'benchmark_task' for goal-directed tasks."""
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
        _download_file(
            file_info["url"],
            data_dir / file_info["name"],
            max_lines,
            sha256=file_info.get("sha256"),
        )


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


# ---------------------------------------------------------------------------
# Fingerprint helpers for benchmark task scoring
# ---------------------------------------------------------------------------


def _ecfp4(mol: "Chem.Mol"):
    """Morgan fingerprint radius=2 (ECFP4).

    Returns:
        RDKit Morgan fingerprint object.
    """
    return AllChem.GetMorganFingerprint(mol, 2)


def _ecfp6(mol: "Chem.Mol"):
    """Morgan fingerprint radius=3 (ECFP6).

    Returns:
        RDKit Morgan fingerprint object.
    """
    return AllChem.GetMorganFingerprint(mol, 3)


def _fcfp4(mol: "Chem.Mol"):
    """Feature-based Morgan fingerprint radius=2 (FCFP4).

    Returns:
        RDKit Morgan fingerprint object using pharmacophoric features.
    """
    return AllChem.GetMorganFingerprint(mol, 2, useFeatures=True)


def _ap(mol: "Chem.Mol"):
    """Atom-pair fingerprint.

    Returns:
        RDKit atom-pair fingerprint object.
    """
    return rdMolDescriptors.GetAtomPairFingerprint(mol)


def _phco(mol: "Chem.Mol"):
    """2D pharmacophore fingerprint (Gobbi).

    Returns:
        RDKit 2D pharmacophore fingerprint object.
    """
    from rdkit.Chem.Pharm2D import (  # noqa: PLC0415
        Generate,
        Gobbi_Pharm2D,
    )

    return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)


def _tanimoto(smiles: str, ref_fp, fp_fn: Callable) -> float:
    """Tanimoto similarity of smiles to ref_fp using the given fingerprint function.

    Returns:
        Tanimoto similarity in [0, 1], or 0.0 for unparseable SMILES.
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp_fn(mol), ref_fp)


# ---------------------------------------------------------------------------
# Isomer and SMARTS scoring helpers
# ---------------------------------------------------------------------------


def _parse_formula(formula: str) -> dict[str, int]:
    """Parse a molecular formula string into {element: count}.

    Example: 'C7H8N2O2' → {'C': 7, 'H': 8, 'N': 2, 'O': 2}

    Returns:
        Mapping of element symbol to atom count.
    """
    return {el: int(cnt or 1) for el, cnt in re.findall(r"([A-Z][a-z]?)(\d*)", formula) if el}


def isomer_score(smiles: str, target_formula: dict[str, int]) -> float:
    """Geometric mean of per-element Gaussian scores (mu=target count, sigma=1).

    Scores 1.0 when the molecule's formula matches target_formula exactly.

    Returns:
        Score in [0, 1]; 0.0 for invalid SMILES.
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return 0.0
    mol_formula = rdMolDescriptors.CalcMolFormula(mol)
    mol_counts = _parse_formula(mol_formula)
    scores = [
        gaussian_score(float(mol_counts.get(el, 0)), mu=float(target_count), sigma=1.0)
        for el, target_count in target_formula.items()
    ]
    return geometric_mean(scores)


def smarts_score(smiles: str, smarts: str, inverse: bool = False) -> float:
    """Returns 1.0 if molecule has (inverse=False) or lacks (inverse=True) the SMARTS match.

    Returns 0.0 for invalid SMILES or unparseable SMARTS.
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return 0.0
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        return 0.0
    has_match = mol.HasSubstructMatch(pattern)
    return 0.0 if (has_match == inverse) else 1.0


# ---------------------------------------------------------------------------
# Reference molecules — computed once at import
# ---------------------------------------------------------------------------

_CELECOXIB = Chem.MolFromSmiles("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F")
_TROGLITAZONE = Chem.MolFromSmiles("Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O")
_THIOTHIXENE = Chem.MolFromSmiles("CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1")

_CELECOXIB_FP4 = _ecfp4(_CELECOXIB)
_TROGLITAZONE_FP4 = _ecfp4(_TROGLITAZONE)
_THIOTHIXENE_FP4 = _ecfp4(_THIOTHIXENE)


# ---------------------------------------------------------------------------
# Rediscovery task scorers
# ---------------------------------------------------------------------------


def celecoxib_rediscovery(smiles: str) -> float:
    """Tanimoto ECFP4 to celecoxib, clipped at 1.0. Score=1.0 for exact match.

    Returns:
        float: Score in [0, 1].
    """
    return clipped_score(_tanimoto(smiles, _CELECOXIB_FP4, _ecfp4), upper=1.0)


def troglitazone_rediscovery(smiles: str) -> float:
    """Tanimoto ECFP4 to troglitazone, clipped at 1.0.

    Returns:
        float: Score in [0, 1].
    """
    return clipped_score(_tanimoto(smiles, _TROGLITAZONE_FP4, _ecfp4), upper=1.0)


def thiothixene_rediscovery(smiles: str) -> float:
    """Tanimoto ECFP4 to thiothixene, clipped at 1.0.

    Returns:
        float: Score in [0, 1].
    """
    return clipped_score(_tanimoto(smiles, _THIOTHIXENE_FP4, _ecfp4), upper=1.0)


_ARIPIPRAZOLE = Chem.MolFromSmiles("Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl")
_ALBUTEROL = Chem.MolFromSmiles("CC(C)(C)NCC(O)c1ccc(O)c(CO)c1")
_MESTRANOL = Chem.MolFromSmiles("COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1")

_ARIPIPRAZOLE_FCFP4 = _fcfp4(_ARIPIPRAZOLE)
_ALBUTEROL_FCFP4 = _fcfp4(_ALBUTEROL)
_MESTRANOL_AP = _ap(_MESTRANOL)

_SIMILARITY_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Similarity task scorers
# ---------------------------------------------------------------------------


def aripiprazole_similarity(smiles: str) -> float:
    """Tanimoto FCFP4 to aripiprazole, clipped at 0.75.

    Returns:
        float: Score in [0, 1].
    """
    return clipped_score(
        _tanimoto(smiles, _ARIPIPRAZOLE_FCFP4, _fcfp4), upper=_SIMILARITY_THRESHOLD
    )


def albuterol_similarity(smiles: str) -> float:
    """Tanimoto FCFP4 to albuterol, clipped at 0.75.

    Returns:
        float: Score in [0, 1].
    """
    return clipped_score(_tanimoto(smiles, _ALBUTEROL_FCFP4, _fcfp4), upper=_SIMILARITY_THRESHOLD)


def mestranol_similarity(smiles: str) -> float:
    """Tanimoto atom-pair to mestranol, clipped at 0.75.

    Returns:
        float: Score in [0, 1].
    """
    return clipped_score(_tanimoto(smiles, _MESTRANOL_AP, _ap), upper=_SIMILARITY_THRESHOLD)


_CAMPHOR = Chem.MolFromSmiles("CC1(C)C2CCC1(C)C(=O)C2")
_MENTHOL = Chem.MolFromSmiles("CC(C)C1CCC(C)CC1O")
_TADALAFIL = Chem.MolFromSmiles("O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C")
_SILDENAFIL = Chem.MolFromSmiles("CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C")

_CAMPHOR_FP4 = _ecfp4(_CAMPHOR)
_MENTHOL_FP4 = _ecfp4(_MENTHOL)
_TADALAFIL_FP6 = _ecfp6(_TADALAFIL)
_SILDENAFIL_FP6 = _ecfp6(_SILDENAFIL)


# ---------------------------------------------------------------------------
# Median molecule task scorers
# ---------------------------------------------------------------------------


def camphor_menthol_median(smiles: str) -> float:
    """Geometric mean of Tanimoto ECFP4 to camphor and menthol.

    Returns:
        float: Score in [0, 1].
    """
    return geometric_mean([
        _tanimoto(smiles, _CAMPHOR_FP4, _ecfp4),
        _tanimoto(smiles, _MENTHOL_FP4, _ecfp4),
    ])


def tadalafil_sildenafil_median(smiles: str) -> float:
    """Geometric mean of Tanimoto ECFP6 to tadalafil and sildenafil.

    Returns:
        float: Score in [0, 1].
    """
    return geometric_mean([
        _tanimoto(smiles, _TADALAFIL_FP6, _ecfp6),
        _tanimoto(smiles, _SILDENAFIL_FP6, _ecfp6),
    ])


_FEXOFENADINE = Chem.MolFromSmiles("CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4")
_OSIMERTINIB = Chem.MolFromSmiles("COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34")
_RANOLAZINE = Chem.MolFromSmiles("COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2")

_FEXOFENADINE_AP = _ap(_FEXOFENADINE)
_OSIMERTINIB_FCFP4 = _fcfp4(_OSIMERTINIB)
_OSIMERTINIB_FP6 = _ecfp6(_OSIMERTINIB)
_RANOLAZINE_AP = _ap(_RANOLAZINE)


def fexofenadine_mpo(smiles: str) -> float:
    """Geometric mean: clipped AP Tanimoto (≤0.8) + high TPSA (μ=90) + low logP (μ=4).

    Returns:
        float: Score in [0, 1].
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return 0.0
    tanimoto_score = clipped_score(_tanimoto(smiles, _FEXOFENADINE_AP, _ap), upper=0.8)
    tpsa_score = max_gaussian_score(float(Descriptors.TPSA(mol)), mu=90.0, sigma=10.0)
    logp_score = min_gaussian_score(float(Descriptors.MolLogP(mol)), mu=4.0, sigma=1.0)
    return geometric_mean([tanimoto_score, tpsa_score, logp_score])


def osimertinib_mpo(smiles: str) -> float:
    """Geometric mean: clipped FCFP4 (≤0.8) + penalised ECFP6 (μ=0.85) + TPSA (μ=100) + logP (μ=1).

    Returns:
        float: Score in [0, 1].
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return 0.0
    tanimoto_fcfp4 = clipped_score(_tanimoto(smiles, _OSIMERTINIB_FCFP4, _fcfp4), upper=0.8)
    tanimoto_fp6 = min_gaussian_score(
        _tanimoto(smiles, _OSIMERTINIB_FP6, _ecfp6), mu=0.85, sigma=0.1
    )
    tpsa_score = max_gaussian_score(float(Descriptors.TPSA(mol)), mu=100.0, sigma=10.0)
    logp_score = min_gaussian_score(float(Descriptors.MolLogP(mol)), mu=1.0, sigma=1.0)
    return geometric_mean([tanimoto_fcfp4, tanimoto_fp6, tpsa_score, logp_score])


def ranolazine_mpo(smiles: str) -> float:
    """Geometric mean: clipped AP Tanimoto (≤0.7) + high logP (μ=7) + high TPSA (μ=95) + 1 F atom.

    Returns:
        float: Score in [0, 1].
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return 0.0
    tanimoto_score = clipped_score(_tanimoto(smiles, _RANOLAZINE_AP, _ap), upper=0.7)
    logp_score = max_gaussian_score(float(Descriptors.MolLogP(mol)), mu=7.0, sigma=1.0)
    tpsa_score = max_gaussian_score(float(Descriptors.TPSA(mol)), mu=95.0, sigma=20.0)
    f_count = float(sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "F"))
    f_score = gaussian_score(f_count, mu=1.0, sigma=1.0)
    return geometric_mean([tanimoto_score, logp_score, tpsa_score, f_score])


_PERINDOPRIL = Chem.MolFromSmiles("O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC")
_AMLODIPINE = Chem.MolFromSmiles(r"Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC")
_SITAGLIPTIN = Chem.MolFromSmiles("Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F")
_ZALEPLON = Chem.MolFromSmiles("O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1")

_PERINDOPRIL_FP4 = _ecfp4(_PERINDOPRIL)
_AMLODIPINE_FP4 = _ecfp4(_AMLODIPINE)
_SITAGLIPTIN_FP4 = _ecfp4(_SITAGLIPTIN)
_ZALEPLON_FP4 = _ecfp4(_ZALEPLON)

_SITAGLIPTIN_LOGP = float(Descriptors.MolLogP(_SITAGLIPTIN))
_SITAGLIPTIN_TPSA = float(Descriptors.TPSA(_SITAGLIPTIN))
_SITAGLIPTIN_FORMULA = _parse_formula(rdMolDescriptors.CalcMolFormula(_SITAGLIPTIN))

_ZALEPLON_FORMULA = _parse_formula("C19H17N3O2")


def perindopril_mpo(smiles: str) -> float:
    """Geometric mean: ECFP4 Tanimoto to perindopril + exactly 2 aromatic rings (μ=2, σ=0.5).

    Returns:
        float: Score in [0, 1].
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return 0.0
    tanimoto_score = _tanimoto(smiles, _PERINDOPRIL_FP4, _ecfp4)
    ring_score = gaussian_score(
        float(rdMolDescriptors.CalcNumAromaticRings(mol)), mu=2.0, sigma=0.5
    )
    return geometric_mean([tanimoto_score, ring_score])


def amlodipine_mpo(smiles: str) -> float:
    """Geometric mean: ECFP4 Tanimoto to amlodipine + exactly 3 rings total (μ=3, σ=0.5).

    Returns:
        float: Score in [0, 1].
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return 0.0
    tanimoto_score = _tanimoto(smiles, _AMLODIPINE_FP4, _ecfp4)
    ring_score = gaussian_score(float(rdMolDescriptors.CalcNumRings(mol)), mu=3.0, sigma=0.5)
    return geometric_mean([tanimoto_score, ring_score])


def sitagliptin_mpo(smiles: str) -> float:
    """Geometric mean: dissimilarity to sitagliptin + logP/TPSA match + formula match.

    Returns:
        float: Score in [0, 1].
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return 0.0
    dissim_score = gaussian_score(_tanimoto(smiles, _SITAGLIPTIN_FP4, _ecfp4), mu=0.0, sigma=0.1)
    logp_score = gaussian_score(float(Descriptors.MolLogP(mol)), mu=_SITAGLIPTIN_LOGP, sigma=0.2)
    tpsa_score = gaussian_score(float(Descriptors.TPSA(mol)), mu=_SITAGLIPTIN_TPSA, sigma=5.0)
    isomer_s = isomer_score(smiles, _SITAGLIPTIN_FORMULA)
    return geometric_mean([dissim_score, logp_score, tpsa_score, isomer_s])


def zaleplon_mpo(smiles: str) -> float:
    """Geometric mean: ECFP4 Tanimoto to zaleplon + formula C19H17N3O2.

    Returns:
        float: Score in [0, 1].
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return 0.0
    tanimoto_score = _tanimoto(smiles, _ZALEPLON_FP4, _ecfp4)
    isomer_s = isomer_score(smiles, _ZALEPLON_FORMULA)
    return geometric_mean([tanimoto_score, isomer_s])


_C7H8N2O2_FORMULA = _parse_formula("C7H8N2O2")
_C9H10N2O2PF2CL_FORMULA = _parse_formula("C9H10N2O2PF2Cl")


def c7h8n2o2_isomer(smiles: str) -> float:
    """Geometric mean of per-element Gaussian scores targeting formula C7H8N2O2.

    Returns:
        float: Score in [0, 1].
    """
    return isomer_score(smiles, _C7H8N2O2_FORMULA)


def c9h10n2o2pf2cl_isomer(smiles: str) -> float:
    """Geometric mean of per-element Gaussian scores targeting formula C9H10N2O2PF2Cl.

    Returns:
        float: Score in [0, 1].
    """
    return isomer_score(smiles, _C9H10N2O2PF2CL_FORMULA)


_HOP_REF = Chem.MolFromSmiles("CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C")
_HOP_REF_PHCO = _phco(_HOP_REF)

_SCAFFOLD_HOP_SMARTS_KEEP = "[#6]-[#6]-[#6]-[#8]-[#6]~[#6]~[#6]~[#6]~[#6]-[#7]-c1ccc2ncsc2c1"
_SCAFFOLD_HOP_SMARTS_REMOVE = "[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12"

_DECORATOR_HOP_SMARTS_REMOVE_SULFONYL = "CS([#6])(=O)=O"
_DECORATOR_HOP_SMARTS_REMOVE_THIENOPYRIDINE = "[#7]-c1ccc2ncsc2c1"
_DECORATOR_HOP_SMARTS_KEEP_PURINONE = "[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12"


def aripiprazole_scaffold_hop(smiles: str) -> float:
    """Arithmetic mean: PHCO Tanimoto (<=0.75) + keep propoxy-thienopyridine + remove
    aminopyrimidine scaffold.

    Returns:
        float: Score in [0, 1].
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return 0.0
    phco_score = clipped_score(_tanimoto(smiles, _HOP_REF_PHCO, _phco), upper=0.75)
    keep_score = smarts_score(smiles, _SCAFFOLD_HOP_SMARTS_KEEP, inverse=False)
    remove_score = smarts_score(smiles, _SCAFFOLD_HOP_SMARTS_REMOVE, inverse=True)
    return arithmetic_mean([phco_score, keep_score, remove_score])


def aripiprazole_decorator_hop(smiles: str) -> float:
    """Arithmetic mean: PHCO (<=0.85) + remove sulfonyl + remove thienopyridine + keep purinone.

    Returns:
        float: Score in [0, 1].
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return 0.0
    phco_score = clipped_score(_tanimoto(smiles, _HOP_REF_PHCO, _phco), upper=0.85)
    rm_sulfonyl = smarts_score(smiles, _DECORATOR_HOP_SMARTS_REMOVE_SULFONYL, inverse=True)
    rm_thienopyridine = smarts_score(
        smiles, _DECORATOR_HOP_SMARTS_REMOVE_THIENOPYRIDINE, inverse=True
    )
    keep_purinone = smarts_score(smiles, _DECORATOR_HOP_SMARTS_KEEP_PURINONE, inverse=False)
    return arithmetic_mean([phco_score, rm_sulfonyl, rm_thienopyridine, keep_purinone])


# ---------------------------------------------------------------------------
# Task scorer dispatch
# ---------------------------------------------------------------------------

_TASK_SCORERS: dict[str, Callable[[str], float]] = {
    "celecoxib_rediscovery": celecoxib_rediscovery,
    "troglitazone_rediscovery": troglitazone_rediscovery,
    "thiothixene_rediscovery": thiothixene_rediscovery,
    "aripiprazole_similarity": aripiprazole_similarity,
    "albuterol_similarity": albuterol_similarity,
    "mestranol_similarity": mestranol_similarity,
    "camphor_menthol_median": camphor_menthol_median,
    "tadalafil_sildenafil_median": tadalafil_sildenafil_median,
    "fexofenadine_mpo": fexofenadine_mpo,
    "osimertinib_mpo": osimertinib_mpo,
    "ranolazine_mpo": ranolazine_mpo,
    "perindopril_mpo": perindopril_mpo,
    "amlodipine_mpo": amlodipine_mpo,
    "sitagliptin_mpo": sitagliptin_mpo,
    "zaleplon_mpo": zaleplon_mpo,
    "c7h8n2o2_isomer": c7h8n2o2_isomer,
    "c9h10n2o2pf2cl_isomer": c9h10n2o2pf2cl_isomer,
    "aripiprazole_scaffold_hop": aripiprazole_scaffold_hop,
    "aripiprazole_decorator_hop": aripiprazole_decorator_hop,
}


def get_task_scorer(task_name: str) -> Callable[[str], float]:
    """Return the scoring function for the given GuacaMol benchmark task name.

    Args:
        task_name: One of the GuacaMolTaskName literal values.

    Returns:
        A callable ``(smiles: str) -> float`` in [0, 1].

    Raises:
        KeyError: If task_name is not a recognised benchmark task.
    """
    return _TASK_SCORERS[task_name]


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
        if _mol_from_smiles(smiles) is None:
            logger.warning("Skipping invalid SMILES: %r", smiles)
            continue
        candidates.append(Candidate(data=smiles, modality=modality, features={}))
        labels.append(scorer(smiles))
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
            split_lc = _label_smiles(smiles_list, properties, target, self.modality)
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

        Stores the three splits in ``self._paper_splits``.

        Returns:
            LabelledCandidates: All scored candidates across train/valid/test splits.
        """
        split_files = {k: v for k, v in GUACAMOL_FILES.items() if k != "ALL"}
        tag_to_key = {"TRAIN": "train", "VALID": "validation", "TEST": "test"}
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
            split_lc = _label_smiles_benchmark(smiles_list, scorer, self.modality)
            self._paper_splits[tag_to_key[tag]] = split_lc
            all_candidates.extend(split_lc.candidates)
            all_labels.extend(split_lc.labels.tolist())
        return LabelledCandidates(
            candidates=all_candidates, labels=np.array(all_labels, dtype=float)
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
