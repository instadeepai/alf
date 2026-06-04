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

import math
import re
from functools import lru_cache
from typing import Any, Callable

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

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


# ---------------------------------------------------------------------------
# Molecule and fingerprint helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=65536)
def _mol_from_smiles(smiles: str) -> "Chem.Mol | None":
    """Return the RDKit Mol for smiles, or None if invalid. Result is cached per unique string."""
    return Chem.MolFromSmiles(smiles)


def _ecfp4(mol: "Chem.Mol") -> Any:
    """Morgan fingerprint radius=2 (ECFP4).

    Returns:
        RDKit Morgan fingerprint object.
    """
    return AllChem.GetMorganFingerprint(mol, 2)


def _ecfp6(mol: "Chem.Mol") -> Any:
    """Morgan fingerprint radius=3 (ECFP6).

    Returns:
        RDKit Morgan fingerprint object.
    """
    return AllChem.GetMorganFingerprint(mol, 3)


def _fcfp4(mol: "Chem.Mol") -> Any:
    """Feature-based Morgan fingerprint radius=2 (FCFP4).

    Returns:
        RDKit Morgan fingerprint object using pharmacophoric features.
    """
    return AllChem.GetMorganFingerprint(mol, 2, useFeatures=True)


def _ap(mol: "Chem.Mol") -> Any:
    """Atom-pair fingerprint with maxLength=10, matching the original GuacaMol implementation.

    Returns:
        RDKit atom-pair fingerprint object.
    """
    return rdMolDescriptors.GetAtomPairFingerprint(mol, maxLength=10)


def _phco(mol: "Chem.Mol") -> Any:
    """2D pharmacophore fingerprint (Gobbi).

    Returns:
        RDKit 2D pharmacophore fingerprint object.
    """
    from rdkit.Chem.Pharm2D import (  # noqa: PLC0415
        Generate,
        Gobbi_Pharm2D,
    )

    return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)


def _tanimoto(smiles: str, ref_fp: Any, fp_fn: Callable[["Chem.Mol"], Any]) -> float:
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
    # Total-atom-count term (H included) with sigma=2, per original IsomerScoringFunction.
    total_mol_atoms = float(Chem.AddHs(mol).GetNumAtoms())
    total_target = float(sum(target_formula.values()))
    scores.append(gaussian_score(total_mol_atoms, mu=total_target, sigma=2.0))
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
        A callable `(smiles: str) -> float` in [0, 1].

    Raises:
        KeyError: If task_name is not a recognised benchmark task.
    """
    return _TASK_SCORERS[task_name]
