# GuacaMol Dataset Tutorial — Design Spec

**Date:** 2026-05-11  
**Output file:** `tutorials/datasets/guacamol_tutorial.ipynb`  
**Approach:** ALF + pandas hybrid (Option B)

---

## Goals

Demonstrate the `GuacaMol` dataset class from `alf_tools` in a self-contained Jupyter notebook. Covers: environment setup, downloading, dataset statistics, property visualisations, querying the API, and a concrete aspirin case study with molecular structure rendering.

## Constraints

- Runs quickly on CPU: `max_molecules=10_000` throughout.
- Uses `seaborn` for all statistical visualisations.
- Placed in a new `tutorials/datasets/` directory.
- Follows existing tutorial conventions (uv setup cell, markdown narrative, alf_tools imports).

---

## Notebook Structure

### Section 0 — Setup
- Markdown: brief intro to GuacaMol (Brown et al. 2019, ChEMBL-derived, 1.6 M drug-like SMILES, 10 RDKit physicochemical properties).
- Install cell: `!uv pip install -e . --python {sys.executable}`
- Import block: `alf_tools.datasets.guacamol`, `rdkit`, `pandas`, `seaborn`, `matplotlib`.

### Section 1 — Download
- Call `download_guacamol(data_dir=..., max_lines=10_000)`.
- Print downloaded file names and sizes.
- Show 5 raw SMILES strings from the file.

### Section 2 — Load & Build DataFrame
- Instantiate `GuacaMolConfig(target_property="QED", max_molecules=10_000)` and `GuacaMol(config)`.
- Convert `_raw_dataset.candidates` → `pd.DataFrame` with columns: `smiles` + all 10 property names.
- Show `.shape` and `.describe()`.

### Section 3 — Dataset Statistics
- `sns.histplot` of SMILES string length (with KDE).
- 2×5 subplot grid: `sns.histplot` (with KDE) for each of the 10 RDKit properties:
  `BertzCT`, `MolLogP`, `MolWt`, `TPSA`, `NumHAcceptors`, `NumHDonors`,
  `NumRotatableBonds`, `NumAliphaticRings`, `NumAromaticRings`, `QED`.
- `sns.heatmap` of the 10×10 property correlation matrix.

### Section 4 — Querying
- Demonstrate `dataset.query()` with 3–5 example SMILES (aspirin, ibuprofen, caffeine).
- Show that the return type is `LabelledCandidates`; extract label and feature dict.
- Print a small DataFrame of query results.

### Section 5 — Aspirin Case Study
- Draw aspirin's 2D molecular structure inline using `rdkit.Chem.Draw.MolToImage` displayed via `IPython.display`.
- Query all 10 properties for aspirin's SMILES (`CC(=O)Oc1ccccc1C(=O)O`).
- `sns.barplot` comparing aspirin's raw property values vs. dataset mean (with ± std error bars).
- Re-plot each of the 10 property histograms (from Section 3) in a 2×5 grid with a vertical line marking aspirin's value.

---

## Key Implementation Notes

- Use `GuacaMolConfig.data_dir` pointing to a local temp path (e.g. `Path.home() / ".cache" / "alf"`) so repeated runs skip the download.
- Extract properties from `candidate.features` dict when building the DataFrame.
- For the aspirin overlay plot, share the histogram data (the DataFrame) from Section 3 — no need to reload.
- `MolToImage` returns a PIL image; use `plt.imshow` or `IPython.display.display` to show it inline.
- Normalise property values for the comparison barplot (z-score vs. dataset mean/std) to make all 10 fit on one axis; show raw values in a separate table.
- **Aspirin multi-property lookup**: `dataset.query()` only computes `target_property` as the label for novel SMILES (features dict is not populated for out-of-corpus molecules). For the full 10-property aspirin profile, call `_compute_properties(smiles, list(ALL_PROPERTIES))` directly from `alf_tools.datasets.guacamol`.

---

## Dependencies

- `alf_core`, `alf_tools` (editable installs, already in `tutorials/pyproject.toml`)
- `rdkit` (via `alf_tools` dependency)
- `matplotlib`, `pandas`, `ipykernel` (already in `tutorials/pyproject.toml`)
- `seaborn` — **must be added** to `tutorials/pyproject.toml`
