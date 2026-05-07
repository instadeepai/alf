# Ensemble MLP Notebook Design

**Date:** 2026-05-07  
**Output:** `tutorials/models/ensemble_tutorial.ipynb`  
**Audience:** Progressive — accessible intro that advances to API comparison

---

## Goal

Showcase `EnsembleWrapper` with `MLPModel` on the GFP dataset across three uncertainty-estimation modes: seed-only, MC dropout-only, and combined seed+dropout. Runs end-to-end on CPU with auto-scaled hyperparameters.

---

## Key Files

| File | Role |
|------|------|
| `tools/alf_tools/models/ensemble.py` | `EnsembleWrapper`, `EnsembleWrapperConfig` |
| `tools/alf_tools/models/mlp.py` | `MLPModel`, `MLPModelConfig`, `MLPTrainConfig` |
| `tools/alf_tools/datasets/gfp.py` | GFP dataset (1000 rows, sequence → brightness) |

---

## Structure (Approach A — linear narrative)

### Section 1 — Setup & Configuration

Single config cell containing all hyperparameters. Device detection sets `FAST_MODE` automatically when no GPU is found.

```python
DEVICE      = "cuda" | "mps" | "cpu"   # auto-detected
FAST_MODE   = DEVICE == "cpu"

N_MEMBERS   = 10
N_EPOCHS    = 25 if FAST_MODE else 50
N_MC_PASSES = 10 if FAST_MODE else 20
BASE_SEED   = 42
DROPOUT_P   = 0.1
```

Prints a banner: device, mode, effective hyperparameters.

---

### Section 2 — Data Loading & Featurisation

- Load GFP via `GFP(BaseDatasetConfig())` (auto-downloads to `datasets/data/`)
- 80/20 train/val split, fixed seed
- Print split sizes, label range, mean brightness
- One-hot encode nucleotide sequences → wrap as `TABULAR` candidates (required by `MLPModel.featurise()`)
- Brief markdown: what GFP is, why it's useful for calibration benchmarks

---

### Section 3 — Seed Ensemble (Deep Ensemble)

**Config:** `EnsembleWrapperConfig(base_seed=BASE_SEED, n_members=N_MEMBERS)` + `MLPModelConfig(n_mc_passes=0)` (deterministic predictions)

- Per-member training progress printed
- Plots (consistent axis scales across Sections 3–5):
  1. Mean vs. ground truth scatter, colour = predicted std
  2. Violin plot of member predictions for 8 sampled val candidates
  3. Histogram of predicted std across val set
- Summary metrics: Spearman ρ, MSE, mean std, wall-clock time

---

### Section 4 — MC Dropout Ensemble

**Config:** `EnsembleWrapperConfig(base_seed=BASE_SEED, n_members=1)` + `MLPModelConfig(dropout=DROPOUT_P, n_mc_passes=N_MC_PASSES, model_seed=BASE_SEED)`

- Note on `dropout_seed=None` → intentional stochasticity at inference
- Same three plots as Section 3 (same axes)
- Additional plot:
  4. Reproducibility check — two `predict()` calls overlaid to show stochastic spread
- Same summary metrics row

---

### Section 5 — Combined Ensemble (Seed + Dropout)

**Config:** 10 members, each with `dropout=DROPOUT_P, n_mc_passes=N_MC_PASSES`; total samples = N_MEMBERS × N_MC_PASSES

- Same four plots as Section 4 (same axes)
- Additional plot:
  5. Member diversity breakdown — per-member KDE of MC passes for one candidate, showing between-member vs. within-member spread
- Same summary metrics row

---

### Section 6 — Comparison & Summary

**Metrics table** (`pandas` DataFrame, one row per mode):

| Mode | Spearman ρ | MSE | Mean std | Std of std | Train time |
|------|-----------|-----|----------|------------|------------|

Best value per column highlighted.

**3×3 plot grid:**
- Row 1: Mean vs. ground truth scatter, all three modes
- Row 2: Std KDEs overlaid on single axes
- Row 3: Calibration curves (expected confidence vs. observed frequency) + perfect-calibration diagonal

**Closing markdown (3–4 sentences):** when to prefer each mode; pointer to `EnsembleWrapperConfig` API.

---

## Notebook Style

- Concise markdown cells — one short paragraph max per section intro
- No redundant commentary; code is self-explanatory via variable names
- All plots use consistent axis scales within each plot type across sections
- No intermediate saves or checkpoints — runs top-to-bottom in one pass

---

## CPU Runtime Estimate (FAST_MODE)

| Mode | Members | Epochs | MC Passes | Approx. time (CPU) |
|------|---------|--------|-----------|---------------------|
| Seed | 10 | 25 | 0 | ~3–4 min |
| Dropout | 1 | 25 | 10 | <1 min |
| Combined | 10 | 25 | 10 | ~3–4 min |

Total: ~8 min on CPU in FAST_MODE.
