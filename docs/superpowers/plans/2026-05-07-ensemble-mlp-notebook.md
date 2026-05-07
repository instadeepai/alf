# Ensemble MLP Tutorial Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `tutorials/models/ensemble_tutorial.ipynb` showcasing seed, MC dropout, and combined ensemble modes with `EnsembleWrapper` + `MLPModel` on GFP brightness data.

**Architecture:** Single self-contained notebook with 6 sections following a linear narrative. Sections 3–5 each train one ensemble variant, run `predict()`, and plot immediately. A shared helpers cell provides reusable metrics, calibration, and plotting functions. Section 6 compares all three modes with a styled metrics table and a 3×3 comparison grid. CPU fallback auto-reduces `N_EPOCHS` (50→25) and `N_MC_PASSES` (20→10).

**Tech Stack:** `alf_tools` (EnsembleWrapper, MLPModel, GFP), `torch`, `numpy`, `scipy.stats`, `matplotlib`, `pandas`, `nbformat` (notebook generation only)

---

## File Overview

| File | Action |
|------|--------|
| `tutorials/models/ensemble_tutorial.ipynb` | **Create** — ~26-cell notebook |

No existing files are modified.

---

### Task 1: Create notebook scaffold with title, imports, and config

**Files:**
- Create: `tutorials/models/ensemble_tutorial.ipynb`

- [ ] **Step 1: Run this script from the repo root to create the file**

```bash
cd /Users/o.gallup/Code/alf && python3 - <<'PYEOF'
import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})

md  = nbformat.v4.new_markdown_cell
src = nbformat.v4.new_code_cell

nb.cells = [
    md(
        "# Ensemble MLP Tutorial\n\n"
        "Three uncertainty-aware surrogate modes on GFP brightness prediction:\n\n"
        "- **Seed ensemble** — N independently-seeded networks\n"
        "- **MC dropout** — one network, T stochastic inference passes\n"
        "- **Combined** — N networks × T passes\n\n"
        "Each section trains one variant, predicts, and plots. Section 6 compares all three."
    ),
    src("""\
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
import torch

from alf_core import BaseDatasetConfig, Candidate, LabelledCandidates, Modality
from alf_tools.datasets.gfp import GFP
from alf_tools.models.mlp import MLPModel, MLPModelConfig, MLPTrainConfig
from alf_tools.models.ensemble import EnsembleWrapper, EnsembleWrapperConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")"""),
    md("## 1. Setup"),
    src("""\
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
FAST_MODE   = DEVICE == "cpu"
N_MEMBERS   = 10
N_EPOCHS    = 25 if FAST_MODE else 50
N_MC_PASSES = 10 if FAST_MODE else 20
BASE_SEED   = 42
DROPOUT_P   = 0.1
HIDDEN_DIMS = [128, 64]

print(f"Device : {DEVICE}")
print(f"Mode   : {'FAST (CPU)' if FAST_MODE else 'FULL (GPU)'}")
print(f"N_MEMBERS={N_MEMBERS}  N_EPOCHS={N_EPOCHS}  N_MC_PASSES={N_MC_PASSES}")"""),
]

with open("tutorials/models/ensemble_tutorial.ipynb", "w") as f:
    nbformat.write(nb, f)
print("Done: tutorials/models/ensemble_tutorial.ipynb")
PYEOF
```

Expected output: `Done: tutorials/models/ensemble_tutorial.ipynb`

- [ ] **Step 2: Commit**

```bash
cd /Users/o.gallup/Code/alf
git add tutorials/models/ensemble_tutorial.ipynb
git commit -m "feat: scaffold ensemble tutorial notebook with imports and config"
```

---

### Task 2: Add GFP data loading and featurisation cells

**Files:**
- Modify: `tutorials/models/ensemble_tutorial.ipynb`

- [ ] **Step 1: Append data cells**

```bash
cd /Users/o.gallup/Code/alf && python3 - <<'PYEOF'
import nbformat

with open("tutorials/models/ensemble_tutorial.ipynb") as f:
    nb = nbformat.read(f, as_version=4)

md  = nbformat.v4.new_markdown_cell
src = nbformat.v4.new_code_cell

nb.cells += [
    md(
        "## 2. GFP Dataset\n\n"
        "GFP (green fluorescent protein) is a standard benchmark for sequence-fitness modelling. "
        "Each of the 1 000 variants is a 237-nt nucleotide sequence; the label is median brightness. "
        "Sequences cluster around wild-type, making calibration informative: "
        "a well-calibrated model should be more uncertain on distant variants."
    ),
    src("""\
gfp = GFP(BaseDatasetConfig())
labelled = gfp.load_dataset()

rng = np.random.default_rng(BASE_SEED)
indices = rng.permutation(len(labelled))
n_train = int(0.8 * len(labelled))
train_idx, val_idx = indices[:n_train], indices[n_train:]
train_raw_cands, train_labels = labelled[train_idx]
val_raw_cands,   val_labels   = labelled[val_idx]

print(f"Sequences : {len(labelled)}")
print(f"Train : {len(train_labels)} | Val : {len(val_labels)}")
print(f"Brightness range : [{labelled.labels.min():.3f}, {labelled.labels.max():.3f}]  mean={labelled.labels.mean():.3f}")"""),
    src("""\
# MLPModel only accepts TABULAR/EMBEDDING modality; one-hot-encode the nucleotide sequences.
NUCLEOTIDES = list("ACGT")
SEQ_LEN     = len(labelled.candidates[0].data)
INPUT_DIM   = SEQ_LEN * 4

def one_hot_encode(seq: str) -> np.ndarray:
    arr = np.zeros(len(seq) * 4, dtype=np.float32)
    for i, nuc in enumerate(seq):
        arr[i * 4 + NUCLEOTIDES.index(nuc)] = 1.0
    return arr

def to_tabular(raw_cands: list) -> list:
    return [Candidate(data=one_hot_encode(c.data), modality=Modality.TABULAR) for c in raw_cands]

train_cands = to_tabular(train_raw_cands)
val_cands   = to_tabular(val_raw_cands)
train_data  = LabelledCandidates(candidates=train_cands, labels=train_labels)
val_data    = LabelledCandidates(candidates=val_cands,   labels=val_labels)

print(f"Seq len: {SEQ_LEN} | Input dim: {INPUT_DIM}")"""),
]

with open("tutorials/models/ensemble_tutorial.ipynb", "w") as f:
    nbformat.write(nb, f)
print("Done")
PYEOF
```

Expected output: `Done`

- [ ] **Step 2: Commit**

```bash
cd /Users/o.gallup/Code/alf
git add tutorials/models/ensemble_tutorial.ipynb
git commit -m "feat: add GFP data loading and one-hot featurisation cells"
```

---

### Task 3: Add shared helpers cell

**Files:**
- Modify: `tutorials/models/ensemble_tutorial.ipynb`

- [ ] **Step 1: Append helpers cell**

```bash
cd /Users/o.gallup/Code/alf && python3 - <<'PYEOF'
import nbformat

with open("tutorials/models/ensemble_tutorial.ipynb") as f:
    nb = nbformat.read(f, as_version=4)

nb.cells.append(nbformat.v4.new_code_cell("""\
# Shared constants and helpers used across all three ensemble sections.
LABEL_LIM  = (float(labelled.labels.min()) - 0.05, float(labelled.labels.max()) + 0.05)
VIOLIN_IDX = np.linspace(0, len(val_cands) - 1, 8, dtype=int)
metrics_table: list[dict] = []


def section_metrics(preds, true_labels, elapsed: float) -> dict:
    rho, _ = spearmanr(preds.means, true_labels)
    mse    = float(np.mean((preds.means - true_labels) ** 2))
    std    = np.sqrt(preds.variances)
    return {
        "Spearman rho": round(float(rho), 4),
        "MSE":          round(mse, 4),
        "Mean std":     round(float(std.mean()), 4),
        "Std of std":   round(float(std.std()), 4),
        "Train time s": round(elapsed, 1),
    }


def calibration_curve(empirical_dist, true_labels, n_bins: int = 10):
    alphas, observed = np.linspace(0.1, 0.9, n_bins), []
    for alpha in alphas:
        lo = np.quantile(empirical_dist, (1 - alpha) / 2, axis=1)
        hi = np.quantile(empirical_dist, (1 + alpha) / 2, axis=1)
        observed.append(float(np.mean((true_labels >= lo) & (true_labels <= hi))))
    return alphas, np.array(observed)


def plot_section(preds, true_labels, title: str) -> None:
    std = np.sqrt(preds.variances)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    sc = axes[0].scatter(preds.means, true_labels, c=std, cmap="viridis", alpha=0.6, s=15)
    plt.colorbar(sc, ax=axes[0], label="std")
    axes[0].plot(LABEL_LIM, LABEL_LIM, "r--", lw=1)
    axes[0].set(xlim=LABEL_LIM, ylim=LABEL_LIM,
                xlabel="Predicted mean", ylabel="True brightness",
                title="Mean vs ground truth")

    data_v = [preds.empirical_dist[i] for i in VIOLIN_IDX]
    axes[1].violinplot(data_v, positions=range(8), showmedians=True)
    axes[1].scatter(range(8), true_labels[VIOLIN_IDX], color="red", zorder=3, s=30, label="True")
    axes[1].set(xlabel="Val index", ylabel="Prediction", title="Member spread (8 samples)")
    axes[1].legend(fontsize=8)

    axes[2].hist(std, bins=30, edgecolor="white", color="steelblue")
    axes[2].set(xlabel="Predicted std", ylabel="Count", title="Uncertainty distribution")

    plt.tight_layout()
    plt.show()"""))

with open("tutorials/models/ensemble_tutorial.ipynb", "w") as f:
    nbformat.write(nb, f)
print("Done")
PYEOF
```

Expected output: `Done`

- [ ] **Step 2: Commit**

```bash
cd /Users/o.gallup/Code/alf
git add tutorials/models/ensemble_tutorial.ipynb
git commit -m "feat: add shared helpers (metrics, calibration, plot_section)"
```

---

### Task 4: Add Section 3 — Seed Ensemble

**Files:**
- Modify: `tutorials/models/ensemble_tutorial.ipynb`

- [ ] **Step 1: Append seed ensemble cells**

```bash
cd /Users/o.gallup/Code/alf && python3 - <<'PYEOF'
import nbformat

with open("tutorials/models/ensemble_tutorial.ipynb") as f:
    nb = nbformat.read(f, as_version=4)

md  = nbformat.v4.new_markdown_cell
src = nbformat.v4.new_code_cell

nb.cells += [
    md(
        "## 3. Seed Ensemble\n\n"
        "N networks are trained from different random initialisations. "
        "Diversity comes entirely from weight-space randomness — no stochasticity at inference. "
        "`EnsembleWrapperConfig(base_seed, n_members)` derives seeds as `[base_seed, base_seed+1, …]`."
    ),
    src("""\
t0 = time.time()

def seed_factory(seed: int) -> MLPModel:
    return MLPModel(
        model_config=MLPModelConfig(hidden_dims=HIDDEN_DIMS, n_mc_passes=0, model_seed=seed),
        train_config=MLPTrainConfig(num_epochs=N_EPOCHS),
        device=DEVICE,
    )

seed_ensemble = EnsembleWrapper(
    model_factory=seed_factory,
    config=EnsembleWrapperConfig(base_seed=BASE_SEED, n_members=N_MEMBERS),
    name="seed_ensemble",
)
seed_ensemble.train(train_data, val_data)
seed_time = time.time() - t0
print(f"Training time: {seed_time:.1f}s")"""),
    src("""\
seed_preds = seed_ensemble.predict(val_cands)
plot_section(seed_preds, val_labels, "Seed Ensemble")
seed_m = section_metrics(seed_preds, val_labels, seed_time)
metrics_table.append({"Mode": "Seed ensemble", **seed_m})
print(pd.Series(seed_m).to_string())"""),
]

with open("tutorials/models/ensemble_tutorial.ipynb", "w") as f:
    nbformat.write(nb, f)
print("Done")
PYEOF
```

Expected output: `Done`

- [ ] **Step 2: Commit**

```bash
cd /Users/o.gallup/Code/alf
git add tutorials/models/ensemble_tutorial.ipynb
git commit -m "feat: add seed ensemble section (Section 3)"
```

---

### Task 5: Add Section 4 — MC Dropout Ensemble

**Files:**
- Modify: `tutorials/models/ensemble_tutorial.ipynb`

- [ ] **Step 1: Append MC dropout cells**

```bash
cd /Users/o.gallup/Code/alf && python3 - <<'PYEOF'
import nbformat

with open("tutorials/models/ensemble_tutorial.ipynb") as f:
    nb = nbformat.read(f, as_version=4)

md  = nbformat.v4.new_markdown_cell
src = nbformat.v4.new_code_cell

nb.cells += [
    md(
        "## 4. MC Dropout Ensemble\n\n"
        "A single network is sampled T times at inference with dropout active. "
        "Training cost is 1× — the same as a plain MLP. "
        "`dropout_seed` is unset here, so the default `model_seed` is used: "
        "predictions are reproducible across calls."
    ),
    src("""\
t0 = time.time()

def dropout_factory(seed: int) -> MLPModel:
    return MLPModel(
        model_config=MLPModelConfig(
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT_P,
            n_mc_passes=N_MC_PASSES,
            model_seed=seed,
        ),
        train_config=MLPTrainConfig(num_epochs=N_EPOCHS),
        device=DEVICE,
    )

dropout_ensemble = EnsembleWrapper(
    model_factory=dropout_factory,
    config=EnsembleWrapperConfig(base_seed=BASE_SEED, n_members=1),
    name="dropout_ensemble",
)
dropout_ensemble.train(train_data, val_data)
dropout_time = time.time() - t0
print(f"Training time: {dropout_time:.1f}s")"""),
    src("""\
dropout_preds = dropout_ensemble.predict(val_cands)
plot_section(dropout_preds, val_labels, "MC Dropout Ensemble")"""),
    src("""\
# Reproducibility check: two calls with the same model_seed return identical predictions.
preds_a = dropout_ensemble.predict(val_cands)
preds_b = dropout_ensemble.predict(val_cands)

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(preds_a.means, preds_b.means, alpha=0.6, s=15, color="steelblue")
ax.plot(LABEL_LIM, LABEL_LIM, "r--", lw=1, label="y = x (perfect reproducibility)")
ax.set(xlim=LABEL_LIM, ylim=LABEL_LIM,
       xlabel="predict() call 1", ylabel="predict() call 2",
       title="MC dropout: reproducibility check")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()"""),
    src("""\
dropout_m = section_metrics(dropout_preds, val_labels, dropout_time)
metrics_table.append({"Mode": "MC Dropout", **dropout_m})
print(pd.Series(dropout_m).to_string())"""),
]

with open("tutorials/models/ensemble_tutorial.ipynb", "w") as f:
    nbformat.write(nb, f)
print("Done")
PYEOF
```

Expected output: `Done`

- [ ] **Step 2: Commit**

```bash
cd /Users/o.gallup/Code/alf
git add tutorials/models/ensemble_tutorial.ipynb
git commit -m "feat: add MC dropout section (Section 4)"
```

---

### Task 6: Add Section 5 — Combined Ensemble

**Files:**
- Modify: `tutorials/models/ensemble_tutorial.ipynb`

- [ ] **Step 1: Append combined ensemble cells**

```bash
cd /Users/o.gallup/Code/alf && python3 - <<'PYEOF'
import nbformat

with open("tutorials/models/ensemble_tutorial.ipynb") as f:
    nb = nbformat.read(f, as_version=4)

md  = nbformat.v4.new_markdown_cell
src = nbformat.v4.new_code_cell

nb.cells += [
    md(
        "## 5. Combined Ensemble (Seed + Dropout)\n\n"
        "Each of N independently-seeded networks performs T MC dropout passes at inference. "
        "Total samples per candidate = N × T. "
        "The `empirical_dist` captures both between-member (seed) and within-member (dropout) variance."
    ),
    src("""\
t0 = time.time()

# Same per-member config as MC Dropout; N_MEMBERS (vs 1) drives the combined diversity.
def combined_factory(seed: int) -> MLPModel:
    return MLPModel(
        model_config=MLPModelConfig(
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT_P,
            n_mc_passes=N_MC_PASSES,
            model_seed=seed,
        ),
        train_config=MLPTrainConfig(num_epochs=N_EPOCHS),
        device=DEVICE,
    )

combined_ensemble = EnsembleWrapper(
    model_factory=combined_factory,
    config=EnsembleWrapperConfig(base_seed=BASE_SEED, n_members=N_MEMBERS),
    name="combined_ensemble",
)
combined_ensemble.train(train_data, val_data)
combined_time = time.time() - t0
print(f"Training time: {combined_time:.1f}s")
print(f"Total samples per candidate: {N_MEMBERS} × {N_MC_PASSES} = {N_MEMBERS * N_MC_PASSES}")"""),
    src("""\
combined_preds = combined_ensemble.predict(val_cands)
plot_section(combined_preds, val_labels, "Combined Ensemble (Seed + Dropout)")"""),
    src("""\
# Member diversity breakdown: per-member KDE for one candidate.
# Between-member spread = seed diversity; within-member spread = dropout diversity.
cand_idx = VIOLIN_IDX[4]
n_mc = N_MC_PASSES

fig, ax = plt.subplots(figsize=(10, 4))
for i in range(N_MEMBERS):
    samples = combined_preds.empirical_dist[cand_idx, i * n_mc : (i + 1) * n_mc]
    xs  = np.linspace(samples.min() - 0.05, samples.max() + 0.05, 200)
    bw  = max(1.06 * samples.std() * n_mc ** -0.2, 1e-4)
    kde = np.mean(
        np.exp(-0.5 * ((xs[:, None] - samples[None, :]) / bw) ** 2) / (bw * np.sqrt(2 * np.pi)),
        axis=1,
    )
    ax.plot(xs, kde, alpha=0.7, label=f"seed {BASE_SEED + i}")

ax.set(xlabel="Prediction", ylabel="Density",
       title=f"Member diversity (candidate {cand_idx}): within=dropout, between=seed")
ax.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()"""),
    src("""\
combined_m = section_metrics(combined_preds, val_labels, combined_time)
metrics_table.append({"Mode": "Combined", **combined_m})
print(pd.Series(combined_m).to_string())"""),
]

with open("tutorials/models/ensemble_tutorial.ipynb", "w") as f:
    nbformat.write(nb, f)
print("Done")
PYEOF
```

Expected output: `Done`

- [ ] **Step 2: Commit**

```bash
cd /Users/o.gallup/Code/alf
git add tutorials/models/ensemble_tutorial.ipynb
git commit -m "feat: add combined ensemble section (Section 5)"
```

---

### Task 7: Add Section 6 — Comparison & Summary

**Files:**
- Modify: `tutorials/models/ensemble_tutorial.ipynb`

- [ ] **Step 1: Append comparison cells**

```bash
cd /Users/o.gallup/Code/alf && python3 - <<'PYEOF'
import nbformat

with open("tutorials/models/ensemble_tutorial.ipynb") as f:
    nb = nbformat.read(f, as_version=4)

md  = nbformat.v4.new_markdown_cell
src = nbformat.v4.new_code_cell

nb.cells += [
    md("## 6. Comparison"),
    src("""\
df = pd.DataFrame(metrics_table).set_index("Mode")

def highlight_best(col):
    styles = [""] * len(col)
    if col.name == "Spearman rho":
        best = col.idxmax()
    elif col.name in ("MSE", "Train time s"):
        best = col.idxmin()
    else:
        return styles
    styles[df.index.get_loc(best)] = "font-weight: bold; background-color: #d4f1d4"
    return styles

display(df.style.apply(highlight_best))"""),
    src("""\
modes     = ["Seed ensemble", "MC Dropout", "Combined"]
all_preds = [seed_preds, dropout_preds, combined_preds]
all_stds  = [np.sqrt(p.variances) for p in all_preds]
colors    = ["#2196F3", "#FF9800", "#4CAF50"]

fig = plt.figure(figsize=(15, 13))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.3)

# Row 0: scatter plots
for col, (mode, preds, std) in enumerate(zip(modes, all_preds, all_stds)):
    ax = fig.add_subplot(gs[0, col])
    sc = ax.scatter(preds.means, val_labels, c=std, cmap="viridis", alpha=0.5, s=10)
    ax.plot(LABEL_LIM, LABEL_LIM, "r--", lw=1)
    ax.set(xlim=LABEL_LIM, ylim=LABEL_LIM, title=mode, xlabel="Predicted mean")
    if col == 0:
        ax.set_ylabel("True brightness")

# Row 1: overlaid std KDEs on a single axes
ax_kde  = fig.add_subplot(gs[1, :])
max_std = max(s.max() for s in all_stds)
xs      = np.linspace(0, max_std * 1.1, 300)
for mode, std, color in zip(modes, all_stds, colors):
    bw  = max(1.06 * std.std() * len(std) ** -0.2, 1e-6)
    kde = np.mean(
        np.exp(-0.5 * ((xs[:, None] - std[None, :]) / bw) ** 2) / (bw * np.sqrt(2 * np.pi)),
        axis=1,
    )
    ax_kde.plot(xs, kde, color=color, label=mode, lw=2)
    ax_kde.fill_between(xs, kde, alpha=0.15, color=color)
ax_kde.set(xlabel="Predicted std", ylabel="Density", title="Uncertainty distributions (overlaid)")
ax_kde.legend()

# Row 2: calibration curves
for col, (mode, preds, color) in enumerate(zip(modes, all_preds, colors)):
    ax = fig.add_subplot(gs[2, col])
    exp_conf, obs_conf = calibration_curve(preds.empirical_dist, val_labels)
    ax.plot(exp_conf, obs_conf, "o-", color=color, label=mode, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Expected conf.", title=mode)
    if col == 0:
        ax.set_ylabel("Observed conf.")
    ax.legend(fontsize=8)

fig.suptitle("Ensemble Mode Comparison", fontsize=14, fontweight="bold")
plt.show()"""),
    md(
        "## Summary\n\n"
        "**Seed ensembles** offer the richest weight-space diversity at N× training cost — "
        "the go-to when compute budget allows. "
        "**MC dropout** is a cheap proxy: one training run, uncertainty from random dropout masks; "
        "calibration is typically weaker. "
        "**Combined** ensembles maximise sample richness (N×T per candidate), "
        "useful for acquisition functions like Thompson sampling that draw directly from "
        "`empirical_dist`. "
        "See [`EnsembleWrapperConfig`](../../tools/alf_tools/models/ensemble.py) for "
        "`member_seeds` and other options."
    ),
]

with open("tutorials/models/ensemble_tutorial.ipynb", "w") as f:
    nbformat.write(nb, f)
print("Done")
PYEOF
```

Expected output: `Done`

- [ ] **Step 2: Commit**

```bash
cd /Users/o.gallup/Code/alf
git add tutorials/models/ensemble_tutorial.ipynb
git commit -m "feat: add comparison and summary section (Section 6)"
```

---

### Task 8: Execute full notebook and verify

**Files:**
- Modify: `tutorials/models/ensemble_tutorial.ipynb` (adds cell outputs)

- [ ] **Step 1: Execute the notebook end-to-end**

```bash
cd /Users/o.gallup/Code/alf
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=900 \
    tutorials/models/ensemble_tutorial.ipynb
echo "Exit: $?"
```

Expected: `Exit: 0` with no tracebacks. On CPU in FAST_MODE this takes ~8 minutes.

- [ ] **Step 2: Verify all code cells have outputs**

```bash
python3 - <<'PYEOF'
import nbformat
with open("tutorials/models/ensemble_tutorial.ipynb") as f:
    nb = nbformat.read(f, as_version=4)
code_cells = [c for c in nb.cells if c.cell_type == "code"]
empty = [i for i, c in enumerate(code_cells) if not c.outputs and not c.source.strip().startswith("#")]
print(f"Code cells: {len(code_cells)}")
print(f"Cells with no output: {empty if empty else 'none — all good'}")
PYEOF
```

Expected: all code cells have outputs.

- [ ] **Step 3: Commit with outputs**

```bash
cd /Users/o.gallup/Code/alf
git add tutorials/models/ensemble_tutorial.ipynb
git commit -m "feat: add executed outputs to ensemble tutorial notebook"
```
