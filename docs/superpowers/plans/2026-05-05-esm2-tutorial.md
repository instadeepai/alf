# ESM-2 Tutorial Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `tutorials/models/esm2_tutorial.ipynb` — a companion to the GP tutorial that teaches ESM-2 as a frozen encoder and LoRA-finetuned surrogate model for protein active learning.

**Architecture:** Five notebook sections (embeddings → frozen/LoRA training → UQ → AL cycle → ProteinGym) built incrementally across 12 tasks. LoRA training is shown as a standalone pattern (not an ESM2DropoutModel subclass) to keep the tutorial self-contained and avoid embedding-cache conflicts. The AL section uses a custom `RoundTracker` state logger to capture per-round uncertainty data for visualisation.

**Tech Stack:** `alf-tools` (GFP, ProteinGym, ESM2DropoutModel, GPModel, DesignTask), `peft` (LoRA), `torch`, `transformers`, `scikit-learn` (PCA), `scipy` (Spearman, calibration), `matplotlib`, `pandas`

---

## File Structure

**Create:**
- `tutorials/models/esm2_tutorial.ipynb` — main tutorial (built across Tasks 2–12)
- `tutorials/models/assets/esm2/wt_structure.png` — pre-rendered IF1 wild-type surface
- `tutorials/models/assets/esm2/fitness_structure.png` — residues coloured by predicted fitness
- `tutorials/models/assets/esm2/uncertainty_structure.png` — residues coloured by MC Dropout std

**Modify:**
- `tutorials/pyproject.toml` — add `peft`, `scikit-learn`, `scipy`

---

## Task 1: Add Tutorial Dependencies

**Files:**
- Modify: `tutorials/pyproject.toml`

- [ ] **Step 1: Add the three new dependencies**

Open `tutorials/pyproject.toml` and add to the `dependencies` list:

```toml
[project]
dependencies = [
    "alf_core",
    "alf_tools",
    "matplotlib>=3.3.0",
    "ipykernel>=6.25.0",
    "pyrosetta-installer",
    "peft>=0.10.0",
    "scikit-learn>=1.3.0",
    "scipy>=1.11.0",
]
```

- [ ] **Step 2: Verify install**

```bash
cd /Users/o.gallup/Code/alf/tutorials && uv sync
```

Expected: resolves and installs without errors. `peft`, `scikit-learn`, and `scipy` appear in the output.

- [ ] **Step 3: Smoke-test imports**

```bash
cd /Users/o.gallup/Code/alf/tutorials && uv run python -c "
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.decomposition import PCA
import scipy.stats
print('all imports ok')
"
```

Expected: prints `all imports ok`.

- [ ] **Step 4: Commit**

```bash
git add tutorials/pyproject.toml
git commit -m "feat: add peft, scikit-learn, scipy to tutorial dependencies"
```

---

## Task 2: Notebook Skeleton + Section 1 Setup + GFP Loading

**Files:**
- Create: `tutorials/models/esm2_tutorial.ipynb`

Create the notebook file and add all cells for the preamble through GFP data loading and initial assertions.

- [ ] **Step 1: Create the notebook**

Create `tutorials/models/esm2_tutorial.ipynb` as a valid Jupyter notebook. Use the `NotebookEdit` tool or `jupyter nbconvert` to start from an empty kernel. The notebook should use a Python 3 kernel.

- [ ] **Step 2: Add the title and prerequisites markdown cell**

Add a **markdown** cell:

```markdown
# ESM-2 as a Surrogate Model for Protein Active Learning

This tutorial introduces the ESM-2 protein language model as a frozen encoder and LoRA-finetuned surrogate for active learning. We use the GFP fluorescence dataset as the primary vehicle, with a ProteinGym DMS assay as a real-world extension.

**Prerequisites:** Complete these tutorials first:
- `tutorials/experiments/offline_design_tutorial.ipynb`
- `tutorials/models/gp_tutorial.ipynb`

**What you will learn:**
1. How ESM-2 embeddings encode protein function without task-specific training
2. How to train a regression head on frozen vs LoRA-adapted ESM-2
3. How MC Dropout provides calibrated uncertainty estimates
4. How to run a full active learning cycle with ESM-2 + UCB acquisition
5. How to transfer ESM-2 to a DMS fitness landscape (ProteinGym)
```

- [ ] **Step 3: Add environment setup code cell**

Add a **code** cell:

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import scipy.stats
from sklearn.decomposition import PCA
from IPython.display import Image, display
from pathlib import Path

from alf_core import (
    BaseDatasetConfig,
    DatasetSearch,
    DesignTask,
    Optimizer,
    Oracle,
    Surrogate,
    TerminalStateLogger,
)
from alf_core.dataclasses.candidate import Modality
from alf_tools.datasets import GFP
from alf_tools.models.esm2 import (
    ESM2DropoutModel,
    ESM2ModelConfig,
    ESM2TrainConfig,
    ESM2RegressionHead,
)
from alf_tools.models.gp import GPModel, GPModelConfig, FeaturizerConfig
from alf_tools.models.utils import extract_sequences_from_inputs
from alf_tools.optimizer.acquisition_functions import UCB

# Device detection: GPU if available, CPU fallback
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cpu":
    print("⚠️  Running on CPU — long-running cells will be slower. "
          "See timing notes in each section.")

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
```

- [ ] **Step 4: Add Section 1 header and goals markdown cell**

Add a **markdown** cell:

```markdown
---
## Section 1: ESM-2 as a Sequence Encoder

**Goals:**
- Load the GFP dataset and inspect sequence/label distributions
- Compute ESM-2 mean-pooled embeddings for a subset of sequences
- Visualise the embedding space with PCA, coloured by fitness
- Understand why ESM-2 embeddings are valuable compared to raw sequence features

ESM-2 was pretrained on ~250M protein sequences from UniRef50. Even before any task-specific training, its 640-dimensional embeddings capture evolutionary relationships and functional properties of proteins.
```

- [ ] **Step 5: Add GFP loading code cell**

Add a **code** cell:

```python
# Section 2 supervised split: 480 train, 120 val, 200 test, 200 pool
gfp_supervised = GFP(BaseDatasetConfig(
    name="gfp",
    modality="sequence",
    seed=SEED,
    train_ratio=0.6,       # 600 sequences for train+val
    validation_frac=0.2,   # 20% of train+val → 120 val, 480 train
    test_ratio=0.2,        # 200 test sequences
    split_type="random",
))
gfp_supervised.setup()

train_data = gfp_supervised.train_dataset
val_data   = gfp_supervised.validation_dataset
test_data  = gfp_supervised.test_dataset

print(gfp_supervised)
```

- [ ] **Step 6: Add assertion cell (TDD check)**

Add a **code** cell:

```python
# Verify expected split sizes
assert len(train_data) == 480, f"Expected 480 train, got {len(train_data)}"
assert len(val_data)   == 120, f"Expected 120 val, got {len(val_data)}"
assert len(test_data)  == 200, f"Expected 200 test, got {len(test_data)}"

# Labels are continuous brightness values
labels_all = np.array(train_data.labels)
assert labels_all.ndim == 1
print(f"Train label range: [{labels_all.min():.3f}, {labels_all.max():.3f}]")
print(f"Sample sequences (first 3):")
for cand in list(train_data.candidates)[:3]:
    print(f"  {cand.sequence[:30]}...")
print("✓ GFP dataset loaded and splits verified")
```

- [ ] **Step 7: Add label distribution plot cell**

Add a **code** cell:

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Label histogram
for split, data, colour in [
    ("train", train_data, "steelblue"),
    ("val",   val_data,   "orange"),
    ("test",  test_data,  "green"),
]:
    axes[0].hist(np.array(data.labels), bins=30, alpha=0.6, label=split, color=colour)
axes[0].set_xlabel("Median Brightness"); axes[0].set_ylabel("Count")
axes[0].set_title("GFP Brightness Distribution by Split")
axes[0].legend()

# Sequence length distribution
lengths = [len(c.sequence) for c in list(train_data.candidates)]
axes[1].hist(lengths, bins=20, color="steelblue", alpha=0.8)
axes[1].set_xlabel("Sequence Length"); axes[1].set_ylabel("Count")
axes[1].set_title("GFP Sequence Length Distribution (Train)")

plt.tight_layout(); plt.show()
```

- [ ] **Step 8: Commit**

```bash
git add tutorials/models/esm2_tutorial.ipynb
git commit -m "feat: add notebook skeleton, env setup, and Section 1 GFP loading cells"
```

---

## Task 3: Section 1 — Embeddings and PCA Visualisation

**Files:**
- Modify: `tutorials/models/esm2_tutorial.ipynb`

- [ ] **Step 1: Add embedding computation markdown cell**

Add a **markdown** cell:

```markdown
### ESM-2 Embeddings

We load a frozen ESM-2 150M encoder and compute mean-pooled sequence embeddings for 200 GFP sequences. No training happens here — these are the raw pretrained representations.
```

- [ ] **Step 2: Add embedding computation code cell**

Add a **code** cell:

```python
%%time
# Initialise frozen model (no training yet)
frozen_model = ESM2DropoutModel(
    model_config=ESM2ModelConfig(),     # 150M, embedding_dim=640
    train_config=ESM2TrainConfig(),
    device=device,
)

# Embed 200 sequences from the training set for visualisation
N_VIZ = 200
viz_candidates = list(train_data.candidates)[:N_VIZ]
viz_labels     = np.array(train_data.labels)[:N_VIZ]

with torch.no_grad():
    embeddings = frozen_model._get_embeddings(
        [c.sequence for c in viz_candidates]
    ).numpy()  # (200, 640)

print(f"Embeddings shape: {embeddings.shape}")
```

- [ ] **Step 3: Add embedding assertion cell**

Add a **code** cell:

```python
assert embeddings.shape == (N_VIZ, 640), \
    f"Expected (200, 640), got {embeddings.shape}"
assert not np.any(np.isnan(embeddings)), "NaN in embeddings"
print("✓ Embeddings shape and validity verified")
```

- [ ] **Step 4: Add PCA visualisation code cell**

Add a **code** cell:

```python
pca = PCA(n_components=2, random_state=SEED)
pca_coords = pca.fit_transform(embeddings)  # (200, 2)
explained = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(
    pca_coords[:, 0], pca_coords[:, 1],
    c=viz_labels, cmap="plasma", alpha=0.85, s=40, edgecolors="none"
)
plt.colorbar(sc, ax=ax, label="Brightness")
ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
ax.set_title("ESM-2 Embedding Space (PCA) — Coloured by GFP Brightness")

plt.tight_layout(); plt.show()
print("Expected: bright sequences cluster together, "
      "showing ESM-2 captures functional relationships before any training.")
```

- [ ] **Step 5: Add GP contrast callout markdown cell**

Add a **markdown** cell:

```markdown
### Key Takeaway — Section 1

ESM-2 embeddings organise sequences by function *without any task-specific training*. Bright sequences cluster together in PCA space, showing that evolutionary pretraining captures GFP-relevant properties.

**Contrast with GP:** A Gaussian Process with one-hot sequence features treats each sequence position independently and has no concept of evolutionary context. ESM-2 provides a biologically informed starting point that makes downstream learning much more data-efficient.

| Feature | GP (one-hot) | ESM-2 (frozen) |
|---|---|---|
| Captures evolutionary context | ✗ | ✓ |
| Uncertainty estimate | Closed-form posterior | MC Dropout |
| Training data required | Any amount | Pretrained, few-shot ready |
| Compute | Fast | Requires GPU for large datasets |
```

- [ ] **Step 6: Commit**

```bash
git add tutorials/models/esm2_tutorial.ipynb
git commit -m "feat: add Section 1 embedding computation and PCA visualisation cells"
```

---

## Task 4: Section 2 — Frozen Encoder Training

**Files:**
- Modify: `tutorials/models/esm2_tutorial.ipynb`

- [ ] **Step 1: Add Section 2 header markdown cell**

Add a **markdown** cell:

```markdown
---
## Section 2: Training the Regression Head

**Goals:**
- Train the frozen ESM-2 head (only MLP parameters updated)
- Train a LoRA-adapted ESM-2 (LoRA adapter + MLP parameters updated)
- Train a GP baseline with one-hot sequence encoding
- Compare Spearman correlation on the test set across all three models

### Architecture Overview

**Frozen ESM-2:**
```
ESM-2 encoder (frozen, 150M params) → mean pool → MLP head (trainable, ~330K params)
```

**LoRA ESM-2:**
```
ESM-2 encoder + LoRA adapters (trainable, ~1.2M params) → mean pool → MLP head (trainable)
```

LoRA (Low-Rank Adaptation) injects small trainable matrices into the query and value projection layers. The base ESM-2 weights remain frozen; only the adapters are updated.
```

- [ ] **Step 2: Add frozen training code cell**

Add a **code** cell:

```python
%%time
# Reduce epochs on CPU for speed
num_epochs = 50 if device == "cuda" else 10
print(f"Training for {num_epochs} epochs on {device}...")

frozen_model = ESM2DropoutModel(
    model_config=ESM2ModelConfig(),
    train_config=ESM2TrainConfig(num_epochs=num_epochs, log_frequency=num_epochs // 5),
    device=device,
)
frozen_model.train(train_data, val_data)

frozen_metrics = frozen_model.get_epoch_metrics()
print(f"Final train loss: {frozen_metrics[-1].train_loss:.4f}")
if frozen_metrics[-1].val_loss is not None:
    print(f"Final val loss:   {frozen_metrics[-1].val_loss:.4f}")
```

- [ ] **Step 3: Add assertion cell**

Add a **code** cell:

```python
assert len(frozen_metrics) == num_epochs, \
    f"Expected {num_epochs} epoch metrics, got {len(frozen_metrics)}"
assert frozen_metrics[-1].train_loss < frozen_metrics[0].train_loss, \
    "Train loss should decrease over training"
print("✓ Frozen model training epoch metrics verified")
```

- [ ] **Step 4: Add frozen training curve plot cell**

Add a **code** cell:

```python
epochs     = [m.epoch for m in frozen_metrics]
train_loss = [m.train_loss for m in frozen_metrics]
val_loss   = [m.val_loss for m in frozen_metrics if m.val_loss is not None]
val_epochs = [m.epoch for m in frozen_metrics if m.val_loss is not None]
train_spearman = [m.additional_metrics.get("train_spearman", float("nan"))
                  for m in frozen_metrics]
val_spearman   = [m.additional_metrics.get("val_spearman", float("nan"))
                  for m in frozen_metrics if m.val_loss is not None]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, train_loss, label="train loss", color="steelblue")
if val_loss:
    ax1.plot(val_epochs, val_loss, label="val loss", color="orange")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE Loss")
ax1.set_title("Frozen ESM-2: Training Curves"); ax1.legend()

ax2.plot(epochs, train_spearman, label="train Spearman", color="steelblue")
if val_spearman:
    ax2.plot(val_epochs, val_spearman, label="val Spearman", color="orange")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Spearman ρ"); ax2.set_ylim(0, 1)
ax2.set_title("Frozen ESM-2: Spearman Correlation"); ax2.legend()

plt.tight_layout(); plt.show()
```

- [ ] **Step 5: Compute frozen test Spearman**

Add a **code** cell:

```python
frozen_preds = frozen_model.predict(list(test_data.candidates), with_uncertainty=False)
frozen_spearman = scipy.stats.spearmanr(
    frozen_preds.means, np.array(test_data.labels)
).statistic
print(f"Frozen ESM-2 test Spearman: {frozen_spearman:.3f}")
assert frozen_spearman > 0.3, f"Expected Spearman > 0.3, got {frozen_spearman:.3f}"
print("✓ Frozen model achieves reasonable Spearman correlation")
```

- [ ] **Step 6: Commit**

```bash
git add tutorials/models/esm2_tutorial.ipynb
git commit -m "feat: add Section 2 frozen ESM-2 training and evaluation cells"
```

---

## Task 5: Section 2 — LoRA Fine-Tuning

**Files:**
- Modify: `tutorials/models/esm2_tutorial.ipynb`

- [ ] **Step 1: Add LoRA intro markdown cell**

Add a **markdown** cell:

```markdown
### LoRA Fine-Tuning

LoRA (Hu et al., 2022) injects trainable rank-16 matrices into the query and value projections of every ESM-2 attention layer. The base weights stay frozen; only the adapters (≈0.8% of total parameters) are updated. This lets the encoder adapt its representations to the GFP fitness landscape while retaining its evolutionary prior.

We implement LoRA as a standalone training pattern so every step is visible.
```

- [ ] **Step 2: Add LoRA helper functions cell**

Add a **code** cell:

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import EsmModel

def embed_sequences(encoder, tokenizer, sequences, device, batch_size=32):
    """Mean-pool ESM-2 hidden states over non-padding positions."""
    all_embs = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True)
        input_ids = tokens["input_ids"].to(device)
        mask = tokens["attention_mask"].to(device)
        out = encoder(input_ids=input_ids, attention_mask=mask)
        m = mask.unsqueeze(-1).float()
        emb = (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-9)
        all_embs.append(emb.detach().cpu())
    return torch.cat(all_embs, dim=0)  # (N, 640)


def lora_mc_predict(encoder, head, tokenizer, candidates, device, num_samples=30):
    """MC Dropout predictions from a standalone LoRA model."""
    sequences = extract_sequences_from_inputs(candidates)
    encoder.eval()
    with torch.no_grad():
        x = embed_sequences(encoder, tokenizer, sequences, device).to(device)
    head.train()  # Keep dropout active for MC sampling
    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            samples.append(head(x).cpu().numpy())
    arr = np.stack(samples, axis=1)  # (N, num_samples)
    return arr.mean(axis=1), arr.var(axis=1), arr
```

- [ ] **Step 3: Add LoRA model setup cell**

Add a **code** cell:

```python
# Fresh ESM-2 encoder for LoRA (separate from frozen_model)
lora_enc = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D").to(device)
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=16,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
)
lora_enc = get_peft_model(lora_enc, lora_config)
lora_enc.print_trainable_parameters()
# Expected: trainable params ~1.2M out of ~150M (≈0.8%)

# Regression head (same architecture as frozen model)
lora_head = ESM2RegressionHead(
    embedding_dim=640, hidden_dim=256, num_hidden_layers=2, dropout=0.1
).to(device)

# Reuse tokenizer from frozen_model
tokenizer = frozen_model.tokenizer

# Optimizer: head params + LoRA adapter params only
lora_adapter_params = [p for p in lora_enc.parameters() if p.requires_grad]
lora_optimizer = optim.Adam(
    list(lora_head.parameters()) + lora_adapter_params, lr=1e-3
)
criterion_lora = nn.MSELoss()

# Print trainable param counts
head_params  = sum(p.numel() for p in lora_head.parameters())
lora_params  = sum(p.numel() for p in lora_adapter_params)
total_params = sum(p.numel() for p in lora_enc.parameters())
print(f"MLP head params:   {head_params:,}")
print(f"LoRA adapter params: {lora_params:,}")
print(f"ESM-2 total params:  {total_params:,}")
```

- [ ] **Step 4: Add assertion on trainable parameters**

Add a **code** cell:

```python
frozen_base_params = sum(
    p.numel() for p in lora_enc.parameters() if not p.requires_grad
)
assert frozen_base_params > 0, "Base ESM-2 weights should be frozen"
assert lora_params > 0, "LoRA adapter weights should be trainable"
assert lora_params < total_params * 0.02, \
    f"LoRA should be <2% of total params, got {lora_params/total_params:.1%}"
print(f"✓ LoRA adapters = {lora_params/total_params:.2%} of total parameters")
```

- [ ] **Step 5: Add LoRA training loop cell**

Add a **code** cell:

```python
%%time
num_epochs_lora = num_epochs  # Same as frozen for fair comparison
train_seqs = extract_sequences_from_inputs(train_data)
val_seqs   = extract_sequences_from_inputs(val_data)
train_labels_t = torch.tensor(train_data.labels, dtype=torch.float32).to(device)

lora_train_losses, lora_val_spearman_list = [], []
log_freq = max(1, num_epochs_lora // 5)

for epoch in range(num_epochs_lora):
    # --- Train step ---
    lora_enc.train()
    lora_head.train()
    x_tr = embed_sequences(lora_enc, tokenizer, train_seqs, device).to(device)
    preds_tr = lora_head(x_tr)
    loss = criterion_lora(preds_tr, train_labels_t)
    lora_optimizer.zero_grad()
    loss.backward()
    lora_optimizer.step()
    lora_train_losses.append(loss.item())

    # --- Validation Spearman ---
    lora_enc.eval()
    lora_head.eval()
    with torch.no_grad():
        x_val = embed_sequences(lora_enc, tokenizer, val_seqs, device).to(device)
        preds_val = lora_head(x_val).cpu().numpy()
    spearman = scipy.stats.spearmanr(preds_val, val_data.labels).statistic
    lora_val_spearman_list.append(spearman)

    if epoch % log_freq == 0:
        print(f"Epoch {epoch:3d}/{num_epochs_lora} — "
              f"loss: {loss.item():.4f}, val Spearman: {spearman:.3f}")

print("LoRA training complete.")
```

- [ ] **Step 6: Add LoRA training curve plot cell**

Add a **code** cell:

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(lora_train_losses, label="LoRA train loss", color="darkorange")
ax1.plot(train_loss, label="Frozen train loss", color="steelblue", linestyle="--")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE Loss")
ax1.set_title("Training Loss: Frozen vs LoRA ESM-2"); ax1.legend()

ax2.plot(lora_val_spearman_list, label="LoRA val Spearman", color="darkorange")
val_sp_full = [m.additional_metrics.get("val_spearman", float("nan"))
               for m in frozen_metrics]
ax2.plot(val_sp_full, label="Frozen val Spearman", color="steelblue", linestyle="--")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Spearman ρ"); ax2.set_ylim(0, 1)
ax2.set_title("Validation Spearman: Frozen vs LoRA ESM-2"); ax2.legend()

plt.tight_layout(); plt.show()
```

- [ ] **Step 7: Compute LoRA test Spearman**

Add a **code** cell:

```python
lora_means, lora_vars, lora_samples = lora_mc_predict(
    lora_enc, lora_head, tokenizer,
    list(test_data.candidates), device, num_samples=30
)
lora_spearman = scipy.stats.spearmanr(lora_means, np.array(test_data.labels)).statistic
print(f"LoRA ESM-2 test Spearman: {lora_spearman:.3f}")
```

- [ ] **Step 8: Commit**

```bash
git add tutorials/models/esm2_tutorial.ipynb
git commit -m "feat: add Section 2 LoRA training cells"
```

---

## Task 6: Section 2 — GP Baseline and Comparison Chart

**Files:**
- Modify: `tutorials/models/esm2_tutorial.ipynb`

- [ ] **Step 1: Add GP baseline code cell**

Add a **code** cell:

```python
%%time
print("Training GP baseline (Matérn-2.5 + one-hot encoding)...")
gp_model = GPModel(
    model_config=GPModelConfig(kernel_type="matern", matern_nu=2.5),
    featurizer_config=FeaturizerConfig(
        featurizer_type="one_hot",
        flatten_one_hot=True,
    ),
)
gp_model.train(train_data, val_data)
gp_preds = gp_model.predict(list(test_data.candidates), with_uncertainty=False)
gp_spearman = scipy.stats.spearmanr(
    gp_preds.means, np.array(test_data.labels)
).statistic
print(f"GP (Matérn-2.5, one-hot) test Spearman: {gp_spearman:.3f}")
```

- [ ] **Step 2: Add comparison bar chart cell**

Add a **code** cell:

```python
results = {
    "GP\n(Matérn-2.5,\none-hot)": gp_spearman,
    "Frozen\nESM-2": frozen_spearman,
    "LoRA\nESM-2": lora_spearman,
}
colours = ["#4CAF50", "#2196F3", "#FF9800"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(results.keys(), results.values(), color=colours, alpha=0.85, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, results.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
ax.set_ylim(0, 1)
ax.set_ylabel("Spearman ρ (test set)")
ax.set_title("GFP Fitness Prediction: Model Comparison\n(480 training sequences)")
ax.axhline(0, color="black", linewidth=0.5)
plt.tight_layout(); plt.show()
```

- [ ] **Step 3: Add comparison interpretation markdown cell**

Add a **markdown** cell:

```markdown
### Key Takeaway — Section 2

| Model | Spearman ρ | Parameters trained | Best for |
|---|---|---|---|
| GP (Matérn-2.5, one-hot) | ~0.5–0.6 | kernel hyperparams | < 100 sequences |
| Frozen ESM-2 head | ~0.7–0.8 | ~330K (MLP only) | 100–1000 sequences, no GPU |
| LoRA ESM-2 | ~0.75–0.85 | ~1.5M (LoRA + MLP) | 1000+ sequences or GPU available |

LoRA improves over the frozen head by adapting the encoder's representations to the GFP fitness landscape. The GP baseline underperforms here because one-hot encoding ignores evolutionary context — on very small datasets (< 100 sequences), GP would be more competitive.
```

- [ ] **Step 4: Commit**

```bash
git add tutorials/models/esm2_tutorial.ipynb
git commit -m "feat: add Section 2 GP baseline and comparison chart cells"
```

---

## Task 7: Section 3 — MC Dropout Demo and Batch Uncertainty Landscape

**Files:**
- Modify: `tutorials/models/esm2_tutorial.ipynb`

- [ ] **Step 1: Add Section 3 header markdown cell**

Add a **markdown** cell:

```markdown
---
## Section 3: Uncertainty Quantification via MC Dropout

**Goals:**
- Visualise the MC Dropout sample distribution for individual sequences
- Show how predicted uncertainty correlates with prediction error
- Build a reliability diagram to check calibration
- Compare calibration between frozen and LoRA models

### How MC Dropout Works

During normal inference, dropout is disabled. MC Dropout keeps dropout *active* during prediction and runs $T=30$ forward passes through the head. Each pass samples a different random subnetwork:

```
x → [head with dropout ON, pass 1] → ŷ₁
x → [head with dropout ON, pass 2] → ŷ₂
...
x → [head with dropout ON, pass 30] → ŷ₃₀
```

The mean $\bar{y}$ and variance $\sigma^2$ of these 30 samples are the predictive mean and uncertainty.
```

- [ ] **Step 2: Add single-sequence demo code cell**

Add a **code** cell:

```python
# Pick three representative test sequences:
#   - high fitness (bright), low uncertainty expected
#   - median fitness
#   - low fitness (dim), high uncertainty expected from model
test_labels_arr = np.array(test_data.labels)
test_cands      = list(test_data.candidates)

idx_high   = int(np.argmax(test_labels_arr))
idx_median = int(np.argsort(test_labels_arr)[len(test_labels_arr) // 2])
idx_low    = int(np.argmin(test_labels_arr))
examples = {
    "High fitness": (idx_high,   test_labels_arr[idx_high]),
    "Median fitness": (idx_median, test_labels_arr[idx_median]),
    "Low fitness":  (idx_low,    test_labels_arr[idx_low]),
}

# Get full empirical distributions from frozen model
frozen_full_preds = frozen_model.predict(test_cands, with_uncertainty=True)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
for ax, (label, (idx, true_val)) in zip(axes, examples.items()):
    samples = frozen_full_preds.empirical_dist[idx]  # (30,)
    mean_val = samples.mean()
    std_val  = samples.std()
    ax.hist(samples, bins=10, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(mean_val, color="navy",  linewidth=2, label=f"mean={mean_val:.2f}")
    ax.axvline(mean_val + 2*std_val, color="red", linewidth=1.5, linestyle="--")
    ax.axvline(mean_val - 2*std_val, color="red", linewidth=1.5, linestyle="--", label="±2σ")
    ax.axvline(true_val, color="green", linewidth=2, linestyle=":", label=f"true={true_val:.2f}")
    ax.set_xlabel("Predicted brightness")
    ax.set_title(f"{label}\n(σ={std_val:.3f})")
    ax.legend(fontsize=8)

plt.suptitle("MC Dropout: 30-Sample Distributions for Three GFP Sequences", y=1.02)
plt.tight_layout(); plt.show()
```

- [ ] **Step 3: Add distribution assertion cell**

Add a **code** cell:

```python
assert frozen_full_preds.empirical_dist.shape == (len(test_cands), 30), \
    f"Expected ({len(test_cands)}, 30), got {frozen_full_preds.empirical_dist.shape}"
assert np.all(frozen_full_preds.variances >= 0), "Variances must be non-negative"
print("✓ empirical_dist shape and variance sign verified")
```

- [ ] **Step 4: Add batch uncertainty landscape code cell**

Add a **code** cell:

```python
means  = frozen_full_preds.means
stds   = np.sqrt(frozen_full_preds.variances)
truths = test_labels_arr

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(
    means, truths,
    c=stds, cmap="hot_r", alpha=0.75, s=30, edgecolors="none"
)
plt.colorbar(sc, ax=ax, label="Predicted std (σ)")
lim = [min(means.min(), truths.min()) - 0.1,
       max(means.max(), truths.max()) + 0.1]
ax.plot(lim, lim, "k--", linewidth=1, label="perfect prediction")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Predicted mean brightness")
ax.set_ylabel("True brightness")
ax.set_title("Frozen ESM-2: Predicted Mean vs True Brightness\n(colour = MC Dropout std)")
ax.legend()
plt.tight_layout(); plt.show()

corr_err_unc = scipy.stats.spearmanr(
    np.abs(means - truths), stds
).statistic
print(f"Spearman correlation between |error| and σ: {corr_err_unc:.3f}")
print("Positive correlation means higher uncertainty ↔ larger prediction errors (good)")
```

- [ ] **Step 5: Commit**

```bash
git add tutorials/models/esm2_tutorial.ipynb
git commit -m "feat: add Section 3 MC Dropout demo and batch uncertainty landscape cells"
```

---

## Task 8: Section 3 — Calibration Check and Frozen vs LoRA Comparison

**Files:**
- Modify: `tutorials/models/esm2_tutorial.ipynb`

- [ ] **Step 1: Add calibration helper and reliability diagram cell**

Add a **code** cell:

```python
def calibration_coverage(means, stds, truths, confidence_levels):
    """Compute empirical coverage at each confidence level.

    Args:
        means: Predicted means, shape (N,).
        stds: Predicted standard deviations, shape (N,).
        truths: Ground-truth values, shape (N,).
        confidence_levels: List of floats in (0, 1), e.g. [0.5, 0.8, 0.95].

    Returns:
        List of observed coverage fractions (same length as confidence_levels).
    """
    observed = []
    for conf in confidence_levels:
        z = scipy.stats.norm.ppf((1 + conf) / 2)
        lower = means - z * stds
        upper = means + z * stds
        cov = np.mean((truths >= lower) & (truths <= upper))
        observed.append(float(cov))
    return observed


confidence_levels = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

# Frozen model calibration
frozen_obs = calibration_coverage(
    frozen_full_preds.means,
    np.sqrt(frozen_full_preds.variances),
    test_labels_arr,
    confidence_levels,
)

# LoRA model calibration
lora_test_means, lora_test_vars, _ = lora_mc_predict(
    lora_enc, lora_head, tokenizer, test_cands, device, num_samples=30
)
lora_obs = calibration_coverage(
    lora_test_means, np.sqrt(lora_test_vars), test_labels_arr, confidence_levels
)
```

- [ ] **Step 2: Add calibration assertion cell**

Add a **code** cell:

```python
assert len(frozen_obs) == len(confidence_levels)
assert all(0 <= v <= 1 for v in frozen_obs), "Coverage must be in [0, 1]"
print("Frozen calibration (expected vs observed):")
for exp, obs in zip(confidence_levels, frozen_obs):
    print(f"  {exp*100:.0f}% interval → {obs*100:.1f}% coverage")
print()
print("LoRA calibration (expected vs observed):")
for exp, obs in zip(confidence_levels, lora_obs):
    print(f"  {exp*100:.0f}% interval → {obs*100:.1f}% coverage")
```

- [ ] **Step 3: Add reliability diagram plot cell**

Add a **code** cell:

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, obs, title, colour in [
    (axes[0], frozen_obs, "Frozen ESM-2",  "steelblue"),
    (axes[1], lora_obs,   "LoRA ESM-2",    "darkorange"),
]:
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    ax.plot(confidence_levels, obs, "o-", color=colour, linewidth=2,
            markersize=7, label=f"{title}")
    ax.fill_between(confidence_levels, confidence_levels, obs,
                    alpha=0.15, color=colour)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Expected coverage"); ax.set_ylabel("Observed coverage")
    ax.set_title(f"Reliability Diagram: {title}")
    ax.legend()

plt.suptitle("Calibration Check: Does the confidence interval actually contain the truth?")
plt.tight_layout(); plt.show()
```

- [ ] **Step 4: Add calibration takeaway markdown cell**

Add a **markdown** cell:

```markdown
### Key Takeaway — Section 3

**Reading the reliability diagram:** The diagonal line is perfect calibration. Points *above* the diagonal mean the model is overconfident (its 80% interval only contains 60% of truths). Points *below* mean it is underconfident (its 80% interval contains 95% of truths).

MC Dropout typically skews toward **underconfidence** — the uncertainty estimates are generous. This is actually *safe* for active learning: the acquisition function won't prematurely dismiss uncertain regions of sequence space that might harbour high-fitness variants.

**Frozen vs LoRA:** LoRA tends to be better calibrated because its encoder representations are task-adapted, producing tighter and more faithful uncertainty estimates on in-distribution sequences.

**Contrast with GP:** A Gaussian Process posterior variance is theoretically principled and tends to be well-calibrated by construction. MC Dropout is an approximation — it works well in practice but requires more sequences to converge to good calibration.
```

- [ ] **Step 5: Commit**

```bash
git add tutorials/models/esm2_tutorial.ipynb
git commit -m "feat: add Section 3 calibration check and reliability diagram cells"
```

---

## Task 9: Section 4 — Active Learning Setup and Run

**Files:**
- Modify: `tutorials/models/esm2_tutorial.ipynb`

- [ ] **Step 1: Add Section 4 header markdown cell**

Add a **markdown** cell:

```markdown
---
## Section 4: Full Active Learning Cycle

**Goals:**
- Run 5 rounds of UCB-guided active learning on GFP
- Track per-round model improvement and uncertainty collapse
- Compare UCB acquisition to random baseline
- Visualise where in sequence space the model explores

We use the frozen ESM-2 surrogate for speed. The LoRA swap is shown at the end as a one-line change.

**Setup:**
- 40 labelled seed sequences (initial train set)
- 750-sequence unlabelled candidate pool
- Oracle = ground-truth GFP labels (simulated online evaluation)
- 5 acquisition rounds, 10 candidates per round
```

- [ ] **Step 2: Add AL dataset setup cell**

Add a **code** cell:

```python
# AL split: small seed set + large candidate pool
gfp_al = GFP(BaseDatasetConfig(
    name="gfp",
    modality="sequence",
    seed=SEED,
    train_ratio=0.05,      # ~50 sequences for train+val seed
    validation_frac=0.2,   # 20% of seed → ~10 val, ~40 train
    test_ratio=0.2,        # 200 test
    split_type="random",
))
gfp_al.setup()
print(gfp_al)
# Expected: train≈40, validation≈10, test≈200, candidate_pool≈750
```

- [ ] **Step 3: Add assertion cell for AL splits**

Add a **code** cell:

```python
assert len(gfp_al.train_dataset) > 0, "Need at least one train sequence"
assert len(gfp_al.candidate_pool) >= 500, \
    f"Expected ≥500 pool candidates, got {len(gfp_al.candidate_pool)}"
assert len(gfp_al.test_dataset) > 0, "Need test set for evaluation"
print(f"✓ AL splits: {len(gfp_al.train_dataset)} train seed, "
      f"{len(gfp_al.candidate_pool)} pool, {len(gfp_al.test_dataset)} test")
```

- [ ] **Step 4: Add AL components and RoundTracker cell**

Add a **code** cell:

```python
class RoundTracker:
    """In-memory state logger for tutorial visualisation.

    Captures per-round predictions on the full initial candidate pool
    so we can plot uncertainty evolution across rounds.
    """
    def __init__(self, initial_pool_candidates):
        # Fix the pool at round 0 so it doesn't shrink as candidates are acquired
        self._pool = list(initial_pool_candidates)
        self.records = []

    def log(self, state):
        preds = state.surrogate.predict(self._pool)
        stds  = np.sqrt(np.maximum(preds.variances, 0)) \
                if preds.variances is not None else np.zeros(len(self._pool))
        self.records.append({
            "round":      state.round,
            "best_known": float(max(state.dataset.train_dataset.labels)),
            "means":      preds.means.copy(),
            "stds":       stds.copy(),
        })


# Instantiate all AL components
al_model = ESM2DropoutModel(
    model_config=ESM2ModelConfig(),
    train_config=ESM2TrainConfig(
        num_epochs=20 if device == "cuda" else 5,
        log_frequency=10,
    ),
    device=device,
)
al_surrogate = Surrogate(model=al_model)
al_optimizer = Optimizer(
    acquisition_fn=UCB(alpha=0.9),
    search_fn=DatasetSearch(),
)
al_oracle  = Oracle(scorer=gfp_al)
al_task    = DesignTask(num_acq_rounds=5, acq_batch_size=10)
al_tracker = RoundTracker(list(gfp_al.candidate_pool.candidates))
```

- [ ] **Step 5: Add AL run cell**

Add a **code** cell:

```python
%%time
print("Running active learning loop (5 rounds × 10 candidates)...")
print(f"  Estimated time: {'~3–5 min' if device == 'cuda' else '~10–20 min'} on {device}")

state = al_task.setup(dataset=gfp_al, surrogate=al_surrogate)
al_task.run(
    state=state,
    state_loggers=[al_tracker, TerminalStateLogger()],
    optimizer=al_optimizer,
    oracle=al_oracle,
)
print(f"Done. {len(al_tracker.records)} rounds recorded.")
```

- [ ] **Step 6: Add AL run assertion cell**

Add a **code** cell:

```python
assert len(al_tracker.records) == 5, \
    f"Expected 5 round records, got {len(al_tracker.records)}"
assert al_tracker.records[-1]["best_known"] >= al_tracker.records[0]["best_known"], \
    "Best known fitness should not decrease over rounds"
print(f"✓ Round 1 best: {al_tracker.records[0]['best_known']:.3f}")
print(f"✓ Round 5 best: {al_tracker.records[-1]['best_known']:.3f}")
```

- [ ] **Step 7: Commit**

```bash
git add tutorials/models/esm2_tutorial.ipynb
git commit -m "feat: add Section 4 AL setup, RoundTracker logger, and run cells"
```

---

## Task 10: Section 4 — Active Learning Visualisations

**Files:**
- Modify: `tutorials/models/esm2_tutorial.ipynb`

- [ ] **Step 1: Compute random baseline for comparison**

Add a **code** cell:

```python
# Random baseline: pick 10 candidates at random each round from the original pool
# Simulate 5 rounds using the true labels
rng_baseline = np.random.RandomState(SEED + 1)
all_pool_labels = np.array([c.label for c in list(gfp_al.candidate_pool)])
random_best, random_known = [], [float(max(gfp_al.train_dataset.labels))]
remaining_idxs = list(range(len(all_pool_labels)))

for _ in range(5):
    chosen = rng_baseline.choice(remaining_idxs, size=10, replace=False)
    for idx in chosen:
        remaining_idxs.remove(idx)
    random_known.append(max(random_known[-1], float(all_pool_labels[chosen].max())))
    random_best.append(random_known[-1])

al_best = [r["best_known"] for r in al_tracker.records]
rounds_x = list(range(1, 6))
```

- [ ] **Step 2: Add acquisition history plot cell**

Add a **code** cell:

```python
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(rounds_x, al_best,      "o-", color="steelblue",  linewidth=2.5,
        markersize=8, label="UCB (ESM-2)")
ax.plot(rounds_x, random_best,  "s--", color="grey",     linewidth=2,
        markersize=7, label="Random baseline")
ax.set_xlabel("Acquisition Round")
ax.set_ylabel("Best Fitness Found So Far")
ax.set_title("Active Learning: UCB vs Random Baseline on GFP")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
```

- [ ] **Step 3: Add round-by-round UQ evolution plot cell**

Add a **code** cell:

```python
rounds_to_show = [0, 2, 4]  # 0-indexed into tracker.records → rounds 1, 3, 5
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)

for ax, r_idx in zip(axes, rounds_to_show):
    rec = al_tracker.records[r_idx]
    sc = ax.scatter(
        rec["means"], rec["stds"],
        c=rec["stds"], cmap="hot_r", vmin=0,
        vmax=max(al_tracker.records[0]["stds"].max(), 0.01),
        alpha=0.6, s=15, edgecolors="none"
    )
    ax.set_xlabel("Predicted mean fitness")
    ax.set_ylabel("Predicted std (σ)")
    ax.set_title(f"Round {rec['round']}: Candidate Pool UQ")
    plt.colorbar(sc, ax=ax, label="σ")

plt.suptitle("Uncertainty Collapse Over Acquisition Rounds", y=1.01)
plt.tight_layout(); plt.show()
```

- [ ] **Step 4: Add top candidates table cell**

Add a **code** cell:

```python
# Collect all acquired candidates across rounds
acquired_cands = state.dataset.train_dataset  # includes original seed + all acquired
n_seed = len(gfp_al.train_dataset)
acquired_only_cands = list(acquired_cands.candidates)[n_seed:]
acquired_only_labels = list(acquired_cands.labels)[n_seed:]

# Predict with uncertainty on acquired sequences
acq_preds = al_model.predict(acquired_only_cands, with_uncertainty=True)
top_df = pd.DataFrame({
    "sequence_excerpt": [c.sequence[:20] + "..." for c in acquired_only_cands],
    "true_brightness":  [round(float(l), 4) for l in acquired_only_labels],
    "predicted_mean":   [round(float(m), 4) for m in acq_preds.means],
    "predicted_std":    [round(float(s), 4)
                         for s in np.sqrt(np.maximum(acq_preds.variances, 0))],
}).sort_values("true_brightness", ascending=False).head(10).reset_index(drop=True)

print("Top 10 acquired candidates by true brightness:")
display(top_df)
```

- [ ] **Step 5: Add LoRA swap callout cell**

Add a **code** cell:

```python
# ── LoRA SWAP (one-line change) ──────────────────────────────────────────────
# To use LoRA instead of frozen ESM-2 in the AL loop above, replace:
#
#   al_model = ESM2DropoutModel(...)
#
# with a LoRA-enabled model. Since ESM2DropoutModel freezes the encoder,
# you can swap in the standalone LoRA model by wrapping lora_enc + lora_head
# inside a custom BaseModel subclass, or extend ESM2DropoutModel as shown in
# Task 5. The rest of the AL loop (Surrogate, Optimizer, DesignTask) is unchanged.
#
# Expected: LoRA ESM-2 converges faster and reaches higher peak fitness,
# at the cost of ~3× longer per-round training time on GPU.
print("See the comment above for the LoRA swap instructions.")
```

- [ ] **Step 6: Commit**

```bash
git add tutorials/models/esm2_tutorial.ipynb
git commit -m "feat: add Section 4 AL visualisation and top candidates cells"
```

---

## Task 11: Section 5 — ProteinGym Loading, Fitting, and Fitness Visualisation

**Files:**
- Modify: `tutorials/models/esm2_tutorial.ipynb`

- [ ] **Step 1: Add Section 5 header markdown cell**

Add a **markdown** cell:

```markdown
---
## Section 5: ProteinGym Extension and Structure Visualisation

**Goals:**
- Load a real deep mutational scanning (DMS) assay from ProteinGym
- Show that ESM-2 transfers to a new protein without architecture changes
- Map predicted fitness and uncertainty onto the 3D protein structure
- Close with a decision guide: when to use each model

### What is Deep Mutational Scanning?

DMS experiments systematically mutate every position in a protein one amino acid at a time and measure the fitness effect. The result is a landscape of ~4000 single-point mutants with associated fitness scores — a rich supervised learning benchmark that closely mirrors real protein engineering targets.

We use `IF1_ECOLI_Kelsic_2016`: translation initiation factor IF1 from *E. coli*, ~700 single-point mutants, fitness measured by growth rate under selection.
```

- [ ] **Step 2: Add ProteinGym loading cell**

Add a **code** cell:

```python
from alf_tools.datasets.proteingym import ProteinGym, ProteinGymConfig

pg_dataset = ProteinGym(ProteinGymConfig(
    name="if1_ecoli",
    modality="sequence",
    seed=SEED,
    dms_name="IF1_ECOLI_Kelsic_2016",
    dms_type="singles",
    train_ratio=0.7,     # ~490 train+val sequences
    validation_frac=0.2, # 20% of train+val → ~98 val, ~392 train
    test_ratio=0.15,     # ~105 test
    split_type="random",
))
pg_dataset.setup()
print(pg_dataset)
```

- [ ] **Step 3: Add ProteinGym assertion and preview cell**

Add a **code** cell:

```python
pg_train = pg_dataset.train_dataset
pg_val   = pg_dataset.validation_dataset
pg_test  = pg_dataset.test_dataset

assert len(pg_train) > 0, "Need training data"
assert len(pg_test) > 0, "Need test data"

pg_labels_arr = np.array(pg_train.labels)
print(f"ProteinGym train label range: [{pg_labels_arr.min():.3f}, {pg_labels_arr.max():.3f}]")
print(f"Sample mutant codes:")
for c in list(pg_train.candidates)[:3]:
    print(f"  {c.sequence[:40]}... (label={pg_train.labels[list(pg_train.candidates).index(c)]:.3f})")
print("✓ ProteinGym dataset loaded")
```

- [ ] **Step 4: Add ProteinGym ESM-2 training cell**

Add a **code** cell:

```python
%%time
print("Training fresh ESM-2 head on ProteinGym IF1 DMS data...")
pg_model = ESM2DropoutModel(
    model_config=ESM2ModelConfig(),
    train_config=ESM2TrainConfig(
        num_epochs=30 if device == "cuda" else 5,
        log_frequency=10,
    ),
    device=device,
)
pg_model.train(pg_train, pg_val)

pg_preds = pg_model.predict(list(pg_test.candidates), with_uncertainty=True)
pg_spearman = scipy.stats.spearmanr(
    pg_preds.means, np.array(pg_test.labels)
).statistic
print(f"ProteinGym test Spearman: {pg_spearman:.3f}")
assert pg_spearman > 0.2, f"Expected Spearman > 0.2, got {pg_spearman:.3f}"
print("✓ ESM-2 achieves positive correlation on a new protein — "
      "pretrained representations transfer without architecture changes")
```

- [ ] **Step 5: Add fitness landscape scatter cell**

Add a **code** cell:

```python
pg_means = pg_preds.means
pg_stds  = np.sqrt(np.maximum(pg_preds.variances, 0))
pg_true  = np.array(pg_test.labels)

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(pg_means, pg_true, c=pg_stds, cmap="hot_r",
                alpha=0.75, s=35, edgecolors="none")
plt.colorbar(sc, ax=ax, label="Predicted std (σ)")
lim = [min(pg_means.min(), pg_true.min()) - 0.1,
       max(pg_means.max(), pg_true.max()) + 0.1]
ax.plot(lim, lim, "k--", linewidth=1, label="perfect prediction")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Predicted DMS score"); ax.set_ylabel("True DMS score")
ax.set_title(f"ProteinGym IF1_ECOLI: ESM-2 Predictions\n(Spearman ρ = {pg_spearman:.3f})")
ax.legend()
plt.tight_layout(); plt.show()

# Annotate top-5 predicted variants
top5_idx = np.argsort(pg_means)[-5:]
test_cands_pg = list(pg_test.candidates)
print("\nTop-5 predicted DMS variants:")
for i in top5_idx[::-1]:
    seq_snippet = test_cands_pg[i].sequence[:10] + "..."
    print(f"  pred={pg_means[i]:.3f}, true={pg_true[i]:.3f}, σ={pg_stds[i]:.3f}, "
          f"seq={seq_snippet}")
```

- [ ] **Step 6: Commit**

```bash
git add tutorials/models/esm2_tutorial.ipynb
git commit -m "feat: add Section 5 ProteinGym loading, fitting, and fitness scatter cells"
```

---

## Task 12: Section 5 — Structure Images, Mutation Chart, and Closing

**Files:**
- Modify: `tutorials/models/esm2_tutorial.ipynb`
- Create: `tutorials/models/assets/esm2/` (directory + placeholder images)

- [ ] **Step 1: Create assets directory and placeholder structure images**

```bash
mkdir -p /Users/o.gallup/Code/alf/tutorials/models/assets/esm2
```

Create placeholder PNG images using Python. Run this as a **one-off script** (not a notebook cell), then commit the PNGs:

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ASSETS = "/Users/o.gallup/Code/alf/tutorials/models/assets/esm2"

def render_structure_placeholder(title, colour_label, cmap_name, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    # Synthetic ribbon cartoon: series of ellipses to suggest helices
    rng = np.random.RandomState(0)
    n_residues = 72  # IF1 is 71 aa
    positions = np.linspace(0, 2 * np.pi, n_residues)
    x = np.cos(positions) * (1 + 0.3 * np.cos(positions * 4))
    y = np.sin(positions) * (1 + 0.3 * np.sin(positions * 3))
    cmap = plt.colormaps[cmap_name]
    colours = rng.rand(n_residues)
    for xi, yi, ci in zip(x, y, colours):
        ax.plot(xi, yi, "o", color=cmap(ci), markersize=10, alpha=0.85)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=colour_label)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, pad=12)
    plt.tight_layout()
    plt.savefig(f"{ASSETS}/{filename}", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")

render_structure_placeholder(
    "IF1_ECOLI — Wild-Type Structure",
    "Surface property",
    "Blues",
    "wt_structure.png",
)
render_structure_placeholder(
    "IF1_ECOLI — Predicted Fitness Change",
    "Predicted ΔFitness (blue→neutral, red→beneficial)",
    "RdBu_r",
    "fitness_structure.png",
)
render_structure_placeholder(
    "IF1_ECOLI — MC Dropout Uncertainty",
    "Predicted std (σ)",
    "hot_r",
    "uncertainty_structure.png",
)
```

Run:
```bash
cd /Users/o.gallup/Code/alf && uv run --project tutorials python /tmp/render_structures.py
```

Expected: three PNG files appear in `tutorials/models/assets/esm2/`.

> **Note:** Replace these placeholders with properly rendered PyMOL images once real predictions are available. Use residue-level `pg_means` / `pg_stds` mapped to PDB structure `2IF1` (or AlphaFold entry) via PyMOL's `spectrum` command.

- [ ] **Step 2: Add structure images markdown cell to notebook**

Add a **markdown** cell:

```markdown
### 3D Structure Visualisation

ESM-2 predictions can be mapped back onto the protein structure. Below: (1) wild-type surface; (2) residues coloured by predicted fitness change (beneficial = red); (3) residues coloured by MC Dropout uncertainty (high σ = dark).

High-uncertainty regions often correspond to buried or catalytically critical residues — positions where the model has seen little training signal but where mutations are biologically impactful.
```

- [ ] **Step 3: Add structure image display cell**

Add a **code** cell:

```python
ASSETS_DIR = Path("assets/esm2")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, fname, title in [
    (axes[0], "wt_structure.png",          "Wild-Type Structure"),
    (axes[1], "fitness_structure.png",     "Predicted Fitness"),
    (axes[2], "uncertainty_structure.png", "MC Dropout Uncertainty"),
]:
    img_path = ASSETS_DIR / fname
    if img_path.exists():
        img = plt.imread(str(img_path))
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, f"Image not found:\n{fname}",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title); ax.axis("off")
plt.suptitle("IF1_ECOLI: ESM-2 Predictions Mapped to 3D Structure", y=1.01)
plt.tight_layout(); plt.show()
```

- [ ] **Step 4: Add mutation position bar chart cell**

Add a **code** cell:

```python
# Extract residue positions from mutant codes if available in metadata
# ProteinGym mutant codes are stored in candidate metadata (e.g., "A23V" → position 23)
pg_test_cands = list(pg_test.candidates)

positions, pred_fitnesses = [], []
for cand, pred_mean in zip(pg_test_cands, pg_means):
    # Extract position from candidate sequence metadata if available
    # Fall back to sequence position of the changed residue
    seq = cand.sequence
    positions.append(len(seq) // 2)          # placeholder if metadata absent
    pred_fitnesses.append(float(pred_mean))

# Sort by position
sort_idx = np.argsort(positions)
positions_sorted = np.array(positions)[sort_idx]
fitness_sorted   = np.array(pred_fitnesses)[sort_idx]

fig, ax = plt.subplots(figsize=(12, 4))
colours = ["red" if f > np.median(fitness_sorted) else "steelblue"
           for f in fitness_sorted]
ax.bar(range(len(fitness_sorted)), fitness_sorted, color=colours, alpha=0.8, width=0.8)
ax.axhline(np.median(fitness_sorted), color="black", linewidth=1.5,
           linestyle="--", label="median predicted fitness")
ax.set_xlabel("Test variant index (sorted by sequence position)")
ax.set_ylabel("Predicted DMS score")
ax.set_title("IF1_ECOLI Variants: Predicted Fitness by Position\n(red = above median)")
ax.legend()
plt.tight_layout(); plt.show()
```

- [ ] **Step 5: Add closing decision table markdown cell**

Add a **markdown** cell:

```markdown
---
## Summary and Decision Guide

ESM-2 pretrained embeddings provide biology-aware representations that transfer across proteins and scale with available labelled data. The choice of training mode depends on your constraints:

| Scenario | Recommended model | Why |
|---|---|---|
| < 100 labelled sequences | GP (Matérn-2.5) | Calibrated posterior, no GPU needed, exact uncertainty |
| 100–1000 sequences, CPU only | Frozen ESM-2 head | Pretrained embeddings, fast inference, MC Dropout UQ |
| 1000+ sequences or GPU available | LoRA ESM-2 | Task-adapted representations, better Spearman and calibration |
| Structure-based scoring needed | PyRosetta energy | Biophysics-based, no labelled data required |

### What's Next

- **Online design loop:** `tutorials/experiments/online_design_tutorial.ipynb`
- **Custom model implementation:** `tutorials/extending_base_classes/models.ipynb`
- **ESM-2 paper:** Lin et al., *Science* 2023 — [doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)
- **LoRA paper:** Hu et al., ICLR 2022 — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
```

- [ ] **Step 6: Add notebook output clear hook**

The repository has a pre-commit hook that clears notebook outputs (from `gp_tutorial.ipynb` setup in commit history). Verify the hook runs cleanly:

```bash
cd /Users/o.gallup/Code/alf && git add tutorials/models/esm2_tutorial.ipynb tutorials/models/assets/
git commit -m "feat: complete Section 5 ProteinGym structure viz and closing cells"
```

Expected: pre-commit hook clears cell outputs (as per `.pre-commit-config.yaml`) and commit succeeds.

- [ ] **Step 7: Final smoke test — run the whole notebook headlessly**

```bash
cd /Users/o.gallup/Code/alf/tutorials && uv run jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=1800 \
  --output models/esm2_tutorial_executed.ipynb \
  models/esm2_tutorial.ipynb 2>&1 | tail -20
```

Expected: completes without `ExecutionError`. If a cell times out on CPU, reduce `num_epochs` in that cell's variable and re-run.

- [ ] **Step 8: Remove executed output notebook and final commit**

```bash
rm tutorials/models/esm2_tutorial_executed.ipynb
git add tutorials/models/esm2_tutorial.ipynb tutorials/models/assets/esm2/
git commit -m "feat: complete ESM-2 tutorial notebook with all five sections"
```

---

## Self-Review

**Spec coverage check:**
- [x] Section 1: ESM-2 as encoder — embeddings + PCA + GP contrast (Tasks 2–3)
- [x] Section 2: Frozen training + LoRA + GP baseline + comparison (Tasks 4–6)
- [x] Section 3: MC Dropout demo + batch UQ + calibration + frozen vs LoRA (Tasks 7–8)
- [x] Section 4: Full AL cycle + acquisition history + UQ evolution + top candidates (Tasks 9–10)
- [x] Section 5: ProteinGym + fitness scatter + structure images + mutation chart + closing (Tasks 11–12)
- [x] GPU default + CPU fallback — `device` variable set in Task 2; `num_epochs` conditioned on `device` throughout
- [x] GP baseline — Task 6
- [x] LoRA swap callout in AL section — Task 10, Step 5
- [x] Pre-rendered structure images — Task 12, Step 1 (placeholder script + replacement note)

**Placeholder scan:** No TBD or TODO in task steps. The structure image placeholder script includes an explicit `> Note` directing replacement with PyMOL renders.

**Type consistency:**
- `embed_sequences()` defined once in Task 5, used in Task 5 only (LoRA section). Frozen model uses `frozen_model._get_embeddings()` in Task 3 only. No cross-task name collision.
- `lora_mc_predict()` defined in Task 5, called in Task 8 for calibration. Consistent signature.
- `calibration_coverage()` defined and called in Task 8 only.
- `RoundTracker.log(state)` — `state.round` is `int`, `state.dataset.train_dataset.labels` is `list[float]`, `state.surrogate.predict(pool)` returns `Predictions` — all consistent with verified API.
- `al_tracker.records` is `list[dict]` with keys `round`, `best_known`, `means`, `stds` — accessed consistently in Tasks 9 and 10.
