# ESM-2 Tutorial Notebook Design

**Date:** 2026-05-08  
**Branch:** `feat/esm-2_simplified`  
**Outcome:** A short tutorial notebook at `tutorials/models/esm2_tutorial.ipynb`

---

## Goal

A self-contained Jupyter notebook that teaches both framework users (clean API) and contributors (internals) how to:

1. Initialise ESM-2 as a frozen embedding extractor
2. Extract protein embeddings from the GFP dataset
3. Visualise those embeddings with UMAP coloured by fluorescence
4. Fine-tune ESM-2 with MLM
5. Fine-tune ESM-2 with log-likelihood
6. Compare the two fine-tuning objectives via summary metrics
7. Run `predict()` on a fine-tuned model

---

## Files Changed

| File | Change |
|------|--------|
| `tutorials/models/esm2_tutorial.ipynb` | **New** — the tutorial notebook |
| `pyproject.toml` | Add `[tutorials]` optional dependency group |

### `pyproject.toml` addition

```toml
[dependency-groups]
tutorials = [
    "matplotlib>=3.7.0",
    "umap-learn>=0.5.0",
]
```

Install with: `uv sync --group tutorials`

---

## Notebook Structure

~20 code cells, ~15 markdown cells. Target runtime: **< 5 minutes on CPU** using `facebook/esm2_t6_8M_UR50D` (8M params).

### Cell-by-cell plan

#### Section 0 — Setup

*Markdown:* Brief intro to ESM-2, what the notebook covers, install note (`uv sync --group tutorials`).

*Code:*
```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap

from alf_tools.datasets.gfp import GFP
from alf_tools.models.esm2 import ESM2Model, ESM2ModelConfig, ESM2TrainConfig
from alf_core.dataclasses import LabelledCandidates

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

---

#### Section 1 — Load GFP Dataset

*Markdown:* Explain GFP (Green Fluorescent Protein), what `medianBrightness` measures, and why it's a good benchmark for protein embeddings.

*Code:*
```python
gfp = GFP()
data: LabelledCandidates = gfp.load_dataset()

print(f"Number of sequences: {len(data.candidates)}")
print(f"Example sequence: {data.candidates[0].data[:30]}...")
print(f"Brightness range: {data.labels.min():.2f} – {data.labels.max():.2f}")
```

*Markdown:* Note that the full dataset has 1000 sequences; we'll use 200 for embeddings and 50 for fine-tuning to keep runtime short.

```python
embed_candidates = data.candidates[:200]
embed_labels     = np.array(data.labels[:200])

finetune_candidates = data.candidates[:50]
finetune_labels     = np.array(data.labels[:50])
```

---

#### Section 2 — Initialise Frozen ESM-2

*Markdown:* Explain `ESM2ModelConfig` fields (`model_id`, `pooling`, `repr_layer`) and why `freeze_backbone=True` gives us a pure embedding extractor. Point contributors to `esm2.py` for internals.

*Code:*
```python
model_cfg = ESM2ModelConfig(
    model_id="facebook/esm2_t6_8M_UR50D",
    pooling="mean",       # pool all token embeddings → one vector per sequence
    repr_layer=-1,        # use the final transformer layer
)
train_cfg = ESM2TrainConfig(freeze_backbone=True)

model = ESM2Model(name="esm2-gfp", model_config=model_cfg, train_config=train_cfg)
print(model)
```

---

#### Section 3 — Extract Embeddings

*Markdown:* `predict()` runs a forward pass and returns a `Predictions` object whose `.means` is a `(N, d)` numpy array — one embedding per sequence. (`featurise()` just tokenizes; the actual forward pass lives in `predict()`.)

*Code:*
```python
predictions_frozen = model.predict(embed_candidates)
X = predictions_frozen.means   # shape (200, 320) — already numpy
print(f"Embedding shape: {X.shape}")
```

---

#### Section 4 — UMAP Visualisation

*Markdown:* We reduce 320-dimensional embeddings to 2D with UMAP and colour by fluorescence. Clusters correspond to regions of sequence space that ESM-2's pre-training has learned to distinguish.

*Code:*
```python
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
coords  = reducer.fit_transform(X)

fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(coords[:, 0], coords[:, 1], c=embed_labels, cmap="viridis",
                s=15, alpha=0.8)
plt.colorbar(sc, ax=ax, label="Median Brightness")
ax.set_title("UMAP of ESM-2 embeddings (frozen) — GFP sequences")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
plt.tight_layout()
plt.show()
```

---

#### Section 5 — Fine-tune with MLM

*Markdown:* Explain what MLM does: randomly masks 15% of amino-acid tokens and trains the model to predict them. This is the same objective ESM-2 was pre-trained with, so fine-tuning on GFP sequences adapts the representations to this protein family. Point contributors to `_mask_tokens()` in `esm2.py` for the 80/10/10 masking split.

*Code — build train/val splits:*
```python
from alf_core.dataclasses import LabelledCandidates

n_train = 40
train_data = LabelledCandidates(
    candidates=finetune_candidates[:n_train],
    labels=finetune_labels[:n_train],
)
val_data = LabelledCandidates(
    candidates=finetune_candidates[n_train:],
    labels=finetune_labels[n_train:],
)
```

*Code — MLM training:*
```python
train_cfg_mlm = ESM2TrainConfig(
    freeze_backbone=False,
    loss_type="mlm",
    num_epochs=2,
    batch_size=4,
    learning_rate=1e-4,
    mask_probability=0.15,
)
model_mlm = ESM2Model(name="esm2-gfp-mlm", model_config=model_cfg, train_config=train_cfg_mlm)
model_mlm.train(train_data, val_data)

metrics_mlm = model_mlm.get_training_summary_metrics()
print("MLM summary metrics:", metrics_mlm)
```

*Code — plot loss curve:*
```python
epoch_metrics = model_mlm.get_epoch_metrics()
train_losses  = [m.train_loss for m in epoch_metrics]

plt.figure(figsize=(5, 3))
plt.plot(train_losses, marker="o", label="Train loss (MLM)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLM fine-tuning loss")
plt.legend()
plt.tight_layout()
plt.show()
```

---

#### Section 6 — Fine-tune with Log-Likelihood

*Markdown:* Log-likelihood training masks **all** non-special tokens simultaneously and trains the model to reconstruct the full sequence. This is a stricter objective: the model must predict every position at once, producing a pseudo-log-likelihood score for the whole sequence. Point contributors to `_compute_log_likelihood_labels()` in `esm2.py`.

*Code:*
```python
train_cfg_ll = ESM2TrainConfig(
    freeze_backbone=False,
    loss_type="log_likelihood",
    num_epochs=2,
    batch_size=4,
    learning_rate=1e-4,
)
model_ll = ESM2Model(name="esm2-gfp-ll", model_config=model_cfg, train_config=train_cfg_ll)
model_ll.train(train_data, val_data)

metrics_ll = model_ll.get_training_summary_metrics()
print("Log-likelihood summary metrics:", metrics_ll)
```

*Code — side-by-side comparison:*
```python
import pandas as pd

comparison = pd.DataFrame({
    "MLM":              metrics_mlm,
    "Log-likelihood":   metrics_ll,
}).T
display(comparison)
```

---

#### Section 7 — Predict with a Fine-tuned Model

*Markdown:* After fine-tuning, `predict()` returns a `Predictions` object with `mean` and `variance` — ready to use as a surrogate in an ALF active learning loop.

*Code:*
```python
test_candidates = data.candidates[200:210]
predictions = model_mlm.predict(test_candidates)

# .means is a (N, hidden_dim) array of embeddings; .variances is None for ESM-2
print(f"Embedding shape:  {predictions.means.shape}")
print(f"First embedding (first 5 dims): {predictions.means[0, :5]}")
```

---

## Constraints

- **Model:** `facebook/esm2_t6_8M_UR50D` throughout — smallest available, downloads ~31 MB.
- **Subset sizes:** 200 sequences for embedding, 50 for fine-tuning (40 train / 10 val).
- **Epochs:** 2 for both fine-tuning runs.
- **Batch size:** 4 — works on CPU without OOM.
- **Random seed:** `random_state=42` for UMAP reproducibility.
- **No `sample()` calls** — not implemented on `ESM2Model`.

---

## Out of Scope

- GPU-specific setup or multi-GPU training
- Hyperparameter tuning
- Downstream active learning integration (covered in experiment tutorials)
- ESM-2 model size comparison
- Before/after UMAP comparison post fine-tuning
