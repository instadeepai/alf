# ESM-2 Tutorial Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `tutorials/models/esm2_tutorial.ipynb` — a self-contained tutorial showing frozen ESM-2 embeddings, UMAP of GFP sequences coloured by fluorescence, and fine-tuning with both MLM and log-likelihood objectives.

**Architecture:** Two-file change only. The root `pyproject.toml` gains a `[tutorials]` dependency group; a new notebook is authored as `.ipynb` JSON. No changes to `alf_tools` or `alf_core` source code.

**Tech Stack:** alf_tools `ESM2Model / ESM2ModelConfig / ESM2TrainConfig`, alf_core `GFP / LabelledCandidates / Predictions`, PyTorch, HuggingFace Transformers (`facebook/esm2_t6_8M_UR50D`), umap-learn, matplotlib, pandas, Jupyter.

---

## Key API Facts (read before implementing)

```
ESM2Model(name: str, model_config: ESM2ModelConfig, train_config: ESM2TrainConfig)
ESM2Model.predict(candidates: list[Candidate]) -> Predictions
  Predictions.means     → np.ndarray  shape (N, hidden_dim)   # embeddings
  Predictions.variances → None for ESM-2

ESM2Model.train(train_data: LabelledCandidates, val_data: LabelledCandidates | None)
ESM2Model.get_epoch_metrics()           -> list[SurrogateEpochMetrics]
  SurrogateEpochMetrics.train_loss      → float
  SurrogateEpochMetrics.val_loss        → float | None
  SurrogateEpochMetrics.additional_metrics → dict with keys:
      "train_perplexity", "train_token_accuracy",
      "train_log_likelihood" (LL only),
      "val_perplexity", "val_token_accuracy", "val_log_likelihood" (LL only)
ESM2Model.get_training_summary_metrics() -> dict  e.g.:
      "final_train_loss", "final_train_perplexity", "final_train_token_accuracy",
      "final_val_loss", "final_val_perplexity", "final_val_token_accuracy",
      "final_train_log_likelihood" / "final_val_log_likelihood" (LL only)

LabelledCandidates(candidates: list[Candidate], labels: np.ndarray)
  .candidates  → list[Candidate]
  .labels      → np.ndarray

Candidate.data  → str   (the sequence string)
GFP().load_dataset() -> LabelledCandidates  (1000 rows, labels = medianBrightness np.ndarray)

featurise() returns {"input_ids": Tensor, "attention_mask": Tensor} — NOT embeddings
```

---

## Files

| File | Action |
|------|--------|
| `pyproject.toml` | Add `tutorials` dependency group |
| `tutorials/models/esm2_tutorial.ipynb` | Create — full notebook |

---

### Task 1: Add `[tutorials]` dependency group to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml` (around line 53, after the `docs` group)

- [ ] **Step 1: Add the tutorials group**

In `pyproject.toml`, insert the `tutorials` block inside `[dependency-groups]` so the full section reads:

```toml
[dependency-groups]
dev = [
    "pytest>=6.2.0",
    "pytest-cov>=2.12.0",
    "black>=21.0.0",
    "flake8>=3.9.0",
    "mypy>=0.910",
    "pre-commit>=4.1.0",
]

docs = [
    "sphinx>=7.0.0",
    "furo>=2023.0.0",
    "myst-parser>=2.0.0",
    "sphinx-autodoc-typehints>=1.24.0",
]

tutorials = [
    "matplotlib>=3.7.0",
    "umap-learn>=0.5.0",
]
```

- [ ] **Step 2: Install and verify**

```bash
uv sync --group tutorials
python -c "import umap; import matplotlib; print('tutorials deps OK')"
```

Expected:
```
tutorials deps OK
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add tutorials dependency group with matplotlib and umap-learn"
```

---

### Task 2: Create `tutorials/models/esm2_tutorial.ipynb`

**Files:**
- Create: `tutorials/models/esm2_tutorial.ipynb`

- [ ] **Step 1: Write the notebook file**

Write the following JSON to `tutorials/models/esm2_tutorial.ipynb` exactly:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "cell-0001",
   "metadata": {},
   "source": [
    "# ESM-2 Tutorial: Embeddings and Fine-Tuning on GFP\n",
    "\n",
    "**ESM-2** (Evolutionary Scale Modeling 2) is a protein language model from Meta AI, pre-trained on hundreds of millions of protein sequences. This notebook shows you how to:\n",
    "\n",
    "1. Load ESM-2 as a frozen embedding extractor\n",
    "2. Extract per-sequence embeddings for GFP (Green Fluorescent Protein) variants\n",
    "3. Visualise the embedding space with UMAP, coloured by fluorescence\n",
    "4. Fine-tune ESM-2 with **Masked Language Modelling (MLM)**\n",
    "5. Fine-tune ESM-2 with **log-likelihood** and compare the two objectives\n",
    "6. Use `predict()` on a fine-tuned model\n",
    "\n",
    "**Prerequisites:** install the `tutorials` dependency group before running:\n",
    "```bash\n",
    "uv sync --group tutorials\n",
    "```\n",
    "\n",
    "**Expected runtime:** < 5 minutes on CPU using `facebook/esm2_t6_8M_UR50D` (8M parameters, ~31 MB download on first run)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-0002",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import torch\n",
    "import umap\n",
    "\n",
    "from alf_core.dataclasses import LabelledCandidates\n",
    "from alf_tools.datasets.gfp import GFP\n",
    "from alf_tools.models.esm2 import ESM2Model, ESM2ModelConfig, ESM2TrainConfig\n",
    "\n",
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "print(f\"Using device: {device}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-0003",
   "metadata": {},
   "source": [
    "## 1. Load the GFP Dataset\n",
    "\n",
    "**Green Fluorescent Protein (GFP)** is a classic protein engineering benchmark. The dataset contains ~1 000 nucleotide sequences encoding GFP variants, each labelled with `medianBrightness` — a proxy for how well the variant fluoresces.\n",
    "\n",
    "`GFP.load_dataset()` downloads and caches the CSV on first call, then returns a `LabelledCandidates` object where each `Candidate.data` is a nucleotide sequence string and `.labels` is a `numpy` array of brightness scores.\n",
    "\n",
    "We use 200 sequences for embedding (fast enough for UMAP) and 50 for fine-tuning (keeps CPU training under 2 minutes)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-0004",
   "metadata": {},
   "outputs": [],
   "source": [
    "gfp = GFP()\n",
    "data: LabelledCandidates = gfp.load_dataset()\n",
    "\n",
    "print(f\"Total sequences : {len(data.candidates)}\")\n",
    "print(f\"Example sequence: {data.candidates[0].data[:40]}...\")\n",
    "print(f\"Brightness range: {data.labels.min():.2f} – {data.labels.max():.2f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-0005",
   "metadata": {},
   "outputs": [],
   "source": [
    "embed_candidates    = data.candidates[:200]\n",
    "embed_labels        = data.labels[:200]       # numpy array, shape (200,)\n",
    "\n",
    "finetune_candidates = data.candidates[:50]\n",
    "finetune_labels     = data.labels[:50]        # numpy array, shape (50,)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-0006",
   "metadata": {},
   "source": [
    "## 2. Initialise ESM-2 as a Frozen Embedding Extractor\n",
    "\n",
    "**`ESM2ModelConfig`** controls the architecture:\n",
    "- `model_id` — any `facebook/esm2_*` HuggingFace checkpoint\n",
    "- `pooling` — how to collapse per-token hidden states to one vector: `\"mean\"` (average over non-padding positions), `\"cls\"` (first token), or `\"last_hidden_state\"` (full sequence tensor)\n",
    "- `repr_layer` — which transformer layer to read; `-1` is the final layer\n",
    "\n",
    "**`ESM2TrainConfig(freeze_backbone=True)`** makes `train()` a no-op: weights are never updated. This gives us a pure feature extractor that runs in inference mode only.\n",
    "\n",
    "> **Contributors:** `ESM2Model.predict()` in `tools/alf_tools/models/esm2.py` tokenizes via `featurise()`, runs a forward pass with `output_hidden_states=True`, then pools the selected hidden layer."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-0007",
   "metadata": {},
   "outputs": [],
   "source": [
    "model_cfg = ESM2ModelConfig(\n",
    "    model_id=\"facebook/esm2_t6_8M_UR50D\",\n",
    "    pooling=\"mean\",\n",
    "    repr_layer=-1,\n",
    ")\n",
    "train_cfg = ESM2TrainConfig(freeze_backbone=True)\n",
    "\n",
    "model = ESM2Model(name=\"esm2-gfp\", model_config=model_cfg, train_config=train_cfg)\n",
    "print(model)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-0008",
   "metadata": {},
   "source": [
    "## 3. Extract Sequence Embeddings\n",
    "\n",
    "`predict()` runs a batched forward pass and returns a `Predictions` object. `predictions.means` is a `(N, hidden_dim)` numpy array — one 320-dimensional vector per sequence.\n",
    "\n",
    "Note: `featurise()` only tokenizes (returns `{\"input_ids\", \"attention_mask\"}`); the forward pass and pooling live entirely in `predict()`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-0009",
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_frozen = model.predict(embed_candidates)\n",
    "X = predictions_frozen.means  # shape (200, 320) — already numpy\n",
    "print(f\"Embedding shape: {X.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-0010",
   "metadata": {},
   "source": [
    "## 4. UMAP of Frozen Embeddings\n",
    "\n",
    "We project 320-dimensional embeddings to 2D with UMAP and colour each point by fluorescence. If ESM-2 has captured meaningful structure in GFP sequence space, high-brightness variants should cluster rather than scatter uniformly."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-0011",
   "metadata": {},
   "outputs": [],
   "source": [
    "reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)\n",
    "coords  = reducer.fit_transform(X)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(7, 5))\n",
    "sc = ax.scatter(coords[:, 0], coords[:, 1], c=embed_labels, cmap=\"viridis\", s=15, alpha=0.8)\n",
    "plt.colorbar(sc, ax=ax, label=\"Median Brightness\")\n",
    "ax.set_title(\"UMAP of ESM-2 embeddings (frozen) — GFP sequences\")\n",
    "ax.set_xlabel(\"UMAP 1\")\n",
    "ax.set_ylabel(\"UMAP 2\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-0012",
   "metadata": {},
   "source": [
    "## 5. Fine-Tune with Masked Language Modelling (MLM)\n",
    "\n",
    "MLM is the same objective ESM-2 was pre-trained with: randomly mask 15% of amino-acid tokens (never special tokens like `[CLS]` / `[EOS]`), then train the model to predict the originals. Fine-tuning on GFP sequences nudges representations toward this protein family.\n",
    "\n",
    "Masked tokens are handled with an 80 / 10 / 10 split:\n",
    "- 80% replaced with `[MASK]`\n",
    "- 10% replaced with a random vocabulary token\n",
    "- 10% left unchanged (still counted in the loss)\n",
    "\n",
    "> **Contributors:** masking logic is in `ESM2Model._mask_tokens()` in `tools/alf_tools/models/esm2.py`.\n",
    "\n",
    "We use 40 train / 10 val sequences and 2 epochs — enough to see the loss move on CPU in under 2 minutes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-0013",
   "metadata": {},
   "outputs": [],
   "source": [
    "n_train    = 40\n",
    "train_data = LabelledCandidates(\n",
    "    candidates=finetune_candidates[:n_train],\n",
    "    labels=finetune_labels[:n_train],\n",
    ")\n",
    "val_data = LabelledCandidates(\n",
    "    candidates=finetune_candidates[n_train:],\n",
    "    labels=finetune_labels[n_train:],\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-0014",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_cfg_mlm = ESM2TrainConfig(\n",
    "    freeze_backbone=False,\n",
    "    loss_type=\"mlm\",\n",
    "    num_epochs=2,\n",
    "    batch_size=4,\n",
    "    learning_rate=1e-4,\n",
    "    mask_probability=0.15,\n",
    ")\n",
    "model_mlm = ESM2Model(name=\"esm2-gfp-mlm\", model_config=model_cfg, train_config=train_cfg_mlm)\n",
    "model_mlm.train(train_data, val_data)\n",
    "\n",
    "metrics_mlm = model_mlm.get_training_summary_metrics()\n",
    "print(\"MLM summary metrics:\", metrics_mlm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-0015",
   "metadata": {},
   "outputs": [],
   "source": [
    "epoch_metrics = model_mlm.get_epoch_metrics()\n",
    "train_losses  = [m.train_loss for m in epoch_metrics]\n",
    "val_losses    = [m.val_loss   for m in epoch_metrics if m.val_loss is not None]\n",
    "epochs        = list(range(1, len(train_losses) + 1))\n",
    "\n",
    "plt.figure(figsize=(5, 3))\n",
    "plt.plot(epochs, train_losses, marker=\"o\", label=\"Train loss\")\n",
    "if val_losses:\n",
    "    plt.plot(epochs[:len(val_losses)], val_losses, marker=\"s\", linestyle=\"--\", label=\"Val loss\")\n",
    "plt.xlabel(\"Epoch\")\n",
    "plt.ylabel(\"Loss\")\n",
    "plt.title(\"MLM fine-tuning loss\")\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-0016",
   "metadata": {},
   "source": [
    "## 6. Fine-Tune with Log-Likelihood\n",
    "\n",
    "The log-likelihood objective differs from MLM in one key way: **all** non-special tokens are masked simultaneously. The model must reconstruct the full sequence at once, approximating its pseudo-log-likelihood (PLL) — a stricter signal that is closer to how ESM-2 fitness scores are typically computed.\n",
    "\n",
    "The only config change vs. MLM is `loss_type=\"log_likelihood\"`. Because all positions are masked, losses will typically be higher than MLM.\n",
    "\n",
    "> **Contributors:** masking logic is in `ESM2Model._compute_log_likelihood_labels()` in `tools/alf_tools/models/esm2.py`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-0017",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_cfg_ll = ESM2TrainConfig(\n",
    "    freeze_backbone=False,\n",
    "    loss_type=\"log_likelihood\",\n",
    "    num_epochs=2,\n",
    "    batch_size=4,\n",
    "    learning_rate=1e-4,\n",
    ")\n",
    "model_ll = ESM2Model(name=\"esm2-gfp-ll\", model_config=model_cfg, train_config=train_cfg_ll)\n",
    "model_ll.train(train_data, val_data)\n",
    "\n",
    "metrics_ll = model_ll.get_training_summary_metrics()\n",
    "print(\"Log-likelihood summary metrics:\", metrics_ll)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-0018",
   "metadata": {},
   "outputs": [],
   "source": [
    "comparison = pd.DataFrame(\n",
    "    {\"MLM\": metrics_mlm, \"Log-likelihood\": metrics_ll}\n",
    ").T\n",
    "display(comparison)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-0019",
   "metadata": {},
   "source": [
    "## 7. Predict with a Fine-Tuned Model\n",
    "\n",
    "After fine-tuning, `predict()` returns embeddings from the adapted model — `predictions.means` is still a `(N, hidden_dim)` numpy array, now reflecting the GFP-adapted representations. These can be fed directly into an ALF active learning loop as surrogate model features.\n",
    "\n",
    "(`predictions.variances` is `None` for ESM-2, since the model does not produce uncertainty estimates.)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-0020",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_candidates = data.candidates[200:210]\n",
    "predictions = model_mlm.predict(test_candidates)\n",
    "\n",
    "print(f\"Embedding shape      : {predictions.means.shape}\")        # (10, 320)\n",
    "print(f\"First 5 dims (seq 0) : {predictions.means[0, :5]}\")\n",
    "print(f\"Variances            : {predictions.variances}\")           # None"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Commit the unexecuted notebook**

```bash
git add tutorials/models/esm2_tutorial.ipynb
git commit -m "feat: add ESM-2 tutorial notebook"
```

---

### Task 3: Execute and verify the notebook

**Files:**
- Modify in-place: `tutorials/models/esm2_tutorial.ipynb` (adds cell outputs)

- [ ] **Step 1: Run the notebook end-to-end**

```bash
cd /Users/o.gallup/Code/alf
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 \
    tutorials/models/esm2_tutorial.ipynb
```

Expected: command exits 0. All 20 cells execute without error. The notebook file is updated with outputs in-place.

If `jupyter` is not on the path, run:
```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 \
    tutorials/models/esm2_tutorial.ipynb
```

- [ ] **Step 2: Spot-check key outputs**

Open the notebook and verify:
- Cell `cell-0002`: prints `Using device: cpu` (or `cuda`)
- Cell `cell-0004`: prints total sequences = 1000, a sequence string, and a brightness range
- Cell `cell-0009`: prints `Embedding shape: (200, 320)`
- Cell `cell-0011`: shows a UMAP scatter plot with a viridis colorbar
- Cell `cell-0014`: prints MLM summary metrics dict (non-empty)
- Cell `cell-0015`: shows a loss curve plot
- Cell `cell-0017`: prints log-likelihood summary metrics dict (non-empty, includes `final_train_log_likelihood`)
- Cell `cell-0018`: displays a 2-row DataFrame with columns like `final_train_loss`, `final_val_loss`, `final_train_perplexity`, etc.
- Cell `cell-0020`: prints `Embedding shape: (10, 320)`, first 5 dims, and `Variances: None`

- [ ] **Step 3: Commit with outputs**

```bash
git add tutorials/models/esm2_tutorial.ipynb
git commit -m "feat: execute ESM-2 tutorial notebook and save outputs"
```
