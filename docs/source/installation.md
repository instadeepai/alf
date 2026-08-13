# ALF Installation Guide

## Package Installation

Use this when you want to use ALF in your own projects. This is what most users need.

### Prerequisites

- Python 3.12 or higher

### Install Packages

```bash
# Install the core package only (minimal dependencies, no PyTorch required)
pip install alf-core

# Install the tools package (includes PyTorch, models, and datasets)
pip install alf-tools
```

**What each package provides:**

- **`alf_core`**: Core data structures, tasks, and utilities. No PyTorch required.
- **`alf_tools`**: Machine learning models (CNNModel, GP, ESM-2, etc.), datasets, and acquisition functions. Requires PyTorch.

### Optional Extras

Heavy ML dependencies (`transformers`, `rdkit`, `chemprop`) are **not** installed by default. Install
only what a given model needs via per-model extras, or grab a whole workflow at once with a workflow
umbrella:

| Extra | Installs | For |
|-------|----------|-----|
| `esm2` | `transformers` | `ESM2Model` |
| `esmfold` | `transformers`, `accelerate` | `ESMFoldModel` |
| `chemprop` | `chemprop` | `ChempropModel` |
| `guacamol` | `rdkit` | GuacaMol dataset/scoring, `GuacaMolOracle`, `SmilesMutationSearch` |
| `protein` | `esm2` + `esmfold` | protein workflow |
| `molecule` | `chemprop` + `guacamol` | small-molecule workflow |

```bash
# ESM2 — protein language model (for ESM2Model)
pip install "alf-tools[esm2]"

# Chemprop — small-molecule MPNN (for ChempropModel)
pip install "alf-tools[chemprop]"

# Whole protein workflow (esm2 + esmfold)
pip install "alf-tools[protein]"

# Several extras together
pip install "alf-tools[esm2,chemprop]"
```

### GPU Support

By default, `alf_tools` installs CPU-optimised PyTorch. To use a GPU build:

```bash
pip install alf-tools
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

> **Note:** Install `alf_tools` first, then override torch separately. Installing both in a single
> step causes an index conflict since `alf_tools` pins torch to the CPU index.

---

## Development Setup

Use this when you want to develop or modify ALF itself.

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- SSH key configured for GitHub

### Quick Start

```bash
# Clone the repository
git clone git@github.com:instadeepai/alf.git
cd alf

# Install all packages with development dependencies
uv sync

# Verify installation
uv run python -c "from alf_tools.models import CNNModel; print('ALF installed successfully')"
```

This installs all ALF packages (`alf_core`, `alf_tools`, `alf_tutorials`) with CPU-optimised
PyTorch by default.

### GPU Support (Optional)

If you have an NVIDIA GPU with CUDA 12.8:

**Option A — Temporary override (no file changes):**

```bash
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu128 --reinstall
```

To revert to CPU, run `uv sync`.

**Option B — Persistent change:**

Edit `tools/pyproject.toml` to change the torch source index from `pytorch-cpu` to `pytorch-gpu`,
then run `uv sync`. Revert this before committing.

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-gpu" },
]
```

> **Note:** After installing the GPU variant, avoid running `uv sync` or `uv run` (which
> auto-syncs) as both will revert torch to the CPU build. To run scripts without triggering a
> sync, use `uv run --no-sync python script.py` or activate the virtualenv directly.

### Optional Extras (Development)

In the cloned repository the `alf_tools` extras are re-exposed as dependency groups, so install them
with `--group` (not `--extra`):

```bash
uv sync --group esm2
uv sync --group chemprop
uv sync --group protein              # esm2 + esmfold
uv sync --group molecule             # chemprop + guacamol
uv sync --group esm2 --group chemprop
```
