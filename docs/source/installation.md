# ALF Installation Guide

---

## Package Installation

Use this when you want to use ALF in your own projects. This is what most users need.

### Prerequisites

- Python 3.12 or higher
- GitHub access token for authentication (ALF is hosted on a private GitHub repository)

### Authentication Setup

Create a `.netrc` file in your home directory with your GitHub credentials:

```
machine github.com login <USERNAME> password <TOKEN>
```

For more information on creating personal access tokens, see [GitHub's documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

### Install Packages

```bash
# Install the core package only (minimal dependencies, no PyTorch required)
pip install git+https://github.com/instadeepai/alf.git#subdirectory=core

# Install the tools package (includes PyTorch, models, and datasets)
pip install git+https://github.com/instadeepai/alf.git#subdirectory=tools
```

**What each package provides:**

- **`alf_core`**: Core data structures, tasks, and utilities. No PyTorch required.
- **`alf_tools`**: Machine learning models (CNNModel, GP, ESM-2, etc.), datasets, and acquisition functions. Requires PyTorch.

### Optional Extras

Some models require additional dependencies not installed by default:

```bash
# ESM2 — protein language model (for ESM2Model)
pip install "alf_tools[esm2] @ git+https://github.com/instadeepai/alf.git#subdirectory=tools"

# Chemprop — small-molecule MPNN (for ChempropModel)
pip install "alf_tools[chemprop] @ git+https://github.com/instadeepai/alf.git#subdirectory=tools"

# Both extras together
pip install "alf_tools[esm2,chemprop] @ git+https://github.com/instadeepai/alf.git#subdirectory=tools"
```

### GPU Support

By default, `alf_tools` installs CPU-optimised PyTorch. To use a GPU build:

```bash
pip install git+https://github.com/instadeepai/alf.git#subdirectory=tools
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

```bash
uv sync --extra esm2
uv sync --extra chemprop
uv sync --extra esm2 --extra chemprop
```
