# ALF Installation Guide

ALF provides two installation methods depending on your use case:

1. **Development Installation** - For developing ALF itself
2. **Package Installation** - For using ALF modules in other projects

---

## Development Installation

Use this when you want to develop or modify ALF itself.

### Quick Start

```bash
# Clone the repository
git clone https://github.com/instadeepai/alf.git
cd alf

# Install all packages with development dependencies
uv sync

# Verify installation
uv run python -c "from alf_tools.models import CNNModel; print('✓ ALF installed successfully')"
```

This installs all ALF packages (`alf_core`, `alf_tools`, `alf_tutorials`) with CPU-optimized PyTorch by default, which works on all platforms (macOS, Linux, Windows).

### GPU Support (Optional)

If you have an NVIDIA GPU and want CUDA acceleration:

**Prerequisites:**

- NVIDIA GPU with CUDA support
- Linux or Windows (macOS does not support CUDA)
- CUDA 12.8 compatible drivers

**Option A — Temporary override (no file changes):**

```bash
# First install all dependencies (CPU torch)
uv sync

# Then override torch with the GPU build
uv pip install torch --index-url https://download.pytorch.org/whl/cu128 --reinstall
```

`uv pip install` bypasses `pyproject.toml` sources, so this works regardless of the index configured there. To revert to CPU, just run `uv sync` again.

**Option B — Edit `tools/pyproject.toml` (persistent):**

Change the torch source index:

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-gpu" },  # Change from "pytorch-cpu"
]
```

Then run `uv sync`. Note: revert this change before committing if you want CPU as the project default.

### Verify GPU Installation

```bash
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If CUDA is properly configured, this prints `CUDA available: True`.

---

## Package Installation

Use this when you want to use ALF modules in other projects. Each module (`alf_core`, `alf_tools`) can be installed independently as a built package.

### Prerequisites

- Python 3.12 or higher
- GitHub access token for authentication

### Authentication Setup

Create a `.netrc` file in your home directory with your GitHub credentials:

```
machine github.com login <USERNAME> password <TOKEN>
```

For more information on creating personal access tokens, see [GitHub&#39;s documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

### Install Packages

```bash
# Install core package only (minimal dependencies)
pip install git+https://github.com/instadeepai/alf.git#subdirectory=core

# Install tools package (includes PyTorch and models)
pip install git+https://github.com/instadeepai/alf.git#subdirectory=tools
```

**What each package provides:**

- **`alf_core`**: Core data structures, tasks, and utilities. Does not require PyTorch.
- **`alf_tools`**: Machine learning models (CNNModel, etc.), datasets, and acquisition functions. Requires PyTorch `>=2.10.0`.

Both packages install CPU-optimized PyTorch by default for broad compatibility.

**GPU variant:** If your project requires GPU PyTorch, install `alf_tools` first (which will pull in the CPU build), then override torch separately:

```bash
pip install git+https://github.com/instadeepai/alf.git#subdirectory=tools
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**Note 1:** Attempting to install a GPU torch alongside `alf_tools` in a single step will cause an index conflict, since `alf_tools` pins torch to the CPU index.

**Note 2:** When using uv to install the alf repo with torch[gpu], make sure to use the `--reinstall` flag as shown below.

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128 --reinstall
```

**Note 3:** After installing the GPU variant, avoid running `uv sync` or `uv run` (which auto-syncs), as both will revert torch back to the CPU variant defined in `pyproject.toml`. To run scripts without triggering a sync, use `--no-sync` as shown below or use python executable after activating the virtual environment.

```bash
uv run --no-sync python script.py
```

---

## Architecture

### Core Package (`alf_core`)

- **Does NOT require PyTorch** for normal operation
- Core dataclasses work with any data type (including torch tensors) without importing torch
- Minimal dependencies: `numpy`, `pandas`, `pydantic`, `scipy`

### Tools Package (`alf_tools`)

- **Requires PyTorch** for CNNModel and other ML components
- Includes `torch>=2.10.0` and `gpytorch>=1.9.0` in dependencies
- Installs CPU-optimized PyTorch by default

### Tutorials Package (`alf_tutorials`)

- Example notebooks and usage guides
- Inherits PyTorch from `alf_tools` dependency

---

## PyTorch Configuration Details

`tools/pyproject.toml` defines two named PyTorch indexes:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-gpu"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

The active index is selected via `[tool.uv.sources]`. CPU is the default:

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cpu" },
]
```

Switch to `pytorch-gpu` and run `uv sync` to install the CUDA 12.8 build. See [GPU Support](#gpu-support-optional) above for the full workflow.
