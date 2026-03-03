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
git clone git@github.com:instadeepai/alf.git
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
- CUDA 12.1 compatible drivers

**Switch to GPU PyTorch:**

```bash
# Install GPU version (CUDA 12.1) - temporary override
./switch_torch.sh gpu

# Switch back to CPU version
./switch_torch.sh cpu

# Check current configuration
./switch_torch.sh status
```

**How it works:**

The script installs the GPU version of PyTorch as a temporary override without modifying `pyproject.toml`:

1. First runs `uv sync` to install all dependencies from `pyproject.toml` (including CPU PyTorch)
2. Then overrides PyTorch using `uv pip install --index-url https://download.pytorch.org/whl/cu121`

**Note:** This is a temporary override. Running `uv sync` will revert to the CPU version defined in `pyproject.toml`.

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

For more information on creating personal access tokens, see [GitHub's documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

### Install Packages

```bash
# Install core package only (minimal dependencies)
pip install git+https://github.com/instadeepai/alf.git#subdirectory=core

# Install tools package (includes PyTorch and models)
pip install git+https://github.com/instadeepai/alf.git#subdirectory=tools
```

**What each package provides:**

- **`alf_core`**: Core data structures, tasks, and utilities. Does not require PyTorch.
- **`alf_tools`**: Machine learning models (CNNModel, etc.), datasets, and acquisition functions. Requires PyTorch.

Both packages install CPU-optimized PyTorch by default for broad compatibility.

---

## Architecture

### Core Package (`alf_core`)
- **Does NOT require PyTorch** for normal operation
- Core dataclasses work with any data type (including torch tensors) without importing torch
- Minimal dependencies: `numpy`, `pandas`, `pydantic`, `scipy`

### Tools Package (`alf_tools`)
- **Requires PyTorch** for CNNModel and other ML components
- Includes `torch>=1.9.0` and `gpytorch>=1.9.0` in dependencies
- Installs CPU-optimized PyTorch by default

### Tutorials Package (`alf_tutorials`)
- Example notebooks and usage guides
- Inherits PyTorch from `alf_tools` dependency

---

## PyTorch Configuration Details

ALF uses explicit PyTorch indexes in `pyproject.toml` to control which PyTorch version gets installed:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-gpu"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }  # Default: CPU version
```

To manually switch to GPU for development, edit `pyproject.toml`:

```toml
[tool.uv.sources]
torch = { index = "pytorch-gpu" }  # Change from "pytorch-cpu"
```

Then run:
```bash
uv sync
```

**Note:** The `switch_torch.sh` script is preferred as it avoids modifying `pyproject.toml` and uses a temporary override instead.
