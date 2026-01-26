# PyTorch Installation Guide for ALF

## Overview

The ALF framework uses PyTorch with a simple configuration that defaults to CPU version for broader compatibility. You can easily switch between CPU and GPU versions using the provided script.

## Architecture

### Core Package (`alf_core`)
- **Does NOT require torch** for normal operation
- Torch is only in `dev` dependencies for testing torch tensor compatibility
- Core dataclasses work with any data type (including torch tensors) without importing torch

### Tools Package (`alf_tools`)
- **Requires torch** as it implements the CNNModel which depends on torch
- Includes both `torch>=1.9.0` and `gpytorch>=1.9.0` in required dependencies
- By default installs CPU version from the PyTorch CPU index

### Tutorials Package (`alf_tutorials`)
- Inherits torch from `alf_tools` dependency
- No torch-specific dependencies needed

## Installation

ALF supports two installation approaches:
1. **Root-level installation** (recommended): Install all packages together from the workspace root
2. **Module-level installation**: Install individual packages (core, tools, tutorials) independently

Both approaches support CPU and GPU PyTorch versions.

---

## Root-Level Installation (Recommended)

This approach installs all ALF packages together using the root `pyproject.toml`.

### Default Installation (CPU PyTorch)

```bash
# From the root directory
uv sync

# This installs torch from the PyTorch CPU index
# Compatible with all platforms: macOS, Linux, Windows
```

### GPU Installation (Linux/Windows with CUDA)

To switch to the GPU version of PyTorch (CUDA 12.1), use the provided helper script:

```bash
# Switch to GPU version
./switch_torch.sh gpu

# This will:
# 1. Update root pyproject.toml to use the pytorch-gpu index
# 2. Run uv sync to install GPU-enabled torch
```

### Switching Back to CPU

```bash
# Switch back to CPU version
./switch_torch.sh cpu
```

### Check Current Configuration

```bash
# See which version is currently configured
./switch_torch.sh status
```

### How it works

The root `pyproject.toml` configures which PyTorch index to use in `[tool.uv.sources]`:
- **pytorch-cpu** (default): `https://download.pytorch.org/whl/cpu`
- **pytorch-gpu**: `https://download.pytorch.org/whl/cu121` (CUDA 12.1)

The `switch_torch.sh` script updates this configuration and runs `uv sync` to reinstall torch from the correct index.

### Why do we need a script?

Neither uv, pip, nor Poetry has built-in support for "install package X from index A or index B based on user choice." While these tools support multiple package indexes, they don't provide a native way to switch between them for specific packages without manually editing configuration files.

The `switch_torch.sh` script solves this by:
- Automating the manual edit process
- Ensuring the correct index is configured before installation
- Providing a simple, user-friendly interface (`./switch_torch.sh gpu`)
- Preventing configuration errors and inconsistencies

Without this script, users would need to manually edit `pyproject.toml` files and remember the exact syntax for index configuration.

---

## Module-Level Installation

Each module (`core`, `tools`, `tutorials`) can be installed independently with its own `pyproject.toml`. This is useful for:
- Developing a single module in isolation
- Using only specific ALF components in other projects
- Testing module-specific dependencies

### Default Installation (CPU PyTorch)

```bash
# Install core module (torch in dev dependencies only)
cd core
uv sync

# Install tools module (torch in required dependencies)
cd tools
uv sync

# Install tutorials module (inherits torch from tools)
cd tutorials
uv sync
```

### GPU Installation (Manual Configuration)

Each module's `pyproject.toml` has its own `[tool.uv.sources]` configuration. To use GPU PyTorch:

**Option 1: Manual edit (for development)**

Edit the module's `pyproject.toml`:

```toml
# In core/pyproject.toml, tools/pyproject.toml, or tutorials/pyproject.toml
[tool.uv.sources]
torch = { index = "pytorch-gpu" }  # Change from "pytorch-cpu"
```

Then sync:
```bash
cd core  # or tools, or tutorials
uv sync
```

**Option 2: Use root-level installation**

The `switch_torch.sh` script only modifies the root `pyproject.toml`. For consistent GPU usage across all modules, use root-level installation (see above).

### Module-Level Index Configuration

Each module defines both PyTorch indexes in its `pyproject.toml`:

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

To switch to GPU, change the index value from `"pytorch-cpu"` to `"pytorch-gpu"` in the `[tool.uv.sources]` section.

## Testing

### Root-Level Testing

```bash
# From root directory
uv sync
uv run python -c "from alf_tools.models import CNNModel; print('✓ Success')"
```

### Module-Level Testing

```bash
# Test core with torch compatibility (dev dependency)
cd core
uv sync
uv run pytest tests/dataclasses/test_candidate.py -k "torch" -v

# Test tools (requires torch)
cd tools
uv sync
uv run python -c "from alf_tools.models import CNNModel; print('✓ Success')"
```

## Configuration Details

### Root `pyproject.toml`

The root configuration manages torch installation for the entire workspace:

```toml
[tool.uv.sources]
torch = { index = "pytorch-cpu" }  # Default: CPU version

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-gpu"
url = "https://download.pytorch.org/whl/cu121"
explicit = true
```

### Module-Level `pyproject.toml` Files

Each module (`core`, `tools`, `tutorials`) has its own torch configuration for independent development.

**Tools `pyproject.toml`** (torch required):
```toml
[project]
dependencies = [
    "torch>=1.9.0",      # Required for CNNModel
    "gpytorch>=1.9.0",   # Required
    # ... other deps
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

# pytorch-gpu index also defined (similar to root)

[tool.uv.sources]
torch = { index = "pytorch-cpu" }  # Can be changed to "pytorch-gpu"
```

**Core `pyproject.toml`** (torch in dev only):
```toml
[dependency-groups]
dev = [
    "torch>=1.9.0",  # For testing torch tensor compatibility
    # ... other dev deps
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

# pytorch-gpu index also defined (similar to root)

[tool.uv.sources]
torch = { index = "pytorch-cpu" }  # Can be changed to "pytorch-gpu"
```

**Tutorials `pyproject.toml`**:
- Inherits torch from `alf_tools` dependency
- No direct torch configuration needed

## Benefits of This Architecture

1. **Clear dependency separation**: Core doesn't require torch, tools does
2. **Flexible installation**: Choose root-level (full workspace) or module-level (individual packages)
3. **Simple defaults**: Default `uv sync` installs CPU version which works everywhere
4. **Easy switching at root**: One-command switch between CPU and GPU versions using `switch_torch.sh`
5. **Module independence**: Each module can be developed and installed independently with its own torch configuration
6. **No package conflicts**: Uses explicit PyTorch indexes, not conda or mixed sources
7. **Version control friendly**: Configuration is just one line in each pyproject.toml

## Which Installation Method Should I Use?

**Use root-level installation if:**
- You're developing or using the full ALF framework
- You want consistent PyTorch versions across all modules
- You want the convenience of `switch_torch.sh` for GPU switching
- You're new to ALF (recommended default)

**Use module-level installation if:**
- You only need specific ALF components (e.g., just `alf_core`)
- You're integrating ALF modules into other projects
- You're developing a single module in isolation
- You need different PyTorch configurations per module (advanced use case)
