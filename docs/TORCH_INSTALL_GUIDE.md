# PyTorch Installation Guide for ALF

## Overview

The ALF framework now uses a cleaner dependency architecture where PyTorch (CPU version) is installed by default, with GPU support available as an optional extra.

## Architecture

### Core Package (`alf_core`)
- **Does NOT require torch** for normal operation
- Torch is only in `dev` dependencies for testing torch tensor compatibility
- Core dataclasses work with any data type (including torch tensors) without importing torch

### Tools Package (`alf_tools`)
- **Requires torch** as it implements the CNNModel which depends on torch
- Includes both `torch>=1.9.0` and `gpytorch>=1.9.0` in required dependencies
- By default installs CPU version from PyPI

### Tutorials Package (`alf_tutorials`)
- Inherits torch from `alf_tools` dependency
- No torch-specific dependencies needed

## Installation

### Default Installation (CPU PyTorch)

```bash
# From the root directory
uv sync

# This installs torch from PyPI (CPU version on most platforms)
# - macOS: CPU/Apple Metal backend
# - Linux: CPU version
# - Windows: CPU version
```

### GPU Installation (Linux/Windows with CUDA)

```bash
# Install with GPU support (CUDA 12.1)
uv sync --extra torch_gpu

# This will:
# - Use PyTorch CUDA 12.1 index on Linux/Windows
# - Fall back to PyPI (CPU) on macOS (no CUDA support)
```

### How it works

The root `pyproject.toml` configures torch to come from the GPU index when:
1. The `torch_gpu` extra is enabled (`--extra torch_gpu`)
2. AND running on Linux or Windows (`sys_platform == 'linux' or sys_platform == 'win32'`)

Otherwise, torch comes from PyPI (default CPU version).

## Testing

### Default install works:
```bash
uv sync
uv run python -c "from alf_tools.models import CNNModel; print('✓ Success')"
```

### Core tests with torch compatibility:
```bash
cd core
uv run pytest tests/dataclasses/test_candidate.py -k "torch" -v
```

## Configuration Details

### Root `pyproject.toml`
```toml
[project.optional-dependencies]
torch_gpu = []  # Marker extra

[tool.uv.sources]
torch = [
  { index = "pytorch-gpu", marker = "extra == 'torch_gpu' and (sys_platform == 'linux' or sys_platform == 'win32')" },
]

[[tool.uv.index]]
name = "pytorch-gpu"
url = "https://download.pytorch.org/whl/cu121"
explicit = true
```

### Tools `pyproject.toml`
```toml
[project]
dependencies = [
    "torch>=1.9.0",      # Required (CPU from PyPI by default)
    "gpytorch>=1.9.0",   # Required
    # ... other deps
]
```

### Core `pyproject.toml`
```toml
[dependency-groups]
dev = [
    "torch>=1.9.0",  # For testing only
    # ... other dev deps
]
```

## Benefits of This Architecture

1. **Clear dependency separation**: Core doesn't require torch, tools does
2. **No extra flag for CPU**: Default `uv sync` just works
3. **GPU opt-in**: Explicitly enable with `--extra torch_gpu`
4. **Platform-aware**: GPU index only used on Linux/Windows
5. **No conflicting indexes**: Single source of truth for torch resolution
