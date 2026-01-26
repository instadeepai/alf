# PyTorch Installation Quick Reference

## CPU Installation (Default)

Works from **any directory** - root or subpackages:

```bash
# From root
uv sync

# From tools
cd tools && uv sync

# From tutorials
cd tutorials && uv sync

# From core (dev only - torch for tests)
cd core && uv sync
```

**Result**: Installs CPU torch from PyPI (works on macOS, Linux, Windows)

---

## GPU Installation (CUDA 12.1)

### Option 1: From Root (Recommended)

```bash
cd /path/to/alf
uv sync --extra torch_gpu
```

**Pros**:
- Clean, configured in `pyproject.toml`
- Only works on Linux/Windows (macOS fallback to CPU)

**Cons**:
- Must be in root directory

---

### Option 2: From Subpackages (CLI Flag)

```bash
# From tools
cd tools
uv sync --extra-index-url https://download.pytorch.org/whl/cu121

# From tutorials
cd tutorials
uv sync --extra-index-url https://download.pytorch.org/whl/cu121

# Or use environment variable
export UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121
cd tools && uv sync
```

**Pros**:
- Works from any subpackage
- Same result as root method

**Cons**:
- Must pass CLI flag/env var
- Not configured in `pyproject.toml`

---

## Summary Table

| Location | CPU | GPU (via extra) | GPU (via CLI) |
|----------|-----|-----------------|---------------|
| **Root** | `uv sync` | `uv sync --extra torch_gpu` ✅ | `uv sync --extra-index-url ...` |
| **Tools** | `uv sync` | ❌ Not available | `uv sync --extra-index-url ...` ✅ |
| **Tutorials** | `uv sync` | ❌ Not available | `uv sync --extra-index-url ...` ✅ |
| **Core** | `uv sync` | ❌ Not needed | ❌ Not needed |

---

## Why This Design?

**Q: Why isn't GPU config in all subpackages?**

A: Adding GPU index configuration to subpackages causes `uv` to attempt GPU resolution even for CPU-only installs, breaking the build. Keeping GPU config only at the root avoids these conflicts.

**Q: Can I still develop with GPU from subpackages?**

A: Yes! Use the CLI flag method shown in Option 2 above.

**Q: Which method should I use?**

A:
- **For most users**: Use the root directory with `--extra torch_gpu`
- **For subpackage development with GPU**: Use the `--extra-index-url` CLI flag
- **For CPU (default)**: Just `uv sync` from anywhere

---

## Verification

Check your torch installation:

```bash
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

**CPU version**: Shows something like `Torch: 2.10.0, CUDA: False`
**GPU version**: Shows something like `Torch: 2.5.1+cu121, CUDA: True` (on CUDA systems)
