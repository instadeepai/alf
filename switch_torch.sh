#!/bin/bash
# Script to switch between PyTorch CPU and GPU versions
# GPU version is installed as a temporary override without modifying pyproject.toml

set -e

PYPROJECT_FILE="pyproject.toml"

function show_current() {
    echo "Current PyTorch configuration:"
    echo ""
    echo "In pyproject.toml (default):"
    grep "torch = {" "$PYPROJECT_FILE" || echo "  Not configured"
    echo ""
    echo "Currently installed version:"
    if command -v python &> /dev/null; then
        python -c "import torch; print(f'  PyTorch {torch.__version__}')" 2>/dev/null || echo "  PyTorch not installed or not available"
        python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || true
    else
        echo "  Python not available in current environment"
    fi
}

function switch_to_gpu() {
    echo "Switching to PyTorch GPU (CUDA 12.1)..."
    echo "Note: pyproject.toml will remain unchanged (keeps CPU as default)"

    # First ensure all dependencies are installed (including CPU torch)
    echo "Step 1: Installing all project dependencies from pyproject.toml..."
    uv sync

    # Then override with GPU version
    echo "Step 2: Overriding with PyTorch GPU packages..."
    uv pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch torchvision torchaudio

    echo "✅ Done! PyTorch GPU version installed."
    echo "Note: This is a temporary override. Running 'uv sync' will revert to the pyproject.toml configuration (CPU)."
}

function switch_to_cpu() {
    echo "Switching to PyTorch CPU (syncing with pyproject.toml)..."

    # Sync with pyproject.toml to restore the default CPU version
    echo "Running uv sync to restore CPU version from pyproject.toml..."
    uv sync

    echo "✅ Done! PyTorch CPU version installed."
}

# Main script logic
case "$1" in
    gpu)
        switch_to_gpu
        ;;
    cpu)
        switch_to_cpu
        ;;
    status)
        show_current
        ;;
    *)
        echo "Usage: $0 {cpu|gpu|status}"
        echo ""
        echo "Commands:"
        echo "  cpu     - Switch to PyTorch CPU version (syncs with pyproject.toml)"
        echo "  gpu     - Install PyTorch GPU version (CUDA 12.1) - temporary override"
        echo "  status  - Show current configuration"
        echo ""
        echo "Note: The GPU version is installed without modifying pyproject.toml."
        echo "      Running 'uv sync' will revert to the CPU version defined in pyproject.toml."
        echo ""
        show_current
        exit 1
        ;;
esac
