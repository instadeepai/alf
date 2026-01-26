#!/bin/bash
# Script to switch between PyTorch CPU and GPU versions

set -e

PYPROJECT_FILE="pyproject.toml"

function show_current() {
    echo "Current PyTorch configuration:"
    grep -A 1 "torch = {" "$PYPROJECT_FILE" || echo "  Not configured"
}

function switch_to_gpu() {
    echo "Switching to PyTorch GPU (CUDA 12.1)..."

    # Update the torch source in pyproject.toml
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' 's|torch = { index = "pytorch-cpu" }.*|torch = { index = "pytorch-gpu" }  # GPU version for CUDA support|' "$PYPROJECT_FILE"
    else
        # Linux
        sed -i 's|torch = { index = "pytorch-cpu" }.*|torch = { index = "pytorch-gpu" }  # GPU version for CUDA support|' "$PYPROJECT_FILE"
    fi

    echo "✅ Switched to GPU. Running uv sync..."
    uv sync
    echo "✅ Done! PyTorch GPU version installed."
}

function switch_to_cpu() {
    echo "Switching to PyTorch CPU..."

    # Update the torch source in pyproject.toml
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' 's|torch = { index = "pytorch-gpu" }.*|torch = { index = "pytorch-cpu" }  # Default: CPU version for broader compatibility|' "$PYPROJECT_FILE"
    else
        # Linux
        sed -i 's|torch = { index = "pytorch-gpu" }.*|torch = { index = "pytorch-cpu" }  # Default: CPU version for broader compatibility|' "$PYPROJECT_FILE"
    fi

    echo "✅ Switched to CPU. Running uv sync..."
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
        echo "  cpu     - Switch to PyTorch CPU version"
        echo "  gpu     - Switch to PyTorch GPU version (CUDA 12.1)"
        echo "  status  - Show current configuration"
        echo ""
        show_current
        exit 1
        ;;
esac
