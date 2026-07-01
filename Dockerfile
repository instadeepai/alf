# syntax=docker/dockerfile:1

# ALF examples & tutorials image
# ------------------------------
# A ready-to-run environment for the ALF tutorial notebooks and the
# benchmark_examples/ scripts, with CPU-only PyTorch (the documented default).
#
# Build:
#   docker build -t alf .
#
# Run the tutorials (JupyterLab on http://localhost:8888):
#   docker run --rm -p 8888:8888 alf
#
# Run a benchmark example instead:
#   docker run --rm alf uv run --no-sync \
#       python benchmark_examples/benchmarking_surrogates.py --dataset gfp --num-seeds 3
#
# See docs/source/how-to/run-with-docker.md for the full guide.

# uv's official image ships Python 3.12 (ALF requires >=3.12) plus the uv binary.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Copy mode avoids hardlink warnings when the cache and the venv span layers.
ENV UV_LINK_MODE=copy

WORKDIR /app

# core, tools and tutorials are editable installs, so their source must be present
# when uv builds them — copy the whole repo before syncing.
COPY . /app

# Install all ALF packages plus the tutorial and benchmark dependency groups
# (jupyterlab, matplotlib, umap, ESM-2/transformers, ...). Torch resolves to the
# CPU build via the index pin in tools/pyproject.toml.
RUN uv sync --group tutorials --group benchmark

EXPOSE 8888

# Default: serve the tutorial notebooks. --no-sync uses the environment built
# above without re-resolving. Token/password auth is disabled for convenience on
# a local machine — do not expose this port to an untrusted network.
CMD ["uv", "run", "--no-sync", "jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--IdentityProvider.token=", \
     "--notebook-dir=/app/tutorials"]
