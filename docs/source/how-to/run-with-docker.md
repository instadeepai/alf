# Run with Docker

The repository ships a `Dockerfile` that builds a ready-to-run environment for the tutorial
notebooks and the `benchmark_examples/` scripts — no local Python, `uv`, or dependency setup
required. It uses CPU-only PyTorch by default.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running.
- A checkout of the repository (the image builds from local source, so no GitHub token is
  needed):

  ```bash
  git clone https://github.com/instadeepai/alf.git
  cd alf
  ```

## Build the image

```bash
docker build -t alf .
```

This installs the `alf_core`, `alf_tools`, and `alf_tutorials` packages along with the
`tutorials` and `benchmark` dependency groups (JupyterLab, matplotlib, umap, ESM-2). The first
build downloads PyTorch and other large wheels, so expect it to take a few minutes.

## Run the tutorials

Launch JupyterLab and open the notebooks in your browser:

```bash
docker run --rm -p 8888:8888 alf
```

Then visit [http://localhost:8888](http://localhost:8888). The server starts in `/app/tutorials`,
so the notebooks listed in the [Tutorials](../tutorials/index.md) are ready to open.

```{note}
Token authentication is disabled for convenience, so anyone who can reach port 8888 can run code
in the container. Only use this on a trusted, local machine — do not publish the port to an
untrusted network.
```

To keep edited notebooks and outputs on your host, mount the tutorials directory:

```bash
docker run --rm -p 8888:8888 -v "$(pwd)/tutorials:/app/tutorials" alf
```

## Run the benchmark examples

Override the default command to run a benchmark script instead. `--no-sync` reuses the
environment baked into the image:

```bash
docker run --rm alf uv run --no-sync \
    python benchmark_examples/benchmarking_surrogates.py \
    --dataset gfp --num-rounds 5 --batch-size 50 --num-seeds 3
```

The scripts write a comparison PNG and `summary.csv` under
`benchmark_examples/outputs/<script>/` inside the container. Mount that directory to keep the
results on your host:

```bash
docker run --rm -v "$(pwd)/benchmark_examples/outputs:/app/benchmark_examples/outputs" \
    alf uv run --no-sync \
    python benchmark_examples/benchmarking_acquisition_functions.py --dataset gfp
```

See the [benchmark examples README](https://github.com/instadeepai/alf/blob/main/benchmark_examples/README.md)
for the full list of flags.

## GPU support

The image installs the CPU build of PyTorch. To use a GPU, run the container with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
and install the CUDA build of torch on top:

```bash
docker run --rm --gpus all -p 8888:8888 alf bash -c \
    "uv pip install torch --index-url https://download.pytorch.org/whl/cu128 && \
     uv run --no-sync jupyter lab --ip=0.0.0.0 --allow-root --IdentityProvider.token="
```

For a persistent GPU image, edit the torch source index in `tools/pyproject.toml` to
`pytorch-gpu` before building (see the GPU Support section of the
[Installation Guide](../installation.md)).
