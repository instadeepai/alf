# ALF (Active Learning Framework)

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Core Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/surana01/114eb5680493468e40f5a528c08f1888/raw/alf-core-coverage.json)](https://github.com/instadeepai/alf/tree/main/core)
[![Tools Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/surana01/25ce4b64d5b9cda177203366146f5bf0/raw/alf-tools-coverage.json)](https://github.com/instadeepai/alf/tree/main/tools)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Tests and Linters 🧪](https://github.com/instadeepai/alf/actions/workflows/tests_and_linters.yaml/badge.svg?branch=main)](https://github.com/instadeepai/alf/actions/workflows/tests_and_linters.yaml)

**ALF** is an active learning framework for computational science, designed to optimize high-dimensional and combinatorially vast search spaces where data acquisition is expensive—from wet-lab experiments and physical measurements to costly simulations. ALF accelerates discovery of optimal designs (proteins, molecules, materials) through intelligent candidate selection, adaptive modelling, and efficient evaluation strategies.

## Why ALF?

In scientific discovery, the bottleneck is rarely compute—it's the **experiment**. Each label costs a wet-lab assay, a simulation, or a measurement, and you can only afford a handful of rounds. ALF runs the full active-learning loop for you: train a surrogate on what you've measured, use an acquisition function to select the most informative next batch, score it, and repeat—all with modular, swappable components so you can change any one part without rewriting the rest.

→ [Full motivation and design rationale](https://instadeepai.github.io/alf/explanation/why-alf.html)

## 📖 Documentation

**[https://instadeepai.github.io/alf/](https://instadeepai.github.io/alf/)** — full docs including API reference, tutorials, and how-to guides.

- 📥 [Installation Guide](docs/INSTALLATION.md)
- 🛠️ [Contributing Guide](docs/CONTRIBUTING.md)

## 📦 Package Architecture

ALF is split into two packages:

**alf-core** — Lightweight framework with base classes, core data structures, and minimal dependencies (numpy, pandas, scipy). No ML framework dependencies. Use this for custom implementations or when integrating into existing systems.

**alf-tools** — Ready-to-use datasets, models, and acquisition functions with heavier dependencies (PyTorch). Depends on alf-core. Use this for quick-start and prototyping.

## 📦 Installation

```bash
# Core package only (no PyTorch required)
pip install git+https://github.com/instadeepai/alf.git#subdirectory=core

# Tools package (includes PyTorch, models, and datasets)
pip install git+https://github.com/instadeepai/alf.git#subdirectory=tools
```

**Authentication:** add your GitHub credentials to `~/.netrc`:

```
machine github.com login <USERNAME> password <TOKEN>
```

For GPU support, optional extras (ESM2, Chemprop), and development setup, see [docs/INSTALLATION.md](docs/INSTALLATION.md).

## 🚀 Quick Start

```python
from alf_core import (
    DatasetSearch,
    DesignTask,
    Optimizer,
    Oracle,
    Surrogate,
    TerminalStateLogger,
)
from alf_tools.datasets.gfp import GFP
from alf_tools.models.cnn import CNNModel
from alf_tools.optimizer.acquisition_functions.greedy import Greedy

# A dataset wraps your candidates and their known labels
dataset = GFP(name="gfp", modality="sequence", seed=42)

# The surrogate is a cheap probabilistic model trained on observed labels
surrogate = Surrogate(model=CNNModel())

# The optimizer picks the next batch: acquisition function scores candidates,
# search function selects from them
optimizer = Optimizer(acquisition_fn=Greedy(), search_fn=DatasetSearch())

# The oracle scores selected candidates (here, it queries the held-out dataset)
oracle = Oracle(scorer=dataset)

# Run the active learning loop for 5 rounds, acquiring 100 candidates per round
task = DesignTask(num_acq_rounds=5, acq_batch_size=100)
state = task.setup(dataset=dataset, surrogate=surrogate)
task.run(
    state=state,
    state_loggers=[TerminalStateLogger()],
    optimizer=optimizer,
    oracle=oracle,
)
```

New to active learning? Start with [Why ALF?](https://instadeepai.github.io/alf/explanation/why-alf.html), then work through the [Tutorials](https://instadeepai.github.io/alf/tutorials/index.html).

## 🎓 Tutorials

### Start with an experiment

- **[Offline Design Tutorial](tutorials/experiments/offline_design_tutorial.ipynb)** — end-to-end active learning loop with labels from a held-out pool. The best entry point.
- **[Online Design Tutorial](tutorials/experiments/online_design_tutorial.ipynb)** — the same loop, with labels from a live scorer.

### Go deeper on models

- **[GP Tutorial](tutorials/models/gp_tutorial.ipynb)** — Gaussian Process surrogate and kernel cheat-sheet
- **[CNN Tutorial](tutorials/models/cnn_tutorial.ipynb)** — convolutional sequence surrogate
- **[Ensemble Tutorial](tutorials/models/ensemble_tutorial.ipynb)** — seed ensembles, MC dropout, and combined ensembles for uncertainty-aware prediction
- **[ESM-2 Tutorial](tutorials/models/esm2_tutorial.ipynb)** — protein language model as surrogate or zero-shot scorer
- **[Chemprop MPNN Tutorial](tutorials/models/chemprop_tutorial.ipynb)** — active learning for small molecules with SMILES inputs

### Lightweight (alf-core only, no PyTorch)

- **[Core-only Tutorial](tutorials/alf_core_quickstart.ipynb)** — synthetic optimisation loop using only `alf-core` and numpy/scipy

### Extend ALF

- **[Models](tutorials/extending_base_classes/models.ipynb)** — create custom models for oracle/surrogate/generator roles
- **[Datasets](tutorials/extending_base_classes/datasets.ipynb)** — add custom data sources
- **[Search Functions](tutorials/extending_base_classes/search_functions.ipynb)** — implement custom search strategies
- **[Acquisition Functions](tutorials/extending_base_classes/acquisition_functions.ipynb)** — create custom acquisition strategies

## 🛠️ Development

```bash
git clone git@github.com:instadeepai/alf.git
cd alf
uv sync
```

```bash
uv run pytest                                         # run all tests
uv run pytest --cov=alf_core --cov-report term-missing  # with coverage
uv run pre-commit install                             # install pre-commit hooks
```

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for GPU configuration and optional extras.

## 🤝 Contributing

Read our [Contributing Guide](docs/CONTRIBUTING.md) for development guidelines and the PR process. For questions or discussions, open an issue.

## 📁 Project Structure

```
alf/
├── core/                  # alf-core package (base classes, tasks, utilities)
├── tools/                 # alf-tools package (models, datasets, acquisition functions)
├── tutorials/             # Tutorial notebooks
└── docs/                  # Documentation source
```

See [core/README.md](core/README.md) and [tools/README.md](tools/README.md) for package-level details.

## 📄 License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
