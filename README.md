# ALF (Active Learning Framework)

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Core Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/surana01/114eb5680493468e40f5a528c08f1888/raw/alf-core-coverage.json)](https://github.com/instadeepai/alf/tree/main/core)
[![Tools Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/surana01/25ce4b64d5b9cda177203366146f5bf0/raw/alf-tools-coverage.json)](https://github.com/instadeepai/alf/tree/main/tools)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Tests and Linters 🧪](https://github.com/instadeepai/alf/actions/workflows/tests_and_linters.yaml/badge.svg?branch=main)](https://github.com/instadeepai/alf/actions/workflows/tests_and_linters.yaml)


**ALF** is a Python package for performing active learning experiments to facilitate
iterative optimization of design targets (e.g., proteins, molecules, materials) through
intelligent candidate selection (data acquisition), model adaptation, and evaluation.

## ✨ Features

- **Modular Architecture**: Flexible, extensible components that can be easily swapped and customized
- **Multiple Experiment Setups**: Multi-round optimization, supervised learning, and zero-shot evaluation
- **Offline and Online Evaluation**: Dataset-based and model-based optimization scenarios
- **Evaluation Metrics**: Metrics for prediction accuracy and uncertainty calibration

## 📦 Installation

### Quick Install

```bash
# Install the core package
pip install git+https://github.com/instadeepai/alf.git#subdirectory=core

# Install the tools package (includes PyTorch)
pip install git+https://github.com/instadeepai/alf.git#subdirectory=tools
```

**Authentication:** Set up a `.netrc` file in your home directory with your GitHub personal access token:

```
machine github.com login <USERNAME> password <TOKEN>
```

For more information on creating personal access tokens, see [GitHub's documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

📖 **For detailed installation instructions, including development setup and GPU configuration, see [INSTALLATION.md](docs/INSTALLATION.md)**.

## 📖 Documentation

Full API documentation is available at **[instadeepai.github.io/alf](https://instadeepai.github.io/alf/)**

### Building Documentation Locally

To build and view the documentation on your local machine:

1. Install documentation dependencies:
   ```bash
   uv sync --group docs
   ```

2. Build the HTML documentation:
   ```bash
   cd docs
   make html
   ```

3. Open the documentation in your browser:
   ```bash
   # macOS
   open build/html/index.html

   # Linux
   xdg-open build/html/index.html

   # Windows
   start build/html/index.html
   ```

## 🚀 Quick Start

### Design Task

```python
from alf_core import Optimizer, DatasetSearch, Oracle, Surrogate, DesignTask, TerminalTaskStateLogger
from alf.tools.datasets.gfp import GFP
from alf.tools.models.cnn import CNNModel
from alf.tools.optimizer.acquisition_functions.greedy import Greedy

# Initialize components
dataset = GFP(name="gfp", modality="sequence", seed=42, split_config=split_config)
surrogate = Surrogate(model=CNNModel())
acquisition_fn = Greedy()
search_fn = DatasetSearch()
optimizer = Optimizer(acquisition_fn=acquisition_fn, search_fn=search_fn)
oracle = Oracle(scorer=dataset)

# Run design task
task = DesignTask(num_acq_rounds=5, acq_batch_size=100)
state = task.setup(dataset=dataset, surrogate=surrogate)
task.run(state=state, task_state_loggers=[TerminalTaskStateLogger()], optimizer=optimizer, oracle=oracle)
```

## 📁 Project Structure

```
alf/
├── core/                  # Core framework (see core/README.md)
│   ├── alf_core/          # Core package
│   │   ├── dataclasses/   # Data structures (Candidate, LabelledCandidates, etc.)
│   │   ├── dataset/       # Dataset base classes and utilities
│   │   ├── model/         # Model base classes
│   │   ├── optimizer/     # Optimizer, acquisition functions, search strategies
│   │   ├── oracle/        # Oracle for candidate evaluation
│   │   ├── surrogate/     # Surrogate model wrapper
│   │   ├── tasks/         # Task implementations (Design, Supervised, ZeroShot)
│   │   └── utils/         # Utilities (metrics, logging)
│   └── tests/             # Core framework tests
├── tools/                 # Example implementations and tools (see tools/README.md)
│   └── alf_tools/         # Tools package
│       ├── datasets/      # Example datasets (e.g., GFP)
│       ├── models/        # Example models (CNN, Random)
│       └── optimizer/     # Example acquisition functions (UCB, Thompson Sampling, etc.) and search strategies
├── tutorials/             # Tutorials and example scripts
└── docs/                  # Documentation
```

**Package Documentation:**
- [Core Framework](core/README.md) - Base classes and task implementations
- [Tools Package](tools/README.md) - Ready-to-use datasets, models, and acquisition functions

## 🎓 Tutorials

Explore the tutorials to learn how to use ALF:

- **[Offline Design Tutorial](tutorials/offline_design_tutorial.ipynb)** - Complete guide
  to running offline design experiments
- **[Example Scripts](tutorials/experiments/)** - Ready-to-run examples for all task types


## 🛠️ Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/instadeepai/alf.git
cd alf

# Install all packages with development dependencies
uv sync
```

For GPU support, see the [GPU Configuration](docs/INSTALLATION.md#gpu-support-optional) section in the installation guide.

### Run Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=alf_core --cov-report term-missing
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Commit your changes (`git commit -m 'Add some amazing feature'`)
3. Push to the branch (`git push origin feature/amazing-feature`)
4. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
