# ALF (Active Learning Framework)

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-check%20CI-orange)](https://github.com/instadeepai/alf/actions)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Tests and Linters 🧪](https://github.com/instadeepai/alf/actions/workflows/tests_and_linters.yaml/badge.svg?branch=main)](https://github.com/instadeepai/alf/actions/workflows/tests_and_linters.yaml)


**ALF** is a Python package for performing active learning experiments to facilitate
iterative optimization of design targets (e.g., proteins, molecules, materials) through
intelligent candidate selection (data acquisition), model adaptation, and evaluation.

## ✨ Features

- **Modular Architecture**: Flexible, extensible components that can be easily swapped
  and customized
- **Multiple experiment setups**: Support for multi-round optimisation, supervised
  learning, and zero-shot evaluation
- **Diverse Acquisition Strategies**: Built-in acquisition functions (Greedy, UCB,
  Expected Improvement, Thompson Sampling)
- **Flexible Search Methods**: Support for local (dataset-based and protocol-based
  e.g. mutagenesis) and global (generative-based) search strategies
- **Offline and Online Evaluation**: Support for both offline (dataset-based) and
  online (model-based or an external objective function) optimization scenarios
- **Evaluation Metrics**: Set of metrics for
  assessing prediction accuracy and uncertainty calibration of surrogate models
- **Comprehensive Testing**: Full test coverage with end-to-end experiments

## 📦 Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Install from source

```bash
# Clone the repository
git clone https://github.com/instadeepai/alf.git
cd alf

# Install with uv (recommended)
uv sync --extra cpu  # or --extra gpu for GPU support
```

Alternatively, this library can be installed with pip from this private GitHub repository, like this:

```
pip install git+https://github.com/instadeepai/alf
```

To authenticate, we recommend to set up a `.netrc` file in your home directory with a GitHub personal access token:

```
machine github.com login <USERNAME> password <TOKEN>
```

The tool `pip` will automatically make use of these credentials for authentication. For more information on creating personal access tokens, see [this](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).



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
oracle = Oracle(module=dataset)

# Run design task
task = DesignTask(num_acq_rounds=5, acq_batch_size=100)
state = task.setup(dataset=dataset, surrogate=surrogate)
task.run(state=state, task_state_loggers=[TerminalTaskStateLogger()], optimizer=optimizer, oracle=oracle)
```

### Supervised Task

```python
from alf_core import Surrogate, SupervisedTask, TerminalTaskStateLogger
from alf.tools.datasets.gfp import GFP
from alf.tools.models.cnn import CNNModel

# Initialize components
dataset = GFP(name="gfp", modality="sequence", seed=42, split_config=split_config)
surrogate = Surrogate(model=CNNModel())

# Run supervised task
task = SupervisedTask()
state = task.setup(dataset=dataset, surrogate=surrogate)
task.run(state=state, task_state_loggers=[TerminalTaskStateLogger()])
```

### Zero-Shot Task

```python
from alf_core import Surrogate, ZeroShotTask, TerminalTaskStateLogger
from alf.tools.datasets.gfp import GFP
from alf.tools.models.random import RandomModel

# Initialize components
dataset = GFP(name="gfp", modality="sequence", seed=42, split_config=split_config)
surrogate = Surrogate(model=RandomModel())  # Pre-trained model

# Run zero-shot task
task = ZeroShotTask()
state = task.setup(dataset=dataset, surrogate=surrogate)
task.run(state=state, task_state_loggers=[TerminalTaskStateLogger()])
```

## 📁 Project Structure

```
alf/
├── core/                  # Core framework
│   ├── alf_core/          # Core package
│   │   ├── dataclasses/   # Data structures (Candidate, LabeledCandidates, etc.)
│   │   ├── dataset/       # Dataset base classes and utilities
│   │   ├── model/         # Model base classes
│   │   ├── optimizer/     # Optimizer, acquisition functions, search strategies
│   │   ├── oracle/        # Oracle for candidate evaluation
│   │   ├── surrogate/     # Surrogate model wrapper
│   │   ├── tasks/         # Task implementations (Design, Supervised, ZeroShot)
│   │   └── utils/         # Utilities (metrics, logging)
│   └── tests/             # Core framework tests
├── tools/                 # Example implementations and tools
│   └── alf_tools/         # Tools package
│       ├── datasets/      # Example datasets (e.g., GFP)
│       ├── models/        # Example models (CNN, Random)
│       └── optimizer/     # Example acquisition functions (UCB, Thompson Sampling, etc.) and search strategies
├── tutorials/             # Tutorials and example scripts
└── docs/                  # Documentation
```

## 📖 Documentation

For detailed documentation on core components, see:
- **[Core Components Documentation](alf/core/README.md)** - Comprehensive guide to all
  core components and their interactions

## 🎓 Tutorials

Explore the tutorials to learn how to use ALF:

- **[Offline Design Tutorial](tutorials/offline_design_tutorial.ipynb)** - Complete guide
  to running offline design experiments
- **[Example Scripts](tutorials/experiments/)** - Ready-to-run examples for all task types

## 🧩 Core Components

ALF is built around a modular architecture with the following key components:

1. **Dataset** (`BaseDataset`) - Data loading, splitting, and querying
2. **Model** (`BaseModel`) - Abstract base class for all models
3. **Surrogate** (`Surrogate`) - Approximates expensive experimental evaluation
4. **Oracle** (`Oracle`) - Provides ground-truth labels for candidates
5. **Optimizer** (`Optimizer`) - Orchestrates the active learning loop (ask-tell interface)
6. **Acquisition Function** (`AcquisitionFunction`) - Scores candidates for acquisition
7. **Search Strategy** (`BaseSearch`) - Defines the candidate pool
8. **Task State** (`TaskState`) - Tracks the state of active learning tasks

See the [Core Components Documentation](alf/core/README.md) for detailed information.

## 🎯 Task Types

ALF supports three main task types:

- **Design Task** - Multi-round active learning loop for iterative optimization
- **Supervised Task** - Train and evaluate models on fixed dataset splits
- **Zero-Shot Task** - Evaluate pre-trained models without training

## 🛠️ Development

### Setup Development Environment

```bash
# Install with development dependencies
uv sync --extra cpu --group dev
```

### Run Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=alf --cov-report term-missing

# Run specific test file
uv run pytest tests/alf/core/tasks/test_design_task.py
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
