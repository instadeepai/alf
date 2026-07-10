# <img src="../docs/imgs/alf_cover.svg" alt="ALF" height="40" align="top"> alf-tools

[![PyPI](https://img.shields.io/pypi/v/alf-tools.svg)](https://pypi.org/project/alf-tools/)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/instadeepai/alf/blob/main/LICENSE)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/surana01/25ce4b64d5b9cda177203366146f5bf0/raw/alf-tools-coverage.json)](https://github.com/instadeepai/alf/tree/main/tools)
[![Docs](https://img.shields.io/badge/docs-instadeepai.github.io%2Falf-blue)](https://instadeepai.github.io/alf/)

**Ready-to-use models, datasets, and acquisition functions for ALF (Active Learning Framework).**

`alf-tools` builds on [alf-core](https://github.com/instadeepai/alf/blob/main/core/README.md)
to give you everything needed to run active-learning experiments out of the box — example
datasets (GFP, ProteinGym, FLIP, GuacaMol), models (CNN, Gaussian Process, ESM-2, Chemprop),
acquisition functions, and search strategies. Use it for quick-start and prototyping; reach for
`alf-core` alone when you want the lightweight framework with no ML-framework dependencies.

<div align="center">
  <img src="https://raw.githubusercontent.com/instadeepai/alf/main/docs/imgs/alf_main_figure.png" alt="ALF active-learning loop overview" width="70%">
</div>

## Installation

```bash
pip install alf-tools
```

This installs PyTorch and the core dependencies. Some models need additional dependencies —
see [Optional Extras](#optional-extras) below.

### Optional Extras

Some models require additional dependencies. Append one or more extras to the package name.
Per-model extras (`esm2`, `esmfold`, `chemprop`, `guacamol`) install exactly one model's
dependencies; workflow umbrellas (`protein`, `molecule`) group the extras you are likely to use
together.

```bash
pip install "alf-tools[esm2]"      # ESM-2 protein language model (ESM2Model)
pip install "alf-tools[esmfold]"   # ESMFold structure-prediction oracle (ESMFoldModel)
pip install "alf-tools[chemprop]"  # Chemprop small-molecule MPNN (ChempropModel)
pip install "alf-tools[guacamol]"  # GuacaMol RDKit-based dataset/scoring
pip install "alf-tools[protein]"   # umbrella: esm2 + esmfold
pip install "alf-tools[molecule]"  # umbrella: chemprop + guacamol
```

## Documentation

- **Full documentation:** [instadeepai.github.io/alf](https://instadeepai.github.io/alf/)
- **API reference:** [alf-tools API](https://instadeepai.github.io/alf/api/alf_tools/index.html)
- **Installation guide:** [instadeepai.github.io/alf/installation.html](https://instadeepai.github.io/alf/installation.html)
- **Core framework:** [alf-core](https://github.com/instadeepai/alf/blob/main/core/README.md)
- **Tutorials:** [tutorials/](https://github.com/instadeepai/alf/tree/main/tutorials)

## What's included

`alf-tools` bundles example datasets (GFP, ProteinGym, FLIP, GuacaMol), surrogate and oracle
models (CNN, Gaussian Process, ESM-2, ESMFold, Chemprop, PyRosetta, plus an ensemble wrapper),
acquisition functions (Greedy, UCB, Expected Improvement, Thompson Sampling, CoreSet, and a
BoTorch wrapper), and search strategies. See the
[API reference](https://instadeepai.github.io/alf/api/alf_tools/index.html) for the full
catalogue and configuration options.

## Quick example

```python
from alf_core import (
    BaseDatasetConfig,
    DatasetSearch,
    DesignTask,
    Optimizer,
    Oracle,
    Surrogate,
    TerminalStateLogger,
)
from alf_tools.datasets import GFP
from alf_tools.models import CNNModel
from alf_tools.optimizer.acquisition_functions import Greedy

# Configure and load the dataset
config = BaseDatasetConfig(
    name="gfp",
    modality="sequence",
    seed=42,
    train_ratio=0.1,
    validation_frac=0.5,
    test_ratio=0.2,
    split_type="random",
    problem_type="regression",
)
dataset = GFP(config)
surrogate = Surrogate(model=CNNModel())
optimizer = Optimizer(acquisition_fn=Greedy(), search_fn=DatasetSearch())
oracle = Oracle(scorer=dataset)

# Run active learning
task = DesignTask(num_acq_rounds=5, acq_batch_size=100)
state = task.setup(dataset=dataset, surrogate=surrogate)
task.run(
    state=state,
    state_loggers=[TerminalStateLogger()],
    optimizer=optimizer,
    oracle=oracle,
)
```

## Normalisation

Models support input normalisation and output standardisation through their train configs
(`BaseTrainConfig`). `GPModel` enables both by default (`minmax` inputs, standardised outputs);
the other models default to neither. See the
[API reference](https://instadeepai.github.io/alf/api/alf_tools/index.html) for per-model
defaults and configuration.

## Creating custom components

All components subclass the `alf_core` base classes (`BaseDataset`, `BaseModel`,
`AcquisitionFunction`, `BaseSearch`). See the
[how-to guides](https://instadeepai.github.io/alf/how-to/index.html) for step-by-step instructions.

## License

Apache License 2.0 — see [LICENSE](https://github.com/instadeepai/alf/blob/main/LICENSE).
