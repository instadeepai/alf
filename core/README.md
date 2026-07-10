# <img src="https://raw.githubusercontent.com/instadeepai/alf/main/docs/imgs/alf_cover.svg" alt="ALF" height="40" align="top"> alf-core

[![PyPI](https://img.shields.io/pypi/v/alf-core.svg)](https://pypi.org/project/alf-core/)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/instadeepai/alf/blob/main/LICENSE)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/surana01/114eb5680493468e40f5a528c08f1888/raw/alf-core-coverage.json)](https://github.com/instadeepai/alf/tree/main/core)
[![Docs](https://img.shields.io/badge/docs-instadeepai.github.io%2Falf-blue)](https://instadeepai.github.io/alf/)

**The lightweight, dependency-minimal foundation of ALF (Active Learning Framework).**

`alf-core` provides the base classes, core data structures, and the active-learning loop
for iterative optimisation in computational science — optimising high-dimensional,
combinatorially vast search spaces where each label is expensive (wet-lab assays,
simulations, measurements). It ships with no ML-framework dependencies (only numpy,
pandas, scipy), so it is standalone and domain-agnostic. For ready-to-use models,
datasets, and acquisition functions, install
[alf-tools](https://github.com/instadeepai/alf/blob/main/tools/README.md).

<div align="center">
  <img src="https://raw.githubusercontent.com/instadeepai/alf/main/docs/imgs/alf_components.svg" alt="ALF Components" width="70%">
</div>

## Installation

```bash
pip install alf-core
```

## Quick start

`alf-core` is the framework layer: you supply your own `BaseDataset` and `BaseModel`
subclasses (or install [alf-tools](https://github.com/instadeepai/alf/blob/main/tools/README.md)
for ready-made ones), then wire them into the active-learning loop.

```python
from alf_core import (
    DatasetSearch,
    DesignTask,
    Optimizer,
    Oracle,
    Surrogate,
    TerminalStateLogger,
)

# Bring your own BaseDataset, BaseModel, and AcquisitionFunction subclasses
dataset = MyDataset(...)
surrogate = Surrogate(model=MyModel())
optimizer = Optimizer(acquisition_fn=MyAcquisition(), search_fn=DatasetSearch())
oracle = Oracle(scorer=dataset)

# Run the active-learning loop for 5 rounds, acquiring 100 candidates per round
task = DesignTask(num_acq_rounds=5, acq_batch_size=100)
state = task.setup(dataset=dataset, surrogate=surrogate)
task.run(
    state=state,
    state_loggers=[TerminalStateLogger()],
    optimizer=optimizer,
    oracle=oracle,
)
```

For a complete, runnable `alf-core`-only example (a bootstrap-ensemble surrogate and a
Probability of Improvement acquisition function built from scratch with numpy/scipy), see the
[ALF Core Quickstart notebook](https://github.com/instadeepai/alf/blob/main/tutorials/alf_core_quickstart.ipynb).

## Key concepts

ALF runs the active-learning loop over a small set of swappable components:

- **Dataset** (`BaseDataset`) — loads, splits, and queries candidate data
- **Model** (`BaseModel`) — the surrogate/oracle/generator backbone you implement
- **Surrogate** (`Surrogate`) — wraps a model to predict fitness and uncertainty
- **Oracle** (`Oracle`) — returns ground-truth labels (offline pool or live scorer)
- **Optimizer** (`Optimizer`) — proposes the next batch via acquisition + search
- **Acquisition function** (`AcquisitionFunction`) — scores candidates to acquire
- **Search strategy** (`BaseSearch`) — defines the candidate pool to score
- **State** (`State`) — tracks rounds, history, and metrics across the loop
- **Tasks** (`DesignTask`, `SupervisedTask`, `ZeroShotTask`) — drive the multi-round
  loop, fixed-data training, or no-train evaluation

## Documentation

- **Core concepts:** [how the components fit together](https://instadeepai.github.io/alf/explanation/core-concepts.html)
- **API reference:** [every class and method](https://instadeepai.github.io/alf/api/alf_core/index.html)
- **Glossary:** [terms and benchmark metrics](https://instadeepai.github.io/alf/reference/glossary.html)
- **Tutorials:** [tutorials/](https://github.com/instadeepai/alf/tree/main/tutorials)
- **Full documentation:** [instadeepai.github.io/alf](https://instadeepai.github.io/alf/)
- **Ready-to-use tools:** [alf-tools](https://github.com/instadeepai/alf/blob/main/tools/README.md)

## License

Apache License 2.0 — see [LICENSE](https://github.com/instadeepai/alf/blob/main/LICENSE).
