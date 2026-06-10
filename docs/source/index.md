# ALF

**ALF** is an active learning framework for iterative optimisation of biological and chemical
design targets. It provides modular components for candidate search, surrogate modelling, and
acquisition, enabling efficient discovery across expensive, high-dimensional search spaces such
as protein sequences and small molecules.

```{note}
This project is under active development.
```

## Start here

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Learning-oriented walkthroughs that take you from zero to a running experiment.
:::

:::{grid-item-card} How-to / Recipes
:link: how-to/index
:link-type: doc

Task-oriented recipes: add your own model, dataset, acquisition or search function.
:::

:::{grid-item-card} Explanation
:link: explanation/index
:link-type: doc

The concepts: why ALF exists, the active-learning loop, and the objects that implement it.
:::

:::{grid-item-card} Reference
:link: reference/index
:link-type: doc

API reference generated from docstrings, plus a glossary of ALF terms.
:::
::::

## Installation

**Prerequisites:** Python 3.12 or higher and [uv](https://docs.astral.sh/uv/) (recommended) or `pip`.

**Quick install** (directly from GitHub):

```bash
# Install the core package (minimal dependencies, no PyTorch required)
pip install git+https://github.com/instadeepai/alf.git#subdirectory=core

# Install the tools package (includes PyTorch, models, and datasets)
pip install git+https://github.com/instadeepai/alf.git#subdirectory=tools
```

Add your GitHub credentials to `~/.netrc` for authentication:

```text
machine github.com login <USERNAME> password <TOKEN>
```

**Development install:**

```bash
git clone git@github.com:instadeepai/alf.git
cd alf
uv sync
```

For GPU support and full installation details, see the
[Installation Guide](https://github.com/instadeepai/alf/blob/main/docs/INSTALLATION.md).

## Quick start

Run an active learning design experiment:

```python
from alf_core import (
    DatasetSearch,
    DesignTask,
    Optimizer,
    Oracle,
    Surrogate,
    TerminalTaskStateLogger,
)
from alf_tools.datasets.gfp import GFP
from alf_tools.models.cnn import CNNModel
from alf_tools.optimizer.acquisition_functions.greedy import Greedy

# Set up dataset and components
dataset = GFP(name="gfp", modality="sequence", seed=42)
surrogate = Surrogate(model=CNNModel())
optimizer = Optimizer(acquisition_fn=Greedy(), search_fn=DatasetSearch())
oracle = Oracle(scorer=dataset)

# Run the design task
task = DesignTask(num_acq_rounds=5, acq_batch_size=100)
state = task.setup(dataset=dataset, surrogate=surrogate)
task.run(
    state=state,
    task_state_loggers=[TerminalTaskStateLogger()],
    optimizer=optimizer,
    oracle=oracle,
)
```

New to active learning? Start with [Why ALF?](explanation/why-alf.md) for the motivation, then
work through the [Tutorials](tutorials/index.md).

```{toctree}
:maxdepth: 2
:caption: Documentation
:hidden:

explanation/index
tutorials/index
how-to/index
reference/index
```

```{toctree}
:maxdepth: 2
:caption: API Reference
:hidden:

Core <api/alf_core/index>
Tools <api/alf_tools/index>
```
