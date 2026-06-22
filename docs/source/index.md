# ALF

**ALF** is an active learning framework for computational science, built to optimise
high-dimensional search spaces where data acquisition is expensive—wet-lab experiments, physical
measurements, or costly simulations. It accelerates the discovery of optimal designs (proteins,
molecules, materials) through intelligent candidate selection, adaptive modelling, and efficient
evaluation.

```{image} _static/alf_loop.svg
:alt: The ALF active learning loop
:align: center
:width: 80%
```

```{note}
This project is under active development.
```

## 🔬 Why ALF?

In scientific discovery, the bottleneck is rarely compute—it's the **experiment**. Each label
costs a wet-lab assay, a simulation, or a measurement, and you can only afford a handful of
rounds. ALF runs the full active-learning loop: train a surrogate on what you've measured, use an
acquisition function to select the most informative next batch, score it, and repeat—all with
modular, swappable components so you can change any one part without rewriting the rest.

→ See [Why ALF?](explanation/why-alf.md) for the full motivation and design rationale.

## 🗺️ Start here

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

## 📦 Installation

```bash
# Core package (minimal dependencies, no PyTorch required)
pip install git+https://github.com/instadeepai/alf.git#subdirectory=core

# Tools package (includes PyTorch, models, and datasets)
pip install git+https://github.com/instadeepai/alf.git#subdirectory=tools
```

Add your GitHub credentials to `~/.netrc` for authentication:

```text
machine github.com login <USERNAME> password <TOKEN>
```

For GPU support, optional extras, and development setup, see the
[Installation Guide](installation.md).

## 🚀 Quick start

The snippet below runs a full active learning design experiment. Each component is swappable:
bring your own dataset, model, acquisition function, or oracle.

What each component does:

- **`Dataset`** — holds your candidates and their labels; handles train/candidate-pool splits
- **`Surrogate`** — wraps a model that is cheaply re-trained each round to predict labels and uncertainty
- **`Optimizer`** — scores the candidate pool with an acquisition function and selects the next batch
- **`Oracle`** — evaluates selected candidates (wet-lab assay, simulation, or held-out dataset)

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
installation
```

```{toctree}
:maxdepth: 2
:caption: API Reference
:hidden:

Core <api/alf_core/index>
Tools <api/alf_tools/index>
```
