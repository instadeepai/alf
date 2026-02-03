# BoTorch Integration - Quick Start

## TL;DR - Three Ways to Use BoTorch

### 1. Generic Wrapper (Recommended for Most Users)

**Best for**: Experimentation, easy switching between acquisition functions

```python
from alf_tools.optimizer.acquisition_functions import BoTorchAcquisition
from alf_tools.optimizer.search import BoTorchMCSampler, ContinuousSearch

sampler = BoTorchMCSampler(sampler_type="sobol", num_samples=512)

# Switch acquisition functions by changing one parameter
acq_fn = BoTorchAcquisition(
    acquisition_type="qEI",  # or "qUCB", "qNEI"
    sampler=sampler,
    bounds=[[0, 1], [0, 1]],
    batch_size=5,
)

optimizer = Optimizer(acquisition_fn=acq_fn, search_fn=ContinuousSearch())
```

### 2. Specific Wrapper (For Advanced Users)

**Best for**: Fine-grained control over a specific acquisition function

```python
from alf_tools.optimizer.acquisition_functions import BoTorchQEI

acq_fn = BoTorchQEI(
    bounds=[[0, 1], [0, 1]],
    num_samples=512,
    num_restarts=10,
    batch_size=5,
)
```

### 3. BoTorch GP Model

**Better defaults, modern priors, seamless BoTorch integration**

```python
from alf_tools.models import BoTorchGPModel

model = BoTorchGPModel(
    normalize_inputs=True,
    standardize_outputs=True,
)
```

## Component Comparison

| Component | Purpose | When to Use |
|-----------|---------|-------------|
| **BoTorchGPModel** | Modern GP with better defaults | Continuous spaces, BoTorch acquisition functions |
| **GPModel** | Flexible GP with custom kernels | Sequence data, custom configurations |
| **BoTorchAcquisition** | Generic acquisition wrapper | Experimenting with different functions |
| **BoTorchQEI** | Specific qEI wrapper | Production use of qEI with fine control |
| **BoTorchMCSampler** | MC sampler configuration | Control sampling strategy |
| **ContinuousSearch** | Continuous optimization | Unbounded continuous search spaces |

## Switching Between Acquisition Functions

```python
# Setup once
sampler = BoTorchMCSampler("sobol", 512)
bounds = [[0, 1], [0, 1]]

# Try different functions
configs = {"sampler": sampler, "bounds": bounds, "batch_size": 5}

acq_ei = BoTorchAcquisition(acquisition_type="qEI", **configs)
acq_ucb = BoTorchAcquisition(acquisition_type="qUCB", beta=0.2, **configs)
acq_nei = BoTorchAcquisition(acquisition_type="qNEI", **configs)

# Use any in your optimizer
optimizer = Optimizer(acquisition_fn=acq_ei, search_fn=ContinuousSearch())
```

## Full Example

```python
from alf_core import DesignTask, Optimizer, Surrogate
from alf_tools.datasets import BoTorchSyntheticDataset
from alf_tools.models import BoTorchGPModel
from alf_tools.optimizer.acquisition_functions import BoTorchAcquisition
from alf_tools.optimizer.search import BoTorchMCSampler, ContinuousSearch

# Dataset
dataset = BoTorchSyntheticDataset("Branin", n_initial=50)

# Model
surrogate = Surrogate(BoTorchGPModel())

# Acquisition
sampler = BoTorchMCSampler("sobol", 512)
acq_fn = BoTorchAcquisition(
    acquisition_type="qEI",
    sampler=sampler,
    bounds=dataset.bounds,
    batch_size=5,
)

# Optimize
optimizer = Optimizer(acq_fn, ContinuousSearch())
task = DesignTask(num_acq_rounds=20, acq_batch_size=5)
state = task.setup(dataset=dataset, surrogate=surrogate)
results = task.run(state, optimizer=optimizer)
```

## What's New?

✅ **Generic Acquisition Wrapper** - Switch functions with one parameter
✅ **BoTorch GP Model** - Modern priors, better defaults
✅ **MC Sampler Configuration** - Control sampling strategy
✅ **Continuous Search** - For unbounded optimization
✅ **Comprehensive Documentation** - Guide + examples
✅ **Backward Compatible** - Existing code still works

## Next Steps

1. **Read the full guide**: [BoTorch Integration Guide](botorch_integration_guide.md)
2. **Try the demo**: `python examples/botorch_integration_demo.py`
3. **Follow the tutorial**: `tutorials/botorch_synthetic_tutorial.ipynb`

## Questions?

- **Which GP model to use?** Start with `BoTorchGPModel` for continuous spaces
- **Which acquisition function?** Start with `qEI` (via `BoTorchAcquisition`)
- **Discrete or continuous search?** Use `ContinuousSearch` for continuous, `DatasetSearch` for discrete
- **How many MC samples?** 512 for Sobol QMC is a good default

For more details, see the [full guide](botorch_integration_guide.md).
