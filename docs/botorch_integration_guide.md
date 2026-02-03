# BoTorch Integration Guide

This guide explains how to use the BoTorch integration components in ALF for advanced Bayesian optimization.

## Overview

The BoTorch integration provides three main components:

1. **`BoTorchGPModel`** - Modern Gaussian Process model with better defaults
2. **`BoTorchMCSampler`** - Configurable Monte Carlo samplers for acquisition functions
3. **`BoTorchAcquisition`** - Generic wrapper for easy switching between acquisition functions
4. **`ContinuousSearch`** - Search function for continuous optimization

## Quick Start

```python
from alf_core import Optimizer, Surrogate
from alf_tools.models import BoTorchGPModel
from alf_tools.optimizer.acquisition_functions import BoTorchAcquisition
from alf_tools.optimizer.search import BoTorchMCSampler, ContinuousSearch

# 1. Create GP model
model = BoTorchGPModel()
surrogate = Surrogate(model)

# 2. Configure sampler
sampler = BoTorchMCSampler(sampler_type="sobol", num_samples=512)

# 3. Create acquisition function
acq_fn = BoTorchAcquisition(
    acquisition_type="qEI",  # or "qUCB", "qNEI"
    sampler=sampler,
    bounds=[[0, 1], [0, 1]],  # Search space bounds
    batch_size=5,
)

# 4. Create optimizer
optimizer = Optimizer(
    acquisition_fn=acq_fn,
    search_fn=ContinuousSearch(),
)
```

## Component Details

### 1. BoTorchGPModel

A Gaussian Process model built on BoTorch's `SingleTaskGP`, offering:
- Modern hyperparameter priors from [Hvarfner et al. 2024](https://github.com/meta-pytorch/botorch/discussions/2451)
- Automatic output standardization
- Efficient model fitting with L-BFGS-B
- Seamless integration with BoTorch acquisition functions

#### Usage

```python
from alf_tools.models import BoTorchGPModel

model = BoTorchGPModel(
    normalize_inputs=True,       # Normalize to [0,1]
    standardize_outputs=True,    # Zero mean, unit variance
    num_iterations=100,          # MLL optimization iterations
    learning_rate=0.1,           # Learning rate (not used with L-BFGS-B)
    device=None,                 # Auto-detect GPU/CPU
    dtype=torch.float32,         # Tensor dtype
)

# Train on data
model.train(train_data, val_data)

# Make predictions
predictions = model.predict(candidates)
```

#### When to Use

- **Use BoTorchGPModel when:**
  - You want modern GP priors and better defaults
  - You're using BoTorch acquisition functions
  - You have continuous input spaces
  - You need seamless BoTorch integration

- **Use standard GPModel when:**
  - You need custom kernel configurations
  - You're working with sequence data (proteins, DNA)
  - You want more control over hyperparameter priors
  - You're not using BoTorch acquisition functions

### 2. BoTorchMCSampler

Configures Monte Carlo samplers for approximating acquisition function expectations.

#### Sampler Types

| Type | Description | Recommended Samples | Use Case |
|------|-------------|---------------------|----------|
| **sobol** | Quasi-Monte Carlo (Sobol sequence) | 256-512 | Better coverage, recommended for most cases |
| **iid** | Independent random sampling | 1024+ | Faster but less efficient |

#### Usage

```python
from alf_tools.optimizer.search import BoTorchMCSampler

# Create sampler configuration
sampler = BoTorchMCSampler(
    sampler_type="sobol",  # or "iid"
    num_samples=512,       # Number of MC samples
    seed=42,               # For reproducibility
)

# Get the BoTorch sampler object
botorch_sampler = sampler.get_sampler()
```

#### Best Practices

- **Sobol QMC** (default): Use 256-512 samples for good accuracy/speed trade-off
- **IID**: Use 1024+ samples if speed is critical and you can tolerate lower accuracy
- Set `seed` for reproducible experiments

### 3. BoTorchAcquisition

Generic wrapper for switching between BoTorch acquisition functions.

#### Supported Acquisition Functions

| Type | Full Name | Description | Best For |
|------|-----------|-------------|----------|
| **qEI** | q-Expected Improvement | Batch expected improvement | General purpose, good default |
| **qUCB** | q-Upper Confidence Bound | Exploration-exploitation with β parameter | When you want to tune exploration |
| **qNEI** | q-Noisy Expected Improvement | Handles noisy observations | Experimental data with noise |
| **qKG** | q-Knowledge Gradient | Most sophisticated (not yet implemented) | Complex optimization problems |

#### Usage

```python
from alf_tools.optimizer.acquisition_functions import BoTorchAcquisition

# Create acquisition function
acq_fn = BoTorchAcquisition(
    acquisition_type="qEI",      # Type of acquisition function
    sampler=sampler,             # BoTorchMCSampler instance
    bounds=[[0, 1], [0, 1]],    # Search space bounds
    num_restarts=10,             # Random restarts for optimization
    raw_samples=512,             # Initial random samples
    batch_size=5,                # Number of candidates to select
    sequential=False,            # Joint vs sequential optimization
    beta=0.2,                    # Exploration parameter (qUCB only)
)
```

#### Switching Acquisition Functions

One of the key benefits is easy switching:

```python
# Try different acquisition functions with same config
configs = {
    "sampler": sampler,
    "bounds": [[0, 1], [0, 1]],
    "batch_size": 5,
}

# Conservative exploration (qEI)
acq_ei = BoTorchAcquisition(acquisition_type="qEI", **configs)

# More exploration (qUCB)
acq_ucb = BoTorchAcquisition(acquisition_type="qUCB", beta=0.3, **configs)

# Handle noisy data (qNEI)
acq_nei = BoTorchAcquisition(acquisition_type="qNEI", **configs)
```

#### Parameters Guide

- **num_restarts**: More restarts = better optimization, slower (10-20 is good)
- **raw_samples**: Initial random samples (512-1024 is typical)
- **batch_size**: How many candidates to select per round
- **sequential**:
  - `False`: Joint optimization (slower, better diversity)
  - `True`: Sequential optimization (faster, less diverse)
- **beta** (qUCB only): Higher = more exploration (0.1-0.5 typical range)

### 4. ContinuousSearch

Search function for continuous optimization with BoTorch.

```python
from alf_tools.optimizer.search import ContinuousSearch

# Returns empty candidate list to signal continuous optimization
search_fn = ContinuousSearch()
```

This tells the acquisition function to use `optimize_acqf` rather than scoring a discrete pool.

## Complete Example

Here's a complete optimization workflow:

```python
from alf_core import DesignTask, Optimizer, Surrogate
from alf_tools.datasets import BoTorchSyntheticDataset
from alf_tools.models import BoTorchGPModel
from alf_tools.optimizer.acquisition_functions import BoTorchAcquisition
from alf_tools.optimizer.search import BoTorchMCSampler, ContinuousSearch

# 1. Create dataset
dataset = BoTorchSyntheticDataset(
    function_name="Branin",
    n_initial=50,
    noise_std=0.1,
)

# 2. Create GP model
model = BoTorchGPModel(normalize_inputs=True)
surrogate = Surrogate(model)

# 3. Configure sampler
sampler = BoTorchMCSampler(sampler_type="sobol", num_samples=512, seed=42)

# 4. Create acquisition function
acq_fn = BoTorchAcquisition(
    acquisition_type="qEI",
    sampler=sampler,
    bounds=dataset.bounds,  # Automatically from dataset
    batch_size=5,
    num_restarts=10,
)

# 5. Create optimizer
optimizer = Optimizer(
    acquisition_fn=acq_fn,
    search_fn=ContinuousSearch(),
)

# 6. Run optimization
task = DesignTask(num_acq_rounds=20, acq_batch_size=5)
state = task.setup(dataset=dataset, surrogate=surrogate)
results = task.run(state, optimizer=optimizer, oracle=None)

# 7. Analyze results
print(f"Final best value: {state.dataset.train_dataset.labels.max():.4f}")
```

## Comparison with Existing Components

### BoTorchGPModel vs GPModel

| Feature | BoTorchGPModel | GPModel |
|---------|---------------|---------|
| **Priors** | Modern (Hvarfner 2024) | Custom configurable |
| **Kernels** | Matern 5/2 (default) | RBF, Matern, Linear, Polynomial |
| **Featurization** | Raw tensors | One-hot encoding for sequences |
| **Training** | L-BFGS-B (BoTorch) | Adam/L-BFGS (custom) |
| **Best for** | Continuous spaces | Protein/sequence spaces |

### BoTorchAcquisition vs Individual Wrappers

| Approach | Pros | Cons |
|----------|------|------|
| **BoTorchAcquisition** | Easy switching, unified interface | Slightly less customization |
| **Individual wrappers** (BoTorchQEI, etc.) | Full control over specific function | Need separate class per function |

**Recommendation**: Use `BoTorchAcquisition` for experimentation and easy switching. Use individual wrappers (like `BoTorchQEI`) when you've settled on a specific acquisition function and need fine-grained control.

## Best Practices

### 1. Data Scaling

BoTorch works best with:
- **Inputs**: Normalized to [0, 1]^d
- **Outputs**: Standardized (zero mean, unit variance) - handled automatically

```python
# Good: Data normalized
bounds = [[0, 1], [0, 1]]  # Unit cube

# Avoid: Wide ranges without normalization
bounds = [[-100, 1000], [0, 500]]  # Will generate warnings
```

### 2. Choosing Acquisition Functions

```python
# Start with qEI (good default)
acq_fn = BoTorchAcquisition(acquisition_type="qEI", ...)

# If underfitting (too exploitative), try qUCB with higher beta
acq_fn = BoTorchAcquisition(acquisition_type="qUCB", beta=0.3, ...)

# If you have noisy data, use qNEI
acq_fn = BoTorchAcquisition(acquisition_type="qNEI", ...)
```

### 3. Batch Size Selection

```python
# Small batches (1-5): Better for sequential experiments
batch_size = 3

# Larger batches (10-20): Better for parallel experiments
batch_size = 10

# Very large batches (50+): Use sequential=True for speed
acq_fn = BoTorchAcquisition(..., batch_size=50, sequential=True)
```

### 4. Debugging Tips

```python
# Enable logging
import logging
logging.basicConfig(level=logging.INFO)

# Check model predictions
predictions = surrogate.predict(test_candidates)
print(f"Mean: {predictions.means.mean():.4f}")
print(f"Std: {np.sqrt(predictions.variances.mean()):.4f}")

# Check acquisition values
acq_values = acq_fn(candidates, state)
print(f"Acquisition range: [{acq_values.labels.min():.4f}, "
      f"{acq_values.labels.max():.4f}]")
```

## Troubleshooting

### Issue: "Data is not contained to the unit cube"

**Solution**: Normalize your input data to [0, 1]

```python
# Before
bounds = [[-5, 10], [0, 15]]

# After: normalize to [0, 1]
# Or set normalize_inputs=True in BoTorchGPModel
```

### Issue: "Model inputs are of type torch.float32"

**Solution**: Use float64 for better numerical stability

```python
model = BoTorchGPModel(dtype=torch.float64)
```

### Issue: Poor optimization performance

**Solutions**:
1. Increase `num_restarts` (10 → 20)
2. Increase `raw_samples` (512 → 1024)
3. Try different acquisition function (qEI → qUCB)
4. Check data normalization

## Advanced Topics

### Custom Acquisition Function Parameters

```python
# Pass custom parameters via kwargs
acq_fn = BoTorchAcquisition(
    acquisition_type="qEI",
    sampler=sampler,
    bounds=bounds,
    # Custom parameters passed to qExpectedImprovement
    objective=custom_objective,
    constraints=[constraint_fn],
)
```

### Using with Discrete Search Spaces

```python
# BoTorchAcquisition works in discrete mode too
from alf_tools.optimizer.search import DatasetSearch

# Score discrete candidates
optimizer = Optimizer(
    acquisition_fn=acq_fn,  # Same acquisition function!
    search_fn=DatasetSearch(),  # Use dataset pool instead
)
```

## References

- [BoTorch Documentation](https://botorch.org/)
- [BoTorch Paper](https://arxiv.org/abs/1910.06403)
- [Modern GP Priors](https://github.com/meta-pytorch/botorch/discussions/2451)
- [ALF Core Documentation](../core/README.md)

## See Also

- [BoTorch Synthetic Tutorial](../tutorials/botorch_synthetic_tutorial.ipynb) - Step-by-step guide
- [Integration Demo](../examples/botorch_integration_demo.py) - Code examples
- [API Reference](../tools/alf_tools/README.md) - Full API documentation
