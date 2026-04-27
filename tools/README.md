# ALF Tools

Ready-to-use implementations for the ALF framework. This package provides example datasets, models, acquisition functions, and search strategies to get you started quickly.

## Installation

```bash
# Install tools package (includes PyTorch)
pip install git+https://github.com/instadeepai/alf.git#subdirectory=tools
```

**Note:** Requires authentication via `.netrc` file (see main [README](../README.md#authentication))

## What's Included

### Datasets
- **GFP** - Green Fluorescent Protein fitness dataset
- **ProteinGym** - Protein sequence datasets from ProteinGym benchmark
- **FLIP** - Fitness Landscape Inference for Proteins benchmark (AAV, GB1, Meltome, SCL, SAV)
- **GuacaMol** — Physicochemical property optimisation over SMILES. Supports `split_mode="paper"` for original train/valid/test boundaries, `split_mode="low_vs_high"` for cold-start AL scenarios, and `query()` for scoring arbitrary novel molecules without a fixed pool.

### Models
- **CNNModel** - Convolutional neural network for sequence modeling with uncertainty quantification
- **PyRosetta** - Rosetta energy function for protein design (requires PyRosetta installation)

### Acquisition Functions
- **Greedy** - Select candidates with highest predicted values
- **UCB** - Upper Confidence Bound for exploration-exploitation
- **ExpectedImprovement** - Expected improvement over current best
- **ThompsonSampling** - Bayesian sampling for exploration

### Search Strategies
- **SingleMutantSearch** - Generate single-mutation variants of reference sequences

## Quick Example

```python
from alf_core import Optimizer, DatasetSearch, Oracle, Surrogate, DesignTask
from alf_tools.datasets import GFP
from alf_tools.models import CNNModel
from alf_tools.optimizer.acquisition_functions import Greedy

# Load dataset and initialize components
dataset = GFP(name="gfp", modality="sequence", seed=42, split_config=split_config)
surrogate = Surrogate(model=CNNModel())
optimizer = Optimizer(acquisition_fn=Greedy(), search_fn=DatasetSearch())
oracle = Oracle(scorer=dataset)

# Run active learning
task = DesignTask(num_acq_rounds=5, acq_batch_size=100)
state = task.setup(dataset=dataset, surrogate=surrogate)
task.run(state=state, optimizer=optimizer, oracle=oracle)
```

## Documentation

For detailed API documentation and tutorials, see:
- **Full documentation:** [instadeepai.github.io/alf](https://instadeepai.github.io/alf/)
- **Core framework:** [../core/README.md](../core/README.md)
- **Installation guide:** [../docs/INSTALLATION.md](../docs/INSTALLATION.md)
- **Tutorials:** [../tutorials/](../tutorials/)

## Creating Custom Components

All components extend base classes from `alf_core`:
- Datasets extend `BaseDataset`
- Models extend `BaseModel`
- Acquisition functions extend `AcquisitionFunction`
- Search strategies extend `BaseSearch`

See the [core documentation](../core/README.md) for implementation details.
