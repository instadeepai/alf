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

### Models
- **CNNModel** - Convolutional neural network for sequence modeling with uncertainty quantification
- **ESMFoldModel** - ESMFold protein structure prediction oracle; returns pTM and/or mean pLDDT scores for amino acid sequence candidates. Use as `Oracle(scorer=ESMFoldModel(ESMFoldConfig(...)))`. Requires `transformers>=4.36.0` and `accelerate>=0.26.0`.
- **GPModel** - Gaussian Process model for sequence fitness prediction with flexible kernel
  selection, input normalisation, and output standardisation enabled by default
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

## Normalisation

Models in `alf-tools` support input normalisation and output standardisation via their train
configs (see `alf_core.model.base_model.BaseTrainConfig`).

| Model | `normalise_inputs` default | `standardise_outputs` default |
|-------|---------------------------|-------------------------------|
| `CNNModel` | `False` | `False` |
| `GPModel` | `True` | `True` |

**`GPTrainConfig`** overrides both defaults to `True`:
- `normalise_inputs=True`: min-max scales features to [0, 1] — GP kernels measure distances and
  benefit from inputs on a common scale.
- `standardise_outputs=True`: Z-score standardises labels before training — improves marginal
  log-likelihood optimisation. Predictions are inverse-transformed back to the original label
  scale before being returned, so **all metrics are computed on the original label scale**.

To disable normalisation for a GP, pass an explicit config:

```python
from alf_tools.models.gp import GPModel, GPTrainConfig

model = GPModel(train_config=GPTrainConfig(normalise_inputs=False, standardise_outputs=False))
```

For implementation details see [`alf_core.model.normaliser`](../core/alf_core/model/normaliser.py)
and the [Core README normalisation section](../core/README.md#9-normalisation-inputnormaliser-outputstandardiser).

## Creating Custom Components

All components extend base classes from `alf_core`:
- Datasets extend `BaseDataset`
- Models extend `BaseModel`
- Acquisition functions extend `AcquisitionFunction`
- Search strategies extend `BaseSearch`

See the [core documentation](../core/README.md) for implementation details.
