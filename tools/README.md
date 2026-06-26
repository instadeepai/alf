# alf-tools

[![PyPI](https://img.shields.io/pypi/v/alf-tools.svg)](https://pypi.org/project/alf-tools/)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/instadeepai/alf/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-instadeepai.github.io%2Falf-blue)](https://instadeepai.github.io/alf/)

**Ready-to-use models, datasets, and acquisition functions for ALF (Active Learning Framework).**

`alf-tools` builds on [alf-core](https://github.com/instadeepai/alf/blob/main/core/README.md)
to give you everything needed to run active-learning experiments out of the box — example
datasets (GFP, ProteinGym, FLIP, GuacaMol), models (CNN, Gaussian Process, ESM-2, Chemprop),
acquisition functions, and search strategies. Use it for quick-start and prototyping; reach for
`alf-core` alone when you want the lightweight framework with no ML-framework dependencies.

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
# ESM2 — protein language model (for ESM2Model)
pip install "alf-tools[esm2]"

# ESMFold — structure-prediction oracle (for ESMFoldModel)
pip install "alf-tools[esmfold]"

# Chemprop — small-molecule MPNN (for ChempropModel)
pip install "alf-tools[chemprop]"

# GuacaMol — RDKit-based small-molecule dataset/scoring
pip install "alf-tools[guacamol]"

# Workflow umbrellas: protein (esm2 + esmfold) or molecule (chemprop + guacamol)
pip install "alf-tools[protein]"
pip install "alf-tools[molecule]"
```

For development installs from the cloned repository, these extras are re-exposed as dependency
groups, so pass `--group` flags to `uv sync`:

```bash
uv sync --group esm2
uv sync --group chemprop
uv sync --group protein              # esm2 + esmfold
uv sync --group molecule             # chemprop + guacamol
uv sync --group esm2 --group chemprop
```

## Documentation

For detailed API documentation and tutorials, see:
- **Full documentation:** [instadeepai.github.io/alf](https://instadeepai.github.io/alf/)
- **Core framework:** [alf-core](https://github.com/instadeepai/alf/blob/main/core/README.md)
- **Installation guide:** [instadeepai.github.io/alf/installation.html](https://instadeepai.github.io/alf/installation.html)
- **Tutorials:** [tutorials/](https://github.com/instadeepai/alf/tree/main/tutorials)

## What's Included

### Datasets
- **GFP** - Green Fluorescent Protein fitness dataset
- **ProteinGym** - Protein sequence datasets from ProteinGym benchmark
- **FLIP** - Fitness Landscape Inference for Proteins benchmark (AAV, GB1, Meltome, SCL, SAV)
- **GuacaMol** - ~1.6 M drug-like SMILES from ChEMBL with 10 RDKit physicochemical properties (MolLogP, TPSA, QED, etc.)

### Models
- **CNNModel** - Convolutional neural network for sequence modeling. Supports `ProblemType.REGRESSION`, `ProblemType.BINARY`, and `ProblemType.MULTICLASS`; output shape and activation are determined automatically from the dataset's `problem_type`
- **GPModel** - Gaussian Process model for sequence fitness prediction with flexible kernel
  selection, input normalisation, and output standardisation enabled by default
- **ESM2Model** - Protein language model surrogate backed by
  [ESM-2](https://huggingface.co/docs/transformers/model_doc/esm). Accepts amino acid sequences
  directly. Three operating modes via `ESM2TrainConfig`:
  `mode='linear_head'` (default) trains a linear head for regression
  (`loss_fn='mse'`) or classification (`loss_fn='cross_entropy'`) on top of the backbone —
  frozen by default, or fine-tuned jointly when `freeze_backbone=False` (the backbone then
  uses `backbone_learning_rate`);
  `mode='likelihoods'` with frozen backbone performs zero-shot pseudo-log-likelihood (PLL)
  scoring with no training; `mode='likelihoods'` with `freeze_backbone=False, loss_fn='mlm'`
  fine-tunes the full ESM-2 backbone via MLM on the input sequences (labels are ignored, but
  `train()` still takes `LabelledCandidates` — placeholder labels are fine) and then returns
  PLL scores.
  Embeddings can be extracted via `embed()`. Requires the `[esm2]` optional extra:
  `pip install "alf-tools[esm2]"`
- **ESMFoldModel** - ESMFold protein structure prediction oracle; returns a scalar confidence score per candidate. Use as `Oracle(scorer=ESMFoldModel(ESMFoldModelConfig(...)))`. Requires `transformers>=4.36.0` and `accelerate>=0.26.0`.
  Three scoring metrics are supported (all in **[0, 1]**, higher is better):
  - `ptm` *(default)* — global fold confidence (pTM). > 0.5 = confident fold; < 0.1 = disordered / very short peptide.
  - `mean_plddt` — per-residue local accuracy averaged over all residues. > 0.7 = well-structured; < 0.5 = disordered.
  - `combined` — weighted average `w * ptm + (1-w) * mean_plddt` (default `w=0.5`); balances global and local confidence.
- **ChempropModel** - Message Passing Neural Network (MPNN) for small-molecule fitness prediction,
  backed by [Chemprop v2.x](https://chemprop.readthedocs.io/). Accepts SMILES strings directly;
  no hand-crafted features required. Requires the `[chemprop]` optional extra:
  `pip install "alf-tools[chemprop]"`
- **PyRosetta** - Rosetta energy function for protein design (requires PyRosetta installation)
- **EnsembleWrapper** - Generic wrapper composing N `BaseModel` instances into a seed ensemble,
  MC dropout ensemble, or combined (seed + dropout) ensemble for uncertainty quantification.
  Configured via `EnsembleWrapperConfig`; optional per-member data subsampling via `SubsampleConfig`

### Acquisition Functions
- **Greedy** - Select candidates with highest predicted values
- **UCB** - Upper Confidence Bound for exploration-exploitation
- **ExpectedImprovement** - Expected improvement over current best
- **ThompsonSampling** - Bayesian sampling for exploration
- **CoreSet** - Greedy k-centres selection for input-space diversity (coverage-based); uses
  `surrogate.featurise()` rather than predictions, so it is compatible with any model and
  does not require uncertainty estimates
- **BoTorchAcquisition** - Unified wrapper around BoTorch's analytic and Monte Carlo acquisition
  functions, selected via a single `acquisition_type` argument (`"qEI"`, `"qLogEI"`, `"qNEI"`,
  `"qUCB"`, `"log_expected_improvement"`, `"upper_confidence_bound"`,
  `"probability_of_improvement"`, `"log_noisy_expected_improvement"`). Supports discrete candidate
  scoring and continuous `optimize_acqf` optimisation. Accepts either a native BoTorch model or an
  ALF `BaseModel` (wrapped automatically via `BotorchModelWrapper`); ALF models must provide
  prediction variances

### Search Strategies
- **SingleMutantSearch** - Generate single-mutation variants of reference sequences

## Quick Example

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

### Diversity-based Selection with CoreSet

`CoreSet` selects candidates that maximise coverage of the input space rather than
predicted fitness. It is a drop-in replacement for any other acquisition function:

```python
from alf_tools.optimizer.acquisition_functions import CoreSet

optimizer = Optimizer(acquisition_fn=CoreSet(), search_fn=DatasetSearch())
```

Candidates are ranked by their greedy k-centres selection order; the first chosen
candidate receives the highest score and unselected candidates receive 0. Because
`CoreSet` calls `surrogate.featurise()` internally — not `predict()` — it works with
any model and requires no uncertainty estimates.

## Normalisation

Models in `alf-tools` support input normalisation and output standardisation via their train
configs (see `alf_core.model.base_model.BaseTrainConfig`).

| Model | `normalise_inputs_strategy` default | `standardise_outputs` default |
|-------|-------------------------------------|-------------------------------|
| `CNNModel` | `None` | `False` |
| `GPModel` | `"minmax"` | `True` |
| `ESM2Model` | `None` | `False` |
| `ChempropModel` | `None` | `False` |

**`GPTrainConfig`** defaults to `normalise_inputs_strategy="minmax"` and `standardise_outputs=True`:
- `"minmax"`: min-max scales features to [0, 1] — GP kernels measure distances and benefit from
  inputs on a common scale.
- `standardise_outputs=True`: Z-score standardises labels before training — improves marginal
  log-likelihood optimisation. Predictions are inverse-transformed back to the original label
  scale before being returned, so **all metrics are computed on the original label scale**.

The `"zscore"` strategy (`InputStandardiser`) zero-centres continuous features and is generally
preferred for deep neural networks. It is **not** enabled by default for `CNNModel`, whose one-hot
sequence inputs are degraded by standardisation; set `normalise_inputs_strategy="zscore"` explicitly
when feeding a CNN continuous features.

To disable input normalisation for a GP, pass an explicit config:

```python
from alf_tools.models.gp import GPModel, GPTrainConfig

model = GPModel(train_config=GPTrainConfig(normalise_inputs_strategy=None, standardise_outputs=False))
```

For implementation details see [`alf_core.model.normaliser`](https://github.com/instadeepai/alf/blob/main/core/alf_core/model/normaliser.py)
and the [Core README normalisation section](https://github.com/instadeepai/alf/blob/main/core/README.md#9-normalisation-inputnormaliser-inputstandardiser-outputstandardiser).

## Creating Custom Components

All components extend base classes from `alf_core`:
- Datasets extend `BaseDataset`
- Models extend `BaseModel`
- Acquisition functions extend `AcquisitionFunction`
- Search strategies extend `BaseSearch`

See the [core documentation](https://github.com/instadeepai/alf/blob/main/core/README.md) for implementation details.
