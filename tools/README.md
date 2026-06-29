# alf-tools

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

Some models require additional dependencies. Append one or more extras to the package URL.
Per-model extras (`esm2`, `esmfold`, `chemprop`, `guacamol`, `mlip`) install exactly one model's
dependencies; workflow umbrellas (`protein`, `molecule`) group the extras you are likely to use
together.

```bash
pip install "alf-tools[esm2]"      # ESM-2 protein language model (ESM2Model)
pip install "alf-tools[esmfold]"   # ESMFold structure-prediction oracle (ESMFoldModel)
pip install "alf-tools[chemprop]"  # Chemprop small-molecule MPNN (ChempropModel)
pip install "alf-tools[guacamol]"  # GuacaMol RDKit-based dataset/scoring
pip install "alf_tools[mlip]       # MLIP — MACE force field for atomistic systems (for MLIPModel)

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
- **MLIPModel** - Machine-learned interatomic potential built on the `mlip-jax` MACE force field.
  Accepts ASE `Atoms` objects directly and predicts per-structure energies.
  Finetunes from a pretrained foundation model by default, or trains from
  scratch when `model_path=None`; supports dynamic training-set-aware hyperparameters and an
  forces→energy weight-flip loss schedule. Requires the `[mlip]` optional extra:
  `pip install "alf-tools[mlip]"`
- **PyRosetta** - Rosetta energy function for protein design (requires PyRosetta installation)
- **EnsembleWrapper** - Generic wrapper composing N `BaseModel` instances into a seed ensemble,
  MC dropout ensemble, or combined (seed + dropout) ensemble for uncertainty quantification.
  Configured via `EnsembleWrapperConfig`; optional per-member data subsampling via `SubsampleConfig`

### Acquisition Functions
- **Greedy** - Select candidates with highest predicted values
- **UCB** - Upper Confidence Bound for exploration-exploitation
- **ExpectedImprovement** - Expected improvement over current best
- **ThompsonSampling** - Bayesian sampling for exploration
- **UncertaintySampling** - Select the highest-variance candidates to reduce model error
  (maximum-variance / query-by-committee); requires an uncertainty-aware surrogate (ensemble or GP)
- **RandomSampling** - Uniform random baseline for comparing against informed acquisitions
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
