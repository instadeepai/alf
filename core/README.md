# ALF Core

This document provides an overview of the core components in the ALF (Active Learning
Framework) library, describes the different task types, and explains how components
interact during execution.

<div align="center">
  <img src="../docs/imgs/alf_components.svg" alt="ALF Components" width="70%">
</div>

## Overview

This README is organized into the following sections:

### Core Components
- **[1. Dataset (`BaseDataset`)](#1-dataset-basedataset)** - Data loading, splitting, and querying
- **[2. Model (`BaseModel`)](#2-model-basemodel)** - Abstract base class for all models
- **[3. Surrogate (`Surrogate`)](#3-surrogate-surrogate)** - Approximates expensive
  experimental evaluation
- **[4. Oracle (`Oracle`)](#4-oracle-oracle)** - Provides ground-truth labels for candidates
- **[5. Optimizer (`Optimizer`)](#5-optimizer-optimizer)** - Orchestrates the active learning loop
- **[6. Acquisition Function (`AcquisitionFunction`)](#6-acquisition-function-acquisitionfunction)**
  - Scores candidates for acquisition
- **[7. Search Strategy (`BaseSearch`)](#7-search-strategy-basesearch)** - Defines the
  candidate pool
- **[8. State (`State`)](#8-state-state)** - Tracks the state of active
  learning tasks
- **[9. Normalisation (`InputNormaliser`, `OutputStandardiser`)](#9-normalisation-inputnormaliser-outputstandardiser)**
  - Feature and label preprocessing

### Task Types
- **[1. Design Task (`DesignTask`)](#1-design-task-designtask)** - Multi-round optimisation loop
- **[2. Supervised Task (`SupervisedTask`)](#2-supervised-task-supervisedtask)** - Train and
  evaluate on fixed data
- **[3. Zero-Shot Task (`ZeroShotTask`)](#3-zero-shot-task-zeroshottask)** - Evaluate
  pre-trained models

### Component Flow
- **[Design Task Flow](#design-task-flow)**
- **[Supervised Task Flow](#supervised-task-flow)**
- **[Zero-Shot Task Flow](#zero-shot-task-flow)**

### Evaluation Utilities
- **[Evaluation Metrics](#evaluation-metrics)** - Metrics for predictions of the surrogate model

## Core Components

### 1. Dataset (`BaseDataset`)

The `BaseDataset` class manages data loading, splitting, and querying. It handles:

- **Data Loading**: Loads raw labelled data through the abstract `load_dataset()` method
- **Data Splitting**: Splits data into train, validation, test, and candidate_pool sets
- **Split Updates**: Distributes newly acquired data into existing data splits
- **Querying**: Provides labels for candidates from the original dataset (used by the
  oracle in offline settings)

**Key Methods:**
- `load_dataset()`: Every child class needs to implement how to load the dataset and
  return it as `LabelledCamdidates`
- `_split_dataset()`: Splits the dataset based on the split config, which specifies the
  ratio of data points in train/val/test/candidate_pool sets and the splitting method
  (random or low_vs_high)
- `update_splits()`: Updates train/validation splits with newly acquired candidates
- `query()`: Returns labels for given candidates (for offline evaluation)

### 2. Model (`BaseModel`)

The `BaseModel` is an abstract base class that defines the interface for all models in
the framework. Models can serve multiple roles depending on the context:

- **Surrogate Model**: Wrapped by `Surrogate` to approximate expensive experimental evaluations
- **Oracle Model**: Wrapped by by `Oracle` for online evaluation (simulating real experiments)
- **Generator Model**: Wrapped by `GeneratorSearch` to sample candidate sequences from the model

**Key Abstract Methods:**
- `featurise()`: Converts inputs (candidates or labelled data) into features suitable for the model
- `train()`: Trains the model on labelled training and validation data
- `predict()`: Generates predictions (means and uncertainties) for candidate sequences
- `sample()`: Samples new candidate sequences from the model (used for generative search strategies)

**Optional Methods:**
- `get_training_summary_metrics()`: Returns training metrics (e.g., loss, accuracy) -
  defaults to empty dict
- `cleanup()`: Cleans up temporary files, checkpoints, or other resources - defaults to no-op

**Training Configuration (`BaseTrainConfig`):**

Concrete model implementations pair with a `BaseTrainConfig` dataclass that exposes normalisation
flags alongside standard training hyperparameters:

- `normalise_inputs: bool` — apply min-max input normalisation (default `False`)
- `standardise_outputs: bool` — apply Z-score output standardisation (default `False`)

See [section 9](#9-normalisation-inputnormaliser-outputstandardiser) for full details on both
normalisation routines.

**Implementation Notes:**
- All concrete model implementations must inherit from `BaseModel` and implement all
  abstract methods, in case a particular method cannot be implemented (e.g., train for
  an oracle model) it should raise a NotImplementedError describing the reason
- The `predict()` method should return a `Predictions` object containing both mean
  predictions and uncertainty estimates
- The `sample()` method is particularly important for generative search strategies,
  where the model generates the candidate pool

### 3. Surrogate (`Surrogate`)

The surrogate approximates the expensive experimental evaluation. It wraps a `BaseModel`
and provides:

- **Training**: Fits the model on labelled training data
- **Prediction**: Makes predictions on candidate sequences
- **Metrics**: Tracks training metrics and performance

**Key Methods:**
- `fit()`: Trains the surrogate on train/validation data
- `predict()`: Generates predictions (means and uncertainties) for candidates
- `get_training_summary_metrics()`: Returns training metrics

### 4. Oracle (`Oracle`)

The oracle provides ground-truth labels for candidates. It can be:

- **Offline**: Uses a `BaseDataset` to query labels from existing data
- **Online**: Uses a `BaseModel` to generate labels (simulating real experiments)

**Key Methods:**
- `evaluate()`: Evaluates candidates and returns labeled results

### 5. Optimizer (`Optimizer`)

The optimizer orchestrates the active learning loop through the ask-tell interface:

- **Ask**: Proposes the next batch of candidates to evaluate
- **Tell**: Updates the surrogate with newly acquired data

**Components:**
- **Acquisition Function**: Scores candidates based on surrogate predictions
- **Search Strategy**: Defines the pool of candidates to acquire from

**Key Methods:**
- `ask()`: Returns the next acquired batch of candidates to evaluate
- `tell()`: Trains the surrogate on updated newly acquired data and returns metrics

### 6. Acquisition Function (`AcquisitionFunction`)

Acquisition functions determine which candidates are most promising to evaluate. They
score candidates based on:

- Surrogate model predictions (means and uncertainties)
- Current task state (training data, round number, etc.)

**Common Acquisition Functions:**
- **Greedy**: Selects candidates with highest predicted values
- **UCB (Upper Confidence Bound)**: Balances exploitation and exploration
- **Expected Improvement**: Selects candidates with highest expected improvement
- **Thompson Sampling**: Uses Bayesian sampling for exploration

### 7. Search Strategy (`BaseSearch`)

Search strategies define the candidate pool available for acquisition. Types include:

- **DatasetSearch**: Uses the candidate pool from the dataset (offline experiments)
- **GeneratorSearch**: Samples candidates from a generative model
- **ProtocolSearch**: Uses a custom protocol to generate candidates
- **ModelProtocolSearch**: Combines a model with a protocol

**Key Methods:**
- `__call__()`: Returns the list of candidates to search over
- `get_metrics()`: Returns search-specific metrics (e.g., recall, regret)

### 8. State (`State`)

The `State` dataclass tracks the complete state of an active learning task:

- **Components**: Dataset, surrogate model, current round
- **History**: Records all acquired candidates per round
- **Metrics**: Tracks performance metrics for each round
- **Configuration**: Acquisition batch size, number of rounds, etc.

**Key Methods:**
- `update()`: Adds newly acquired candidates to history, updates dataset splits, and increments the round counter

### 9. Normalisation (`InputNormaliser`, `OutputStandardiser`)

ALF provides two preprocessing classes in `alf_core.model.normaliser` for feature and label scaling.
Both are fitted exclusively on training data and applied consistently at predict time to avoid data leakage.

**`InputNormaliser`** — min-max scaling of input features to [0, 1]:
- Statistics (per-feature min and range) are computed over the batch dimension, so each feature
  dimension is scaled independently.
- Supports 2-D inputs `(n_samples, n_features)` and higher-dimensional tensors such as one-hot
  encoded sequences `(n_samples, alphabet_size, seq_len)`.
- Edge case: a feature with zero range is clamped to `_MIN_RANGE = 1e-8` to avoid division by zero.
  For one-hot inputs, ensure all amino acids appear at all positions in training, or clip outputs.
- Well suited for GP models, where kernels measure distances between inputs and benefit from inputs
  spanning the unit cube [0, 1].

**`OutputStandardiser`** — Z-score standardisation of output labels to zero mean and unit variance:
- `inverse_transform(mean, var)` maps predictions back to the original label scale:
  `mean_orig = mean_std * std + mean_train`, `var_orig = var_std * std²`
- Edge case: near-zero training standard deviation is clamped to `_MIN_STD = 1e-8`.
- **When `standardise_outputs=True`, all evaluation metrics are computed on the original
  (inverse-transformed) label scale.** Predictions returned by `predict()` are always in the
  original label space.

Both are controlled via `BaseTrainConfig` flags (see section 2):

| Flag | Default | Effect |
|------|---------|--------|
| `normalise_inputs` | `False` | Apply `InputNormaliser` (min-max) to input features |
| `standardise_outputs` | `False` | Apply `OutputStandardiser` (Z-score) to output labels |

Concrete model configs may override these defaults; for example, `GPTrainConfig` sets both to `True`
because GP kernels operate in distance space and benefit from standardised targets.

## Task Types

### 1. Design Task (`DesignTask`)

The design task implements a multi-round active learning loop for optimizing sequences:

**Workflow:**
1. **Initialization**: Setup dataset and surrogate
2. **For each round:**
   - **Ask**: Optimizer proposes candidates using search + acquisition
   - **Evaluate**: Oracle labels the candidates
   - **Update**: Add labeled candidates to training data
   - **Tell**: Retrain surrogate on updated data
   - **Evaluate**: Assess surrogate performance on test set
   - **Log**: Record metrics and save results

**Use Case**: Iteratively improve sequences by actively selecting and evaluating
promising candidates.

**Components Required:**
- Dataset
- Surrogate model
- Optimizer (with acquisition function and search strategy)
- Oracle
- Logger

### 2. Supervised Task (`SupervisedTask`)

The supervised task trains a model on fixed training data and evaluates it:

**Workflow:**
1. **Setup**: Initialize dataset and surrogate
2. **Train**: Fit surrogate on train/validation splits
3. **Evaluate**: Assess performance on test set
4. **Save**: Save predictions and metrics

**Use Case**: Evaluate model performance on a fixed dataset split (no active learning).

**Components Required:**
- Dataset
- Surrogate model
- Logger

### 3. Zero-Shot Task (`ZeroShotTask`)

The zero-shot task evaluates a pre-trained or untrained model without training:

**Workflow:**
1. **Setup**: Initialize dataset and surrogate
2. **Evaluate**: Make predictions on test set (no training)
3. **Save**: Save predictions and metrics

**Use Case**: Evaluate pre-trained models or baseline performance without training.

**Components Required:**
- Dataset
- Surrogate model (pre-trained)
- Logger

## Component Flow

### Design Task Flow
1. **Setup Phase**:
   ```
   Task.setup(dataset, surrogate) → State
   ```

2. **Round Loop** (for each acquisition round):
   ```
   a. Optimizer.ask(state) → candidates
      ├─ Search(state) → search_candidates
      ├─ Surrogate.predict(search_candidates) → predictions
      └─ Acquisition(predictions) → top_k candidates

   b. Oracle.evaluate(candidates, state) → labeled_candidates

   c. State.update(labeled_candidates)
      └─ Dataset.update_splits(labeled_candidates)

   d. Optimizer.tell(state, logger) → updated_state
      └─ Surrogate.fit(train_data, val_data)

   e. Task.evaluate(state) → updated_state
      ├─ Surrogate.predict(test_data) → predictions
      ├─ Results(predictions, targets) → metrics
      └─ State.save(metrics, history)
   ```

### Supervised Task Flow
1. **Setup Phase**:
   ```
   Task.setup(dataset, surrogate) → State
   ```

2. **Training Phase**:
   ```
   Surrogate.fit(train_data, val_data, logger)
   ```

3. **Evaluation Phase**:
   ```
   Task.evaluate(state)
   ├─ Surrogate.predict(test_data) → predictions
   ├─ Results(predictions, targets) → metrics
   └─ State.save(metrics)
   ```

### Zero-Shot Task Flow
1. **Setup Phase**:
   ```
   Task.setup(dataset, surrogate) → State
   ```

2. **Evaluation Phase** (no training):
   ```
   Task.evaluate(state)
   ├─ Surrogate.predict(test_data) → predictions
   ├─ Results(predictions, targets) → metrics
   └─ State.save(metrics)
   ```

## Evaluation Metrics

ALF provides comprehensive utilities for evaluating surrogate model predictions through metrics (see `utils/metrics.py`). Metrics are automatically added to the regsistry and categorized by whether variance is needed in the calculation of the metric:

**Accuracy Metrics** (no variance required):
- **MSE**: Mean Squared Error between predictions and targets
- **Pearson**: Pearson correlation between predictions and targets
- **Spearman**: Spearman correlation between predictions and targets
- **Pairwise XEnt**: Ranking loss for pairwise classification

**Calibration Metrics** (variance required):
- **ECE** (Expected Calibration Error): Area between observed coverage and ideal calibration curve (see [this](https://arxiv.org/abs/1706.04599) paper for more details)
- **Rank ECE**: ECE computed in rank space using Monte Carlo ranking
- **Coverage**: Percentage of targets falling within confidence intervals at a given alpha level
- **Rank Coverage**: Coverage computed in rank space
- **Width**: Average confidence interval width normalized by dataset range
- **Rank Width**: Width computed in rank space

**Uncertainty Quantification (UQ) Metrics** (variance required):
- **Residual Spearman**: Spearman correlation between absolute residuals and predicted variances
- **Residual Pearson**: Pearson correlation between absolute residuals and standard deviations

**Acquisition Performance Metrics** (variance required):
- **Regret UCB Alpha**: UCB acquisition regret comparing selected vs optimal candidates
- **Regret UCB Alpha Sweep**: UCB regret computed across multiple alpha exploration parameters

All metrics accept predictions (means, variances, targets) and return a dictionary of computed values. Metrics requiring variance will validate that uncertainty estimates are provided.

> **Normalisation and metrics:** When `standardise_outputs=True` in the model's train config,
> predictions are inverse-transformed back to the original label scale before metrics are computed.
> Metrics therefore always reflect performance in original label units, regardless of whether
> output standardisation was used during training.
