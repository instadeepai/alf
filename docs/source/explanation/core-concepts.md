# Core Concepts

This page is the mental model of ALF's objects: what each one is, and how they combine into a
single round of the [ask/tell loop](intro-to-active-learning.md). All of these live in
`alf_core`; concrete implementations live in `alf_tools`.

## The objects

| Object | Role | Base class |
|--------|------|------------|
| `Dataset` | The candidate pool and its labels; train/validation splits and a query interface | {py:class}`BaseDataset <alf_core.dataset.base_dataset.BaseDataset>` |
| `Model` | A learnable predictor with `featurise` / `train` / `predict` / `sample` | {py:class}`BaseModel <alf_core.model.base_model.BaseModel>` |
| `Surrogate` | Wraps a model; the *cheap* approximation retrained each round on acquired data | {py:class}`Surrogate <alf_core.surrogate.surrogate.Surrogate>` |
| `Oracle` | Wraps a scorer (a model **or** a dataset); returns the *true* label for a batch | {py:class}`Oracle <alf_core.oracle.oracle.Oracle>` |
| `Acquisition function` | Scores candidates by expected value (e.g. Greedy, UCB, EI, Thompson, CoreSet, Random) | {py:class}`AcquisitionFunction <alf_core.optimizer.acquisition_function.AcquisitionFunction>` |
| `Search function` | Generates the candidate pool to score (e.g. from a dataset or a generator) | {py:class}`BaseSearch <alf_core.optimizer.search.BaseSearch>` |
| `Optimizer` | Bundles an acquisition + search function; implements `ask` and `tell` | {py:class}`Optimizer <alf_core.optimizer.optimizer.Optimizer>` |
| `Task` | Orchestrates the loop end to end (`setup` → `run`) | {py:class}`BaseTask <alf_core.tasks.base_task.BaseTask>` |
| `State` | The carrier passed through the loop: dataset, surrogate, round, history, metrics | {py:class}`State <alf_core.dataclasses.state.State>` |

The key relationships: a **Surrogate wraps a Model**, an **Oracle wraps a scorer**, and an
**Optimizer bundles an acquisition function and a search function**. The **Task** owns the loop and
threads a single **State** object through every step.

## How one round flows

A round is exactly the `ask` then `tell` of the {py:class}`Optimizer <alf_core.optimizer.optimizer.Optimizer>`:

```text
ask(state):
    pool      = search_fn(state)              # Search proposes candidates
    scored    = acquisition_fn(pool, state)   # Acquisition scores them
    batch     = scored.get_top_k(acq_batch_size)
    → return the batch to evaluate

oracle.evaluate(batch)                        # true labels revealed
state.dataset += labelled batch               # knowledge grows

tell(state):
    surrogate.fit(train, val)                 # retrain on the larger dataset
    → update round metrics (regret, recall, calibration, …)
```

After `tell`, the round's metrics are recorded in `state.round_metrics` and the loop repeats for
the configured number of rounds.

## Three task families

The kind of `Task` you pick determines *what* is being optimised. These three map directly onto
ALF's experiment taxonomy:

```text
experiment
├── design      → DesignTask      pick candidates that maximise an objective (the AL loop above)
├── supervised  → SupervisedTask  iteratively label to improve a predictor on a fixed pool
└── zero-shot   → ZeroShotTask    score with a pretrained model, no surrogate training
```

Within a family, an experiment is further specified by **{term}`modality <Modality>`** (e.g. protein sequence,
small-molecule SMILES), **dataset**, and **mode** (`Offline` or `Online`). The full
cross-product, `dataset × surrogate × acquisition × search × oracle × seed`, is exactly what the
ALF benchmark suite sweeps over; see [Why ALF?](why-alf.md).

## Putting it together

The 15-line quickstart on the [home page](../index.md) is precisely these objects wired up: a
`GFP` dataset, a `CNNModel` inside a `Surrogate`, an `Optimizer` holding `Greedy` + `DatasetSearch`,
an `Oracle` scoring against the dataset, run by a `DesignTask`. Once the mental model above is
clear, that snippet reads as "compose the loop from parts."

→ Look up any unfamiliar term in the [Glossary](../reference/glossary.md), or
[try a tutorial](../tutorials/index.md).
