# Glossary

Definitions of the terms used throughout the ALF documentation.

```{glossary}
Active learning
  A sequential strategy that, given a fixed labelling budget, repeatedly chooses the most
  informative or highest-value candidates to label next, rather than labelling everything or a
  random sample. ALF implements active learning as an [ask/tell loop](../explanation/intro-to-active-learning.md).

Acquisition function
  The rule that scores candidates by their *expected value to the search* and decides which to
  pick next, trading off exploitation (high predicted value) against exploration (high
  uncertainty). Built-in examples: Greedy, UCB, Expected Improvement, Thompson Sampling, CoreSet.
  Distinct from a `Search function`, which produces the pool the acquisition function scores.

Best-found
  The value of the best candidate discovered so far, tracked as a function of round. The headline
  "are we winning?" curve for a design experiment (see `Task`).

Calibration
  How well a model's stated uncertainty matches its actual error. A well-calibrated
  `Surrogate` "knows what it doesn't know", which is what lets an `Acquisition function`
  explore intelligently. Often summarised by `ECE`.

Candidate
  A single point in the search space (e.g. a protein sequence or a SMILES string), optionally
  carrying features and a label. Represented by `Candidate` / `LabelledCandidates` in `alf_core`.

Coverage
  The fraction of the truly high-value region that a experiment has discovered. Distinguishes finding
  *the* peak from finding *all* the good designs; related to `Recall`.

Dataset
  The candidate pool together with its labels and train/validation splits, plus a query interface.
  Base class `BaseDataset`; examples include GFP, FLIP, ProteinGym, and GuacaMol.

ECE
  *Expected Calibration Error*: a scalar summary of `Calibration`, the average gap between
  predicted confidence and observed accuracy across confidence bins.

Modality
  The representation of a candidate, e.g. a protein `sequence` or a small-molecule SMILES string.
  Determines which `Model` and `Dataset` implementations are compatible.

Model
  A learnable predictor with a common interface: `featurise`, `train`, `predict`, `sample`. Base
  class `BaseModel`; examples include CNN, Gaussian Process, ESM-2, and Chemprop. A model becomes a
  `Surrogate` or an `Oracle` depending on the role it plays in the loop.

Offline
  A loop in which the `Oracle` reads labels from a held-out, pre-scored pool. Fully
  reproducible with no external calls; the mode used for benchmarking and method development.

Online
  A loop in which the `Oracle` obtains labels from a live scorer (a trained model or
  simulator) on demand; the mode used to drive a real experiment. The loop is otherwise identical to
  `Offline`.

Optimizer
  The object that bundles an `Acquisition function` and a `Search function` and exposes
  the `ask` (propose a batch) and `tell` (retrain on new labels) interface. Class `Optimizer`.

Oracle
  The source of *ground-truth* labels for a proposed batch. Wraps a scorer that is either a trained
  `Model` (online) or a `Dataset` (offline). Class `Oracle`.

Precision
  Of the candidates a method selected as high-value, the fraction that truly are. Reported
  alongside `Recall` for classification-style objectives.

Recall
  Of all the truly high-value candidates, the fraction the experiment has selected. Measures whether
  the method is finding the good region, not just one point; see also `Coverage`.

Regret
  The gap between the best achievable value and the best value found so far. Lower is better;
  regret-versus-round is the primary way ALF compares the *speed* of methods.

Search function
  The component that generates the pool of `Candidate` objects an
  `Acquisition function` then scores, e.g. enumerating a dataset (`DatasetSearch`) or
  sampling from a generative `Model` (`GeneratorSearch`). Base class `BaseSearch`.

Spearman correlation
  A rank-correlation metric used to judge how well a `Surrogate`'s predicted ordering of
  candidates matches the true ordering: what matters for selection, regardless of absolute scale.

State
  The mutable carrier threaded through the loop: the current `Dataset`, `Surrogate`,
  round index, acquisition history, and per-round metrics. Class `State`.

Surrogate
  A cheap-to-evaluate probabilistic approximation of the true objective, wrapping a `Model`
  and **retrained each round** on the data acquired so far. Its predictions (and uncertainty) drive
  the `Acquisition function`. Class `Surrogate`.

Task
  The orchestrator that runs the loop end to end via `setup` then `run`. The family of task
  (`DesignTask`, `SupervisedTask`, or `ZeroShotTask`) determines *what* is being optimised. Base
  class `BaseTask`.
```
