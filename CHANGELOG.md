# Changelog

## Release 0.1.0

- Releasing ALF as an open-source project under the Apache 2.0 license.
- Introducing `alf_core`, a lightweight framework package providing base classes,
  core data structures (`Candidate`, `LabelledCandidates`, `Predictions`,
  `State`), and an active learning loop abstraction with no ML-framework
  dependencies.
- Introducing `alf_tools`, a ready-to-use package of surrogate models, datasets,
  and acquisition functions built on top of `alf_core` with PyTorch.
- Adding surrogate models: `CNNModel`, `MLPModel`, Gaussian Process (`GPModel`),
  `EnsembleWrapper` for deep ensembles, and `ChempropModel` (message-passing neural
  network via Chemprop v2).
- Adding `ESM-2` protein language model surrogate with BERT-style masked language
  model fine-tuning, and `ESMFold` oracle for protein structure confidence scoring.
- Integrating BoTorch for continuous search spaces, including BoTorch-based surrogate
  models and acquisition functions (Expected Improvement, Upper Confidence Bound,
  Thompson Sampling) alongside discrete acquisition functions (Greedy, CoreSet with
  greedy k-centres).
- Adding benchmark datasets: GFP (green fluorescent protein), FLIP, ProteinGym
  (with cross-validation support), and GuacaMol (small-molecule optimisation).
- Adding classification support across the core framework, surrogate models, and
  metrics.
- Adding input normalisation utilities including `InputStandardiser` and a
  configurable normalisation strategy.
- Adding benchmarking metrics for active learning campaigns (recall, regret, and
  acquisition batch metrics) and benchmark example scripts comparing surrogates and
  acquisition functions.
- Adding training metric logging via `TerminalStateLogger` and configurable
  per-round state recording.
- Publishing comprehensive documentation including installation guide, API reference,
  conceptual explanations, how-to recipes, and tutorials following the Diátaxis
  structure.
