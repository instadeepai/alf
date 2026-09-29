# Changelog

## Release 0.1.0

### Added

- `RandomAcquisition` and `UncertaintySampling` acquisition functions, giving a
  random baseline and a pure-exploration strategy.
- `Matbench` dataset and `Modality.MATERIALS`, behind a new optional
  `matbench`/`materials` extra.
- `SmilesMutationSearch` and `ElementSubstitutionSearch`, search protocols that
  propose novel molecules and crystals each round.
- `GuacaMolOracle`, for scoring arbitrary SMILES in online loops.
- `top_k` on `SingleMutantSearch`, to seed mutants from more than just the
  single best sequence.
- `BaseModel.embed()` and `Surrogate.embed()`, giving embeddings a contract
  separate from `featurise()`.

### Changed

- **`CoreSet` now calls `embed()` instead of `featurise()`.** `BaseModel.embed()`
  raises `NotImplementedError` by default, so a custom model that relied on
  `featurise()` returning a 2-D embedding must now override `embed()` —
  `return self.featurise(inputs)` restores the previous behaviour.
- **Tied labels in `LabelledCandidates.get_top_k` and `sort` now break by
  position**, rather than in an unspecified order that varied with array size.
  Ties are routine, so benchmark numbers may shift and reference results may
  need regenerating.
- **`SingleMutantSearch` now deduplicates its proposals**, so the pool it
  returns can be smaller than before. It also rejects empty training sets and
  multi-output labels instead of silently mis-selecting.
- `chemprop` is capped below 2.3.0, and `uv` overrides raise the `matminer`,
  `scipy`, `monty` and `scikit-learn` floors workspace-wide.

### Fixed

- `one_hot_encode()` pads variable-length batches instead of raising
  `IndexError`.
- `BoTorchAcquisition` featurises candidates, fixing GP models whose
  `Candidate.data` is not already numeric.

## Release 0.1.0b0

- Released ALF as beta version under the Apache 2.0 license. First stable release
  will follow shortly.
- Introduced `alf_core`, a lightweight framework package providing base classes,
  core data structures (`Candidate`, `LabelledCandidates`, `Predictions`,
  `State`), and an active learning loop abstraction with no ML-framework
  dependencies.
- Introduced `alf_tools`, a ready-to-use package of surrogate models, datasets,
  and acquisition functions built on top of `alf_core` with PyTorch.
- Added surrogate models: `CNNModel`, `MLPModel`, Gaussian Process (`GPModel`),
  `EnsembleWrapper` for deep ensembles, and `ChempropModel` (message-passing neural
  network via Chemprop v2).
- Added `ESM-2` protein language model surrogate with BERT-style masked language
  model fine-tuning, and `ESMFold` oracle for protein structure confidence scoring.
- Integrated BoTorch for continuous search spaces, including BoTorch-based surrogate
  models and acquisition functions (Expected Improvement, Upper Confidence Bound,
  Thompson Sampling) alongside discrete acquisition functions (Greedy, CoreSet with
  greedy k-centres).
- Added benchmark datasets: GFP (green fluorescent protein), FLIP, ProteinGym
  (with cross-validation support), and GuacaMol (small-molecule optimisation).
- Added classification support across the core framework, surrogate models, and
  metrics.
- Added input normalisation utilities including `InputStandardiser` and a
  configurable normalisation strategy.
- Added benchmarking metrics for active learning campaigns (recall, regret, and
  acquisition batch metrics) and benchmark example scripts comparing surrogates and
  acquisition functions.
- Added training metric logging via `TerminalStateLogger` and configurable
  per-round state recording.
- Published comprehensive documentation including installation guide, API reference,
  conceptual explanations, how-to recipes, and tutorials following the Diátaxis
  structure.
