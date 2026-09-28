# Changelog

## Release 0.1.0

### Added

- `RandomAcquisition`, a random-scoring baseline acquisition function. Takes a
  required `seed`, from which a per-round RNG is derived.
- `UncertaintySampling`, a pure-exploration acquisition function scoring
  candidates by predictive standard deviation.
- `Matbench` dataset (13 materials-property tasks), behind a new optional
  `matbench`/`materials` extra.
- `Modality.MATERIALS`, used by `Matbench` candidates in place of
  `Modality.TABULAR`. The `intra_batch_diversity` metric does not support it
  yet and raises `NotImplementedError`.
- `GuacaMolOracle`, for scoring arbitrary SMILES in online loops without
  downloading the GuacaMol corpus. Returns `0.0` for invalid SMILES rather than
  raising.
- `SmilesMutationSearch` and `ElementSubstitutionSearch`, search protocols that
  propose novel molecules and crystals each round.
  `ElementSubstitutionSearch` requires an `allowed_elements` whitelist.
- `top_k` on `SingleMutantSearch`, seeding mutants from the top-k best-labelled
  sequences rather than only the best. Defaults to `1`.
- `BaseModel.embed()` and `Surrogate.embed()`, giving embeddings a contract
  separate from `featurise()`.

### Changed

- `CoreSet` now calls `embed()` instead of `featurise()`, fixing a crash on
  `ESM2Model` surrogates. **`BaseModel.embed()` raises `NotImplementedError` by
  default**, so a custom model that relied on `featurise()` returning a 2-D
  embedding must now also override `embed()`; a one-line
  `return self.featurise(inputs)` restores the previous behaviour.
- Tied labels in `LabelledCandidates.get_top_k` and `sort` now break by
  position, rather than in an unspecified order that varied with array size.
  Tied acquisition scores are routine (`ThompsonSampling`, `BoTorchAcquisition`
  q-batches, `CoreSet`), so **benchmark numbers may shift** and reference
  results may need regenerating.
- `SingleMutantSearch` now deduplicates its proposals, so the pool it returns
  can be smaller than before. It also raises `ValueError` on an empty training
  set, and on multi-output labels that cannot be squeezed to one dimension,
  where it previously selected against the wrong axis.
- `chemprop` is capped below 2.3.0 and will resolve to 2.2.x. Version 2.3.0
  promotes `cuik_molmaker` to a required dependency, which needs native X11
  libraries at import time.
- Installing the workspace now applies `uv` dependency overrides raising
  `matminer`, `scipy`, `monty` and `scikit-learn` to the floors `matbench`
  needs on Python 3.12. These apply workspace-wide, not just to the `matbench`
  extra.

### Fixed

- `one_hot_encode()` now zero-pads variable-length batches instead of raising
  `IndexError`, fixing featurisation of datasets such as FLIP's AAV.
- `BoTorchAcquisition` now featurises candidates, fixing GP models configured
  with `featurizer_type="one_hot"` or `"custom"`.

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
