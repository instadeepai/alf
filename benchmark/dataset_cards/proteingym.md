# Dataset card: ProteinGym

**Registry name:** `proteingym` · **Modality:** sequence · **Problem type:** regression

## Summary

ProteinGym is a large suite of deep mutational scanning (DMS) assays for
benchmarking variant-effect prediction. In ALF it is an *extended-tier* problem
(requires a download) and is the natural home for the **zero-shot** family once
that family lands; today it is used as a supervised/design regression target.

## Provenance

- **Benchmark:** Notin et al., *"ProteinGym: Large-Scale Benchmarks for Protein
  Fitness Prediction and Design"*, NeurIPS 2023 Datasets & Benchmarks.
- **As consumed by ALF:** selected via `ProteinGymConfig` with `dms_name`,
  `dms_type ∈ {singles, multiples}`, and optional cross-validation
  (`cross_validation`, `cross_validation_type`, `cross_validation_fold`). See
  [`tools/alf_tools/datasets/proteingym.py`](../../tools/alf_tools/datasets/proteingym.py)
  for the source and split semantics.

## Split

When `cross_validation` is enabled, the assay's predefined CV folds are used
(`cross_validation_fold ∈ {0..4}`); otherwise the standard ratio split applies.
Use the predefined folds for results comparable to the ProteinGym literature.

## License

See the ProteinGym project for per-assay licences and citation requirements;
confirm terms before redistribution.

## Known limitations / leakage notes

- DMS assays differ in size, sparsity, and measurement type — aggregate across
  assays with care.
- For zero-shot scoring, the model must not have seen the target during
  pretraining; ALF does not enforce this, so document the model's pretraining
  corpus when reporting zero-shot numbers.
