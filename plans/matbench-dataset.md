# Feature Plan: Matbench Dataset Integration

**Created**: 2026-03-10
**Status**: Approved

## Summary

Add Matbench as a dataset in `alf_tools/datasets/`, supporting all 13 tasks across
composition-based (TABULAR) and structure-based (STRUCTURE) inputs. Follows the same
`BaseDataset` pattern as FLIP and ProteinGym with a custom config and custom splitting
logic to accommodate Matbench's predefined 5-fold cross-validation.

## Decisions Made

| Decision | Selected Option | Rationale |
|---|---|---|
| Task scope | All 13 tasks | Full coverage of regression + classification, composition + structure |
| Composition modality | TABULAR | Chemical formula strings map naturally to TABULAR |
| Structure modality | STRUCTURE | pymatgen Structure objects map to STRUCTURE |
| Fold handling | `fold_number` in config | Single fold → predefined Matbench split; `None` → merge all folds + ratio-based split |
| Fold traceability | `fold_id` in Candidate.features | Stored on every candidate regardless of fold mode |

## Technical Approach

### MatbenchConfig

Extends `BaseDatasetConfig` with:
- `task_name: str` — one of the 13 Matbench task names (validated)
- `fold_number: int | None` — `0–4` for a predefined fold; `None` to merge all folds

When `fold_number` is set:
- Matbench's predefined 60/20/20 train/val/test split is used
- `train_ratio`, `validation_frac`, `test_ratio` are ignored
- Remaining FLIP-style: non-test train rows become `candidate_pool`

When `fold_number=None`:
- All 5 folds are loaded and merged into a single `LabelledCandidates`
- `fold_id` stored in each candidate's `features` dict for traceability
- Standard ratio-based `_split_dataset()` applies (`train_ratio`, `test_ratio`, etc.)

### Matbench class

- Inherits `BaseDataset`
- `load_dataset()`: uses `matminer.datasets.load_dataset(task_name)` to download/cache data; converts each row to a `Candidate` with appropriate modality + `fold_id` feature
- `_split_dataset()`: overrides base class; dispatches on `fold_number is None`
- Modality is inferred automatically from task name (composition → TABULAR, structure → STRUCTURE); user-provided `modality` in config is validated against this

### Matbench task metadata

Hardcode a mapping of task name → `(input_type, output_type)`:
```python
MATBENCH_TASKS = {
    "matbench_steels":        ("composition", "regression"),
    "matbench_jdft2d":        ("structure",   "regression"),
    "matbench_phonons":       ("structure",   "regression"),
    "matbench_expt_gap":      ("composition", "regression"),
    "matbench_dielectric":    ("composition", "regression"),
    "matbench_log_gvrh":      ("structure",   "regression"),
    "matbench_log_kvrh":      ("structure",   "regression"),
    "matbench_perovskites":   ("structure",   "regression"),
    "matbench_mp_gap":        ("structure",   "regression"),
    "matbench_mp_e_form":     ("structure",   "regression"),
    "matbench_expt_is_metal": ("composition", "classification"),
    "matbench_glass":         ("composition", "classification"),
    "matbench_mp_is_metal":   ("structure",   "classification"),
}
```

### Structure serialisation

pymatgen `Structure` objects are stored as-is in `Candidate.data` under STRUCTURE modality.
`Candidate.to_serializable()` for STRUCTURE currently returns the raw array/tensor —
we extend it to also handle pymatgen Structure (convert to JSON string via `structure.to(fmt="json")`).

### Classification labels

Matbench classification tasks have boolean labels. These are cast to `float` (0.0 / 1.0)
to stay consistent with ALF's `np.ndarray` label convention.

## Implementation Steps

1. [ ] Add `matbench`, `matminer`, and `pymatgen` to `tools/pyproject.toml` as optional extras
      (e.g. `[project.optional-dependencies] matbench = [...]`)
2. [ ] Create `tools/alf_tools/datasets/matbench.py`:
      - `MATBENCH_TASKS` mapping
      - `MatbenchConfig` (Pydantic, validates task_name, fold_number 0-4 or None)
      - `Matbench(BaseDataset)` with `load_dataset()` and `_split_dataset()`
3. [ ] Extend `Candidate.to_serializable()` in `core/alf_core/dataclasses/candidate.py`
      to handle pymatgen Structure objects under STRUCTURE modality
4. [ ] Register `Matbench` in `tools/alf_tools/datasets/__init__.py`
5. [ ] Create `tools/tests/datasets/test_matbench.py` — unit tests (mock matminer download)
6. [ ] Add a usage example to the tutorials or docs

## Files to Create

- `tools/alf_tools/datasets/matbench.py` — dataset implementation
- `tools/tests/datasets/test_matbench.py` — unit tests

## Files to Modify

- `tools/pyproject.toml` — add optional `matbench` extras group
- `tools/alf_tools/datasets/__init__.py` — export `Matbench`, `MatbenchConfig`
- `core/alf_core/dataclasses/candidate.py` — extend `to_serializable()` for pymatgen Structure

## Dependencies

- `matbench` — benchmark harness and fold definitions
- `matminer` — dataset loading and materials feature extraction
- `pymatgen` — crystal structure handling (heavy; added as optional extras)

All three added under `[project.optional-dependencies]` as `alf_tools[matbench]` to avoid
forcing pymatgen on users who only need sequence/tabular datasets.

## Testing Strategy

- Mock `matminer.datasets.load_dataset` to return a synthetic DataFrame (same pattern as FLIP's `_load_split_dataframe` mock)
- Test both fold modes:
  - `fold_number=0`: assert predefined split sizes, no data leakage between splits
  - `fold_number=None`: assert merged pool size, ratio-based splitting, fold_id in features
- Test composition task → TABULAR modality inferred
- Test structure task → STRUCTURE modality inferred
- Test classification labels cast to float
- Test invalid task_name raises `ValueError`
- Test invalid fold_number (e.g. 5) raises `ValueError`

## Notes

- Matbench's predefined folds use `seed=18012019` internally; this is fixed by Matbench
  and cannot be overridden — document clearly in the docstring
- For active learning use cases, the `candidate_pool` in fold mode comes from the
  remaining train rows not allocated to the initial labelled set (same as FLIP)
- Classification tasks (boolean labels) are cast to float for consistency with ALF's
  regression-first design; users should be aware the surrogate will treat these as regression
- pymatgen lazy-imported inside the class to avoid import errors for users without it installed
