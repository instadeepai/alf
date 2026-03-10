# Feature Plan: Matbench Dataset Integration

**Created**: 2026-03-10
**Status**: Approved (post-challenge review)

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
| Structure modality | STRUCTURE | pymatgen Structure JSON string stored in Candidate.data |
| Fold handling | `fold_number` in config | Single fold → predefined Matbench split; `None` → merge all folds + ratio-based split |
| Fold traceability | `fold_id` in Candidate.features | Stored on every candidate regardless of fold mode |
| Serialisation | JSON string in matbench.py | Convert Structure before creating Candidate; no changes to alf_core |
| Loading API | `MatbenchBenchmark` only | Correct benchmark-comparable splits; matminer not needed |
| Modality | Auto-inferred from task_name | Set in `validate_config()`; user never specifies it |
| Config validation | Override `validate_config()` | Skip ratio sum check (ignored in fold mode); mirrors FLIP pattern |

## Technical Approach

### MatbenchConfig

Extends `BaseDatasetConfig` with:
- `task_name: str` — one of the 13 Matbench task names (validated against `MATBENCH_TASKS`)
- `fold_number: int | None = None` — `0–4` for a predefined fold; `None` to merge all folds
- `split_type: Literal["random", "low_vs_high"] = "random"` — used only in `fold_number=None` mode
- `modality` — **auto-set** in `validate_config()` from task_name; user does not provide it

`validate_config()` overrides the base class validator to:
1. Look up the task in `MATBENCH_TASKS`, raise `ValueError` for unknown tasks
2. Validate `fold_number` is `None` or in `0–4`
3. Set `self.modality` from the task's input type (composition → TABULAR, structure → STRUCTURE)
4. Skip the `train_ratio + test_ratio <= 1` check (ratios are ignored in fold mode)

### MatbenchConfig example

```python
# Fold mode — ratios and modality not required
config = MatbenchConfig(
    name="matbench_mp_e_form_fold0",
    task_name="matbench_mp_e_form",
    fold_number=0,
    seed=42,
    train_ratio=0.1,   # used only for candidate_pool split within fold train set
    validation_frac=0.1,
    test_ratio=0.2,    # ignored in fold mode; Matbench test set used directly
)

# Merged mode — all 5 folds combined, ratio-based splitting
config = MatbenchConfig(
    name="matbench_mp_e_form_merged",
    task_name="matbench_mp_e_form",
    fold_number=None,
    seed=42,
    train_ratio=0.1,
    validation_frac=0.1,
    test_ratio=0.2,
)
```

When `fold_number` is set (0–4):
- `MatbenchBenchmark` API provides the predefined train and test sets for that fold
- `train_ratio` controls what fraction of the Matbench train set becomes the initial
  labelled set; the remainder becomes `candidate_pool` (same as FLIP)
- `test_ratio` and `split_type` are ignored; Matbench test set used directly

When `fold_number=None`:
- All 5 folds are loaded via `MatbenchBenchmark` and merged into a single `LabelledCandidates`
- `fold_id` (0–4) stored in each candidate's `features` dict for traceability
- Standard ratio-based `_split_dataset()` applies (`train_ratio`, `test_ratio`, `split_type`)
- ⚠️ Loses Matbench's benchmark integrity guarantees — document clearly

### Matbench class

- Inherits `BaseDataset`
- `load_dataset()`: uses `MatbenchBenchmark` API to download/cache and load all data;
  for structure tasks, converts pymatgen `Structure` to JSON string before creating `Candidate`;
  stores `fold_id` in candidate features
- `_split_dataset()`: overrides base class; dispatches on `fold_number is None`

### Structure serialisation

pymatgen `Structure` objects are converted to JSON strings **in `matbench.py`** before
creating `Candidate` objects:

```python
# In matbench.py load_dataset()
data = structure.to(fmt="json")  # str
candidate = Candidate(data=data, modality=Modality.STRUCTURE, features={"fold_id": fold_id})
```

No changes to `alf_core/dataclasses/candidate.py`. pymatgen stays entirely within `alf_tools`.

### Matbench task metadata

```python
MATBENCH_TASKS: dict[str, tuple[str, str]] = {
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

### Classification labels

Matbench classification tasks have boolean labels. Cast to `float` (0.0 / 1.0) for
consistency with ALF's `np.ndarray` label convention.

## Implementation Steps

1. [ ] Add `matbench` and `pymatgen` to `tools/pyproject.toml` as optional extras
      (`[project.optional-dependencies] matbench = ["matbench", "pymatgen"]`)
2. [ ] Create `tools/alf_tools/datasets/matbench.py`:
      - `MATBENCH_TASKS` mapping
      - `MatbenchConfig` with auto-inferred modality, overridden `validate_config()`
      - `Matbench(BaseDataset)` with `load_dataset()` using `MatbenchBenchmark` API
        and `_split_dataset()` dispatching on `fold_number`
3. [ ] Register `Matbench` and `MatbenchConfig` in `tools/alf_tools/datasets/__init__.py`
4. [ ] Create `tools/tests/datasets/test_matbench.py` — unit tests (mock `MatbenchBenchmark`)
5. [ ] Add a usage example to the tutorials or docs

## Files to Create

- `tools/alf_tools/datasets/matbench.py` — dataset implementation
- `tools/tests/datasets/test_matbench.py` — unit tests

## Files to Modify

- `tools/pyproject.toml` — add optional `matbench` extras group
- `tools/alf_tools/datasets/__init__.py` — export `Matbench`, `MatbenchConfig`

## Dependencies

- `matbench` — benchmark harness, fold definitions, and data loading via `MatbenchBenchmark`
- `pymatgen` — crystal structure handling (for structure tasks only)

Both added under `[project.optional-dependencies]` as `alf_tools[matbench]`.
`matminer` is **not** required — `MatbenchBenchmark` handles all data loading.

## Testing Strategy

- Mock `MatbenchBenchmark` to return a synthetic task with known train/test data
- Test fold mode (`fold_number=0`):
  - Assert predefined split sizes
  - Assert no data leakage between train/test splits
  - Assert `fold_id=0` in all candidate features
- Test merged mode (`fold_number=None`):
  - Assert all 5 folds' data is present
  - Assert ratio-based splitting applies
  - Assert `fold_id` in `{0,1,2,3,4}` across candidates
- Test composition task → TABULAR modality auto-inferred
- Test structure task → STRUCTURE modality auto-inferred
- Test structure candidates have JSON string data (not raw Structure object)
- Test classification labels cast to float
- Test invalid `task_name` raises `ValueError`
- Test invalid `fold_number` (e.g. 5) raises `ValueError`

## Notes

- Matbench's predefined folds use `seed=18012019` internally; this is fixed by Matbench
  and cannot be overridden — document clearly in the docstring
- `fold_number=None` (merged mode) loses Matbench's benchmark integrity guarantees;
  document with a warning in the docstring
- For active learning in fold mode, `candidate_pool` comes from the remaining train rows
  not allocated to the initial labelled set (same as FLIP)
- Classification tasks (boolean labels) cast to float; surrogate treats them as regression —
  document in docstring
- pymatgen lazy-imported inside `matbench.py` to avoid import errors for users who have
  not installed the `alf_tools[matbench]` extras
