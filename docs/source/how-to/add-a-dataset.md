# Add a dataset

Make a new candidate pool available to ALF by subclassing `BaseDataset`. The dataset supplies the
candidate pool, its labels, train/validation splits, and a query interface, so it can serve both
the search space and (in offline mode) the `Oracle`.

**Base class:** `BaseDataset`. **Key methods:** `load_dataset`, `query`.

## Steps

1. Subclass `BaseDataset` under `tools/alf_tools/datasets/`.
2. Implement `load_dataset` (build the candidate pool and labels) and `query`.
3. Pick a compatible `modality` (e.g. protein `sequence` or SMILES) so models line up.
4. Add tests and run the build / pre-commit checks.

## Reference notebook

- [Datasets](https://github.com/instadeepai/alf/blob/main/tutorials/extending_base_classes/datasets.ipynb): implement a custom `BaseDataset`.
