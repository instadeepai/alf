# Add a search function

A search function generates the pool of candidates that an `Acquisition function` then scores.
Subclass `BaseSearch` to control where candidates come from (e.g. enumerating a dataset with
`DatasetSearch`, or sampling from a generative model with `GeneratorSearch`).

**Base class:** `BaseSearch`. **Key method:** `__call__`.

## Steps

1. Subclass `BaseSearch` under `tools/alf_tools/optimizer/search/`.
2. Implement `__call__` to return the candidate pool for the current round.
3. Drop it into an `Optimizer` alongside an acquisition function and run a `Task`.
4. Add tests and run the build / pre-commit checks.

## Reference notebook

- [Search functions](https://github.com/instadeepai/alf/blob/main/tutorials/extending_base_classes/search_functions.ipynb): implement a custom `BaseSearch`.
