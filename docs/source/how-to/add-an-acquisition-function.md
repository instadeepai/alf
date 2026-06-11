# Add an acquisition function

An acquisition function scores candidates by their *expected value to the search*, trading off
exploitation against exploration. Subclass `AcquisitionFunction` to add a new scoring rule
(alongside the built-in Greedy, UCB, Expected Improvement, Thompson Sampling, and CoreSet).

**Base class:** `AcquisitionFunction`. **Key method:** `__call__`.

It scores the pool produced by a `Search function`; the `Optimizer` bundles the two and
selects the top batch. See [Core Concepts](../explanation/core-concepts.md) for how one round flows.

## Steps

1. Subclass `AcquisitionFunction` under `tools/alf_tools/optimizer/acquisition_functions/`.
2. Implement `__call__` to score candidates given the surrogate's predictions and uncertainty.
3. Drop it into an `Optimizer` and run a `Task`.
4. Add tests and run the build / pre-commit checks.

## Reference notebook

- [Acquisition functions](https://github.com/instadeepai/alf/blob/main/tutorials/extending_base_classes/acquisition_functions.ipynb): implement a custom `AcquisitionFunction`.
