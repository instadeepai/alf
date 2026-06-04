# ALF Experiment Scripts

Hydra-based CLI for running ALF experiments. `run.py` is the single entry point; Hydra composes a full config from the YAML hierarchy in `conf/` and passes it to `factory.py`, which wires the ALF objects together.

## Quick start

```bash
# Run from the repo root
uv run python scripts/run.py \
    dataset=gfp \
    model=mlp_default \
    optimizer=ucb \
    oracle=dataset \
    task=design \
    experiment=development
```

Outputs are written to `outputs/<phase>/<name>/<timestamp>/`.

## Config groups

| Group        | Available values                                  | Default       |
|--------------|---------------------------------------------------|---------------|
| `dataset`    | `gfp`, `flip`, `proteingym`                       | `gfp`         |
| `model`      | `gp_rbf`, `gp_matern`, `mlp_default`, `cnn_default`, `ensemble_mlp` | `gp_rbf` |
| `optimizer`  | `ucb`, `ei`, `thompson`, `greedy`                 | `ucb`         |
| `oracle`     | `dataset`, `model`                                | `dataset`     |
| `task`       | `design`, `supervised`, `zeroshot`                | `design`      |
| `experiment` | `development`, `model-optimisation`               | `development` |

Override a group on the command line: `model=cnn_default`.

Override individual keys: `model.train.num_epochs=100`.

## Acquisition functions

The `optimizer.acquisition_fn.name` field selects the acquisition function. Registered names:

| Name         | Class                | Extra params   |
|--------------|----------------------|----------------|
| `ucb`        | `UCB`                | `alpha` (float) |
| `ei`         | `ExpectedImprovement`| —              |
| `greedy`     | `Greedy`             | —              |
| `thompson`   | `ThompsonSampling`   | —              |
| `core_set`   | `CoreSet`            | —              |

## Oracle modes

- `oracle=dataset` — uses the dataset itself as the oracle (offline).
- `oracle=model` — uses an external scorer. You must provide the scorer class on the CLI:

  ```bash
  uv run python scripts/run.py oracle=model \
      '+oracle.scorer._target_=alf_tools.models.esmfold.ESMFoldModel'
  ```

## Adding a new model

1. Add a YAML file in `conf/model/` following the pattern of `mlp_default.yaml`.
2. Register the builder in `factory.py`:
   - For a new model type, add a `_build_<type>` function and register it in `build_model`'s dispatch dict.
   - For a new ensemble member type, also add it to `_build_ensemble`'s `member_dispatch`.

## Adding a new acquisition function

1. Implement the class in `alf_tools/optimizer/acquisition_functions/`.
2. Add it to `_ACQ_FN_REGISTRY` in `factory.py`.
3. Add a YAML file in `conf/optimizer/` with `acquisition_fn.name: <your_name>`.

## Running tests

```bash
uv run pytest scripts/tests/
```
