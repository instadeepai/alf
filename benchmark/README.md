# alf-benchmark

A benchmarking layer over `alf-core` and `alf-tools`. It turns ALF from "a
framework that runs one active-learning experiment" into "a benchmark suite you
can run, extend, and compare", with the headline axis being **sequential /
active-learning performance** (best-found, regret, recall, calibration **vs
acquisition round**) — the gap that static benchmarks like ProteinGym and FLIP
leave open.

> **Status: Phase 1 (usability core).** This phase delivers the registry,
> config-as-code models, and a resumable sweep runner that reuses the existing
> ALF tasks unchanged. Aggregation/plots (Phase 2), a YAML + CLI front end
> (Phase 3), and a versioned suite with reference results (Phase 4) follow.

## Concepts

| Object | Role |
|--------|------|
| `Registry` | Resolves alf components by name from Python **entry points**, and builds them from config dicts. Providers register components without importing `alf-benchmark`. |
| `ProblemConfig` / `BenchmarkProblem` | *What* is optimised: a configured dataset, replication seeds, and a primary metric. Frozen and versioned. |
| `MethodConfig` / `BenchmarkMethod` | *How* it is optimised: surrogate, acquisition, search, oracle, and task. Frozen and versioned. |
| `BenchmarkSuite` | A named, versioned collection of problems. |
| `BenchmarkRunner` | Expands `(problem × method × seed)` into replications. Each replication is one unchanged `task.run()`. |
| `Manifest` | Per-replication reproducibility record (resolved configs, versions, git SHA, status). |

One replication writes to `runs/<problem>/<method>/seed=<n>/`:

- `metrics.csv` — one row per round (from `FileStateLogger`), with `round`,
  `problem_id`, `method_id`, and `seed` columns appended by the runner.
- `manifest.json` — the reproducibility record; also used to **skip already
  completed replications** on a resumed run.

## Quick start (config-as-code)

```python
from alf_benchmark import (
    BenchmarkMethod,
    BenchmarkRunner,
    BenchmarkSuite,
    ComponentSpec,
    MethodConfig,
    ProblemConfig,
    TaskConfig,
)

problem = ProblemConfig(
    name="gfp_offline",
    dataset=ComponentSpec(
        name="gfp",
        config={
            "name": "gfp",
            "modality": "sequence",
            "train_ratio": 0.4,
            "validation_frac": 0.2,
            "test_ratio": 0.2,
            "problem_type": "regression",
        },
    ),
    seeds=[0, 1],
    primary_metric="optimizer/regret",
)

method = MethodConfig(
    name="cnn_greedy",
    family="design",
    surrogate=ComponentSpec(name="cnn"),
    acquisition=ComponentSpec(name="greedy"),
    search=ComponentSpec(name="dataset_search"),
    task=TaskConfig(num_acq_rounds=5, acq_batch_size=100),
)

suite = BenchmarkSuite.from_configs("adhoc", "0.1.0", [problem])
runner = BenchmarkRunner()
manifests = runner.run(suite, [BenchmarkMethod(method)], output_dir="runs")
```

## Registering your own component

Expose a class (model, dataset, acquisition or search function) under an
entry-point group in your package's `pyproject.toml` — no `alf-benchmark`
import required:

```toml
[project.entry-points."alf.models"]
my_model = "my_package.models:MyModel"
```

After installing your package, the component is referable by name
(`ComponentSpec(name="my_model", config={...})`). The registry infers the config
classes from the constructor's type annotations, so dataclass/pydantic config
parameters are coerced from sub-dictionaries automatically.

## Current scope and limits (Phase 1)

- Only the **`design`** family (the active-learning loop) is implemented;
  `supervised` and `zero_shot` raise `NotImplementedError` for now.
- The runner is **sequential**; parallel execution is a planned follow-up.
- Full AL curves (regret/recall/calibration vs round) require a `DatasetSearch`
  method, and calibration additionally requires an uncertainty-capable model.
  The `DatasetSearch` recall metric assumes a candidate pool of at least 100.
