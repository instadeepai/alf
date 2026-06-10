# alf-benchmark

A benchmarking layer over `alf-core` and `alf-tools`. It turns ALF from "a
framework that runs one active-learning experiment" into "a benchmark suite you
can run, extend, and compare", with the headline axis being **sequential /
active-learning performance** (best-found, regret, recall, calibration **vs
acquisition round**) — the gap that static benchmarks like ProteinGym and FLIP
leave open.

> **Status: Phase 3 (YAML + CLI).** Phases 1–3 deliver the registry,
> config-as-code models, a resumable sweep runner (reusing the existing ALF tasks
> unchanged), analysis (mean ± bootstrap-CI aggregation, paired significance tests,
> a leaderboard, active-learning curve plots), and a declarative YAML + `alf-bench`
> CLI front end. A versioned suite with reference results (Phase 4) follows.

## Concepts

| Object | Role |
|--------|------|
| `Registry` | Resolves alf components by name from Python **entry points**, and builds them from config dicts. Providers register components without importing `alf-benchmark`. |
| `ProblemConfig` / `BenchmarkProblem` | *What* is optimised: a configured dataset, replication seeds, and a primary metric. Frozen and versioned. |
| `MethodConfig` / `BenchmarkMethod` | *How* it is optimised: surrogate, acquisition, search, oracle, and task. Frozen and versioned. |
| `BenchmarkSuite` | A named, versioned collection of problems. |
| `BenchmarkRunner` | Expands `(problem × method × seed)` into replications. Each replication is one unchanged `task.run()`. |
| `Manifest` | Per-replication reproducibility record (resolved configs, versions, git SHA, status). |
| `BenchmarkResults` | Loads replications into a tidy long-form table; computes mean ± bootstrap CI, paired significance tests, and a leaderboard. |

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

## Analysing results

```python
from alf_benchmark import BenchmarkResults
from alf_benchmark.plotting import plot_curves, plot_leaderboard

results = BenchmarkResults.from_dir("runs")
results.aggregate()        # mean ± bootstrap CI per (problem, method, round, metric)
results.leaderboard()      # final-round ranking on each problem's primary metric
results.significance()     # paired t-tests between methods

# Active-learning curves with CI bands (best-found, regret, recall, calibration vs round)
plot_curves(results, metric="derived/best_found").savefig("best_found.png")
plot_leaderboard(results).savefig("leaderboard.png")
```

`best-found-so-far` is *derived* as `derived/best_found` (running max of the
per-round acquired batch maxima). `optimizer/regret` is already a *simple*
(best-found-so-far) regret and is plotted directly as the regret-vs-round curve.
*Cumulative* regret is intentionally not derived: it needs per-round instantaneous
regret against the fixed pool optimum, which the current schema does not log.
Curves guard missing metrics: regret/recall require a `DatasetSearch` method, and
calibration requires an uncertainty-capable model, so a metric only appears for the
methods that produced it.

## Command line (`alf-bench`)

Run, aggregate, plot, and inspect without writing Python. A run is described by a
YAML file that parses into the same validated config models (see
[`configs/example.yaml`](configs/example.yaml)):

```bash
alf-bench run benchmark/configs/example.yaml   # sweep (problem x method x seed)
alf-bench aggregate runs/example               # print the leaderboard
alf-bench aggregate runs/example --output agg.csv
alf-bench plot runs/example                    # save AL-curve figures
alf-bench list                                 # show registered components
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
