# ALF Examples

Runnable, self-contained scripts showing how to run each ALF experiment type with
the plain Python API. No config system, no extra framework — just edit the flags (or
the code) and run.

Run them from the repo root, which already provides `alf-core` and `alf-tools`:

```bash
uv run python examples/offline_design.py --help
```

`_common.py` holds the shared wiring (argument parsing, dataset/model/optimizer
builders). Each script reads top-to-bottom and can be copied out as a starting point.

## Scripts

| Script | Task | Oracle | Search | Runs out-of-the-box? |
|--------|------|--------|--------|----------------------|
| [`offline_design.py`](offline_design.py) | Multi-round active learning | dataset (offline) | fixed dataset pool | ✅ |
| [`online_design.py`](online_design.py) | Multi-round active learning | **synthetic** model (demo) | generated single mutants | ✅ |
| [`supervised.py`](supervised.py) | Train once, evaluate | — | — | ✅ |
| [`zeroshot.py`](zeroshot.py) | Pre-trained scoring, no training | ESM-2 (PLL) | — | ⚠️ needs `esm2` extra + model download |

## Quick start

```bash
# Offline design (the flagship example): GFP + CNN, greedy acquisition
uv run python examples/offline_design.py --dataset gfp --model cnn \
    --num-rounds 5 --batch-size 50 --seed 42

# Supervised baseline
uv run python examples/supervised.py --dataset gfp --model cnn --epochs 20

# Online design (synthetic oracle, generative search)
uv run python examples/online_design.py --num-rounds 3 --batch-size 5

# Zero-shot with ESM-2 (install the extra first)
uv sync --extra esm2
uv run python examples/zeroshot.py --dataset gfp
```

Each script writes `metrics.csv` (and per-round prediction CSVs) under
`--output-dir`, which defaults to `examples/outputs/<script>/`.

## Flags

| Flag | Scripts | Default | Notes |
|------|---------|---------|-------|
| `--dataset {gfp,flip}` | offline, supervised, zeroshot | `gfp` | GFP needs no token; FLIP downloads from GitHub (`gb1`/`one_vs_rest`). |
| `--model {cnn,gp}` | offline, supervised | `cnn` | MLP is omitted — it takes tabular/embedding inputs, not raw sequences. |
| `--acquisition {greedy,ucb,ei}` | offline | `greedy` | `ucb`/`ei` need uncertainty → require `--model gp`. |
| `--num-rounds` / `--batch-size` | offline, online | `5` / `50` | Keep their product below the candidate-pool size (see below). |
| `--epochs` | offline, online, supervised | `20` | CNN epochs / GP iterations. |
| `--seed` | all | `42` | — |
| `--output-dir` | all | `examples/outputs/<script>/` | Created if missing. |
| `--model-id` | zeroshot | `facebook/esm2_t6_8M_UR50D` | Any ESM-2 HuggingFace checkpoint. |

## Notes & gotchas

- **Acquisition budget.** Offline design draws from a finite, shrinking pool. The
  scripts check up front that `num_rounds × batch_size` fits and exit with a clear
  message otherwise. GFP loads ~1000 rows, so the post-split pool is a few hundred —
  keep the product small (e.g. `5 × 50`).
- **`online_design.py` is a demonstration.** Its oracle is a toy synthetic scoring
  function (a stand-in for a real assay or simulator like PyRosetta), so the metrics
  are illustrative only. `SingleMutantSearch` enumerates every single mutant of the
  current best sequence (~length × 19 candidates per round), so rounds are heavier
  than offline; results also vary with `--seed`.
- **`zeroshot.py` needs the ESM-2 extra** (`transformers`) and downloads a model
  checkpoint on first run. The default 8M model is small; larger checkpoints are
  slower and need more memory.
- **ProteinGym** is available in the API but requires an `HF_TOKEN`, so it is not
  exposed via `--dataset` here. Construct `ProteinGymConfig` directly if you need it.
- **Acquisition × model compatibility.** `ucb`/`ei` require a model that predicts
  uncertainty (use `--model gp`); pairing them with `--model cnn` exits early with an
  explanation rather than crashing deep in the acquisition function.
