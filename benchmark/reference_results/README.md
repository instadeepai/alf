# Reference results — `alf-protein-v1`

Committed baseline results for the frozen `alf-protein-v1` suite. These are the
numbers a fresh clone should reproduce on CPU.

## Files

- [`LEADERBOARD.md`](LEADERBOARD.md) — final-round ranking (mean ± 95% bootstrap CI).
- `aggregate.csv` — the full mean ± CI table per `(problem, method, round, metric)`.

## How these were produced

```bash
bash benchmark/reproduce.sh
```

CPU only, offline (bundled GFP dataset), `deterministic: true`, three seeds.
Re-running regenerates these files; the **scientific** metrics (regret, best-found,
recall, calibration, spearman, MSE) reproduce bit-for-bit. The wall-clock timing
rows in `aggregate.csv` (`ask_time` / `tell_time` / `oracle_time`) naturally vary
from machine to machine — compare on the scientific metrics, not timing. The raw
per-replication outputs are written to `runs/alf_protein_v1/` (git-ignored,
regenerable) rather than committed, since their manifests carry machine-specific
provenance (platform, timestamps, git SHA).

## Interpreting this baseline

- On final `optimizer/regret`, **`gp_greedy` ranks above `cnn_greedy`** (mean
  regret ≈ −0.021 vs ≈ +0.025): with a small initial train split the GP's
  uncertainty-aware fit guides acquisition to better designs than the CNN here.
- The 95% bootstrap CIs **overlap** at three seeds, so this ordering is suggestive,
  not statistically significant — `alf-bench`'s paired test would not call it. This
  is the point of committing CIs rather than bare means. Add seeds (and/or the
  heavier FLIP/ProteinGym tier in the dataset cards) to sharpen the comparison.
- `optimizer/regret` is *simple* regret against the accumulated training set; the
  small `train_ratio=0.1` keeps the pool optimum out of the initial split so
  acquisition actually drives the result. The GP additionally emits calibration
  metrics (`ece`, `width_0.95`, `coverage_0.95`) that the CNN does not.
