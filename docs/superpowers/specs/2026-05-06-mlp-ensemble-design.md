# MLP Model & Generic Ensemble Wrapper — Design Spec

**Date:** 2026-05-06  
**Status:** Approved  

---

## Overview

Add a flexible MLP surrogate model and a generic ensemble wrapper to `alf_tools`. The MLP handles pre-computed vector inputs (molecular fingerprints, protein embeddings, or any tabular features) and supports optional MC dropout for uncertainty quantification at inference. The ensemble wrapper is a generic `BaseModel` that can wrap any `BaseModel` via a factory callable, supporting deep ensembles, MC dropout ensembles, or combinations of both — populating `Predictions.empirical_dist` for use by existing acquisition functions (Thompson Sampling, EI, UCB).

---

## Files

```
alf/tools/alf_tools/models/
  mlp.py       ← MLP(nn.Module), MLPModelConfig, MLPTrainConfig, MLPModel(BaseModel)
  ensemble.py  ← EnsembleWrapperConfig, EnsembleWrapper(BaseModel)

alf/tools/tests/models/
  test_mlp.py
  test_ensemble.py
```

No changes to core. No new dependencies.

---

## Section 1: Architecture

### `MLPModel`

Standalone `BaseModel` for pre-computed vector inputs. Featurisation is a passthrough. Supports optional MC dropout at inference: when `n_mc_passes > 0`, `predict()` runs T stochastic forward passes with the model kept in `train()` mode, returning a full `empirical_dist` alongside aggregated means and variances. When `n_mc_passes == 0`, evaluates normally and returns means only.

### `EnsembleWrapper`

Generic `BaseModel` wrapping N instances of any `BaseModel` via a factory callable `Callable[[int], BaseModel]`. At construction, resolves member seeds and instantiates members. At train time, fits all members independently. At predict time, assembles `empirical_dist` by concatenating each member's contribution (either their `empirical_dist` columns if they produce one, or their `means` as a single column).

### Uncertainty modes

| Mode | Setup | `empirical_dist` shape |
|---|---|---|
| Deep ensemble | N MLPModels, `n_mc_passes=0` | `(N_candidates, N_members)` |
| MC dropout | 1 MLPModel, `n_mc_passes=T` | `(N_candidates, T)` |
| Combined | N MLPModels, `n_mc_passes=T` each | `(N_candidates, N_members × T)` |

`means` and `variances` on the returned `Predictions` are always the row-wise mean and variance of `empirical_dist`, so all existing acquisition functions work without modification.

---

## Section 2: Configuration Dataclasses

### `MLPModelConfig`

```python
@dataclass
class MLPModelConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    activation: Literal["relu", "gelu", "silu"] = "relu"
    norm: Literal["none", "batch", "layer"] = "none"
    dropout: float = 0.0
    n_mc_passes: int = 0
    model_seed: int = 0
    dropout_seed: int | None = None
```

- `hidden_dims`: depth and width of the network; variable length determines number of hidden layers
- `activation`: applied after each hidden layer's norm
- `norm`: applied before activation (pre-norm); `"none"` skips normalisation entirely
- `dropout`: applied after activation in each hidden layer; must be `> 0` if `n_mc_passes > 0`
- `n_mc_passes`: number of stochastic forward passes at inference; `0` disables MC dropout
- `model_seed`: global seed for weight initialisation, training data shuffling, and (fallback) dropout
- `dropout_seed`: if set, overrides `model_seed` exclusively for the MC dropout pass generator

Validation: if `n_mc_passes > 0` then `dropout > 0`.

### `MLPTrainConfig`

```python
@dataclass
class MLPTrainConfig:
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 50
    optimizer: Literal["adam", "adamw"] = "adam"
    weight_decay: float = 0.0
```

### `EnsembleWrapperConfig`

```python
@dataclass
class EnsembleWrapperConfig:
    base_seed: int | None = None
    member_seeds: list[int] | None = None
    n_members: int | None = None
```

- Exactly one of `base_seed` or `member_seeds` must be set
- If `base_seed` is set, `n_members` must also be set; member seeds derived as `[base_seed + i for i in range(n_members)]`
- If `member_seeds` is set, `n_members` is ignored; `len(member_seeds)` determines the number of members

Validation enforced via Pydantic `model_validator`.

---

## Section 3: `MLP(nn.Module)`

Pure PyTorch module with no ALF dependencies. Architecture:

```
input (input_dim,)
→ [Linear(in, hidden) → Norm → Activation → Dropout] × len(hidden_dims)
→ Linear(hidden_dims[-1], 1)
```

- Pre-norm (norm before activation) for stability with LayerNorm
- `nn.BatchNorm1d` or `nn.LayerNorm` per hidden layer; skipped when `norm="none"`
- Output layer: bare `Linear` with no norm or activation — intentionally minimal; a future variance head extends this layer without touching the hidden block
- Weight initialisation seeded via `torch.manual_seed(model_seed)` immediately before parameter construction

---

## Section 4: `MLPModel(BaseModel)`

### `featurise(inputs: list[Candidate]) -> Tensor`

Converts candidates to a `(N, D)` float32 tensor. Accepts `Modality.TABULAR` and `Modality.EMBEDDING`. Raises `ValueError` for any other modality. No normalisation applied.

### `train(train_data, val_data)`

1. Seeds `np.random` and `torch` with `model_seed`
2. Constructs `DataLoader` with `batch_size`; shuffles training data
3. Runs `num_epochs` of Adam or AdamW with MSE loss
4. Records one `SurrogateEpochMetrics` per epoch: train loss, val loss, Spearman ρ on val set
5. Sets model to `eval()` after training (predict with `n_mc_passes > 0` switches back to `train()` mode internally; predict with `n_mc_passes == 0` explicitly sets `eval()` mode before the forward pass)

### `predict(candidates) -> Predictions`

**`n_mc_passes == 0` (standard eval):**
- `model.eval()`, single forward pass
- Returns `Predictions(means=shape(N,), variances=None, empirical_dist=None)`

**`n_mc_passes > 0` (MC dropout):**
- Model stays in `train()` mode (activates dropout)
- Creates `torch.Generator` seeded with `dropout_seed or model_seed`
- Runs T forward passes; each pass uses the generator for dropout mask sampling
- `empirical_dist` shape `(N, T)`
- `means` = row-wise mean of `empirical_dist`
- `variances` = row-wise variance of `empirical_dist`
- Returns `Predictions(means, variances, empirical_dist)`

### `sample()`

Raises `NotImplementedError` — not meaningful for a fixed-input regression model.

### `get_epoch_metrics()` / `get_training_summary_metrics()`

Mirrors `CNNModel` implementation exactly.

---

## Section 5: `EnsembleWrapper(BaseModel)`

### Construction

```python
EnsembleWrapper(
    model_factory: Callable[[int], BaseModel],
    config: EnsembleWrapperConfig,
)
```

Resolves seeds at init time; calls `model_factory(seed)` for each resolved seed; stores members as `list[BaseModel]`.

### `featurise(inputs)`

Delegates to `self.members[0].featurise(inputs)`. This assumes all members produced by `model_factory` have identical featurisation — a convention enforced by the caller, not the type.

### `train(train_data, val_data)`

Trains each member independently and sequentially. Each member manages its own seeding internally — the wrapper does not re-seed. Collects and concatenates `SurrogateEpochMetrics` from all members; per-member metrics are tagged with member index (e.g. `"member_0/val_spearman"`).

### `predict(candidates) -> Predictions`

For each member:
- Calls `member.predict(candidates)` → `p_i`
- If `p_i.empirical_dist is not None`: appends all T columns
- Else: appends `p_i.means[:, None]` as a single column

Stacks all columns horizontally into final `empirical_dist` of shape `(N, total_columns)`.

```
means     = empirical_dist.mean(axis=1)
variances = empirical_dist.var(axis=1)
```

Returns `Predictions(means, variances, empirical_dist)`.

### `sample()`

Raises `NotImplementedError`.

### `get_epoch_metrics()` / `get_training_summary_metrics()`

Aggregates across members; training summary includes per-member scalar keys.

---

## Section 6: Testing

### `TestMLPModel`

- Network builds correctly for varied `hidden_dims`, activations, and norm modes
- `featurise` handles `TABULAR` and `EMBEDDING` candidates; raises `ValueError` for unsupported modalities
- `predict` output shape: means `(N,)`, `empirical_dist` `(N, T)` when `n_mc_passes > 0`
- MC dropout predictions vary across passes (stochastic) but are reproducible given same `dropout_seed`
- Same `model_seed` → same weights → same eval predictions (weight init reproducibility)
- Different `dropout_seed` → different pass sequence, same weights (seed independence)
- `dropout_seed=None` falls back to `model_seed` for dropout generator
- `n_mc_passes > 0` with `dropout == 0` raises `ValueError` at config validation
- Per-epoch metrics recorded correctly (train loss, val loss, Spearman ρ)

### `TestEnsembleWrapper`

- `member_seeds` produces correct number of members with correct seeds
- `base_seed + n_members` produces correct number of members with derived seeds
- Mutual exclusivity: both or neither `base_seed`/`member_seeds` set raises `ValueError`
- `base_seed` without `n_members` raises `ValueError`
- Deep ensemble mode: `empirical_dist` shape `(N, n_members)`
- MC dropout mode (1 member, `n_mc_passes=T`): `empirical_dist` shape `(N, T)`
- Combined mode (N members, `n_mc_passes=T`): `empirical_dist` shape `(N, N×T)`
- `means` and `variances` derived correctly from `empirical_dist`
- Different `member_seeds` → different member weights → different per-member predictions
- Per-member metric keys are tagged correctly (e.g. `"member_0/val_spearman"`, `"member_1/val_spearman"`)

---

## Constraints and Non-Goals

- No input normalisation — caller is responsible
- No chemistry dependency — fingerprints arrive pre-computed
- No Hydra — configuration is Pydantic dataclasses only, consistent with existing models
- No variance head in this iteration — output layer is designed to support it later (bare `Linear` with no activation)
- `EnsembleWrapper` does not implement `sample()` — not meaningful for this use case
- `EnsembleWrapper` is generic but not placed in `alf_core` — it depends on the `BaseModel` contract and numpy/torch patterns that belong in `alf_tools`
