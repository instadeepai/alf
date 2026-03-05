# Plan: Multi-Output GP via Linear Model of Coregionalization (LMC)

## Context

The user models multiple correlated performance metrics simultaneously (e.g., Va, Vb, Vc, Vjunction for a circuit), all driven by the same underlying physics. A plain `ModelListGP` (independent GPs per output) throws away this correlation. LMC/ICM models each output as a linear combination of shared latent GPs, enabling transfer of information across outputs:

```
Va(θ)  = w₁₁·f₁(θ) + w₁₂·f₂(θ) + ε₁
Vb(θ)  = w₂₁·f₁(θ) + w₂₂·f₂(θ) + ε₂
...
```

BoTorch's `MultiTaskGP` implements ICM (rank=1, single latent GP) and LMC (rank>1, multiple latent GPs) via its `rank` parameter. This is already a native BoTorch model, so the acquisition adapter path that directly accesses `state.surrogate.model.model` will work without changes.

**Goal:** Add `BoTorchMultiOutputGP` (wrapping `MultiTaskGP`) to ALF, update the data flow to handle multi-output predictions, and add scalarized acquisition support so existing acquisition types continue to work.

---

## Files to Create

### 1. `tools/alf_tools/models/botorch_multi_output_gp.py` (new)

New `BoTorchMultiOutputGP(BaseModel)` class that:

**Constructor:**
```python
def __init__(
    self,
    num_outputs: int,               # Required: number of correlated outputs
    rank: int = 1,                  # LMC rank (1=ICM, 2+ = full LMC)
    num_iterations: int = 100,
    learning_rate: float = 0.1,
    optimizer: str = "scipy",
    max_attempts: int = 5,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
)
```

**`train(train_data, val_data=None)`:**
- Input: `train_data.labels` of shape `(n, num_outputs)`
- Convert to BoTorch MultiTaskGP format: expand X with task index column
  ```python
  # train_X: (n, d) → expand to (n*num_outputs, d+1)
  X_list, Y_list = [], []
  for task_i in range(num_outputs):
      X_list.append(torch.cat([train_X, torch.full((n, 1), task_i)], dim=1))
      Y_list.append(train_Y[:, task_i:task_i+1])
  X_mt = torch.cat(X_list, dim=0)   # (n*num_outputs, d+1)
  Y_mt = torch.cat(Y_list, dim=0)   # (n*num_outputs, 1)
  ```
- Initialize: `MultiTaskGP(train_X=X_mt, train_Y=Y_mt, task_feature=-1, rank=self.rank)`
- Fit with `fit_gpytorch_mll` (same scipy/torch paths as `BoTorchGPModel`)

**`predict(candidate_points)`:**
- Converts candidates → `test_X: (m, d)`
- Gets posterior over all tasks: `self.model.posterior(test_X_expanded)` where expanded X has task index appended for each output
- Actually: use `self.model.posterior(test_X)` after setting task indices appropriately — `MultiTaskGP.posterior` accepts `(m, d)` test points and returns a `(num_outputs, m)` batched posterior by default when no task feature is in test X
- Returns `Predictions(means=means_2d, variances=variances_2d)` where both are shape `(m, num_outputs)`

**`num_outputs` property:** returns `self._num_outputs`

**`botroch_model` property:** returns `self.model` (the `MultiTaskGP` instance)

---

## Files to Modify

### 2. `tools/alf_tools/models/__init__.py`

Add export:
```python
from alf_tools.models.botorch_multi_output_gp import BoTorchMultiOutputGP
__all__ = [..., "BoTorchMultiOutputGP"]
```

### 3. `tools/alf_tools/utils/botorch_utils.py`

Update `predictions_to_posterior` to handle 2D predictions (multi-output):

```python
# After existing single-output path, add multi-output branch:
if mean.dim() == 2:
    # Multi-output: means shape (n, num_outputs)
    # Need batched MVN: batch_shape=(num_outputs,), event_shape=(n,)
    mean_t = mean.T                          # (num_outputs, n)
    variance_clamped = torch.clamp(variance.T, min=1e-6)  # (num_outputs, n)
    covar = torch.diag_embed(variance_clamped)             # (num_outputs, n, n)
    mvn = MultivariateNormal(mean_t, covar)
else:
    # Existing single-output path (unchanged)
    ...
```

### 4. `tools/alf_tools/models/utils/botorch_model_adapter.py`

Update `num_outputs` property to check for `num_outputs` attribute on ALF models:

```python
@property
def num_outputs(self) -> int:
    if self._is_botorch_model:
        return int(self._wrapped_model.num_outputs)
    # Check if ALF model exposes num_outputs (e.g., BoTorchMultiOutputGP)
    if hasattr(self._wrapped_model, "num_outputs"):
        return int(self._wrapped_model.num_outputs)
    return 1  # default single-output
```

Also update `posterior()` for multi-output ALF models: after calling `predictions_to_posterior()`, the batched MVN needs proper reshaping when `needs_reshape=True` and `num_outputs > 1`. Handle the `(batch_size, num_outputs, q, q)` covariance case.

### 5. `tools/alf_tools/optimizer/acquisition_functions/botorch_acquisition.py`

Add optional `weights` parameter and `ScalarizedPosteriorTransform` support:

**Constructor addition:**
```python
weights: Optional[list[float]] = None,  # scalarization weights, len = num_outputs
```

**`_create_acquisition_function` update:**
```python
from botorch.acquisition.objective import ScalarizedPosteriorTransform

posterior_transform = None
if self.weights is not None:
    w = torch.tensor(self.weights, dtype=torch.float32)
    posterior_transform = ScalarizedPosteriorTransform(weights=w)
# Pass posterior_transform= to qEI/qNEI/qUCB
```

**`__call__` update:**
- For multi-output models, `best_f = train_data.labels.max()` still works as a scalar best (global max across outputs, or use scalarized best if `weights` provided)
- When `weights` is not None and model is multi-output, compute `best_f` as scalarized: `(labels @ weights).max()`

**`_optimize_continuous` update:**
- The native BoTorch model path (`state.surrogate.model.model`) handles multi-output correctly since `MultiTaskGP` is a native BoTorch model — no changes needed for gradient-based optimization

---

## Files to Create (Tests)

### 6. `tools/tests/models/test_botorch_multi_output_gp.py` (new)

Test classes:
- `TestBoTorchMultiOutputGPInit` — validates `num_outputs`, `rank`, device setup
- `TestBoTorchMultiOutputGPTrain` — trains on synthetic 2D→3-output data, checks model fitted
- `TestBoTorchMultiOutputGPPredict` — predictions return shape `(m, num_outputs)` means/variances
- `TestBoTorchMultiOutputGPLMCRank` — verify `rank=2` trains a different model from `rank=1`
- `TestBoTorchMultiOutputGPErrors` — 1D labels raises clear error, untrained model raises RuntimeError

### 7. Update `tools/tests/models/utils/test_botorch_model_adapter.py`

Add tests for `num_outputs` returning correct value for multi-output ALF models.

### 8. Update `tools/tests/models/utils/test_botorch_utils.py` (or create if not exists)

Add tests for `predictions_to_posterior` with 2D inputs (multi-output case).

---

## Critical Files (paths for reference)

| Role | Path |
|---|---|
| New model | `tools/alf_tools/models/botorch_multi_output_gp.py` |
| Model exports | `tools/alf_tools/models/__init__.py` |
| Posterior conversion | `tools/alf_tools/utils/botorch_utils.py` |
| Adapter | `tools/alf_tools/models/utils/botorch_model_adapter.py` |
| Acquisition | `tools/alf_tools/optimizer/acquisition_functions/botorch_acquisition.py` |
| Predictions dataclass | `core/alf_core/dataclasses/predictions.py` (no changes needed) |
| LabelledCandidates | `core/alf_core/dataclasses/labelled_candidates.py` (no changes needed) |

## Reuse Notes

- `candidates_to_tensor` from `botorch_utils.py` — reuse as-is for `featurise()`
- `fit_gpytorch_mll` / `fit_gpytorch_mll_torch` — same training loop as `BoTorchGPModel`
- `ExactMarginalLogLikelihood` — same MLL setup
- `get_device` from `torch_utils.py` — reuse for device setup
- `BoTorchGPModel.train()` — mirror the scipy/torch optimizer branching pattern

---

## Verification

```bash
# Unit tests
cd tools && python -m pytest tests/models/test_botorch_multi_output_gp.py -v
cd tools && python -m pytest tests/models/utils/test_botorch_model_adapter.py -v

# Quick smoke test (can be run as a script)
from alf_tools.models.botorch_multi_output_gp import BoTorchMultiOutputGP
from alf_core import LabelledCandidates, Candidate, Predictions
from alf_core.dataclasses.candidate import Modality
import numpy as np

n, d, t = 20, 3, 4   # 20 samples, 3 dims, 4 outputs
X = np.random.rand(n, d).astype(np.float32)
Y = np.random.rand(n, t).astype(np.float32)  # 4 correlated outputs
candidates = [Candidate(data=x, modality=Modality.TABULAR) for x in X]
train_data = LabelledCandidates(candidates=candidates, labels=Y)

model = BoTorchMultiOutputGP(num_outputs=4, rank=2)
model.train(train_data)
preds = model.predict(candidates[:5])
assert preds.means.shape == (5, 4)
assert preds.variances.shape == (5, 4)
print("Multi-output GP works!")
```
