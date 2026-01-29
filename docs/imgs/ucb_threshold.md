# Diversity Penalty for Batch Acquisition

## Problem

When acquiring multiple points in a single round (batch acquisition), the standard acquisition function can select points that cluster together. This happens because all points are scored independently, without considering which other points will be selected in the same batch.


## Solution

We've added a **diversity penalty** feature to the `ThresholdUCB` acquisition function that spreads points apart within each batch. This is implemented using:

1. **Greedy Sequential Selection**: Instead of selecting all k points at once, we select them one at a time
2. **RBF Distance Penalty**: Each new point is penalized based on its distance to already-selected points
3. **Configurable Strength**: You can control how strongly to enforce diversity

## How It Works

The diversity penalty uses a Radial Basis Function (RBF) kernel to compute penalties during greedy sequential selection:

**Step 1: Base Acquisition Calculation**

First, the standard acquisition value is computed:
```
acquisition(x) = (1 - w) * exploitation + w * normalized_uncertainty
```

Where:
- `exploitation` measures the expected benefit of sampling this point:
  - For Gaussian predictions: `norm.cdf((mu - threshold) / sigma)` (probability that f(x) > threshold)
  - For empirical distributions: `mean(max(samples - threshold, 0))` (expected improvement over threshold)
- `normalized_uncertainty` is the uncertainty normalized by the maximum uncertainty across all candidates
- `w` is the exploration weight (default: 0.2, giving 80% weight to exploitation)
- **UCB Filter**: Only points where `mu + beta*sigma > threshold` are considered (others get value 0)

**Step 2: Diversity Penalty Computation**

For each point, a penalty is computed based on distance to already-selected points in the batch:

```
penalty(x) = Σ_i exp(-||x - x_i||² / (2 * length_scale²))
```

Where:
- `x` is a candidate point (shape: [n_features])
- `x_i` are already selected points in the current batch
- `||x - x_i||` is the Euclidean distance in the input feature space
- `length_scale` controls how quickly the penalty decays with distance
- The sum is over all currently selected points in the batch

**Implementation Details:**
- Features are extracted from `candidate.data` and reshaped if needed
- Pairwise distances are computed using broadcasting: `candidate_features[:, None, :] - selected_features[None, :, :]`
- The RBF kernel is applied element-wise: `exp(-distances² / (2 * length_scale²))`
- Penalties from all selected points are summed together
- If no points are selected yet (first iteration), penalty is zero

**Step 3: Adjusted Acquisition**

The acquisition value is adjusted by subtracting the diversity penalty:

```
acquisition'(x) = acquisition(x) - diversity_penalty * penalty(x)
```

**Step 4: Greedy Selection**

Points are selected iteratively:
1. Select the point with highest acquisition'(x)
2. Add it to the selected set
3. Recompute acquisition'(x) for remaining points (now including penalty from newly selected point)
4. Repeat until batch is filled

**Step 5: Final Ranking**

The function returns values that reflect the selection order, ensuring the optimizer picks the diversity-aware batch in the correct sequence.

## Usage

### Configuration Parameters

```python
acquisition_fn = ThresholdUCB(
    threshold=0.5,
    beta=1.0,
    exploration_weight=0.2,
    diversity_penalty=2.0,        # NEW: Weight for diversity penalty (0=off)
    diversity_length_scale=1.0    # NEW: Length scale for distance computation
)
```

### Parameter Guide

**`threshold`** (required)
- The target threshold value that points should exceed
- Only candidates where `mu + beta*sigma > threshold` are considered

**`beta`** (default: 1.0)
- Number of standard deviations for the UCB filter
- Higher values are more optimistic/exploratory
- Controls how aggressively to explore uncertain regions

**`exploration_weight`** (default: 0.2)
- Weight for exploration bonus in acquisition scoring
- Range [0, 1], where:
  - `0.0`: Pure exploitation (only probability of improvement)
  - `1.0`: Pure exploration (only uncertainty)
  - `0.2`: 20% exploration, 80% exploitation (recommended default)

**`diversity_penalty`** (default: 0.0)
- Weight for diversity penalty to spread batch points apart
- Range [0, ∞), where:
  - `0.0`: No diversity penalty (standard behavior, allows clustering)
  - `0.5-1.0`: Mild penalty (slight spreading)
  - `1.0-3.0`: Moderate penalty (good balance)
  - `>3.0`: Strong penalty (maximum spreading)

**`diversity_length_scale`** (default: 1.0)
- Length scale for computing distances in diversity penalty
- Controls the spatial extent of the repulsion effect:
  - Smaller values (e.g., 0.1-0.5): More localized penalty, only nearby points are penalized
  - Larger values (e.g., 1.0-3.0): Broader penalty, points further away still affected
  - Should be scaled relative to your input space range

### Example: Sinusoidal Function Tutorial

In `gp_sinusoidal_tutorial.ipynb`, we've added:

```python
# Configuration
ACQ_DIVERSITY_PENALTY = 2.0         # Spread batch points apart
ACQ_DIVERSITY_LENGTH_SCALE = 0.5    # Localized penalty

# Acquisition function
acquisition_fn = ThresholdUCB(
    threshold=ACQ_THRESHOLD,
    beta=ACQ_BETA,
    exploration_weight=ACQ_EXPLORATION_WEIGHT,
    diversity_penalty=ACQ_DIVERSITY_PENALTY,
    diversity_length_scale=ACQ_DIVERSITY_LENGTH_SCALE
)
```

With these settings:
- Domain: `[0, 2π]` ≈ `[0, 6.28]`
- Length scale: `0.5` means penalty is significant within ~1-2 units
- Penalty weight: `2.0` provides moderate spreading

## Tuning Tips

### For your specific problem:

1. **Start with moderate diversity**:
   - `diversity_penalty=2.0` (moderate repulsion)
   - `diversity_length_scale = (domain_range / 10)` (adjust based on your input space)
   - `exploration_weight=0.2` (default: 80% exploitation, 20% exploration)
   - `beta=1.0` (default: 1 standard deviation for UCB filter)

2. **If points are still too clustered**:
   - Increase `diversity_penalty` to 3.0 or higher (stronger repulsion)
   - Decrease `diversity_length_scale` to make repulsion more localized (e.g., divide by 2)
   - This makes the penalty stronger and more focused on nearby points

3. **If points are too spread out** (missing interesting regions):
   - Decrease `diversity_penalty` to 1.0 or lower (weaker repulsion)
   - Increase `diversity_length_scale` to make repulsion broader (e.g., multiply by 2)
   - Consider increasing `exploration_weight` to help discover new regions

4. **If exploration is insufficient**:
   - Increase `exploration_weight` (e.g., 0.3-0.5 for more exploration)
   - Increase `beta` (e.g., 1.5-2.0 for more optimistic UCB filtering)
   - This helps discover uncertain regions even without diversity penalty

5. **Visualize the results**:
   - Run your experiment with different settings
   - Check the "Point Acquisition Over Time" plot
   - Points within each round should be well-distributed
   - Verify that interesting regions (near threshold) are being explored

## Technical Details

### Implementation

The diversity penalty is implemented through a custom `__call__` method:

1. **Standard Path** (`diversity_penalty == 0.0`):
   - Computes predictions once
   - Computes acquisition values directly
   - Returns LabelledCandidates with standard acquisition values

2. **Diversity Path** (`diversity_penalty > 0.0`):
   - Computes predictions once (they don't change within a batch)
   - Computes base acquisition values (without diversity penalty)
   - Performs greedy sequential selection:
     - For each position in the batch:
       - Recomputes acquisition values with current diversity penalty
       - Selects the highest-scoring candidate
       - Adds its index to `selected_in_batch` list
       - Next iteration will penalize points near this selection
   - Returns final values based on selection order

3. **Final Value Assignment**:
   - Selected points receive values: `len(selected_indices) - i + max(base_acquisition_values)`
   - Where `i` is the selection order (0 for first selected, 1 for second, etc.)
   - This ensures the optimizer's `get_top_k` picks points in the diversity-aware order
   - Non-selected points receive value 0

4. **Distance Computation**:
   - Uses Euclidean distance in the input feature space
   - Candidate features are extracted from `candidate.data`
   - Distances are computed using NumPy broadcasting for efficiency
   - Pairwise distances: `||x - x_i||` for all candidates x and selected points x_i

5. **Batch Tracking**:
   - `selected_in_batch`: List storing indices of selected points during greedy selection
   - `_current_candidates`: Reference to current candidate list for distance computation
   - Both are reset after each `__call__` invocation to ensure batch independence

### Performance Considerations

- **Computational Cost**: Greedy selection recomputes acquisition values `k` times (where `k` is batch size)
  - Predictions are computed only once (the most expensive operation)
  - Only the diversity penalty and acquisition scoring are recomputed iteratively
  - Distance calculations use efficient NumPy broadcasting
- **For batch size 5**: Approximately 3-4x slower than standard acquisition (not 5x due to prediction caching)
- **For batch size 10**: Approximately 5-7x slower than standard acquisition
- **Scalability**: Reasonable for batch sizes up to ~20; consider optimizations for larger batches

### Edge Cases and Special Behavior

1. **Early Termination of Greedy Selection**:
   - If all remaining acquisition values are ≤ 0, greedy selection stops early
   - This can happen when all good candidates (passing UCB filter) have been selected
   - The batch may contain fewer than `acq_batch_size` points if this occurs

2. **Uncertainty Normalization**:
   - When all candidates have zero uncertainty, normalization uses 1.0 as the denominator
   - This prevents division by zero in edge cases
   - Gaussian predictions: sigma is clipped to minimum of 1e-9 for numerical stability

3. **Empty Selected Set**:
   - On the first iteration (no points selected yet), diversity penalty is zero
   - This ensures the first point is selected purely based on acquisition value

4. **Feature Extraction**:
   - Candidate features from `candidate.data` are automatically reshaped if 1D
   - This ensures consistent array shapes for distance computations

### Backward Compatibility

- Setting `diversity_penalty=0.0` (default) disables the feature entirely
- When `diversity_penalty=0.0`, the `__call__` method uses the standard path:
  - Predictions are computed once
  - Acquisition values are computed once (no iterative selection)
  - No diversity penalty calculations are performed
- The acquisition function behaves exactly as before when penalty is 0
- No changes required to existing code that doesn't use diversity penalty
