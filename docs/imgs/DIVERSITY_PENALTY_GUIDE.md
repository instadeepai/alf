# Diversity Penalty for Batch Acquisition

## Problem

When acquiring multiple points in a single round (batch acquisition), the standard acquisition function can select points that cluster together. This happens because all points are scored independently, without considering which other points will be selected in the same batch.

In your visualization, you noticed that points within each round are sampled very close to each other, leading to inefficient exploration of the search space.

## Solution

We've added a **diversity penalty** feature to the `ThresholdUCB` acquisition function that spreads points apart within each batch. This is implemented using:

1. **Greedy Sequential Selection**: Instead of selecting all k points at once, we select them one at a time
2. **RBF Distance Penalty**: Each new point is penalized based on its distance to already-selected points
3. **Configurable Strength**: You can control how strongly to enforce diversity

## How It Works

The diversity penalty uses a Radial Basis Function (RBF) kernel to compute penalties:

```
penalty(x) = Σ exp(-||x - x_i||² / (2 * length_scale²))
```

Where:
- `x` is a candidate point
- `x_i` are already selected points in the current batch
- `length_scale` controls how quickly the penalty decays with distance

The acquisition value is then adjusted:

```
acquisition'(x) = acquisition(x) - diversity_penalty * penalty(x)
```

## Usage

### Configuration Parameters

```python
acquisition_fn = ThresholdUCB(
    threshold=0.5,
    beta=0.8,
    exploration_weight=0.8,
    diversity_penalty=2.0,        # NEW: Weight for diversity penalty (0=off)
    diversity_length_scale=0.5    # NEW: Length scale for distance computation
)
```

### Parameter Guide

**`diversity_penalty`** (default: 0.0)
- `0.0`: No diversity penalty (standard behavior, allows clustering)
- `0.5-1.0`: Mild penalty (slight spreading)
- `1.0-3.0`: Moderate penalty (good balance)
- `>3.0`: Strong penalty (maximum spreading)

**`diversity_length_scale`** (default: 1.0)
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

1. **Start with the defaults**:
   - `diversity_penalty=2.0`
   - `diversity_length_scale = (domain_range / 10)`

2. **If points are still too clustered**:
   - Increase `diversity_penalty` to 3.0 or higher
   - Decrease `diversity_length_scale` for more localized repulsion

3. **If points are too spread out** (missing interesting regions):
   - Decrease `diversity_penalty` to 1.0 or lower
   - Increase `diversity_length_scale` for broader repulsion

4. **Visualize the results**:
   - Run your experiment with different settings
   - Check the "Point Acquisition Over Time" plot
   - Points within each round should be well-distributed

## Technical Details

### Implementation

The diversity penalty is implemented by:

1. **Overriding `__call__`**: The acquisition function's `__call__` method now performs greedy sequential selection when `diversity_penalty > 0`

2. **Iterative Selection**: For each point in the batch:
   - Compute acquisition values for all candidates
   - Apply penalty based on distance to already-selected points
   - Select the best candidate
   - Add it to the selected list
   - Repeat for next point

3. **Returning Ordered Values**: The final acquisition values reflect the greedy selection order, ensuring the optimizer's `get_top_k` selects the diversity-aware batch

### Performance Considerations

- **Computational Cost**: Greedy selection requires computing acquisition values `k` times (where `k` is batch size), instead of once
- **For batch size 5**: ~5x slower acquisition computation
- **Not recommended** for very large batch sizes (>20) without optimization

### Backward Compatibility

- Setting `diversity_penalty=0.0` (default) disables the feature entirely
- The acquisition function behaves exactly as before when penalty is 0
- No changes required to existing code

## Related Files

- **Implementation**: `tools/alf_tools/optimizer/acquisition_functions/threshold_ucb.py`
- **Tutorial**: `tutorials/gp_sinusoidal_tutorial.ipynb`
- **Configuration**: Cell 7 (configuration parameters) and Cell 23 (optimizer setup)

## Next Steps

1. **Run the tutorial** with the new diversity penalty enabled
2. **Compare visualizations** before/after enabling diversity
3. **Experiment** with different penalty weights and length scales
4. **Adapt** the approach to other acquisition functions if needed

Enjoy better space exploration with diversity-aware batch acquisition!
