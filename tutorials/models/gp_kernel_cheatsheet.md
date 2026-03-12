# GP Kernel Selection Cheatsheet

A practitioner's guide: when to use which kernel, what they assume, and how to handle high-dimensional inputs.

---

## Quick Decision Tree

```
Is the function periodic?
├── Yes → Periodic kernel  (+RBF or +Linear for trend)
└── No →
    Is it rough / spiky / non-differentiable?
    ├── Yes → Matern 1/2
    └── No →
        Does it have a global linear or polynomial trend?
        ├── Yes → Linear or Polynomial (+ stationary kernel)
        └── No →
            Does it vary at multiple spatial scales?
            ├── Yes → Rational Quadratic  (or sum of RBFs with diff ℓ)
            └── No →
                Unknown smoothness?
                ├── Yes → Matern 5/2  ← safe default
                └── Known smooth → RBF/SE
```

For Bayesian Optimization specifically: **Matern 5/2 with ARD** is the standard default (used in BoTorch).

---

## Kernel Reference

### RBF / Squared Exponential

```
k(x, x') = σ² exp(−||x−x'||² / 2ℓ²)
```

**Intuition:** "Nearby points are correlated; correlation decays like a Gaussian bell curve."
Assumes the function is infinitely smooth (has derivatives of all orders).

|                      |                                                                                                             |
| -------------------- | ----------------------------------------------------------------------------------------------------------- |
| Smoothness           | Infinitely differentiable                                                                                   |
| Stationary           | Yes                                                                                                         |
| Hyperparams          | `ℓ` (length-scale), `σ²` (output variance)                                                           |
| **Use when**   | Very smooth underlying phenomena: physical simulations, smooth response surfaces                            |
| **Avoid when** | True function has sharp transitions — the kernel fights against roughness and produces "ringing" artefacts |

**`ℓ` intuition:** Small `ℓ` → rapid wiggles. Large `ℓ` → slow, sweeping variation.

---

### Matern 1/2  (= Ornstein-Uhlenbeck)

```
k(x, x') = σ² exp(−||x−x'|| / ℓ)
```

**Intuition:** "Exponentially decaying correlation — the GP equivalent of a random walk."
Continuous everywhere but differentiable nowhere. The roughest of the Matern family.

|                    |                                                                                                                 |
| ------------------ | --------------------------------------------------------------------------------------------------------------- |
| Smoothness         | 0 (continuous, not differentiable)                                                                              |
| Stationary         | Yes                                                                                                             |
| Hyperparams        | `ℓ`, `σ²`                                                                                                |
| **Use when** | Rough/spiky signals: financial time series, noisy physical measurements, processes with sharp local transitions |

---

### Matern 3/2

```
k(r) = σ²(1 + √3 r/ℓ) exp(−√3 r/ℓ),   r = ||x−x'||
```

**Intuition:** "Moderately smooth — once differentiable. A good middle ground between OU and RBF."
A popular default when you suspect RBF over-smooths.

|                    |                                                                                                                            |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| Smoothness         | 1 time mean-square differentiable                                                                                          |
| Stationary         | Yes                                                                                                                        |
| Hyperparams        | `ℓ`, `σ²`                                                                                                           |
| **Use when** | General empirical data where the function is somewhat smooth but not infinitely so; good default for real-world regression |

---

### Matern 5/2  ← recommended default

```
k(r) = σ²(1 + √5 r/ℓ + 5r²/3ℓ²) exp(−√5 r/ℓ)
```

**Intuition:** "Twice differentiable — smooth enough for most tasks, without the over-smoothing of RBF."
The BoTorch and GPyTorch default for Bayesian Optimization.

|                    |                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------- |
| Smoothness         | 2 times mean-square differentiable                                                                |
| Stationary         | Yes                                                                                               |
| Hyperparams        | `ℓ`, `σ²`                                                                                  |
| **Use when** | Bayesian optimization, surrogate modeling, any task where smoothness is unknown — safest default |

**Practical note:** Matern {1/2, 3/2, 5/2} have closed-form expressions and are ~10x cheaper to compute than general `ν`. Stick to these three.

---

### Periodic (ExpSineSquared)

```
k(x, x') = σ² exp(−2 sin²(π||x−x'||/p) / ℓ²)
```

**Intuition:** "Wraps the input onto a circle of circumference `p`, then applies RBF on the circle."
Produces functions that repeat exactly with period `p`.

|                      |                                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------------------ |
| Smoothness           | Infinitely differentiable within each period                                                     |
| Stationary           | Yes                                                                                              |
| Hyperparams          | `p` (period), `ℓ` (within-period smoothness), `σ²`                                      |
| **Use when**   | Time series with known/suspected periodicity: seasonal data, circadian rhythms, harmonic signals |
| **Limitation** | Cannot capture trends or aperiodic variation on its own                                          |

**Almost always combined:**

- `Periodic × RBF` → quasi-periodic (pattern fades with distance, like sunspot data)
- `Periodic + RBF` → periodic component + independent smooth trend

---

### Rational Quadratic

```
k(r) = σ²(1 + r² / 2αℓ²)^{−α}
```

**Intuition:** "An infinite mixture of RBF kernels with different length-scales."
As α → ∞ it converges to the RBF. Small α = heavy tails = multi-scale behavior.

|                    |                                                                                                              |
| ------------------ | ------------------------------------------------------------------------------------------------------------ |
| Smoothness         | Infinitely differentiable                                                                                    |
| Stationary         | Yes                                                                                                          |
| Hyperparams        | `ℓ`, `α` (scale-mixture weight), `σ²`                                                              |
| **Use when** | Function varies at multiple spatial scales simultaneously; when a single RBF length-scale is too restrictive |

---

### Linear (Dot Product)

```
k(x, x') = σ₀² + x·x'
```

**Intuition:** "A GP with this kernel is Bayesian linear regression."
Sample paths are linear functions.

|                    |                                                                                                        |
| ------------------ | ------------------------------------------------------------------------------------------------------ |
| Smoothness         | Infinitely differentiable (linear)                                                                     |
| Stationary         | **No** (depends on absolute position)                                                            |
| Hyperparams        | `σ₀²` (bias variance)                                                                             |
| **Use when** | Strong prior that relationship is linear; or as a component in composite kernels to add a linear trend |

---

### Polynomial

```
k(x, x') = (x·x' + c)^d
```

**Intuition:** "Bayesian polynomial regression of degree `d`."
Product of `d` linear kernels; models global polynomial structure.

|                    |                                                                      |
| ------------------ | -------------------------------------------------------------------- |
| Stationary         | **No**                                                         |
| Hyperparams        | `d` (degree, integer), `c` (inhomogeneity)                       |
| **Use when** | Expected polynomial relationship (physical scaling laws)             |
| **Warning**  | Poor extrapolation behavior — extreme values far from training data |

---

### Spectral Mixture (Wilson & Adams, 2013)

```
k(τ) = Σ_q w_q cos(2π µ_q τ) exp(−2π² τ² v_q)
```

**Intuition:** "Model the power spectrum of the signal as a Gaussian mixture in frequency space."
By Bochner's theorem, this is a universal approximator for stationary kernels.

|                     |                                                                                                                                                           |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Smoothness          | Infinitely differentiable                                                                                                                                 |
| Stationary          | Yes                                                                                                                                                       |
| Hyperparams         | Per component `q`: `µ_q` (frequency), `v_q` (bandwidth), `w_q` (weight). Total: Q × 3 params                                                    |
| **Use when**  | Time series with multiple/unknown periodicities; pattern discovery and extrapolation; when you want the kernel to discover the signal structure from data |
| **Trade-off** | Many hyperparameters → requires careful initialization (e.g., from a periodogram); harder to optimize                                                    |

---

## Composite Kernels

| Combination          | Meaning                                                   | Example use case                                                  |
| -------------------- | --------------------------------------------------------- | ----------------------------------------------------------------- |
| `k₁ + k₂`        | Function is a superposition of two independent components | `Matern52 + Periodic`: smooth trend + seasonal variation        |
| `k₁ × k₂`       | One kernel modulates the other; restricts correlation     | `Periodic × RBF`: quasi-periodic (pattern fades over distance) |
| `RBF + Linear`     | Smooth local variation + global linear trend              | Regression with drift                                             |
| `RBF + WhiteNoise` | Signal + iid noise                                        | Adding a nugget / noise term                                      |
| `Linear × Linear` | Quadratic (polynomial degree 2)                           | Bayesian quadratic regression                                     |

**Rule of thumb:** Use `+` to combine independent signal sources; use `×` to restrict one kernel by another.

---

## High-Dimensional Inputs

### Why standard kernels break down (D > ~20)

Standard kernels compute distances in the full input space. As D increases:

1. **Distance concentration** — every pair of points becomes approximately equidistant. The kernel matrix approaches rank-1 (all entries nearly identical) → encodes no useful structure.
2. **Exponential coverage requirement** — you need ~exp(D) training points to cover the space.
3. **ARD optimization degrades** — the MLE landscape for D length-scales becomes flat and multimodal with typical dataset sizes.

### Solutions by dimensionality regime

| Input dimension D | Recommendation                                                                                |
| ----------------- | --------------------------------------------------------------------------------------------- |
| D ≤ 5            | Any kernel works. Isotropic or ARD variants both fine.                                        |
| 5 < D ≤ 20       | **ARD-Matern 5/2** — standard default. Optimize length-scales carefully.               |
| 20 < D ≤ 50      | ARD with careful regularization; consider **additive kernels** or **SAASBO**.    |
| D > 50            | Deep kernels, additive kernels, or dimensionality-reduction methods are essentially required. |

---

### ARD (Automatic Relevance Determination)

Any stationary kernel can have a separate length-scale per dimension:

```
k_ARD(x, x') = k(sqrt(Σ_d (x_d − x'_d)² / ℓ_d²))
```

**Intuition:** If `ℓ_d → ∞`, dimension `d` contributes nothing to the distance — the kernel ignores that input. ARD discovers which inputs matter during hyperparameter optimization.

- **Works well:** D ≤ 20–30 with adequate data
- **Breaks down:** D > 30–50 — too many length-scales to optimize reliably

---

### Additive Kernels (Duvenaud et al., 2011)

```
k(x, x') = Σ_d k_d(x_d, x'_d)       # fully additive
```

or higher-order interactions: sum over subsets of dimensions.

**Intuition:** "The function is approximately a sum of univariate functions — one per dimension."
Each 1D kernel only needs marginal data coverage, sidestepping the curse of dimensionality.

- **Use when:** Function follows a generalized additive model structure; input effects are mostly independent
- **Limitation:** Cannot capture high-order interaction effects between dimensions (by design)
- **Reference:** [Duvenaud et al., 2011 — Additive Kernels for GP Modeling](https://arxiv.org/abs/1103.4023)

---

### Deep Kernels / Deep Kernel Learning (Wilson et al., 2016)

```
k(x, x') = k_base(φ_θ(x), φ_θ(x'))
```

where `φ_θ` is a neural network that learns a low-dimensional representation.

**Intuition:** "The neural network learns what the relevant representation of high-dimensional inputs is; the base GP kernel then operates in that learned space where distances are meaningful."

- **Use when:** High-dimensional structured inputs (images, molecular fingerprints, embeddings, text features) where raw input distances are not informative
- **Trade-off:** Requires more data to train the NN component; loses interpretability of standard GP hyperparameters
- **Reference:** [Wilson et al., 2016 — Deep Kernel Learning](https://arxiv.org/abs/1511.02222)

---

### Dimensionality Reduction Approaches

| Method                       | Core Idea                                                     | Assumption                                   |
| ---------------------------- | ------------------------------------------------------------- | -------------------------------------------- |
| **REMBO**              | Random projection to low-D subspace; optimize there           | True optimum lies in a low-D linear subspace |
| **ALEBO**              | Improved REMBO with better embedding handling                 | Same as REMBO, more robust                   |
| **SAASBO**             | Sparsity-inducing half-Cauchy priors on inverse length-scales | Only a few dimensions are "active"           |
| **Active Subspace GP** | Learn linear projection A s.t. f(x) ≈ g(Ax)                  | Function varies in a linear subspace         |
| **GPLVM**              | Treat low-D latent coordinates as model parameters            | Manifold structure in inputs                 |

**SAASBO** (Eriksson & Jankowiak, 2021) is currently among the strongest for high-D Bayesian optimization: sparsity prior strongly regularizes the model toward using few dimensions without hard feature selection.

---

## Summary Table

| Kernel             | Stationary | Differentiable | Key Params       | Best Use Case                      | High-D?                |
| ------------------ | ---------- | -------------- | ---------------- | ---------------------------------- | ---------------------- |
| RBF/SE             | Yes        | ∞             | ℓ, σ²         | Very smooth functions, simulations | Poor (D > 10)          |
| Matern 1/2         | Yes        | 0              | ℓ, σ²         | Rough/spiky signals, finance       | Poor                   |
| Matern 3/2         | Yes        | 1              | ℓ, σ²         | General empirical data             | Poor (use ARD variant) |
| Matern 5/2         | Yes        | 2              | ℓ, σ²         | BO default, surrogate modeling     | OK with ARD (D ≤ 30)  |
| Periodic           | Yes        | ∞             | ℓ, p, σ²      | Time series with periodicity       | N/A (1D time)          |
| Rational Quadratic | Yes        | ∞             | ℓ, α, σ²     | Multi-scale variation              | Poor                   |
| Linear             | No         | ∞ (linear)    | σ₀²           | Linear trend component             | OK (linear is exact)   |
| Polynomial         | No         | ∞             | d, c             | Polynomial trends                  | Poor                   |
| Spectral Mixture   | Yes        | ∞             | µ, v, w ×Q     | Pattern discovery, time series     | Poor                   |
| ARD-Matern/RBF     | Yes        | varies         | ℓ_1..ℓ_D, σ² | Moderate-D, irrelevant features    | OK (D ≤ 30)           |
| Additive           | Varies     | varies         | per-dim          | High-D, weak interactions          | Good (D ≤ 100+)       |
| Deep Kernel        | Yes        | ∞             | NN + base        | High-D structured inputs           | Excellent              |
| SAASBO             | Yes        | 2 (Matern)     | sparse ARD       | High-D BO, few active dims         | Excellent (D ≤ 1000)  |

---

## Model Selection in Practice

Use **Marginal Log-Likelihood (MLL)** as the primary selection criterion — it automatically penalizes complexity (Bayesian Occam's razor):

```python
# GPyTorch / BoTorch pattern
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)
# Compare mll values across kernel choices
```

If multiple kernels give similar MLL: prefer the simpler one (fewer hyperparameters, better-conditioned optimization).

---

## References

- Rasmussen & Williams, *Gaussian Processes for Machine Learning*, Ch. 4 — [gpml](http://gaussianprocess.org/gpml/chapters/RW4.pdf)
- Duvenaud, *The Kernel Cookbook* — [toronto.edu](https://www.cs.toronto.edu/~duvenaud/cookbook/)
- Wilson & Adams (2013), *Spectral Mixture Kernels* — [arxiv:1302.4245](https://arxiv.org/abs/1302.4245)
- Wilson et al. (2016), *Deep Kernel Learning* — [arxiv:1511.02222](https://arxiv.org/abs/1511.02222)
- Duvenaud et al. (2011), *Additive Kernels* — [arxiv:1103.4023](https://arxiv.org/abs/1103.4023)
- Eriksson & Jankowiak (2021), *SAASBO* — [PMLR](https://proceedings.mlr.press/v161/eriksson21a/eriksson21a.pdf)
