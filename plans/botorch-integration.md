# Feature Plan: BoTorch Integration with ALF

**Created**: January 27, 2026
**Status**: Draft (Pending Approval)

## Summary

Integrate BoTorch (Bayesian Optimization library) into the ALF framework to provide advanced batch acquisition functions, multi-objective optimization capabilities, and enhanced Gaussian Process models. The integration will be comprehensive but implemented as an MVP first, with BoTorch as a required dependency only in `alf_tools` (keeping `alf_core` independent). This maintains ALF's modular architecture while unlocking state-of-the-art Bayesian optimization capabilities for general-purpose biological design tasks.

**Key Innovation: Domain-Neutral Testing** - The integration will use BoTorch's synthetic test functions (Branin, Hartmann, Ackley, DTLZ, ZDT) as the primary benchmarking suite. This ensures the implementation is truly domain-agnostic and can be validated against known ground truth optimums, making it robust for any application domain (proteins, molecules, materials, etc.).

## Decisions Made

| Decision | Selected Option | Rationale |
|----------|-----------------|-----------|
| **Integration Scope** | Comprehensive (acquisition functions + GP models + multi-objective) | Maximize value by providing full BoTorch capabilities |
| **Dependency Strategy** | Required in `alf_tools` only | Keeps `alf_core` lightweight and independent |
| **Primary Domain** | General purpose | Flexible framework for proteins, molecules, materials, etc. |
| **Implementation Approach** | MVP first, then iterate | Build minimal working example to validate integration, then expand |

## Technical Approach

### Architecture Overview

BoTorch integration will follow ALF's existing patterns:

1. **Acquisition Functions** (`alf_tools/optimizer/acquisition_functions/botorch_*.py`)
   - Wrap BoTorch's batch acquisition functions (qEI, qNEI, qUCB, qKG)
   - Implement ALF's `AcquisitionFunction` interface
   - Convert between ALF's `Predictions` and BoTorch's model outputs

2. **GP Surrogate Models** (`alf_tools/models/botorch_gp.py`)
   - Enhance existing GPyTorch GP with BoTorch's SingleTaskGP
   - Maintain compatibility with ALF's `BaseModel` interface
   - Support BoTorch's model fitting utilities (fit_gpytorch_mll)

3. **Multi-Objective Support** (`alf_tools/optimizer/acquisition_functions/multi_objective/`)
   - New acquisition functions: qEHVI, qNEHVI for Pareto optimization
   - Multi-objective task extension (future phase)

### Domain-Neutral Testing Strategy

**Why BoTorch Test Functions?**

Using BoTorch's synthetic test functions provides several critical advantages:

1. **Known Ground Truth**: Functions like Branin, Hartmann have known global optima
   - Can measure true regret: `regret = f(x) - f(x*)`
   - Validate convergence behavior objectively

2. **Fast Iteration**: Synthetic functions evaluate in microseconds
   - Test on 1000s of evaluations quickly
   - Rapid prototyping and debugging

3. **Controlled Complexity**: Test across different landscapes
   - Convex (Quadratic)
   - Multimodal (Ackley, Hartmann)
   - High-dimensional (Ackley, Rosenbrock with d>10)

4. **Domain Agnostic**: Not tied to biology/chemistry
   - Implementation works for any optimization problem
   - Easy to extend to new domains

5. **Standard Benchmarks**: Widely used in BO literature
   - Compare with published results
   - Reproducible experiments

**Test Function Suite**

| Function | Dim | Properties | Use Case |
|----------|-----|------------|----------|
| **Branin** | 2 | 3 global minima, simple | MVP validation, quick tests |
| **Hartmann3** | 3 | 4 local minima | Medium complexity |
| **Hartmann6** | 6 | 6 local minima | High-dimensional |
| **Ackley** | 2-20 | Highly multimodal | Exploration stress test |
| **Rosenbrock** | 2-20 | Narrow valley | Exploitation stress test |
| **BraninCurrin** | 2 | Bi-objective | Multi-objective testing |
| **DTLZ2** | 5-10 | Concave Pareto front | Multi-objective testing |
| **ZDT1** | 5-10 | Convex Pareto front | Multi-objective testing |

**Adapter Design**

```python
from botorch.test_functions import Branin
from alf_core import BaseDataset, Candidate, LabeledCandidates

class BoTorchSyntheticDataset(BaseDataset):
    """Adapter to use BoTorch test functions with ALF."""

    def __init__(self, function_name="Branin", dim=None, noise_std=0.0, ...):
        self.test_fn = get_botorch_function(function_name, dim)
        self.bounds = self.test_fn.bounds  # e.g., [0,1]^d

    def load_dataset(self) -> LabeledCandidates:
        # Generate random initial samples
        X = random_sample(bounds=self.bounds, n_samples=1000)
        y = self.test_fn(X).numpy()

        # Convert to Candidates (store tensors as data)
        candidates = [Candidate(data=x, modality="continuous") for x in X]
        return LabeledCandidates(candidates=candidates, labels=y)

    def query(self, candidates: list[Candidate]) -> np.ndarray:
        # Oracle evaluation: evaluate test function
        X = torch.stack([c.data for c in candidates])
        return self.test_fn(X).numpy()
```

**Validation Workflow**

```python
# 1. Create synthetic dataset
dataset = BoTorchSyntheticDataset(
    function_name="Branin",
    n_initial=50,
    noise_std=0.1
)

# 2. Use ALF framework as normal
surrogate = Surrogate(model=BoTorchGPModel())
optimizer = Optimizer(
    acquisition_fn=BoTorchQEI(),
    search_fn=DatasetSearch()
)
oracle = Oracle(scorer=dataset)  # Uses dataset.query()

# 3. Run optimization
task = DesignTask(num_acq_rounds=20, acq_batch_size=5)
state = task.setup(dataset=dataset, surrogate=surrogate)
task.run(state, optimizer=optimizer, oracle=oracle)

# 4. Validate results
best_value = state.dataset.train_dataset.labels.max()
true_optimum = -0.397887  # Known Branin optimum
regret = true_optimum - best_value
print(f"Final regret: {regret:.4f}")  # Should decrease over rounds
```

This makes testing completely independent of biological data while ensuring the implementation works for any domain.

### Key Integration Points

```
ALF Architecture with BoTorch:

┌─────────────────────────────────────────────────────────────┐
│  alf_core (Independent, no BoTorch dependency)              │
│  ├── AcquisitionFunction (abstract base)                    │
│  ├── BaseModel (abstract base)                              │
│  ├── Surrogate (wrapper)                                    │
│  └── Predictions (dataclass with means/variances)           │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │
┌─────────────────────────────────────────────────────────────┐
│  alf_tools (BoTorch implementations)                        │
│  ├── acquisition_functions/                                 │
│  │   ├── botorch_qei.py (qExpectedImprovement)             │
│  │   ├── botorch_qnei.py (qNoisyExpectedImprovement)       │
│  │   ├── botorch_qucb.py (qUpperConfidenceBound)           │
│  │   └── botorch_qkg.py (qKnowledgeGradient)               │
│  ├── models/                                                │
│  │   ├── gp.py (existing GPyTorch model)                   │
│  │   └── botorch_gp.py (BoTorch-enhanced GP)               │
│  └── utils/                                                 │
│      └── botorch_utils.py (conversion helpers)             │
└─────────────────────────────────────────────────────────────┘
```

## MVP Implementation Steps

### Phase 1: Foundation & Single Acquisition Function (Week 1)

1. **Add BoTorch Dependency**
   - [ ] Add `botorch>=0.16.0` to `tools/pyproject.toml`
   - [ ] Update documentation with BoTorch installation notes
   - [ ] Verify compatibility with existing GPyTorch version (>=1.9.0)

2. **Create BoTorch Test Function Dataset Adapter**
   - [ ] Create `tools/alf_tools/datasets/botorch_test_functions.py`
   - [ ] Implement `BoTorchSyntheticDataset(BaseDataset)` class:
     - Wrap BoTorch test functions (Branin, Hartmann, Ackley, etc.)
     - Convert continuous tensor inputs → ALF `Candidate` objects
     - Generate initial dataset by random sampling
     - Support both single and multi-objective functions
   - [ ] Support all major test functions:
     - Branin (2D), Hartmann (3D/6D), Ackley (2D+)
     - For multi-objective: BraninCurrin, DTLZ2, ZDT1
   - [ ] Add configuration for bounds, noise, dimensionality

3. **Create Utility Module**
   - [ ] Create `tools/alf_tools/utils/botorch_utils.py`
   - [ ] Implement conversion functions:
     - `candidates_to_tensor()`: Convert ALF Candidates → torch.Tensor for BoTorch
     - `tensor_to_candidates()`: Convert torch.Tensor → ALF Candidates
     - `predictions_to_posterior()`: Convert ALF Predictions → BoTorch GPyTorchPosterior
     - `wrap_gp_model()`: Wrap ALF GP model for BoTorch acquisition functions

4. **Implement First Acquisition Function: qEI**
   - [ ] Create `tools/alf_tools/optimizer/acquisition_functions/botorch_qei.py`
   - [ ] Implement `BoTorchQEI` class:
     - Inherits from `AcquisitionFunction`
     - Wraps `botorch.acquisition.qExpectedImprovement`
     - Handles batch optimization with `optimize_acqf()`
     - Works with continuous spaces from BoTorch test functions
   - [ ] Add comprehensive docstrings with usage examples

5. **Create Tutorial Notebook**
   - [ ] Create `tutorials/botorch_synthetic_tutorial.ipynb`
   - [ ] Demonstrate optimization on Branin function (2D, known optimum)
   - [ ] Compare ALF's EI vs BoTorch's qEI
   - [ ] Show convergence plots, regret curves
   - [ ] Include batch diversity analysis
   - [ ] Use different test functions (Hartmann, Ackley) to show generalization

6. **Add Tests with Synthetic Functions**
   - [ ] Create `tools/tests/optimizer/acquisition_functions/test_botorch_qei.py`
   - [ ] Test cases using synthetic functions:
     - Basic functionality with Branin function
     - Batch selection (compare q=1 vs q=10) on Hartmann
     - Convergence to known optimum
     - Edge cases (bounds, dimensionality)
   - [ ] Create `tools/tests/datasets/test_botorch_test_functions.py`
   - [ ] Test dataset adapter with various BoTorch functions

### Phase 2: Additional Acquisition Functions (Week 2)

6. **Implement qNEI (Noisy Expected Improvement)**
   - [ ] Create `botorch_qnei.py`
   - [ ] Handle noisy observations (important for experimental data)
   - [ ] Add tests

7. **Implement qUCB (Upper Confidence Bound)**
   - [ ] Create `botorch_qucb.py`
   - [ ] Support beta parameter for exploration-exploitation tradeoff
   - [ ] Add tests

8. **Implement qKG (Knowledge Gradient)**
   - [ ] Create `botorch_qkg.py`
   - [ ] More sophisticated but computationally expensive
   - [ ] Add tests

9. **Update Documentation**
   - [ ] Add acquisition function comparison table to docs
   - [ ] Document when to use each acquisition function
   - [ ] Add performance benchmarks

### Phase 3: Enhanced GP Model (Week 3)

10. **Create BoTorch-Enhanced GP Model**
    - [ ] Create `tools/alf_tools/models/botorch_gp.py`
    - [ ] Implement `BoTorchGPModel(BaseModel)`:
      - Use `botorch.models.SingleTaskGP`
      - Use `botorch.fit.fit_gpytorch_mll` for optimization
      - Support `ModelListGP` for multi-output scenarios
    - [ ] Maintain compatibility with existing `GPModelTrainer`

11. **Add Advanced Features**
    - [ ] Support for heteroscedastic noise (different noise per observation)
    - [ ] Input transformations (normalize, standardize)
    - [ ] Output transformations (log, standardize)
    - [ ] Add comprehensive tests

12. **Create Comparison Tutorial**
    - [ ] Notebook comparing GPyTorch GP vs BoTorch GP
    - [ ] Show when BoTorch's utilities provide value
    - [ ] Performance and accuracy comparison

### Phase 4: Multi-Objective Support (Week 4)

13. **Create Multi-Objective Acquisition Functions**
    - [ ] Create `tools/alf_tools/optimizer/acquisition_functions/multi_objective/`
    - [ ] Implement `BoTorchQEHVI` (Expected Hypervolume Improvement)
    - [ ] Implement `BoTorchQNEHVI` (Noisy EHVI)
    - [ ] Add tests for Pareto frontier optimization

14. **Extend Predictions Dataclass** (if needed)
    - [ ] Consider adding multi-objective support to `Predictions`
    - [ ] May need separate dataclass for multi-objective predictions

15. **Create Multi-Objective Tutorial**
    - [ ] Notebook demonstrating protein optimization for:
      - Stability + Activity
      - Expression + Binding affinity
    - [ ] Show Pareto frontier visualization
    - [ ] Compare to sequential single-objective optimization

### Phase 5: Integration & Polish (Week 5)

16. **End-to-End Integration Testing**
    - [ ] Create `tools/tests/e2e_experiments/test_botorch_design_gfp.py`
    - [ ] Full experiment: GFP design with BoTorch qEI
    - [ ] Verify all metrics computed correctly
    - [ ] Compare to existing acquisition functions

17. **Performance Benchmarking**
    - [ ] Create benchmarking script
    - [ ] Compare acquisition time: ALF vs BoTorch
    - [ ] Compare optimization performance (regret, top-k recovery)
    - [ ] Document tradeoffs

18. **Documentation Updates**
    - [ ] Update main README with BoTorch integration
    - [ ] Add BoTorch section to `core/README.md`
    - [ ] Create `docs/BOTORCH_GUIDE.md` with:
      - Installation instructions
      - Usage examples
      - Best practices
      - Troubleshooting

19. **Code Quality**
    - [ ] Ensure all tests pass
    - [ ] Run linters and formatters
    - [ ] Add type hints throughout
    - [ ] Update pre-commit hooks if needed

## Files to Create

### New Files
- `tools/alf_tools/datasets/botorch_test_functions.py` - **BoTorch test function adapter for domain-neutral testing**
- `tools/alf_tools/utils/botorch_utils.py` - Conversion utilities (Candidates ↔ Tensors, GP wrapping)
- `tools/alf_tools/optimizer/acquisition_functions/botorch_qei.py` - qEI implementation
- `tools/alf_tools/optimizer/acquisition_functions/botorch_qnei.py` - qNEI implementation
- `tools/alf_tools/optimizer/acquisition_functions/botorch_qucb.py` - qUCB implementation
- `tools/alf_tools/optimizer/acquisition_functions/botorch_qkg.py` - qKG implementation
- `tools/alf_tools/optimizer/acquisition_functions/multi_objective/__init__.py`
- `tools/alf_tools/optimizer/acquisition_functions/multi_objective/botorch_qehvi.py`
- `tools/alf_tools/optimizer/acquisition_functions/multi_objective/botorch_qnehvi.py`
- `tools/alf_tools/models/botorch_gp.py` - BoTorch-enhanced GP model
- `tutorials/botorch_synthetic_tutorial.ipynb` - **MVP tutorial using synthetic test functions**
- `tutorials/botorch_multi_objective_tutorial.ipynb` - Multi-objective tutorial with DTLZ/ZDT
- `docs/BOTORCH_GUIDE.md` - Comprehensive guide

### Test Files
- `tools/tests/datasets/test_botorch_test_functions.py` - **Test synthetic function adapter**
- `tools/tests/utils/test_botorch_utils.py`
- `tools/tests/optimizer/acquisition_functions/test_botorch_qei.py` - **Uses Branin/Hartmann**
- `tools/tests/optimizer/acquisition_functions/test_botorch_qnei.py`
- `tools/tests/optimizer/acquisition_functions/test_botorch_qucb.py`
- `tools/tests/optimizer/acquisition_functions/test_botorch_qkg.py`
- `tools/tests/optimizer/acquisition_functions/multi_objective/test_botorch_qehvi.py`
- `tools/tests/models/test_botorch_gp.py`
- `tools/tests/e2e_experiments/test_botorch_branin.py` - **E2E test with Branin function**
- `tools/tests/e2e_experiments/test_botorch_hartmann.py` - **E2E test with Hartmann function**

## Files to Modify

- `tools/pyproject.toml` - Add BoTorch dependency
- `tools/alf_tools/optimizer/acquisition_functions/__init__.py` - Export new classes
- `tools/alf_tools/models/__init__.py` - Export BoTorchGPModel
- `README.md` - Add BoTorch integration overview
- `core/README.md` - Document BoTorch compatibility
- `.github/workflows/tests_and_linters.yaml` - Ensure CI tests BoTorch

## Dependencies

### New Dependencies
```toml
[project.dependencies]
botorch = ">=0.16.0"  # Main BoTorch library
```

### Version Compatibility
- Python: >=3.12 (existing requirement)
- PyTorch: >=1.9.0 (existing, BoTorch requires >=1.13.1, should update)
- GPyTorch: >=1.9.0 (existing, BoTorch requires >=1.11, should update)

### Recommended Updates to `tools/pyproject.toml`
```toml
dependencies = [
    "alf_core",
    "requests>=2.25.0",
    "huggingface_hub==0.27.0",
    "jaxtyping>=0.3.5",
    "torch>=2.0.0",  # Updated for BoTorch compatibility
    "gpytorch>=1.11.0",  # Updated for BoTorch compatibility
    "botorch>=0.16.0",  # New dependency
]
```

## Testing Strategy

### Unit Tests
- Test each acquisition function independently with mock GP models
- Test utility functions for data conversion
- Test BoTorchGPModel initialization, training, and prediction

### Integration Tests
- Test acquisition functions with real ALF GP models
- Test compatibility with existing datasets (GFP, ProteinGym)
- Test multi-round optimization loops

### End-to-End Tests
- Full design task experiments comparing BoTorch vs existing acquisition functions
- Measure: regret, top-k recovery, batch diversity, runtime
- Test on multiple seeds for statistical significance

### Benchmarking
- Compare acquisition time (seconds per batch)
- Compare optimization performance (regret curves)
- Compare uncertainty calibration (ECE, coverage)
- Document when BoTorch provides most value

## Success Criteria

The MVP is considered successful when:

1. ✅ **Functional Integration**: At least one BoTorch acquisition function (qEI) works end-to-end with synthetic test functions
2. ✅ **Known Optimum Convergence**: On Branin function, achieves regret < 0.1 within 20 rounds (known optimum: -0.397887)
3. ✅ **Performance**: BoTorch qEI achieves better regret than ALF's greedy/EI on standard benchmarks (Branin, Hartmann)
4. ✅ **Batch Diversity**: BoTorch's batch selection (q>1) shows better diversity than independent scoring
5. ✅ **Domain Agnostic**: Same code works for 2D Branin, 6D Hartmann without modification
6. ✅ **Documentation**: Tutorial notebook demonstrates usage on multiple test functions
7. ✅ **Tests Pass**: All unit, integration, and E2E tests pass with synthetic functions
8. ✅ **Code Quality**: Passes linters, has type hints, follows ALF conventions

## Future Extensions (Beyond MVP)

After MVP validation:

1. **Constrained Optimization**
   - Add `qNoisyConstrainedEI` for constrained design
   - Support constraints like synthesizability, solubility, toxicity

2. **Multi-Task Learning**
   - Use `MultiTaskGP` for related design tasks
   - Transfer learning across protein families

3. **High-Dimensional Optimization**
   - Add SAASBO (Sparse Axis-Aligned Subspace BO) for high-dim spaces
   - Useful for full protein sequences (100+ residues)

4. **Preference Learning**
   - Add preference-based acquisition functions
   - Learn from pairwise comparisons instead of absolute values

5. **Parallel Optimization**
   - Add support for parallel batch evaluation
   - Asynchronous acquisition for long-running experiments

6. **Custom Kernels**
   - String kernels for sequences
   - Graph kernels for molecular structures
   - Integrate with BoTorch's custom kernel framework

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **BoTorch API changes** | High | Pin specific BoTorch version, monitor releases |
| **Performance overhead** | Medium | Benchmark early, optimize hot paths, add caching |
| **Complex API for users** | Medium | Provide high-level wrappers, comprehensive tutorials |
| **Dependency conflicts** | High | Test on fresh environments, document conflicts |
| **GPU memory issues** | Medium | Add batch size limits, support CPU fallback |

## Notes

### Why BoTorch + ALF is Powerful

1. **ALF provides the framework**: Data management, experiment loop, metrics, logging
2. **BoTorch provides the algorithms**: State-of-the-art acquisition functions, GP models
3. **Together**: Best-in-class active learning for biological design

### Architectural Principles

1. **Keep `alf_core` independent**: No BoTorch in core, only in tools
2. **Maintain backwards compatibility**: Existing code should work unchanged
3. **Follow ALF patterns**: Implement standard interfaces, use same testing patterns
4. **Documentation first**: Every feature needs tutorial + docstrings

### Development Workflow

1. Start with Phase 1 (qEI MVP) using synthetic test functions
2. Validate on standard benchmarks (Branin, Hartmann, Ackley)
3. Once validated, optionally test on real datasets (GFP, ProteinGym) for sanity check
4. Get feedback from users/stakeholders
5. Iterate based on real use cases
6. Gradually add more features (Phases 2-5)
7. Maintain high test coverage throughout

### Testing Philosophy

**Primary Tests: BoTorch Synthetic Functions**
- Fast, deterministic, known ground truth
- Used for unit tests, integration tests, CI/CD
- Enables rapid iteration and debugging

**Secondary Validation: Real Datasets (Optional)**
- GFP, ProteinGym for sanity checking
- Slower, domain-specific validation
- Not required for core functionality

This ensures the implementation is robust and domain-agnostic from day one.

---

**Next Steps After Approval:**
1. Create implementation branches
2. Start with Phase 1, Step 1 (add BoTorch dependency)
3. Implement incrementally with tests at each step
4. Review and merge after each phase completion
