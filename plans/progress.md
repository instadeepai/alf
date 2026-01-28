# Implementation Progress: BoTorch Integration

## Completed Phases

### Phase 1: Foundation & Single Acquisition Function (qEI)
**Completed**: January 27, 2026
**Status**: Complete

#### Changes Made
- **Dependencies**:
  - Added `botorch>=0.16.0` to `tools/pyproject.toml`
  - Updated PyTorch to `>=2.0.0` and GPyTorch to `>=1.11.0` for compatibility

- **Core Implementation Files**:
  - `tools/alf_tools/datasets/botorch_test_functions.py` - BoTorch synthetic test function adapter
  - `tools/alf_tools/utils/botorch_utils.py` - Conversion utilities between ALF and BoTorch formats
  - `tools/alf_tools/optimizer/acquisition_functions/botorch_qei.py` - qExpectedImprovement acquisition function

- **Tutorial**:
  - `tutorials/botorch_synthetic_tutorial.ipynb` - Complete tutorial demonstrating optimization on Branin function

- **Tests**:
  - `tools/tests/datasets/test_botorch_test_functions.py` - 13 tests for synthetic dataset adapter
  - `tools/tests/utils/test_botorch_utils.py` - 17 tests for utility functions
  - `tools/tests/optimizer/acquisition_functions/test_botorch_qei.py` - 15 tests for qEI acquisition function

- **Module Exports**:
  - Updated `tools/alf_tools/datasets/__init__.py` to export `BoTorchSyntheticDataset`
  - Updated `tools/alf_tools/optimizer/acquisition_functions/__init__.py` to export `BoTorchQEI`
  - Fixed `tools/alf_tools/models/__init__.py` to properly export `GPModel`

- **Bug Fixes**:
  - Fixed pre-existing typo: `LabeledCandidates` → `LabelledCandidates` in:
    - `tools/alf_tools/models/gp.py`
    - `tools/alf_tools/models/model_utils.py`

#### Key Learnings
- BoTorch test functions provide excellent domain-neutral benchmarking capabilities
- The adapter pattern works well for bridging BoTorch's tensor-based API with ALF's Candidate-based system
- qEI supports both:
  1. **Scoring mode**: Evaluate acquisition values for discrete candidate pools
  2. **Optimization mode**: Continuous optimization for unbounded search spaces
- BoTorch's batch acquisition (q>1) jointly optimizes candidates for diversity and quality

#### Implementation Highlights

**BoTorch Synthetic Dataset Adapter**:
- Supports 8 test functions: Branin, Hartmann (3D/6D), Ackley, Rosenbrock, BraninCurrin, DTLZ2, ZDT1
- Configurable noise, dimensionality, and bounds
- Automatic negation for maximization (BoTorch functions are minimization by default)
- Integration with ALF's dataset splitting and query interface

**Conversion Utilities**:
- `candidates_to_tensor()`: Convert ALF Candidates → torch.Tensor
- `tensor_to_candidates()`: Convert torch.Tensor → ALF Candidates
- `predictions_to_posterior()`: Convert ALF Predictions → BoTorch GPyTorchPosterior
- `get_bounds_tensor()`: Convert bounds to BoTorch format

**BoTorchQEI Acquisition Function**:
- Dual-mode operation (scoring vs optimization)
- MC sampling with Sobol sequences for approximation
- Configurable batch size, restarts, and samples
- Sequential vs joint batch optimization support

#### Test Coverage
- **45 total tests** across all three modules
- All tests passing with comprehensive coverage:
  - Dataset initialization and configuration
  - Function evaluation with/without noise
  - Conversion utilities with various data formats
  - qEI in both scoring and optimization modes
  - Reproducibility and edge cases

#### Verification Status
- ✅ All pre-commit checks passing (ruff linter + formatter)
- ✅ All 45 unit/integration tests passing
- ✅ Code quality: Type hints, docstrings, follows ALF conventions
- ✅ No import errors or dependency issues

#### Notes for Future Phases
- **Dependencies created**:
  - BoTorch test functions can be used for benchmarking future acquisition functions
  - Conversion utilities are reusable for all BoTorch integrations
  - Pattern established for wrapping BoTorch acquisition functions

- **Things to watch out for**:
  - BoTorch expects float32 tensors by default; ensure dtype consistency
  - Posterior shapes can vary ([n] vs [n, 1]); tests should handle both
  - Optimization mode requires bounds; validate configuration

- **Patterns established**:
  - Synthetic datasets use TABULAR modality for continuous optimization
  - Acquisition functions check for empty `search_candidates` to switch modes
  - Tests use fixtures for common setup (dataset, surrogate, task_state)

## Next Phase
**Phase 2: Additional Acquisition Functions** - Ready to implement
- qNEI (Noisy Expected Improvement)
- qUCB (Upper Confidence Bound)
- qKG (Knowledge Gradient)

## Overall Verification Status
- ✅ Phase 1 complete and verified
- ✅ All linting checks passing
- ✅ All tests passing
- ✅ Tutorial notebook created
- ✅ Documentation complete
