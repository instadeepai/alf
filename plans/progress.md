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

## Phase 1.5: Generic BoTorch Wrappers & GP Models (Extended)
**Completed**: February 3, 2026
**Status**: Complete

### Changes Made

#### New Components Implemented

1. **`BoTorchGPModel`** (`tools/alf_tools/models/botorch_gp_models.py`)
   - Modern GP model built on BoTorch's `SingleTaskGP`
   - Better default hyperparameter priors (Hvarfner et al. 2024)
   - Automatic output standardization
   - Efficient model fitting with L-BFGS-B via `fit_gpytorch_mll`
   - Implements full `BaseModel` interface (featurise, train, predict, sample)
   - Works seamlessly with both continuous and discrete search spaces

2. **`BoTorchMCSampler`** (`tools/alf_tools/optimizer/search/botorch_search_functions.py`)
   - Wrapper for BoTorch's Monte Carlo samplers
   - Supports Sobol QMC (recommended) and IID sampling
   - Configurable number of samples and random seed
   - Clean interface: create config, get sampler when needed
   - Used by acquisition functions to control MC approximation

3. **`ContinuousSearch`** (`tools/alf_tools/optimizer/search/botorch_search_functions.py`)
   - Search function for continuous optimization
   - Returns empty candidate list to signal `optimize_acqf` mode
   - Works with BoTorch acquisition functions

4. **`BoTorchAcquisition`** (`tools/alf_tools/optimizer/acquisition_functions/botorch_acquisition.py`)
   - **Generic wrapper for ALL BoTorch acquisition functions**
   - Easy switching via `acquisition_type` parameter
   - Supports: qEI, qUCB, qNEI (qKG coming soon)
   - Works in both discrete (scoring) and continuous (optimization) modes
   - Configurable MC sampling, restarts, batch optimization
   - Replaces need for individual wrapper classes

#### Documentation & Examples

- **Comprehensive Guide** (`docs/botorch_integration_guide.md`)
  - Complete API documentation for all new components
  - Best practices and troubleshooting
  - Comparison tables (BoTorchGPModel vs GPModel, etc.)
  - When to use which component
  - Advanced usage patterns

- **Demo Script** (`examples/botorch_integration_demo.py`)
  - Working examples of all components
  - Shows how to switch between acquisition functions
  - Demonstrates full optimization workflow
  - Validates all integrations work correctly

#### Module Exports Updated

- `tools/alf_tools/models/__init__.py` - Added `BoTorchGPModel`
- `tools/alf_tools/optimizer/search/__init__.py` - Added `ContinuousSearch`, `BoTorchMCSampler`
- `tools/alf_tools/optimizer/acquisition_functions/__init__.py` - Added `BoTorchAcquisition`

### Key Design Decisions

1. **Generic Acquisition Wrapper**
   - Created single `BoTorchAcquisition` class instead of separate wrappers
   - Users switch functions via parameter: `acquisition_type="qEI"` or `"qUCB"`
   - Reduces code duplication and makes experimentation easier
   - Individual wrappers (like `BoTorchQEI`) still available for advanced use

2. **BoTorchGPModel vs GPModel**
   - Created **alongside** existing `GPModel`, not replacing it
   - `BoTorchGPModel`: Better for continuous spaces, BoTorch integration
   - `GPModel`: Better for sequence spaces, custom kernels, more control
   - Both share same `BaseModel` interface - easy to swap

3. **MC Sampler Configuration**
   - Separated sampler configuration from acquisition function
   - `BoTorchMCSampler` is a config object, not a search function
   - Pass to acquisition function for MC approximation control
   - Consistent with BoTorch's design patterns

4. **Clean, Understandable Code**
   - Comprehensive docstrings with examples
   - Type hints throughout
   - Clear parameter documentation
   - Follows ALF conventions and patterns

### Code Quality

- ✅ All linting checks passing (ruff, basedpyright)
- ✅ Proper type hints throughout
- ✅ Comprehensive docstrings with usage examples
- ✅ Follows ALF architectural patterns
- ✅ Demo script runs successfully
- ✅ No breaking changes to existing code

### Usage Example

```python
from alf_tools.models import BoTorchGPModel
from alf_tools.optimizer.acquisition_functions import BoTorchAcquisition
from alf_tools.optimizer.search import BoTorchMCSampler, ContinuousSearch

# Create components
model = BoTorchGPModel()
sampler = BoTorchMCSampler(sampler_type="sobol", num_samples=512)

# Easy switching between acquisition functions
acq_qei = BoTorchAcquisition(acquisition_type="qEI", sampler=sampler, ...)
acq_qucb = BoTorchAcquisition(acquisition_type="qUCB", beta=0.2, sampler=sampler, ...)
acq_qnei = BoTorchAcquisition(acquisition_type="qNEI", sampler=sampler, ...)

# Use in optimizer
optimizer = Optimizer(acquisition_fn=acq_qei, search_fn=ContinuousSearch())
```

### Benefits Delivered

1. **Easy Switching**: Change acquisition function with one parameter
2. **Clean Architecture**: Generic wrapper reduces code duplication
3. **Backward Compatible**: Existing code works unchanged
4. **Well Documented**: Comprehensive guide + working examples
5. **Production Ready**: All linting passing, proper error handling

## Next Phase
**Phase 2: Testing & Validation** - Ready to implement
- Unit tests for `BoTorchGPModel`
- Unit tests for `BoTorchAcquisition` (generic wrapper)
- Unit tests for `BoTorchMCSampler`
- Integration tests with existing ALF components
- End-to-end tests on synthetic functions

## Overall Verification Status
- ✅ Phase 1 complete and verified (BoTorchQEI, utilities, synthetic dataset)
- ✅ Phase 1.5 complete and verified (Generic wrappers, GP model, samplers)
- ✅ All linting checks passing
- ✅ Demo script working
- ✅ Comprehensive documentation created
- 🔄 Tests needed for new components (Phase 2)
