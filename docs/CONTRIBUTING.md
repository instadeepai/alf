# Contributing to ALF

Thank you for your interest in contributing to ALF (Active Learning Framework)!

ALF's mission is to maximise information gained from multiple rounds of experimentation, applying active learning and Bayesian experimental design to accelerate scientific discovery. ALF targets domains where exploration is constrained by expensive data acquisition—wet-lab experiments, computational simulations, or physical measurements—and search spaces are high-dimensional or combinatorially vast. By contributing to ALF, you're helping researchers across computational biology, materials science, and chemistry optimize their experimental campaigns more efficiently.

This guide will help you get started with extending the framework and contributing code.

## Getting Started

### Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** installed on your system
- **Git** for version control
- **SSH key** configured for GitHub ([guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh))
- **uv** package manager ([installation guide](https://github.com/astral-sh/uv))

### Development Environment Setup

1. **Clone the repository:**
   ```bash
   git clone git@github.com:instadeepai/alf.git
   cd alf
   ```

2. **Create virtual environment and install dependencies:**
   ```bash
   # Install all packages with development dependencies
   uv sync
   ```

   For detailed installation instructions, including GPU support, see [docs/INSTALLATION.md](INSTALLATION.md).

3. **Install pre-commit hooks:**
   ```bash
   uv run pre-commit install
   ```

### Running Tests

Always run tests to ensure your changes don't break existing functionality:

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=alf_core --cov-report term-missing

# Run specific test file
uv run pytest core/alf_core/tests/test_specific.py
```

### Running Pre-commit Hooks

Pre-commit hooks automatically check code quality before commits:

```bash
# Run on all files
uv run pre-commit run --all-files

# Run on staged files only
uv run pre-commit run
```

### Understanding ALF Packages

ALF is organized into two packages to balance flexibility and usability:

- **alf-core** (`core/alf_core/`) - Framework backbone with base classes, core data structures, and minimal dependencies (no ML frameworks)
- **alf-tools** (`tools/alf_tools/`) - Ready-to-use implementations (models, datasets, acquisition functions) with heavier dependencies (PyTorch)

📖 **For detailed architecture and dependency information, see [INSTALLATION.md](INSTALLATION.md#architecture)**.

## Contributing Guidelines

**When adding new implementations:**
- ✅ Add to `alf-tools` for general-purpose implementations (new models, datasets, acquisition functions)
- ✅ Modify `alf-core` only for framework enhancements (new base classes, core utilities, task types)
- ✅ Keep alf-core dependencies minimal - propose heavy dependencies only if critical to the framework

**Examples:**
- Adding a new transformer model → Add to `tools/alf_tools/models/`
- Adding a new task type → Modify `core/alf_core/tasks/`
- Adding a new acquisition function → Add to `tools/alf_tools/optimizer/acquisition_functions/`
- Adding a new base class → Modify `core/alf_core/`

## Extending ALF Components

ALF is designed to be extendable. You can create custom implementations of core components by extending base classes. Below is a quick reference of extension points:

| Component | Base Class | Key Methods | Tutorial |
|-----------|------------|-------------|----------|
| **Models** | `BaseModel` | `featurise()`, `train()`, `predict()`, `sample()` | [Models](../tutorials/extending_base_classes/models.ipynb) |
| **Datasets** | `BaseDataset` | `load_dataset()`, `query()` | [Datasets](../tutorials/extending_base_classes/datasets.ipynb) |
| **Search Functions** | `BaseSearch` | `__call__()` | [Search Functions](../tutorials/extending_base_classes/search_functions.ipynb) |
| **Acquisition Functions** | `AcquisitionFunction` | `__call__()` | [Acquisition Functions](../tutorials/extending_base_classes/acquisition_functions.ipynb) |

### Quick Start: Extending a Component

1. **Choose the component** you want to extend (e.g., BaseModel, BaseDataset)
2. **Review the tutorial** linked in the table above
3. **Implement required methods** as shown in the tutorial
4. **Add your implementation** to the appropriate directory in `tools/alf_tools/`
5. **Write tests** for your implementation
6. **Run tests and pre-commit checks** to validate

Example directory structure for new implementations:
```
tools/alf_tools/
├── models/              # Custom model implementations
├── datasets/            # Custom dataset implementations
└── optimizer/           # Custom search/acquisition functions
```

#### Adding a new dataset

1. Create `tools/alf_tools/datasets/<name>.py` implementing `BaseDataset`.
   - `load_dataset()` must return `LabelledCandidates` with 1-D labels.
   - Download logic must check for a local cache before hitting the network.
2. Create `tools/tests/fixtures/<name>/` with synthetic fixture files that
   exercise all code paths (valid, invalid, empty, large samples).
   Run `generate.py` once and commit the output.
3. Create `tools/tests/datasets/test_<name>.py`. Mark every test with
   `pytestmark = pytest.mark.<name>` and run `pytest -m <name>` to verify
   no network calls are needed.
4. Register the marker in `tools/pyproject.toml` under `[tool.pytest.ini_options]`.

## Understanding Model Roles

Models in ALF can serve three distinct roles depending on how they're used in the active learning loop:

### Oracle
- **Purpose:** Evaluate candidates by providing ground truth scores
- **When to use:** When you require ground truth feedback during the optimization loop
- **Required methods:** `predict()`
- **Example use case:** Using a trained model, such as ESM-2, to score newly generated candidates

### Surrogate
- **Purpose:** A cheap-to-evaluate probabilistic approximation of the true objective function, used to predict values (and uncertainty) where the function hasn't been evaluated
- **When to use:** When evaluating the true objective function is expensive, slow, or limited, so you require a cheaper model to guide where to sample next
- **Required methods:** `train()`, `predict()`
- **Optional methods:** `get_training_summary_metrics()`
- **Example use case:** Training a Gaussian Process (GP) to approximate expensive wet-lab or computational experiments

### Generator
- **Purpose:** Propose new candidates to evaluate and explore a large combinatorial search space
- **When to use:** When you want to generate new candidates to evaluate under your surrogate model
- **Required methods:** `sample()`
- **Example use case:** Using a variational autoencoder to propose new protein sequences

For detailed examples of implementing models for each role, see the [Model Roles Tutorial](../tutorials/extending_base_classes/model_roles.ipynb).

## Code Contribution Workflow

### Testing Requirements

All contributions must include appropriate tests:

1. **Unit Tests**
   - Test individual functions and methods
   - Place in the same directory as the code: `tests/test_<module>.py`
   - Aim for high coverage of your new code

2. **Integration Tests**
   - Test how components work together
   - Ensure compatibility with existing code

3. **Test Guidelines**
   - Use descriptive test names: `test_acquisition_function_handles_empty_candidates()`
   - Include edge cases and error conditions
   - Mock external dependencies where appropriate
   - Ensure tests are deterministic (use fixed seeds)

### Code Style

ALF uses automated tools to maintain code quality:

- **Ruff** - Fast Python linter and formatter
- **MyPy** - Static type checker
- **Pre-commit hooks** - Automated checks before commits

Your code must pass all checks:

```bash
# These will run automatically on commit, but you can run manually:
uv run pre-commit run --all-files
```

**Style guidelines:**
- Follow PEP 8 conventions (enforced by ruff)
- Use type hints for function signatures
- Write clear, descriptive docstrings (Google style)
- Keep functions focused and modular
- Add comments for complex logic

### Pull Request Process

Before starting work on a new feature or bug fix, **tag a repo maintainer in the relevant issue** to discuss the approach and confirm it aligns with the project direction. This avoids duplicate effort and ensures your PR will be accepted.

1. **Create a feature branch:**
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes:**
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Verify your changes:**
   ```bash
   # Run tests
   uv run pytest

   # Run pre-commit checks
   uv run pre-commit run --all-files
   ```

4. **Commit your changes:**
   ```bash
   git add <changed files>
   git commit -m "feat: add your feature description"
   ```

5. **Push to your fork:**
   ```bash
   git push origin feat/your-feature-name
   ```

6. **Open a Pull Request:**
   - Go to the [ALF repository](https://github.com/instadeepai/alf)
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template with:
     - Description of changes
     - Related issues (if any)
     - Testing performed
     - Screenshots (if applicable)

7. **Address review feedback:**
   - Respond to reviewer comments
   - Make requested changes
   - Push updates to the same branch

### AI-Assisted PR Reviews

ALF uses [Claude Code](https://claude.ai/code) for AI-assisted pull request reviews. When you open a PR, Claude will automatically review your changes and leave inline comments. You are encouraged to respond to and address these comments as you would with any human reviewer.

### PR Review Checklist

Before submitting, ensure:

- [ ] Code follows the project's style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated (if needed)
- [ ] Commit messages follow conventional format
- [ ] No merge conflicts with main branch
- [ ] Pre-commit hooks pass

## Troubleshooting

### Common Setup Issues

#### Issue: `uv` command not found
**Solution:** Install uv following the [official instructions](https://github.com/astral-sh/uv):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Issue: Pre-commit hooks failing
**Solution:** Ensure pre-commit is installed and run on all files:
```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

If hooks continue to fail, check the specific error messages and fix the reported issues.

#### Issue: Import errors when running tests
**Solution:** Make sure you've synced all dependencies:
```bash
uv sync
```

And run tests using `uv run`:
```bash
uv run pytest
```

#### Issue: MyPy type checking errors
**Solution:** Ensure your code has proper type hints. For complex types, you may need to:
```python
from typing import Any, Union
from alf_core.dataclasses import Candidate, LabelledCandidates
```

Check the error message for specific type issues and add appropriate annotations.

#### Issue: CUDA/GPU errors during testing
**Solution:** If you don't have a GPU or CUDA installed, install the CPU-only version:
```bash
# See docs/INSTALLATION.md for GPU-specific setup
uv sync
```

#### Issue: Switching to GPU PyTorch
**Solution:** There are two options — see [GPU Support in INSTALLATION.md](INSTALLATION.md#gpu-support-optional) for full prerequisites and verification steps.

**Option A — Temporary override (no file changes):**
```bash
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu128 --reinstall
```
To revert to CPU, run `uv sync` again.

**Option B — Persistent change:** Edit the torch source index in `tools/pyproject.toml` from `pytorch-cpu` to `pytorch-gpu`, then run `uv sync`. Revert before committing.

#### Issue: Tests fail with "ModuleNotFoundError"
**Solution:** Ensure you're in the correct directory and have installed the package:
```bash
cd /path/to/alf
uv sync
uv run pytest
```

### Getting Help

If you encounter issues not covered here:

1. **Search existing issues:** Check [GitHub Issues](https://github.com/instadeepai/alf/issues) for similar problems
2. **Review documentation:** See the [full documentation](https://instadeepai.github.io/alf/)
3. **Open an issue:** Create a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Your environment details (OS, Python version, etc.)
   - Error messages and logs

## Additional Resources

- **Architecture Overview:** See [core/README.md](../core/README.md) for detailed framework design
- **API Documentation:** [https://instadeepai.github.io/alf/](https://instadeepai.github.io/alf/)
- **Understanding ALF Packages:** See [Package Architecture](../README.md#-package-architecture)
- **Tutorial Notebooks:**
  - [Offline Design Tutorial](../tutorials/experiments/offline_design_tutorial.ipynb)
  - [Online Design Tutorial](../tutorials/experiments/online_design_tutorial.ipynb)
  - [Extension Tutorials](../tutorials/extending_base_classes/)

Thank you for contributing to ALF! Your contributions help make active learning more accessible and powerful for everyone.
