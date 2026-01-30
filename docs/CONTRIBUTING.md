# Contributing to ALF

Thank you for your interest in contributing to ALF (Active Learning Framework)! This guide will help you get started with extending the framework and contributing code.

## Getting Started

### Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** installed on your system
- **Git** for version control
- **uv** package manager ([installation guide](https://github.com/astral-sh/uv))

### Development Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/instadeepai/alf.git
   cd alf
   ```

2. **Install dependencies:**
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

## Extending ALF Components

ALF is designed to be extensible. You can create custom implementations of core components by extending base classes. Below is a quick reference of extension points:

| Component | Base Class | Key Methods | Tutorial |
|-----------|------------|-------------|----------|
| **Models** | `BaseModel` | `featurise()`, `train()`, `predict()`, `sample()` | [Extending Models](../tutorials/extending/extending_models.ipynb) |
| **Datasets** | `BaseDataset` | `load_dataset()`, `query()` | [Extending Datasets](../tutorials/extending/extending_datasets.ipynb) |
| **Search Functions** | `BaseSearch` | `__call__()` | [Extending Search Functions](../tutorials/extending/extending_search_functions.ipynb) |
| **Acquisition Functions** | `AcquisitionFunction` | `__call__()` | [Extending Acquisition Functions](../tutorials/extending/extending_acquisition_functions.ipynb) |

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

## Understanding Model Roles

Models in ALF can serve three distinct roles depending on how they're used in the active learning loop:

### Oracle
- **Purpose:** Evaluate candidates online (during the learning loop)
- **When to use:** When you need real-time scoring of candidates
- **Required methods:** `predict()`
- **Example use case:** Using a trained model to score generated candidates

### Surrogate
- **Purpose:** Approximate expensive evaluation functions
- **When to use:** When direct evaluation is too costly and you need a fast approximation
- **Required methods:** `train()`, `predict()`
- **Optional methods:** `get_training_summary_metrics()`
- **Example use case:** Training a neural network to approximate expensive simulations

### Generator
- **Purpose:** Generate new candidate points to explore
- **When to use:** When the search space is continuous or you want to generate novel candidates
- **Required methods:** `sample()`
- **Example use case:** Using a generative model to propose new molecular structures

For detailed examples of implementing models for each role, see the [Model Roles Tutorial](../tutorials/extending/model_roles.ipynb).

## Code Contribution Workflow

### Branch Naming Conventions

Use descriptive branch names with prefixes:

- `feat/` - New features (e.g., `feat/add-uncertainty-sampling`)
- `fix/` - Bug fixes (e.g., `fix/acquisition-function-nan`)
- `docs/` - Documentation updates (e.g., `docs/update-api-reference`)
- `refactor/` - Code refactoring (e.g., `refactor/simplify-dataset-loading`)
- `test/` - Test additions/updates (e.g., `test/add-model-integration-tests`)
- `perf/` - Performance improvements (e.g., `perf/optimize-batch-prediction`)

### Commit Message Format

This project uses [Conventional Commits](https://www.conventionalcommits.org/). Each commit message must follow this format:

```
<type>: <description>

[optional body]

[optional footer]
```

**Allowed types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `style` - Code style changes (formatting, no logic change)
- `refactor` - Code refactoring
- `perf` - Performance improvements
- `test` - Adding or updating tests
- `build` - Build system changes
- `ci` - CI/CD changes
- `chore` - Other changes (dependencies, configs)
- `revert` - Revert a previous commit

**Examples:**
```bash
feat: add Thompson sampling acquisition function

fix: handle NaN values in surrogate predictions

docs: update extending models tutorial with edge cases

test: add unit tests for dataset splitting logic
```

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
   git add .
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
- **Tutorial Notebooks:**
  - [Offline Design Tutorial](../tutorials/experiments/offline_design_tutorial.ipynb)
  - [Online Design Tutorial](../tutorials/experiments/online_design_tutorial.ipynb)
  - [Extension Tutorials](../tutorials/extending/)

Thank you for contributing to ALF! Your contributions help make active learning more accessible and powerful for everyone.
