# Feature Plan: Contributing Documentation & Extension Tutorials

**Created**: 2026-01-30
**Branch**: docs-contributing-guide
**Status**: Approved

## Summary

Create comprehensive contributing documentation that enables developers to extend ALF's base classes. This includes:
1. A technical CONTRIBUTING.md guide in the docs folder
2. Five focused tutorial notebooks showing how to extend base classes
3. Updated main README linking to all new documentation

## Decisions Made

| Decision | Selected Option | Rationale |
|----------|-----------------|-----------|
| Tutorial depth | Minimal working examples | Experienced users need quick reference, not extensive tutorials. Keep notebooks ~50-100 lines with essential code. |
| Contribution scope | Technical only | Focus on code contributions: setup, extending classes, testing, PR process. Skip CoC/community guidelines. |
| Notebook organization | Separate notebooks per component | 5 focused notebooks easier to navigate and discover than one large file. |
| Architecture docs | Keep core/README.md in place | Existing location is fine; just improve discoverability through better linking. |

## Technical Approach

### 1. CONTRIBUTING.md (docs/CONTRIBUTING.md)

A technical contribution guide structured as:

**Section A: Getting Started**
- Prerequisites (Python 3.10+, git, uv)
- Development environment setup (reference docs/INSTALLATION.md)
- Running tests and pre-commit hooks

**Section B: Extending ALF Components**
- Overview of extension points with links to tutorial notebooks
- Quick reference table:
  | Component | Base Class | Key Methods | Tutorial |
  |-----------|------------|-------------|----------|
  | Models | BaseModel | train(), predict(), sample() | tutorials/extending/extending_models.ipynb |
  | Datasets | BaseDataset | load_dataset(), query() | tutorials/extending/extending_datasets.ipynb |
  | Search Functions | BaseSearch | __call__() | tutorials/extending/extending_search_functions.ipynb |
  | Acquisition Functions | AcquisitionFunction | __call__() | tutorials/extending/extending_acquisition_functions.ipynb |

**Section C: Understanding Model Roles**
- Oracle: Online evaluation of candidates
- Surrogate: Approximating expensive evaluations
- Generator: Sampling new candidates
- Link to tutorials/extending/model_roles.ipynb

**Section D: Code Contribution Workflow**
- Branch naming conventions (feat/, fix/, docs/, refactor/)
- Commit message format
- Testing requirements (unit tests, integration tests, coverage)
- Code style (ruff, mypy, pre-commit)
- PR submission and review process

### 2. Tutorial Directory Restructure

Reorganize the tutorials directory into two clear sections:

**New Structure:**
```
tutorials/
├── pyproject.toml            # Dependencies for all tutorials (stays in root)
├── experiments/              # End-to-end experiment tutorials
│   ├── offline_design_tutorial.ipynb  (existing - move here)
│   ├── online_design_tutorial.ipynb   (existing - move here)
│   ├── results/              (existing - move here)
│   └── structures/           (existing - move here)
└── extending/                # Extension tutorials for developers
    ├── extending_models.ipynb
    ├── extending_datasets.ipynb
    ├── extending_search_functions.ipynb
    ├── extending_acquisition_functions.ipynb
    └── model_roles.ipynb
```

### 3. Extension Tutorial Notebooks (tutorials/extending/)

Create 5 new notebooks, each following this minimal template:

```python
# 1. Imports (2-3 lines)
# 2. Define custom class (10-20 lines)
# 3. Configuration example (5-10 lines)
# 4. Usage example (10-20 lines)
# 5. Key points summary (3-5 bullet points)
```

#### A. `tutorials/extending/extending_models.ipynb`
- Extend BaseModel with a simple custom model
- Implement required methods: `featurise()`, `train()`, `predict()`
- Optional methods: `sample()` for generative models
- Show BaseModelConfig usage
- Example: Simple polynomial regression model

#### B. `tutorials/extending/extending_datasets.ipynb`
- Extend BaseDataset for custom data sources
- Implement required method: `load_dataset()`
- Show BaseDatasetConfig with modality selection
- Demonstrate data splitting and querying
- Example: CSV dataset with custom features

#### C. `tutorials/extending/extending_search_functions.ipynb`
- Extend BaseSearch for custom search strategies
- Implement `__call__(task_state)` returning candidates
- Show integration with task state
- Example: Random search and grid search

#### D. `tutorials/extending/extending_acquisition_functions.ipynb`
- Extend AcquisitionFunction for custom acquisition strategies
- Implement `__call__(search_candidates, state)` returning scored candidates
- Show how to access predictions from task state
- Example: Simple uncertainty sampling and diversity-based acquisition

#### E. `tutorials/extending/model_roles.ipynb`
- Demonstrate the three model roles: Oracle, Surrogate, Generator
- For each role:
  - When to use it
  - How to wrap your custom model
  - Required methods
  - Usage example (5-10 lines each)
- Comparison table showing which methods are needed per role:
  | Role | Required Methods | Optional Methods | Use Case |
  |------|------------------|------------------|----------|
  | Oracle | predict() | - | Online evaluation |
  | Surrogate | train(), predict() | get_training_summary_metrics() | Approximate expensive functions |
  | Generator | sample() | - | Generate new candidates |

### 4. README Updates (README.md)

Update the main README to link to new documentation:

**Location 1: Documentation Section (around line 42-46)**
```markdown
## Documentation

- 📖 [Full Documentation](https://instadeepai.github.io/alf/)
- 🛠️ [Contributing Guide](docs/CONTRIBUTING.md) - How to extend ALF and contribute code
- 📥 [Installation Guide](docs/INSTALLATION.md)
```

**Location 2: Tutorials Section (around line 127-129)**
Replace existing tutorials section with reorganized structure:
```markdown
## Tutorials

### Experiment Tutorials

End-to-end guides for running active learning experiments:
- [Offline Design Tutorial](tutorials/experiments/offline_design_tutorial.ipynb) - Dataset-based optimization
- [Online Design Tutorial](tutorials/experiments/online_design_tutorial.ipynb) - Model-based optimization

### Extension Tutorials

Learn how to extend ALF's base classes for custom implementations:
- [Extending Models](tutorials/extending/extending_models.ipynb) - Create custom models for oracle/surrogate/generator roles
- [Extending Datasets](tutorials/extending/extending_datasets.ipynb) - Add custom data sources
- [Extending Search Functions](tutorials/extending/extending_search_functions.ipynb) - Implement custom search strategies
- [Extending Acquisition Functions](tutorials/extending/extending_acquisition_functions.ipynb) - Create custom acquisition strategies
- [Understanding Model Roles](tutorials/extending/model_roles.ipynb) - Oracle, Surrogate, and Generator patterns
```

**Location 3: Contributing Section (around line 167-174)**
Replace existing content with:
```markdown
## Contributing

We welcome contributions! To get started:

1. Read our [Contributing Guide](docs/CONTRIBUTING.md) for development setup and guidelines
2. Check out the [Extension Tutorials](tutorials/extending/) to learn how to extend ALF's base classes

For questions or discussions, please open an issue.
```

## Implementation Steps

### Phase 1: Restructure Tutorials Directory
1. [ ] Create `tutorials/experiments/` directory
2. [ ] Create `tutorials/extending/` directory
3. [ ] Move `tutorials/offline_design_tutorial.ipynb` to `tutorials/experiments/`
4. [ ] Move `tutorials/online_design_tutorial.ipynb` to `tutorials/experiments/`
5. [ ] Move `tutorials/results/` directory to `tutorials/experiments/`
6. [ ] Move `tutorials/structures/` directory to `tutorials/experiments/`

### Phase 2: Create CONTRIBUTING.md
8. [ ] Create `docs/CONTRIBUTING.md` with sections A-D (without Section E)
9. [ ] Add links to tutorial notebooks in extending/ folder
10. [ ] Include code examples for branch naming, commit messages
11. [ ] Add troubleshooting section for common setup issues

### Phase 3: Create Extension Tutorial Notebooks
12. [ ] Create `tutorials/extending/extending_models.ipynb` with minimal working example
13. [ ] Create `tutorials/extending/extending_datasets.ipynb` with CSV dataset example
14. [ ] Create `tutorials/extending/extending_search_functions.ipynb` with search examples
15. [ ] Create `tutorials/extending/extending_acquisition_functions.ipynb` with acquisition examples
16. [ ] Create `tutorials/extending/model_roles.ipynb` with oracle/surrogate/generator patterns

### Phase 4: Update Main README
17. [ ] Update Documentation section (remove core/README.md link)
18. [ ] Replace Tutorials section with new structure (Experiments + Extensions)
19. [ ] Update Contributing section (remove core/README.md link)
20. [ ] Verify all links are correct and files exist

### Phase 5: Testing & Polish
21. [ ] Run all tutorial notebooks to ensure they execute without errors
22. [ ] Check markdown formatting and links in CONTRIBUTING.md
23. [ ] Verify main README renders correctly on GitHub
24. [ ] Update any internal links that reference old tutorial paths
25. [ ] Run pre-commit hooks to ensure code style compliance

## Files to Create

- `docs/CONTRIBUTING.md` - Technical contribution guide without Section E (~250-350 lines)
- `tutorials/extending/extending_models.ipynb` - Model extension tutorial (~50-80 lines)
- `tutorials/extending/extending_datasets.ipynb` - Dataset extension tutorial (~50-80 lines)
- `tutorials/extending/extending_search_functions.ipynb` - Search function tutorial (~50-80 lines)
- `tutorials/extending/extending_acquisition_functions.ipynb` - Acquisition function tutorial (~50-80 lines)
- `tutorials/extending/model_roles.ipynb` - Model roles explanation (~80-100 lines)

## Directories to Create

- `tutorials/experiments/` - For existing experiment tutorials
- `tutorials/extending/` - For new extension tutorials

## Files to Move

- `tutorials/offline_design_tutorial.ipynb` → `tutorials/experiments/offline_design_tutorial.ipynb`
- `tutorials/online_design_tutorial.ipynb` → `tutorials/experiments/online_design_tutorial.ipynb`
- `tutorials/results/` → `tutorials/experiments/results/`
- `tutorials/structures/` → `tutorials/experiments/structures/`

Note: `tutorials/pyproject.toml` stays in the tutorials root directory.

## Files to Modify

- `README.md` - Update Documentation, Tutorials, and Contributing sections (~40 lines changed)

## Dependencies

None - all work uses existing dependencies and documentation tooling.

## Testing Strategy

1. **Notebook Execution**: Run each notebook in a fresh environment to verify:
   - All imports work
   - Code executes without errors
   - Examples produce expected output

2. **Link Validation**: Check all markdown links:
   - Internal links (to files in repo)
   - External links (to hosted docs)
   - Relative paths work from different locations

3. **Documentation Review**:
   - Verify CONTRIBUTING.md is clear and actionable
   - Check that tutorial notebooks are concise (<100 lines)
   - Ensure README updates flow naturally

4. **Integration Test**: Follow the contributing guide as a new developer would:
   - Clone repo
   - Follow setup instructions
   - Run a tutorial notebook
   - Verify the workflow makes sense

## Rollback Plan

If issues arise, this branch can be easily abandoned or cherry-picked:
- All changes are in new files (5 notebooks + CONTRIBUTING.md)
- README changes are isolated to 3 small sections
- No existing functionality is modified
- Easy to revert individual files if needed

## Notes

### Key Design Principles

1. **Minimal Examples**: Tutorials are concise reference material, not comprehensive guides
2. **Clear Navigation**: Separate notebooks make it easy to find specific extension patterns
3. **Link Everything**: CONTRIBUTING.md, README, and core/README.md form a web of documentation
4. **Technical Focus**: Documentation targets developers who want to extend ALF, not end users

### Content Sources

- Base class information from: `core/alf_core/` (model, dataset, optimizer)
- Architecture details from: `core/README.md` (comprehensive existing documentation)
- Code examples from: `tools/alf_tools/` (existing implementations as reference)
- Tutorial structure from: existing `tutorials/offline_design_tutorial.ipynb`

### Future Enhancements (Not in Scope)

- Video tutorials or screencasts
- Community guidelines and Code of Conduct
- Detailed API reference (already covered by Sphinx docs)
- Advanced optimization techniques guide
- Performance tuning documentation
