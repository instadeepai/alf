---
name: implement-phase
description: Implements the next phase from a plan folder. Reads overview.md and progress.md to determine current phase, implements it with tests, runs quality checks ensuring zero errors, and updates progress.md. Use when user says "implement phase", "run next phase", "execute phase", or provides a plan folder path.
---

# Implement Phase

Implement the next phase from a structured plan folder.

## Instructions

### Step 1: Understand the Plan Context

1. **Read the overview file** (`overview.md`) in the provided plan folder to understand:
   - Total number of phases
   - Phase dependencies
   - Key architectural notes
   - Overall verification criteria

2. **Read progress.md** (if it exists) in the plan folder to understand:
   - Which phases have been completed
   - Key learnings from previous phases
   - Any issues or notes to carry forward

3. **Determine the current phase** to implement:
   - If no progress.md exists, start with Phase 1
   - Otherwise, implement the next incomplete phase based on progress.md
   - Respect phase dependencies from the overview

### Step 2: Read the Phase Details

1. Read the phase file (e.g., `phase-1-create-mask-utilities.md`)
2. Understand:
   - Objective and prerequisites
   - All tasks with their checkboxes
   - Files to create/modify
   - Implementation details and code snippets
   - Verification criteria

### Step 3: Implement the Phase

1. **Create a todo list** tracking all tasks from the phase file
2. **Implement each task** following the phase's implementation details exactly
3. **Write tests** for every piece of functionality you implement
4. Mark todos as completed as you finish each task

### Step 4: Quality Checks (CRITICAL)

This step is non-negotiable. ALL checks must pass with ZERO errors.

1. **Run pre-commit checks** on all files:
   ```bash
   pre-commit run --all-files
   ```
   This runs the configured hooks (ruff linter and ruff-format) on the entire codebase.

2. **Run the full test suite**:
   ```bash
   pytest core/flow/tests/ core/model/tests/ -v
   ```

3. **Check for any pre-existing errors**:
   - If you find errors from previous code, YOU MUST FIX THEM
   - Do not proceed until all errors are resolved
   - This includes errors not related to your changes

4. **Verify all checks pass**:
   - All pre-commit hooks must pass
   - Every existing test must still pass
   - All new tests must pass
   - No exceptions, no skipped tests, no warnings treated as acceptable

### Step 5: Update Progress File

Create or update `progress.md` in the plan folder with:

```markdown
# Implementation Progress

## Completed Phases

### Phase X: [Phase Name]
**Completed**: [Date]
**Status**: Complete

#### Changes Made
- [List of files created/modified]

#### Key Learnings
- [Important discoveries during implementation]
- [Gotchas or tricky parts]
- [Decisions made and why]

#### Notes for Future Phases
- [Dependencies created]
- [Things to watch out for]
- [Patterns established]

## Next Phase
Phase Y: [Name] - Ready to implement

## Verification Status
- [ ] All pre-commit checks passing
- [ ] All tests passing
```

### Step 6: Final Report

Provide a summary to the user:
- What was implemented
- Tests added and their results
- Quality check results
- Any issues encountered and how they were resolved
- What's next

## Important Rules

1. **DO NOT COMMIT** - The user explicitly does not want commits after implementation
2. **ZERO TOLERANCE FOR ERRORS** - All quality checks must pass with no errors
3. **FIX PRE-EXISTING ERRORS** - If you find errors in existing code, fix them
4. **TEST EVERYTHING** - Every new functionality needs tests
5. **ALL TESTS MUST PASS** - No exceptions, including existing tests
6. **UPDATE PROGRESS.MD** - Always update after completing a phase
7. **NEVER SKIP TESTS** - This is CRITICAL:
   - NEVER use `--ignore`, `--deselect`, or any flag to skip tests
   - NEVER exclude test files or directories from the test run
   - If tests fail due to missing dependencies, INSTALL THE DEPENDENCIES
   - If tests fail due to environment issues, FIX THE ENVIRONMENT
   - If tests are truly broken and unrelated to your work, FIX THEM ANYWAY
   - The only acceptable outcome is ALL tests passing with zero exclusions

## Error Handling

If you encounter errors during quality checks:

1. **Identify the source** - Is it from your changes or pre-existing?
2. **Fix ALL errors** - Both new and pre-existing
3. **Re-run quality checks** - Verify fixes work
4. **Document in progress.md** - Note any pre-existing issues you fixed

## Example Usage

User: "Implement the next phase from plans/consistent-mask-dimensions/"

1. Read `plans/consistent-mask-dimensions/overview.md`
2. Read `plans/consistent-mask-dimensions/progress.md` (if exists)
3. Determine current phase (e.g., Phase 1)
4. Read `plans/consistent-mask-dimensions/phase-1-create-mask-utilities.md`
5. Implement all tasks with tests
6. Run pre-commit and pytest - fix any errors
7. Update `plans/consistent-mask-dimensions/progress.md`
8. Report results (no commit)
