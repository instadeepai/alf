---
name: split-plan
description: Split a feature plan into executable phases. Use when user says "split plan", "break into phases", "phase this plan", or wants to divide implementation work into stable, testable increments.
---

# Split Plan into Phases

Take a feature plan document and break it into discrete phases that can each be completed in a single session while keeping the system stable.

## Instructions

### Step 1: Read and Analyze the Plan

1. Read the provided plan file completely
2. Identify:
   - The feature name (from title or filename)
   - All implementation steps/tasks
   - Dependencies between tasks
   - Files to create vs files to modify
   - Testing requirements

### Step 2: Design Phases

Create phases following these principles:

**Phase Boundaries**
- Each phase must leave the system in a stable, working state
- All tests must pass at the end of each phase
- No broken imports or undefined references between phases

**Phase Contents**
- Each phase includes its own unit tests (tests are NOT a separate phase)
- Group related changes together (e.g., create module + its tests)
- Consider dependency order: create before consume
- Atomic migrations should stay in a single phase

**Phase Size**
- Target: completable in one focused session (2-4 hours of work)
- If a phase is too large, split it further
- If a phase is trivial, consider merging with adjacent phase

### Step 3: Create Output Structure

Create a directory under `plans/` with the feature name:

```
plans/
└── <feature-name>/
    ├── overview.md          # Summary and phase index
    ├── phase-1-<name>.md    # First phase details
    ├── phase-2-<name>.md    # Second phase details
    └── ...
```

### Step 4: Write Overview File

The `overview.md` file should contain:

```markdown
# <Feature Name> - Implementation Phases

**Source Plan**: [link to original plan]
**Total Phases**: N
**Created**: <date>

## Phase Summary

| Phase | Name | Description | Key Deliverables |
|-------|------|-------------|------------------|
| 1 | ... | ... | ... |
| 2 | ... | ... | ... |

## Dependencies

[Describe any cross-phase dependencies or ordering constraints]

## Verification

After all phases complete:
- [ ] All original plan items addressed
- [ ] Full test suite passes
- [ ] No temporary workarounds remain
```

### Step 5: Write Phase Files

Each `phase-N-<name>.md` file should contain:

```markdown
# Phase N: <Phase Name>

**Status**: Not Started | In Progress | Complete
**Estimated Scope**: <brief size indicator>

## Objective

<One paragraph describing what this phase accomplishes>

## Prerequisites

- [ ] Phase N-1 complete (if applicable)
- [ ] Any other prerequisites

## Tasks

### N.1 <First Task Group>
- [ ] Subtask 1
- [ ] Subtask 2

### N.2 <Second Task Group>
- [ ] Subtask 1
- [ ] Subtask 2

## Files to Create
| File | Purpose |
|------|---------|
| ... | ... |

## Files to Modify
| File | Changes |
|------|---------|
| ... | ... |

## Tests

<List specific tests to write/update in this phase>

## Verification

- [ ] All new tests pass
- [ ] Existing tests still pass
- [ ] No import errors
- [ ] <Any phase-specific verification>

## Notes

<Any additional context, gotchas, or decisions specific to this phase>
```

## Example Phase Split

For a plan with "Create utility module" and "Update consumers":

**Phase 1: Create Foundation**
- Create new utility module
- Write unit tests for new module
- Verify tests pass

**Phase 2: Migrate Consumers**
- Update all consumers to use new module
- Remove old implementation
- Update consumer tests
- Verify all tests pass

**Phase 3: Cleanup**
- Remove workarounds in dependent code
- Final verification

## Output

After running this skill:
1. Confirm the phase structure with the user
2. Create the `plans/<feature-name>/` directory
3. Write `overview.md` with phase summary
4. Write individual `phase-N-<name>.md` files
5. Report what was created
