---
name: plan-feature
description: Plan new features by deeply exploring the codebase, understanding patterns, and asking clarifying questions. Use when the user wants to plan a feature, design a new capability, or needs help understanding how to implement something new in the codebase.
---

# Plan Feature

A methodical feature planning skill that explores the codebase, asks clarifying questions one at a time, presents options with pros/cons, and produces an approved implementation plan.

## Core Principles

1. **Never assume** - Always ask when uncertain
2. **One question at a time** - Don't overwhelm the user
3. **Present options clearly** - Show pros/cons and recommend one
4. **Iterate on feedback** - Refine until approved
5. **Document the plan** - Save approved plans to `plans/` folder

## Process

### Phase 1: Initial Understanding

1. Read the feature description provided by the user
2. Use the Task tool with `subagent_type: Explore` to deeply explore the codebase:
   - Identify related existing patterns
   - Find similar implementations
   - Understand the architecture
   - Locate relevant files and modules

### Phase 2: Clarifying Questions

Ask clarifying questions **one at a time** using the `AskUserQuestion` tool:

- Start with the most fundamental questions first
- Wait for each answer before asking the next
- Continue until you have full clarity on:
  - Scope and boundaries of the feature
  - Expected behavior and edge cases
  - Integration points with existing code
  - Performance or scalability requirements
  - Any constraints or preferences

Example flow:
```
Question 1: "What is the primary use case for this feature?"
[Wait for answer]
Question 2: "Should this integrate with X or Y system?"
[Wait for answer]
...continue as needed
```

### Phase 3: Options Presentation

When multiple valid approaches exist:

1. Present options **one decision at a time**
2. For each option, clearly show:
   - **Description**: What this approach entails
   - **Pros**: Benefits and advantages
   - **Cons**: Drawbacks and tradeoffs
   - **Recommendation**: Mark the suggested option with "(Recommended)"
3. Wait for user selection before moving to the next decision

Use the `AskUserQuestion` tool with options structured like:
```
Option A: [Name] (Recommended)
  - Pros: ...
  - Cons: ...

Option B: [Name]
  - Pros: ...
  - Cons: ...
```

### Phase 4: Plan Compilation

Once all questions are answered and options selected, compile the implementation plan:

1. **Summary**: Brief overview of the feature
2. **Technical Approach**: Selected architecture and patterns
3. **Implementation Steps**: Ordered list of tasks
4. **Files to Modify/Create**: Specific file paths
5. **Dependencies**: Any new dependencies needed
6. **Testing Strategy**: How to verify the implementation
7. **Rollback Plan**: How to undo if needed

### Phase 5: Approval Loop

Use the `AskUserQuestion` tool to get approval:

```
Question: "Do you approve this plan, or do you have feedback to refine it?"
Options:
- "Approve plan"
- "I have feedback"
```

If feedback is provided:
1. Incorporate the feedback
2. Update the plan
3. Present the revised plan
4. Ask for approval again
5. Repeat until approved

### Phase 6: Save the Plan

Once approved:

1. Create the `plans/` directory if it doesn't exist
2. Save the plan as a markdown file: `plans/[feature-name].md`
3. Use a descriptive filename based on the feature
4. Include a timestamp in the document

## Plan Document Template

```markdown
# Feature Plan: [Feature Name]

**Created**: [Date]
**Status**: Approved

## Summary

[Brief description of the feature]

## Decisions Made

| Decision | Selected Option | Rationale |
|----------|-----------------|-----------|
| ... | ... | ... |

## Technical Approach

[Detailed technical approach based on selected options]

## Implementation Steps

1. [ ] Step 1
2. [ ] Step 2
...

## Files to Modify

- `path/to/file.py` - [what changes]
- ...

## Files to Create

- `path/to/new/file.py` - [purpose]
- ...

## Dependencies

- [Any new packages or dependencies]

## Testing Strategy

- [How to test the feature]

## Notes

[Any additional context or considerations]
```

## Important Reminders

- Use `Task` tool with `subagent_type: Explore` for codebase exploration
- Use `AskUserQuestion` for all user interactions
- Always wait for user response before proceeding
- Keep questions focused and one at a time
- Present options with clear pros/cons
- Mark recommended options explicitly
- Iterate on feedback until explicit approval
- Save final plan to `plans/` folder
