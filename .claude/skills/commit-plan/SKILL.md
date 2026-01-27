---
name: commit-plan
description: Create a feature branch, stage and commit changes, push to remote, and create a GitHub PR with a description linking to an issue. Use when the user says "commit plan", "create PR", "push feature", "submit changes", or wants to create a pull request from a plan.
---

# Commit Plan

Create a feature branch, commit changes, push to remote, and open a GitHub PR linked to an issue.

## Instructions

### Step 1: Gather Information

Ask the user for:
1. **Feature name** - Used for branch name (will be formatted as `feature/<name>`)
2. **Issue number** - GitHub issue to link in the PR (e.g., `#123`)
3. **Commit message** - Brief description of the changes (optional, can be auto-generated)

### Step 2: Create and Switch to Feature Branch

```bash
git checkout -b feature/<feature-name>
```

If the branch already exists, ask the user whether to use it or create a new name.

### Step 3: Stage Changes

Review unstaged changes:
```bash
git status
git diff
```

Stage all relevant changes:
```bash
git add <files>
```

Or stage all changes if appropriate:
```bash
git add -A
```

### Step 4: Commit Changes

Create a commit with a descriptive message:
```bash
git commit -m "$(cat <<'EOF'
<commit message>

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

### Step 5: Push to Remote

Push the branch to origin with upstream tracking:
```bash
git push -u origin feature/<feature-name>
```

### Step 6: Create Pull Request

Use GitHub CLI to create the PR with a well-structured description:

```bash
gh pr create --title "<PR title>" --body "$(cat <<'EOF'
## Summary

<Brief description of what this PR does>

## Changes

- <List of key changes>

## Related Issue

Closes #<issue-number>

---
Generated with [Claude Code](https://claude.ai/code)
EOF
)"
```

### Step 7: Report Success

Provide the user with:
- The PR URL
- Branch name
- Linked issue number

## Example Usage

User: "commit plan for the new authentication feature, links to issue #42"

Claude will:
1. Create branch `feature/authentication`
2. Stage and commit changes
3. Push to origin
4. Create PR linking to #42
5. Return the PR URL

## Notes

- Always verify there are changes to commit before proceeding
- If no issue number is provided, ask the user or create PR without the "Closes" reference
- Use meaningful commit messages that describe the "why" not just the "what"
- Never force push or use destructive git commands without explicit user consent
