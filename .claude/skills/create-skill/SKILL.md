---
name: create-skill
description: Creates new Claude Code skills. Use when the user wants to create a skill, make a new skill, or asks about skill creation.
---

# Create Skill

Help the user create a new Claude Code skill by following this process:

## Step 1: Gather Requirements

Ask the user:
1. **Skill name**: What should the skill be called? (lowercase, hyphens, max 64 chars)
2. **Purpose**: What should this skill do?
3. **Trigger conditions**: When should Claude use this skill?
4. **Scope**: Personal (`~/.claude/skills/`) or project (`.claude/skills/`)?

## Step 2: Create the Skill Structure

### Basic Structure
```
skill-name/
└── SKILL.md
```

### Multi-file Structure (for complex skills)
```
skill-name/
├── SKILL.md              # Main instructions (keep under 500 lines)
├── reference.md          # Detailed documentation
├── examples.md           # Usage examples
└── scripts/
    └── helper.py         # Utility scripts
```

## Step 3: Write SKILL.md

Use this template:

```yaml
---
name: skill-name
description: Brief description of what this skill does and when to use it. Include trigger keywords users would say.
---

# Skill Title

## Instructions

[Clear, step-by-step guidance for Claude]

## Examples

[Concrete usage examples]
```

### Available Metadata Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Lowercase letters, numbers, hyphens only (max 64 chars) |
| `description` | Yes | What it does + when to use it (max 1024 chars) |
| `allowed-tools` | No | Restrict tools: `Read, Grep, Glob` or YAML list |
| `model` | No | Specific model: `claude-sonnet-4-20250514` |
| `context` | No | Set to `fork` for isolated sub-agent context |
| `agent` | No | Agent type when `context: fork`: `Explore`, `Plan`, `general-purpose` |
| `hooks` | No | Lifecycle hooks: `PreToolUse`, `PostToolUse`, `Stop` |
| `user-invocable` | No | Hide from slash menu: `false` |

## Step 4: Write an Effective Description

Good descriptions answer:
1. **What does this skill do?** List specific capabilities.
2. **When should Claude use it?** Include trigger keywords.

**Bad**: `Helps with documents`
**Good**: `Extract text and tables from PDF files, fill forms, merge documents. Use when working with PDF files or when the user mentions PDFs, forms, or document extraction.`

## Step 5: Create the Skill

Execute these commands based on scope:

### Personal Skill (available across all projects)
```bash
mkdir -p ~/.claude/skills/SKILL_NAME
```

### Project Skill (shared with team via version control)
```bash
mkdir -p .claude/skills/SKILL_NAME
```

Then write the SKILL.md file to the created directory.

## Step 6: Verify

Ask Claude: "What Skills are available?" to confirm the skill loaded.

## Advanced Options

### Restrict Tool Access
```yaml
allowed-tools: Read, Grep, Glob
```

### Run in Forked Context
```yaml
context: fork
agent: Explore
```

### Add Lifecycle Hooks
```yaml
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/check.sh $TOOL_INPUT"
```

### Progressive Disclosure
Keep SKILL.md focused. Link to supporting files:
```markdown
For detailed API reference, see [reference.md](reference.md).
```

## Documentation Reference

For more details, fetch these URLs:
- Skills overview: https://code.claude.com/docs/en/skills.md
- Hooks reference: https://code.claude.com/docs/en/hooks.md
- Sub-agents: https://code.claude.com/docs/en/sub-agents.md
- Plugins: https://code.claude.com/docs/en/plugins.md
