# Create Command

## Overview
Interactive command creator that helps you build new Cursor commands. This command will guide you through the process by asking clarifying questions one at a time to ensure your command is well-defined and complete.

## Process

When you use this command, I will:

1. **Analyze your input**: Review the prompt you provide about the command you want to create
2. **Ask clarifying questions**: If any aspect is unclear or missing, I'll ask one question at a time, waiting for your response before moving to the next
3. **Create the command file**: Once all details are clear, I'll create a properly formatted `.md` file in the `.cursor/commands` directory

## Information I'll gather

I'll ask about (one at a time if needed):
- **Command name**: What should the command be called? (e.g., `review-code`, `setup-feature`)
- **Purpose**: What is the main goal of this command?
- **Steps/Workflow**: What specific steps or actions should this command perform?
- **Context needed**: What information or context should the command require or use?
- **Output/Result**: What should the command produce or accomplish?

## Usage

Type `/create-command` followed by a description of the command you want to create. For example:

```
/create-command I want a command that reviews my code for security issues
```

or

```
/create-command Create a command for setting up new React components
```

If your description is vague or missing details, I'll ask you one clarifying question at a time until I have enough information to create a complete, useful command.

## Example Flow

**You**: `/create-command I want a command for code review`

**Me**: "What specific aspects of code should this review focus on? (e.g., security, performance, style, functionality, or all of the above?)"

**You**: "Security and performance"

**Me**: "Should this command automatically fix issues it finds, or just report them?"

**You**: "Just report them"

**Me**: [Creates the command file with all the gathered information]

---

**Note**: After I ask each clarifying question, please respond with your answer, and I'll continue with the next question or proceed to create the command if everything is clear.
