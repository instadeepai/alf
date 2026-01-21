# Create Rule

## Overview
Interactive rule creator that helps you build new Cursor rules. This command will guide you through the configuration process step-by-step, presenting available options and collecting your choices to create a properly formatted rule.

You can create rules in two ways:
- **Manual entry**: Provide rule content directly through the interactive prompts
- **URL fetch**: Provide a URL and I'll fetch the content, analyze it, and generate comprehensive rules based on the fetched material (documentation, style guides, API specs, etc.)

## Process

When you use this command, I will:

1. **Determine input method**: Ask if you want to provide content manually or via URL
2. **Gather rule information**:
   - If manual: Ask you about the rule's purpose and content
   - If URL: Fetch and analyze the URL content, then generate comprehensive rules
3. **Configure rule type**: Present the four rule types and ask you to choose one
4. **Collect type-specific settings**: Based on your choice, ask for additional configuration (description, file patterns, etc.)
5. **Create the rule**: Generate the rule folder structure with `RULE.md` file in `.cursor/rules`

## Rule Configuration Options

### Rule Types (you'll choose one):

1. **Always Apply**
   - Rule is applied to every chat session
   - Sets `alwaysApply: true`
   - No additional configuration needed

2. **Apply Intelligently**
   - Agent decides when rule is relevant based on description
   - Requires a description field
   - Sets `alwaysApply: false` with a description

3. **Apply to Specific Files**
   - Rule applies when files match specified patterns
   - Requires glob patterns (e.g., `*.ts`, `**/components/**`)
   - Sets `globs` array in frontmatter

4. **Apply Manually**
   - Rule is applied when @-mentioned in chat (e.g., `@my-rule`)
   - Sets `alwaysApply: false`
   - No globs or description needed

## Information I'll gather

I'll ask about (one at a time, in order):

1. **Input method**: Would you like to provide the rule content manually or via URL?
2. **Rule name**: What should this rule be called? (will be used as folder name)
3. **Rule content or URL**:
   - If manual: What are the actual instructions/content for this rule?
   - If URL: What URL should I fetch? (I'll analyze the content and generate comprehensive rules)
4. **Rule purpose** (if URL): What is this rule for? (brief description - I may infer this from the URL content)
5. **Rule type**: Which type should this rule be? (I'll present the 4 options)
6. **Type-specific config**:
   - If "Apply Intelligently": Ask for description
   - If "Apply to Specific Files": Ask for glob patterns
   - If "Always Apply" or "Apply Manually": No additional config needed

## URL-Based Rule Generation

When you provide a URL, I will:

1. **Fetch the content**: Retrieve the webpage, documentation, or file from the URL
2. **Analyze the content**: Understand the structure, purpose, and key information
3. **Generate comprehensive rules**: Create well-structured rule content that captures:
   - Key principles and guidelines from the source
   - Best practices and standards
   - Specific requirements and constraints
   - Examples and patterns when applicable
4. **Present for review**: Show you the generated rule content for approval or modification before creating the rule file

**Supported URL types:**
- Documentation websites (GitHub docs, API docs, style guides)
- Markdown files (GitHub raw files, Gists)
- Text-based content (articles, blog posts, specifications)
- Code repositories (README files, contributing guides)

**Note**: I'll do my best to extract relevant information, but you may want to review and refine the generated content to ensure it matches your specific needs.

## Usage

Type `/create-rule` to start the interactive rule creation process. I'll guide you through each step.

Example:
```
/create-rule
```

Then I'll ask:
- "Would you like to provide the rule content manually or via URL? (Type 'manual' or 'url')"
- "What should this rule be called?"
- "What are the instructions/content for this rule?" (if manual) OR "What URL should I fetch?" (if URL)
- "What is the purpose of this rule?" (if URL, I may infer this)
- "Which rule type would you like? (1) Always Apply, (2) Apply Intelligently, (3) Apply to Specific Files, (4) Apply Manually"
- [Additional questions based on your choice]

**URL Example:**
```
/create-rule
```
Then:
- "Would you like to provide the rule content manually or via URL?" → `url`
- "What URL should I fetch?" → `https://github.com/airbnb/javascript`
- [I fetch and analyze the content, then generate comprehensive rules]

## Example Flow

### Manual Entry Example

**You**: `/create-rule`

**Me**: "Would you like to provide the rule content manually or via URL? (Type 'manual' or 'url')"

**You**: "manual"

**Me**: "What should this rule be called?"

**You**: "frontend-standards"

**Me**: "What are the instructions/content for this rule? (You can provide the full rule content)"

**You**: "When working in components directory:
- Always use Tailwind for styling
- Use Framer Motion for animations
- Follow component naming conventions"

**Me**: "Which rule type would you like?
1. Always Apply - Applied to every chat session
2. Apply Intelligently - Agent decides when relevant (requires description)
3. Apply to Specific Files - When files match patterns (requires globs)
4. Apply Manually - When @-mentioned in chat

Please choose 1, 2, 3, or 4:"

**You**: "3"

**Me**: "What file patterns should trigger this rule? (e.g., `*.tsx`, `**/components/**`, `src/components/*.tsx`)"

**You**: "**/components/**/*.tsx"

**Me**: [Creates the rule folder structure with RULE.md containing frontmatter and content]

### URL Fetch Example

**You**: `/create-rule`

**Me**: "Would you like to provide the rule content manually or via URL? (Type 'manual' or 'url')"

**You**: "url"

**Me**: "What should this rule be called?"

**You**: "airbnb-javascript-style"

**Me**: "What URL should I fetch?"

**You**: "https://github.com/airbnb/javascript"

**Me**: [Fetches the URL content, analyzes it, and generates comprehensive rules]

**Me**: "I've analyzed the content and generated the following rule:

# Airbnb JavaScript Style Guide

## General Principles
- Use ES6+ features when possible
- Prefer const/let over var
- Use meaningful variable names
...

Would you like me to proceed with creating this rule, or would you like to modify the content first?"

**You**: "proceed"

**Me**: "Which rule type would you like?
1. Always Apply - Applied to every chat session
2. Apply Intelligently - Agent decides when relevant (requires description)
3. Apply to Specific Files - When files match patterns (requires globs)
4. Apply Manually - When @-mentioned in chat

Please choose 1, 2, 3, or 4:"

**You**: "3"

**Me**: "What file patterns should trigger this rule? (e.g., `*.js`, `**/*.js`, `src/**/*.js`)"

**You**: "**/*.js"

**Me**: [Creates the rule folder structure with RULE.md containing frontmatter and generated content]

## Rule File Structure

The command will create:
```
.cursor/rules/
  your-rule-name/
    RULE.md
```

The `RULE.md` file will contain:
- Frontmatter with appropriate metadata (alwaysApply, globs, description)
- Your rule content in markdown format

## Notes

- Rule names should be descriptive and use kebab-case (e.g., `frontend-standards`, `api-validation`)
- Keep rule content focused and under 500 lines (split large rules into multiple rules)
- You can reference files in rules using `@filename.ts` syntax
- After each question, respond with your answer and I'll move to the next step
- **URL fetching**: When using URLs, I'll fetch and analyze the content to generate comprehensive rules. Large documents may be summarized, but key principles will be captured. You can always refine the generated content before finalizing the rule.
- **URL limitations**: Some URLs may require authentication or may not be accessible. If a URL cannot be fetched, I'll let you know and you can provide the content manually instead.

---

**Note**: This command follows the same interactive pattern as `/create-command` - one question at a time, waiting for your response before proceeding.
