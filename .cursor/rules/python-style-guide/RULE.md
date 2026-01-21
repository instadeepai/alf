---
alwaysApply: false
description: Follow python code guidelines
---

# PEP 8 - Python Style Guide

## Core Principles

- **Readability counts**: Code is read more often than written
- **Consistency**: Consistency within a project is more important than consistency with this guide
- **Pragmatism**: Know when to be inconsistent - don't break backwards compatibility just to comply

## Code Layout

### Indentation
- Use **4 spaces** per indentation level
- Never mix tabs and spaces
- Prefer spaces over tabs

### Maximum Line Length
- Limit all lines to **79 characters** maximum
- For docstrings and comments, limit to **72 characters**
- Use Python's implicit line continuation inside parentheses, brackets, and braces

### Blank Lines
- Two blank lines between top-level function and class definitions
- One blank line between method definitions inside a class
- Use blank lines sparingly within functions to indicate logical sections

### Imports
- Imports should be on separate lines
- Order: standard library → related third party → local application/library
- Use absolute imports when possible
- Avoid wildcard imports (`from module import *`)
- Group imports with blank lines between groups

### Source File Encoding
- Python 3: UTF-8 (default)
- Python 2: UTF-8 or ASCII

## String Quotes

- Use single quotes for strings that contain double quotes
- Use double quotes for strings that contain single quotes
- Use triple double quotes for docstrings
- Be consistent within a module

## Whitespace in Expressions and Statements

### Pet Peeves
- Avoid extraneous whitespace:
  - Immediately inside parentheses, brackets, or braces
  - Between a trailing comma and a following close parenthesis
  - Immediately before a comma, semicolon, or colon
  - More than one space around an assignment operator

### Other Recommendations
- Always surround binary operators with a single space on either side
- Don't use spaces around `=` when used for keyword arguments or default parameter values
- Use spaces around operators, but group them sensibly

## Comments

### Block Comments
- Start each line with `#` followed by a single space
- Paragraphs within block comments should be separated by a line containing a single `#`

### Inline Comments
- Use sparingly
- Separate inline comments by at least two spaces from the statement
- Start with `#` and a single space

### Documentation Strings
- Write docstrings for all public modules, functions, classes, and methods
- Use triple double quotes: `"""docstring"""`
- One-line docstrings: closing quotes on same line
- Multi-line docstrings: summary line, blank line, detailed description

## Naming Conventions

### Overriding Principle
- Names visible to the user as public parts of the API should follow conventions that reflect usage rather than implementation

### Naming Styles
- **b** (single lowercase letter)
- **B** (single uppercase letter)
- **lowercase**
- **lower_case_with_underscores**
- **UPPERCASE**
- **UPPER_CASE_WITH_UNDERSCORES**
- **CapitalizedWords** (CapWords, PascalCase)
- **mixedCase**
- **Capitalized_Words_With_Underscores**
- **\_single_leading_underscore**: weak "internal use" indicator
- **single_trailing_underscore_**: avoids conflict with Python keyword
- **\_\_double_leading_underscore**: name mangling (inside class FooBar, `__boo` becomes `_FooBar__boo`)
- **\_\_double_leading_and_trailing_underscore__**: special objects (e.g., `__init__`)

### Prescriptive Naming Conventions

#### Names to Avoid
- Never use `l`, `O`, or `I` as single character names (confusable with `1` and `0`)

#### Package and Module Names
- Short, all-lowercase names
- Underscores can be used if it improves readability
- Example: `mypackage`, `my_module`

#### Class Names
- Use **CapWords** convention
- Example: `MyClass`, `HttpConnection`

#### Type Variable Names
- Use short CapWords names, preferably single letters
- Add `_co` or `_contra` variance suffix if needed
- Example: `T`, `KT`, `VT`, `T_co`

#### Exception Names
- Use **CapWords** convention
- Should end in "Error" if the exception is actually an error
- Example: `MyError`, `ValueError`

#### Function and Variable Names
- Use **lowercase** with words separated by underscores
- Example: `my_function`, `my_variable`

#### Function and Method Arguments
- Use `self` as the first argument to instance methods
- Use `cls` as the first argument to class methods
- If a function argument name conflicts with a reserved keyword, append a single trailing underscore

#### Method Names and Instance Variables
- Use function naming convention: **lowercase** with underscores
- Use one leading underscore only for non-public methods and instance variables
- Example: `_internal_method`, `_private_var`

#### Constants
- Use **UPPER_CASE_WITH_UNDERSCORES**
- Example: `MAX_OVERFLOW`, `TOTAL`

## Programming Recommendations

### General
- Code should be written in a way that does not disadvantage other implementations of Python
- Comparisons to singletons like `None` should always be done with `is` or `is not`, never the equality operators
- Use `is not` rather than `not ... is`

### String Comparisons
- Use `''.startswith()` and `''.endswith()` instead of string slicing to check for prefixes or suffixes
- Example: `if foo.startswith('bar'):` not `if foo[:3] == 'bar':`

### Object Type Comparisons
- Always use `isinstance()` instead of comparing types directly
- Example: `if isinstance(obj, int):` not `if type(obj) is type(1):`

### Sequences
- For sequences (strings, lists, tuples), use the fact that empty sequences are false
- Example: `if not seq:` not `if len(seq):`

### Boolean Comparisons
- Don't compare boolean values to `True` or `False` using `==`
- Example: `if greeting:` not `if greeting == True:`

### Exception Handling
- Be specific in exception handling: catch specific exceptions, not bare `except:`
- When catching operating system errors, prefer the explicit exception hierarchy
- Limit `try` clause to the absolute minimum amount of code necessary
- Use `with` statements for resource management

### Return Statements
- Be consistent in return statements
- Either all return statements in a function should return an expression, or none should
- If any return statement returns an expression, any return statements where no value is returned should explicitly state `return None`

### Function and Variable Annotations
- Use PEP 484 syntax for type annotations
- Annotations should have a single space after the colon
- No space before the colon
- If an assignment has a right hand side, the equality sign should have exactly one space on both sides
- Example: `code: int`, `result: int = 0`

### Context Managers
- Use `with` statements to ensure resources are cleaned up promptly
- Context managers should be invoked through separate functions or methods when they do something other than acquire and release resources

### Flow Control
- Avoid using `return`/`break`/`continue` within the `finally` suite of a `try...finally` where the flow control statement would jump outside the finally suite

## References

- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
