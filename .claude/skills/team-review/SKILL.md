---
name: team-review
description: >
  Review a PR for team-style consistency: CHANGELOG verb tense, documentation currency,
  naming conventions, and open-source hygiene. Use when the user says "team-review",
  "check style consistency", "review for conventions", or wants a quick conventions pass
  before requesting a full code review.
---

# Team Review Skill

You are a team reviewer checking that a PR meets ALF's house style and open-source hygiene
standards. This is NOT a deep code review — it is a consistency and conventions pass.
Flag deviations clearly so the author can fix them before requesting peer review.

---

## Step 0 — Gather Context

Collect the PR diff and any changed documentation files. Pay particular attention to:
- `CHANGELOG.md`
- `README.md`, `docs/`, `*.md` files
- `pyproject.toml` files
- `.github/` workflow files

---

## Step 1 — CHANGELOG Verb Tense Consistency

**ALF uses past tense throughout CHANGELOG entries.** Every bullet must start with a
past-tense verb. This is the house rule chosen to match the keepachangelog.com convention
and to feel natural when reading a historical record.

Accepted first words (examples): `Added`, `Fixed`, `Removed`, `Changed`, `Introduced`,
`Released`, `Integrated`, `Published`, `Updated`, `Deprecated`.

Flag any bullet that starts with:
- A gerund (`Adding`, `Introducing`, `Releasing`, `Publishing`, …) — **wrong**
- An infinitive (`Add`, `Fix`, `Remove`, …) — **wrong**
- A noun or article (`The`, `A`, `Support for`, …) — **wrong**

For each violation, write:
```
CHANGELOG line N: "<offending text>" → suggest "<past-tense fix>"
```

If the entire CHANGELOG is consistent, confirm it with one line.

---

## Step 2 — README & Documentation Currency

- [ ] Does the PR change behaviour, CLI flags, class names, or API surface? If yes, is
  the README or relevant doc page updated?
- [ ] Are installation instructions using `pip install git+https://github.com/...` still
  accurate? (Do not switch to PyPI names until packages are published on PyPI.)
- [ ] Are any internal hostnames, private GitHub token flows, or `.netrc` instructions
  present? These must be removed before the repo goes public.

---

## Step 3 — Open-Source Hygiene

- [ ] License headers: new source files should not add per-file copyright blocks (Apache
  2.0 top-level `LICENSE` file covers the whole repo).
- [ ] No secrets or internal credentials committed (API keys, tokens, `.netrc` examples
  with real values, internal service URLs).
- [ ] CI runners: while the repo is still private, runners must stay as `instadeep-ci` /
  `instadeep-ci-4`. Switching to `ubuntu-latest` is deferred to the final public-release PR.
- [ ] `pyproject.toml` metadata (`license`, `authors`, `keywords`, `classifiers`,
  `[project.urls]`) is present and accurate for both `core/` and `tools/`.

---

## Step 4 — Write the Review

Output a short structured report:

### CHANGELOG
State whether verb tense is consistent (past tense). List any violations with suggested fixes.

### Documentation
List any gaps between the diff and the docs.

### Open-Source Hygiene
List any hygiene issues found, or confirm all clear.

### Summary
One sentence: ready to request review, or list the blockers.

---

## Severity

| Level | Meaning |
|-------|---------|
| **Block** | Must fix before requesting review (wrong verb tense throughout CHANGELOG, exposed secrets, wrong CI runners). |
| **Fix** | Should fix before merge (stale docs, missing pyproject metadata). |
| **Note** | Worth mentioning; author can decide (minor wording, nit-level inconsistency). |
