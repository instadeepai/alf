# Docs UX Improvements Plan

Branch: `improve/docs-ux`

## Goals

Make docs and README more user-friendly: reduce duplication, improve structure and visual quality,
clarify the audience split between users and contributors, and remove internal tooling files before
open-sourcing.

---

## 1. Restructure `INSTALLATION.md` (docs-first)

**File:** `docs/INSTALLATION.md`

Current structure leads with "Development Installation", which is wrong for most readers.
Flip the order and simplify language:

**New structure:**
1. **Package Installation** (first, prominent) — simple `pip install` commands, no mention of `uv`.
   Make clear this is all most users need.
2. **Optional extras** (ESM2, Chemprop) — stays here, briefly.
3. **Authentication setup** — move here (needed for pip install from private GitHub).
4. **GPU support** — brief, `pip install torch --index-url ...` only; no `uv` instructions.
5. **Development setup** (last, clearly scoped to contributors) — `uv sync`, GPU via uv, etc.
   This section can be short since CONTRIBUTING.md covers contribution workflow.

Remove or collapse the "Architecture" and "PyTorch Configuration Details" sections — they belong
in explanation docs, not an installation guide.

---

## 3. Restructure README

**File:** `README.md`

Current order buries important entry points. New order:

1. Badges + one-line description (keep as-is)
2. **"Why ALF?"** — pull the key paragraph up here (2–3 sentences max, linking to full page)
3. **Documentation link** — prominent, early
4. **Package Architecture** (`alf-core` vs `alf-tools`) — stays, it's short
5. **Installation** — simplified (pip install only, point to INSTALLATION.md for details)
6. **Quick Start** — extend slightly with inline comments explaining each step
7. **Tutorials** — move up, before Contributing
8. **Contributing / Dev setup** — stays near end
9. **Project Structure** — move to end or remove entirely (it's low value for most readers)

---

## 4. Extend Quick Start explanations (README + docs landing page)

**Files:** `README.md`, `docs/source/index.md`

Both have identical Quick Start code blocks with no explanatory text. They should:
- Stay in sync (or reference one from the other — preferred to avoid duplication)
- Have brief inline comments or a short paragraph after the code block explaining what each
  component does (Surrogate, Oracle, Optimizer, DesignTask)

Preferred: keep the full annotated version in `docs/source/index.md` and have README point to it.

---

## 5. Improve docs landing page

**File:** `docs/source/index.md`

- Add an image/banner at the top (can reuse an existing one from `imgs/` or create a simple one)
- Move "Why ALF?" reference card higher — above the four navigation cards
- Quick Start: extend with short explanation of each component (see item 4)
- Remove the installation section from here — it duplicates README; replace with a single line
  linking to INSTALLATION.md

---

## 6. Fix code block styling

**File:** `docs/source/conf.py` or relevant CSS/theme config

The yellow code block highlight colour is too bright. Identify where it's set (likely a custom
CSS override or Furo theme variable) and change it to grey.

---

## 7. Polish niche tutorials

**Files:** `docs/source/tutorials/index.md` + linked notebooks

The "Go deeper" tutorials (ESM-2, Chemprop, GuacaMol) have thinner explanations than the core
ones. For each:
- Ensure there's a short intro paragraph explaining *why* you'd use this model/dataset
- Ensure the notebook has a clear "what we'll do" section at the top

---

## 8. Add an `alf-core`-only tutorial

**Files:** new notebook in `tutorials/`, entry in `docs/source/tutorials/index.md`

Add one tutorial that uses only `alf-core` — no PyTorch required. Goal: showcase the lightweight
package and lower the barrier to entry. Good candidate: a simple synthetic optimization loop with
a custom scorer (e.g., optimise a quadratic function over sequences using random search + GP from
scipy).

---

## 9. Tighten CONTRIBUTING.md scope

**File:** `docs/CONTRIBUTING.md`

Currently comprehensive. Before OSS, decide: is this aimed at external contributors or just
internal? For external OSS:
- Keep: code style, PR process, how to extend base classes
- Remove or shorten: internal-only CI details, InstaDeep-specific tooling
- Consider whether a short "how to add your own model" recipe in the how-to section replaces
  most of what's in CONTRIBUTING.md

---

## Acceptance criteria

- [ ] INSTALLATION.md leads with `pip install`, dev setup is last
- [ ] README has docs link and "Why ALF?" in the top third
- [ ] README quick start has inline explanation
- [ ] Tutorials link appears before Contributing in README
- [ ] Project structure section is at the end of README (or removed)
- [ ] Docs landing page has image and "Why ALF?" above nav cards
- [ ] No duplicate installation content between README and docs landing page
- [ ] Code block colour is not bright yellow
- [ ] All tutorial pages have intro paragraphs
- [ ] At least one `alf-core`-only tutorial exists
