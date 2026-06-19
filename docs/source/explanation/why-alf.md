# Why ALF?

## The problem

In scientific discovery (protein engineering, small-molecule design, materials), the
bottleneck is rarely compute. It is the **experiment**: each label costs a wet-lab assay, a
simulation, or a measurement. You can afford only a handful of rounds, and each round you must
decide *which* candidates are worth the spend.

This is a **sequential decision problem**, not a one-shot prediction problem. The question is not
"how accurately can a model predict fitness?" but "given everything measured so far, which batch
should we measure *next* to reach a good design in as few rounds as possible?"

## What ALF does

ALF (Active Learning Framework) runs the full **active-learning loop** for you: train a
`surrogate` on what you've measured, use an `acquisition function` to score candidates
by their expected value, select a batch, score it with an `oracle`, and repeat. It provides
modular, swappable components (`Dataset`, `Model`, `Acquisition function`,
and `Search function` objects), so you can compose an
experiment from parts and change any one of them without rewriting the rest.

See [Intro to Active Learning](intro-to-active-learning.md) for the loop itself, and
[Core Concepts](core-concepts.md) for the objects that implement it.

## The differentiator: *sequential* benchmarking

Existing protein benchmarks such as **ProteinGym** and **FLIP** are, at heart, *static* supervised
benchmarks: fixed train/test splits that measure how well a model predicts fitness. That is a real
and useful question, but it is not the question a discovery experiment actually faces.

ALF is built to measure the part those benchmarks leave open: **how good a method is at the
sequential loop**. Instead of a single accuracy number, ALF tracks performance *as a function of
round*:

- **`Regret`** and **`best-found`**: are we converging on the best
  designs, and how fast?
- **`Recall`** / **`coverage`**: are we finding the high-value region, not
  just one peak?
- **`Calibration`**: does the surrogate *know what it doesn't know*, so acquisition can
  trust its uncertainty?

This "information per round" framing is the same story that anchors the ALF benchmark suite and
keeps the narrative consistent across the library, the benchmark, and the paper.

## Who it's for

- **Researchers** running real optimisation experiments who want a tested loop instead of re-implementing the ask/tell cycle themselves.
- **Method developers** who want to drop in a new acquisition function or surrogate and measure it
  against baselines on equal footing.

→ Ready to see it run? Start with the [Tutorials](../tutorials/index.md).
