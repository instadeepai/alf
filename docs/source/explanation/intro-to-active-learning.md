# Intro to Active Learning

This page explains the *method* ALF implements, with as little API as possible. For the concrete
objects, see [Core Concepts](core-concepts.md).

## The ask / tell loop

Active learning turns a fixed-budget search into a feedback loop. Each round has two halves:

- **ask**: given everything measured so far, *propose* the next batch of candidates to evaluate.
- **tell**: *reveal* the true labels for that batch and fold them back into what we know.

```text
        ┌──────────────────────────────────────────────┐
        │                                                │
        ▼                                                │
   ┌─────────┐   ask    ┌──────────────┐   batch   ┌────────────┐
   │ Surrogate│ ───────▶ │ Acquisition  │ ────────▶ │   Oracle   │
   │  (model) │          │  + Search    │           │ (true score)│
   └─────────┘          └──────────────┘           └────────────┘
        ▲                                                │
        │                  tell (retrain)                │
        └────────────────────────────────────────────────┘
                     repeat for N rounds
```

Each turn of the loop, the `surrogate` is **retrained** on the growing set of labelled data,
so its predictions (and the acquisition decisions built on them) improve round over round. The
goal is to reach good designs in **as few rounds as possible**, because rounds are what cost money.

## Why information-per-round matters

A model that is 2% more accurate offline may be worthless if it leads you to measure the wrong
batch first. Conversely, a model with good `Calibration` can win even when less accurate,
because acquisition can trust its uncertainty to explore where it matters. This is why ALF
measures performance *versus round* (`regret`, `best-found`,
`recall`) rather than as a single static score; see [Why ALF?](why-alf.md).

## Offline vs online

The loop is the same; what differs is **where the labels come from**.

| Mode | Oracle source | Use case |
|------|---------------|----------|
| **Offline** | A held-out dataset / pre-scored pool | Benchmarking and method development; fully reproducible, no external calls |
| **Online** | A live scorer (a trained model, a simulator) | Driving a real campaign where labels are generated on demand |

In ALF both modes run through the *same* task and loop; you swap the `oracle`'s scorer. This
is why a method validated offline can move to an online campaign without rewriting it.

→ Next: [Core Concepts](core-concepts.md), the objects that make up the loop.
