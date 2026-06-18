# Switch offline to online

The loop is identical in both modes; what changes is **where the labels come from**, which is set
by the `Oracle`'s scorer. This means a method validated offline moves to a real experiment without
rewriting it.

| Mode | Oracle scorer | Use case |
|------|---------------|----------|
| **Offline** | a `Dataset` (held-out, pre-scored pool) | benchmarking and method development; reproducible, no external calls |
| **Online** | a trained `Model` (or simulator) | driving a real experiment where labels come on demand |

## Steps

Keep the `Dataset`, `Surrogate`, `Optimizer`, and `Task` the same, and swap only the
`Oracle`'s scorer:

```python
# Offline: score against the dataset's held-out labels
oracle = Oracle(scorer=dataset)

# Online: score with a trained model on demand
oracle = Oracle(scorer=trained_model)
```

See [Intro to Active Learning](../explanation/intro-to-active-learning.md) for the concept.

## Reference notebooks

- [Offline Design Tutorial](https://github.com/instadeepai/alf/blob/main/tutorials/experiments/offline_design_tutorial.ipynb)
- [Online Design Tutorial](https://github.com/instadeepai/alf/blob/main/tutorials/experiments/online_design_tutorial.ipynb)
