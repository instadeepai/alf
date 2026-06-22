# Add your own model

Plug a new predictor into ALF by subclassing {py:class}`BaseModel <alf_core.model.base_model.BaseModel>`. Once wrapped in a {py:class}`Surrogate <alf_core.surrogate.surrogate.Surrogate>` (or an
{py:class}`Oracle <alf_core.oracle.oracle.Oracle>`), it slots into the loop with no other changes.

**Base class:** {py:class}`BaseModel <alf_core.model.base_model.BaseModel>`. **Key methods:** `featurise`, `train`, `predict`, `sample`.

A model can play three roles depending on which methods it implements and how it is used:
a `Surrogate` (needs `train` + `predict`), an `Oracle` (needs `predict`), or a generator
(needs `sample`). See [Core Concepts](../explanation/core-concepts.md) for the roles.

## Steps

1. Subclass {py:class}`BaseModel <alf_core.model.base_model.BaseModel>` under `tools/alf_tools/models/`.
2. Implement `featurise`, `train`, `predict`, and `sample` as needed for the role.
3. Wrap it in a `Surrogate` (or `Oracle`) and pass it to a {py:class}`Task <alf_core.tasks.base_task.BaseTask>`.
4. Add tests and run the build / pre-commit checks.

## Reference notebooks

- [Models](https://github.com/instadeepai/alf/blob/main/tutorials/extending_base_classes/models.ipynb): implement a custom {py:class}`BaseModel <alf_core.model.base_model.BaseModel>`.
- [Model roles](https://github.com/instadeepai/alf/blob/main/tutorials/extending_base_classes/model_roles.ipynb): the surrogate / oracle / generator roles in detail.
