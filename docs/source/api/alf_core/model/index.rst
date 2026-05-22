Model
=====

The ``BaseModel`` is an abstract base class that defines the interface for all models in the framework.
Models can serve multiple roles: as surrogate models (wrapped by ``Surrogate`` to approximate expensive
experimental evaluations), as oracle models (wrapped by ``Oracle`` for online evaluation), or as
generator models (wrapped by ``GeneratorSearch`` to sample candidate sequences).

All concrete model implementations must inherit from ``BaseModel`` and implement the abstract methods:
``featurise()``, ``train()``, ``predict()``, and ``sample()``.

After ``setup()`` is called with a dataset, the model exposes two attributes derived from the
dataset config: ``problem_type`` (the ``ProblemType`` enum value) and ``output_dim`` (the number
of output neurons — ``1`` for regression and binary classification, ``num_classes`` for
multiclass).

.. automodule:: alf_core.model.base_model
   :members:
   :show-inheritance:
   :undoc-members:
