Model
=====

The :class:`~alf_core.model.base_model.BaseModel` is an abstract base class that defines the interface for all models in the framework.
Models can serve multiple roles: as surrogate models (wrapped by ``Surrogate`` to approximate expensive
experimental evaluations), as oracle models (wrapped by ``Oracle`` for online evaluation), or as
generator models (wrapped by :class:`~alf_core.optimizer.search.GeneratorSearch` to sample candidate sequences).

All concrete model implementations must inherit from :class:`~alf_core.model.base_model.BaseModel` and implement the abstract methods:
``featurise()``, ``train()``, ``predict()``, and ``sample()``.

After ``setup()`` is called with a dataset, the model exposes two attributes derived from the
dataset config: ``problem_type`` (the :class:`~alf_core.utils.enums.ProblemType` enum value) and ``output_dim`` (the number
of output neurons — ``1`` for regression and binary classification, ``num_classes`` for
multiclass).

The :class:`~alf_core.model.base_model.BaseTrainConfig` dataclass provides shared training fields — including
``normalise_inputs_strategy`` and ``standardise_outputs`` — that all concrete training configs
inherit from.

.. toctree::
   :maxdepth: 1

   Base Model <base_model/index>
   Normalisers <normaliser/index>
