Results
=======

The ``Results`` dataclass aggregates the metrics computed on a surrogate's predictions
relative to the targets (for example, accuracy, recall, regret), storing
per-round metrics and summary statistics. The metrics computed depend on whether the
task is classification or regression; see ``alf_core.utils.metrics`` for more
details and examples. It is the primary return value of a completed ALF experiment
and is used to evaluate and compare experiment runs.

.. automodule:: alf_core.dataclasses.results
   :members:
   :show-inheritance:
   :undoc-members:
