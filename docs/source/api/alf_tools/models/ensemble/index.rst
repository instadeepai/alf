Ensemble Wrapper
================

A generic ensemble wrapper that composes N ``BaseModel`` instances into a single model.
Assembles ``Predictions.empirical_dist`` from per-member outputs, supporting seed ensembles,
MC dropout ensembles, and combined (seed + dropout) ensembles for uncertainty quantification.

.. automodule:: alf_tools.models.ensemble
   :members:
   :show-inheritance:
   :undoc-members:
