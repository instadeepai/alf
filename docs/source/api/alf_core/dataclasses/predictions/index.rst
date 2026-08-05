Predictions
===========

The :class:`~alf_core.dataclasses.predictions.Predictions` dataclass stores model predictions including mean predictions and uncertainty
estimates. This is returned by surrogate models and used for acquisition scoring.

The ``to_dataframe()`` method requires a ``problem_type`` argument to determine the output
format: regression predictions produce ``mean`` and ``variance`` columns, while classification
predictions produce one ``prob_class_N`` column per class (e.g., ``prob_class_0``,
``prob_class_1``).

.. automodule:: alf_core.dataclasses.predictions
   :members:
   :show-inheritance:
   :undoc-members:
