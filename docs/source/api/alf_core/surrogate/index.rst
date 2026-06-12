Surrogate
=========

The `Surrogate` approximates expensive experimental evaluation by wrapping a `BaseModel`. It
provides training functionality to fit the model on labelled training data, prediction capabilities
to make predictions on candidate sequences, and a `featurise()` method that delegates to the
underlying model to obtain feature embeddings — used by diversity-based acquisition functions such
as `CoreSet`. The surrogate is a key component in active learning, enabling efficient exploration
by reducing the need for costly experimental evaluations.

.. automodule:: alf_core.surrogate.surrogate
   :members:
   :show-inheritance:
   :undoc-members:
