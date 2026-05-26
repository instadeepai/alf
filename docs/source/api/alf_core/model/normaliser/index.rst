Normalisers
===========

Stateful normalisation and standardisation classes used during model training
and inference. ``InputNormaliser`` applies min-max scaling to input features;
``OutputStandardiser`` applies Z-score standardisation to output labels.
Both classes are fit on training data only — never on validation or test data.

.. automodule:: alf_core.model.normaliser
   :members:
   :show-inheritance:
   :undoc-members:
