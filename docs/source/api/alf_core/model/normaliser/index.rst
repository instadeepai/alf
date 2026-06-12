Normalisers
===========

Stateful normalisation and standardisation classes used during model training
and inference. `InputNormaliser` applies min-max scaling to input features;
`InputStandardiser` applies Z-score standardisation (zero mean, unit variance)
to input features; `OutputStandardiser` applies Z-score standardisation to
output labels. All classes are fit on training data only — never on validation
or test data.

The `make_input_transform` factory maps a `normalise_inputs_strategy` value
from `BaseTrainConfig` to the matching transform: `"minmax"` constructs an
`InputNormaliser`, `"zscore"` an `InputStandardiser`. Min-max scaling
suits GP models, whose kernels measure distances between inputs; Z-score
standardisation is generally preferred for deep neural networks, but degrades
one-hot sequence inputs and is therefore not enabled by default for
`CNNModel`.

.. automodule:: alf_core.model.normaliser
   :members:
   :show-inheritance:
   :undoc-members:
