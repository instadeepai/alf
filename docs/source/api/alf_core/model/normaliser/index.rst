Normalisers
===========

Stateful normalisation and standardisation classes used during model training
and inference. :class:`~alf_core.model.normaliser.InputNormaliser` applies min-max scaling to input features;
:class:`~alf_core.model.normaliser.InputStandardiser` applies Z-score standardisation (zero mean, unit variance)
to input features; :class:`~alf_core.model.normaliser.OutputStandardiser` applies Z-score standardisation to
output labels. All classes are fit on training data only — never on validation
or test data.

The :func:`~alf_core.model.normaliser.make_input_transform` factory maps a ``normalise_inputs_strategy`` value
from :class:`~alf_core.model.base_model.BaseTrainConfig` to the matching transform: ``"minmax"`` constructs an
:class:`~alf_core.model.normaliser.InputNormaliser`, ``"zscore"`` an :class:`~alf_core.model.normaliser.InputStandardiser`. Min-max scaling
suits GP models, whose kernels measure distances between inputs; Z-score
standardisation is generally preferred for deep neural networks, but degrades
one-hot sequence inputs and is therefore not enabled by default for
:class:`~alf_tools.models.cnn.CNNModel`.

.. automodule:: alf_core.model.normaliser
   :members:
   :show-inheritance:
   :undoc-members:
