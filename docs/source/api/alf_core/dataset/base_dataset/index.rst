Base Dataset
============

The ``BaseDataset`` class is an abstract base class that defines the interface for all datasets in
the framework. It manages data loading through the abstract ``load_dataset()`` method, splits data
into train/validation/test/candidate_pool sets, updates splits with newly acquired candidates, and
provides labels for candidates from the original dataset (used by the oracle in offline settings).

``BaseDatasetConfig`` requires a ``problem_type`` field (``ProblemType.REGRESSION``,
``ProblemType.BINARY``, or ``ProblemType.MULTICLASS``) that drives metric selection, model output
dimensionality, and split strategy validation. Note that ``split_type="stratified"`` is only valid
for classification tasks and will raise a validation error if used with
``ProblemType.REGRESSION``.

.. automodule:: alf_core.dataset.base_dataset
   :members:
   :show-inheritance:
   :undoc-members:
