Base Dataset
============

The ``BaseDataset`` class is an abstract base class that defines the interface for all datasets in
the framework. It manages data loading through the abstract ``load_dataset()`` method, splits data
into train/validation/test/candidate_pool sets, updates splits with newly acquired candidates, and
provides labels for candidates from the original dataset (used by the oracle in offline settings).

.. automodule:: alf_core.dataset.base_dataset
   :members:
   :show-inheritance:
   :undoc-members:
