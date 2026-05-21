Dataset

The ``BaseDataset`` class manages data loading, splitting, and querying. It handles data loading through
the abstract ``load_dataset()`` method, splits data into train/validation/test/candidate_pool sets,
distributes newly acquired data into existing splits, and provides labels for candidates from the
original dataset (used by the oracle in offline settings).

.. toctree::
   :maxdepth: 1

   Base Dataset <base_dataset/index>
   Splitting Utils <splitting_utils/index>
