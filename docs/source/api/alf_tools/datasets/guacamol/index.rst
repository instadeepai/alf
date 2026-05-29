GuacaMol Dataset
================

The GuacaMol dataset implementation. This dataset wraps the GuacaMol benchmark corpus
(~1.6 million drug-like SMILES from ChEMBL) and computes RDKit physicochemical properties
(e.g. ``MolLogP``, ``TPSA``, ``QED``) as regression targets. Supports random, low-vs-high,
and original paper train/valid/test splits.

.. automodule:: alf_tools.datasets.guacamol
   :members:
   :show-inheritance:
   :undoc-members:
