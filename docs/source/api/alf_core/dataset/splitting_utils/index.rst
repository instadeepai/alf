Splitting Utils
===============

The :mod:`alf_core.dataset.splitting_utils` module provides utility functions for splitting datasets into train, validation,
test, and candidate pool sets. It supports three splitting strategies: random splitting
(``split_type="random"``), low-vs-high splitting that separates top and bottom performers by label
value (``split_type="low_vs_high"``), and stratified splitting that preserves class proportions
across splits for classification tasks (``split_type="stratified"``). Stratified splitting is only
valid with ``ProblemType.BINARY`` or ``ProblemType.MULTICLASS``.

.. automodule:: alf_core.dataset.splitting_utils
   :members:
   :show-inheritance:
   :undoc-members:
