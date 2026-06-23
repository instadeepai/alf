Search
======

The :class:`~alf_core.optimizer.search.BaseSearch` class defines the interface for search strategies that generate candidate pools
for evaluation. Search strategies can operate on predefined datasets (:class:`~alf_core.optimizer.search.DatasetSearch`), generate
new candidates using models (:class:`~alf_core.optimizer.search.GeneratorSearch`), or apply mutations to existing sequences
(``MutationSearch``).

.. automodule:: alf_core.optimizer.search
   :members:
   :show-inheritance:
   :undoc-members:
