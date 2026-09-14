Single Mutant Search
====================

Single Mutant Search is a mutation-based search strategy that generates candidate pools by applying
single-point mutations to existing sequences. This is particularly useful for local exploration around
known high-performing sequences in protein engineering.

The ``top_k`` parameter controls how many training sequences are mutated. With the default of 1,
only the best-labelled sequence is mutated, so the search explores a single neighbourhood at a time.
With a larger value, the ``top_k`` best-labelled sequences are each mutated and the results pooled
together, so the search covers several local optima at once rather than stalling when no neighbour
of the current best improves. A mutant reachable from more than one sequence appears only once in
the pool. Sequences with equal labels are ranked by their position in the training set (earliest
first), so ``top_k=1`` always selects the same sequence as single-best selection.

.. automodule:: alf_tools.optimizer.search.single_mutant_search
   :members:
   :show-inheritance:
   :undoc-members:
