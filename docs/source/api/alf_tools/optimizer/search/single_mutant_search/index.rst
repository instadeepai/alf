Single Mutant Search
====================

Single Mutant Search is a mutation-based search strategy that generates candidate pools by applying
single-point mutations to existing sequences. This is particularly useful for local exploration around
known high-performing sequences in protein engineering.

By default (``top_k=1``) it seeds from the single best-labelled training sequence, exploring one
neighbourhood at a time. Setting ``top_k`` higher seeds from that many top training sequences and
unions their neighbourhoods, so the search explores around several local optima at once instead of
stalling when no neighbour of the current incumbent improves. Overlapping neighbourhoods are
deduplicated, and ties in labels are broken by training-set position (earliest index ranks higher),
which keeps ``top_k=1`` identical to single-best selection.

.. automodule:: alf_tools.optimizer.search.single_mutant_search
   :members:
   :show-inheritance:
   :undoc-members:
