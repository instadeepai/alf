Diversity Metrics
=================

Diversity metrics for evaluating the spread of acquired candidate batches.
These metrics operate on :class:`~alf_core.dataclasses.candidate.Candidate` objects
rather than prediction arrays, so they are not registered in the regression or
classification registries.

``intra_batch_diversity`` computes the average pairwise dissimilarity within a batch,
using normalised edit distance for SEQUENCE candidates and cosine distance for
EMBEDDING and TABULAR candidates.

.. automodule:: alf_core.utils.metrics.diversity
   :members:
   :show-inheritance:
   :undoc-members:
