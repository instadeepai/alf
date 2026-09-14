Element Substitution Search
===========================

Element Substitution Search generates candidate crystals by swapping one element for another in
the best-labelled training structures. Swaps are drawn from the Hautier et al. (2011)
substitution table, data-mined from the ICSD, so only pairs with precedent in known chemistry
are proposed, and only charge-balanced results are kept. Candidates are returned ordered by
substitution probability, most plausible first, and anything already evaluated in the run is
filtered out.

``allowed_elements`` lists the element symbols the oracle can score, and is required. The
substitution table reaches further up the periodic table than most scoring models, and a
candidate the oracle cannot score returns NaN, which propagates into the training labels and
costs a round.

``top_k`` sets how many training structures are used as parents, counting only those that
actually yield new candidates. Structures are tried in label order, skipping any that cannot be
assigned oxidation states — bond-valence analysis has no parameters for some elements, notably
several actinides — and any whose substitutions have all been proposed before. Such structures
tend to cluster at the top of the ranking, so a fixed top-k slice would stall on them.

Two limits are worth knowing:

- The number of distinct elements is preserved, so a binary parent gives binary children and
  the reachable space is bounded by the stoichiometries already present in the training set.
- The parent's lattice is reused without relaxation, leaving bond lengths that are wrong for
  the new chemistry. Harmless when the featuriser uses only composition, but a structural
  featuriser needs a volume rescaling or relaxation first.

Requires the ``matbench`` extra, which supplies ``pymatgen``.

.. Structure is re-exported from pymatgen, not part of this module's API; excluding it
   stops sphinx_autodoc_typehints probing its annotations and warning about ArrayLike.

.. automodule:: alf_tools.optimizer.search.element_substitution_search
   :members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: Structure
