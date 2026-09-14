Element Substitution Search
===========================

The materials-domain counterpart to :doc:`SMILES Mutation Search </api/alf_tools/optimizer/search/smiles_mutation_search/index>`:
a search protocol that proposes novel crystals each round by making single-species swaps on the
best-labelled training structures. Candidate swaps come from the Hautier et al. (2011) lambda
table mined from the ICSD, so only substitutions with real precedent in known chemistry are
proposed, and each parent is decorated with oxidation states by bond-valence analysis first.
Charge balance is enforced during prediction: for rocksalt LiF this cuts 481 raw species maps
down to 65.

Parent selection walks the *entire* label-descending ranking rather than taking a fixed top-k
slice, because the aim is ``top_k`` **usable** parents. Oxidation-state decoration fails on
exactly the structures the loop drives toward — bond-valence parameters are missing for Ac, Th
and several other actinides, which a scoring model such as MACE rates happily — so undecorable
structures accumulate at the top of the ranking. A parent counts toward the quota only if it
contributed at least one novel child, which stops exhausted parents from ending the walk with
most of the ranking unexamined.

``allowed_elements`` is required rather than optional: the lambda table covers 230 species
including Am, Cm and Cf, while a model such as MACE-MPA-0 stops at Z=94. A child the oracle
cannot score costs a whole round, since the resulting NaN reaches the training labels and the
next surrogate fit is rejected.

Two limitations are worth knowing before reaching for this search. Species count is preserved —
the lambda table maps one species onto another, so a binary parent yields binary children and the
reachable space is bounded by the stoichiometries in the seed set. And the lattice is not
relaxed: species are swapped onto the parent's fixed lattice, so bond lengths are the parent's
and physically wrong for the new chemistry. The latter costs nothing when the downstream
featurizer reduces a structure to its composition, but a structural featurizer would need volume
rescaling or a relaxation first.

Requires the ``matbench`` extra, which supplies ``pymatgen``.

.. Structure is re-exported from pymatgen, not part of this module's API; excluding it
   stops sphinx_autodoc_typehints probing its annotations and warning about ArrayLike.

.. automodule:: alf_tools.optimizer.search.element_substitution_search
   :members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: Structure
