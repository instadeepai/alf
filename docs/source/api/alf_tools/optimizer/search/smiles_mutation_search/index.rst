SMILES Mutation Search
=======================

The molecule-domain counterpart to :doc:`Single Mutant Search </api/alf_tools/optimizer/search/single_mutant_search/index>`:
a search protocol that proposes novel SMILES each round by mutating the top-K best-labelled
training molecules, filtered for RDKit validity. Unlike protein sequences, most single-character
SMILES edits are structurally invalid, so this filtering step is required; RDKit's error logger is
silenced accordingly since invalid mutations are expected and handled internally.

Mutating only the single current best (``top_k=1``) makes the search a pure hill-climb: once no
neighbour of the incumbent beats it, the same neighbourhood is regenerated every round and the
loop stalls in that local optimum. Setting ``top_k`` higher keeps several regions of the search
space under active exploration at once, so a stall in one neighbourhood doesn't stall the whole
search.

.. automodule:: alf_tools.optimizer.search.smiles_mutation_search
   :members:
   :show-inheritance:
   :undoc-members:
