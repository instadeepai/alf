# Copyright 2026 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import numpy as np
from alf_core import Candidate, Modality, SearchProtocol, State
from rdkit import Chem
from rdkit.rdBase import BlockLogs

# Covers common organic elements (C, N, O, S, F), aromatic lowercase forms, ring-bond
# digits, and the most common bond symbols. Deliberately excludes multi-character
# tokens (Cl, Br, bracket atoms like [nH]) since mutation here is a single-character
# substitution; molecules that need those atoms/tokens are reachable only by starting
# from a training SMILES that already contains them, not by mutating one that doesn't.
# Pass a custom `alphabet` to reach a different slice of chemical space.
DEFAULT_SMILES_ALPHABET = "CNOSFcnos()=#123456789"


class SmilesMutationSearch(SearchProtocol):
    """Search protocol that mutates the top-K best-scoring SMILES seen so far.

    This is the online, generative counterpart to SingleMutantSearch for molecules:
    for each of the top-K training candidates, it enumerates single-character
    substitutions of that SMILES string (mirroring SingleMutantSearch's
    enumerate-all-single-point-mutations shape), then keeps only the mutations
    RDKit can parse as valid molecules, deduplicated by canonical SMILES across
    the whole pool. Unlike protein sequences, most single-character SMILES edits
    are structurally invalid, so this filtering step is required.

    Mutating only the single current best (top_k=1) makes the search a pure
    hill-climb: once no neighbour of the incumbent beats it, the same
    neighbourhood is regenerated every round and the loop stalls in that local
    optimum. Mutating several top candidates each round keeps multiple regions
    of the space under active exploration simultaneously, so a stall in one
    neighbourhood doesn't stall the whole search.
    """

    def __init__(
        self,
        alphabet: str = DEFAULT_SMILES_ALPHABET,
        max_candidates: int | None = None,
        top_k: int = 1,
    ):
        """Initialize the SMILES mutation search protocol.

        Args:
            alphabet: Single-character SMILES tokens to substitute at each position.
            max_candidates: Optional cap on the number of candidates returned.
            top_k: Number of best-labelled training candidates to mutate from each
                round. 1 reproduces plain single-best hill-climbing; higher values
                keep several neighbourhoods under exploration at once.
        """
        self.alphabet = alphabet
        self.max_candidates = max_candidates
        self.top_k = top_k

    def __call__(self, state: State) -> List[Candidate]:
        """Generate valid single-character mutants of the top-K training SMILES.

        Args:
            state: The task state containing the dataset and surrogate model.

        Returns:
            A list of candidates with novel, valid, deduplicated SMILES.

        Raises:
            ValueError: If the training set is empty, or if no valid, novel
                mutation of the top-K training SMILES was found.
        """
        train_dataset = state.dataset.train_dataset
        num_bases = min(self.top_k, len(train_dataset.candidates))
        if num_bases == 0:
            raise ValueError(
                "SmilesMutationSearch requires at least one training candidate to "
                "mutate, but state.dataset.train_dataset is empty."
            )
        top_indices = np.argsort(train_dataset.labels)[::-1][:num_bases]
        base_smiles_list = [train_dataset.candidates[i].data for i in top_indices]

        # Most single-character SMILES mutations are structurally invalid by design
        # (unlike SingleMutantSearch's protein-sequence mutations, which are always
        # valid amino acids). RDKit's C++ logger would otherwise write a parse error
        # straight to stderr for each one; BlockLogs scopes the suppression to this
        # call only, since disabling it at import time would silence RDKit's logger
        # process-wide for any other code sharing the interpreter.
        with BlockLogs():
            seen = set()
            for base_smiles in base_smiles_list:
                base_mol = Chem.MolFromSmiles(base_smiles)
                seen.add(Chem.MolToSmiles(base_mol) if base_mol is not None else base_smiles)

            mutant_pool = []
            for base_smiles in base_smiles_list:
                for i in range(len(base_smiles)):
                    for ch in self.alphabet:
                        if base_smiles[i] == ch:
                            continue
                        mutant = base_smiles[:i] + ch + base_smiles[i + 1 :]
                        mol = Chem.MolFromSmiles(mutant)
                        if mol is None:
                            continue
                        canonical = Chem.MolToSmiles(mol)
                        if canonical in seen:
                            continue
                        seen.add(canonical)
                        mutant_pool.append(Candidate(data=canonical, modality=Modality.MOLECULE))

        if not mutant_pool:
            raise ValueError(
                f"SmilesMutationSearch found no valid, novel single-character mutation of "
                f"the top-{num_bases} training SMILES {base_smiles_list!r} using alphabet "
                f"{self.alphabet!r}. Try a larger alphabet or a higher top_k."
            )

        if self.max_candidates is not None:
            mutant_pool = mutant_pool[: self.max_candidates]
        return mutant_pool
