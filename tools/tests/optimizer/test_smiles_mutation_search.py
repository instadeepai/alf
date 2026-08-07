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

from types import SimpleNamespace

import numpy as np
import pytest
from alf_core import Candidate, LabelledCandidates, Modality, ProtocolSearch, State
from alf_core.dataset.base_dataset import BaseDataset, BaseDatasetConfig
from alf_core.surrogate.surrogate import Surrogate
from alf_core.utils.enums import ProblemType
from rdkit import Chem

from alf_tools.models.guacamol_oracle import GuacaMolOracleModel, GuacaMolOracleModelConfig
from alf_tools.optimizer.search.smiles_mutation_search import SmilesMutationSearch

_BENZENE = "c1ccccc1"
_ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


class _TinyMoleculeDataset(BaseDataset):
    """Minimal molecule dataset with one clear best-scoring SMILES."""

    def __init__(self, config: BaseDatasetConfig, best_smiles: str, other_smiles: list[str]):
        """Initialize with a designated best SMILES and lower-scoring others.

        Args:
            config: Dataset configuration.
            best_smiles: SMILES that should receive the highest label.
            other_smiles: Additional SMILES with lower labels.
        """
        super().__init__(config)
        self._best_smiles = best_smiles
        self._other_smiles = other_smiles

    def load_dataset(self) -> LabelledCandidates:
        """Load the best SMILES plus others, with the best given the highest label.

        Returns:
            LabelledCandidates over the best and other SMILES.
        """
        smiles_list = [self._best_smiles, *self._other_smiles]
        labels = np.array([1.0] + [0.1] * len(self._other_smiles))
        candidates = [Candidate(data=s, modality=Modality.MOLECULE) for s in smiles_list]
        return LabelledCandidates(candidates=candidates, labels=labels)


def _make_state(best_smiles: str, other_smiles: list[str]) -> State:
    config = BaseDatasetConfig(
        name="tiny_molecules",
        modality=Modality.MOLECULE,
        seed=0,
        train_ratio=1.0,
        validation_frac=0.0,
        test_ratio=0.0,
        split_type="random",
        problem_type=ProblemType.REGRESSION,
    )
    dataset = _TinyMoleculeDataset(config, best_smiles=best_smiles, other_smiles=other_smiles)
    dataset.setup()
    surrogate = Surrogate(model=GuacaMolOracleModel(GuacaMolOracleModelConfig(task_name="osimertinib_mpo")))
    return State(dataset=dataset, surrogate=surrogate)


class _RankedMoleculeDataset(BaseDataset):
    """Molecule dataset with an explicit, caller-controlled label ranking."""

    def __init__(self, config: BaseDatasetConfig, ranked_smiles: list[str]):
        """Initialize with SMILES in descending label order.

        Args:
            config: Dataset configuration.
            ranked_smiles: SMILES ordered from highest to lowest intended label.
        """
        super().__init__(config)
        self._ranked_smiles = ranked_smiles

    def load_dataset(self) -> LabelledCandidates:
        """Load candidates with strictly descending labels matching input order.

        Returns:
            LabelledCandidates with labels 1.0, 0.9, 0.8, ... in input order.
        """
        candidates = [Candidate(data=s, modality=Modality.MOLECULE) for s in self._ranked_smiles]
        labels = np.array([1.0 - 0.1 * i for i in range(len(self._ranked_smiles))])
        return LabelledCandidates(candidates=candidates, labels=labels)


def _make_ranked_state(ranked_smiles: list[str]) -> State:
    config = BaseDatasetConfig(
        name="ranked_molecules",
        modality=Modality.MOLECULE,
        seed=0,
        train_ratio=1.0,
        validation_frac=0.0,
        test_ratio=0.0,
        split_type="random",
        problem_type=ProblemType.REGRESSION,
    )
    dataset = _RankedMoleculeDataset(config, ranked_smiles=ranked_smiles)
    dataset.setup()
    surrogate = Surrogate(model=GuacaMolOracleModel(GuacaMolOracleModelConfig(task_name="osimertinib_mpo")))
    return State(dataset=dataset, surrogate=surrogate)


def _canonical_set(candidates) -> set:
    return {Chem.MolToSmiles(Chem.MolFromSmiles(c.data)) for c in candidates}


class TestSmilesMutationSearch:
    """Tests for SmilesMutationSearch."""

    def test_all_returned_candidates_are_valid_smiles(self):
        """Every candidate produced should parse successfully with RDKit."""
        state = _make_state(best_smiles=_BENZENE, other_smiles=[_ASPIRIN])
        search = SmilesMutationSearch()
        candidates = search(state)
        assert len(candidates) > 0
        for candidate in candidates:
            assert Chem.MolFromSmiles(candidate.data) is not None
            assert candidate.modality == Modality.MOLECULE

    def test_returned_candidates_are_unique_by_canonical_smiles(self):
        """No two returned candidates should be canonically identical."""
        state = _make_state(best_smiles=_BENZENE, other_smiles=[_ASPIRIN])
        candidates = SmilesMutationSearch()(state)
        canonical = [Chem.MolToSmiles(Chem.MolFromSmiles(c.data)) for c in candidates]
        assert len(canonical) == len(set(canonical))

    def test_best_molecule_itself_is_excluded(self):
        """The best molecule should not be included among its own mutants."""
        state = _make_state(best_smiles=_BENZENE, other_smiles=[_ASPIRIN])
        candidates = SmilesMutationSearch()(state)
        best_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(_BENZENE))
        canonical = {Chem.MolToSmiles(Chem.MolFromSmiles(c.data)) for c in candidates}
        assert best_canonical not in canonical

    def test_max_candidates_truncates_pool(self):
        """max_candidates should cap the number of returned candidates."""
        state = _make_state(best_smiles=_ASPIRIN, other_smiles=[_BENZENE])
        candidates = SmilesMutationSearch(max_candidates=2)(state)
        assert len(candidates) <= 2

    def test_mutates_the_highest_labelled_molecule(self):
        """Mutants should be neighbours of the best-labelled SMILES, not the others."""
        aspirin_best_pool = {
            Chem.MolToSmiles(Chem.MolFromSmiles(c.data))
            for c in SmilesMutationSearch()(_make_state(best_smiles=_ASPIRIN, other_smiles=[_BENZENE]))
        }
        benzene_best_pool = {
            Chem.MolToSmiles(Chem.MolFromSmiles(c.data))
            for c in SmilesMutationSearch()(_make_state(best_smiles=_BENZENE, other_smiles=[_ASPIRIN]))
        }
        # Swapping which molecule has the highest label should swap which
        # molecule gets mutated, so the two candidate pools must differ.
        assert aspirin_best_pool != benzene_best_pool

    def test_works_with_protocol_search(self):
        """SmilesMutationSearch should compose with ProtocolSearch from alf_core."""
        state = _make_state(best_smiles=_BENZENE, other_smiles=[_ASPIRIN])
        search_fn = ProtocolSearch(protocol=SmilesMutationSearch())
        candidates = search_fn(state)
        assert len(candidates) > 0
        assert all(isinstance(c, Candidate) for c in candidates)

    def test_empty_train_dataset_raises_clear_error(self):
        """An empty training set should raise a clear ValueError, not crash downstream."""
        fake_state = SimpleNamespace(
            dataset=SimpleNamespace(
                train_dataset=SimpleNamespace(candidates=[], labels=np.array([]))
            )
        )
        with pytest.raises(ValueError, match="at least one training candidate"):
            SmilesMutationSearch()(fake_state)

    def test_no_valid_mutations_raises_clear_error(self):
        """If every mutation is invalid/degenerate, raise instead of returning an empty pool."""
        state = _make_state(best_smiles="C", other_smiles=[])
        with pytest.raises(ValueError, match="no valid, novel"):
            SmilesMutationSearch(alphabet="C")(state)


class TestSmilesMutationSearchTopK:
    """Tests for the top_k multi-basis mutation behaviour."""

    def test_top_k_default_matches_single_best_behaviour(self):
        """top_k=1 (the default) should only mutate the single best molecule."""
        state = _make_ranked_state([_ASPIRIN, _BENZENE])
        default_pool = _canonical_set(SmilesMutationSearch()(state))
        explicit_pool = _canonical_set(SmilesMutationSearch(top_k=1)(state))
        assert default_pool == explicit_pool

    def test_top_k_two_includes_mutants_of_the_second_best(self):
        """top_k=2 should include mutants that only the second-best molecule can produce."""
        state = _make_ranked_state([_ASPIRIN, _BENZENE])

        top1_pool = _canonical_set(SmilesMutationSearch(top_k=1)(state))
        top2_pool = _canonical_set(SmilesMutationSearch(top_k=2)(state))

        # Mutants of benzene alone (computed by making it the sole best molecule).
        benzene_only_pool = _canonical_set(
            SmilesMutationSearch()(_make_state(best_smiles=_BENZENE, other_smiles=[_ASPIRIN]))
        )
        benzene_specific = benzene_only_pool - top1_pool

        assert benzene_specific, "expected at least one mutant unique to benzene's neighbourhood"
        assert benzene_specific.issubset(top2_pool)
        assert top2_pool.issuperset(top1_pool)

    def test_top_k_excludes_all_base_molecules(self):
        """None of the top-K base molecules themselves should appear as mutants."""
        state = _make_ranked_state([_ASPIRIN, _BENZENE])
        pool = _canonical_set(SmilesMutationSearch(top_k=2)(state))
        assert Chem.MolToSmiles(Chem.MolFromSmiles(_ASPIRIN)) not in pool
        assert Chem.MolToSmiles(Chem.MolFromSmiles(_BENZENE)) not in pool

    def test_top_k_larger_than_dataset_is_clamped(self):
        """top_k greater than the training set size should not raise."""
        state = _make_ranked_state([_ASPIRIN, _BENZENE])
        candidates = SmilesMutationSearch(top_k=100)(state)
        assert len(candidates) > 0
