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
from alf_core import Candidate, LabelledCandidates, Modality
from alf_tools.optimizer.search.single_mutant_search import SingleMutantSearch

# A tiny alphabet keeps the enumerated neighbourhoods small enough to assert on exactly.
_ABC = "ABC"


def _make_state(sequences: list[str], labels: list[float]) -> SimpleNamespace:
    """Build a minimal stand-in state exposing only what SingleMutantSearch reads."""
    return SimpleNamespace(
        dataset=SimpleNamespace(
            train_dataset=LabelledCandidates(
                candidates=[Candidate(data=s, modality=Modality.SEQUENCE) for s in sequences],
                labels=np.array(labels),
            )
        )
    )


def _single_mutants(sequence: str, alphabet: str) -> list[str]:
    """Enumerate every single-position substitution of a sequence, in generation order."""
    return [
        sequence[:i] + character + sequence[i + 1 :]
        for i in range(len(sequence))
        for character in alphabet
        if sequence[i] != character
    ]


def _data(candidates: list[Candidate]) -> list[str]:
    return [c.data for c in candidates]


class TestSingleMutantSearch:
    """Tests for the baseline single-seed behaviour of SingleMutantSearch."""

    def test_enumerates_neighbourhood_of_the_best_sequence(self):
        """The default pool should be exactly the best sequence's single-mutant set."""
        state = _make_state(["AAA", "BBB"], [1.0, 0.0])
        pool = _data(SingleMutantSearch(alphabet=_ABC)(state))
        assert pool == _single_mutants("AAA", _ABC)

    def test_mutants_differ_from_the_seed_in_exactly_one_position(self):
        """Every candidate should be a genuine single-point mutant of the seed."""
        state = _make_state(["ABCABC"], [1.0])
        for mutant in _data(SingleMutantSearch(alphabet=_ABC)(state)):
            assert len(mutant) == len("ABCABC")
            differences = sum(a != b for a, b in zip(mutant, "ABCABC"))
            assert differences == 1

    def test_seed_itself_is_not_returned(self):
        """The unmutated seed must not appear in its own neighbourhood."""
        state = _make_state(["AAA", "BBB"], [1.0, 0.0])
        assert "AAA" not in _data(SingleMutantSearch(alphabet=_ABC)(state))

    def test_candidates_use_the_sequence_modality(self):
        """All returned candidates should be tagged as sequences."""
        state = _make_state(["AAA"], [1.0])
        pool = SingleMutantSearch(alphabet=_ABC)(state)
        assert pool
        assert all(c.modality == Modality.SEQUENCE for c in pool)

    def test_selects_the_highest_labelled_sequence_not_the_first(self):
        """Ranking should follow labels, not training-set position."""
        state = _make_state(["AAA", "BBB"], [0.0, 1.0])
        pool = _data(SingleMutantSearch(alphabet=_ABC)(state))
        assert pool == _single_mutants("BBB", _ABC)

    def test_empty_train_dataset_raises_clear_error(self):
        """An empty training set should raise a clear ValueError."""
        state = _make_state([], [])
        with pytest.raises(ValueError, match="at least one training candidate"):
            SingleMutantSearch(alphabet=_ABC)(state)


class TestSingleMutantSearchTopK:
    """Tests for the top_k multi-seed behaviour."""

    def test_top_k_one_is_the_default(self):
        """Passing top_k=1 explicitly should match the default exactly."""
        state = _make_state(["AAA", "BBB", "CCC"], [1.0, 0.5, 0.2])
        default_pool = _data(SingleMutantSearch(alphabet=_ABC)(state))
        explicit_pool = _data(SingleMutantSearch(alphabet=_ABC, top_k=1)(state))
        assert default_pool == explicit_pool

    def test_top_k_one_reproduces_legacy_argmax_selection(self):
        """top_k=1 must seed from the same sequence that labels.argmax() would pick."""
        sequences = ["AAA", "BBB", "CCC", "ABC"]
        labels = [1.0, 3.0, 2.0, 3.0]
        state = _make_state(sequences, labels)
        expected_seed = sequences[int(np.array(labels).argmax())]
        pool = _data(SingleMutantSearch(alphabet=_ABC, top_k=1)(state))
        assert pool == _single_mutants(expected_seed, _ABC)

    def test_top_k_two_unions_both_neighbourhoods(self):
        """top_k=2 should return the union of the top two seeds' neighbourhoods."""
        state = _make_state(["AAA", "BBB", "CCC"], [1.0, 0.5, 0.2])
        pool = _data(SingleMutantSearch(alphabet=_ABC, top_k=2)(state))

        first = _single_mutants("AAA", _ABC)
        second = _single_mutants("BBB", _ABC)
        assert set(pool) == set(first) | set(second)
        # "CCC" is outside the top 2, so its neighbourhood-only members are absent.
        assert "CCB" not in pool

    def test_top_k_is_ordered_by_seed_rank(self):
        """Higher-ranked seeds' mutants should come first in the returned pool."""
        state = _make_state(["AAA", "BBB"], [1.0, 0.5])
        pool = _data(SingleMutantSearch(alphabet=_ABC, top_k=2)(state))
        first = _single_mutants("AAA", _ABC)
        assert pool[: len(first)] == first

    def test_pool_is_deduplicated_across_overlapping_seeds(self):
        """Overlapping neighbourhoods must not yield duplicate candidates."""
        # "AAA" and "AAB" are themselves single mutants of each other, so their
        # neighbourhoods overlap heavily.
        state = _make_state(["AAA", "AAB"], [1.0, 0.9])
        pool = _data(SingleMutantSearch(alphabet=_ABC, top_k=2)(state))
        assert len(pool) == len(set(pool))

    def test_dedup_keeps_every_distinct_mutant(self):
        """Deduplication must remove only repeats, never distinct sequences."""
        state = _make_state(["AAA", "AAB"], [1.0, 0.9])
        pool = _data(SingleMutantSearch(alphabet=_ABC, top_k=2)(state))
        expected = set(_single_mutants("AAA", _ABC)) | set(_single_mutants("AAB", _ABC))
        assert set(pool) == expected

    def test_mutant_equal_to_another_seed_appears_once(self):
        """A mutant that coincides with another seed should not be duplicated."""
        # "AAB" is a mutant of seed "AAA" and is itself the second seed.
        state = _make_state(["AAA", "AAB"], [1.0, 0.9])
        pool = _data(SingleMutantSearch(alphabet=_ABC, top_k=2)(state))
        assert pool.count("AAB") == 1

    def test_ordering_is_deterministic_across_repeated_calls(self):
        """Repeated calls must return an identically ordered pool."""
        state = _make_state(["AAA", "AAB", "ABC"], [1.0, 0.9, 0.8])
        search = SingleMutantSearch(alphabet=_ABC, top_k=3)
        assert _data(search(state)) == _data(search(state))

    def test_top_k_larger_than_dataset_is_clamped(self):
        """top_k above the training set size should use every sequence, not crash."""
        state = _make_state(["AAA", "BBB"], [1.0, 0.5])
        clamped = _data(SingleMutantSearch(alphabet=_ABC, top_k=100)(state))
        all_seeds = _data(SingleMutantSearch(alphabet=_ABC, top_k=2)(state))
        assert clamped == all_seeds

    @pytest.mark.parametrize("top_k", [0, -1, -5])
    def test_invalid_top_k_raises(self, top_k):
        """top_k below 1 should be rejected at construction time."""
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            SingleMutantSearch(alphabet=_ABC, top_k=top_k)


class TestSingleMutantSearchLabelShape:
    """Tests for how label array shapes are handled when ranking seeds."""

    def test_column_vector_labels_are_accepted(self):
        """Labels of shape (n, 1) are scalar labels and should rank normally."""
        state = _make_state(["AAA", "BBB"], [0.0, 0.0])
        state.dataset.train_dataset.labels = np.array([[0.0], [1.0]])
        pool = _data(SingleMutantSearch(alphabet=_ABC, top_k=1)(state))
        assert pool == _single_mutants("BBB", _ABC)

    def test_multi_output_labels_raise_instead_of_mis_ranking(self):
        """Genuinely multi-output labels should fail loudly rather than mis-select seeds."""
        state = _make_state(["AAA", "BBB"], [0.0, 0.0])
        state.dataset.train_dataset.labels = np.array([[1.0, 5.0], [3.0, 0.0]])
        with pytest.raises(ValueError, match="single scalar label per candidate"):
            SingleMutantSearch(alphabet=_ABC, top_k=1)(state)
