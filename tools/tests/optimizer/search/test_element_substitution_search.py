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

import os
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from alf_core import Candidate, Modality
from alf_tools.optimizer.search.element_substitution_search import (
    ElementSubstitutionSearch,
    _ranking_key,  # noqa: PLC2701
)
from pymatgen.core import Lattice, Structure

# A generous allowlist covering the light main-group and 3d chemistry reachable from
# the tiny rocksalt parents used below. Deliberately excludes the actinides and heavy
# species the lambda table would otherwise happily propose.
ALLOWED = frozenset({
    "H",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Cs",
    "Ba",
    "Tl",
    "Pb",
    "Bi",
})


def _rocksalt(cation: str, anion: str, a: float = 4.02) -> Structure:
    """Build a rocksalt structure for the given cation/anion pair."""
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(a), [cation, anion], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def _fcc_metal(symbol: str, a: float = 3.61) -> Structure:
    """Build an elemental fcc metal, which bond-valence analysis cannot decorate."""
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), [symbol], [[0, 0, 0]])


def _candidate(structure: Structure) -> Candidate:
    return Candidate(data=structure.to_json(), modality=Modality.MATERIALS)


def _make_state(
    train: list[Structure],
    labels: list[float] | None = None,
    validation: list[Structure] | None = None,
) -> SimpleNamespace:
    """Build a minimal state stub with explicit train/validation splits.

    Labels default to strictly descending in input order, so the ranking walk visits
    the training structures in exactly the order given.
    """
    if labels is None:
        labels = [1.0 - 0.1 * i for i in range(len(train))]
    return SimpleNamespace(
        dataset=SimpleNamespace(
            train_dataset=SimpleNamespace(
                candidates=[_candidate(s) for s in train], labels=np.array(labels)
            ),
            validation_dataset=SimpleNamespace(
                candidates=[_candidate(s) for s in (validation or [])],
                labels=np.array([0.0] * len(validation or [])),
            ),
        )
    )


def _formulas(candidates: list[Candidate]) -> list[str]:
    """Reduced formulas of the returned candidates, in returned order."""
    return [
        Structure.from_str(c.data, fmt="json").composition.remove_charges().reduced_formula
        for c in candidates
    ]


class TestElementSubstitutionSearch:
    """End-to-end behaviour against the real Hautier lambda table."""

    def test_generates_novel_charge_balanced_children_of_lif(self):
        """A decorable rocksalt parent should yield novel, charge-balanced children."""
        search = ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=1)
        candidates = search(_make_state([_rocksalt("Li", "F")]))

        assert len(candidates) > 0
        for candidate in candidates:
            assert candidate.modality == Modality.MATERIALS
            structure = Structure.from_str(candidate.data, fmt="json")
            # Every child carries oxidation states that sum to zero: the charge
            # balance filter lives inside composition_prediction.
            charge = sum(site.specie.oxi_state for site in structure)
            assert charge == pytest.approx(0.0, abs=1e-8)

        # The parent must never be re-proposed as its own child.
        assert "LiF" not in _formulas(candidates)

    def test_children_are_unique_within_a_round(self):
        """No formula should be emitted twice in a single round."""
        candidates = ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=2)(
            _make_state([_rocksalt("Li", "F"), _rocksalt("Na", "Cl", a=5.64)])
        )
        formulas = _formulas(candidates)
        assert len(formulas) == len(set(formulas))

    def test_two_parents_cannot_both_emit_the_same_child(self):
        """Overlapping neighbourhoods must not double-propose a shared child.

        LiF and NaCl both reach a large shared region of alkali-halide space, so
        without claiming each identity as it is accepted the same formula would be
        emitted by both parents.
        """
        state = _make_state([_rocksalt("Li", "F"), _rocksalt("Na", "Cl", a=5.64)])
        formulas = _formulas(ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=2)(state))

        lif_only = set(
            _formulas(
                ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=1)(
                    _make_state([_rocksalt("Li", "F")])
                )
            )
        )
        nacl_only = set(
            _formulas(
                ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=1)(
                    _make_state([_rocksalt("Na", "Cl", a=5.64)])
                )
            )
        )
        overlap = lif_only & nacl_only
        assert overlap, "expected the two neighbourhoods to overlap for this test to bite"
        assert len(formulas) == len(set(formulas))

    def test_allowed_elements_filters_out_disallowed_children(self):
        """A child containing an element outside the allowlist must be discarded."""
        state = _make_state([_rocksalt("Li", "F")])
        permissive = set(
            _formulas(ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=1)(state))
        )
        assert "NaF" in permissive, "expected NaF to be reachable from LiF"

        restricted = set(
            _formulas(ElementSubstitutionSearch(allowed_elements=ALLOWED - {"Na"}, top_k=1)(state))
        )
        assert "NaF" not in restricted
        assert restricted < permissive

    def test_children_already_in_train_are_filtered(self):
        """A material already in the training split must not be re-proposed."""
        state = _make_state([_rocksalt("Li", "F"), _rocksalt("Na", "F")])
        formulas = _formulas(ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=2)(state))
        assert "NaF" not in formulas

    def test_children_already_in_validation_are_filtered(self):
        """A material sitting in the validation split must also be filtered.

        ALF divides each acquired batch between train and validation, so seeding
        `seen` from train alone leaks previously-acquired materials back in.
        """
        baseline = _formulas(
            ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=1)(
                _make_state([_rocksalt("Li", "F")])
            )
        )
        assert "NaF" in baseline

        with_val = _formulas(
            ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=1)(
                _make_state([_rocksalt("Li", "F")], validation=[_rocksalt("Na", "F")])
            )
        )
        assert "NaF" not in with_val


class TestRankingWalk:
    """The parent-selection loop, where the non-obvious correctness lives."""

    def test_walks_past_an_undecorable_parent(self):
        """An undecorable top-ranked parent must not abort or empty the round.

        Elemental fcc copper cannot be assigned valences by BVAnalyzer, but it
        outranks the decorable LiF below it. A fixed top-1 slice would return
        nothing; the walk must step past it.
        """
        state = _make_state([_fcc_metal("Cu"), _rocksalt("Li", "F")], labels=[1.0, 0.5])
        candidates = ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=1)(state)

        assert len(candidates) > 0
        # The children must come from LiF, the second-ranked parent.
        assert "NaF" in _formulas(candidates)

    def test_exhausted_parent_does_not_consume_the_quota(self):
        """A parent that decorates but yields nothing novel must not count as usable.

        This is the shard-killing bug: counting a parent merely for decorating
        successfully lets exhausted parents satisfy `top_k`, stopping the walk early
        with most of the ranking unexamined.
        """
        # Rank an exhausted parent first: LiF decorates fine, but every one of its
        # children is pre-registered in validation, so it contributes nothing novel.
        lif = _rocksalt("Li", "F")
        exhausted_children = [
            Structure.from_str(c.data, fmt="json")
            for c in ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=1)(
                _make_state([lif])
            )
        ]
        state = _make_state(
            [lif, _rocksalt("Na", "Cl", a=5.64)],
            labels=[1.0, 0.5],
            validation=exhausted_children,
        )

        candidates = ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=1)(state)

        # With the bug, LiF would consume the single quota slot and the round would
        # raise. Correct behaviour walks on to NaCl and returns its children.
        assert len(candidates) > 0
        produced = set(_formulas(candidates))
        assert produced.isdisjoint({s.composition.reduced_formula for s in exhausted_children})

    def test_top_k_counts_usable_parents_only(self):
        """top_k=2 with an undecorable parent in between should still use two parents."""
        state = _make_state(
            [_rocksalt("Li", "F"), _fcc_metal("Cu"), _rocksalt("Na", "Cl", a=5.64)],
            labels=[1.0, 0.9, 0.8],
        )
        two = _formulas(ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=2)(state))
        one = _formulas(ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=1)(state))
        # The second usable parent (NaCl, past the undecorable Cu) must contribute.
        assert len(two) > len(one)


class TestOrderingAndCapping:
    """Deterministic ordering and the post-sort cap."""

    def test_output_is_sorted_by_descending_probability(self):
        """Candidates must come back in descending lambda-table probability order."""
        search = ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=1)
        state = _make_state([_rocksalt("Li", "F")])

        # Recover the probabilities the search sorted on by re-running the internals.
        parent = Structure.from_str(state.dataset.train_dataset.candidates[0].data, fmt="json")
        scored = search._children_of(parent, set())
        by_identity = {identity: prob for prob, identity, _ in scored}

        probs = [by_identity[f] for f in _formulas(search(state))]
        assert probs == sorted(probs, reverse=True)

    def test_ordering_is_reproducible_across_instances(self):
        """Identical inputs must produce an identical output order.

        pymatgen's prediction order varies between processes, so probability alone
        does not pin the order down; the identity tie-break makes it total.
        """
        state = _make_state([_rocksalt("Li", "F")])
        first = _formulas(ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=1)(state))
        second = _formulas(
            ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=1)(
                _make_state([_rocksalt("Li", "F")])
            )
        )
        assert first == second

    def test_ordering_is_reproducible_across_processes(self):
        """Output order must survive a different PYTHONHASHSEED.

        pymatgen's `composition_prediction` returns predictions in a hash-seed
        dependent order, so this is the case the identity tie-break exists for. A
        subprocess with an explicit, different seed must produce the same ordering.
        """
        script = (
            "import warnings; warnings.filterwarnings('ignore')\n"
            "from pymatgen.core import Structure, Lattice\n"
            "from alf_tools.optimizer.search.element_substitution_search import "
            "ElementSubstitutionSearch, _ranking_key\n"
            f"allowed = {sorted(ALLOWED)!r}\n"
            "lif = Structure.from_spacegroup('Fm-3m', Lattice.cubic(4.02), ['Li','F'],"
            " [[0,0,0],[0.5,0.5,0.5]])\n"
            "s = ElementSubstitutionSearch(allowed_elements=allowed, top_k=1)\n"
            "out = sorted(s._children_of(lif, set()), key=_ranking_key)\n"
            "print(','.join(i[1] for i in out))\n"
        )
        orders = []
        for seed in ("0", "12345"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            result = subprocess.run(
                [sys.executable, "-c", script], check=False, capture_output=True, text=True, env=env
            )
            assert result.returncode == 0, result.stderr
            orders.append(result.stdout.strip().splitlines()[-1])

        assert orders[0] == orders[1]
        assert orders[0]

    def test_probability_ties_are_broken_by_identity(self):
        """Equal-probability children must be ordered by identity string.

        The tie-break has to make the order *total*, not merely stable: pymatgen
        yields predictions in a process-dependent order, so a sort keyed on
        probability alone would return tied children in whatever order they happened
        to be generated. This drives the same tied pair in from both input orders and
        requires the same output, which a stability-only sort cannot satisfy.
        """
        search = ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=1)
        parent = _rocksalt("Li", "F")
        scored = search._children_of(parent, set())

        # Exact float ties are rare in the real table, so force one: flatten every
        # probability to a single value. The identity tie-break in the production
        # sort key is then the only thing that can determine the order.
        entries = [(1.0, identity, structure) for _, identity, structure in scored]
        assert len(entries) > 1

        # Feeding the same entries in opposite input orders must give the same
        # result. A stability-only sort (probability alone) would not.
        forward = [identity for _, identity, _ in sorted(entries, key=_ranking_key)]
        backward = [identity for _, identity, _ in sorted(entries[::-1], key=_ranking_key)]
        assert forward == backward
        assert forward == sorted(identity for _, identity, _ in entries)

    def test_max_candidates_keeps_the_most_probable_children(self):
        """The cap is applied after sorting, so it keeps the top of the ranking."""
        state = _make_state([_rocksalt("Li", "F")])
        full = _formulas(ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=1)(state))
        capped = _formulas(
            ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=1, max_candidates=3)(
                _make_state([_rocksalt("Li", "F")])
            )
        )
        assert len(capped) == 3
        assert capped == full[:3]


class TestFailureHandling:
    """Error paths must stay legible rather than silently emptying the round."""

    def test_empty_training_set_raises(self):
        """An empty training set should raise a clear ValueError."""
        state = _make_state([])
        with pytest.raises(ValueError, match="at least one training candidate"):
            ElementSubstitutionSearch(allowed_elements=ALLOWED)(state)

    def test_all_parents_failing_raises_with_per_parent_outcomes(self):
        """If nothing is novel, the error must carry why each parent failed."""
        state = _make_state([_fcc_metal("Cu"), _fcc_metal("Ni", a=3.52)], labels=[1.0, 0.5])
        with pytest.raises(ValueError) as excinfo:
            ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=2)(state)

        message = str(excinfo.value)
        assert "no novel, allowed substitution" in message
        assert "Per-parent outcomes" in message
        # Both undecorable parents must be individually accounted for.
        assert message.count("parent ") >= 2
        assert "Valences cannot be assigned" in message

    def test_one_bad_parent_does_not_abort_the_round(self):
        """An unparseable training entry must be recorded, not raised."""
        state = _make_state([_rocksalt("Li", "F")])
        state.dataset.train_dataset.candidates.insert(
            0, Candidate(data="not-json", modality=Modality.MATERIALS)
        )
        state.dataset.train_dataset.labels = np.array([2.0, 1.0])

        candidates = ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=1)(state)
        assert len(candidates) > 0


class TestPickling:
    """The search must ship cheaply to worker processes."""

    def test_getstate_drops_the_predictor(self):
        """__getstate__ must clear the cached predictor."""
        search = ElementSubstitutionSearch(allowed_elements=ALLOWED)
        assert search.predictor is not None
        assert search.__dict__["_predictor"] is not None
        assert search.__getstate__()["_predictor"] is None

    def test_round_trips_through_pickle_after_use(self):
        """A used search must pickle and still work after unpickling."""
        search = ElementSubstitutionSearch(allowed_elements=ALLOWED, top_k=1)
        state = _make_state([_rocksalt("Li", "F")])
        before = _formulas(search(state))

        restored = pickle.loads(pickle.dumps(search))
        assert restored.__dict__["_predictor"] is None
        assert restored.allowed_elements == search.allowed_elements
        assert restored.threshold == search.threshold
        assert restored.top_k == search.top_k

        after = _formulas(restored(_make_state([_rocksalt("Li", "F")])))
        assert after == before
