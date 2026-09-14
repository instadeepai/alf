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

import warnings
from typing import Any, Iterable, List

import numpy as np
from alf_core import Candidate, Modality, SearchProtocol, State
from pymatgen.analysis.structure_prediction.substitution_probability import (
    SubstitutionPredictor,
)
from pymatgen.core import Structure
from pymatgen.transformations.standard_transformations import (
    AutoOxiStateDecorationTransformation,
    SubstitutionTransformation,
)

# The Hautier et al. (2011) lambda table is mined from the ICSD at this threshold in
# pymatgen's own defaults; substitutions rarer than this are dominated by data-mining
# noise rather than real chemistry.
DEFAULT_SUBSTITUTION_THRESHOLD = 1e-3


def _ranking_key(item: tuple[float, str, Structure]) -> tuple[float, str]:
    """Sort key ordering children by descending probability, then by identity.

    The identity component makes the order total rather than merely stable: pymatgen's
    prediction order is hash-seed dependent, so probability alone would order tied
    children differently between processes.

    Args:
        item: A ``(probability, identity, structure)`` tuple.

    Returns:
        A ``(-probability, identity)`` sort key.
    """
    probability, identity, _ = item
    return (-probability, identity)


def _identity(structure: Structure) -> str:
    """Return the deduplication identity of a structure.

    The oxidation-state-stripped reduced formula, so a freshly decorated child matches
    an undecorated parent from a dataset such as Matbench. A geometric
    ``StructureMatcher`` is unnecessary because substitution never alters geometry:
    same-formula candidates are the same material, not distinct polymorphs.

    Args:
        structure: Structure to identify.

    Returns:
        The oxidation-state-stripped reduced formula.
    """
    return structure.composition.remove_charges().reduced_formula


class ElementSubstitutionSearch(SearchProtocol):
    """Search protocol that swaps single elements in the best-labelled crystals.

    Parents are taken from the training structures in label order and one element is
    swapped for another, using the Hautier et al. (2011) substitution table
    (Inorg. Chem. 50(2):656-663, mined from the ICSD) to keep only swaps with precedent
    in known chemistry. Children are charge-balanced, restricted to ``allowed_elements``,
    deduplicated against everything already evaluated, and returned most-plausible first.

    ``top_k`` counts *usable* parents -- those yielding at least one new child -- rather
    than top-ranked ones, so parents that cannot be assigned oxidation states or whose
    substitutions are exhausted are skipped instead of consuming the quota.

    Two limits are worth knowing:

    - The number of distinct elements is preserved, so a binary parent yields binary
      children and the reachable space is bounded by the stoichiometries already in the
      training set.
    - The parent's lattice is reused without relaxation, leaving bond lengths wrong for
      the new chemistry. Harmless when the featuriser uses only composition, not for a
      structural one.
    """

    def __init__(
        self,
        allowed_elements: Iterable[str],
        threshold: float = DEFAULT_SUBSTITUTION_THRESHOLD,
        top_k: int = 3,
        max_candidates: int | None = None,
    ):
        """Initialize the element substitution search protocol.

        Args:
            allowed_elements: Element symbols the downstream oracle can score; children
                using anything else are discarded. Required: the substitution table
                reaches further up the periodic table than most scoring models, and an
                unscoreable child returns NaN, which reaches the training labels and
                costs the round.
            threshold: Minimum lambda-table probability for a substitution to be
                considered plausible.
            top_k: Number of *usable* parents to draw children from each round. A parent
                is usable only if it yields at least one novel child.
            max_candidates: Optional cap on the number of candidates returned, applied
                after sorting so a cut keeps the most plausible children.
        """
        self.allowed_elements = frozenset(allowed_elements)
        self.threshold = threshold
        self.top_k = top_k
        self.max_candidates = max_candidates
        self._predictor: SubstitutionPredictor | None = None

    @property
    def predictor(self) -> SubstitutionPredictor:
        """The substitution table predictor, built lazily and reused.

        Returns:
            The cached :class:`SubstitutionPredictor`.
        """
        # Construction parses the table and precomputes a 230x230 species
        # normalisation, so build once and share across rounds.
        if self._predictor is None:
            self._predictor = SubstitutionPredictor(threshold=self.threshold)
        return self._predictor

    def __getstate__(self) -> dict[str, Any]:
        """Drop the cached predictor so the search pickles cheaply to workers.

        Returns:
            The instance dictionary with the cached predictor cleared. It is rebuilt
            lazily on first use in the receiving process.
        """
        state = self.__dict__.copy()
        state["_predictor"] = None
        return state

    def _children_of(self, parent: Structure, seen: set[str]) -> list[tuple[float, str, Structure]]:
        """Generate the novel, allowed children of a single parent structure.

        Args:
            parent: Parent structure to substitute species on.
            seen: Identities already evaluated or accepted this round. Mutated in place,
                so two parents cannot both propose the same child.

        Returns:
            List of ``(probability, identity, structure)`` tuples for accepted children.

        Raises:
            ValueError: If oxidation-state decoration of the parent fails.
        """
        # Bond-valence analysis raises for metals and anything lacking BV parameters
        # (notably several actinides); the caller records it and walks on.
        decorated = AutoOxiStateDecorationTransformation().apply_transformation(parent)

        # Predict *outward* from the parent (`to_this_composition=False`). The
        # alternative, `Substitutor.pred_from_structures`, is a retrieval API needing the
        # target chemistry up front -- backwards here, where not knowing the target
        # chemistry is the whole problem -- and it silently returns an empty list for an
        # undecorated parent. This call also applies the charge-balance filter itself
        # (for rocksalt LiF, 481 raw species maps down to 65).
        predictions = self.predictor.composition_prediction(
            decorated.composition, to_this_composition=False
        )

        children = []
        for prediction in predictions:
            # Keys are the parent's species, values the replacements. Entries mapping a
            # species onto itself are not part of the swap; if nothing remains, the
            # prediction merely regenerates the parent, so skip it.
            swap = {
                species: replacement
                for species, replacement in prediction["substitutions"].items()
                if species != replacement
            }
            if not swap:
                continue

            try:
                child = SubstitutionTransformation(swap).apply_transformation(decorated)
            except Exception:
                # One bad swap must not kill the parent.
                continue

            symbols = {element.symbol for element in child.composition.remove_charges().elements}
            if not symbols <= self.allowed_elements:
                continue

            identity = _identity(child)
            if identity in seen:
                continue
            # Claim the identity immediately so a later parent cannot re-propose it.
            seen.add(identity)
            children.append((float(prediction["probability"]), identity, child))

        return children

    def __call__(self, state: State) -> List[Candidate]:
        """Generate novel, charge-balanced single-element substitutions of top parents.

        Args:
            state: The task state containing the dataset and surrogate model.

        Returns:
            Candidates holding serialized structures, most plausible first and truncated
            to ``max_candidates``.

        Raises:
            ValueError: If the training set is empty, or if no parent yielded a novel
                child. The per-parent outcomes are attached so the failure is legible.
        """
        train_dataset = state.dataset.train_dataset
        if len(train_dataset.candidates) == 0:
            raise ValueError(
                "ElementSubstitutionSearch requires at least one training candidate to "
                "substitute, but state.dataset.train_dataset is empty."
            )

        # Walk the whole ranking, not a fixed top-k slice: the aim is `top_k` *usable*
        # parents (see the class docstring).
        ranking = np.argsort(train_dataset.labels)[::-1]

        collected: list[tuple[float, str, Structure]] = []
        failures: list[str] = []
        usable_parents = 0

        # pymatgen's valence analysis and transformations warn liberally. Suppress only
        # inside the generation block: a global filter would silence warnings for any
        # other code sharing the process.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Seed from train *and* validation: ALF splits each acquired batch between
            # the two, so train alone leaks previously-acquired materials back in.
            seen: set[str] = set()
            for candidate in [
                *train_dataset.candidates,
                *state.dataset.validation_dataset.candidates,
            ]:
                try:
                    seen.add(_identity(Structure.from_str(candidate.data, fmt="json")))
                except Exception as error:
                    failures.append(f"unparseable evaluated candidate: {error}")

            for index in ranking:
                if usable_parents >= self.top_k:
                    break

                raw = train_dataset.candidates[index].data
                try:
                    parent = Structure.from_str(raw, fmt="json")
                except Exception as error:
                    failures.append(f"parent {index}: unparseable structure: {error}")
                    continue

                try:
                    children = self._children_of(parent, seen)
                except Exception as error:
                    failures.append(
                        f"parent {index} ({_safe_formula(parent)}): {type(error).__name__}: {error}"
                    )
                    continue

                if not children:
                    # Decorating successfully is not enough: an exhausted parent must
                    # not consume a quota slot and stop the walk early.
                    failures.append(f"parent {index} ({_safe_formula(parent)}): no novel children")
                    continue

                collected.extend(children)
                usable_parents += 1

        if not collected:
            outcomes = "\n  ".join(failures) if failures else "no parents examined"
            raise ValueError(
                "ElementSubstitutionSearch found no novel, allowed substitution of any "
                f"of the {len(ranking)} training structures, so the round has no "
                f"candidates. Per-parent outcomes:\n  {outcomes}"
            )

        # Descending probability, ties broken on identity so the order is reproducible
        # across processes; see `_ranking_key`.
        collected.sort(key=_ranking_key)

        if self.max_candidates is not None:
            collected = collected[: self.max_candidates]

        return [
            Candidate(data=structure.to_json(), modality=Modality.MATERIALS)
            for _, _, structure in collected
        ]


def _safe_formula(structure: Structure) -> str:
    """Return a structure's reduced formula for error messages, never raising.

    Args:
        structure: Structure to describe.

    Returns:
        The reduced formula, or ``"<unknown>"`` if it cannot be computed.
    """
    try:
        return _identity(structure)
    except Exception:
        return "<unknown>"
