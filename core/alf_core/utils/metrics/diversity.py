# Copyright 2023 InstaDeep Ltd. All rights reserved.
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

"""Diversity metrics for acquired candidate batches.

These metrics operate on lists of :class:`~alf_core.dataclasses.candidate.Candidate`
objects rather than on prediction arrays, so they are not registered in the
standard regression or classification registries.
"""

from difflib import SequenceMatcher

import numpy as np
from alf_core.dataclasses.candidate import Candidate, Modality
from scipy.spatial.distance import pdist


def intra_batch_diversity(candidates: list[Candidate]) -> dict[str, float]:
    """Compute average pairwise dissimilarity within a batch of candidates.

    Measures how spread out the acquired batch is in the design space.
    Low diversity indicates the surrogate model is proposing near-duplicate
    candidates, which wastes the oracle budget.

    Dissimilarity is computed as follows depending on candidate modality:

    - **SEQUENCE**: normalised edit distance derived from
      :class:`difflib.SequenceMatcher`.  `dissimilarity = 1 - ratio`,
      where `ratio` is 0 for completely different sequences and 1 for
      identical ones.
    - **EMBEDDING** / **TABULAR**: cosine distance computed via
      :func:`scipy.spatial.distance.pdist`.  Candidates are flattened to 1-D
      feature vectors before comparison.

    The average of all pairwise dissimilarities is returned.

    Args:
        candidates: List of :class:`~alf_core.dataclasses.candidate.Candidate`
            objects representing the candidates acquired in one round.  All
            candidates must share the same modality.

    Returns:
        Dictionary with key `intra_batch_diversity` mapping to the average
        pairwise dissimilarity in [0, 1].  Returns an empty dict when fewer
        than 2 candidates are provided.

    Raises:
        ValueError: If candidates span multiple modalities, if the modality
            is not one of SEQUENCE, EMBEDDING, or TABULAR, or if any candidate
            has an all-zero feature vector (cosine distance undefined).
    """
    if len(candidates) < 2:
        return {}

    modalities = {c.modality for c in candidates}
    if len(modalities) > 1:
        raise ValueError(
            f"intra_batch_diversity requires all candidates to share the same modality, "
            f"got {modalities}"
        )

    modality = candidates[0].modality

    if modality == Modality.SEQUENCE:
        pairs = [
            1.0 - SequenceMatcher(None, str(candidates[i].data), str(candidates[j].data)).ratio()
            for i in range(len(candidates))
            for j in range(i + 1, len(candidates))
        ]
        return {"intra_batch_diversity": float(np.mean(pairs))}

    if modality in (Modality.EMBEDDING, Modality.TABULAR):
        try:
            features = np.stack([np.asarray(c.data).flatten().astype(float) for c in candidates])
        except Exception as exc:
            raise ValueError(
                f"Cannot convert candidate data to a numeric feature matrix: {exc}"
            ) from exc
        if np.any(np.all(features == 0, axis=1)):
            raise ValueError(
                "intra_batch_diversity: one or more candidates have an all-zero feature "
                "vector; cosine distance is undefined."
            )
        distances = pdist(features, metric="cosine")
        return {"intra_batch_diversity": float(np.mean(distances))}

    raise ValueError(
        f"intra_batch_diversity does not support modality '{modality}'. "
        f"Supported modalities: SEQUENCE, EMBEDDING, TABULAR."
    )
