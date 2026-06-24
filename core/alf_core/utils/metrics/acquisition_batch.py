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

"""Metrics for the acquisition function on the acquired batch.

Covers both within-batch diversity (how spread out the batch is) and
pool-relative quality metrics (recall and regret against the full
candidate pool).
"""

from itertools import combinations

import numpy as np
from alf_core.dataclasses import LabelledCandidates
from alf_core.dataclasses.candidate import Candidate, Modality
from scipy.spatial.distance import pdist


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings.

    Returns:
        int: The minimum number of single-character edits (insertions,
            deletions, or substitutions) required to change string `a` into string `b`.
    """
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for ch_a in a:
        curr = [prev[0] + 1]
        for j, ch_b in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[-1] + 1, prev[j] + (ch_a != ch_b)))
        prev = curr
    return prev[-1]


def intra_batch_diversity(candidates: list[Candidate]) -> dict[str, float]:
    """Compute average pairwise dissimilarity within a batch of candidates.

    Measures how spread out the acquired batch is in the design space.
    Low diversity indicates the acquisition function is proposing near-duplicate
    candidates, which wastes the oracle budget.

    Dissimilarity is computed as follows depending on candidate modality:

    - **SEQUENCE**: normalised Levenshtein distance.
      `dissimilarity = levenshtein(a, b) / max(len(a), len(b))`,
      yielding 0 for identical sequences and 1 when every character must
      be substituted or the sequences differ by their full length.
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
        pairs = []
        for ci, cj in combinations(candidates, 2):
            a, b = str(ci.data), str(cj.data)
            max_len = max(len(a), len(b))
            dist = 0.0 if max_len == 0 else _levenshtein(a, b) / max_len
            pairs.append(dist)
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


def compute_recall(
    init_candidate_pool: LabelledCandidates,
    acquired_candidates: LabelledCandidates,
    top_percentile: float = 0.1,
    top_n: int = 100,
) -> dict[str, float]:
    """Compute recall metrics for acquired candidates.

    Measures how many of the acquired candidates are in the top performers of
    the initial candidate pool, using both percentile-based and top-N thresholds.

    Args:
        init_candidate_pool: Initial candidate pool before acquisition.
        acquired_candidates: Candidates acquired during optimization, excluding
            the initial labelled seed.
        top_percentile: Percentile threshold (e.g., 0.1 for top 10%). Defaults to 0.1.
        top_n: Number of top candidates to consider. Defaults to 100.

    Returns:
        Dictionary containing:
        - "optimizer/top_{top_percentile * 100:.0f}pc_recall": Recall at the
          top_percentile threshold
        - "optimizer/top_{top_n}_recall": Recall at the top_n threshold

        Rank counts are clamped to [1, pool size]; the key names always reflect
        the nominal top_percentile/top_n arguments, so for pools smaller than
        top_n the "top_{top_n}_recall" value is computed over the whole pool.
    """
    init_candidate_pool = init_candidate_pool.sort(ascending=False)
    pool_size = len(init_candidate_pool)

    # Clamp the rank thresholds so small pools don't produce a negative index
    # (int(pool_size * top_percentile) can be 0) or an out-of-range index
    # (top_n can exceed the pool size).
    top_percentile_count = max(1, int(pool_size * top_percentile))
    top_n_count = min(top_n, pool_size)

    top_percentile_threshold = init_candidate_pool[top_percentile_count - 1][1]
    top_n_threshold = init_candidate_pool[top_n_count - 1][1]

    top_percentile_recall = (
        sum(acquired_candidates.labels >= top_percentile_threshold) / top_percentile_count
    )
    top_n_recall = sum(acquired_candidates.labels >= top_n_threshold) / top_n_count

    # If there are candidates with the same label, e.g., the threshold label is Y and
    # there are more than top_percentile and/or top_n candidates with label equal to
    # or greater than Y, then the recall will be greater than 1 and thus needs to be
    # clipped at 1.
    top_percentile_recall = min(top_percentile_recall, 1)
    top_n_recall = min(top_n_recall, 1)

    return {
        f"optimizer/top_{top_percentile * 100:.0f}pc_recall": top_percentile_recall,
        f"optimizer/top_{top_n}_recall": top_n_recall,
    }


def compute_regret(
    init_candidate_pool: LabelledCandidates, acquired_candidates: LabelledCandidates
) -> dict[str, float]:
    """Compute regret of acquired candidates relative to the best possible candidate.

    Regret is the difference between the best possible label in the initial pool
    and the best label found in the acquired candidates.

    Args:
        init_candidate_pool: Initial candidate pool before acquisition.
        acquired_candidates: Candidates acquired during optimization, excluding
            the initial labelled seed.

    Returns:
        Dictionary containing:
        - "optimizer/regret": The regret value (lower is better).
    """
    best_possible_candidate_label = init_candidate_pool.labels.max()
    best_acquired_candidate_label = acquired_candidates.labels.max()
    regret = best_possible_candidate_label - best_acquired_candidate_label
    return {"optimizer/regret": regret}
