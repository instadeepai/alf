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


from alf_core.dataclasses import LabelledCandidates


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
        acquired_candidates: Candidates that were acquired during optimization.
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
        acquired_candidates: Candidates that were acquired during optimization.

    Returns:
        Dictionary containing:
        - "optimizer/regret": The regret value (lower is better).
    """
    best_possible_candidate_label = init_candidate_pool.labels.max()
    best_acquired_candidate_label = acquired_candidates.labels.max()
    regret = best_possible_candidate_label - best_acquired_candidate_label
    return {"optimizer/regret": regret}
