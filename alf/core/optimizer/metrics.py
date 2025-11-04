from alf.core.dataclasses import LabeledCandidates
from typing import Dict

def compute_recall(init_candidate_pool: LabeledCandidates, acquired_candidates: LabeledCandidates, top_percentile: float = 0.1, top_n: int = 100) -> Dict[str, float]:
    """Compute how many of the acquired candidates are in the top percentile of the initial candidate pool."""
    init_candidate_pool = init_candidate_pool.sort(ascending=False)
    top_percentile_threshold = init_candidate_pool[int(len(init_candidate_pool) * top_percentile) - 1].labels[0]
    top_n_threshold = init_candidate_pool[top_n - 1].labels[0]

    top_percentile_recall = sum(acquired_candidates.labels >= top_percentile_threshold) / int(len(init_candidate_pool) * top_percentile)
    top_n_recall = sum(acquired_candidates.labels >= top_n_threshold) / top_n

    # If there are candidates with the same label, e.g., the threshold label is Y and 
    # there are more than top_percentile and/or top_n candidates with label equal to or greater than Y, 
    # then the recall will be greater than 1 and thus needs to be clipped at 1.
    top_percentile_recall = min(top_percentile_recall, 1)
    top_n_recall = min(top_n_recall, 1)

    return {"optimizer/top_percentile_recall": top_percentile_recall, "optimizer/top_n_recall": top_n_recall}

def compute_regret(init_candidate_pool: LabeledCandidates, acquired_candidates: LabeledCandidates) -> Dict[str, float]:
    """Compute the regret of the acquired candidates w.r.t. the best possible candidate in the initial candidate pool."""
    best_possible_candidate_label = init_candidate_pool.labels.max()
    best_acquired_candidate_label = acquired_candidates.labels.max()
    regret = best_possible_candidate_label - best_acquired_candidate_label
    return {"optimizer/regret": regret}
