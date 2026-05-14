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

import numpy as np
from scipy.spatial.distance import cdist

from alf_core import AcquisitionFunction, Candidate, LabelledCandidates, State

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _to_numpy(features: object) -> np.ndarray:
    """Convert model features to a numpy array.

    Args:
        features: Feature output from a model's featurise method.

    Returns:
        Numpy array representation of the features.

    Raises:
        ValueError: If features is None or cannot be converted to a numpy array.
    """
    if features is None:
        raise ValueError(
            "featurise returned None. CoreSet requires a model with a featurise "
            "implementation that returns numerical embeddings."
        )
    if HAS_TORCH and isinstance(features, torch.Tensor):
        return features.detach().cpu().numpy()
    try:
        return np.asarray(features)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"featurise returned a value that cannot be converted to a numpy array: {e}"
        ) from e


class CoreSet(AcquisitionFunction):
    """Core-set acquisition function using greedy k-centres.

    Greedily selects candidates that maximise the minimum distance to
    the training set and previously selected candidates (greedy k-centres).

    Candidates are scored by their minimum L2 distance to all centres at
    the time of selection. Unselected candidates receive a score of 0.
    This is a maximising acquisition function.
    """

    def __call__(
        self,
        search_candidates: list[Candidate],
        state: State,
    ) -> LabelledCandidates:
        """Compute CoreSet acquisition values for unlabelled candidates.

        Args:
            search_candidates: List of unlabelled candidates to score.
            state: The task state containing the current datasets and surrogate model.

        Raises:
            ValueError: If featurise returns None or a non-array-like object.

        Returns:
            LabelledCandidates with CoreSet acquisition values.
        """
        training_candidates = state.dataset.train_dataset.candidates
        features = state.surrogate.model.featurise(training_candidates + search_candidates)
        embeddings = _to_numpy(features)

        n_train = len(training_candidates)
        training_embs = embeddings[:n_train]
        candidate_embs = embeddings[n_train:]

        n_cands = len(search_candidates)
        n_select = min(state.acq_batch_size, n_cands)

        min_dists = cdist(candidate_embs, training_embs).min(axis=1)
        selected_mask = np.zeros(n_cands, dtype=bool)
        acquisition_values = np.zeros(n_cands)

        for _ in range(n_select):
            masked = np.where(~selected_mask, min_dists, -np.inf)
            best_idx = int(np.argmax(masked))
            acquisition_values[best_idx] = min_dists[best_idx]
            selected_mask[best_idx] = True
            dists_to_new = cdist(candidate_embs, candidate_embs[[best_idx]])[:, 0]
            min_dists = np.minimum(min_dists, dists_to_new)

        return LabelledCandidates(candidates=search_candidates, labels=acquisition_values)
