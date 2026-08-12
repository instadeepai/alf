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

"""Unit tests for BaseModel.embed()'s default behaviour and Surrogate.embed() delegation."""

from typing import Any

import numpy as np
import pytest
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions, Surrogate


class _FeaturiseOnlyModel(BaseModel):
    """Model that only implements featurise(); embed() is not overridden."""

    def featurise(self, inputs: list[Candidate] | LabelledCandidates) -> np.ndarray:
        """Return a (n, 2) array of ones for the given inputs.

        Args:
            inputs: Candidates or labelled candidates to featurise.

        Returns:
            Array of shape (n, 2) filled with ones.
        """
        candidates = inputs if isinstance(inputs, list) else inputs.candidates
        return np.ones((len(candidates), 2))

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Return zero-mean predictions.

        Args:
            candidate_points: Candidates to predict for.

        Returns:
            Predictions with zero means.
        """
        return Predictions(means=np.zeros(len(candidate_points)))

    def train(self, train_data: LabelledCandidates, val_data: Any = None) -> None:
        """No-op training.

        Args:
            train_data: Training data (unused).
            val_data: Validation data (unused).
        """

    def sample(self, condition: Any = None) -> list[Candidate]:
        """Not implemented for this test model.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError


class _EmbedEqualsFeaturiseModel(_FeaturiseOnlyModel):
    """Model that opts in to embed() by explicitly delegating to featurise()."""

    def embed(self, inputs: list[Candidate] | LabelledCandidates) -> np.ndarray:
        """Return featurise()'s output as the embedding.

        Args:
            inputs: Candidates or labelled candidates to embed.

        Returns:
            Array of shape (n, 2) filled with ones (featurise()'s output).
        """
        return self.featurise(inputs)


class _TokenizingModel(_FeaturiseOnlyModel):
    """Model where featurise() returns tokenized inputs (not embeddings), like ESM2Model.

    Overrides embed() to return proper numeric embeddings, distinct from featurise().
    """

    def featurise(self, inputs: list[Candidate] | LabelledCandidates) -> dict[str, Any]:
        """Return a dict of fake tokenized tensors, not a numeric embedding array.

        Args:
            inputs: Candidates or labelled candidates to featurise.

        Returns:
            Dict with a single 'input_ids' key.
        """
        candidates = inputs if isinstance(inputs, list) else inputs.candidates
        return {"input_ids": [0] * len(candidates)}

    def embed(self, inputs: list[Candidate] | LabelledCandidates) -> np.ndarray:
        """Return a (n, 3) array of twos, distinct from featurise()'s output.

        Args:
            inputs: Candidates or labelled candidates to embed.

        Returns:
            Array of shape (n, 3) filled with twos.
        """
        candidates = inputs if isinstance(inputs, list) else inputs.candidates
        return np.full((len(candidates), 3), 2.0)


class _RaisingEmbedModel(_FeaturiseOnlyModel):
    """Model whose embed() always raises RuntimeError."""

    def embed(self, inputs: list[Candidate] | LabelledCandidates) -> np.ndarray:
        """Raise RuntimeError unconditionally.

        Args:
            inputs: Ignored.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("embed failed intentionally")


def _make_candidates(n: int = 2) -> list[Candidate]:
    """Create n simple sequence candidates.

    Args:
        n: Number of candidates.

    Returns:
        List of n Candidate objects.
    """
    return [Candidate(data=f"seq_{i}", modality="sequence") for i in range(n)]


def test_base_model_embed_raises_not_implemented_by_default() -> None:
    """BaseModel.embed() raises NotImplementedError unless a subclass overrides it."""
    model = _FeaturiseOnlyModel()
    candidates = _make_candidates(3)
    with pytest.raises(NotImplementedError, match="does not implement embed"):
        model.embed(candidates)


def test_surrogate_embed_propagates_not_implemented_by_default() -> None:
    """Surrogate.embed() propagates the model's default NotImplementedError."""
    surrogate = Surrogate(model=_FeaturiseOnlyModel())
    candidates = _make_candidates(3)
    with pytest.raises(NotImplementedError, match="does not implement embed"):
        surrogate.embed(candidates)


def test_surrogate_embed_delegates_to_model_opt_in() -> None:
    """Surrogate.embed() returns whatever the underlying model's embed() returns,
    once the model explicitly opts in by overriding embed().
    """
    surrogate = Surrogate(model=_EmbedEqualsFeaturiseModel())
    candidates = _make_candidates(3)
    result = surrogate.embed(candidates)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 2)


def test_surrogate_embed_uses_override_not_featurise() -> None:
    """When a model overrides embed() distinctly from featurise(), Surrogate.embed()
    returns the embed() output, not the featurise() output.
    """
    surrogate = Surrogate(model=_TokenizingModel())
    candidates = _make_candidates(2)

    embedded = surrogate.embed(candidates)
    featurised = surrogate.featurise(candidates)

    assert isinstance(embedded, np.ndarray)
    assert embedded.shape == (2, 3)
    assert np.all(embedded == 2.0)
    assert isinstance(featurised, dict)


def test_surrogate_embed_propagates_model_exception() -> None:
    """Surrogate.embed() does not swallow exceptions from model.embed()."""
    surrogate = Surrogate(model=_RaisingEmbedModel())
    with pytest.raises(RuntimeError, match="embed failed intentionally"):
        surrogate.embed(_make_candidates(1))
