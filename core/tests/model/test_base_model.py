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

from typing import Any

import pytest
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions


class ModelWithoutEmbed(BaseModel):
    """Satisfies BaseModel's abstract contract without overriding embed()."""

    def featurise(self, inputs: list[Candidate] | LabelledCandidates) -> Any:
        raise NotImplementedError

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        raise NotImplementedError

    def train(self, train_data: LabelledCandidates, val_data: Any = None) -> None:
        raise NotImplementedError

    def sample(self, condition: Any = None) -> list[Candidate]:
        raise NotImplementedError


class TestBaseModel:
    """Tests for BaseModel's default method implementations."""

    def test_embed_raises_not_implemented_by_default(self) -> None:
        """embed() raises NotImplementedError unless a subclass overrides it."""
        model = ModelWithoutEmbed()
        candidates = [Candidate(data="seq_0", modality="sequence")]
        with pytest.raises(NotImplementedError, match="does not implement embed"):
            model.embed(candidates)
