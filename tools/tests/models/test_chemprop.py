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
import pytest
from alf_core import Candidate, LabelledCandidates
from alf_tools.models.chemprop import ChempropModel, ChempropModelConfig, ChempropTrainConfig

SMILES = ["C", "CC", "CCC", "c1ccccc1", "CCO", "CC(=O)O"]


def _make_labelled(smiles: list[str], labels: list[float]) -> LabelledCandidates:
    candidates = [Candidate(data=smi, modality="graph") for smi in smiles]
    return LabelledCandidates(candidates, np.array(labels))


@pytest.fixture
def fast_model() -> ChempropModel:
    """Create a small ChempropModel suitable for fast unit tests.

    Returns:
        ChempropModel with minimal hidden_size, depth, and num_epochs.
    """
    model_config = ChempropModelConfig(hidden_size=32, depth=1, ffn_num_layers=1)
    train_config = ChempropTrainConfig(batch_size=4, num_epochs=2)
    return ChempropModel(model_config=model_config, train_config=train_config, device="cpu")


@pytest.fixture
def train_data() -> LabelledCandidates:
    """Create a labelled training set from the SMILES list.

    Returns:
        LabelledCandidates containing all six SMILES with ascending labels.
    """
    return _make_labelled(SMILES, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


@pytest.fixture
def val_data() -> LabelledCandidates:
    """Create a labelled validation set from the first three SMILES.

    Returns:
        LabelledCandidates containing the first three SMILES with labels.
    """
    return _make_labelled(SMILES[:3], [1.5, 2.5, 3.5])


class TestChempropConfig:
    """Tests for ChempropModelConfig and ChempropTrainConfig defaults."""

    def test_model_config_defaults(self) -> None:
        """ChempropModelConfig should have the expected default field values."""
        cfg = ChempropModelConfig()
        assert cfg.hidden_size == 300
        assert cfg.depth == 3
        assert cfg.ffn_num_layers == 2
        assert cfg.dropout == 0.0
        assert cfg.aggregation == "mean"

    def test_train_config_defaults(self) -> None:
        """ChempropTrainConfig should have the expected default field values."""
        cfg = ChempropTrainConfig()
        assert cfg.learning_rate == 1e-3
        assert cfg.batch_size == 50
        assert cfg.num_epochs == 50
        assert cfg.optimizer == "adam"
        assert cfg.weight_init is None
        assert cfg.seed is None


class TestFeaturise:
    """Tests for ChempropModel.featurise()."""

    def test_featurise_labelled_candidates(
        self, fast_model: ChempropModel, train_data: LabelledCandidates
    ) -> None:
        """featurise() should extract SMILES from a LabelledCandidates object."""
        result = fast_model.featurise(train_data)
        assert result == SMILES

    def test_featurise_list_of_candidates(self, fast_model: ChempropModel) -> None:
        """featurise() should extract SMILES from a plain list of Candidate objects."""
        candidates = [Candidate(data=smi, modality="graph") for smi in SMILES]
        assert fast_model.featurise(candidates) == SMILES


class TestGuards:
    """Tests for pre-condition guards in ChempropModel."""

    def test_predict_before_train_raises(self, fast_model: ChempropModel) -> None:
        """predict() should raise RuntimeError when called before train()."""
        candidates = [Candidate(data="C", modality="graph")]
        with pytest.raises(RuntimeError, match="train"):
            fast_model.predict(candidates)

    def test_sample_raises_not_implemented(self, fast_model: ChempropModel) -> None:
        """sample() should always raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            fast_model.sample()
