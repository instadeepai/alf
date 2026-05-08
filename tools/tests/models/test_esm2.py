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
import torch
from alf_core import Candidate, LabelledCandidates
from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics
from alf_tools.models.esm2 import ESM2Model, ESM2ModelConfig, ESM2TrainConfig

MODEL_ID = "facebook/esm2_t6_8M_UR50D"


@pytest.fixture(scope="session")
def model_config():
    return ESM2ModelConfig(model_id=MODEL_ID)


@pytest.fixture(scope="session")
def train_config():
    return ESM2TrainConfig(freeze_backbone=True)


@pytest.fixture(scope="session")
def esm2_model(model_config, train_config):
    """Frozen ESM-2 model — downloaded once per test session."""
    return ESM2Model(name="test_esm2", model_config=model_config, train_config=train_config, device="cpu")


@pytest.fixture
def sample_data():
    sequences = ["ACDEFGHIKL", "MNPQRSTVWY", "ACMNPQRST"]
    candidates = [Candidate(data=seq, modality="sequence") for seq in sequences]
    labels = np.array([1.0, 2.0, 1.5])
    return LabelledCandidates(candidates, labels)


class TestConfigs:
    def test_model_config_requires_model_id(self):
        config = ESM2ModelConfig(model_id=MODEL_ID)
        assert config.model_id == MODEL_ID

    def test_model_config_defaults(self):
        config = ESM2ModelConfig(model_id=MODEL_ID)
        assert config.pooling == "mean"
        assert config.repr_layer == -1

    def test_train_config_defaults(self):
        config = ESM2TrainConfig()
        assert config.freeze_backbone is True
        assert config.learning_rate == 1e-4
        assert config.optimizer_type == "adamw"
        assert config.batch_size == 8
        assert config.num_epochs == 10
        assert config.mask_probability == 0.15
        assert config.log_frequency == 1


class TestFeaturise:
    def test_returns_dict_with_required_keys(self, esm2_model, sample_data):
        result = esm2_model.featurise(sample_data)
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_tensors_have_correct_batch_size(self, esm2_model, sample_data):
        result = esm2_model.featurise(sample_data)
        assert result["input_ids"].shape[0] == len(sample_data)
        assert result["attention_mask"].shape[0] == len(sample_data)

    def test_tensors_are_2d(self, esm2_model, sample_data):
        result = esm2_model.featurise(sample_data)
        assert result["input_ids"].ndim == 2
        assert result["attention_mask"].ndim == 2

    def test_accepts_list_of_candidates(self, esm2_model, sample_data):
        result = esm2_model.featurise(sample_data.candidates)
        assert result["input_ids"].shape[0] == len(sample_data)

    def test_returns_cpu_tensors(self, esm2_model, sample_data):
        result = esm2_model.featurise(sample_data)
        assert result["input_ids"].device.type == "cpu"
        assert result["attention_mask"].device.type == "cpu"
