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
    return ESM2Model(
        name="test_esm2", model_config=model_config, train_config=train_config, device="cpu"
    )


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


class TestPredict:
    def test_mean_pooling_shape(self, esm2_model, sample_data):
        predictions = esm2_model.predict(sample_data.candidates)
        hidden_dim = esm2_model.esm_model.config.hidden_size
        assert predictions.means.shape == (len(sample_data), hidden_dim)

    def test_cls_pooling_shape(self, model_config, train_config, sample_data):
        config = ESM2ModelConfig(model_id=MODEL_ID, pooling="cls")
        model = ESM2Model(
            name="cls_model", model_config=config, train_config=train_config, device="cpu"
        )
        predictions = model.predict(sample_data.candidates)
        hidden_dim = model.esm_model.config.hidden_size
        assert predictions.means.shape == (len(sample_data), hidden_dim)

    def test_last_hidden_state_pooling_shape(self, model_config, train_config, sample_data):
        config = ESM2ModelConfig(model_id=MODEL_ID, pooling="last_hidden_state")
        model = ESM2Model(
            name="lhs_model", model_config=config, train_config=train_config, device="cpu"
        )
        predictions = model.predict(sample_data.candidates)
        # Shape: (n_seqs, seq_len, hidden_dim) — seq_len includes special tokens and padding
        assert predictions.means.ndim == 3
        assert predictions.means.shape[0] == len(sample_data)
        assert predictions.means.shape[2] == model.esm_model.config.hidden_size

    def test_variances_are_none(self, esm2_model, sample_data):
        predictions = esm2_model.predict(sample_data.candidates)
        assert predictions.variances is None

    def test_embeddings_are_finite(self, esm2_model, sample_data):
        predictions = esm2_model.predict(sample_data.candidates)
        assert np.all(np.isfinite(predictions.means))

    def test_repr_layer_produces_different_embeddings(self, train_config, sample_data):
        config_final = ESM2ModelConfig(model_id=MODEL_ID, repr_layer=-1)
        config_first = ESM2ModelConfig(model_id=MODEL_ID, repr_layer=1)
        model_final = ESM2Model(
            name="final", model_config=config_final, train_config=train_config, device="cpu"
        )
        model_first = ESM2Model(
            name="first", model_config=config_first, train_config=train_config, device="cpu"
        )
        preds_final = model_final.predict(sample_data.candidates)
        preds_first = model_first.predict(sample_data.candidates)
        assert not np.allclose(preds_final.means, preds_first.means)


class TestTrainFrozen:
    def test_frozen_train_is_noop(self, esm2_model, sample_data):
        """When freeze_backbone=True, train() must not change any weights."""
        initial_params = {
            name: param.clone() for name, param in esm2_model.esm_model.named_parameters()
        }
        esm2_model.train(sample_data)
        for name, param in esm2_model.esm_model.named_parameters():
            assert torch.equal(initial_params[name], param), f"Parameter {name} changed"

    def test_frozen_epoch_metrics_empty(self, esm2_model, sample_data):
        esm2_model.train(sample_data)
        assert esm2_model.get_epoch_metrics() == []

    def test_frozen_summary_metrics_empty(self, esm2_model, sample_data):
        esm2_model.train(sample_data)
        assert esm2_model.get_training_summary_metrics() == {}
