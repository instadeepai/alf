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
from alf_tools.models.mlp import MLPModel, MLPModelConfig, MLPTrainConfig


BENZENE = "c1ccccc1"
ETHANOL = "CCO"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def _make_smiles_candidates(smiles_list: list[str]) -> list[Candidate]:
    return [Candidate(data=s, modality="sequence") for s in smiles_list]


def _make_labelled(smiles_list: list[str], labels: list[float]) -> LabelledCandidates:
    return LabelledCandidates(
        candidates=_make_smiles_candidates(smiles_list),
        labels=np.array(labels, dtype=np.float32),
    )


def _make_precomputed_candidates(smiles_list: list[str]) -> list[Candidate]:
    """Candidates with MolWt and MolLogP pre-populated in features."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    candidates = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        features = {
            "MolWt": float(Descriptors.MolWt(mol)),
            "MolLogP": float(Descriptors.MolLogP(mol)),
        }
        candidates.append(Candidate(data=s, modality="sequence", features=features))
    return candidates


def _make_precomputed_labelled(smiles_list: list[str], labels: list[float]) -> LabelledCandidates:
    return LabelledCandidates(
        candidates=_make_precomputed_candidates(smiles_list),
        labels=np.array(labels, dtype=np.float32),
    )


SMILES = [BENZENE, ETHANOL, ASPIRIN, "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"]
LABELS = [0.1, 0.5, 0.3, 0.8, 0.6]


@pytest.fixture
def mlp_model():
    return MLPModel(
        model_config=MLPModelConfig(hidden_dims=[32, 16], dropout=0.0),
        train_config=MLPTrainConfig(batch_size=4, num_epochs=2),
        device="cpu",
    )


@pytest.fixture
def train_data():
    return _make_labelled(SMILES, LABELS)


@pytest.fixture
def train_data_precomputed():
    return _make_precomputed_labelled(SMILES, LABELS)


class TestMLPModelInit:
    def test_model_is_none_before_training(self, mlp_model):
        """Model attribute must be None before any training call."""
        assert mlp_model.model is None

    def test_epoch_metrics_empty_before_training(self, mlp_model):
        """get_epoch_metrics() must return an empty list before training."""
        assert mlp_model.get_epoch_metrics() == []

    def test_predict_before_train_raises(self, mlp_model):
        """predict() must raise RuntimeError with 'Model not trained' before train() is called."""
        candidates = _make_smiles_candidates([BENZENE])
        with pytest.raises(RuntimeError, match="Model not trained"):
            mlp_model.predict(candidates)


class TestMLPModelFeaturise:
    def test_featurise_smiles_returns_tensor(self, mlp_model):
        """featurise() must return a 2-D torch.Tensor with one row per candidate."""
        candidates = _make_smiles_candidates([BENZENE, ETHANOL])
        result = mlp_model.featurise(candidates)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 2
        assert result.shape[1] > 0

    def test_featurise_smiles_finite_values(self, mlp_model):
        """All values in the featurised SMILES tensor must be finite."""
        candidates = _make_smiles_candidates([BENZENE, ASPIRIN])
        result = mlp_model.featurise(candidates)
        assert torch.all(torch.isfinite(result))

    def test_featurise_precomputed_returns_tensor(self, mlp_model):
        """featurise() with pre-populated features must return a tensor of shape (n, num_features)."""
        candidates = _make_precomputed_candidates([BENZENE, ETHANOL])
        result = mlp_model.featurise(candidates)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 2)  # 2 precomputed features: MolWt, MolLogP

    def test_featurise_precomputed_values_match_rdkit(self, mlp_model):
        """Precomputed feature values in the tensor must match the RDKit-computed values."""
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(ETHANOL)
        expected_mw = float(Descriptors.MolWt(mol))
        expected_lp = float(Descriptors.MolLogP(mol))
        candidates = _make_precomputed_candidates([ETHANOL])
        result = mlp_model.featurise(candidates)
        # sorted keys: MolLogP < MolWt alphabetically
        assert torch.isclose(result[0, 0], torch.tensor(expected_lp), atol=1e-3)
        assert torch.isclose(result[0, 1], torch.tensor(expected_mw), atol=1e-3)

    def test_featurise_smiles_path_used_when_no_features(self, mlp_model):
        """Without pre-populated features, featurise() must use the SMILES fingerprint path (2057 dims)."""
        candidates = _make_smiles_candidates([BENZENE])
        result_smiles = mlp_model.featurise(candidates)
        # SMILES path: 2048-bit fingerprint + 9 descriptors = 2057
        assert result_smiles.shape[1] == 2057

    def test_featurise_precomputed_path_used_when_features_present(self, mlp_model):
        """With pre-populated features, featurise() must use only those features (2 dims here)."""
        candidates = _make_precomputed_candidates([BENZENE])
        result_precomputed = mlp_model.featurise(candidates)
        # Precomputed path: 2 features (MolLogP, MolWt)
        assert result_precomputed.shape[1] == 2

    def test_featurise_labelled_candidates(self, mlp_model, train_data):
        """featurise() must accept a LabelledCandidates object and return one row per sample."""
        result = mlp_model.featurise(train_data)
        assert result.shape[0] == len(train_data)

    def test_featurise_filters_string_features(self, mlp_model):
        """Candidates with 'split' string tag should still use only numeric features."""
        c = Candidate(
            data=BENZENE,
            modality="sequence",
            features={"MolWt": 78.11, "split": "train"},
        )
        result = mlp_model.featurise([c])
        assert result.shape == (1, 1)  # only MolWt is numeric


class TestMLPModelTrainPredict:
    def test_train_initialises_model(self, mlp_model, train_data):
        """train() must set the model attribute to a non-None value."""
        mlp_model.train(train_data)
        assert mlp_model.model is not None

    def test_predict_returns_predictions_object(self, mlp_model, train_data):
        """predict() must return a Predictions object with finite means of correct shape."""
        mlp_model.train(train_data)
        from alf_core import Predictions
        preds = mlp_model.predict(_make_smiles_candidates([BENZENE, ETHANOL]))
        assert isinstance(preds, Predictions)
        assert preds.means.shape == (2,)
        assert np.all(np.isfinite(preds.means))

    def test_predict_with_precomputed_features(self, mlp_model, train_data_precomputed):
        """predict() must work correctly when candidates have pre-populated features."""
        mlp_model.train(train_data_precomputed)
        candidates = _make_precomputed_candidates([BENZENE, ETHANOL])
        preds = mlp_model.predict(candidates)
        assert preds.means.shape == (2,)

    def test_train_with_validation_data(self, mlp_model, train_data):
        """Training with val_data must populate 'val_loss' in the summary metrics."""
        val_data = _make_labelled([BENZENE, ASPIRIN], [0.2, 0.7])
        mlp_model.train(train_data, val_data=val_data)
        metrics = mlp_model.get_training_summary_metrics()
        assert "val_loss" in metrics

    def test_train_without_validation_data(self, mlp_model, train_data):
        """Training without val_data must populate 'train_loss' but not 'val_loss'."""
        mlp_model.train(train_data)
        metrics = mlp_model.get_training_summary_metrics()
        assert "train_loss" in metrics
        assert "val_loss" not in metrics

    def test_train_loss_is_finite(self, mlp_model, train_data):
        """Train loss must be a finite float after training."""
        mlp_model.train(train_data)
        metrics = mlp_model.get_training_summary_metrics()
        assert np.isfinite(metrics["train_loss"])

    def test_epoch_metrics_length_matches_num_epochs(self, mlp_model, train_data):
        """get_epoch_metrics() must return exactly num_epochs entries after training."""
        mlp_model.train(train_data)
        assert len(mlp_model.get_epoch_metrics()) == mlp_model.train_config.num_epochs

    def test_epoch_metrics_reset_on_retrain(self, mlp_model, train_data):
        """Calling train() twice must reset epoch metrics so only the latest run is kept."""
        mlp_model.train(train_data)
        mlp_model.train(train_data)
        assert len(mlp_model.get_epoch_metrics()) == mlp_model.train_config.num_epochs

    def test_epoch_metrics_are_surrogate_epoch_metrics(self, mlp_model, train_data):
        """Every entry returned by get_epoch_metrics() must be a SurrogateEpochMetrics instance."""
        mlp_model.train(train_data)
        assert all(isinstance(em, SurrogateEpochMetrics) for em in mlp_model.get_epoch_metrics())

    def test_val_loss_none_without_val_data(self, mlp_model, train_data):
        """Without val_data, val_loss must be None on every SurrogateEpochMetrics."""
        mlp_model.train(train_data)
        for em in mlp_model.get_epoch_metrics():
            assert em.val_loss is None

    def test_val_loss_set_with_val_data(self, mlp_model, train_data):
        """With val_data, every SurrogateEpochMetrics must have a finite val_loss."""
        val_data = _make_labelled([BENZENE, ETHANOL], [0.1, 0.4])
        mlp_model.train(train_data, val_data=val_data)
        for em in mlp_model.get_epoch_metrics():
            assert em.val_loss is not None
            assert np.isfinite(em.val_loss)

    def test_sample_raises_not_implemented(self, mlp_model):
        """sample() must raise NotImplementedError as it is not implemented for MLPModel."""
        with pytest.raises(NotImplementedError):
            mlp_model.sample()
