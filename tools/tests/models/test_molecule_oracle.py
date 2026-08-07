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

import numpy as np
import pytest
from alf_core import Candidate, LabelledCandidates, Modality, Predictions

from alf_tools.models.molecule_oracle import MoleculeOracleModel

_SMILES = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "CCO"]


def _candidates(smiles_list: list[str]) -> list[Candidate]:
    return [Candidate(data=s, modality=Modality.MOLECULE) for s in smiles_list]


class TestMoleculeOracleModelPredict:
    """Tests for MoleculeOracleModel.predict() with an arbitrary user scorer."""

    def test_predict_uses_the_supplied_scorer(self):
        """predict() should call the supplied scorer once per candidate."""
        model = MoleculeOracleModel(scorer=lambda smiles: float(len(smiles)))
        predictions = model.predict(_candidates(_SMILES))
        assert isinstance(predictions, Predictions)
        np.testing.assert_allclose(predictions.means, [float(len(s)) for s in _SMILES])

    def test_predict_output_length_matches_input(self):
        """predict() should return exactly one score per candidate."""
        model = MoleculeOracleModel(scorer=lambda smiles: 1.0)
        predictions = model.predict(_candidates(_SMILES))
        assert len(predictions.means) == len(_SMILES)

    def test_predict_composes_with_a_guacamol_scorer(self):
        """A GuacaMol task scorer should work as a plain callable, unchanged."""
        from alf_tools.datasets.guacamol.guacamol_scoring import get_task_scorer

        scorer = get_task_scorer("osimertinib_mpo")
        model = MoleculeOracleModel(scorer=scorer)
        predictions = model.predict(_candidates(_SMILES))
        expected = np.array([scorer(s) for s in _SMILES])
        np.testing.assert_allclose(predictions.means, expected)


class TestMoleculeOracleModelNoOps:
    """Tests for the no-op / unsupported parts of the BaseModel contract."""

    def test_train_is_a_no_op(self):
        """train() should not raise and should not change predict() behaviour."""
        model = MoleculeOracleModel(scorer=lambda smiles: float(len(smiles)))
        before = model.predict(_candidates(_SMILES)).means
        model.train(train_data=LabelledCandidates(candidates=[], labels=np.array([])))
        after = model.predict(_candidates(_SMILES)).means
        np.testing.assert_allclose(before, after)

    def test_featurise_is_a_no_op(self):
        """featurise() should not raise."""
        model = MoleculeOracleModel(scorer=lambda smiles: 1.0)
        model.featurise(_candidates(_SMILES))

    def test_sample_raises_not_implemented(self):
        """sample() is unsupported: this oracle scores, it does not generate."""
        model = MoleculeOracleModel(scorer=lambda smiles: 1.0)
        with pytest.raises(NotImplementedError):
            model.sample()


class TestGuacaMolOracleModelIsAMoleculeOracleModel:
    """GuacaMolOracleModel should be a preset built on top of MoleculeOracleModel."""

    def test_guacamol_oracle_model_is_a_subclass(self):
        """GuacaMolOracleModel must remain a MoleculeOracleModel subclass."""
        from alf_tools.models.guacamol_oracle import GuacaMolOracleModel, GuacaMolOracleModelConfig

        model = GuacaMolOracleModel(GuacaMolOracleModelConfig(task_name="osimertinib_mpo"))
        assert isinstance(model, MoleculeOracleModel)
