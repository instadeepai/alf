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
from alf_core import Candidate, LabelledCandidates, Modality, Oracle, Predictions, State
from alf_core.dataset.base_dataset import BaseDataset, BaseDatasetConfig
from alf_core.surrogate.surrogate import Surrogate
from alf_core.utils.enums import ProblemType

from alf_tools.datasets.guacamol.guacamol_scoring import osimertinib_mpo
from alf_tools.models.guacamol_oracle import GuacaMolOracleModel, GuacaMolOracleModelConfig

_OSIMERTINIB_SMILES = "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34"
_VALID_SMILES = [_OSIMERTINIB_SMILES, "CC(=O)O", "c1ccccc1"]
_INVALID_SMILES = "NOTASMILES!!!!"


def _candidates(smiles_list: list[str]) -> list[Candidate]:
    return [Candidate(data=s, modality=Modality.MOLECULE) for s in smiles_list]


@pytest.fixture
def oracle_model() -> GuacaMolOracleModel:
    """GuacaMolOracleModel targeting osimertinib_mpo.

    Returns:
        A configured GuacaMolOracleModel instance.
    """
    return GuacaMolOracleModel(GuacaMolOracleModelConfig(task_name="osimertinib_mpo"))


class TestGuacaMolOracleModelPredict:
    """Tests for GuacaMolOracleModel.predict()."""

    def test_predict_matches_direct_scorer_call(self, oracle_model):
        """predict() means should match calling osimertinib_mpo directly per SMILES."""
        candidates = _candidates(_VALID_SMILES)
        predictions = oracle_model.predict(candidates)
        expected = np.array([osimertinib_mpo(s) for s in _VALID_SMILES])
        assert isinstance(predictions, Predictions)
        np.testing.assert_allclose(predictions.means, expected)

    def test_predict_scores_are_in_unit_interval(self, oracle_model):
        """All scores from predict() should be within [0, 1]."""
        predictions = oracle_model.predict(_candidates(_VALID_SMILES))
        assert np.all(predictions.means >= 0.0)
        assert np.all(predictions.means <= 1.0)

    def test_predict_returns_zero_for_invalid_smiles(self, oracle_model):
        """Invalid SMILES should score 0.0, not raise."""
        predictions = oracle_model.predict(_candidates([_INVALID_SMILES]))
        assert predictions.means[0] == pytest.approx(0.0)

    def test_predict_output_length_matches_input(self, oracle_model):
        """predict() should return exactly one score per candidate."""
        predictions = oracle_model.predict(_candidates(_VALID_SMILES))
        assert len(predictions.means) == len(_VALID_SMILES)

    def test_task_name_is_configurable(self):
        """A different task_name should use that task's scorer, not osimertinib_mpo."""
        from alf_tools.datasets.guacamol.guacamol_scoring import ranolazine_mpo

        model = GuacaMolOracleModel(GuacaMolOracleModelConfig(task_name="ranolazine_mpo"))
        predictions = model.predict(_candidates(_VALID_SMILES))
        expected = np.array([ranolazine_mpo(s) for s in _VALID_SMILES])
        np.testing.assert_allclose(predictions.means, expected)


class TestGuacaMolOracleModelNoOps:
    """Tests for the no-op / unsupported parts of the BaseModel contract."""

    def test_train_is_a_no_op(self, oracle_model):
        """train() should not raise and should not change predict() behaviour."""
        before = oracle_model.predict(_candidates(_VALID_SMILES)).means
        oracle_model.train(train_data=LabelledCandidates(candidates=[], labels=np.array([])))
        after = oracle_model.predict(_candidates(_VALID_SMILES)).means
        np.testing.assert_allclose(before, after)

    def test_featurise_is_a_no_op(self, oracle_model):
        """featurise() should not raise."""
        oracle_model.featurise(_candidates(_VALID_SMILES))

    def test_sample_raises_not_implemented(self, oracle_model):
        """sample() is unsupported: this oracle scores, it does not generate."""
        with pytest.raises(NotImplementedError):
            oracle_model.sample()


class _TinyMoleculeDataset(BaseDataset):
    """Minimal molecule dataset used only to build a State for Oracle tests."""

    def load_dataset(self) -> LabelledCandidates:
        """Load a tiny fixed set of molecule candidates with dummy labels.

        Returns:
            LabelledCandidates over a handful of valid SMILES.
        """
        candidates = _candidates(_VALID_SMILES)
        return LabelledCandidates(candidates=candidates, labels=np.zeros(len(candidates)))


@pytest.fixture
def state() -> State:
    """Minimal task State for Oracle.evaluate() tests.

    Returns:
        A State wrapping a tiny molecule dataset and a dummy surrogate.
    """
    config = BaseDatasetConfig(
        name="tiny_molecules",
        modality=Modality.MOLECULE,
        seed=0,
        train_ratio=0.5,
        validation_frac=0.0,
        test_ratio=0.0,
        split_type="random",
        problem_type=ProblemType.REGRESSION,
    )
    dataset = _TinyMoleculeDataset(config)
    dataset.setup()
    surrogate = Surrogate(model=GuacaMolOracleModel(GuacaMolOracleModelConfig(task_name="osimertinib_mpo")))
    return State(dataset=dataset, surrogate=surrogate)


class TestGuacaMolOracleModelViaOracle:
    """Tests for GuacaMolOracleModel wrapped in the Oracle interface."""

    def test_evaluate_returns_labelled_candidates_and_records_time(self, oracle_model, state):
        """Oracle.evaluate() should return LabelledCandidates and record oracle_time."""
        oracle = Oracle(scorer=oracle_model)
        candidates = _candidates(_VALID_SMILES)
        labelled, new_state = oracle.evaluate(candidates, state)
        assert isinstance(labelled, LabelledCandidates)
        assert len(labelled.candidates) == len(candidates)
        assert "oracle_time" in new_state.round_metrics.metrics

    def test_evaluate_labels_match_direct_scorer_call(self, oracle_model, state):
        """Labels returned via Oracle.evaluate() should match the raw scorer."""
        oracle = Oracle(scorer=oracle_model)
        candidates = _candidates(_VALID_SMILES)
        labelled, _ = oracle.evaluate(candidates, state)
        expected = np.array([osimertinib_mpo(s) for s in _VALID_SMILES])
        np.testing.assert_allclose(labelled.labels, expected)
