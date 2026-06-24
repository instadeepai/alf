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

pytest.importorskip("mlip", reason="mlip not installed; install alf_tools[mlip]")

from alf_core import Candidate  # noqa: E402
from alf_core.dataclasses.labelled_candidates import LabelledCandidates  # noqa: E402
from alf_tools.models.mlip import MLIPModel, MLIPModelConfig, MLIPTrainConfig  # noqa: E402
from alf_tools.models.utils.mlip_utils import (  # noqa: E402
    _candidate_to_chemical_system,  # noqa: PLC2701
    _compute_batching_limits,  # noqa: PLC2701
    _filter_valid_graphs,  # noqa: PLC2701
)
from ase import Atoms  # noqa: E402
from mlip.graph import Graph  # noqa: E402


def _water() -> Atoms:
    """Build a small water molecule for graph/system construction.

    Returns:
        An ASE Atoms object for H2O with non-trivial geometry.
    """
    return Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])


def _scratch_model() -> MLIPModel:
    """Build an MLIPModel that trains from scratch (no pretrained download).

    Returns:
        An MLIPModel whose model_path is None, so __init__ performs no S3 download.
    """
    return MLIPModel(
        model_config=MLIPModelConfig(model_path=None),
        train_config=MLIPTrainConfig(),
    )


class TestDynamicTrainingSettings:
    """Tests for the documented dynamic training schedule."""

    def test_small_regime(self) -> None:
        """train_size <= 20 uses batch_size=1, lr=0.001."""
        model = _scratch_model()
        batch_size, lr, epochs = model._get_dynamic_training_settings(10)
        assert batch_size == 1
        assert lr == 0.001
        assert epochs == 100  # ceil(1000 * 1 / 10)

    def test_medium_regime(self) -> None:
        """21 <= train_size <= 100 uses batch_size=2, lr=0.005."""
        model = _scratch_model()
        batch_size, lr, epochs = model._get_dynamic_training_settings(50)
        assert batch_size == 2
        assert lr == 0.005
        assert epochs == 40  # ceil(1000 * 2 / 50)

    def test_large_regime(self) -> None:
        """train_size > 100 uses batch_size=4, lr=0.01."""
        model = _scratch_model()
        batch_size, lr, epochs = model._get_dynamic_training_settings(200)
        assert batch_size == 4
        assert lr == 0.01
        assert epochs == 20  # ceil(1000 * 4 / 200)

    def test_min_epochs_floor(self) -> None:
        """Epochs never drop below the minimum of 10 for very large datasets."""
        model = _scratch_model()
        _, _, epochs = model._get_dynamic_training_settings(1000)
        assert epochs == 10  # max(10, ceil(4000 / 1000))

    @pytest.mark.parametrize("size", [20, 21, 100, 101])
    def test_regime_boundaries(self, size: int) -> None:
        """Batch size steps at the documented 20/100 boundaries."""
        model = _scratch_model()
        batch_size, _, _ = model._get_dynamic_training_settings(size)
        expected = 1 if size <= 20 else 2 if size <= 100 else 4
        assert batch_size == expected


class TestCandidateToChemicalSystem:
    """Tests for converting candidates to mlip ChemicalSystems."""

    def test_charge_defaults_to_neutral(self) -> None:
        """A candidate with no charge feature defaults to charge 0."""
        candidate = Candidate(data=_water(), modality="structure")
        system = _candidate_to_chemical_system(candidate, energy=-1.5)
        assert system.charge == 0
        assert system.energy == -1.5
        assert system.forces is None
        assert len(system.atomic_numbers) == 3

    def test_charge_and_forces_passed_through(self) -> None:
        """Charge and forces features are forwarded to the ChemicalSystem."""
        forces = np.zeros((3, 3))
        candidate = Candidate(
            data=_water(), modality="structure", features={"charge": 2, "forces": forces}
        )
        system = _candidate_to_chemical_system(candidate, energy=0.0)
        assert system.charge == 2
        assert system.forces is not None
        assert system.forces.shape == (3, 3)


class TestGraphHelpers:
    """Tests for graph filtering and batching-limit computation."""

    def test_filter_drops_empty_and_none_graphs(self) -> None:
        """Single-atom (edgeless) graphs and None entries are filtered out."""
        single = _candidate_to_chemical_system(
            Candidate(data=Atoms("H", positions=[[0, 0, 0]]), modality="structure"), 0.0
        )
        water = _candidate_to_chemical_system(Candidate(data=_water(), modality="structure"), 0.0)
        g_single = Graph.from_chemical_system(single, 5.0)
        g_water = Graph.from_chemical_system(water, 5.0)
        assert int(g_single.n_edge.sum()) == 0
        assert int(g_water.n_edge.sum()) > 0

        filtered = _filter_valid_graphs([g_single, g_water, None])
        assert len(filtered) == 1
        assert filtered[0] is g_water

    def test_compute_batching_limits_positive(self) -> None:
        """Batching limits are positive and cover the provided systems."""
        water = _candidate_to_chemical_system(Candidate(data=_water(), modality="structure"), 0.0)
        graph = Graph.from_chemical_system(water, 5.0)
        max_n_node, max_n_edge = _compute_batching_limits([water, water], [graph, graph], 2)
        assert max_n_node >= 1
        assert max_n_edge >= 1


class TestTrainValDataRequired:
    """Tests for the val_data contract on train."""

    def test_train_rejects_none_val_data(self) -> None:
        """Train raises ValueError when val_data is None (BaseModel-conforming signature)."""
        model = _scratch_model()
        train_data = LabelledCandidates(candidates=[], labels=np.array([]))
        with pytest.raises(ValueError, match="requires val_data"):
            model.train(train_data, val_data=None)


class TestPredictWithForces:
    """Tests for predict_with_forces."""

    def test_empty_candidates_returns_empty(self) -> None:
        """No candidates yields an empty energy array and an empty force list."""
        model = _scratch_model()
        energies, forces = model.predict_with_forces([])
        assert energies.shape == (0,)
        assert forces == []

    def test_unpacks_energies_and_forces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Energies and per-structure forces are extracted from each prediction."""
        model = _scratch_model()

        class _Pred:
            def __init__(self, energy: float, forces: np.ndarray) -> None:
                self.energy = energy
                self.forces = forces

        fake = [
            _Pred(1.0, np.array([[0.1, 0.0, 0.0]])),
            _Pred(2.0, np.array([[0.0, 0.2, 0.0]])),
        ]
        monkeypatch.setattr(model, "_run_inference", lambda structures: fake)

        candidates = [
            Candidate(data=_water(), modality="structure"),
            Candidate(data=_water(), modality="structure"),
        ]
        energies, forces = model.predict_with_forces(candidates)

        np.testing.assert_array_equal(energies, np.array([1.0, 2.0]))
        assert len(forces) == 2
        np.testing.assert_array_equal(forces[0], np.array([[0.1, 0.0, 0.0]]))


class TestPerReactionMetrics:
    """Tests for _compute_and_log_per_reaction_metrics index alignment."""

    def test_energy_and_force_metrics_per_reaction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Per-reaction energy/forces metrics align predictions, labels, and masks."""
        model = _scratch_model()
        candidates = [
            Candidate(
                data=_water(),
                modality="structure",
                features={"reaction_id": rid, "forces": np.zeros((3, 3))},
            )
            for rid in (0, 0, 1, 1)
        ]
        true_energies = np.array([1.0, 2.0, 3.0, 4.0])
        model._test_data = LabelledCandidates(candidates=candidates, labels=true_energies)
        model._test_labels = true_energies

        pred_energies = np.array([1.3, 2.0, 2.7, 4.0])
        pred_forces = [np.full((3, 3), 0.1) for _ in range(4)]
        monkeypatch.setattr(model, "predict_with_forces", lambda c: (pred_energies, pred_forces))

        metrics = model._compute_and_log_per_reaction_metrics()

        # rxn 0 (3-atom systems): per-atom energy errors = [0.3/3, 0.0/3] = [0.1, 0.0].
        assert metrics["rxn00000/energy_rmse_per_atom"] == pytest.approx(np.sqrt(0.005))
        assert metrics["rxn00000/energy_mae_per_atom"] == pytest.approx(0.05)
        # Constant 0.1 force error everywhere.
        assert metrics["rxn00000/force_rmse"] == pytest.approx(0.1)
        assert metrics["rxn00000/force_mae"] == pytest.approx(0.1)
        # Both reactions reported.
        assert "rxn00001/energy_rmse_per_atom" in metrics
        assert "rxn00001/force_rmse" in metrics

    def test_returns_empty_without_reactions(self) -> None:
        """No reaction_id features yields an empty metrics dict."""
        model = _scratch_model()
        candidates = [Candidate(data=_water(), modality="structure")]
        model._test_data = LabelledCandidates(candidates=candidates, labels=np.array([1.0]))
        model._test_labels = np.array([1.0])
        assert model._compute_and_log_per_reaction_metrics() == {}


class TestTrainPredictFromScratch:
    """End-to-end smoke test: train a tiny from-scratch model and predict."""

    def _labelled(self, n: int) -> LabelledCandidates:
        """Build n water candidates with random forces and energies.

        Returns:
            LabelledCandidates of n 3-atom water systems.
        """
        rng = np.random.default_rng(0)
        candidates = [
            Candidate(
                data=_water(),
                modality="structure",
                features={"forces": rng.normal(size=(3, 3))},
            )
            for _ in range(n)
        ]
        labels = rng.normal(size=n)
        return LabelledCandidates(candidates=candidates, labels=labels)

    def test_train_then_predict(self) -> None:
        """Training one epoch from scratch yields a force field that predicts energies."""
        model = MLIPModel(
            model_config=MLIPModelConfig(model_path=None, num_channels=4),
            train_config=MLIPTrainConfig(
                epochs=1,
                batch_size=2,
                use_weight_flip=False,
                energy_weight=1.0,
                forces_weight=1.0,
            ),
        )
        train_data = self._labelled(4)
        val_data = self._labelled(2)

        model.train(train_data, val_data=val_data)
        assert model.force_field is not None

        preds = model.predict(train_data.candidates)
        assert preds.means.shape == (4,)
        assert np.all(np.isfinite(preds.means))
