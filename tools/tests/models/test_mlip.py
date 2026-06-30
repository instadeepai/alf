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

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mlip", reason="mlip not installed; install alf_tools[mlip]")

import alf_tools.models.mlip as mlip_module  # noqa: E402
from alf_core import Candidate  # noqa: E402
from alf_core.dataclasses.labelled_candidates import LabelledCandidates  # noqa: E402
from alf_tools.models.mlip import (  # noqa: E402
    MLIPModel,
    MLIPModelConfig,
    MLIPTrainConfig,
)
from alf_tools.models.utils.mlip_utils import (  # noqa: E402
    MODEL_TYPES,
    build_finetuning_graph_datasets,
    build_graph_dataset,
    build_graph_datasets,
    candidate_to_chemical_system,
    chemical_system_to_candidate,
    chemical_systems_to_labelled_candidates,
    load_extxyz_as_labelled_candidates,
)
from ase import Atoms  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402
from ase.io import write as ase_write  # noqa: E402
from mlip.data import ChemicalSystem, DatasetInfo  # noqa: E402
from mlip.models import Mace  # noqa: E402
from mlip.models.loss import HuberLoss  # noqa: E402
from mlip.training.optimizer_config import OptimizerConfig  # noqa: E402


def _water() -> dict[str, object]:
    """Build a small dependency-free water structure payload.

    Returns:
        A structure dictionary for H2O with non-trivial geometry.
    """
    return {
        "atomic_numbers": np.array([1, 1, 8]),
        "positions": np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float),
        "cell": None,
        "pbc": None,
    }


def _diatomic(atomic_number: int) -> dict[str, object]:
    """Build a small same-element diatomic structure payload."""
    return {
        "atomic_numbers": np.array([atomic_number, atomic_number]),
        "positions": np.array([[0, 0, 0], [0, 0, 1]], dtype=float),
        "cell": None,
        "pbc": None,
    }


def _train_config(num_epochs: int = 1) -> MLIPTrainConfig:
    """Build an explicit mlip-backed training config for tests."""
    return MLIPTrainConfig(
        optimizer_config=OptimizerConfig(),
        training_loop_config=mlip_module.TrainingLoop.Config(num_epochs=num_epochs),
    )


def _scratch_model() -> MLIPModel:
    """Build an MLIPModel that trains from scratch.

    Returns:
        An MLIPModel whose model_path is None.
    """
    return MLIPModel(
        model_config=MLIPModelConfig(model_path=None),
        train_config=_train_config(),
    )


class TestCandidateToChemicalSystem:
    """Tests for converting candidates to mlip ChemicalSystems."""

    def test_missing_charge_remains_unspecified(self) -> None:
        """A candidate with no charge data leaves the ChemicalSystem charge unset."""
        candidate = Candidate(data=_water(), modality="tabular")
        system = candidate_to_chemical_system(candidate)
        assert system.charge is None
        assert system.energy is None
        assert system.forces is None
        assert len(system.atomic_numbers) == 3

    def test_charge_data_and_force_features_passed_through(self) -> None:
        """Charge data and force features are forwarded to the ChemicalSystem."""
        forces = np.zeros((3, 3))
        candidate = Candidate(
            data=_water() | {"charge": 2},
            modality="tabular",
            features={"forces": forces},
        )
        system = candidate_to_chemical_system(candidate, energy=0.0)
        assert system.charge == 2
        assert system.forces is not None
        assert system.forces.shape == (3, 3)

    def test_chemical_system_round_trip_preserves_targets(self) -> None:
        """ChemicalSystem adapters preserve structure inputs and target arrays."""
        forces = np.ones((3, 3))
        system = ChemicalSystem(
            **_water(),
            energy=-1.25,
            forces=forces,
            charge=1,
        )

        candidate = chemical_system_to_candidate(system)
        assert "energy" not in candidate.data
        assert candidate.data["charge"] == 1
        assert "forces" not in candidate.data
        np.testing.assert_array_equal(candidate.features["forces"], forces)

        round_tripped = candidate_to_chemical_system(candidate)
        assert round_tripped.energy is None
        assert round_tripped.charge == 1
        np.testing.assert_array_equal(round_tripped.forces, forces)

        labelled_round_tripped = candidate_to_chemical_system(candidate, energy=-2.0)
        assert labelled_round_tripped.energy == -2.0

    def test_chemical_systems_to_labelled_candidates(self) -> None:
        """ChemicalSystem lists convert into labelled ALF candidates."""
        systems = [
            ChemicalSystem(**_diatomic(1), energy=2.0, forces=np.zeros((2, 3))),
            ChemicalSystem(**_diatomic(8), energy=6.0, forces=np.ones((2, 3))),
        ]

        labelled = chemical_systems_to_labelled_candidates(systems)

        assert len(labelled) == 2
        np.testing.assert_array_equal(labelled.labels, np.array([2.0, 6.0]))
        np.testing.assert_array_equal(
            labelled.candidates[1].features["forces"],
            np.ones((2, 3)),
        )

    def test_load_extxyz_as_labelled_candidates(self, tmp_path) -> None:
        """Extxyz loading delegates to mlip's reader and returns ALF data."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=2.0,
            forces=np.ones((2, 3)),
        )
        xyz_path = tmp_path / "systems.xyz"
        ase_write(xyz_path, [atoms], format="extxyz")

        labelled = load_extxyz_as_labelled_candidates(xyz_path)

        assert len(labelled) == 1
        assert labelled.labels[0] == pytest.approx(2.0)
        np.testing.assert_array_equal(labelled.candidates[0].data["atomic_numbers"], [1, 1])
        assert "forces" not in labelled.candidates[0].data
        np.testing.assert_array_equal(
            labelled.candidates[0].features["forces"],
            np.ones((2, 3)),
        )


class TestGraphDatasetConstruction:
    """Tests for ALF's graph dataset boundary around mlip builders."""

    def test_build_graph_dataset_uses_mlip_builder(self) -> None:
        """GraphDataset construction delegates graph creation and batching to mlip."""
        water = candidate_to_chemical_system(Candidate(data=_water(), modality="tabular"), 0.0)
        dataset = build_graph_dataset([water, water], cutoff=5.0, batch_size=2)

        assert dataset.max_n_node >= 1
        assert dataset.max_n_edge >= 1
        assert len(dataset.graphs) == 2

    def test_build_graph_datasets_defaults_missing_charges_to_zero(self) -> None:
        """Training graph builder normalises missing total charges to neutral."""
        water = candidate_to_chemical_system(Candidate(data=_water(), modality="tabular"), 0.0)
        datasets, dataset_info = build_graph_datasets(
            {"train": [water, water], "valid": [water]},
            cutoff=5.0,
            batch_size=2,
        )

        assert set(datasets) == {"train", "valid"}
        assert dataset_info.available_total_charges == [0]

    def test_build_finetuning_graph_datasets_uses_target_train_e0s(self) -> None:
        """Finetuning graph building reuses mlip's MULTI dataset-info merge."""
        hydrogen = ChemicalSystem(
            **_diatomic(1),
            energy=2.0,
            forces=np.zeros((2, 3)),
        )
        pretrained_info = DatasetInfo(
            atomic_energies_map={1: -1.0, 6: -6.0, 8: -8.0},
            total_charge_set={0, 1},
            graph_cutoff_angstrom=5.0,
        )

        datasets, dataset_info = build_finetuning_graph_datasets(
            {"train": [hydrogen, hydrogen], "valid": [hydrogen]},
            pretrained_dataset_info=pretrained_info,
            batch_size=2,
        )

        assert set(datasets) == {"train", "valid"}
        assert dataset_info is not pretrained_info
        assert isinstance(dataset_info.atomic_energies_map, dict)
        assert dataset_info.atomic_energies_map[1] == pytest.approx(1.0)
        assert dataset_info.atomic_energies_map[6] == -6.0
        assert dataset_info.atomic_energies_map[8] == -8.0

    def test_build_finetuning_graph_datasets_rejects_valid_species_absent_from_train(
        self,
    ) -> None:
        """Target validation/test species must already be present in target train."""
        hydrogen = ChemicalSystem(
            **_diatomic(1),
            energy=2.0,
            forces=np.zeros((2, 3)),
        )
        oxygen = ChemicalSystem(
            **_diatomic(8),
            energy=6.0,
            forces=np.zeros((2, 3)),
        )
        pretrained_info = DatasetInfo(
            atomic_energies_map={1: -1.0, 8: -8.0},
            graph_cutoff_angstrom=5.0,
        )

        with pytest.raises(ValueError, match="unseen atomic"):
            build_finetuning_graph_datasets(
                {"train": [hydrogen], "valid": [oxygen]},
                pretrained_dataset_info=pretrained_info,
                batch_size=2,
            )

    def test_build_finetuning_graph_datasets_rejects_species_absent_from_pretrained(
        self,
    ) -> None:
        """Target train species must already be covered by the pretrained z-table."""
        carbon = ChemicalSystem(
            **_diatomic(6),
            energy=12.0,
            forces=np.zeros((2, 3)),
        )
        pretrained_info = DatasetInfo(
            atomic_energies_map={1: -1.0, 8: -8.0},
            graph_cutoff_angstrom=5.0,
        )

        with pytest.raises(ValueError, match="absent from its z-table"):
            build_finetuning_graph_datasets(
                {"train": [carbon], "valid": [carbon]},
                pretrained_dataset_info=pretrained_info,
                batch_size=2,
            )

    def test_build_finetuning_graph_datasets_rejects_multi_head_pretrained_info(
        self,
    ) -> None:
        """Naive finetuning only supports scalar pretrained atomic energy maps."""
        hydrogen = ChemicalSystem(
            **_diatomic(1),
            energy=2.0,
            forces=np.zeros((2, 3)),
        )
        pretrained_info = DatasetInfo(
            dataset_name=["source_a", "source_b"],
            atomic_energies_map=[{1: -1.0}, {8: -8.0}],
            graph_cutoff_angstrom=5.0,
        )

        with pytest.raises(ValueError, match="single-head pretrained checkpoints"):
            build_finetuning_graph_datasets(
                {"train": [hydrogen], "valid": [hydrogen]},
                pretrained_dataset_info=pretrained_info,
                batch_size=2,
            )


class TestTrainValDataRequired:
    """Tests for the val_data contract on train."""

    def test_train_rejects_none_val_data(self) -> None:
        """Train raises ValueError when val_data is None (BaseModel-conforming signature)."""
        model = _scratch_model()
        train_data = LabelledCandidates(candidates=[], labels=np.array([]))
        with pytest.raises(ValueError, match="requires val_data"):
            model.train(train_data, val_data=None)


class TestTrainConfig:
    """Tests for MLIP training config construction."""

    def test_native_mlip_configs_are_required(self) -> None:
        """The ALF wrapper does not choose optimizer or epoch defaults."""
        with pytest.raises(TypeError):
            MLIPTrainConfig()

    def test_explicit_native_mlip_configs_are_preserved(self) -> None:
        """Explicit optimizer and training-loop config objects are used as-is."""
        optimizer_config = OptimizerConfig()
        training_loop_config = mlip_module.TrainingLoop.Config(num_epochs=3)

        train_config = MLIPTrainConfig(
            optimizer_config=optimizer_config,
            training_loop_config=training_loop_config,
        )

        assert train_config.optimizer_config is optimizer_config
        assert train_config.training_loop_config is training_loop_config


class TestModelTypeConfig:
    """Tests for architecture selection."""

    def test_supported_model_types_available(self) -> None:
        """The wrapper exposes the supported mlip architectures."""
        assert set(MODEL_TYPES) == {"mace", "nequip", "visnet", "esen"}

    def test_invalid_model_type_rejected(self) -> None:
        """Unknown architecture names fail before model loading or training."""
        with pytest.raises(ValueError, match="Unsupported MLIP model_type"):
            MLIPModel(
                model_config=MLIPModelConfig(model_path=None, model_type="not-a-model"),
                train_config=_train_config(),
            )

    def test_pretrained_model_path_is_loaded_directly(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Configured pretrained zip paths are delegated to mlip's model IO."""
        model_path = tmp_path / "model.zip"
        captured: dict[str, Any] = {}
        fake_force_field = SimpleNamespace(
            dataset_info=SimpleNamespace(atomic_energies_map={1: -1.0}),
        )

        def fake_load_mlip_force_field(model_type, load_path):
            captured["model_type"] = model_type
            captured["load_path"] = load_path
            return fake_force_field

        monkeypatch.setattr(mlip_module, "load_mlip_force_field", fake_load_mlip_force_field)

        model = MLIPModel(
            model_config=MLIPModelConfig(model_path=model_path, model_type="mace"),
            train_config=_train_config(),
        )

        assert captured["model_type"] == "mace"
        assert captured["load_path"] == model_path
        assert model._pretrained_force_field is fake_force_field
        assert model.force_field is None


class TestPredict:
    """Tests for MLIP prediction payloads."""

    def test_empty_candidates_returns_empty(self) -> None:
        """No candidates follows the core Predictions non-empty contract."""
        model = _scratch_model()
        with pytest.raises(AssertionError):
            model.predict([])

    def test_unpacks_energies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Energies are extracted into standard Predictions."""
        model = _scratch_model()

        class _Pred:
            def __init__(self, energy: float) -> None:
                self.energy = energy

        fake = [
            _Pred(1.0),
            _Pred(2.0),
        ]
        monkeypatch.setattr(model, "_run_inference", lambda structures: fake)

        candidates = [
            Candidate(data=_water(), modality="tabular"),
            Candidate(data=_water(), modality="tabular"),
        ]
        predictions = model.predict(candidates)

        np.testing.assert_array_equal(predictions.means, np.array([1.0, 2.0]))
        assert not hasattr(predictions, "forces")


class TestTrainFromScratch:
    """Fast unit tests for scratch-training setup."""

    def _labelled(self, n: int) -> LabelledCandidates:
        """Build n water candidates with random forces and energies.

        Returns:
            LabelledCandidates of n 3-atom water systems.
        """
        rng = np.random.default_rng(0)
        candidates = [
            Candidate(
                data=_water(),
                modality="tabular",
                features={"forces": rng.normal(size=(3, 3))},
            )
            for _ in range(n)
        ]
        labels = rng.normal(size=n)
        return LabelledCandidates(candidates=candidates, labels=labels)

    def test_train_wires_scratch_model_without_running_jax(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scratch train constructs the selected network without real JAX training."""
        network_config = Mace.Config(num_channels=4)
        training_loop_config = mlip_module.TrainingLoop.Config(num_epochs=1)
        optimizer_config = OptimizerConfig(
            init_learning_rate=2e-3,
            peak_learning_rate=3e-3,
            final_learning_rate=1e-3,
        )
        custom_loss = HuberLoss(stress_weight_schedule=lambda _: 0.0)
        model = MLIPModel(
            model_config=MLIPModelConfig(
                model_path=None,
                network_config=network_config,
            ),
            train_config=MLIPTrainConfig(
                batch_size=2,
                optimizer_config=optimizer_config,
                training_loop_config=training_loop_config,
                loss=custom_loss,
            ),
        )
        dataset_info = SimpleNamespace(graph_cutoff_angstrom=5.0)
        captured: dict[str, Any] = {}

        class FakeNetwork:
            def __init__(self, config, dataset_info_arg) -> None:
                captured["network_config"] = config
                captured["network_dataset_info"] = dataset_info_arg

        class FakeTrainingLoop:
            def __init__(self, **kwargs) -> None:
                captured["training_loop_kwargs"] = kwargs
                self.best_model = SimpleNamespace(
                    params={"trained": True},
                    predictor="trained-predictor",
                )

            def run(self) -> None:
                captured["training_loop_ran"] = True

            def test(self, test_set) -> None:
                captured["test_set"] = test_set

        model._mlip_model_cls = FakeNetwork
        monkeypatch.setattr(
            mlip_module,
            "build_graph_datasets",
            lambda systems_by_split, cutoff, batch_size: (
                {
                    "train": {
                        "systems": systems_by_split["train"],
                        "batch_size": batch_size,
                        "cutoff": cutoff,
                    },
                    "valid": {
                        "systems": systems_by_split["valid"],
                        "batch_size": batch_size,
                        "cutoff": cutoff,
                    },
                },
                dataset_info,
            ),
        )
        monkeypatch.setattr(
            mlip_module,
            "labelled_candidates_to_chemical_systems",
            lambda data: data.candidates,
        )
        monkeypatch.setattr(
            mlip_module.ForceField,
            "from_mlip_network",
            staticmethod(
                lambda mlip_network, seed: SimpleNamespace(
                    params={"initial": True},
                    predictor="initial-predictor",
                )
            ),
        )
        monkeypatch.setattr(mlip_module, "TrainingLoop", FakeTrainingLoop)

        def fake_get_default_mlip_optimizer(config):
            captured["optimizer_config"] = config
            return "optimizer"

        monkeypatch.setattr(
            mlip_module,
            "get_default_mlip_optimizer",
            fake_get_default_mlip_optimizer,
        )
        monkeypatch.setattr(mlip_module.jax, "device_put", lambda params: params)

        train_data = self._labelled(1)
        val_data = self._labelled(1)
        model.train(train_data, val_data=val_data)

        assert captured["training_loop_ran"] is True
        assert captured["network_dataset_info"] is dataset_info
        assert captured["network_config"] is network_config
        assert captured["optimizer_config"] is optimizer_config
        assert captured["training_loop_kwargs"]["optimizer"] == "optimizer"
        assert captured["training_loop_kwargs"]["config"] is training_loop_config
        assert captured["training_loop_kwargs"]["loss"] is custom_loss
        assert model.get_training_summary_metrics()["peak_learning_rate"] == 3e-3
        assert model.force_field is not None
        assert model.force_field.params == {"trained": True}

    def test_train_with_charge_embedding_defaults_missing_charges(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scratch training can initialise charge-embedding models from neutral data."""
        training_loop_config = mlip_module.TrainingLoop.Config(num_epochs=1)
        model = MLIPModel(
            model_config=MLIPModelConfig(
                model_path=None,
                network_config=Mace.Config(
                    num_channels=4,
                    use_total_charge_embedding=True,
                ),
            ),
            train_config=MLIPTrainConfig(
                batch_size=2,
                optimizer_config=OptimizerConfig(),
                training_loop_config=training_loop_config,
            ),
        )
        captured: dict[str, Any] = {}

        class FakeNetwork:
            def __init__(self, config, dataset_info_arg) -> None:
                captured["network_config"] = config
                captured["network_dataset_info"] = dataset_info_arg

        class FakeTrainingLoop:
            def __init__(self, **kwargs) -> None:
                captured["training_loop_kwargs"] = kwargs
                self.best_model = SimpleNamespace(
                    params={"trained": True},
                    predictor="trained-predictor",
                )

            def run(self) -> None:
                captured["training_loop_ran"] = True

        model._mlip_model_cls = FakeNetwork
        monkeypatch.setattr(
            mlip_module.ForceField,
            "from_mlip_network",
            staticmethod(
                lambda mlip_network, seed: SimpleNamespace(
                    params={"initial": True},
                    predictor="initial-predictor",
                )
            ),
        )
        monkeypatch.setattr(mlip_module, "TrainingLoop", FakeTrainingLoop)
        monkeypatch.setattr(
            mlip_module,
            "get_default_mlip_optimizer",
            lambda config: "optimizer",
        )
        monkeypatch.setattr(mlip_module.jax, "device_put", lambda params: params)

        model.train(self._labelled(2), val_data=self._labelled(1))

        dataset_info = captured["network_dataset_info"]
        assert getattr(dataset_info, "available_total_charges") == [0]
        assert getattr(captured["network_config"], "use_total_charge_embedding") is True
        assert captured["training_loop_ran"] is True


class TestFinetuning:
    """Fast unit tests for single-head finetuning setup."""

    def _labelled(
        self,
        n: int,
        atomic_numbers: np.ndarray | None = None,
    ) -> LabelledCandidates:
        """Build n labelled candidates for finetuning tests."""
        rng = np.random.default_rng(0)
        base = _water()
        if atomic_numbers is not None:
            base = base | {"atomic_numbers": atomic_numbers}
        atomic_numbers_array = np.asarray(base["atomic_numbers"])
        candidates = [
            Candidate(
                data=base,
                modality="tabular",
                features={"forces": rng.normal(size=(len(atomic_numbers_array), 3))},
            )
            for _ in range(n)
        ]
        labels = rng.normal(size=n)
        return LabelledCandidates(candidates=candidates, labels=labels)

    def test_train_uses_finetuning_builder_and_loaded_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Finetuning uses the retargeted DatasetInfo with loaded parameter shapes."""
        captured: dict[str, Any] = {}

        pretrained_info = DatasetInfo(
            atomic_energies_map={1: -1.0, 6: -6.0, 8: -8.0},
            total_charge_set={-1, 0, 1},
            graph_cutoff_angstrom=4.0,
        )
        target_info = pretrained_info.model_copy(
            update={"atomic_energies_map": {1: 1.0, 6: -6.0, 8: 3.0}}
        )

        @dataclass(frozen=True)
        class FakeNetwork:
            config: object
            dataset_info: DatasetInfo

        @dataclass(frozen=True)
        class FakePredictor:
            mlip_network: FakeNetwork

        pretrained = SimpleNamespace(
            dataset_info=pretrained_info,
            params={"source": True},
            predictor=FakePredictor(
                mlip_network=FakeNetwork(
                    config=SimpleNamespace(name="config"),
                    dataset_info=pretrained_info,
                )
            ),
            inference_context=None,
        )
        model = MLIPModel(
            model_config=MLIPModelConfig(model_path=None),
            train_config=MLIPTrainConfig(
                optimizer_config=OptimizerConfig(),
                training_loop_config=mlip_module.TrainingLoop.Config(num_epochs=1),
            ),
        )
        model._pretrained_force_field = pretrained

        def fake_build_finetuning_graph_datasets(
            systems_by_split,
            pretrained_dataset_info_arg,
            batch_size,
        ):
            captured["systems_by_split"] = systems_by_split
            captured["pretrained_dataset_info"] = pretrained_dataset_info_arg
            captured["batch_size"] = batch_size
            return (
                {
                    "train": "train-dataset",
                    "valid": "valid-dataset",
                },
                target_info,
            )

        monkeypatch.setattr(
            mlip_module,
            "build_finetuning_graph_datasets",
            fake_build_finetuning_graph_datasets,
        )
        monkeypatch.setattr(mlip_module.jax, "device_put", lambda params: params)

        class FakeTrainingLoop:
            def __init__(self, **kwargs) -> None:
                captured["training_loop_kwargs"] = kwargs
                self.best_model = SimpleNamespace(
                    params={"trained": True},
                    predictor=kwargs["force_field"].predictor,
                )

            def run(self) -> None:
                captured["training_loop_ran"] = True

        monkeypatch.setattr(mlip_module, "TrainingLoop", FakeTrainingLoop)
        monkeypatch.setattr(
            mlip_module,
            "get_default_mlip_optimizer",
            lambda config: "optimizer",
        )

        train_data = LabelledCandidates(
            candidates=[
                Candidate(
                    data=_diatomic(1),
                    modality="tabular",
                    features={"forces": np.zeros((2, 3))},
                ),
                Candidate(
                    data=_diatomic(8),
                    modality="tabular",
                    features={"forces": np.zeros((2, 3))},
                ),
            ],
            labels=np.array([2.0, 6.0]),
        )
        model.train(train_data, val_data=self._labelled(1))

        initial_force_field = captured["training_loop_kwargs"]["force_field"]
        assert captured["pretrained_dataset_info"] is pretrained_info
        assert captured["batch_size"] == model.train_config.batch_size
        assert len(captured["systems_by_split"]["train"]) == 2
        assert pretrained_info.atomic_energies_map[1] == -1.0
        assert initial_force_field is not pretrained
        assert initial_force_field.params == {"source": True}
        assert initial_force_field.predictor.mlip_network.dataset_info is target_info
        assert captured["training_loop_kwargs"]["train_dataset"] == "train-dataset"
        assert captured["training_loop_kwargs"]["validation_dataset"] == "valid-dataset"
        assert captured["training_loop_kwargs"]["optimizer"] == "optimizer"
        assert captured["training_loop_ran"] is True
        assert model.force_field.params == {"trained": True}
        assert model.force_field.dataset_info.atomic_energies_map[1] == pytest.approx(1.0)
