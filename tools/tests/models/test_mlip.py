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
