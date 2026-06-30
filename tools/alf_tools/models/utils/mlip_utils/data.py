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

"""MLIP-specific helpers for ALF data conversion and graph construction."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from alf_core import Candidate, LabelledCandidates
from mlip.data import (
    BuilderMode,
    ChemicalSystem,
    GraphDatasetBuilder,
    SingleGraphDatasetBuilder,
)
from mlip.data.chemical_systems_readers.extxyz_reader import ExtxyzReader
from mlip.data.dataset_info import DatasetInfo
from mlip.data.graph_dataset import GraphDataset


_REPLAY_DATASET_KEY = "replay"
_TARGET_DATASET_KEY = "target"


def _reader_from_systems(systems: list[ChemicalSystem]) -> SimpleNamespace:
    """Return the minimal reader interface expected by mlip dataset builders."""
    return SimpleNamespace(load=lambda: list(systems))


def candidate_to_chemical_system(
    candidate: Candidate,
    energy: float | None = None,
) -> ChemicalSystem:
    """Convert an ALF structure candidate into an mlip ChemicalSystem."""
    system_kwargs = dict(candidate.data)
    if energy is not None:
        system_kwargs["energy"] = float(energy)
    if "forces" in candidate.features:
        system_kwargs["forces"] = np.asarray(candidate.features["forces"], dtype=float)
    return ChemicalSystem(**system_kwargs)


def chemical_system_to_candidate(
    system: ChemicalSystem,
    modality: str = "tabular",
) -> Candidate:
    """Convert an mlip ChemicalSystem into an ALF Candidate."""
    data = system.model_dump(exclude_none=True)
    data.pop("energy", None)
    forces = data.pop("forces", None)
    features = {}
    if forces is not None:
        features["forces"] = np.asarray(forces, dtype=float)
    return Candidate(data=data, modality=modality, features=features)


def labelled_candidates_to_chemical_systems(
    data: LabelledCandidates,
) -> list[ChemicalSystem]:
    """Convert ALF labelled candidates into mlip ChemicalSystems."""
    return [
        candidate_to_chemical_system(candidate, energy)
        for candidate, energy in zip(data.candidates, data.labels)
    ]


def chemical_systems_to_labelled_candidates(
    systems: list[ChemicalSystem],
    modality: str = "tabular",
) -> LabelledCandidates:
    """Convert labelled mlip ChemicalSystems into ALF LabelledCandidates."""
    missing_energy = [i for i, system in enumerate(systems) if system.energy is None]
    if missing_energy:
        raise ValueError(
            "Cannot build LabelledCandidates from ChemicalSystems without energies: "
            f"indices {missing_energy}"
        )
    return LabelledCandidates(
        candidates=[
            chemical_system_to_candidate(system, modality=modality) for system in systems
        ],
        labels=np.asarray([system.energy for system in systems], dtype=float),
    )


def load_extxyz_as_labelled_candidates(
    xyz_path: Path | str,
    modality: str = "tabular",
    property_name_mapping: dict[str, str] | None = None,
) -> LabelledCandidates:
    """Load an extxyz file into ALF LabelledCandidates via mlip's ExtxyzReader."""
    systems = ExtxyzReader(
        filepaths=xyz_path,
        property_name_mapping=property_name_mapping,
    ).load()
    return chemical_systems_to_labelled_candidates(systems, modality=modality)


def build_graph_datasets(
    systems_by_split: dict[str, list[ChemicalSystem]],
    cutoff: float,
    batch_size: int,
) -> tuple[dict[str, GraphDataset], DatasetInfo]:
    """Build graph datasets and dataset info with mlip's dataset builder."""
    readers = {
        split: _reader_from_systems(systems)
        for split, systems in systems_by_split.items()
        if len(systems) > 0
    }
    builder = GraphDatasetBuilder(
        readers=readers,
        builder_config=GraphDatasetBuilder.Config(
            graph_cutoff_angstrom=cutoff,
            batch_size=batch_size,
            set_none_charges_to_zero=True,
            homogenize=True,
        ),
        mode=BuilderMode.TRAINING,
    )
    datasets = builder.get_datasets(prefetch=False)
    built_dataset_info = builder.dataset_info
    if isinstance(built_dataset_info, dict) or built_dataset_info is None:
        raise ValueError("MLIP TRAINING dataset builder did not return a DatasetInfo")
    return datasets, built_dataset_info


def build_finetuning_graph_datasets(
    systems_by_split: dict[str, list[ChemicalSystem]],
    pretrained_dataset_info: DatasetInfo,
    batch_size: int,
) -> tuple[dict[str, GraphDataset], DatasetInfo]:
    """Build finetuning datasets and scalar target DatasetInfo via mlip MULTI mode."""
    pretrained_e0s = pretrained_dataset_info.atomic_energies_map
    if not isinstance(pretrained_e0s, dict):
        raise ValueError(
            "Naive MLIP finetuning only supports single-head pretrained checkpoints "
            "with a scalar atomic_energies_map. Multi-dataset or multi-head "
            "pretrained checkpoints are not supported."
        )

    target_readers = {
        split: _reader_from_systems(systems)
        for split, systems in systems_by_split.items()
        if len(systems) > 0
    }
    builder = GraphDatasetBuilder(
        readers={
            _REPLAY_DATASET_KEY: {},
            _TARGET_DATASET_KEY: target_readers,
        },
        builder_config=GraphDatasetBuilder.Config(
            graph_cutoff_angstrom=pretrained_dataset_info.graph_cutoff_angstrom,
            batch_size=batch_size,
            set_none_charges_to_zero=True,
            homogenize=True,
        ),
        mode=BuilderMode.MULTI,
        dataset_info=pretrained_dataset_info,
    )
    datasets = builder.get_datasets(prefetch=False)
    multi_dataset_info = builder.dataset_info
    if not isinstance(multi_dataset_info, DatasetInfo):
        raise ValueError("MLIP finetuning dataset builder did not return a DatasetInfo")
    dataset_names = multi_dataset_info.dataset_name
    if not isinstance(dataset_names, list):
        raise ValueError("MLIP finetuning DatasetInfo is missing dataset names")
    atomic_energies_maps = multi_dataset_info.atomic_energies_map
    if not isinstance(atomic_energies_maps, list):
        raise ValueError("MLIP finetuning DatasetInfo is missing per-dataset E0 maps")

    try:
        target_idx = dataset_names.index(_TARGET_DATASET_KEY)
    except ValueError as exc:
        raise ValueError("MLIP finetuning DatasetInfo is missing target E0s") from exc

    target_merged_e0s = atomic_energies_maps[target_idx]
    missing_atomic_numbers = set(target_merged_e0s) - set(pretrained_e0s)
    if missing_atomic_numbers:
        raise ValueError(
            "Cannot finetune a pretrained MLIP model on elements absent from "
            f"its z-table: {sorted(missing_atomic_numbers)}"
        )

    return datasets, pretrained_dataset_info.model_copy(
        update={"atomic_energies_map": target_merged_e0s}
    )


def build_graph_dataset(
    systems: list[ChemicalSystem],
    cutoff: float,
    batch_size: int,
    *,
    dataset_info: DatasetInfo | bool = False,
    long_range_cutoff: float | None = None,
) -> GraphDataset:
    """Build a GraphDataset from ChemicalSystems using mlip's dataset builder."""
    if any(len(system.atomic_numbers) <= 1 for system in systems):
        raise ValueError("Single atom systems are not supported yet.")

    builder = SingleGraphDatasetBuilder(
        _reader_from_systems(systems),
        GraphDatasetBuilder.Config(
            graph_cutoff_angstrom=cutoff,
            long_range_cutoff_angstrom=long_range_cutoff,
            batch_size=batch_size,
            set_none_charges_to_zero=True,
        ),
        dataset_info=dataset_info,
        shuffle=False,
    )
    return builder.get_dataset(prefetch=False)
