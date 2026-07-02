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


def _reader_from_systems(systems: list[ChemicalSystem]) -> SimpleNamespace:
    """Return the minimal reader interface expected by mlip dataset builders."""
    return SimpleNamespace(load=lambda: list(systems))


def _atomic_numbers_in_systems(systems: list[ChemicalSystem]) -> set[int]:
    """Return all atomic numbers present in a list of systems."""
    atomic_numbers: set[int] = set()
    for system in systems:
        atomic_numbers.update(np.asarray(system.atomic_numbers, dtype=int).tolist())
    return atomic_numbers


def _total_charges_in_systems(systems: list[ChemicalSystem]) -> set[int]:
    """Return total charges present in systems, treating missing charges as neutral."""
    return {0 if system.charge is None else int(system.charge) for system in systems}


def _validate_non_train_species_seen_in_train(
    systems_by_split: dict[str, list[ChemicalSystem]],
) -> None:
    """Raise if validation/test splits contain elements absent from target train.

    Raises:
        ValueError: If a non-training split contains atomic numbers absent from
            the target training split.
    """
    train_atomic_numbers = _atomic_numbers_in_systems(systems_by_split.get("train", []))
    for split, systems in systems_by_split.items():
        if split == "train" or not systems:
            continue
        unseen = _atomic_numbers_in_systems(systems) - train_atomic_numbers
        if unseen:
            raise ValueError(
                f"Split {split!r} contains unseen atomic numbers {sorted(unseen)} "
                "not present in the target train split."
            )


def _validate_total_charges_supported_by_pretrained(
    systems_by_split: dict[str, list[ChemicalSystem]],
    pretrained_dataset_info: DatasetInfo,
) -> None:
    """Raise if target systems contain total charges absent from the pretrained table.

    Raises:
        ValueError: If any target split contains total charges absent from the
            pretrained total-charge table.
    """
    pretrained_charges = set(pretrained_dataset_info.available_total_charges)
    for split, systems in systems_by_split.items():
        if not systems:
            continue
        unseen = _total_charges_in_systems(systems) - pretrained_charges
        if unseen:
            raise ValueError(
                f"Split {split!r} contains unseen total charges {sorted(unseen)} "
                "not present in the pretrained total-charge table."
            )


def candidate_to_chemical_system(
    candidate: Candidate,
    energy: float | None = None,
) -> ChemicalSystem:
    """Convert an ALF structure candidate into an mlip ChemicalSystem.

    Returns:
        The corresponding mlip ChemicalSystem.
    """
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
    """Convert an mlip ChemicalSystem into an ALF Candidate.

    Returns:
        The corresponding ALF candidate.
    """
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
    """Convert ALF labelled candidates into mlip ChemicalSystems.

    Returns:
        ChemicalSystems with labels copied into their energy fields.
    """
    return [
        candidate_to_chemical_system(candidate, energy)
        for candidate, energy in zip(data.candidates, data.labels)
    ]


def chemical_systems_to_labelled_candidates(
    systems: list[ChemicalSystem],
    modality: str = "tabular",
) -> LabelledCandidates:
    """Convert labelled mlip ChemicalSystems into ALF LabelledCandidates.

    Returns:
        ALF labelled candidates built from the ChemicalSystem energies.

    Raises:
        ValueError: If any ChemicalSystem is missing an energy value.
    """
    missing_energy = [i for i, system in enumerate(systems) if system.energy is None]
    if missing_energy:
        raise ValueError(
            "Cannot build LabelledCandidates from ChemicalSystems without energies: "
            f"indices {missing_energy}"
        )
    return LabelledCandidates(
        candidates=[chemical_system_to_candidate(system, modality=modality) for system in systems],
        labels=np.asarray([system.energy for system in systems], dtype=float),
    )


def load_extxyz_as_labelled_candidates(
    xyz_path: Path | str,
    modality: str = "tabular",
    property_name_mapping: dict[str, str] | None = None,
) -> LabelledCandidates:
    """Load an extxyz file into ALF LabelledCandidates via mlip's ExtxyzReader.

    Returns:
        Labelled candidates parsed from the extxyz file.
    """
    systems = ExtxyzReader(
        filepaths=xyz_path,
        property_name_mapping=property_name_mapping,
    ).load()
    return chemical_systems_to_labelled_candidates(systems, modality=modality)


def build_graph_datasets(
    systems_by_split: dict[str, list[ChemicalSystem]],
    cutoff: float,
    batch_size: int,
    *,
    pretrained_dataset_info: DatasetInfo | None = None,
    validate_total_charges: bool = False,
) -> tuple[dict[str, GraphDataset], DatasetInfo]:
    """Build graph datasets and dataset info with mlip's dataset builder.

    Passing `pretrained_dataset_info` enables the single-head finetuning path:
    target-domain E0s are computed from the target train split, and
    the returned DatasetInfo keeps pretrained species
    and charge metadata while replacing E0s for target train species.

    Returns:
        A mapping of split names to graph datasets and the inferred DatasetInfo.

    Raises:
        ValueError: If mlip does not return a scalar DatasetInfo, if a pretrained
            DatasetInfo is unsupported, or if target data is incompatible with it.
    """
    if pretrained_dataset_info is not None:
        pretrained_e0s = pretrained_dataset_info.atomic_energies_map
        if not isinstance(pretrained_e0s, dict):
            raise ValueError(
                "Naive MLIP finetuning only supports single-head pretrained checkpoints "
                "with a scalar atomic_energies_map. Multi-dataset or multi-head "
                "pretrained checkpoints are not supported."
            )
        _validate_non_train_species_seen_in_train(systems_by_split)
        if validate_total_charges:
            _validate_total_charges_supported_by_pretrained(
                systems_by_split,
                pretrained_dataset_info,
            )

    readers = {
        split: _reader_from_systems(systems)
        for split, systems in systems_by_split.items()
        if len(systems) > 0
    }
    graph_cutoff = (
        pretrained_dataset_info.graph_cutoff_angstrom
        if pretrained_dataset_info is not None
        else cutoff
    )
    long_range_cutoff = (
        pretrained_dataset_info.long_range_cutoff_angstrom
        if pretrained_dataset_info is not None
        else None
    )
    builder = GraphDatasetBuilder(
        readers=readers,
        builder_config=GraphDatasetBuilder.Config(
            graph_cutoff_angstrom=graph_cutoff,
            long_range_cutoff_angstrom=long_range_cutoff,
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

    if pretrained_dataset_info is None:
        return datasets, built_dataset_info

    target_e0s = built_dataset_info.atomic_energies_map
    if not isinstance(target_e0s, dict):
        raise ValueError("MLIP finetuning dataset builder did not return scalar target E0s")
    missing_atomic_numbers = set(target_e0s) - set(pretrained_e0s)
    if missing_atomic_numbers:
        raise ValueError(
            "Cannot finetune a pretrained MLIP model on elements absent from "
            f"its z-table: {sorted(missing_atomic_numbers)}"
        )

    return datasets, pretrained_dataset_info.model_copy(
        update={"atomic_energies_map": pretrained_e0s | target_e0s}
    )


def build_finetuning_graph_datasets(
    systems_by_split: dict[str, list[ChemicalSystem]],
    pretrained_dataset_info: DatasetInfo,
    batch_size: int,
    *,
    validate_total_charges: bool = False,
) -> tuple[dict[str, GraphDataset], DatasetInfo]:
    """Build finetuning datasets and scalar target DatasetInfo.

    Returns:
        A mapping of split names to graph datasets and the retargeted DatasetInfo.

    Raises:
        ValueError: If the pretrained DatasetInfo is unsupported or target species
            are absent from the pretrained z-table.
    """
    return build_graph_datasets(
        systems_by_split,
        cutoff=pretrained_dataset_info.graph_cutoff_angstrom,
        batch_size=batch_size,
        pretrained_dataset_info=pretrained_dataset_info,
        validate_total_charges=validate_total_charges,
    )


def build_graph_dataset(
    systems: list[ChemicalSystem],
    cutoff: float,
    batch_size: int,
    *,
    dataset_info: DatasetInfo | bool = False,
    long_range_cutoff: float | None = None,
) -> GraphDataset:
    """Build a GraphDataset from ChemicalSystems using mlip's dataset builder.

    Returns:
        A graph dataset ready for mlip inference.

    Raises:
        ValueError: If any input system has a single atom.
    """
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
