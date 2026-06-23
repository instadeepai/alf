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

"""Utility functions for the MLIP model.

Data conversion, graph building, batching, E0 computation, and pretrained model
loading helpers used by `alf_tools.models.mlip.MLIPModel`.
"""

import logging
import os
from pathlib import Path

import fsspec
import numpy as np
from alf_core import Candidate, LabelledCandidates
from mlip.data import ChemicalSystem
from mlip.data.helpers.atomic_energies import compute_average_e0s_from_graphs
from mlip.graph import Graph
from mlip.models import ForceField
from mlip.models.model_io import load_model_from_zip

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_BUCKET = "mlip-jax-2420d80efc6f4f6b-inputs"

# Models live in the package's models/ directory (sibling of this utils/ package).
_MODELS_DIR = Path(__file__).parent.parent / "models"

_mlipjax_fs = fsspec.filesystem(
    "s3",
    client_kwargs={"endpoint_url": os.getenv("FSSPEC_S3_ENDPOINT_URL")},
)


def _download_model(load_path: Path | str) -> None:
    """Download pretrained model from S3 if not cached locally."""
    load_path = Path(load_path)
    file_path = _MODELS_DIR / load_path
    if file_path.exists():
        logger.info(f"Model already exists locally: {load_path}")
        return

    parts = load_path.parts
    if len(parts) > 1 and parts[0].startswith("mlip-jax-"):
        bucket_name = Path(parts[0])
        key_path = Path(*parts[1:])
    else:
        bucket_name = Path(_DEFAULT_MODEL_BUCKET)
        key_path = load_path

    file_path.parent.mkdir(parents=True, exist_ok=True)
    bucket_path = bucket_name / key_path
    logger.info(f"Downloading model from S3: {bucket_path}")
    with _mlipjax_fs.open(bucket_path, "rb") as f:
        with open(file_path, "wb") as f_out:
            f_out.write(f.read())


def _load_model_from_zip(model_type: type, load_path: str) -> ForceField:
    """Load a MACE force field from a zip file using mlip's model IO.

    Delegates to `mlip.models.model_io.load_model_from_zip`, which handles both
    legacy (v1) and current model archives.

    Returns:
        The loaded ForceField.
    """
    model_path = _MODELS_DIR / Path(load_path)
    return load_model_from_zip(model_type, model_path)


def _candidate_to_chemical_system(candidate: Candidate, energy: float) -> ChemicalSystem:
    """Convert a Candidate wrapping ASE Atoms to a ChemicalSystem.

    Returns:
        The ChemicalSystem built from the candidate's atoms and energy.
    """
    atoms = candidate.data
    forces = candidate.features.get("forces") if candidate.features else None
    return ChemicalSystem(
        atomic_numbers=np.asarray(atoms.numbers),
        positions=np.asarray(atoms.get_positions()),
        energy=float(energy),
        forces=None if forces is None else np.asarray(forces),
        stress=None,
        cell=np.asarray(atoms.get_cell()),
        pbc=tuple(atoms.get_pbc()),
        weight=1.0,
        partial_charges=None,
        charge=None,
        dipole_moment=None,
    )


def _labeled_candidates_to_systems(data: LabelledCandidates) -> list[ChemicalSystem]:
    """Convert LabelledCandidates to a list of ChemicalSystems.

    Returns:
        A list of ChemicalSystems, one per candidate.
    """
    return [
        _candidate_to_chemical_system(candidate, energy)
        for candidate, energy in zip(data.candidates, data.labels)
    ]


def _filter_valid_graphs(graphs: list) -> list:
    """Drop graphs that are None or have no edges (single-atom or cutoff too small).

    Returns:
        The graphs that are non-None and have at least one edge.
    """
    return [g for g in graphs if g is not None and int(g.n_edge.sum()) > 0]


def _median_neighbors_and_max_total_edges(graphs: list) -> tuple[int, int]:
    """Return (median neighbours per node, max total edges) across graphs.

    Mirrors mlip's former GraphDatasetBuilder helper used for batch sizing.
    """
    num_neighbors = []
    max_total_edges = 0
    for graph in graphs:
        counts = np.bincount(np.asarray(graph.receivers))
        max_total_edges = max(max_total_edges, int(counts.sum()))
        num_neighbors.append(counts)
    median = int(np.ceil(np.median(np.concatenate(num_neighbors)).item()))
    return median, max_total_edges


def _build_graphs(
    systems: list[ChemicalSystem],
    cutoff: float,
    name: str,
) -> list:
    """Build and filter graphs from chemical systems, raising if none are valid.

    Returns:
        The list of valid graphs.

    Raises:
        ValueError: If no valid graphs are produced.
    """
    graphs = [Graph.from_chemical_system(system, cutoff) for system in systems]
    filtered = _filter_valid_graphs(graphs)
    if len(filtered) == 0:
        raise ValueError(
            f"{name} set produced 0 valid graphs "
            "(cutoff too small, unseen species, or single-atom systems)."
        )
    return filtered


def _compute_batching_limits(
    systems: list[ChemicalSystem],
    graphs: list,
    batch_size: int,
) -> tuple[int, int]:
    """Compute max_n_node and max_n_edge for batching a GraphDataset.

    Returns:
        Tuple of (max_n_node, max_n_edge).
    """
    n_atoms = [len(s.atomic_numbers) for s in systems]
    median_n_atoms = int(np.median(n_atoms))
    max_n_atoms = int(np.max(n_atoms))
    max_n_node = median_n_atoms
    if batch_size * max_n_node < max_n_atoms:
        max_n_node = int(np.ceil(max_n_atoms / batch_size))

    median_n_nei, max_total_edges = _median_neighbors_and_max_total_edges(graphs)
    max_n_edge = median_n_nei * max_n_node // 2
    if max_n_edge * batch_size * 2 < max_total_edges:
        max_n_edge = int(np.ceil(max_total_edges / (2 * batch_size)))

    return max_n_node, max_n_edge


def compute_e0s_from_labeled_candidates(
    data: LabelledCandidates,
    cutoff: float,
) -> dict[int, float]:
    """Compute E0s (average atomic energies) from LabelledCandidates.

    Uses least-squares regression to estimate per-atom energy contributions.

    Args:
        data: LabelledCandidates containing structures and energies.
        cutoff: Graph cutoff distance in Angstrom.

    Returns:
        Dictionary mapping atomic number to average energy contribution.

    Raises:
        ValueError: If no valid graphs are produced from the data.
    """
    systems = _labeled_candidates_to_systems(data)

    graphs = [Graph.from_chemical_system(s, cutoff) for s in systems]
    valid_graphs = _filter_valid_graphs(graphs)

    if len(valid_graphs) == 0:
        raise ValueError("No valid graphs produced from data for E0 computation")

    squeezed_graphs = []
    for graph in valid_graphs:
        squeezed = graph
        if graph.globals.energy is not None:
            energy = graph.globals.energy
            if hasattr(energy, "shape") and len(energy.shape) > 0:
                squeezed = graph.replace_globals(energy=np.squeeze(energy))
        squeezed_graphs.append(squeezed)

    return compute_average_e0s_from_graphs(squeezed_graphs)
