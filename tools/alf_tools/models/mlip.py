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

"""MLIP model for active learning over atomistic systems.

Implements the ALF BaseModel interface around the mlip-jax MACE force-field,
supporting both training from scratch and finetuning from a pretrained model.
"""

import copy
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import fsspec
import jax
import numpy as np
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions
from mlip.data import ChemicalSystem, DatasetInfo
from mlip.data.dataset_info import compute_dataset_info_from_graphs
from mlip.data.graph_dataset import GraphDataset
from mlip.data.helpers.atomic_energies import compute_average_e0s_from_graphs
from mlip.graph import Graph
from mlip.graph.mask_helpers import get_graph_padding_mask
from mlip.inference.batched_inference import run_batched_inference
from mlip.models import ForceField
from mlip.models.loss import HuberLoss
from mlip.models.mace.network import Mace
from mlip.models.model_io import load_model_from_zip, save_model_to_zip
from mlip.models.params_transfer import transfer_params
from mlip.training import TrainingLoop
from mlip.training.optimizer import get_default_mlip_optimizer
from mlip.training.optimizer_config import OptimizerConfig
from mlip.training.training_io_handler import TrainingIOHandler
from mlip.training.training_loggers import log_metrics_to_line

from alf_tools.models.mlip_utils.config import MLIPModelConfig, MLIPTrainConfig

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_BUCKET = "mlip-jax-2420d80efc6f4f6b-inputs"

_mlipjax_fs = fsspec.filesystem(
    "s3",
    client_kwargs={"endpoint_url": os.getenv("FSSPEC_S3_ENDPOINT_URL")},
)


def _download_model(load_path: Path | str) -> None:
    """Download pretrained model from S3 if not cached locally."""
    load_path = Path(load_path)
    file_path = Path(__file__).parent / "models" / load_path
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
    """
    model_path = Path(__file__).parent / "models" / Path(load_path)
    return load_model_from_zip(model_type, model_path)


def _candidate_to_chemical_system(candidate: Candidate, energy: float) -> ChemicalSystem:
    """Convert a Candidate wrapping ASE Atoms to a ChemicalSystem."""
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
    """Convert LabelledCandidates to a list of ChemicalSystems."""
    return [
        _candidate_to_chemical_system(candidate, energy)
        for candidate, energy in zip(data.candidates, data.labels)
    ]


def _filter_valid_graphs(graphs: list) -> list:
    """Drop graphs that are None or have no edges (single-atom or cutoff too small)."""
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
    """Build and filter graphs from chemical systems, raising if none are valid."""
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
    """Compute max_n_node and max_n_edge for batching a GraphDataset."""
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
    """
    systems = _labeled_candidates_to_systems(data)

    graphs = [Graph.from_chemical_system(s, cutoff) for s in systems]
    valid_graphs = _filter_valid_graphs(graphs)

    if len(valid_graphs) == 0:
        raise ValueError("No valid graphs produced from data for E0 computation")

    squeezed_graphs = []
    for g in valid_graphs:
        if g.globals.energy is not None:
            energy = g.globals.energy
            if hasattr(energy, "shape") and len(energy.shape) > 0:
                g = g.replace_globals(energy=np.squeeze(energy))
        squeezed_graphs.append(g)

    return compute_average_e0s_from_graphs(squeezed_graphs)


class MLIPModel(BaseModel):
    """MLIP MACE force-field wrapper implementing the ALF BaseModel interface.

    Supports finetuning from a pretrained foundation model (default) or training
    from scratch. When finetuning, model weights are reinitialised to pretrained
    values at the start of each train() call.

    Args:
        model_config: Model architecture and path configuration.
        train_config: Training hyperparameter configuration.
        seed: Random seed for reproducibility.
        precomputed_e0s: Optional pre-computed per-element reference energies
            (atomic number -> energy). When provided, these are reused across all
            AL iterations instead of recomputing from each training batch.
    """

    def __init__(
        self,
        model_config: MLIPModelConfig,
        train_config: MLIPTrainConfig,
        seed: int = 42,
        precomputed_e0s: Optional[dict[int, float]] = None,
    ):
        self.model_config = model_config
        self.train_config = train_config
        self.seed = seed
        self._precomputed_e0s = precomputed_e0s

        self.force_field: Optional[ForceField] = None
        self._pretrained_force_field: Optional[ForceField] = None
        self._test_data: Optional[LabelledCandidates] = None

        self._test_graphs: Optional[list] = None
        self._test_batching: Optional[tuple[int, int, int]] = None
        self._test_labels: Optional[np.ndarray] = None
        self._valid_test_systems: Optional[list] = None

        self._last_training_metrics: dict = {}

        self._load_pretrained_model()

    def _load_pretrained_model(self) -> None:
        """Load pretrained model from disk (downloading from S3 if needed)."""
        if self.model_config.model_path is not None:
            _download_model(self.model_config.model_path)
            self._pretrained_force_field = _load_model_from_zip(Mace, self.model_config.model_path)
            self.force_field = None
            logger.info(f"Loaded pretrained model from {self.model_config.model_path}")
            logger.info(
                f"  z_table: "
                f"{sorted(self._pretrained_force_field.dataset_info.atomic_energies_map.keys())}"
            )
        else:
            self._pretrained_force_field = None
            self.force_field = None

    def set_test_data(self, test_data: LabelledCandidates) -> None:
        """Set held-out test data to evaluate after each training run.

        Args:
            test_data: LabelledCandidates to use as the test set.
        """
        self._test_data = test_data

    def _get_dynamic_training_settings(self, train_size: int) -> tuple[int, float, int]:
        """Derive batch_size, learning_rate, and epochs that keep ~1000 gradient updates.

        Args:
            train_size: Number of training samples.

        Returns:
            Tuple of (batch_size, learning_rate, epochs).
        """
        target_gradient_updates = 1000
        min_epochs = 10

        if train_size <= 20:
            batch_size = 1
            lr = 0.001
        elif train_size <= 100:
            batch_size = 2
            lr = 0.005
        else:
            batch_size = 4
            lr = 0.01

        epochs = max(min_epochs, int(np.ceil(target_gradient_updates * batch_size / train_size)))
        return batch_size, lr, epochs

    def featurise(self, inputs: list[Candidate]) -> Any:
        """No-op: data is already ASE Atoms objects used directly by the model."""
        return inputs

    def _squeeze_graph_energies(self, graphs: list) -> list:
        """Remove batch dimension from graph energies if present."""
        if len(graphs) > 0 and graphs[0].globals.energy is not None:
            sample_energy = graphs[0].globals.energy
            if hasattr(sample_energy, "shape") and len(sample_energy.shape) > 0:
                return [g.replace_globals(energy=np.squeeze(g.globals.energy)) for g in graphs]
        return graphs

    def _compute_e0s_from_graphs(self, graphs: list) -> dict[int, float]:
        """Compute average E0s from graphs, handling batch dimension."""
        squeezed = self._squeeze_graph_energies(graphs)
        return compute_average_e0s_from_graphs(squeezed)

    def _create_finetuning_force_field(
        self,
        train_graphs: list,
    ) -> ForceField:
        """Build a force field initialised from pretrained weights with updated E0s.

        Uses precomputed_e0s if provided at init, otherwise computes from train_graphs.

        Args:
            train_graphs: Graphs used to compute E0s when precomputed_e0s is None.

        Returns:
            ForceField with pretrained parameters transferred to new E0 configuration.
        """
        pretrained = self._pretrained_force_field
        if pretrained is None:
            raise ValueError("Pretrained force field is required for finetuning")
        pretrained_e0s = pretrained.dataset_info.atomic_energies_map

        if self._precomputed_e0s is not None:
            finetuning_e0s = self._precomputed_e0s
            e0_source = "precomputed (full dataset)"
        else:
            finetuning_e0s = self._compute_e0s_from_graphs(train_graphs)
            e0_source = "computed from training data"

        merged_e0s = {**pretrained_e0s, **finetuning_e0s}

        logger.info("========== FINETUNING E0 CONFIGURATION ==========")
        logger.info(f"E0 source: {e0_source}")
        logger.info(f"Pretrained E0s ({len(pretrained_e0s)} species): {pretrained_e0s}")
        logger.info(f"Finetuning E0s ({len(finetuning_e0s)} species): {finetuning_e0s}")
        logger.info(f"Merged E0s ({len(merged_e0s)} species): {merged_e0s}")
        logger.info("==================================================")

        updated_dataset_info = DatasetInfo(
            atomic_energies_map=merged_e0s,
            graph_cutoff_angstrom=pretrained.dataset_info.graph_cutoff_angstrom,
            avg_num_neighbors=pretrained.dataset_info.avg_num_neighbors,
            avg_r_min_angstrom=pretrained.dataset_info.avg_r_min_angstrom,
            scaling_mean=0.0,
            scaling_stdev=1.0,
        )
        updated_config = pretrained.config.model_copy(
            update={
                "avg_num_neighbors": pretrained.dataset_info.avg_num_neighbors,
                "avg_r_min": pretrained.dataset_info.avg_r_min_angstrom,
            }
        )

        model = Mace(config=updated_config, dataset_info=updated_dataset_info)
        new_force_field = ForceField.from_mlip_network(model, seed=self.seed)
        transferred_params = transfer_params(
            pretrained.params,
            new_force_field.params,
            scale_factor=1.0,
        )

        logger.info("Created finetuning force field with updated E0s")
        return ForceField(new_force_field.predictor, transferred_params)

    def train(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates,
        metrics_collector: Optional[Any] = None,
        reference_train_data: Optional[LabelledCandidates] = None,
    ) -> None:
        """Train or finetune the model on the provided data.

        Args:
            train_data: Training structures and energies/forces.
            val_data: Validation structures and energies/forces.
            metrics_collector: Optional callable(category, metrics, epoch, head_id) for
                external metric logging.
            reference_train_data: Optional larger dataset used only for computing E0s
                and batching limits (e.g. when train_data is a subset of a full dataset).
                Defaults to train_data.
        """
        is_finetuning = self._pretrained_force_field is not None
        current_train_size = len(train_data)
        reference_train_data = train_data if reference_train_data is None else reference_train_data

        if self.train_config.dynamic_training:
            effective_batch_size, effective_lr, effective_epochs = (
                self._get_dynamic_training_settings(current_train_size)
            )
            logger.info(
                f"Dynamic training — {current_train_size} samples: "
                f"batch={effective_batch_size}, lr={effective_lr}, epochs={effective_epochs}"
            )
        else:
            effective_batch_size = self.train_config.batch_size
            effective_lr = self.train_config.learning_rate
            effective_epochs = self.train_config.epochs

        logger.info(
            f"Starting {'finetuning' if is_finetuning else 'training from scratch'} "
            f"with {current_train_size} train samples, {len(val_data)} val samples"
        )
        logger.info(
            f"  Epochs: {effective_epochs}, Batch size: {effective_batch_size}, LR: {effective_lr}"
        )

        if is_finetuning:
            if self._pretrained_force_field is None:
                raise ValueError("Pretrained force field missing during finetuning")
            cutoff = self._pretrained_force_field.dataset_info.graph_cutoff_angstrom
        else:
            cutoff = self.model_config.graph_cutoff_angstrom

        train_systems = _labeled_candidates_to_systems(train_data)
        reference_train_systems = _labeled_candidates_to_systems(reference_train_data)
        val_systems = _labeled_candidates_to_systems(val_data)

        train_graphs = _build_graphs(train_systems, cutoff, name="train")
        reference_train_graphs = (
            train_graphs
            if reference_train_data is train_data
            else _build_graphs(reference_train_systems, cutoff, name="reference train")
        )
        val_graphs = _build_graphs(val_systems, cutoff, name="val")

        test_graphs = None
        if self._test_data is not None and len(self._test_data) > 0:
            test_systems = _labeled_candidates_to_systems(self._test_data)
            original_test_graphs = [Graph.from_chemical_system(s, cutoff) for s in test_systems]
            test_graphs = []
            valid_test_systems = []
            valid_test_candidates = []
            valid_test_labels = []
            for i, g in enumerate(original_test_graphs):
                if g is not None and hasattr(g, "n_edge") and int(g.n_edge.sum()) > 0:
                    test_graphs.append(g)
                    valid_test_systems.append(test_systems[i])
                    valid_test_candidates.append(self._test_data.candidates[i])
                    valid_test_labels.append(self._test_data.labels[i])

            n_removed = len(test_systems) - len(test_graphs)
            if n_removed > 0:
                logger.warning(
                    f"Filtered test set: {len(test_systems)} -> {len(test_graphs)} "
                    f"({n_removed} empty graphs removed)"
                )
            if len(test_graphs) == 0:
                raise ValueError("Test set produced 0 valid graphs")

            self._test_data = LabelledCandidates(
                candidates=valid_test_candidates,
                labels=np.array(valid_test_labels),
            )
            self._test_labels = np.array([s.energy for s in valid_test_systems])
            self._valid_test_systems = valid_test_systems

        max_n_node, max_n_edge = _compute_batching_limits(
            reference_train_systems, reference_train_graphs, effective_batch_size
        )
        val_max_n_node, val_max_n_edge = _compute_batching_limits(
            val_systems, val_graphs, effective_batch_size
        )

        # Shuffle with model seed so committee members (identical pretrained init)
        # see different batch orderings from the first epoch.
        rng = np.random.RandomState(self.seed)
        shuffle_indices = rng.permutation(len(train_graphs))
        train_graphs_shuffled = [train_graphs[i] for i in shuffle_indices]
        logger.info(f"  Shuffled {len(train_graphs)} training graphs with seed={self.seed}")

        train_set = GraphDataset(
            train_graphs_shuffled,
            effective_batch_size,
            max_n_node,
            max_n_edge,
            should_shuffle=False,
        )
        val_set = GraphDataset(
            val_graphs,
            effective_batch_size,
            val_max_n_node,
            val_max_n_edge,
            should_shuffle=False,
        )

        if is_finetuning:
            self.force_field = self._create_finetuning_force_field(reference_train_graphs)
        else:
            squeezed_graphs = self._squeeze_graph_energies(train_graphs)
            dataset_info = compute_dataset_info_from_graphs(
                squeezed_graphs,
                graph_cutoff_angstrom=cutoff,
            )
            mlip_network = Mace(
                Mace.Config(
                    num_channels=self.model_config.num_channels,
                    correlation=self.model_config.correlation,
                    avg_num_neighbors=dataset_info.avg_num_neighbors,
                    avg_r_min=dataset_info.avg_r_min_angstrom,
                ),
                dataset_info,
            )
            self.force_field = ForceField.from_mlip_network(mlip_network, seed=self.seed)

        def _log_wrapper(
            category: str, to_log: dict, epoch_number: int, head_id: Any = None
        ) -> None:
            log_metrics_to_line(category, to_log, epoch_number)
            if metrics_collector is not None:
                metrics_collector(category, to_log, epoch_number, head_id)

        io_handler = TrainingIOHandler()
        io_handler.attach_logger(_log_wrapper)

        force_field_for_training = copy.deepcopy(self.force_field)
        training_config = TrainingLoop.Config(num_epochs=effective_epochs)

        if self.train_config.use_weight_flip:
            flip_epoch = self.train_config.flip_epoch or int(effective_epochs * 0.7)
            logger.info(f"  Using weight flip at epoch {flip_epoch}")
            loss_fn = HuberLoss(
                energy_weight_schedule=lambda epoch: 40.0 if epoch < flip_epoch else 1000.0,
                forces_weight_schedule=lambda epoch: 1000.0 if epoch < flip_epoch else 40.0,
                extended_metrics=True,
            )
        else:
            energy_w = self.train_config.energy_weight
            forces_w = self.train_config.forces_weight
            logger.info(f"  Huber loss: energy={energy_w}, forces={forces_w}")
            loss_fn = HuberLoss(
                energy_weight_schedule=lambda _: energy_w,
                forces_weight_schedule=lambda _: forces_w,
                extended_metrics=True,
            )

        optimizer_config = OptimizerConfig(
            init_learning_rate=effective_lr,
            peak_learning_rate=effective_lr,
            final_learning_rate=effective_lr,
        )

        training_loop = TrainingLoop(
            train_dataset=train_set,
            validation_dataset=val_set,
            force_field=force_field_for_training,
            loss=loss_fn,
            optimizer=get_default_mlip_optimizer(optimizer_config),
            config=training_config,
            io_handler=io_handler,
        )

        start_time = time.time()
        training_loop.run()
        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.2f}s")

        best_model = training_loop.best_model
        has_valid_params = (
            best_model is not None
            and best_model.params is not None
            and hasattr(training_loop, "_best_params")
            and training_loop._best_params is not None
        )

        if has_valid_params:
            self.force_field = best_model

            if test_graphs is not None and len(test_graphs) > 0:
                cached_test_systems = self._valid_test_systems
                if cached_test_systems is None:
                    raise ValueError("valid_test_systems missing for test evaluation")
                test_max_n_node, test_max_n_edge = _compute_batching_limits(
                    cached_test_systems, test_graphs, effective_batch_size
                )
                test_set = GraphDataset(
                    test_graphs,
                    effective_batch_size,
                    test_max_n_node,
                    test_max_n_edge,
                    should_shuffle=False,
                )
                training_loop.test(test_set)
                self._test_graphs = test_graphs
                self._test_batching = (effective_batch_size, test_max_n_node, test_max_n_edge)
                self._compute_and_log_per_reaction_metrics()
        else:
            logger.warning("Training loop returned invalid model, keeping previous model")

        self._last_training_metrics = {
            "train_size": current_train_size,
            "val_size": len(val_data),
            "epochs": effective_epochs,
            "batch_size": effective_batch_size,
            "learning_rate": effective_lr,
            "training_time": elapsed,
        }

    def _compute_and_log_per_reaction_metrics(self) -> dict:
        """Compute and log per-reaction energy (and forces) metrics on the test set."""
        if self._test_data is None or self._test_labels is None:
            return {}

        has_rxn = any(
            c.features
            and c.features.get("reaction_id") is not None
            and c.features.get("reaction_id") != -1
            for c in self._test_data.candidates
        )
        if not has_rxn:
            return {}

        preds = self.evaluate_test()
        if preds is None:
            return {}

        pred_energies = preds.means
        true_energies = self._test_labels
        reaction_ids = np.array([
            c.features.get("reaction_id", -1) if c.features else -1
            for c in self._test_data.candidates
        ])
        n_atoms = np.array([len(c.data) for c in self._test_data.candidates], dtype=float)

        pred_forces = None
        true_forces = None
        try:
            structures = [c.data for c in self._test_data.candidates]
            true_forces_list = [
                np.asarray(c.features["forces"])
                for c in self._test_data.candidates
                if c.features and c.features.get("forces") is not None
            ]
            if len(true_forces_list) == len(structures):
                true_forces = true_forces_list
                batch_size, max_n_node, max_n_edge = self._safe_batching_limits(structures)
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, "w")
                try:
                    mlip_preds = run_batched_inference(
                        structures=structures,
                        force_field=self.force_field,
                        batch_size=batch_size,
                        max_n_node=max_n_node,
                        max_n_edge=max_n_edge,
                    )
                finally:
                    sys.stdout.close()
                    sys.stdout = old_stdout
                pred_forces = [np.array(p.forces) for p in mlip_preds]
        except Exception:
            pred_forces = None
            true_forces = None

        metrics: dict = {}
        logger.info("  Per-reaction test metrics:")
        for rxn_id in sorted(set(reaction_ids)):
            mask = reaction_ids == rxn_id
            indices = np.where(mask)[0]
            n = mask.sum()

            errors_pa = (pred_energies[mask] - true_energies[mask]) / n_atoms[mask]
            rmse_pa = float(np.sqrt(np.mean(errors_pa**2)))
            mae_pa = float(np.mean(np.abs(errors_pa)))
            metrics[f"rxn{rxn_id:05d}/energy_rmse_per_atom"] = rmse_pa
            metrics[f"rxn{rxn_id:05d}/energy_mae_per_atom"] = mae_pa

            line = (
                f"    rxn{rxn_id:05d} ({n:3d}): E_RMSE/atom={rmse_pa:.6f}, E_MAE/atom={mae_pa:.6f}"
            )

            if pred_forces is not None and true_forces is not None:
                rxn_pf = np.concatenate([pred_forces[i].flatten() for i in indices])
                rxn_tf = np.concatenate([true_forces[i].flatten() for i in indices])
                f_errors = rxn_pf - rxn_tf
                f_rmse = float(np.sqrt(np.mean(f_errors**2)))
                f_mae = float(np.mean(np.abs(f_errors)))
                metrics[f"rxn{rxn_id:05d}/force_rmse"] = f_rmse
                metrics[f"rxn{rxn_id:05d}/force_mae"] = f_mae
                line += f", F_RMSE={f_rmse:.4f}, F_MAE={f_mae:.4f}"

            logger.info(line)

        self._last_training_metrics.update(metrics)
        return metrics

    def _safe_batching_limits(self, structures: list) -> tuple[int, int, int]:
        """Compute batch_size, max_n_node, max_n_edge covering every structure.

        Uses actual maxima rather than medians so no structure is silently dropped.

        Args:
            structures: List of ASE Atoms objects.

        Returns:
            Tuple of (batch_size, max_n_node, max_n_edge).
        """
        force_field = self.force_field
        if force_field is None:
            raise RuntimeError("Model must be trained before prediction")

        cutoff = force_field.cutoff_distance
        graphs = [
            Graph.from_chemical_system(
                ChemicalSystem(
                    atomic_numbers=s.numbers,
                    positions=s.get_positions(),
                ),
                cutoff,
            )
            for s in structures
        ]

        batch_size = self.train_config.batch_size
        max_n_atoms = max(int(g.n_node.sum()) for g in graphs)
        max_n_edges = max(int(g.n_edge.sum()) for g in graphs)
        max_n_node = max(1, int(np.ceil(max_n_atoms / batch_size)))
        max_n_edge = max(1, int(np.ceil(max_n_edges / (2 * batch_size))))
        return batch_size, max_n_node, max_n_edge

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Predict energies for the given candidates.

        Args:
            candidate_points: Candidates whose .data are ASE Atoms objects.

        Returns:
            Predictions with means as predicted energies.
        """
        structures = [c.data for c in candidate_points]
        batch_size, max_n_node, max_n_edge = self._safe_batching_limits(structures)

        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            force_field = self.force_field
            if force_field is None:
                raise RuntimeError("Model must be trained before prediction")
            mlip_predictions = run_batched_inference(
                structures=structures,
                force_field=force_field,
                batch_size=batch_size,
                max_n_node=max_n_node,
                max_n_edge=max_n_edge,
            )
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        return Predictions(means=np.array([p.energy for p in mlip_predictions]))

    def predict_with_forces(
        self, candidate_points: list[Candidate]
    ) -> tuple[Predictions, list[np.ndarray]]:
        """Predict energies and forces for the given candidates.

        Args:
            candidate_points: Candidates whose .data are ASE Atoms objects.

        Returns:
            Tuple of (Predictions with energy means, list of force arrays per structure).
        """
        structures = [c.data for c in candidate_points]
        batch_size, max_n_node, max_n_edge = self._safe_batching_limits(structures)

        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            force_field = self.force_field
            if force_field is None:
                raise RuntimeError("Model must be trained before prediction")
            mlip_predictions = run_batched_inference(
                structures=structures,
                force_field=force_field,
                batch_size=batch_size,
                max_n_node=max_n_node,
                max_n_edge=max_n_edge,
            )
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        energies = np.array([p.energy for p in mlip_predictions])
        forces = [np.array(p.forces) for p in mlip_predictions]
        return Predictions(means=energies), forces

    def evaluate_test(self) -> Optional[Predictions]:
        """Run inference on the cached test graphs.

        Returns:
            Predictions with energy means, or None if no test graphs are cached.
        """
        if self._test_graphs is None or self._test_batching is None:
            return None

        batch_size, max_n_node, max_n_edge = self._test_batching
        test_set = GraphDataset(
            self._test_graphs, batch_size, max_n_node, max_n_edge, should_shuffle=False
        )

        force_field = self.force_field
        if force_field is None:
            raise RuntimeError("Model must be trained before test evaluation")
        jitted_force_field = jax.jit(force_field)

        energies = []
        for batch in test_set:
            output = jitted_force_field(batch)
            mask = get_graph_padding_mask(batch)
            for i in range(output.energy.shape[0]):
                if mask[i]:
                    energies.append(float(output.energy[i]))

        return Predictions(means=np.array(energies))

    def sample(self, condition: Any = None) -> list[Candidate]:
        """Not implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Sampling is not implemented for MLIPModel")

    def get_training_summary_metrics(self) -> dict[str, float]:
        """Return summary metrics from the most recent train() call.

        Returns:
            Dictionary with training metadata and per-reaction test metrics.
        """
        return self._last_training_metrics

    def save(self, save_path: Path | str) -> None:
        """Save the trained force field to a zip file.

        Args:
            save_path: Destination path for the zip file.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model = self.force_field
        if model is None:
            raise RuntimeError("Cannot save an untrained model")

        save_model_to_zip(save_path, model)
        logger.info(f"Model saved to: {save_path}")
