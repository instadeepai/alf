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

Implements the ALF BaseModel interface around mlip force fields, supporting
both training from scratch and finetuning from pretrained models.
"""

import contextlib
import copy
import io
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import numpy as np
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions
from mlip.data import DatasetInfo
from mlip.inference.batched_inference import run_batched_inference
from mlip.models import ForceField
from mlip.models.model_io import save_model_to_zip
from mlip.training import TrainingLoop
from mlip.training.optimizer import get_default_mlip_optimizer
from mlip.training.training_io_handler import TrainingIOHandler
from mlip.training.training_loggers import log_metrics_to_line

from alf_tools.models.utils.mlip_utils import (
    MLIPModelConfig,
    MLIPTrainConfig,
    build_finetuning_graph_datasets,
    build_graph_dataset,
    build_graph_datasets,
    candidate_to_chemical_system,
    labelled_candidates_to_chemical_systems,
    load_mlip_force_field,
    resolve_mlip_model_cls,
)

logger = logging.getLogger("alf-tools")


class MLIPModel(BaseModel):
    """MLIP force-field wrapper implementing the ALF BaseModel interface.

    Supports training from scratch or finetuning from a pretrained foundation model.
    When finetuning, model weights are reinitialised to pretrained
    values at the start of each train() call.

    Args:
        model_config: Model architecture and path configuration.
        train_config: Training hyperparameter configuration.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        model_config: MLIPModelConfig,
        train_config: MLIPTrainConfig,
        seed: int = 42,
    ):
        """Initialise the model, loading the pretrained force field if configured."""
        self.model_config = model_config
        self.train_config = train_config
        self.seed = seed

        self.force_field: ForceField | None = None
        self._pretrained_force_field: ForceField | None = None
        self._test_data: LabelledCandidates | None = None

        self._last_training_metrics: dict[str, float | int | np.number] = {}
        self._mlip_model_cls = resolve_mlip_model_cls(model_config.model_type)

        self._load_pretrained_model()

    def _load_pretrained_model(self) -> None:
        """Load pretrained model from a zip file when configured."""
        if self.model_config.model_path is not None:
            self._pretrained_force_field = load_mlip_force_field(
                self.model_config.model_type, self.model_config.model_path
            )
            self.force_field = None
            logger.info(f"Loaded pretrained model from {self.model_config.model_path}")
            atomic_energies_map = self._pretrained_force_field.dataset_info.atomic_energies_map
            if isinstance(atomic_energies_map, dict):
                z_table = sorted(atomic_energies_map)
            else:
                z_table = sorted({
                    atomic_number for e0s in atomic_energies_map for atomic_number in e0s
                })
            logger.info(f"  z_table: {z_table}")
        else:
            self._pretrained_force_field = None
            self.force_field = None

    def set_test_data(self, test_data: LabelledCandidates) -> None:
        """Set held-out test data to evaluate after each training run.

        Args:
            test_data: LabelledCandidates to use as the test set.
        """
        self._test_data = test_data

    def featurise(self, inputs: list[Candidate]) -> Any:
        """No-op: structure data is converted directly at the model boundary.

        Returns:
            The inputs unchanged.
        """
        return inputs

    def _initialise_finetuning_force_field(
        self,
        dataset_info: DatasetInfo,
    ) -> ForceField:
        """Return a pretrained force field whose static dataset info is retargeted.

        Raises:
            ValueError: If no pretrained force field is configured.
        """
        pretrained = self._pretrained_force_field
        if pretrained is None:
            raise ValueError("Pretrained force field is required for finetuning")

        pretrained_network = pretrained.predictor.mlip_network
        mlip_network = type(pretrained_network)(
            config=pretrained_network.config,
            dataset_info=dataset_info,
        )
        predictor = replace(pretrained.predictor, mlip_network=mlip_network)
        logger.info("Initialised finetuning force field from retargeted pretrained model")
        return ForceField(
            predictor=predictor,
            params=copy.deepcopy(pretrained.params),
            inference_context=pretrained.inference_context,
        )

    def train(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates | None = None,
        metrics_collector: Any | None = None,
    ) -> None:
        """Train or finetune the model on the provided data.

        Args:
            train_data: Training structures with energy labels and optional force
                labels in ``Candidate.features["forces"]``.
            val_data: Validation structures with the same label convention. Required
                by this model — it is declared optional only to conform to `BaseModel.train`,
                and a `ValueError` is raised when it is `None`.
            metrics_collector: Optional callable(category, metrics, epoch, head_id) for
                external metric logging.

        Raises:
            ValueError: If `val_data` is None, if a pretrained force field is missing
                during finetuning, or if any data split produces no valid graphs.
        """
        if val_data is None:
            raise ValueError("MLIPModel requires val_data; received None.")

        is_finetuning = self._pretrained_force_field is not None
        current_train_size = len(train_data)
        effective_batch_size = self.train_config.batch_size
        optimizer_config = self.train_config.optimizer_config
        training_config = self.train_config.training_loop_config

        logger.info(
            f"Starting {'finetuning' if is_finetuning else 'training from scratch'} "
            f"with {current_train_size} train samples, {len(val_data)} val samples"
        )
        logger.info(
            f"  Epochs: {training_config.num_epochs}, Batch size: {effective_batch_size}, "
            f"Peak LR: {optimizer_config.peak_learning_rate}"
        )

        pretrained = self._pretrained_force_field
        pretrained_dataset_info = None
        if pretrained is not None:
            pretrained_dataset_info = pretrained.dataset_info
            cutoff = pretrained_dataset_info.graph_cutoff_angstrom
        else:
            cutoff = self.model_config.graph_cutoff_angstrom

        train_systems = labelled_candidates_to_chemical_systems(train_data)
        val_systems = labelled_candidates_to_chemical_systems(val_data)
        systems_by_split = {"train": train_systems, "valid": val_systems}

        if self._test_data is not None and len(self._test_data) > 0:
            systems_by_split["test"] = labelled_candidates_to_chemical_systems(self._test_data)

        if pretrained_dataset_info is None:
            datasets, dataset_info = build_graph_datasets(
                systems_by_split,
                cutoff,
                effective_batch_size,
            )
        else:
            datasets, dataset_info = build_finetuning_graph_datasets(
                systems_by_split,
                pretrained_dataset_info,
                effective_batch_size,
            )
        train_set = datasets["train"]
        val_set = datasets["valid"]
        test_set = datasets.get("test")

        if is_finetuning:
            self.force_field = self._initialise_finetuning_force_field(dataset_info)
        else:
            network_config = self.model_config.network_config or self._mlip_model_cls.Config()
            mlip_network = self._mlip_model_cls(network_config, dataset_info)
            self.force_field = ForceField.from_mlip_network(mlip_network, seed=self.seed)

        def _log_wrapper(
            category: str, to_log: dict, epoch_number: int, head_id: Any = None
        ) -> None:
            log_metrics_to_line(category, to_log, epoch_number)
            if metrics_collector is not None:
                metrics_collector(category, to_log, epoch_number, head_id)

        io_handler = TrainingIOHandler()
        io_handler.attach_logger(_log_wrapper)

        training_loop = TrainingLoop(
            train_dataset=train_set,
            validation_dataset=val_set,
            force_field=self.force_field,
            loss=self.train_config.loss,
            optimizer=get_default_mlip_optimizer(optimizer_config),
            config=training_config,
            io_handler=io_handler,
        )

        start_time = time.time()
        training_loop.run()
        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.2f}s")

        best_model = training_loop.best_model
        has_valid_params = best_model is not None and best_model.params is not None

        if has_valid_params:
            # jax.device_get in best_model produces numpy arrays; convert back to JAX
            # arrays so that run_batched_inference can JIT-compile correctly.
            jax_params = jax.device_put(best_model.params)
            inference_context = getattr(self.force_field, "inference_context", None)
            self.force_field = ForceField(
                best_model.predictor,
                jax_params,
                inference_context=inference_context,
            )

            if test_set is not None:
                training_loop.test(test_set)
        else:
            logger.warning(
                "Training loop returned invalid model, keeping initial model for this run"
            )

        self._last_training_metrics = {
            "train_size": current_train_size,
            "val_size": len(val_data),
            "epochs": training_config.num_epochs,
            "batch_size": effective_batch_size,
            "init_learning_rate": optimizer_config.init_learning_rate,
            "peak_learning_rate": optimizer_config.peak_learning_rate,
            "final_learning_rate": optimizer_config.final_learning_rate,
            "training_time": elapsed,
        }

    def _run_inference(self, candidate_points: list[Candidate]) -> list:
        """Run batched inference over structure candidates, silencing mlip's stdout.

        Args:
            candidate_points: List of dict-backed structure candidates.

        Returns:
            The list of per-structure mlip prediction objects (energy and forces).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        force_field = self.force_field
        if force_field is None:
            raise RuntimeError("Model must be trained before prediction")

        systems = [candidate_to_chemical_system(candidate) for candidate in candidate_points]
        batch_size = self.train_config.inference_batch_size or self.train_config.batch_size
        graph_dataset = build_graph_dataset(
            systems,
            cutoff=force_field.cutoff_distance,
            batch_size=batch_size,
            dataset_info=force_field.dataset_info,
            long_range_cutoff=force_field.long_range_cutoff_distance,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            return run_batched_inference(
                structures=graph_dataset,
                force_field=force_field,
            )

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Predict energies for the given candidates.

        Args:
            candidate_points: Candidates whose .data are ChemicalSystem kwargs.

        Returns:
            Predictions with ``means`` as predicted energies.

        Raises:
            RuntimeError: If the model has not been trained yet.
            ValueError: If no candidates are provided, or if any candidate is a
                single-atom system (unsupported by mlip's batched inference).
        """
        if not candidate_points:
            raise ValueError("MLIPModel.predict requires at least one candidate")

        mlip_predictions = self._run_inference(candidate_points)
        return Predictions(means=np.array([p.energy for p in mlip_predictions]))

    def sample(self, condition: Any = None) -> list[Candidate]:
        """Not implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Sampling is not implemented for MLIPModel")

    def get_training_summary_metrics(self) -> dict[str, float | int | np.number]:
        """Return summary metrics from the most recent train() call.

        Returns:
            Dictionary with training metadata. Values are a mix of ints
            (sizes, epochs) and floats (learning rate, training time).
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
