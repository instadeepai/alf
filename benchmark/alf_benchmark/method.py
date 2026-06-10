# Copyright 2023 InstaDeep Ltd. All rights reserved.
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

"""Benchmark methods: *how* a problem is optimised, frozen and versioned."""

from alf_core.dataset.base_dataset import BaseDataset
from alf_core.optimizer.optimizer import Optimizer
from alf_core.oracle.oracle import Oracle
from alf_core.surrogate.surrogate import Surrogate
from alf_core.tasks.base_task import BaseTask
from alf_core.tasks.design_task import DesignTask

from alf_benchmark.config import MethodConfig
from alf_benchmark.registry import Registry, default_registry


class BenchmarkMethod:
    """A frozen method: resolves surrogate, optimizer, oracle, and task by name."""

    def __init__(self, config: MethodConfig, registry: Registry | None = None) -> None:
        """Initialise a benchmark method.

        Args:
            config: The method configuration.
            registry: Registry used to build components (defaults to shared).
        """
        self.config = config
        self.name = config.name
        self.family = config.family
        self.version = config.version
        self._registry = registry if registry is not None else default_registry()

    def build_surrogate(self) -> Surrogate:
        """Build the surrogate model wrapped for the active-learning loop.

        Returns:
            A :class:`Surrogate` wrapping the configured model.
        """
        model = self._registry.build(
            "alf.models", self.config.surrogate.name, self.config.surrogate.config
        )
        return Surrogate(model=model)

    def build_optimizer(self) -> Optimizer:
        """Build the optimizer bundling the acquisition and search functions.

        Returns:
            The configured :class:`Optimizer`.

        Raises:
            ValueError: If the acquisition or search component is missing.
        """
        if self.config.acquisition is None or self.config.search is None:
            raise ValueError("design methods require both 'acquisition' and 'search' components.")
        acquisition_fn = self._registry.build(
            "alf.acquisition_functions",
            self.config.acquisition.name,
            self.config.acquisition.config,
        )
        search_fn = self._registry.build(
            "alf.searches", self.config.search.name, self.config.search.config
        )
        return Optimizer(acquisition_fn=acquisition_fn, search_fn=search_fn)

    def build_oracle(self, dataset: BaseDataset) -> Oracle:
        """Build the oracle that supplies ground-truth labels.

        Offline binds the oracle to the problem's dataset instance; online
        builds a model scorer from the oracle spec.

        Args:
            dataset: The problem's dataset (used as scorer in offline mode).

        Returns:
            The configured :class:`Oracle`.
        """
        oracle_spec = self.config.oracle
        if oracle_spec.mode == "offline":
            return Oracle(scorer=dataset)
        scorer_spec = oracle_spec.scorer
        assert scorer_spec is not None  # guaranteed by OracleSpec validation
        scorer = self._registry.build("alf.models", scorer_spec.name, scorer_spec.config)
        return Oracle(scorer=scorer)

    def build_task(self) -> BaseTask:
        """Build the ALF task for this method's family.

        Returns:
            The configured task.

        Raises:
            NotImplementedError: If the family is not yet supported (Phase 1
                implements the ``design`` family only).
        """
        task_config = self.config.task
        if self.family == "design":
            return DesignTask(
                num_acq_rounds=task_config.num_acq_rounds,
                acq_batch_size=task_config.acq_batch_size,
                save_round_predictions=task_config.save_round_predictions,
            )
        raise NotImplementedError(
            f"Family '{self.family}' is not implemented yet; Phase 1 supports 'design'."
        )
