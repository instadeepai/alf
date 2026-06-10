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

"""Benchmark problems and suites: *what* is optimised, frozen and versioned."""

import dataclasses

from alf_core.dataset.base_dataset import BaseDataset

from alf_benchmark.config import ProblemConfig
from alf_benchmark.registry import Registry, default_registry


class BenchmarkProblem:
    """A frozen problem: a configured dataset, replication seeds, and a metric."""

    def __init__(self, config: ProblemConfig, registry: Registry | None = None) -> None:
        """Initialise a benchmark problem.

        Args:
            config: The problem configuration.
            registry: Registry used to build the dataset (defaults to shared).
        """
        self.config = config
        self.name = config.name
        self.seeds = config.seeds
        self.primary_metric = config.primary_metric
        self.version = config.version
        self._registry = registry if registry is not None else default_registry()

    def build_dataset(self, seed: int) -> BaseDataset:
        """Build and load the problem's dataset for a given replication seed.

        The seed is injected into the dataset config so the split is
        reproducible, then :meth:`BaseDataset.setup` is called if the dataset
        has not already loaded itself.

        Args:
            seed: Replication seed controlling the dataset split.

        Returns:
            The loaded, split dataset instance.
        """
        dataset_config = dict(self.config.dataset.config)
        dataset_config["seed"] = seed
        dataset = self._registry.build("alf.datasets", self.config.dataset.name, dataset_config)
        if not dataset.splits:
            dataset.setup()
        return dataset


@dataclasses.dataclass
class BenchmarkSuite:
    """A named, versioned collection of benchmark problems.

    Attributes:
        name: Suite identifier (e.g. ``"alf-protein-v1"``).
        version: Frozen version string for reproducibility.
        problems: The problems that make up the suite.
    """

    name: str
    version: str
    problems: list[BenchmarkProblem]

    @classmethod
    def from_configs(
        cls,
        name: str,
        version: str,
        problem_configs: list[ProblemConfig],
        registry: Registry | None = None,
    ) -> "BenchmarkSuite":
        """Build a suite from problem configurations.

        Args:
            name: Suite identifier.
            version: Suite version string.
            problem_configs: Problem configurations to include.
            registry: Registry used to build datasets (defaults to shared).

        Returns:
            The constructed suite.
        """
        problems = [BenchmarkProblem(config, registry) for config in problem_configs]
        return cls(name=name, version=version, problems=problems)
