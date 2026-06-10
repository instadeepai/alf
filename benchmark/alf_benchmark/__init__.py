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

"""alf-benchmark: a benchmark suite layer over alf-core and alf-tools."""

from alf_benchmark.config import (
    ComponentSpec,
    MethodConfig,
    OracleSpec,
    ProblemConfig,
    RunConfig,
    TaskConfig,
)
from alf_benchmark.method import BenchmarkMethod
from alf_benchmark.problem import BenchmarkProblem, BenchmarkSuite
from alf_benchmark.registry import (
    GROUPS,
    Registry,
    RegistryEntry,
    default_registry,
    register,
)
from alf_benchmark.results import BenchmarkResults
from alf_benchmark.runner import BenchmarkRunner, run_benchmark
from alf_benchmark.schema import Manifest, read_manifest, write_manifest
from alf_benchmark.seeding import seed_everything
from alf_benchmark.yaml_loader import load_run_config

__all__ = [
    "GROUPS",
    "BenchmarkMethod",
    "BenchmarkProblem",
    "BenchmarkResults",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "ComponentSpec",
    "Manifest",
    "MethodConfig",
    "OracleSpec",
    "ProblemConfig",
    "Registry",
    "RegistryEntry",
    "RunConfig",
    "TaskConfig",
    "default_registry",
    "load_run_config",
    "read_manifest",
    "register",
    "run_benchmark",
    "seed_everything",
    "write_manifest",
]
