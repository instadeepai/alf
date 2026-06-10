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

"""Pydantic configuration models: the config-as-code core API.

These declarative, validated models describe *what* to benchmark
(:class:`ProblemConfig`) and *how* (:class:`MethodConfig`). They are the single
source of truth: the Python API constructs them directly and the YAML loader
(Phase 3) parses into the same models, so validation is shared.
"""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Family = Literal["design", "supervised", "zero_shot"]
"""Experiment family, mapping one-to-one onto an ALF task."""

OracleMode = Literal["offline", "online"]
"""Oracle label source: a held-out dataset (offline) or a live model (online)."""


class ComponentSpec(BaseModel):
    """A reference to a registered component plus its configuration.

    Attributes:
        name: Registry name of the component (e.g. ``"cnn"``, ``"gfp"``).
        config: Configuration dictionary passed to the registry builder.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    config: dict[str, Any] = Field(default_factory=dict)


class OracleSpec(BaseModel):
    """Specification of the oracle that supplies ground-truth labels.

    Offline binds the oracle to the problem's dataset (set by the runner);
    online builds a model scorer from ``scorer``.

    Attributes:
        mode: ``"offline"`` (dataset scorer) or ``"online"`` (model scorer).
        scorer: Model component for online mode; must be ``None`` offline.
    """

    model_config = ConfigDict(extra="forbid")

    mode: OracleMode = "offline"
    scorer: ComponentSpec | None = None

    @model_validator(mode="after")
    def _validate_scorer(self) -> Self:
        """Validate the scorer matches the oracle mode.

        Returns:
            The validated specification.

        Raises:
            ValueError: If online mode has no scorer, or offline mode has one.
        """
        if self.mode == "online" and self.scorer is None:
            raise ValueError("oracle.mode='online' requires a 'scorer' model spec.")
        if self.mode == "offline" and self.scorer is not None:
            raise ValueError(
                "oracle.mode='offline' binds to the problem dataset; 'scorer' must be unset."
            )
        return self


class TaskConfig(BaseModel):
    """Configuration of the ALF task that runs one replication.

    Attributes:
        num_acq_rounds: Number of acquisition rounds (design family).
        acq_batch_size: Candidates acquired per round (design family).
        save_round_predictions: Whether to persist per-round test predictions.
    """

    model_config = ConfigDict(extra="forbid")

    num_acq_rounds: int = Field(default=0, ge=0)
    acq_batch_size: int = Field(default=0, ge=0)
    save_round_predictions: bool = False


class MethodConfig(BaseModel):
    """A frozen, versioned definition of *how* to optimise.

    Attributes:
        name: Identifier for the method (used in result paths and leaderboards).
        family: Experiment family selecting the ALF task.
        surrogate: Surrogate model component.
        acquisition: Acquisition function component (required for design).
        search: Search function component (required for design).
        oracle: Oracle specification (offline/online).
        task: Task configuration (rounds, batch size).
        version: Version string; frozen so results stay comparable.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    family: Family = "design"
    surrogate: ComponentSpec
    acquisition: ComponentSpec | None = None
    search: ComponentSpec | None = None
    oracle: OracleSpec = Field(default_factory=OracleSpec)
    task: TaskConfig = Field(default_factory=TaskConfig)
    version: str = "0.1.0"

    @model_validator(mode="after")
    def _validate_family(self) -> Self:
        """Validate components required by the chosen family are present.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If a design method is missing an acquisition or search
                function, or the online oracle pairs with a dataset search.
        """
        if self.family == "design":
            if self.acquisition is None or self.search is None:
                raise ValueError(
                    "design methods require both 'acquisition' and 'search' components."
                )
        return self


class ProblemConfig(BaseModel):
    """A frozen, versioned definition of *what* to optimise.

    Attributes:
        name: Identifier for the problem (used in result paths and leaderboards).
        dataset: Dataset component specification.
        seeds: Replication seeds; each seed is one independent run.
        primary_metric: Metric column that defines success for this problem.
        version: Version string; frozen so results stay comparable.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    dataset: ComponentSpec
    seeds: list[int] = Field(default_factory=lambda: [0])
    primary_metric: str = "optimizer/regret"
    version: str = "0.1.0"

    @model_validator(mode="after")
    def _validate_seeds(self) -> Self:
        """Validate the seed list is non-empty.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If no seeds are provided.
        """
        if not self.seeds:
            raise ValueError("problem must define at least one seed.")
        return self


class RunConfig(BaseModel):
    """Top-level configuration for a benchmark run (suite + methods).

    Attributes:
        suite_name: Name of the suite being run.
        suite_version: Version of the suite being run.
        output_dir: Directory results are written to.
        problems: Problems forming the suite.
        methods: Methods evaluated against every problem.
        deterministic: Request deterministic kernels where supported.
    """

    model_config = ConfigDict(extra="forbid")

    suite_name: str = "adhoc"
    suite_version: str = "0.1.0"
    output_dir: str
    problems: list[ProblemConfig]
    methods: list[MethodConfig]
    deterministic: bool = False
