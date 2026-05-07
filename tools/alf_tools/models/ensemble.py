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

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Union

import numpy as np
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions
from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics

logger = logging.getLogger("alf-tools")


@dataclass
class EnsembleWrapperConfig:
    """Configuration for the generic ensemble wrapper.

    Exactly one of `base_seed` or `member_seeds` must be provided.

    Args:
        base_seed: Base integer from which member seeds are derived as
            [base_seed, base_seed+1, ..., base_seed+n_members-1].
            Requires n_members to also be set.
        member_seeds: Explicit list of seeds, one per member.
            len(member_seeds) determines the number of members.
            n_members is ignored when this is set.
        n_members: Number of ensemble members. Only used when base_seed is set.
    """

    base_seed: int | None = None
    member_seeds: list[int] | None = None
    n_members: int | None = None
    _resolved_seeds: list[int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate that exactly one seed strategy is specified, then resolve seeds.

        Raises:
            ValueError: If neither or both seed strategies are provided, or if
                base_seed is set without n_members, or if n_members < 1, or if
                member_seeds is empty.
            AssertionError: If the logic for resolving seeds is somehow
                incorrect (should be unreachable).
        """
        if self.base_seed is None and self.member_seeds is None:
            raise ValueError("Exactly one of base_seed or member_seeds must be set; got neither.")
        if self.base_seed is not None and self.member_seeds is not None:
            raise ValueError("Exactly one of base_seed or member_seeds must be set; got both.")
        if self.base_seed is not None and self.n_members is None:
            raise ValueError("n_members must be set when base_seed is provided.")
        if self.base_seed is not None and self.n_members is not None and self.n_members < 1:
            raise ValueError(f"n_members must be >= 1, got {self.n_members}")
        if self.member_seeds is not None and len(self.member_seeds) == 0:
            raise ValueError("member_seeds must not be empty.")

        if self.member_seeds is not None:
            self._resolved_seeds = list(self.member_seeds)
        elif self.base_seed is not None and self.n_members is not None:
            self._resolved_seeds = [self.base_seed + i for i in range(self.n_members)]
        else:
            raise AssertionError("unreachable: validation above guarantees one branch is taken")

    def resolve_seeds(self) -> list[int]:
        """Return the ordered list of per-member seeds."""
        return self._resolved_seeds


class EnsembleWrapper(BaseModel):
    """Generic ensemble wrapper that composes N BaseModel instances.

    Assembles Predictions.empirical_dist from per-member outputs:
    - If a member returns empirical_dist, all its columns are concatenated.
    - If a member returns only means, that column is appended as a single column.
    """

    def __init__(
        self,
        model_factory: Callable[[int], BaseModel],
        config: EnsembleWrapperConfig,
        name: str = "ensemble_wrapper",
    ):
        """Instantiate members by calling model_factory with each resolved seed."""
        self.name = name
        self.config = config
        seeds = config.resolve_seeds()
        self.members: list[BaseModel] = [model_factory(seed) for seed in seeds]

    def featurise(self, inputs: Union[LabelledCandidates, list[Candidate]]) -> Any:
        """Delegate featurisation to the first ensemble member.

        Returns:
            Feature representation returned by the first member's featurise().
        """
        return self.members[0].featurise(inputs)

    def train(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates | None = None,
    ) -> None:
        """Train each member sequentially, logging progress."""
        for i, member in enumerate(self.members):
            logger.info(
                f"EnsembleWrapper '{self.name}': training member {i + 1}/{len(self.members)}"
            )
            member.train(train_data, val_data)

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Aggregate per-member predictions into a single Predictions object.

        Returns:
            Predictions with means, variances, and empirical_dist assembled from
            all member outputs concatenated along the sample axis.
        """
        columns: list[np.ndarray] = []
        for member in self.members:
            p_i = member.predict(candidate_points)
            if p_i.empirical_dist is not None:
                columns.append(p_i.empirical_dist)
            else:
                columns.append(p_i.means[:, np.newaxis])

        empirical_dist = np.concatenate(columns, axis=1)
        means = empirical_dist.mean(axis=1)
        variances = empirical_dist.var(axis=1)
        return Predictions(means=means, variances=variances, empirical_dist=empirical_dist)

    def sample(self, condition: Any | None = None) -> list[Candidate]:
        """Not implemented; raises NotImplementedError."""
        raise NotImplementedError("Sampling is not implemented for EnsembleWrapper.")

    def get_epoch_metrics(self) -> list[SurrogateEpochMetrics]:
        """Return per-epoch metrics from all members, tagged with member index."""
        result: list[SurrogateEpochMetrics] = []
        for i, member in enumerate(self.members):
            for em in member.get_epoch_metrics():
                tagged = {f"member_{i}/{k}": v for k, v in em.additional_metrics.items()}
                result.append(
                    SurrogateEpochMetrics(
                        epoch=em.epoch,
                        train_loss=em.train_loss,
                        val_loss=em.val_loss,
                        additional_metrics=tagged,
                    )
                )
        return result

    def get_training_summary_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Return summary metrics from all members, tagged with member index."""
        result: dict[str, Union[float, int, np.number]] = {}
        for i, member in enumerate(self.members):
            for k, v in member.get_training_summary_metrics().items():
                result[f"member_{i}/{k}"] = v
        return result

    def cleanup(self) -> None:
        """Delegate cleanup to each ensemble member."""
        for member in self.members:
            member.cleanup()
