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
from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        if self.base_seed is None and self.member_seeds is None:
            raise ValueError(
                "Exactly one of base_seed or member_seeds must be set; got neither."
            )
        if self.base_seed is not None and self.member_seeds is not None:
            raise ValueError(
                "Exactly one of base_seed or member_seeds must be set; got both."
            )
        if self.base_seed is not None and self.n_members is None:
            raise ValueError("n_members must be set when base_seed is provided.")
        if self.n_members is not None and self.n_members < 1:
            raise ValueError(f"n_members must be >= 1, got {self.n_members}")
        if self.member_seeds is not None and len(self.member_seeds) == 0:
            raise ValueError("member_seeds must not be empty.")

    def resolve_seeds(self) -> list[int]:
        """Return the ordered list of per-member seeds."""
        if self.member_seeds is not None:
            return list(self.member_seeds)
        assert self.base_seed is not None and self.n_members is not None
        return [self.base_seed + i for i in range(self.n_members)]


class EnsembleWrapper(BaseModel):
    """Generic ensemble wrapper that composes N BaseModel instances.

    Assembles Predictions.empirical_dist from per-member outputs:
    - If a member returns empirical_dist (e.g. MC dropout MLPModel), all its
      columns are concatenated.
    - If a member returns only means, that column is appended as a single column.

    This gives three modes when wrapping MLPModel:
        deep ensemble  : N members, n_mc_passes=0  → empirical_dist (N_cand, N)
        MC dropout     : 1 member,  n_mc_passes=T  → empirical_dist (N_cand, T)
        combined       : N members, n_mc_passes=T  → empirical_dist (N_cand, N*T)
    """

    def __init__(
        self,
        model_factory: Callable[[int], BaseModel],
        config: EnsembleWrapperConfig,
        name: str = "ensemble_wrapper",
    ):
        self.name = name
        self.config = config
        seeds = config.resolve_seeds()
        self.members: list[BaseModel] = [model_factory(seed) for seed in seeds]

    def featurise(self, inputs: Union[LabelledCandidates, list[Candidate]]) -> Any:
        return self.members[0].featurise(inputs)

    def train(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates | None = None,
    ) -> None:
        for i, member in enumerate(self.members):
            logger.info(f"EnsembleWrapper '{self.name}': training member {i + 1}/{len(self.members)}")
            member.train(train_data, val_data)

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
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
        raise NotImplementedError("Sampling is not implemented for EnsembleWrapper.")

    def get_epoch_metrics(self) -> list[SurrogateEpochMetrics]:
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
        result: dict[str, Union[float, int, np.number]] = {}
        for i, member in enumerate(self.members):
            for k, v in member.get_training_summary_metrics().items():
                result[f"member_{i}/{k}"] = v
        return result

    def cleanup(self) -> None:
        for member in self.members:
            member.cleanup()
