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

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from alf_core import BaseDataset, BaseModel, Candidate, LabelledCandidates, Predictions
from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics

logger = logging.getLogger("alf-tools")


@dataclass
class SubsampleConfig:
    """Per-member training data subsampling strategy.

    Args:
        fraction: Fraction of training samples to use per member. Must be in (0, 1].
            1.0 with replace=False is a no-op (full dataset, no shuffle).
        replace: Sample with replacement when True (bootstrap), without when False.
        seeds: Explicit per-member seeds for the sampler. Length must match the
            number of ensemble members when provided. When None, each member's
            model-init seed is used for data sampling.
    """

    fraction: float
    replace: bool = False
    seeds: list[int] | None = None

    def __post_init__(self) -> None:
        """Validate fraction and seeds.

        Raises:
            ValueError: If fraction not in (0, 1], or if seeds is an empty list.
        """
        if not (0.0 < self.fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")
        if self.seeds is not None and len(self.seeds) == 0:
            raise ValueError("seeds must not be empty when provided.")


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
        subsample: Optional per-member subsampling config. When None (default),
            each member trains on the full training dataset.
    """

    base_seed: int | None = None
    member_seeds: list[int] | None = None
    n_members: int | None = None
    subsample: SubsampleConfig | None = None
    seeds: list[int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate that exactly one seed strategy is specified, then resolve seeds.

        Raises:
            ValueError: If neither or both seed strategies are provided, or if
                base_seed is set without n_members, or if n_members < 1, or if
                member_seeds is empty.
            AssertionError: If the code logic is incorrect and fails to resolve seeds when
                one valid strategy is provided (should be unreachable due to validation above).
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
        if self.member_seeds is not None and self.n_members is not None:
            logger.warning(
                "EnsembleWrapperConfig: n_members=%d is ignored because "
                "member_seeds is provided (len=%d).",
                self.n_members,
                len(self.member_seeds),
            )

        if self.member_seeds is not None:
            self.seeds = list(self.member_seeds)
        elif self.base_seed is not None and self.n_members is not None:
            self.seeds = [self.base_seed + i for i in range(self.n_members)]
        else:
            raise AssertionError("unreachable: validation above guarantees one branch is taken")

        negative = [s for s in self.seeds if s < 0]
        if negative:
            raise ValueError(
                f"All member seeds must be non-negative; got negative seed(s): {negative}"
            )

        if (
            self.subsample is not None
            and self.subsample.seeds is not None
            and len(self.subsample.seeds) != len(self.seeds)
        ):
            raise ValueError(
                f"subsample.seeds length ({len(self.subsample.seeds)}) must equal "
                f"the number of ensemble members ({len(self.seeds)})."
            )


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
        name: str = "ensemble",
    ):
        """Instantiate members by calling model_factory with each resolved seed.

        Args:
            model_factory: Callable that takes a seed integer and returns a BaseModel instance.
            config: Configuration specifying seeds, member count, and optional subsampling.
            name: Name identifier for this ensemble, used in logging.
        """
        self.name = name
        self.config = config
        self.members: list[BaseModel] = [model_factory(seed) for seed in config.seeds]

    def featurise(
        self,
        inputs: LabelledCandidates | list[Candidate],
        member_index: int = 0,
    ) -> Any:
        """Delegate featurisation to the specified ensemble member.

        Args:
            inputs: Input samples to featurise.
            member_index: Index of the member to use for featurisation.
                Defaults to 0. Use non-zero values for heterogeneous ensembles
                where members have distinct featurise implementations.

        Returns:
            Feature representation returned by the specified member's featurise().
        """
        return self.members[member_index].featurise(inputs)

    def _subsample(self, data: LabelledCandidates, seed: int) -> LabelledCandidates:
        """Return a random subset of data using the given seed.

        Args:
            data: Full training dataset.
            seed: RNG seed for reproducible sampling.

        Returns:
            A new LabelledCandidates containing k = max(1, round(fraction * n)) samples.

        Raises:
            RuntimeError: If subsample config is None.
        """
        cfg = self.config.subsample
        if cfg is None:
            raise RuntimeError("_subsample called but subsample config is None")
        rng = np.random.default_rng(seed)
        n = len(data)
        k = max(1, round(cfg.fraction * n))
        indices = rng.choice(n, size=k, replace=cfg.replace)
        candidates, labels = data[indices]
        return LabelledCandidates(candidates=candidates, labels=labels)

    def setup(self, dataset: BaseDataset) -> None:
        """Delegate setup to each ensemble member.

        Args:
            dataset: The dataset this ensemble will be trained on.
        """
        super().setup(dataset)
        for member in self.members:
            member.setup(dataset)

    def train(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates | None = None,
    ) -> None:
        """Train each member sequentially on full or subsampled training data.

        When subsample is configured, each member receives a random subset of
        train_data. The subsample seed is taken from subsample.seeds[i] if
        provided, otherwise from the member's model-init seed.

        val_data is forwarded directly to each member unchanged (including None).
        Members that require a validation set must handle None gracefully.

        If any member's train() call raises, a warning is logged for that member,
        training continues for the remaining members, and a summary RuntimeError
        is raised at the end listing all failures.

        Raises:
            RuntimeError: If one or more members raise an exception during training.
                The exception message will summarize all failures.
        """
        errors: list[tuple[int, Exception]] = []
        resolved_seeds = self.config.seeds
        for i, member in enumerate(self.members):
            if self.config.subsample is not None:
                subsample_seed = (
                    self.config.subsample.seeds[i]
                    if self.config.subsample.seeds is not None
                    else resolved_seeds[i]
                )
                data = self._subsample(train_data, subsample_seed)
            else:
                data = train_data
            logger.info(
                "EnsembleWrapper '%s': training member %d/%d", self.name, i + 1, len(self.members)
            )
            try:
                member.train(data, val_data)
            except Exception as exc:
                logger.warning(
                    "EnsembleWrapper: member %d/%d training failed: %s",
                    i + 1,
                    len(self.members),
                    exc,
                )
                errors.append((i, exc))
        if errors:
            summary = "; ".join(f"member {i}: {exc}" for i, exc in errors)
            raise RuntimeError(
                f"EnsembleWrapper.train() failed for {len(errors)} member(s): {summary}"
            )

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Aggregate per-member predictions into a single Predictions object.

        Returns:
            Predictions with means, variances, and empirical_dist assembled from
            all member outputs concatenated along the sample axis.

        Raises:
            ValueError: If members return empirical_dist columns of different widths.
        """
        columns: list[np.ndarray] = []
        for i, member in enumerate(self.members):
            p_i = member.predict(candidate_points)
            if p_i.empirical_dist is not None:
                columns.append(p_i.empirical_dist)
            elif p_i.means is None:
                raise ValueError(
                    f"EnsembleWrapper.predict(): member {i} returned Predictions "
                    "with both means and empirical_dist as None."
                )
            elif p_i.means.ndim == 1:
                columns.append(p_i.means[:, np.newaxis])
            else:
                columns.append(p_i.means)

        # Width check is deferred until all members have predicted because column
        # widths (MC samples vs. means-only) are not known until predict() returns.
        width_map = {i: c.shape[1] for i, c in enumerate(columns)}
        unique_widths = set(width_map.values())
        if len(unique_widths) > 1:
            detail = ", ".join(f"member {i}: {w}" for i, w in width_map.items())
            raise ValueError(
                f"EnsembleWrapper.predict(): members returned empirical_dist columns of "
                f"different widths ({detail}). All members must return the same number of "
                "columns (use consistent MC-dropout samples or means-only outputs)."
            )

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

    def get_training_summary_metrics(self) -> dict[str, float | int | np.number]:
        """Return summary metrics from all members, tagged with member index."""
        result: dict[str, float | int | np.number] = {}
        for i, member in enumerate(self.members):
            for k, v in member.get_training_summary_metrics().items():
                result[f"member_{i}/{k}"] = v
        return result

    def cleanup(self) -> None:
        """Delegate cleanup to each ensemble member.

        Attempts cleanup on every member regardless of individual failures,
        then raises a summary RuntimeError if any cleanup calls failed.

        Raises:
            RuntimeError: If one or more members raise an exception during cleanup.
                The exception message will summarize all failures.
        """
        errors: list[tuple[int, Exception]] = []
        for i, member in enumerate(self.members):
            try:
                member.cleanup()
            except Exception as exc:
                errors.append((i, exc))
        if errors:
            summary = "; ".join(f"member {i}: {exc}" for i, exc in errors)
            raise RuntimeError(
                f"EnsembleWrapper.cleanup() failed for {len(errors)} member(s): {summary}"
            )
