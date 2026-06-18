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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics


@dataclass
class RoundMetrics:
    """All metrics for a single acquisition round.

    `round` is the canonical round number — it is not duplicated inside
    `metrics`.  `training_history` carries per-epoch `SurrogateEpochMetrics`
    objects for backends that support step-based logging

    Attributes:
        round: The round number this instance describes.
        metrics: Flat dict of scalar metrics for this round (e.g. tell_time,
            surrogate/test_spearman, dataset/num_train).
        training_history: Per-epoch metrics recorded during surrogate training
            in this round.  Empty list when no training occurred (e.g. zero-shot
            tasks) or before `Surrogate.fit()` has been called.
    """

    round: int
    metrics: dict[str, Any] = field(default_factory=dict)
    training_history: list[SurrogateEpochMetrics] = field(default_factory=list)
