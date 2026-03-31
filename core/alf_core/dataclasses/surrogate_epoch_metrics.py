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

from dataclasses import dataclass, field


@dataclass
class SurrogateEpochMetrics:
    """Per-epoch training metrics for a surrogate model.

    Explicit fields cover the standard required metrics.
    The ``additional_metrics`` dict holds model-specific metrics (e.g. spearman,
    mse) and any other optional values.  ``None`` values are excluded when
    converting to a flat dict, so backends only receive metrics that were
    actually computed.

    Attributes:
        epoch: Zero-based epoch index.
        train_loss: Training loss for this epoch.
        val_loss: Validation loss, or None if no validation data was provided.
        additional_metrics: Model-specific metrics (e.g. ``train_spearman``,
            ``val_spearman``, ``train_mse``, ``val_mse``) passed through to backends.
    """

    epoch: int
    train_loss: float
    val_loss: float | None = None
    additional_metrics: dict[str, float] = field(default_factory=dict)

    def to_metrics_dict(self) -> dict[str, int | float]:
        """Return a flat dict of non-None metric values merged with additional_metrics.

        Returns:
            Flat dict with ``epoch`` (as float), ``train_loss``, any non-None
            optional fields, and all entries from ``additional_metrics``.
        """
        result: dict[str, int | float] = {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
        }
        if self.val_loss is not None:
            result["val_loss"] = self.val_loss
        result.update(self.additional_metrics)
        return result
