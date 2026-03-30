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

    Explicit fields provide discoverability for the standard CNN metrics.
    The ``extra`` dict is an escape hatch for model-specific additional metrics.
    ``None`` values are excluded when converting to a flat dict, so backends
    only receive metrics that were actually computed.

    Attributes:
        epoch: Zero-based epoch index.
        train_loss: Training loss for this epoch.
        val_loss: Validation loss, or None if no validation data was provided.
        train_spearman: Spearman correlation on training set, or None.
        val_spearman: Spearman correlation on validation set, or None.
        train_mse: MSE on training set, or None.
        val_mse: MSE on validation set, or None.
        extra: Additional model-specific metrics passed through to backends.
    """

    epoch: int
    train_loss: float
    val_loss: float | None = None
    train_spearman: float | None = None
    val_spearman: float | None = None
    train_mse: float | None = None
    val_mse: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    def to_metrics_dict(self) -> dict[str, float]:
        """Return a flat dict of non-None metric values merged with extra.

        Returns:
            Flat dict with ``epoch`` (as float), ``train_loss``, any non-None
            optional fields, and all entries from ``extra``.
        """
        result: dict[str, float] = {
            "epoch": float(self.epoch),
            "train_loss": self.train_loss,
        }
        for key in ("val_loss", "train_spearman", "val_spearman", "train_mse", "val_mse"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        result.update(self.extra)
        return result
