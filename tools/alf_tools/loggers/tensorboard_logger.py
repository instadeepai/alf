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

from alf_core.dataclasses import State
from alf_core.utils.state_logger import StateLogger


class TensorBoardLogger(StateLogger):
    """Logs round-level and per-epoch training metrics to TensorBoard.

    Uses two distinct namespaces to keep step axes independent:
    - ``training/<metric>`` at ``step=epoch`` for per-epoch training curves.
    - ``round/<metric>`` at ``step=round`` for per-round evaluation metrics.

    Args:
        writer: A ``torch.utils.tensorboard.SummaryWriter`` instance.
    """

    def __init__(self, writer) -> None:
        """Initialize the TensorBoardLogger with a SummaryWriter.

        Args:
            writer: A ``torch.utils.tensorboard.SummaryWriter`` instance.
        """
        self.writer = writer

    def log(self, state: State, round_name: str | None = None) -> None:
        """Log training history and round metrics to TensorBoard.

        Args:
            state: Task state containing ``round_metrics`` with ``training_history``
                and scalar ``metrics``.
            round_name: Unused by TensorBoard (step axes provide the x-axis).
        """
        self._log_training_history_per_round(state)
        self._log_round_metrics(state)

    def _log_round_metrics(self, state: State) -> None:
        """Write each scalar in round_metrics.metrics as a TensorBoard scalar.

        Tag: ``round/<key>``, step: ``state.round_metrics.round``.
        """
        step = state.round_metrics.round
        for key, value in state.round_metrics.metrics.items():
            self.writer.add_scalar(f"round/{key}", value, step)

    def _log_training_history_per_round(self, state: State) -> None:
        """Write per-epoch metrics from training_history to TensorBoard.

        Tag: ``training/<key>``, step: ``epoch_metrics.epoch``.
        """
        for em in state.round_metrics.training_history:
            for key, value in em.to_metrics_dict().items():
                self.writer.add_scalar(f"training/{key}", value, em.epoch)
