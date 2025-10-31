from dataclasses import dataclass, field
from typing import List, Any, TYPE_CHECKING
import copy
from core.dataclasses import LabeledCandidates
import logging
import pandas as pd
import os
from core.utils.io import input_handler
import matplotlib.pyplot as plt

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]")
log = logging.getLogger("rich")

if TYPE_CHECKING:
    from core.dataset.base_dataset import BaseDataset
    from core.surrogate.surrogate import Surrogate


@dataclass
class TaskState:
    """Tracks the state of an active learning / optimization task."""

    dataset: "BaseDataset"
    surrogate: "Surrogate"
    round: int = 0
    acq_batch_size: int = 0
    num_acq_rounds: int = 0
    save_round_predictions: bool = False
    history: List = field(default_factory=list)
    round_metrics: dict[str, Any] = field(default_factory=dict)

    def update(self, acquired_candidates: LabeledCandidates) -> None:
        """Update the task state, including the dataset, with the acquired candidates."""
        self.history.append(copy.copy(acquired_candidates))
        if self.round != 0:
            self.dataset.update_splits(acquired_candidates)
        self.round += 1

    def print_metrics(self, round_name: int | str) -> None:
        """Stringify the metrics."""
        metrics_list: list = []
        for key, value in self.round_metrics.items():
            # Skip the "round" key explicitly; only accept float and int types
            if (key != "round") and isinstance(value, (float, int)):
                metrics_list.append(f"{key}: {value:.3f}")
        metrics_str = "\t".join(metrics_list)
        log.info(f"Round {round_name}:\t{metrics_str}")  # noqa: E231

    def save_metrics(
        self,
        output_dir: str,
    ) -> None:
        """Save the multiround metrics to the output directory"""
        numeric_metrics = {
            k: v for k, v in self.round_metrics.items() if not isinstance(v, plt.Figure)
        }
        metrics_df = pd.DataFrame.from_records([numeric_metrics])

        if input_handler.isfile(os.path.join(output_dir, "metrics.csv")):
            saved_df = input_handler.read_csv(os.path.join(output_dir, "metrics.csv"))
            metrics_df = pd.concat([saved_df, metrics_df])

        input_handler.save_csv(os.path.join(output_dir, "metrics.csv"), metrics_df)

    def save_history(self, output_dir: str) -> None:
        """Save the multiround history of acquisitions to the output directory"""
        for acq_round, acq_points in enumerate(self.history):
            input_handler.save_csv(
                os.path.join(output_dir, f"acq_round_{acq_round}.csv"),
                acq_points.to_dataframe(),
            )

    def save(self, save_path: str | None, _verbose: bool = False) -> None:
        """Save the metrics and history to the output directory"""
        if _verbose:
            log.info(f"Saving metrics and history to {save_path}")

        if save_path:
            self.save_metrics(save_path)
            self.save_history(save_path)

    def should_terminate(self) -> bool:
        """Check if the task should be terminated."""
        if len(self.dataset.candidate_pool) < self.acq_batch_size:
            log.info(
                "Optimizer is signalling that optimization is complete, i.e. batch size "
                + f"({self.acq_batch_size}) > remaining candidate pool"
                + f"({len(self.dataset.candidate_pool)}), breaking"
            )
            return True
        return False
