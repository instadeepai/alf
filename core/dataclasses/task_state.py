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
    from core.datasets.base_dataset import BaseDataset
    from core.surrogate.surrogate import Surrogate


@dataclass
class TaskState:
    """Tracks the state of an active learning / optimization task."""

    dataset: "BaseDataset"
    surrogate: "Surrogate"
    step_count: int = 0
    acq_batch_size: int = 0
    num_acq_rounds: int = 0
    save_round_predictions: bool = False
    alphabet: Any = None
    history: List = field(default_factory=list)
    round_metrics: dict[str, Any] = field(default_factory=dict)

    def update(self, acquired_candidates: LabeledCandidates) -> None:
        self.history.append(copy.copy(acquired_candidates))
        if self.step_count != 0:
            self.dataset.update_splits(acquired_candidates)
        self.step_count += 1

    def save_metrics(
        self,
        output_dir: str,
        _verbose: bool = False,
    ) -> None:
        """Save the multiround metrics to the output directory"""
        if _verbose:
            log.info(f"Saving metrics history to {output_dir}")

        # Drop plot entries (e.g., plt.Figure)
        numeric_metrics = {
            k: v for k, v in self.round_metrics.items() if not isinstance(v, plt.Figure)
        }

        metrics_df = pd.DataFrame.from_records([numeric_metrics])

        if input_handler.isfile(os.path.join(output_dir, "metrics.csv")):
            saved_df = input_handler.read_csv(os.path.join(output_dir, "metrics.csv"))
            metrics_df = pd.concat([saved_df, metrics_df])

        input_handler.save_csv(os.path.join(output_dir, "metrics.csv"), metrics_df)

    # def save_history(self, output_dir: str, _verbose: bool = False) -> None:
    #     """Save the multiround history of acquisitions to the output directory"""
    #     pass

    def save(self, save_path: str, _verbose: bool = False) -> None:
        self.save_metrics(save_path, _verbose)
        # self.save_history(save_path, _verbose)


    def should_terminate(self) -> bool:
        """Check if the task should be terminated."""
        raise NotImplementedError("should_terminate not implemented for TaskState")
