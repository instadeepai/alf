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

"""Shared helpers for the benchmark_examples taster scripts.

Runs active-learning (DesignTask) experiments, collects per-round metrics from the
real FileStateLogger output, aggregates across seeds, and plots comparison curves.
A small taster for the kind of comparison the (deferred) alf_benchmark layer formalises.
"""

from __future__ import annotations

import argparse
import logging
import random
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from alf_core import (
    BaseDatasetConfig,
    DatasetSearch,
    DesignTask,
    FileStateLogger,
    Modality,
    Optimizer,
    Oracle,
    ProblemType,
    Surrogate,
)
from alf_core.dataset.base_dataset import BaseDataset
from alf_core.model.base_model import BaseModel
from alf_tools.datasets import FLIP, GFP, FLIPConfig, ProteinGym
from alf_tools.datasets.proteingym import ProteinGymConfig
from alf_tools.models import (
    CNNModel,
    CNNModelConfig,
    CNNTrainConfig,
    FeaturizerConfig,
    GPModel,
    GPModelConfig,
    GPTrainConfig,
)
from alf_tools.optimizer.acquisition_functions import (
    UCB,
    CoreSet,
    ExpectedImprovement,
    Greedy,
)

matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt  # noqa: E402

try:  # ESM-2 needs the optional 'esm2' extra (transformers); bundled in the benchmark group.
    from alf_tools.models import ESM2Model, ESM2ModelConfig, ESM2TrainConfig

    _ESM2_AVAILABLE = True
except ImportError:
    _ESM2_AVAILABLE = False

log = logging.getLogger("benchmark_examples")

DEFAULT_ESM_MODEL_ID = "facebook/esm2_t6_8M_UR50D"  # small (~8M) checkpoint, fast on CPU

# Acquisition functions that need per-candidate uncertainty (variances).
_VARIANCE_REQUIRING = {"ucb", "ei"}


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def set_seeds(seed: int) -> None:
    """Seed Python, NumPy and torch for reproducible runs.

    Args:
        seed: Random seed applied to all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_logging() -> None:
    """Send the scripts' progress logging to the console."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def build_dataset(name: str, seed: int) -> BaseDataset:
    """Build a sequence dataset by short name (un-setup; caller calls setup()).

    Args:
        name: One of ``"gfp"``, ``"flip"``, ``"proteingym"``.
        seed: Split seed — the SAME seed must be used across configs for a fair
            (paired) comparison.

    Returns:
        An un-setup dataset.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    if name == "gfp":
        return GFP(
            BaseDatasetConfig(
                name="gfp",
                modality=Modality.SEQUENCE,
                seed=seed,
                train_ratio=0.4,
                validation_frac=0.1,
                test_ratio=0.1,
                split_type="random",
                problem_type=ProblemType.REGRESSION,
            )
        )
    if name == "flip":
        return FLIP(
            FLIPConfig(
                name="flip_gb1",
                modality=Modality.SEQUENCE,
                seed=seed,
                flip_dataset="gb1",
                flip_split="one_vs_rest",
                train_ratio=0.4,
                validation_frac=0.1,
                test_ratio=0.4,
                problem_type=ProblemType.REGRESSION,
            )
        )
    if name == "proteingym":
        return ProteinGym(
            ProteinGymConfig(
                name="proteingym_if1",
                modality=Modality.SEQUENCE,
                seed=seed,
                dms_name="IF1_ECOLI_Kelsic_2016",
                dms_type="singles",
                train_ratio=0.4,
                validation_frac=0.1,
                test_ratio=0.1,
                split_type="random",
                problem_type=ProblemType.REGRESSION,
            )
        )
    raise ValueError(f"Unknown dataset: {name!r}. Expected 'gfp', 'flip' or 'proteingym'.")


def build_cnn(seed: int, epochs: int) -> BaseModel:
    """Build a CNN surrogate (sequence model).

    Args:
        seed: Unused directly (global RNG is seeded by the caller); kept for a uniform
            builder signature.
        epochs: Number of training epochs.

    Returns:
        An untrained CNN model.
    """
    return CNNModel(
        name="cnn", model_config=CNNModelConfig(), train_config=CNNTrainConfig(num_epochs=epochs)
    )


def build_gp(seed: int, epochs: int) -> BaseModel:
    """Build a GP surrogate with one-hot sequence featurisation.

    Args:
        seed: Unused directly (global RNG is seeded by the caller); kept for a uniform
            builder signature.
        epochs: Number of marginal-likelihood optimisation iterations.

    Returns:
        An untrained GP model.
    """
    return GPModel(
        name="gp",
        model_config=GPModelConfig(kernel_type="rbf"),
        train_config=GPTrainConfig(num_iterations=epochs),
        featurizer_config=FeaturizerConfig(featurizer_type="one_hot"),
    )


def build_esm(seed: int, epochs: int, model_id: str = DEFAULT_ESM_MODEL_ID) -> BaseModel:
    """Build an ESM-2 surrogate (frozen backbone + trained linear head).

    Args:
        seed: Seed for linear-head initialisation.
        epochs: Number of linear-head training epochs.
        model_id: HuggingFace ESM-2 checkpoint.

    Returns:
        An untrained ESM-2 model.

    Raises:
        SystemExit: If the ``esm2`` extra (transformers) is not installed.
    """
    if not _ESM2_AVAILABLE:
        raise SystemExit(
            "The ESM-2 surrogate needs the 'esm2' extra (transformers).\n"
            "  Run with:  uv run --group benchmark python benchmark_examples/...\n"
            '  or install: pip install "alf_tools[esm2]"'
        )
    return ESM2Model(
        name="esm2",
        model_config=ESM2ModelConfig(model_id=model_id, seed=seed),
        train_config=ESM2TrainConfig(mode="linear_head", loss_fn="mse", num_epochs=epochs),
    )


def build_acquisition(name: str, model_name: str):
    """Build an acquisition function, validating it is compatible with the model.

    Args:
        name: One of ``"greedy"``, ``"ucb"``, ``"ei"``, ``"core_set"``.
        model_name: Surrogate short name, for the compatibility check.

    Returns:
        The constructed acquisition function.

    Raises:
        SystemExit: If a variance-requiring acquisition is paired with a model that
            does not produce uncertainty (e.g. ``ucb`` with ``cnn``/``esm``).
        ValueError: If ``name`` is unknown.
    """
    if name in _VARIANCE_REQUIRING and model_name != "gp":
        raise SystemExit(
            f"Acquisition {name!r} needs per-candidate uncertainty, which model "
            f"{model_name!r} does not provide. Use a GP surrogate or 'greedy'/'core_set'."
        )
    if name == "greedy":
        return Greedy()
    if name == "ucb":
        return UCB(alpha=1.0)
    if name == "ei":
        return ExpectedImprovement()
    if name == "core_set":
        return CoreSet()
    raise ValueError(f"Unknown acquisition: {name!r}.")


def _check_budget(dataset: BaseDataset, num_rounds: int, batch_size: int) -> None:
    """Fail fast if the acquisition budget would exhaust the candidate pool.

    Args:
        dataset: A dataset on which ``setup()`` has already been called.
        num_rounds: Planned acquisition rounds.
        batch_size: Candidates acquired per round.

    Raises:
        SystemExit: If ``num_rounds * batch_size`` would empty the pool.
    """
    pool = len(dataset.candidate_pool)
    requested = num_rounds * batch_size
    if requested >= pool:
        raise SystemExit(
            f"Acquisition budget too large: {num_rounds} rounds x {batch_size} = {requested}, "
            f"but the pool holds only {pool}. Lower --num-rounds/--batch-size "
            f"(keep their product below {pool})."
        )


# --------------------------------------------------------------------------- #
# Experiment runner
# --------------------------------------------------------------------------- #
def run_al_experiment(
    dataset_name: str,
    model_name: str,
    build_model,
    acquisition: str,
    seed: int,
    num_rounds: int,
    batch_size: int,
    epochs: int,
    workdir: Path,
) -> pd.DataFrame:
    """Run one design-task experiment and return its per-round metrics.

    Builds its OWN dataset with ``seed`` (paired split) and seeds all RNGs before
    constructing the model, so configs sharing a seed are directly comparable.

    Args:
        dataset_name: Dataset short name.
        model_name: Surrogate short name (for the acquisition compatibility check).
        build_model: Callable ``(seed, epochs) -> BaseModel``.
        acquisition: Acquisition function short name.
        seed: Shared split + RNG seed.
        num_rounds: Acquisition rounds.
        batch_size: Candidates acquired per round.
        epochs: Training length (CNN epochs / GP iterations / ESM head epochs).
        workdir: Directory for this run's FileStateLogger output.

    Returns:
        DataFrame of per-round metrics with an explicit ``round`` column (0 = initial)
        and a derived ``best_found_so_far`` column.
    """
    set_seeds(seed)
    dataset = build_dataset(dataset_name, seed)
    dataset.setup()
    _check_budget(dataset, num_rounds, batch_size)
    init_best = float(dataset.train_dataset.labels.max())

    surrogate = Surrogate(model=build_model(seed, epochs))
    optimizer = Optimizer(
        acquisition_fn=build_acquisition(acquisition, model_name), search_fn=DatasetSearch()
    )
    oracle = Oracle(scorer=dataset)

    workdir.mkdir(parents=True, exist_ok=True)
    task = DesignTask(num_acq_rounds=num_rounds, acq_batch_size=batch_size)
    state = task.setup(dataset=dataset, surrogate=surrogate, seed=seed)
    task.run(
        state=state,
        state_loggers=[FileStateLogger(output_path=workdir)],
        optimizer=optimizer,
        oracle=oracle,
    )
    return _read_metrics(workdir / "metrics.csv", init_best)


def _read_metrics(metrics_path: Path, init_best: float) -> pd.DataFrame:
    """Parse a FileStateLogger metrics.csv defensively into a per-round DataFrame.

    metrics.csv has no explicit round column and a ragged initial row (acquisition
    metrics are NaN before the first acquisition). We add an explicit ``round`` index
    (0 = initial) and a ``best_found_so_far`` column seeded with the initial train max.

    Args:
        metrics_path: Path to the run's metrics.csv.
        init_best: Max label in the initial training set (round-0 best-found seed).

    Returns:
        DataFrame with the original columns plus ``round`` and ``best_found_so_far``.
    """
    df = pd.read_csv(metrics_path).reset_index(drop=True)
    df["round"] = range(len(df))

    running, current = [], init_best
    round_max = (
        df["acquired_candidates/round_max"] if "acquired_candidates/round_max" in df else None
    )
    series = round_max if round_max is not None else [np.nan] * len(df)
    for value in series:
        if not pd.isna(value):
            current = max(current, float(value))
        running.append(current)
    df["best_found_so_far"] = running
    return df


def find_recall_column(df: pd.DataFrame) -> str | None:
    """Return a top-K recall column name if present (prefers the percentile recall).

    Args:
        df: A per-round metrics DataFrame.

    Returns:
        A recall column name, or None if no recall column is present.
    """
    recall_cols = [c for c in df.columns if c.startswith("optimizer/top_") and c.endswith("recall")]
    if not recall_cols:
        return None
    pc = [c for c in recall_cols if "pc_recall" in c]
    return pc[0] if pc else recall_cols[0]


# --------------------------------------------------------------------------- #
# Aggregation + plotting
# --------------------------------------------------------------------------- #
def _stack(per_seed: list[pd.DataFrame], col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack a metric column across seed runs aligned by round.

    Args:
        per_seed: One metrics DataFrame per seed (same number of rounds).
        col: Metric column to stack.

    Returns:
        Tuple ``(rounds, matrix[seeds, rounds], mean[rounds])``.
    """
    rounds = per_seed[0]["round"].to_numpy()
    matrix = np.vstack([d[col].to_numpy(dtype=float) for d in per_seed])
    with warnings.catch_warnings():  # rounds with no acquisition metrics are all-NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(matrix, axis=0)
    return rounds, matrix, mean


def plot_metric(
    ax, per_seed_by_config: dict[str, list[pd.DataFrame]], col: str, title: str, ylabel: str
) -> None:
    """Plot mean curves (bold) with faint per-seed lines, one colour per config.

    Skips any config missing ``col``. Honest about run-to-run variance.

    Args:
        ax: Matplotlib axis to draw on.
        per_seed_by_config: Maps config label -> list of per-seed metric DataFrames.
        col: Metric column to plot.
        title: Axis title.
        ylabel: Y-axis label.
    """
    cmap = plt.get_cmap("tab10")
    for idx, (label, per_seed) in enumerate(per_seed_by_config.items()):
        if any(col not in d.columns for d in per_seed):
            log.info("  (skipping %s for %r: column not present)", col, label)
            continue
        colour = cmap(idx % 10)
        rounds, matrix, mean = _stack(per_seed, col)
        for row in matrix:
            ax.plot(rounds, row, color=colour, alpha=0.2, linewidth=0.8)
        ax.plot(rounds, mean, color=colour, linewidth=2.2, label=label)
    ax.set_title(title)
    ax.set_xlabel("acquisition round")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)


def save_figure(fig, path: Path) -> None:
    """Write a figure to PNG and close it.

    Args:
        fig: Matplotlib figure.
        path: Destination PNG path (parent dirs created if missing).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info("Saved plot: %s", path)


def summarise_final(
    per_seed_by_config: dict[str, list[pd.DataFrame]], columns: dict[str, str]
) -> pd.DataFrame:
    """Build a summary table of final-round metrics (mean ± std over seeds) per config.

    Args:
        per_seed_by_config: Maps config label -> list of per-seed metric DataFrames.
        columns: Maps a display name -> metric column to summarise at the last round.

    Returns:
        Summary DataFrame indexed by config label.
    """
    rows = {}
    for label, per_seed in per_seed_by_config.items():
        row = {}
        for display, col in columns.items():
            if all(col in d.columns for d in per_seed):
                # Last NON-NaN value: metrics.csv ends with a final eval row whose
                # acquisition metrics (regret/recall) are NaN — take the last real round.
                finals = []
                for d in per_seed:
                    valid = d[col].dropna()
                    finals.append(float(valid.iloc[-1]) if len(valid) else float("nan"))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    row[f"{display} (mean)"] = round(float(np.nanmean(finals)), 4)
                    row[f"{display} (std)"] = round(float(np.nanstd(finals)), 4)
        rows[label] = row
    return pd.DataFrame.from_dict(rows, orient="index")


# --------------------------------------------------------------------------- #
# Argparse
# --------------------------------------------------------------------------- #
def base_arg_parser(description: str, default_output: str) -> argparse.ArgumentParser:
    """Build the argument parser shared by both benchmark scripts.

    Args:
        description: Parser description (the script's module docstring).
        default_output: Default ``--output-dir`` for the script.

    Returns:
        A configured argument parser.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset",
        choices=["gfp", "flip", "proteingym"],
        default="gfp",
        help="Dataset (default: gfp).",
    )
    parser.add_argument(
        "--num-rounds", type=int, default=5, help="Acquisition rounds (default: 5)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Candidates per round (default: 50)."
    )
    parser.add_argument(
        "--num-seeds", type=int, default=3, help="Seed replications to average (default: 3)."
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="CNN epochs / GP iters / ESM head epochs (def: 20)."
    )
    parser.add_argument("--output-dir", type=str, default=default_output, help="Output directory.")
    return parser
