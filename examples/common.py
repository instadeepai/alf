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

"""Shared helpers for the ALF example scripts.

These helpers keep the individual example scripts short while letting each one
read top-to-bottom. They only wrap the public ALF API; nothing here is required
to use ALF -- the scripts could equally well inline this wiring.
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from alf_core import (
    BaseDatasetConfig,
    DatasetSearch,
    FileStateLogger,
    Modality,
    Optimizer,
    ProblemType,
    StateLogger,
    TerminalStateLogger,
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
from alf_tools.optimizer.acquisition_functions import UCB, ExpectedImprovement, Greedy

# Acquisition functions that need per-candidate uncertainty (variances). A plain
# CNN only predicts a mean, so pairing these with --model cnn cannot work.
_VARIANCE_REQUIRING = {"ucb", "ei"}


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy and torch for reproducible example runs.

    Args:
        seed: Random seed applied to all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_logging() -> None:
    """Send ALF's loggers to the console so metrics are visible when run as a script."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


# --------------------------------------------------------------------------- #
# Argparse helpers (compose the subset each script needs)
# --------------------------------------------------------------------------- #
def add_seed_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``--seed`` argument."""
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")


def add_output_arg(parser: argparse.ArgumentParser, default: str) -> None:
    """Add the ``--output-dir`` argument with a script-specific default."""
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default,
        help=f"Directory for metrics/prediction CSVs (default: {default}).",
    )


def add_dataset_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``--dataset`` argument (sequence datasets only)."""
    parser.add_argument(
        "--dataset",
        choices=["gfp", "flip", "proteingym"],
        default="gfp",
        help="Dataset to use (default: gfp). proteingym loads a default DMS assay "
        "(IF1_ECOLI_Kelsic_2016) from the public ProteinGym_v1 repo -- no token needed.",
    )


def add_model_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``--model`` argument (sequence-capable surrogates only)."""
    parser.add_argument(
        "--model",
        choices=["cnn", "gp"],
        default="cnn",
        help="Surrogate model (default: cnn). MLP is omitted as it only accepts "
        "tabular/embedding inputs, not raw sequences.",
    )


def add_acquisition_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``--acquisition`` argument."""
    parser.add_argument(
        "--acquisition",
        choices=["greedy", "ucb", "ei"],
        default="greedy",
        help="Acquisition function (default: greedy). ucb/ei need uncertainty, so they "
        "require --model gp.",
    )


def add_design_args(parser: argparse.ArgumentParser) -> None:
    """Add the active-learning loop arguments (rounds + batch size)."""
    parser.add_argument(
        "--num-rounds", type=int, default=5, help="Number of acquisition rounds (default: 5)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Candidates acquired per round (default: 50).",
    )


def add_epochs_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``--epochs`` training-length argument."""
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs for CNN / iterations for GP (default: 20).",
    )


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def build_dataset(name: str, seed: int) -> BaseDataset:
    """Build a sequence dataset by short name.

    Args:
        name: One of ``"gfp"`` or ``"flip"``.
        seed: Seed used for the train/val/test split.

    Returns:
        An un-setup dataset; the caller must invoke ``dataset.setup()``.

    Raises:
        ValueError: If ``name`` is not a known dataset.
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
        # A single-substitution DMS assay; edit dms_name/dms_type for other assays.
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


def build_model(name: str, seed: int, epochs: int) -> BaseModel:
    """Build a sequence-capable surrogate model by short name.

    Args:
        name: One of ``"cnn"`` or ``"gp"``.
        seed: Seed (applied globally via :func:`set_seeds`; passed through for clarity).
        epochs: CNN training epochs / GP optimisation iterations.

    Returns:
        A configured, untrained model.

    Raises:
        ValueError: If ``name`` is not a known model.
    """
    if name == "cnn":
        return CNNModel(
            name="cnn",
            model_config=CNNModelConfig(),
            train_config=CNNTrainConfig(num_epochs=epochs),
        )
    if name == "gp":
        return GPModel(
            name="gp",
            model_config=GPModelConfig(kernel_type="rbf"),
            train_config=GPTrainConfig(num_iterations=epochs),
            featurizer_config=FeaturizerConfig(featurizer_type="one_hot"),
        )
    raise ValueError(f"Unknown model: {name!r}. Expected 'cnn' or 'gp'.")


def build_optimizer(acquisition: str, model: str) -> Optimizer:
    """Build an offline (dataset-search) optimizer, validating acquisition/model fit.

    Args:
        acquisition: One of ``"greedy"``, ``"ucb"`` or ``"ei"``.
        model: The ``--model`` value, used only for the compatibility check.

    Returns:
        An :class:`Optimizer` over a fixed dataset candidate pool.

    Raises:
        SystemExit: If a variance-requiring acquisition is paired with a model that
            does not produce uncertainty (e.g. ``ucb`` with a plain ``cnn``).
        ValueError: If ``acquisition`` is unknown.
    """
    if acquisition in _VARIANCE_REQUIRING and model == "cnn":
        raise SystemExit(
            f"--acquisition {acquisition} needs per-candidate uncertainty, but --model cnn "
            f"only predicts a mean. Use --model gp, or --acquisition greedy."
        )
    if acquisition == "greedy":
        acq_fn = Greedy()
    elif acquisition == "ucb":
        acq_fn = UCB(alpha=1.0)
    elif acquisition == "ei":
        acq_fn = ExpectedImprovement()
    else:
        raise ValueError(f"Unknown acquisition: {acquisition!r}.")
    return Optimizer(acquisition_fn=acq_fn, search_fn=DatasetSearch())


def build_loggers(output_dir: str) -> list[StateLogger]:
    """Build a terminal logger plus a file logger writing under ``output_dir``.

    Args:
        output_dir: Directory for metrics/prediction CSVs; created if missing.

    Returns:
        ``[TerminalStateLogger(), FileStateLogger(output_path=output_dir)]``.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return [TerminalStateLogger(), FileStateLogger(output_path=path)]


def validate_acquisition_budget(dataset: BaseDataset, num_rounds: int, batch_size: int) -> None:
    """Fail fast if the acquisition budget would exhaust the candidate pool mid-run.

    ``DatasetSearch`` raises once the batch size meets or exceeds the remaining pool,
    which shrinks every round. Checking up front turns an opaque mid-run crash into a
    clear message before any compute is spent.

    Args:
        dataset: A dataset on which ``setup()`` has already been called.
        num_rounds: Planned number of acquisition rounds.
        batch_size: Candidates acquired per round.

    Raises:
        SystemExit: If ``num_rounds * batch_size`` would empty the pool.
    """
    pool_size = len(dataset.candidate_pool)
    requested = num_rounds * batch_size
    if requested >= pool_size:
        raise SystemExit(
            f"Acquisition budget too large: --num-rounds {num_rounds} x --batch-size "
            f"{batch_size} = {requested} candidates, but the pool only holds {pool_size}. "
            f"Lower --num-rounds or --batch-size (keep their product below {pool_size})."
        )
