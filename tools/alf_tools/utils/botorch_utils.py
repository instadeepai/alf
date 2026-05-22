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

"""Utility functions for BoTorch integration with ALF.

This module provides conversion utilities between ALF's data structures
(Candidates, Predictions) and BoTorch's expected formats (tensors, posteriors).
"""

import logging
from typing import Any

import numpy as np
import torch
from alf_core import Candidate, Predictions
from alf_core.dataclasses.candidate import Modality
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DiagLinearOperator

logger = logging.getLogger(__name__)


def candidates_to_tensor(
    candidates: list[Candidate],
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Convert a list of ALF Candidates to a PyTorch tensor.

    This function extracts the data from each candidate and stacks them
    into a single tensor suitable for BoTorch operations.

    Args:
        candidates: List of Candidate objects. Each candidate's data should be
            a numpy array or torch tensor of shape (d,) where d is the feature dimension.
        device: Optional device to place the tensor on. If None, uses CPU.
        dtype: Desired data type of the output tensor. Default is torch.float64.

    Returns:
        Tensor of shape (n, d) where n is the number of candidates and d is
        the feature dimension.

    Raises:
        ValueError: If candidates have inconsistent shapes or unsupported data types.

    Example:
        >>> candidates = [
        ...     Candidate(data=np.array([1.0, 2.0]), modality=Modality.TABULAR),
        ...     Candidate(data=np.array([3.0, 4.0]), modality=Modality.TABULAR)
        ... ]
        >>> X = candidates_to_tensor(candidates)
        >>> print(X.shape)
        torch.Size([2, 2])
    """
    if not candidates:
        raise ValueError("Cannot convert empty list of candidates to tensor")

    if device is None:
        device = torch.device("cpu")

    # Extract data from candidates
    data_list = []
    for cand in candidates:
        data = cand.data

        # Convert to numpy array if torch tensor
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        elif not isinstance(data, np.ndarray):
            # Try to convert to numpy
            data = np.asarray(data)

        data_list.append(data)

    # Stack into single array
    try:
        X = np.stack(data_list, axis=0)
    except ValueError as e:
        raise ValueError(
            f"Failed to stack candidate data. Ensure all candidates have the same shape. Error: {e}"
        ) from e

    # Convert to tensor
    X_tensor = torch.from_numpy(X).to(dtype).to(device)
    return X_tensor


def tensor_to_candidates(
    X: torch.Tensor,
    modality: Modality = Modality.TABULAR,
    features: dict[str, Any] | None = None,
) -> list[Candidate]:
    """Convert a PyTorch tensor to a list of ALF Candidates.

    Args:
        X: Tensor of shape (n, d) where n is the number of candidates and
            d is the feature dimension.
        modality: Modality to assign to the candidates. Default is TABULAR
            for continuous optimization.
        features: Optional dictionary of features to assign to all candidates.

    Returns:
        List of n Candidate objects.

    Raises:
        ValueError: If X is not a 2D tensor.

    Example:
        >>> X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> candidates = tensor_to_candidates(X)
        >>> print(len(candidates))
        2
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D tensor of shape (n, d), got shape {tuple(X.shape)}")
    X_numpy = X.detach().cpu().numpy()
    candidates = [
        Candidate(
            data=x, modality=modality, features=dict(features) if features is not None else None
        )
        for x in X_numpy
    ]
    return candidates


def predictions_to_posterior(
    predictions: Predictions,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> GPyTorchPosterior:
    """Convert ALF Predictions to a BoTorch GPyTorchPosterior.

    This allows using ALF's surrogate model predictions with BoTorch's
    acquisition functions.

    Args:
        predictions: ALF Predictions object containing means and variances.
        device: Optional device to place tensors on. If None, uses CPU.
        dtype: Desired data type of the output tensor. Default is torch.float64.

    Returns:
        BoTorch GPyTorchPosterior wrapping the predictions.

    Raises:
        ValueError: If predictions don't contain variances (required for posterior).
        RuntimeError: If variances contain significantly negative values, which
            may indicate a problem with the surrogate model.

    Example:
        >>> means = np.array([1.0, 2.0, 3.0])
        >>> variances = np.array([0.1, 0.2, 0.15])
        >>> predictions = Predictions(means=means, variances=variances)
        >>> posterior = predictions_to_posterior(predictions)
        >>> print(posterior.mean.shape)
        torch.Size([3, 1])
    """
    if predictions.variances is None:
        raise ValueError(
            "Cannot convert predictions to posterior without variances. "
            "Ensure the surrogate model provides uncertainty estimates."
        )

    if device is None:
        device = torch.device("cpu")

    mean = torch.from_numpy(predictions.means).to(dtype).to(device)
    variance = torch.from_numpy(predictions.variances).to(dtype).to(device)

    min_variance = variance.min().item()
    if min_variance < -1e-4:
        raise RuntimeError(
            "Predictions contain significantly negative variances (min=%.6g). "
            "This may indicate a problem with the surrogate model.",
            min_variance,
        )

    variance_clamped = torch.clamp(variance, min=1e-6)
    covar = DiagLinearOperator(variance_clamped)

    # Create multivariate normal distribution
    mvn = MultivariateNormal(mean, covar)

    # Wrap in GPyTorchPosterior
    posterior = GPyTorchPosterior(mvn)
    return posterior


def get_bounds_tensor(
    bounds: np.ndarray | list[tuple[float, float]],
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Convert bounds to BoTorch format.

    BoTorch expects bounds as a tensor of shape (2, d) where the first row
    contains lower bounds and the second row contains upper bounds.

    Args:
        bounds: Bounds in one of the following formats:
            - numpy array of shape (2, d): [[lower_1, ..., lower_d],
                                            [upper_1, ..., upper_d]]
            - list of tuples: [(lower_1, upper_1), ..., (lower_d, upper_d)]
        device: Optional device to place the tensor on. If None, uses CPU.
        dtype: Desired data type of the output tensor. Default is torch.float64.

    Raises:
        ValueError: If bounds are not in a valid format or if lower
        bounds are not <= upper bounds.

    Returns:
        Tensor of shape (2, d) in BoTorch format.

    Example:
        >>> bounds = [(0.0, 1.0), (0.0, 1.0)]
        >>> bounds_tensor = get_bounds_tensor(bounds)
        >>> print(bounds_tensor.shape)
        torch.Size([2, 2])
    """
    if device is None:
        device = torch.device("cpu")

    def is_valid_bounds_array(arr):
        return (
            isinstance(arr, np.ndarray)
            and arr.ndim == 2
            and arr.shape[0] == 2
            and np.all(arr[0] <= arr[1])  # Ensure lower <= upper
        )

    if isinstance(bounds, np.ndarray):
        # Already in (2, d) format
        bounds_array = bounds
    else:
        # Convert list of tuples to (2, d) array
        bounds_array = np.array(bounds).T  # Transpose to get (2, d)

    if not is_valid_bounds_array(bounds_array):
        raise ValueError(
            "Bounds must be a numpy array of shape (2, d) with lower bounds <= upper bounds"
        )

    bounds_tensor = torch.from_numpy(bounds_array).to(dtype).to(device)
    return bounds_tensor
