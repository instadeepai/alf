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

import abc

import numpy as np


class Normaliser(abc.ABC):
    """Abstract base class for label normalisers.

    Normalisers fit statistics from training labels, transform labels to a
    normalised space, and invert predictions back for reporting. Used by
    Surrogate to pre-process training labels and post-process predictions.
    """

    @abc.abstractmethod
    def fit(self, labels: np.ndarray) -> None:
        """Compute normalisation statistics from training labels.

        Args:
            labels: 1-D array of training target values.
        """

    @abc.abstractmethod
    def transform(self, labels: np.ndarray) -> np.ndarray:
        """Normalise labels using fitted statistics.

        Args:
            labels: 1-D array of target values.

        Returns:
            Normalised labels of the same shape.
        """

    @abc.abstractmethod
    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Denormalise values back to the original label space.

        Args:
            values: 1-D array of normalised values (e.g. model predictions).

        Returns:
            Values in the original label space.
        """

    def inverse_transform_variance(self, variances: np.ndarray) -> np.ndarray:
        """Denormalise predictive variances back to the original label space.

        Default implementation returns variances unchanged. Override for
        normalisers that scale labels (e.g. ZScoreNormaliser scales by std^2).

        Args:
            variances: 1-D array of normalised predictive variances.

        Returns:
            Variances in the original label space.
        """
        return variances


class IdentityNormaliser(Normaliser):
    """No-op normaliser — passes labels through unchanged."""

    def fit(self, labels: np.ndarray) -> None:
        """No-op fit.

        Args:
            labels: Ignored.
        """

    def transform(self, labels: np.ndarray) -> np.ndarray:
        """Return labels unchanged.

        Args:
            labels: Input labels.

        Returns:
            Same array unchanged.
        """
        return labels

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Return values unchanged.

        Args:
            values: Input values.

        Returns:
            Same array unchanged.
        """
        return values


class ZScoreNormaliser(Normaliser):
    """Normalises labels to zero mean and unit standard deviation.

    If standard deviation is zero (constant labels), outputs zeros.
    """

    def __init__(self) -> None:
        """Initialize ZScoreNormaliser with unset statistics."""
        self._mean: float | None = None
        self._std: float | None = None

    def fit(self, labels: np.ndarray) -> None:
        """Compute mean and standard deviation from labels.

        Args:
            labels: 1-D array of training target values.
        """
        self._mean = float(np.mean(labels))
        self._std = float(np.std(labels))

    def transform(self, labels: np.ndarray) -> np.ndarray:
        """Standardise labels to zero mean and unit variance.

        Args:
            labels: 1-D array of target values.

        Returns:
            Standardised labels.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("ZScoreNormaliser.fit() must be called before transform().")
        if self._std < 1e-10:
            return np.zeros_like(labels, dtype=float)
        return (labels - self._mean) / self._std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Denormalise predictions back to original label space.

        Args:
            values: 1-D array of normalised predictions.

        Returns:
            Predictions in original label space.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("ZScoreNormaliser.fit() must be called before inverse_transform().")
        if self._std < 1e-10:
            return np.full_like(values, fill_value=self._mean, dtype=float)
        return values * self._std + self._mean

    def inverse_transform_variance(self, variances: np.ndarray) -> np.ndarray:
        """Denormalise predictive variances by scaling by std^2.

        Args:
            variances: 1-D array of normalised predictive variances.

        Returns:
            Variances in original label space.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._std is None:
            raise RuntimeError(
                "ZScoreNormaliser.fit() must be called before inverse_transform_variance()."
            )
        if self._std < 1e-10:
            return np.zeros_like(variances, dtype=float)
        return variances * (self._std**2)


class MinMaxNormaliser(Normaliser):
    """Normalises labels to the [0, 1] range.

    If min == max (constant labels), outputs zeros.
    """

    def __init__(self) -> None:
        """Initialize MinMaxNormaliser with unset statistics."""
        self._min: float | None = None
        self._max: float | None = None

    def fit(self, labels: np.ndarray) -> None:
        """Compute min and max from labels.

        Args:
            labels: 1-D array of training target values.
        """
        self._min = float(np.min(labels))
        self._max = float(np.max(labels))

    def transform(self, labels: np.ndarray) -> np.ndarray:
        """Scale labels to [0, 1].

        Args:
            labels: 1-D array of target values.

        Returns:
            Scaled labels in [0, 1].

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._min is None or self._max is None:
            raise RuntimeError("MinMaxNormaliser.fit() must be called before transform().")
        if (self._max - self._min) < 1e-10:
            return np.zeros_like(labels, dtype=float)
        return (labels - self._min) / (self._max - self._min)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Rescale predictions back to original label range.

        Args:
            values: 1-D array of normalised predictions in [0, 1].

        Returns:
            Predictions in original label space. Returns a constant array equal
            to _min when min == max (constant training labels).

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._min is None or self._max is None:
            raise RuntimeError("MinMaxNormaliser.fit() must be called before inverse_transform().")
        if (self._max - self._min) < 1e-10:
            return np.full_like(values, fill_value=self._min, dtype=float)
        return values * (self._max - self._min) + self._min

    def inverse_transform_variance(self, variances: np.ndarray) -> np.ndarray:
        """Denormalise predictive variances by scaling by (max - min)^2.

        Args:
            variances: 1-D array of normalised predictive variances.

        Returns:
            Variances in original label space. Returns zeros when min == max
            (constant training labels).

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._min is None or self._max is None:
            raise RuntimeError(
                "MinMaxNormaliser.fit() must be called before inverse_transform_variance()."
            )
        scale = self._max - self._min
        if scale < 1e-10:
            return np.zeros_like(variances, dtype=float)
        return variances * (scale**2)
