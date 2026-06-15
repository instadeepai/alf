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

import logging
from typing import Literal

import numpy as np
from jaxtyping import Float

logger = logging.getLogger("alf-core")


class OutputStandardiser:
    """Standardises output labels to zero mean and unit variance.

    Fit on training labels, then use transform/inverse_transform to convert
    between original and standardised space. The model always returns predictions
    in the original label scale via inverse_transform.

    Handles the variance inverse transform correctly:
        var_orig = var_standardised * std²

    Edge case: if the training labels have near-zero standard deviation
    (constant targets), std is clamped to _MIN_STD to avoid division by zero.
    """

    _MIN_STD: float = 1e-8

    def __init__(self) -> None:
        """Initialize with no fitted parameters."""
        self._mean: float | None = None
        self._std: float | None = None

    @property
    def is_fitted(self) -> bool:
        """Check if fit() has been called and mean/std are available."""
        return self._mean is not None and self._std is not None

    def fit(self, Y: Float[np.ndarray, "n_samples"]) -> None:
        """Compute mean and std from training labels.

        Args:
            Y: 1-D array of training labels, shape (n_samples,).
        """
        self._mean = float(np.mean(Y))
        raw_std = float(np.std(Y))
        self._std = max(raw_std, self._MIN_STD)
        if raw_std < self._MIN_STD:
            logger.warning(
                "OutputStandardiser: training labels have near-zero std (%.2e). "
                "Clamping to %.2e. Standardisation may not be meaningful.",
                raw_std,
                self._MIN_STD,
            )

    def transform(self, Y: Float[np.ndarray, "n_samples"]) -> Float[np.ndarray, "n_samples"]:
        """Standardise labels to zero mean, unit variance.

        Args:
            Y: Array of labels to transform, shape (n_samples,).

        Returns:
            Standardised labels, shape (n_samples,).

        Raises:
            RuntimeError: If called before fit().
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("OutputStandardiser must be fitted before calling transform.")
        return (Y - self._mean) / self._std

    def inverse_transform(
        self,
        mean: Float[np.ndarray, "n_samples"],
        var: Float[np.ndarray, "n_samples"] | None = None,
    ) -> tuple[Float[np.ndarray, "n_samples"], Float[np.ndarray, "n_samples"] | None]:
        """Transform predictions back to the original label scale.

        Applies the correct variance scaling: var_orig = var_standardised * std².

        Args:
            mean: Predicted means in standardised space, shape (n_samples,).
            var: Predicted variances in standardised space, shape (n_samples,), or None.

        Returns:
            Tuple of (mean_original, var_original). var_original is None if var is None.

        Raises:
            RuntimeError: If called before fit().
        """
        if self._mean is None or self._std is None:
            raise RuntimeError(
                "OutputStandardiser must be fitted before calling inverse_transform."
            )
        mean_orig = mean * self._std + self._mean
        var_orig = var * (self._std**2) if var is not None else None
        return mean_orig, var_orig


class InputNormaliser:
    """Normalises input features to the [0, 1] range via min-max scaling.

    Fit on training features, then apply transform at both train and predict time.
    Each feature dimension is scaled independently.

    Supports both 2-D inputs (n_samples, n_features) and higher-dimensional inputs
    such as the CNN's 3-D one-hot tensors (n_samples, alphabet_size, seq_len).
    Statistics are always computed over the batch dimension (dim=0).

    Edge case: if a feature has zero range (constant column), the range is set to
    1.0 so the feature is only min-shifted (i.e., (X - min) == 0 for training).
    This prevents unseen non-zero test values from being amplified by division by a
    tiny epsilon. For one-hot inputs, ensure the training set covers all amino acids
    at all positions, or clip outputs after transform.

    Note:
        Min-max scaling is well suited for GP models, where the kernel computes
        distances between input points and benefits from inputs spanning the unit
        cube [0, 1]. For deep neural networks (e.g. CNNModel), Z-score
        standardisation (zero mean, unit variance) is generally preferred as it
        zero-centres inputs and avoids the gradient bias that arises from
        non-zero-centred activations.
    """

    _MIN_RANGE: float = 1e-8

    def __init__(self) -> None:
        """Initialize with no fitted parameters."""
        self._min: np.ndarray | None = None
        self._range: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        """Check if fit() has been called and min/range are available."""
        return self._min is not None and self._range is not None

    def fit(
        self,
        X: Float[np.ndarray, "n_samples ..."],
    ) -> None:
        """Compute per-feature min and range from training features.

        Args:
            X: Training feature tensor, shape (n_samples, ...).
               Statistics are computed over the batch dimension (dim=0),
               so each feature position gets its own min/max.
        """
        x_min, x_max = (np.min(X, axis=0), np.max(X, axis=0))
        self._min = x_min

        raw_range = x_max - x_min
        near_zero = raw_range < self._MIN_RANGE

        # Constant features get a range of 1.0 so they are only min-shifted
        # (i.e. (X - min) == 0 for training), keeping unseen non-zero values
        # bounded at predict time rather than being amplified by division by
        # a tiny epsilon. Mirror the behaviour of `InputStandardiser.fit`.
        self._range = np.where(near_zero, 1.0, raw_range)
        if np.any(near_zero):
            logger.warning(
                "InputNormaliser: %d feature(s) have near-zero range (< %.2e). "
                "Setting their range to 1.0; those features are only min-shifted.",
                int(np.sum(near_zero)),
                self._MIN_RANGE,
            )

    def transform(
        self,
        X: Float[np.ndarray, "n_samples ..."],
    ) -> Float[np.ndarray, "n_samples ..."]:
        """Scale features to [0, 1].

        Args:
            X: Feature tensor, shape (n_samples, ...), matching the dimensionality
               of the data passed to fit().

        Returns:
            Normalised feature tensor, same shape as input, values in [0, 1].

        Raises:
            RuntimeError: If called before fit().
        """
        if not self.is_fitted:
            raise RuntimeError("InputNormaliser must be fitted before calling transform.")
        return (X - self._min) / self._range

    def inverse_transform(
        self,
        X: Float[np.ndarray, "n_samples ..."],
    ) -> Float[np.ndarray, "n_samples ..."]:
        """Map normalised features in [0, 1] back to the original scale.

        Args:
            X: Normalised feature tensor, shape (n_samples, ...), matching the
               dimensionality of the data passed to fit().

        Returns:
            Feature tensor in the original scale, same shape as input.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self.is_fitted:
            raise RuntimeError("InputNormaliser must be fitted before calling inverse_transform.")
        return X * self._range + self._min


class InputStandardiser:
    """Standardises input features to zero mean and unit variance (Z-score).

    Fit on training features, then apply transform at both train and predict time.
    Each feature dimension is standardised independently.

    Supports both 2-D inputs (n_samples, n_features) and higher-dimensional inputs
    such as the CNN's 3-D one-hot tensors (n_samples, alphabet_size, seq_len).
    Statistics are always computed over the batch dimension (dim=0).

    Edge case: if a feature has near-zero standard deviation (constant column), its
    scale is set to 1.0 (rather than a tiny epsilon), and a warning is logged. The
    column is therefore only mean-centred, so unseen non-constant values at predict
    time (e.g. a one-hot position never varied in training) stay bounded instead of
    being amplified by division by a near-zero std.

    Note:
        Z-score standardisation is generally preferred for deep neural networks
        (e.g. CNNModel) as it zero-centres inputs and avoids the gradient bias that
        arises from non-zero-centred activations. For GP models, the min-max
        `InputNormaliser` is usually preferred, since the kernel computes distances
        between input points and benefits from inputs spanning the unit cube [0, 1].
    """

    _MIN_STD: float = 1e-8

    def __init__(self) -> None:
        """Initialize with no fitted parameters."""
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        """Check if fit() has been called and mean/std are available."""
        return self._mean is not None and self._std is not None

    def fit(
        self,
        X: Float[np.ndarray, "n_samples ..."],
    ) -> None:
        """Compute per-feature mean and std from training features.

        Args:
            X: Training feature tensor, shape (n_samples, ...).
               Statistics are computed over the batch dimension (dim=0),
               so each feature position gets its own mean/std.
        """
        self._mean = np.mean(X, axis=0)
        raw_std = np.std(X, axis=0)
        # Constant features get a scale of 1.0 so they are only mean-centred,
        # keeping unseen non-constant values bounded at predict time.
        near_zero = raw_std < self._MIN_STD
        self._std = np.where(near_zero, 1.0, raw_std)
        if np.any(near_zero):
            logger.warning(
                "InputStandardiser: %d feature(s) have near-zero std (< %.2e). "
                "Setting their scale to 1.0; those features are only mean-centred.",
                int(np.sum(near_zero)),
                self._MIN_STD,
            )

    def transform(
        self,
        X: Float[np.ndarray, "n_samples ..."],
    ) -> Float[np.ndarray, "n_samples ..."]:
        """Standardise features to zero mean, unit variance.

        Args:
            X: Feature tensor, shape (n_samples, ...), matching the dimensionality
               of the data passed to fit().

        Returns:
            Standardised feature tensor, same shape as input.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self.is_fitted:
            raise RuntimeError("InputStandardiser must be fitted before calling transform.")
        return (X - self._mean) / self._std

    def inverse_transform(
        self,
        X: Float[np.ndarray, "n_samples ..."],
    ) -> Float[np.ndarray, "n_samples ..."]:
        """Map standardised features back to the original scale.

        Args:
            X: Standardised feature tensor, shape (n_samples, ...), matching the
               dimensionality of the data passed to fit().

        Returns:
            Feature tensor in the original scale, same shape as input.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self.is_fitted:
            raise RuntimeError("InputStandardiser must be fitted before calling inverse_transform.")
        return X * self._std + self._mean


def make_input_transform(
    strategy: Literal["minmax", "zscore"],
) -> InputNormaliser | InputStandardiser:
    """Construct an unfitted input transform for the given strategy.

    Args:
        strategy: `minmax` for `InputNormaliser` (scaling to [0, 1]) or `zscore`
            for `InputStandardiser` (zero mean, unit variance).

    Returns:
        An unfitted `InputNormaliser` or `InputStandardiser`.

    Raises:
        ValueError: If `strategy` is not one of `minmax` or `zscore`.
    """
    if strategy == "minmax":
        return InputNormaliser()
    if strategy == "zscore":
        return InputStandardiser()
    raise ValueError(
        f"Unknown input normalisation strategy: {strategy!r}. Expected 'minmax' or 'zscore'."
    )
