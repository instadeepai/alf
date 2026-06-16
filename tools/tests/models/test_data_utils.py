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

import numpy as np
import torch
from alf_core.model.normaliser import InputNormaliser, InputStandardiser
from alf_tools.models.utils.data_utils import transform_data


def _make_inputs(n: int = 8) -> tuple[torch.Tensor, np.ndarray]:
    """Return a small (n, 4) float tensor and a label array."""
    x = torch.rand(n, 4)
    labels = np.random.randn(n).astype(np.float32)
    return x, labels


class TestTransformDataNormaliseInputs:
    """transform_data with the various normalise_inputs_strategy values."""

    def test_minmax_fits_input_normaliser(self):
        """The 'minmax' strategy returns a fitted InputNormaliser."""
        x, labels = _make_inputs()
        _, _, normaliser, _ = transform_data(
            x, labels, "minmax", False, torch.float32, torch.device("cpu")
        )
        assert isinstance(normaliser, InputNormaliser)
        assert normaliser.is_fitted

    def test_zscore_fits_input_standardiser(self):
        """The 'zscore' strategy returns a fitted InputStandardiser."""
        x, labels = _make_inputs()
        _, _, normaliser, _ = transform_data(
            x, labels, "zscore", False, torch.float32, torch.device("cpu")
        )
        assert isinstance(normaliser, InputStandardiser)
        assert normaliser.is_fitted

    def test_none_strategy_returns_none(self):
        """A None strategy leaves input_normaliser as None."""
        x, labels = _make_inputs()
        _, _, normaliser, _ = transform_data(
            x, labels, None, False, torch.float32, torch.device("cpu")
        )
        assert normaliser is None

    def test_minmax_produces_unit_range(self):
        """After min-max normalisation, each feature column spans [0, 1]."""
        x, labels = _make_inputs(16)
        train_x, _, _, _ = transform_data(
            x, labels, "minmax", False, torch.float32, torch.device("cpu")
        )
        x_np = train_x.numpy()
        assert x_np.min() >= -1e-6
        assert x_np.max() <= 1.0 + 1e-6

    def test_zscore_produces_zero_mean(self):
        """After z-score standardisation, each feature column has approximately zero mean."""
        x, labels = _make_inputs(32)
        train_x, _, _, _ = transform_data(
            x, labels, "zscore", False, torch.float32, torch.device("cpu")
        )
        np.testing.assert_allclose(train_x.numpy().mean(axis=0), np.zeros(4), atol=1e-5)

    def test_normalise_inputs_tensor_on_device(self):
        """Output tensor lives on the requested device."""
        x, labels = _make_inputs()
        train_x, _, _, _ = transform_data(
            x, labels, "minmax", False, torch.float32, torch.device("cpu")
        )
        assert train_x.device.type == "cpu"


class TestTransformDataStandardiseOutputs:
    """transform_data with standardise_outputs=True/False."""

    def test_standardise_outputs_true_fits_standardiser(self):
        """standardise_outputs=True returns a fitted OutputStandardiser."""
        x, labels = _make_inputs()
        _, _, _, standardiser = transform_data(
            x, labels, None, True, torch.float32, torch.device("cpu")
        )
        assert standardiser is not None
        assert standardiser.is_fitted

    def test_standardise_outputs_false_returns_none(self):
        """standardise_outputs=False leaves output_standardiser as None."""
        x, labels = _make_inputs()
        _, _, _, standardiser = transform_data(
            x, labels, None, False, torch.float32, torch.device("cpu")
        )
        assert standardiser is None

    def test_standardise_outputs_produces_zero_mean(self):
        """After standardisation, training labels have approximately zero mean."""
        x, labels = _make_inputs(32)
        _, train_y, _, _ = transform_data(x, labels, None, True, torch.float32, torch.device("cpu"))
        assert abs(train_y.numpy().mean()) < 1e-5

    def test_standardise_outputs_label_dtype_respected(self):
        """label_dtype is applied to the output tensor."""
        x, labels = _make_inputs()
        _, train_y, _, _ = transform_data(
            x, labels, None, False, torch.float64, torch.device("cpu")
        )
        assert train_y.dtype == torch.float64


class TestTransformDataBothEnabled:
    """transform_data with both input normalisation and output standardisation."""

    def test_both_enabled_returns_fitted_normalisers(self):
        """Both normaliser and standardiser are returned fitted."""
        x, labels = _make_inputs()
        _, _, normaliser, standardiser = transform_data(
            x, labels, "minmax", True, torch.float32, torch.device("cpu")
        )
        assert normaliser is not None and normaliser.is_fitted
        assert standardiser is not None and standardiser.is_fitted

    def test_both_disabled_returns_none_normalisers(self):
        """Neither normaliser nor standardiser is returned when both are disabled."""
        x, labels = _make_inputs()
        _, _, normaliser, standardiser = transform_data(
            x, labels, None, False, torch.float32, torch.device("cpu")
        )
        assert normaliser is None
        assert standardiser is None
