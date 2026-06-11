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

"""Data transformation utilities for ALF model implementations."""

from typing import Literal

import numpy as np
import torch
from alf_core.model.normaliser import (
    InputNormaliser,
    InputStandardiser,
    OutputStandardiser,
    make_input_transform,
)
from torch import dtype as TorchDtype


def transform_data(
    train_x: torch.Tensor,
    labels: np.ndarray,
    normalise_inputs_strategy: Literal["minmax", "zscore"] | None,
    standardise_outputs: bool,
    label_dtype: TorchDtype,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    InputNormaliser | InputStandardiser | None,
    OutputStandardiser | None,
]:
    """Apply input normalisation and output standardisation to training data.

    Args:
        train_x: Training features tensor.
        labels: Training labels array.
        normalise_inputs_strategy: Which input normalisation to apply: `minmax`,
            `zscore`, or None to disable input normalisation.
        standardise_outputs: Whether to apply Z-score standardisation to output labels.
        label_dtype: Data type for the output labels.
        device: Device to move tensors to.

    Returns:
        Transformed training features, training labels, and the fitted normalisers.
    """
    input_normaliser = None
    output_standardiser = None

    if normalise_inputs_strategy is not None:
        train_x_np = train_x.cpu().numpy()
        input_normaliser = make_input_transform(normalise_inputs_strategy)
        input_normaliser.fit(train_x_np)
        train_x_np = input_normaliser.transform(train_x_np)
        train_x = torch.tensor(train_x_np, dtype=torch.float32).to(device)
    else:
        train_x = train_x.to(device)

    if standardise_outputs:
        output_standardiser = OutputStandardiser()
        output_standardiser.fit(labels)
        labels = output_standardiser.transform(labels)

    train_y = torch.tensor(labels, dtype=label_dtype).to(device)
    return train_x, train_y, input_normaliser, output_standardiser
