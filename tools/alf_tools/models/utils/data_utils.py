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

import numpy as np
import torch
from alf_core.model.normaliser import InputNormaliser, OutputStandardiser
from torch import dtype as TorchDtype


def transform_data(
    train_x: torch.Tensor,
    labels: np.ndarray,
    normalise_inputs: bool,
    standardise_outputs: bool,
    label_dtype: TorchDtype,
    device: torch.device,
    feature_dtype: TorchDtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, InputNormaliser | None, OutputStandardiser | None]:
    """Apply input normalisation and output standardisation to training data.

    Args:
        train_x: Training features tensor.
        labels: Training labels array.
        normalise_inputs: Whether to apply min-max normalisation to input features.
        standardise_outputs: Whether to apply Z-score standardisation to output labels.
        label_dtype: Data type for the output labels.
        device: Device to move tensors to.
        feature_dtype: Data type for the input features. Defaults to float32.

    Returns:
        Transformed training features, training labels, and the fitted normalisers.
    """
    input_normaliser = None
    output_standardiser = None

    if normalise_inputs:
        train_x_np = train_x.cpu().numpy()
        input_normaliser = InputNormaliser()
        input_normaliser.fit(train_x_np)
        train_x_np = input_normaliser.transform(train_x_np)
        train_x = torch.tensor(train_x_np, dtype=feature_dtype).to(device)
    else:
        train_x = train_x.to(device=device, dtype=feature_dtype)

    if standardise_outputs:
        output_standardiser = OutputStandardiser()
        output_standardiser.fit(labels)
        labels = output_standardiser.transform(labels)

    train_y = torch.tensor(labels, dtype=label_dtype).to(device)
    return train_x, train_y, input_normaliser, output_standardiser
