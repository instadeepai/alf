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
from alf_core import BaseModel
from alf_core.model.normaliser import InputNormaliser, OutputStandardiser


class SurrogateModel(BaseModel):
    """Base class for surrogate models with shared normalisation helpers.

    Extends BaseModel with static helper methods for fitting and applying
    InputNormaliser (min-max) and OutputStandardiser (Z-score). Both GPModel
    and CNNModel inherit from this class to avoid duplicating the fit/transform
    logic.
    """

    @staticmethod
    def _fit_input_normaliser(
        train_x: torch.Tensor,
        apply: bool,
    ) -> tuple[torch.Tensor, InputNormaliser | None]:
        """Fit an InputNormaliser on train_x and apply it.

        Args:
            train_x: Training feature tensor, shape (n_samples, ...).
            apply: If False, returns (train_x, None) unchanged.

        Returns:
            Tuple of (normalised_x, fitted_normaliser). normalised_x equals
            train_x when apply is False; normaliser is None when apply is False.
        """
        if apply:
            normaliser = InputNormaliser()
            normaliser.fit(train_x)
            return normaliser.transform(train_x), normaliser
        return train_x, None

    @staticmethod
    def _fit_output_standardiser(
        train_y: np.ndarray,
        apply: bool,
    ) -> tuple[np.ndarray, OutputStandardiser | None]:
        """Fit an OutputStandardiser on train_y and apply it.

        Args:
            train_y: Training label array, shape (n_samples,).
            apply: If False, returns (train_y, None) unchanged.

        Returns:
            Tuple of (standardised_y, fitted_standardiser). standardised_y equals
            train_y when apply is False; standardiser is None when apply is False.
        """
        if apply:
            standardiser = OutputStandardiser()
            standardiser.fit(train_y)
            return standardiser.transform(train_y), standardiser
        return train_y, None
