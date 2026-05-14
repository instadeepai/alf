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

from dataclasses import dataclass


@dataclass
class BaseTrainConfig:
    """Shared training configuration for all models.

    Attributes:
        standardise_outputs: Whether to standardise training labels to zero mean
            and unit variance before training. The inverse transform is applied
            automatically at prediction time so predictions are always returned
            in the original label scale. Defaults to True.
        normalise_inputs: Whether to normalise input features to the [0, 1] range
            (min-max normalisation) before training. Useful for precomputed
            continuous features with heterogeneous scales. Has little effect on
            one-hot encoded features which are already in {0, 1}. Defaults to False.
    """

    standardise_outputs: bool = True
    normalise_inputs: bool = False
