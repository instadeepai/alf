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

"""Monte Carlo samplers for BoTorch acquisition functions.

This module provides sampler configurations used by BoTorch acquisition functions
to approximate expectations via Monte Carlo sampling.
"""

from typing import Literal

import torch
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler

# Type for supported MC sampler types
MCSamplerType = Literal["sobol", "iid"]


class BoTorchMCSampler:
    """Wrapper for BoTorch Monte Carlo samplers.

    This class provides a unified interface to BoTorch's MC samplers, which are
    used by acquisition functions to approximate expectations via Monte Carlo.

    Different samplers offer different trade-offs:
    - **Sobol QMC**: Quasi-Monte Carlo with better coverage (recommended)
    - **IID**: Independent sampling (faster but less efficient)

    The sampler is not a search function itself - it's a configuration object
    that gets passed to BoTorch acquisition functions to control how they
    approximate the acquisition value.

    Example:
        >>> from alf_tools.optimizer.acquisition_functions import BoTorchMCSampler
        >>> from alf_tools.optimizer.acquisition_functions import BoTorchAcquisition
        >>>
        >>> # Create Sobol QMC sampler
        >>> sampler = BoTorchMCSampler(sampler_type="sobol", num_samples=512)
        >>>
        >>> # Pass to acquisition function
        >>> acq_fn = BoTorchAcquisition(
        ...     acquisition_type="qEI",
        ...     sampler=sampler,
        ...     bounds=[[0, 1], [0, 1]]
        ... )

    Args:
        sampler_type: Type of sampler to use. Options:
            - "sobol": Sobol QMC sampler (better coverage, recommended)
            - "iid": Independent sampling (faster)
        num_samples: Number of MC samples to draw. More samples = more accurate
            but slower. Typical values: 256-512 for Sobol, 1024+ for IID.
        seed: Random seed for reproducibility. If None, uses random seed.

    Raises:
        ValueError: If sampler_type is not 'sobol' or 'iid', or if num_samples
            is not positive.
    """

    def __init__(
        self,
        sampler_type: MCSamplerType = "sobol",
        num_samples: int = 512,
        seed: int | None = None,
    ):
        """Initialize MC sampler configuration.

        Raises:
            ValueError: If sampler_type is not valid or num_samples is not positive.
        """
        self.sampler_type = sampler_type
        self.num_samples = num_samples
        self.seed = seed

        # Validate inputs
        if sampler_type not in ["sobol", "iid"]:
            raise ValueError(f"Invalid sampler_type: {sampler_type}. Must be 'sobol' or 'iid'")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

    def get_sampler(self):
        """Create and return the configured BoTorch sampler.

        Returns:
            A BoTorch sampler instance (SobolQMCNormalSampler or IIDNormalSampler).

        Raises:
            ValueError: If sampler_type is unknown.

        Example:
            >>> sampler_config = BoTorchMCSampler("sobol", 512)
            >>> sampler = sampler_config.get_sampler()
            >>> # Use with BoTorch acquisition function
        """
        if self.sampler_type == "sobol":
            return SobolQMCNormalSampler(
                sample_shape=torch.Size([self.num_samples]),
                seed=self.seed,
            )
        elif self.sampler_type == "iid":
            return IIDNormalSampler(
                sample_shape=torch.Size([self.num_samples]),
                seed=self.seed,
            )
        else:
            raise ValueError(f"Unknown sampler type: {self.sampler_type}")

    def __repr__(self) -> str:
        """String representation of the sampler configuration.

        Returns:
            String describing the sampler configuration.
        """
        return (
            f"BoTorchMCSampler(type={self.sampler_type}, "
            f"num_samples={self.num_samples}, seed={self.seed})"
        )
