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

"""Shared pytest fixtures for tools tests.

This module provides fixtures needed by the data-conversion utility tests
(test_botorch_utils.py). GP model, acquisition, and dataset fixtures are
added in later PRs.
"""

import numpy as np
import pytest
import torch
from alf_core import Candidate
from alf_core.dataclasses.candidate import Modality


@pytest.fixture
def test_candidates_2d():
    """Create 2D test candidates (tabular data).

    Returns:
        List of 3 Candidate objects with 2D numpy arrays.

    Example:
        >>> def test_with_candidates(test_candidates_2d):
        ...     assert len(test_candidates_2d) == 3
        ...     assert test_candidates_2d[0].data.shape == (2,)
    """
    return [
        Candidate(data=np.array([0.5, 0.5], dtype=np.float32), modality=Modality.TABULAR),
        Candidate(data=np.array([0.3, 0.7], dtype=np.float32), modality=Modality.TABULAR),
        Candidate(data=np.array([0.8, 0.2], dtype=np.float32), modality=Modality.TABULAR),
    ]


@pytest.fixture
def test_tensor_2d():
    """Create 2D test tensor for BoTorch operations.

    Returns:
        Torch tensor of shape (3, 2) with float32 dtype.

    Example:
        >>> def test_with_tensor(test_tensor_2d):
        ...     posterior = model.posterior(test_tensor_2d)
        ...     assert posterior.mean.shape[0] == 3
    """
    return torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]], dtype=torch.float32)


@pytest.fixture
def test_bounds_2d():
    """Create 2D bounds for optimization problems.

    Returns:
        List of tuples representing bounds: [(0.0, 1.0), (0.0, 1.0)].

    Example:
        >>> def test_optimization(test_bounds_2d):
        ...     bounds_tensor = get_bounds_tensor(test_bounds_2d)
        ...     assert bounds_tensor.shape == torch.Size([2, 2])
    """
    return [(0.0, 1.0), (0.0, 1.0)]


@pytest.fixture
def random_seed():
    """Set a fixed random seed for reproducibility.

    Yields:
        The seed value (42), and resets numpy/torch random state after test.

    Example:
        >>> def test_reproducible(random_seed):
        ...     data = np.random.randn(100)
    """
    seed = 42
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    yield seed
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
