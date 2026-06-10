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

"""Tests that seeding makes RNG draws reproducible."""

import numpy as np
from alf_benchmark.seeding import seed_everything


def test_numpy_reproducible_across_calls():
    """Two seedings with the same seed produce identical numpy draws."""
    seed_everything(123)
    first = np.random.randn(5)
    seed_everything(123)
    second = np.random.randn(5)
    np.testing.assert_array_equal(first, second)


def test_different_seeds_differ():
    """Different seeds produce different numpy draws."""
    seed_everything(1)
    first = np.random.randn(5)
    seed_everything(2)
    second = np.random.randn(5)
    assert not np.array_equal(first, second)


def test_torch_reproducible_when_available():
    """Two seedings with the same seed produce identical torch draws."""
    torch = __import__("torch")
    seed_everything(7)
    first = torch.randn(4)
    seed_everything(7)
    second = torch.randn(4)
    assert torch.equal(first, second)
