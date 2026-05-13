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

"""conftest for alf_tools/models tests.

Stubs out the broken transformers → huggingface_hub import chain before
chemprop/lightning/torchmetrics are loaded.

Root cause: torchmetrics tries to import transformers, which requires
huggingface_hub>=1.5.0, but the installed version is 0.27.0. A minimal stub
prevents the ImportError without affecting chemprop functionality.

TODO: remove once huggingface_hub is upgraded to >=1.5.0.
"""

import sys
import types


def _make_stub(name: str) -> types.ModuleType:
    """Create a minimal module stub and register it in sys.modules.

    Args:
        name: Fully-qualified module name to stub.

    Returns:
        The stub module.
    """
    stub = types.ModuleType(name)
    sys.modules[name] = stub
    return stub


for _mod in [
    "transformers",
    "transformers.utils",
    "transformers.utils.hub",
    "transformers.utils.versions",
    "transformers.dependency_versions_check",
]:
    if _mod not in sys.modules:
        _make_stub(_mod)

_transformers_stub = sys.modules["transformers"]
if not hasattr(_transformers_stub, "AutoModel"):
    _transformers_stub.AutoModel = None  # type: ignore[attr-defined]
if not hasattr(_transformers_stub, "AutoTokenizer"):
    _transformers_stub.AutoTokenizer = None  # type: ignore[attr-defined]
