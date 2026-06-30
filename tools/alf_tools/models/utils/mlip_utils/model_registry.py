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

"""Model-type registry and loading helpers for mlip force fields."""

from pathlib import Path
from typing import Any

from mlip.models import Esen, ForceField, Mace, Nequip, Visnet
from mlip.models.model_io import load_model_from_zip


MODEL_TYPES: dict[str, type[Any]] = {
    "esen": Esen,
    "mace": Mace,
    "nequip": Nequip,
    "visnet": Visnet,
}


def resolve_mlip_model_cls(model_type: str) -> type[Any]:
    """Return the mlip model class for a configured model type."""
    try:
        return MODEL_TYPES[model_type]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_TYPES))
        raise ValueError(
            f"Unsupported MLIP model_type {model_type!r}; "
            f"supported values are: {supported}"
        ) from exc


def load_mlip_force_field(model_type: str, model_path: str | Path) -> ForceField:
    """Load a pretrained mlip force field from a zip file."""
    return load_model_from_zip(resolve_mlip_model_cls(model_type), model_path)
