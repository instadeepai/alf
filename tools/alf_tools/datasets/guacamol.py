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

import copy
import logging
from pathlib import Path
from typing import Literal, get_args

import numpy as np
import requests
from alf_core import BaseDataset, Candidate, LabelledCandidates
from alf_core.dataset.base_dataset import BaseDatasetConfig
from pydantic import model_validator

logger = logging.getLogger("alf-tools")

DATAPATH = Path(__file__).parent / "data"
FILENAME_ALL = "guacamol_v1_all.smiles"
FILENAME_TRAIN = "guacamol_v1_train.smiles"
FILENAME_VALID = "guacamol_v1_valid.smiles"
FILENAME_TEST = "guacamol_v1_test.smiles"

URL_ALL = "https://figshare.com/ndownloader/files/13612745"
URL_TRAIN = "https://figshare.com/ndownloader/files/13612760"
URL_VALID = "https://figshare.com/ndownloader/files/13612766"
URL_TEST = "https://figshare.com/ndownloader/files/13612757"

GuacaMolPropertyName = Literal[
    "BertzCT", "MolLogP", "MolWt", "TPSA",
    "NumHAcceptors", "NumHDonors", "NumRotatableBonds",
    "NumAliphaticRings", "NumAromaticRings", "QED",
]
GuacaMolTaskName = Literal[
    "celecoxib_rediscovery", "troglitazone_rediscovery", "thiothixene_rediscovery",
    "aripiprazole_similarity", "albuterol_similarity", "mestranol_similarity",
    "camphor_menthol_median", "tadalafil_sildenafil_median",
    "fexofenadine_mpo", "osimertinib_mpo", "ranolazine_mpo",
    "perindopril_mpo", "amlodipine_mpo", "sitagliptin_mpo", "zaleplon_mpo",
    "c7h8n2o2_isomer", "c9h10n2o2pf2cl_isomer",
    "aripiprazole_scaffold_hop", "aripiprazole_decorator_hop",
]

ALL_PROPERTIES: frozenset[str] = frozenset(get_args(GuacaMolPropertyName))
ALL_TASKS: frozenset[str] = frozenset(get_args(GuacaMolTaskName))

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, GraphDescriptors, rdMolDescriptors
    from rdkit.Chem import QED as RDKitQED
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False


def _require_rdkit() -> None:
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for the GuacaMol dataset. "
            "Install it with: pip install 'alf_tools[benchmarks]'"
        )
