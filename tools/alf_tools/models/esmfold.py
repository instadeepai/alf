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

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions
from alf_core.dataclasses.candidate import Modality

from alf_tools.utils.constants import PROTEIN_ALPHABET

logger = logging.getLogger("alf-tools")

try:
    from transformers import AutoTokenizer, EsmForProteinFolding
except ImportError:
    raise ImportError(
        "transformers is not installed. Install it with:\n"
        "  pip install 'transformers>=4.36.0' 'accelerate>=0.26.0'"
    )

_VALID_AA: frozenset[str] = frozenset(PROTEIN_ALPHABET)


@dataclass
class ESMFoldConfig:
    """Configuration for ESMFold protein structure prediction model.

    Attributes:
        model_name: HuggingFace hub ID or absolute local path to the ESMFold checkpoint.
        device: PyTorch device string ('cpu', 'cuda', 'cuda:0', 'mps').
        scoring_metric: Scalar metric returned as the oracle score.
        combined_ptm_weight: Weight of pTM in the combined metric; (1-w) applied to mean_plddt.
        batch_size: Number of sequences processed per forward pass.
        chunk_size: Axial-attention chunk size for long sequences; None disables chunking.
        low_memory: Offload encoder layers to CPU between forward passes to reduce VRAM usage.
    """

    model_name: str = "facebook/esmfold_v1"
    device: str = "cpu"
    scoring_metric: Literal["ptm", "mean_plddt", "combined"] = "ptm"
    combined_ptm_weight: float = 0.5
    batch_size: int = 1
    chunk_size: int | None = None
    low_memory: bool = False


class ESMFoldModel(BaseModel):
    """ESMFold protein structure prediction oracle model.

    Uses the ESMFold model from Meta to predict protein structure and
    return confidence scores (pTM and/or mean pLDDT) as oracle scores.
    """

    def featurise(self, inputs: list[Candidate]) -> Any:
        """Convert candidates to sequence strings for ESMFold input.

        Args:
            inputs: List of Candidate objects to featurize.

        Returns:
            List of amino acid sequence strings.
        """
        raise NotImplementedError

    def predict(self, inputs: list[Candidate]) -> Predictions:
        """Run ESMFold forward pass and return structure confidence scores.

        Args:
            inputs: List of Candidate objects containing protein sequences.

        Returns:
            Predictions containing oracle scores derived from pTM/pLDDT.
        """
        raise NotImplementedError

    def train(self, labelled_candidates: LabelledCandidates) -> None:
        """ESMFold is a pretrained model and does not support fine-tuning.

        Args:
            labelled_candidates: Labelled candidates (unused).
        """
        raise NotImplementedError
