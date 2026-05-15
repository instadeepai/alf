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
from transformers import AutoTokenizer, EsmForProteinFolding

from alf_tools.utils.constants import PROTEIN_ALPHABET

logger = logging.getLogger("alf-tools")


_VALID_AA: frozenset[str] = frozenset(PROTEIN_ALPHABET)


@dataclass
class ESMFoldConfig:
    """Configuration for ESMFold protein structure prediction model.

    Attributes:
        model_name: HuggingFace hub ID or absolute local path to the ESMFold checkpoint.
        device: PyTorch device string ('cpu', 'cuda', 'cuda:0', 'mps').
        scoring_metric: Scalar metric returned as the oracle score.
        combined_ptm_weight: Weight of pTM in the combined metric; (1-w) applied to mean_plddt.
        batch_size: Number of sequences processed per forward pass. Values > 1 are only valid
            with scoring_metric="mean_plddt"; ptm and combined require batch_size=1 because
            ESMFold returns a single pTM scalar per batch, not per sequence.
        chunk_size: Axial-attention chunk size for long sequences; None disables chunking.
        low_memory: Pass low_cpu_mem_usage=True to from_pretrained, deferring weight
            materialization to the first forward pass (latency spike on first call).
    """

    model_name: str = "facebook/esmfold_v1"
    device: str = "cpu"
    scoring_metric: Literal["ptm", "mean_plddt", "combined"] = "ptm"
    combined_ptm_weight: float = 0.5
    batch_size: int = 1
    chunk_size: int | None = None
    low_memory: bool = False


class ESMFoldModel(BaseModel):
    """ESMFold protein structure prediction oracle for active learning.

    Predicts pTM and/or mean pLDDT for amino acid sequence candidates using
    HuggingFace EsmForProteinFolding. Plugs into Oracle via:
        Oracle(scorer=ESMFoldModel(ESMFoldConfig(...)))
    """

    def __init__(self, config: ESMFoldConfig) -> None:
        """Load ESMFold tokenizer and model from HuggingFace or a local path.

        Args:
            config: Model source, device, and scoring configuration.

        Raises:
            ImportError: If transformers or accelerate is not installed.
            ValueError: If config parameters are out of valid ranges.
        """
        if not 0.0 <= config.combined_ptm_weight <= 1.0:
            raise ValueError(
                f"combined_ptm_weight must be in [0, 1], got {config.combined_ptm_weight}"
            )
        if config.chunk_size is not None and config.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {config.chunk_size}")
        if config.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {config.batch_size}")
        if config.batch_size > 1 and config.scoring_metric in ("ptm", "combined"):
            raise ValueError(
                f"batch_size > 1 is not supported with scoring_metric='{config.scoring_metric}'. "
                "ESMFold's pTM score is a single scalar per batch, not per-sequence. "
                "Use batch_size=1 or scoring_metric='mean_plddt'."
            )

        self.config = config
        try:
            self.device = torch.device(config.device)
        except RuntimeError as exc:
            raise ValueError(f"Invalid device string in ESMFoldConfig: {config.device!r}") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = EsmForProteinFolding.from_pretrained(
            config.model_name,
            low_cpu_mem_usage=config.low_memory,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        if config.chunk_size is not None:
            self.model.esm.encoder.set_chunk_size(config.chunk_size)

        if self.device.type == "cpu":
            # EsmForProteinFolding casts the ESM backbone to fp16 by default (fp16_esm=True).
            # CPU lacks native fp16 hardware, so emulated fp16 produces noisy representations
            # that corrupt pTM and pLDDT scores. Converting to fp32 restores accuracy.
            self.model.esm.float()
            logger.warning(
                "ESMFoldModel is running on CPU. Inference will be very slow for real proteins."
                " Use device='cuda' for production workloads."
            )

        logger.info(
            "ESMFoldModel loaded: model=%s device=%s scoring_metric=%s",
            config.model_name,
            config.device,
            config.scoring_metric,
        )

    def _validate_candidates(self, candidates: list[Candidate]) -> None:
        """Validate candidates before prediction.

        Args:
            candidates: Candidates to validate.

        Raises:
            ValueError: If empty list, wrong modality, empty sequence, or invalid AA chars.
        """
        if not candidates:
            raise ValueError("No candidates provided to predict().")
        for i, cand in enumerate(candidates):
            if cand.modality != Modality.SEQUENCE:
                raise ValueError(
                    f"ESMFoldModel only accepts Modality.SEQUENCE. "
                    f"Got {cand.modality} at index {i}."
                )
            if not cand.data:
                raise ValueError(f"Candidate at index {i} has an empty sequence.")
            invalid = [c for c in cand.data if c not in _VALID_AA]
            if invalid:
                raise ValueError(
                    f"Candidate at index {i} contains invalid amino acid character(s): "
                    f"{sorted(set(invalid))}. Valid characters: {PROTEIN_ALPHABET}"
                )

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Run ESMFold inference and return structure confidence scores.

        Args:
            candidate_points: Protein sequence candidates (Modality.SEQUENCE).

        Returns:
            Predictions with means containing the configured scoring_metric per candidate.

        Raises:
            ValueError: If any candidate fails validation.
            RuntimeError: If the forward pass fails (e.g., GPU OOM).
        """
        self._validate_candidates(candidate_points)

        sequences = [c.data for c in candidate_points]
        n = len(sequences)
        metric = self.config.scoring_metric
        # ptm_batch_scores: one float per *batch* (pTM is a scalar for the whole batch).
        # batch_size=1 is enforced in __init__ for ptm/combined, so one float per sequence.

        ptm_scores = np.zeros(n, dtype=np.float64) if metric != "mean_plddt" else None
        plddt_scores = np.zeros(n, dtype=np.float64) if metric != "ptm" else None

        with torch.no_grad():
            for i in range(0, n, self.config.batch_size):
                batch = sequences[i : i + self.config.batch_size]
                dest = slice(i, i + len(batch))  # handles partial last batch

                tokens = self.tokenizer(
                    batch, return_tensors="pt", padding=True, add_special_tokens=False
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                output = self.model(**tokens)

                if ptm_scores is not None:
                    # pTM is a batch-level scalar; replicate across every sequence
                    # in this batch so ptm_scores[dest] stays aligned with candidates.
                    ptm_scores[dest] = output.ptm.item()

                if plddt_scores is not None:
                    mask = tokens["attention_mask"]  # (B, L)
                    n_atoms = output.plddt.shape[-1]
                    # Cast to float32: avoids fp16 overflow (seq_len * n_atoms can exceed 65504).
                    # Sum atoms first to avoid an O(B*L*n_atoms) broadcast copy.
                    plddt_per_residue = output.plddt.float().sum(dim=-1)  # (B, L)
                    real_count = mask.float().sum(dim=1) * n_atoms  # (B,)
                    if (real_count == 0).any():
                        raise RuntimeError(
                            f"Batch at offset {i} contains sequences with all-padding "
                            "attention mask after tokenization."
                        )
                    masked_sum = (plddt_per_residue * mask.float()).sum(dim=1)  # (B,)
                    plddt_scores[dest] = (masked_sum / real_count).cpu().numpy()

        if metric == "ptm":
            if ptm_scores is None:
                raise RuntimeError("ptm_scores is None but metric='ptm'; this is a bug")
            means = ptm_scores
        elif metric == "mean_plddt":
            if plddt_scores is None:
                raise RuntimeError("plddt_scores is None but metric='mean_plddt'; this is a bug")
            means = plddt_scores
        else:
            if ptm_scores is None or plddt_scores is None:
                raise RuntimeError("Score arrays are None but metric='combined'; this is a bug")
            w = self.config.combined_ptm_weight
            means = w * ptm_scores + (1.0 - w) * plddt_scores

        return Predictions(means=means)

    def featurise(self, inputs: LabelledCandidates | list[Candidate]) -> None:
        """Not implemented for ESMFoldModel.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Featurisation is not implemented for ESMFoldModel.")

    def train(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates | None = None,
    ) -> None:
        """Not implemented for ESMFoldModel.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Training is not implemented for ESMFoldModel.")

    def sample(self, condition: Any | None = None) -> list[Candidate]:
        """Not implemented for ESMFoldModel.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Sampling is not implemented for ESMFoldModel.")

    def get_training_summary_metrics(self) -> dict[str, float]:
        """Return empty metrics dict (ESMFoldModel is inference-only).

        Returns:
            Empty dict; ESMFoldModel does not produce training metrics.
        """
        return {}

    def cleanup(self) -> None:
        """Move model to CPU and clear CUDA cache to free GPU memory."""
        self.model = self.model.to("cpu")
        self.device = torch.device("cpu")
        # ESM backbone was fp16 on GPU; restore fp32 so the model is usable on CPU after cleanup.
        self.model.esm.float()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
