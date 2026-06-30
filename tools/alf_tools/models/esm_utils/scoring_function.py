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

import logging
from typing import Any

import numpy as np
import torch
from alf_core import Predictions
from alf_tools.models.esm_utils.loss import special_tokens_mask

logger = logging.getLogger("alf-tools")


def compute_pll(
    esm_model: Any,
    tokeniser: Any,
    device: str | torch.device,
    all_input_ids: torch.Tensor,
    all_attention_mask: torch.Tensor,
    batch_threshold: int,
) -> Predictions:
    """Compute zero-shot pseudo-log-likelihood (PLL) scores for sequences.

    Scores each sequence by masking one residue at a time and accumulating the
    log-probability the model assigns to the correct residue at that position. The
    per-sequence score is the mean (average) log-probability across all non-special
    (i.e. amino-acid) positions. Two execution modes trade off memory and speed:

    - If the number of scoreable residues is ≤ `batch_threshold`, residues are masked
        in a single batched forward pass (one masked position per batch row) to leverage
        GPU parallelism. This creates a forward pass of shape (n_residues × padded_seq_len),
        which can spike GPU memory for sequences near the threshold on large models.
    - For longer sequences, positions are masked and scored one-at-a-time to avoid
        excessive memory usage.

    Args:
        esm_model: A HuggingFace masked-LM model returning `.logits`.
        tokeniser: A HuggingFace tokeniser exposing `mask_token_id`.
        device: Device to run the forward passes on.
        all_input_ids: Tensor of shape (n_candidates, seq_len) with token IDs.
        all_attention_mask: Tensor of shape (n_candidates, seq_len) with 1
            for non-padding tokens.
        batch_threshold: Maximum number of scoreable residues to score in a single
            batched forward pass before falling back to position-by-position scoring.

    Returns:
        Predictions: means is a float32 numpy array of per-sequence PLL
            scores (average log-likelihood per residue).

    Raises:
        ValueError: If any sequence has no scoreable residue positions
            (e.g. all special tokens).
    """
    n_candidates = all_input_ids.shape[0]

    # PLL: mask one residue at a time, scored per sequence
    log_likelihoods: list[float] = []
    _pll_oom_warned = False
    with torch.no_grad():
        for i in range(n_candidates):
            input_ids_i = all_input_ids[i].unsqueeze(0).to(device)  # (1, L)
            attention_mask_i = all_attention_mask[i].unsqueeze(0).to(device)

            special_mask = special_tokens_mask(input_ids_i[0], tokeniser)
            residue_positions = (~special_mask).nonzero(as_tuple=True)[0].tolist()

            if not residue_positions:
                raise ValueError(
                    "One or more sequences have no scoreable positions (all special tokens). "
                    "Ensure each sequence contains at least one amino acid residue, "
                    "or increase max_length to avoid full truncation."
                )

            if len(residue_positions) <= batch_threshold:
                n = len(residue_positions)
                seq_len = input_ids_i.shape[1]
                if not _pll_oom_warned and n * seq_len > 50_000:
                    logger.warning(
                        "Zero-shot PLL batched forward pass: %d residues × %d padded tokens "
                        "= %d tokens. This may cause OOM on memory-constrained devices. "
                        "Reduce ESM2ModelConfig.max_length or call predict() on "
                        "smaller batches.",
                        n,
                        seq_len,
                        n * seq_len,
                    )
                    _pll_oom_warned = True
                batch_input = input_ids_i.expand(n, -1).clone()  # (N, L)
                for row, pos in enumerate(residue_positions):
                    batch_input[row, pos] = tokeniser.mask_token_id
                logits = esm_model(
                    input_ids=batch_input,
                    attention_mask=attention_mask_i.expand(n, -1),
                ).logits  # (N, L, vocab_size)
                ll = sum(
                    torch.nn.functional.log_softmax(logits[row, pos], dim=-1)[
                        input_ids_i[0, pos]
                    ].item()
                    for row, pos in enumerate(residue_positions)
                )
            else:
                ll = 0.0
                for pos in residue_positions:
                    masked_input = input_ids_i.clone()
                    masked_input[0, pos] = tokeniser.mask_token_id
                    logits = esm_model(
                        input_ids=masked_input,
                        attention_mask=attention_mask_i,
                    ).logits  # (1, L, vocab_size)
                    ll += torch.nn.functional.log_softmax(logits[0, pos], dim=-1)[
                        input_ids_i[0, pos]
                    ].item()

            log_likelihoods.append(ll / len(residue_positions))

    return Predictions(means=np.array(log_likelihoods, dtype=np.float32))
