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
from typing import Any

import torch

logger = logging.getLogger("alf-tools")


def special_tokens_mask(input_ids: torch.Tensor, tokeniser: Any) -> torch.Tensor:
    """Return a boolean tensor True at positions occupied by CLS, EOS, PAD, or UNK tokens.

    Args:
        input_ids: Token IDs of any shape.
        tokeniser: A HuggingFace tokeniser exposing the special-token id attributes.

    Returns:
        A boolean tensor matching `input_ids` shape, True at special-token positions.
    """
    special_ids = {
        tokeniser.cls_token_id,
        tokeniser.eos_token_id,
        tokeniser.pad_token_id,
        tokeniser.unk_token_id,
    } - {None}
    special_id_tensor = torch.tensor(
        list(special_ids), dtype=input_ids.dtype, device=input_ids.device
    )
    return torch.isin(input_ids, special_id_tensor)


def compute_supervised_loss(
    preds: torch.Tensor, targets: torch.Tensor, loss_fn: str | None
) -> torch.Tensor:
    """Compute the linear-head loss between predictions and targets.

    Args:
        preds: Linear-head outputs of shape (batch, output_dim).
        targets: Target values of shape (batch,).
        loss_fn: Either 'mse' (regression) or 'cross_entropy' (classification).

    Returns:
        Scalar loss tensor.

    Raises:
        AssertionError: If `loss_fn` is not 'mse' or 'cross_entropy' (should be unreachable
            given `ESM2TrainConfig.__post_init__` validation).
    """
    if loss_fn == "mse":
        return torch.nn.functional.mse_loss(preds.squeeze(-1), targets)
    elif loss_fn == "cross_entropy":
        return torch.nn.functional.cross_entropy(preds, targets.long())
    else:
        raise AssertionError(
            f"Unreachable: loss_fn={loss_fn!r} "
            "should have been caught by ESM2TrainConfig.__post_init__"
        )


def mask_tokens(
    input_ids: torch.Tensor,
    tokeniser: Any,
    mask_probability: float,
    mask_splitting: tuple[float, float, float],
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply random token masking for MLM.

    Non-masked positions in labels are set to -100 so CrossEntropyLoss ignores them.
    Special tokens (cls, eos, pad, unk) are never masked.

    Args:
        input_ids: Token IDs of shape (batch, seq_len).
        tokeniser: A HuggingFace tokeniser exposing `mask_token_id` and `vocab_size`.
        mask_probability: Fraction of eligible tokens to mask per sequence.
        mask_splitting: (p_mask, p_random, p_unchanged) 3-way replacement probabilities.
        generator: Optional torch.Generator for reproducible masking. Pass a seeded
            generator (e.g. during validation) to keep the masking fixed across calls;
            None uses the global RNG.

    Returns:
        Tuple of (masked_input_ids, labels), both of shape (batch, seq_len).

    Raises:
        ValueError: If the tokeniser does not have a mask token.
    """
    if tokeniser.mask_token_id is None:
        raise ValueError(
            "Tokeniser has no mask token. Cannot perform MLM masking. "
            "Ensure the tokeniser is initialised with a [MASK] token."
        )

    labels = input_ids.clone()

    is_special = special_tokens_mask(input_ids, tokeniser)
    eligible = ~is_special

    prob_matrix = torch.full(input_ids.shape, mask_probability, device=input_ids.device)
    prob_matrix.masked_fill_(is_special, 0.0)
    masked = torch.bernoulli(prob_matrix, generator=generator).bool()

    rows_with_no_mask = ~masked.any(dim=1)
    if rows_with_no_mask.any():
        eligible_float = eligible[rows_with_no_mask].float()
        if eligible_float.sum(dim=1).eq(0).any():
            logger.warning(
                "One or more sequences consist entirely of special tokens. "
                "These rows will contribute zero loss. Check your data pipeline."
            )
            has_eligible = eligible_float.sum(dim=1) > 0
            if has_eligible.any():
                picks = torch.multinomial(
                    eligible_float[has_eligible], num_samples=1, generator=generator
                ).squeeze(1)
                target_rows = rows_with_no_mask.nonzero(as_tuple=True)[0][has_eligible]
                masked[target_rows, picks] = True
        else:
            picks = torch.multinomial(eligible_float, num_samples=1, generator=generator).squeeze(1)
            target_rows = rows_with_no_mask.nonzero(as_tuple=True)[0]
            masked[target_rows, picks] = True

    labels[~masked] = -100

    masked_input_ids = input_ids.clone()

    masked_indices = masked.nonzero(as_tuple=False)
    n_masked = masked_indices.shape[0]
    p_mask, p_random, _ = mask_splitting
    if n_masked > 0:
        split = torch.rand(n_masked, device=input_ids.device, generator=generator)

        replace_with_mask = split < p_mask
        if replace_with_mask.any():
            idx = masked_indices[replace_with_mask]
            masked_input_ids[idx[:, 0], idx[:, 1]] = tokeniser.mask_token_id

        replace_with_random = (split >= p_mask) & (split < (p_mask + p_random))
        if replace_with_random.any():
            idx = masked_indices[replace_with_random]
            # Sample replacements from non-special (amino-acid) tokens only, excluding
            # special tokens and [MASK] so the random branch stays distinct from masking
            # and never injects padding/CLS/EOS into the input.
            all_ids = torch.arange(tokeniser.vocab_size, device=input_ids.device)
            excluded = special_tokens_mask(all_ids, tokeniser)
            excluded |= all_ids == tokeniser.mask_token_id
            allowed_ids = all_ids[~excluded]
            random_ids = allowed_ids[
                torch.randint(
                    low=0,
                    high=allowed_ids.numel(),
                    size=(idx.shape[0],),
                    device=input_ids.device,
                    generator=generator,
                )
            ]
            masked_input_ids[idx[:, 0], idx[:, 1]] = random_ids

    return masked_input_ids, labels
