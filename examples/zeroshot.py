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

"""Zero-shot evaluation experiment.

Scores the test split with a pre-trained ESM-2 protein language model *without any
training*, using masked-marginal pseudo-log-likelihood (``scoring_function='pll'``).
This is a genuine zero-shot variant-effect baseline.

Unlike the other examples, this one needs the optional ESM-2 extra and downloads a
model checkpoint from HuggingFace on first run:

    uv sync --extra esm2          # or: pip install "alf_tools[esm2]"

Example:
    uv run python examples/zeroshot.py --dataset gfp --seed 42
"""

import argparse
import sys

import common
from alf_core import Surrogate, ZeroShotTask

try:
    from alf_tools.models import ESM2Model, ESM2ModelConfig, ESM2TrainConfig

    _ESM2_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # the optional 'transformers' extra is not installed
    _ESM2_IMPORT_ERROR = exc


def build_zeroshot_dataset(name: str, seed: int):
    """Build a dataset whose entire pool is the test split (no training data).

    Reuses :func:`common.build_dataset` then overrides the split ratios so 100% of
    the data is held out for zero-shot evaluation.

    Args:
        name: Dataset short name (``"gfp"`` or ``"flip"``).
        seed: Split seed.

    Returns:
        An un-setup dataset configured with ``test_ratio=1.0``.
    """
    dataset = common.build_dataset(name, seed)
    dataset.config.train_ratio = 0.0
    dataset.config.validation_frac = 0.0
    dataset.config.test_ratio = 1.0
    return dataset


def main() -> None:
    """Parse arguments and run the zero-shot evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_dataset_arg(parser)
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="HuggingFace ESM-2 checkpoint (default: the small 8M model).",
    )
    common.add_seed_arg(parser)
    common.add_output_arg(parser, default="examples/outputs/zeroshot")
    args = parser.parse_args()

    # Checked after argparse so --help still works without the optional extra.
    if _ESM2_IMPORT_ERROR is not None:
        sys.exit(
            "Zero-shot evaluation requires the ESM-2 extra (the 'transformers' package).\n"
            '  Install it with:  uv sync --extra esm2   (or: pip install "alf_tools[esm2]")'
        )

    common.configure_logging()
    common.set_seeds(args.seed)

    dataset = build_zeroshot_dataset(args.dataset, args.seed)
    dataset.setup()

    model = ESM2Model(
        name="esm2_zeroshot",
        model_config=ESM2ModelConfig(model_id=args.model_id, seed=args.seed),
        train_config=ESM2TrainConfig(scoring_function="pll"),
    )
    surrogate = Surrogate(model=model)
    loggers = common.build_loggers(args.output_dir)

    task = ZeroShotTask()
    state = task.setup(dataset=dataset, surrogate=surrogate)
    task.run(state=state, state_loggers=loggers)


if __name__ == "__main__":
    main()
