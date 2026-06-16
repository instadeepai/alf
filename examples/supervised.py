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

"""Supervised experiment.

Trains the surrogate once on the training split and evaluates it on the test
split -- a plain supervised baseline, no acquisition loop. Useful for checking how
well a model fits a dataset before running active learning.

Example:
    uv run python examples/supervised.py --dataset gfp --model cnn --epochs 20 --seed 42
"""

import argparse

import common
from alf_core import SupervisedTask, Surrogate


def main() -> None:
    """Parse arguments and run the supervised task."""
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_dataset_arg(parser)
    common.add_model_arg(parser)
    common.add_epochs_arg(parser)
    common.add_seed_arg(parser)
    common.add_output_arg(parser, default="examples/outputs/supervised")
    args = parser.parse_args()

    common.configure_logging()
    common.set_seeds(args.seed)

    dataset = common.build_dataset(args.dataset, args.seed)
    dataset.setup()

    surrogate = Surrogate(model=common.build_model(args.model, args.seed, epochs=args.epochs))
    loggers = common.build_loggers(args.output_dir)

    task = SupervisedTask()
    state = task.setup(dataset=dataset, surrogate=surrogate)
    task.run(state=state, state_loggers=loggers)


if __name__ == "__main__":
    main()
