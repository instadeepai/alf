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

r"""Offline design experiment.

Runs a multi-round active-learning loop where the oracle is the dataset itself:
acquired candidates are "labelled" by looking up their known values. This is the
standard benchmarking setup -- the search pool is the fixed dataset.

Example:
    uv run python examples/offline_design.py --dataset gfp --model cnn \\
        --num-rounds 5 --batch-size 50 --seed 42
"""

import argparse

import common
from alf_core import DesignTask, Oracle, Surrogate


def main() -> None:
    """Parse arguments and run the offline design loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_dataset_arg(parser)
    common.add_model_arg(parser)
    common.add_acquisition_arg(parser)
    common.add_design_args(parser)
    common.add_epochs_arg(parser)
    common.add_seed_arg(parser)
    common.add_output_arg(parser, default="examples/outputs/offline_design")
    args = parser.parse_args()

    common.configure_logging()
    common.set_seeds(args.seed)

    dataset = common.build_dataset(args.dataset, args.seed)
    dataset.setup()
    common.validate_acquisition_budget(dataset, args.num_rounds, args.batch_size)

    surrogate = Surrogate(model=common.build_model(args.model, args.seed, epochs=args.epochs))
    optimizer = common.build_optimizer(args.acquisition, args.model)
    oracle = Oracle(scorer=dataset)  # offline: the dataset answers acquisition queries
    loggers = common.build_loggers(args.output_dir)

    task = DesignTask(num_acq_rounds=args.num_rounds, acq_batch_size=args.batch_size)
    state = task.setup(dataset=dataset, surrogate=surrogate)
    task.run(state=state, state_loggers=loggers, optimizer=optimizer, oracle=oracle)


if __name__ == "__main__":
    main()
