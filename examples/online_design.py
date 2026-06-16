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

"""Online design experiment (API demonstration).

Unlike offline design, the search pool is *generated* each round rather than read
from a fixed dataset, and the oracle is a *model* that scores freshly proposed
sequences. Here we use:

  * ``ProtocolSearch(SingleMutantSearch())`` -- each round enumerates every single
    mutant of the current best training sequence. For a protein of length L over a
    20-letter alphabet that is ~L*19 candidates per round (e.g. ~4.5k for GFP), so
    each round is heavier than offline design.
  * A small **synthetic** sequence oracle (defined below) standing in for a real
    wet-lab assay or physics model (e.g. PyRosetta in the online tutorial). It is a
    toy scoring function, so the numbers are illustrative, not meaningful.

Because the CNN surrogate predicts only a mean (no uncertainty), this script uses
greedy acquisition. Results depend on the initial random split (the starting "best"
sequence), so they vary with --seed.

Example:
    uv run python examples/online_design.py --num-rounds 3 --batch-size 5 --seed 42
"""

import argparse

import common
import numpy as np
from alf_core import (
    Candidate,
    DesignTask,
    Optimizer,
    Oracle,
    Predictions,
    ProtocolSearch,
    Surrogate,
)
from alf_core.model.base_model import BaseModel
from alf_tools.optimizer.acquisition_functions import Greedy
from alf_tools.optimizer.search import SingleMutantSearch
from alf_tools.utils.constants import PROTEIN_ALPHABET


class SyntheticSequenceOracle(BaseModel):
    """Toy oracle: scores a sequence by its mean per-residue weight (fixed seed).

    A placeholder for a real online scorer (assay, simulator, structure model).
    Deterministic given ``seed`` and cheap to evaluate on arbitrary sequences.
    """

    def __init__(self, seed: int = 0) -> None:
        """Assign each amino acid a fixed random weight.

        Args:
            seed: Seed for the per-residue weight table.
        """
        rng = np.random.RandomState(seed)
        self._weights = dict(zip(PROTEIN_ALPHABET, rng.randn(len(PROTEIN_ALPHABET))))

    def featurise(self, inputs):  # noqa: D102 - no-op; oracle scores analytically
        return None

    def train(self, train_data, val_data):  # noqa: D102 - no-op; closed-form scorer
        return None

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Score each candidate by the mean weight of its residues.

        Args:
            candidate_points: Sequence candidates to score.

        Returns:
            Predictions whose means are the per-sequence average residue weights.
        """
        means = np.array([
            np.mean([self._weights.get(residue, 0.0) for residue in str(cand.data)])
            for cand in candidate_points
        ])
        return Predictions(means=means)

    def sample(self, condition=None):  # noqa: D102 - not used as a generator
        raise NotImplementedError("SyntheticSequenceOracle does not generate samples.")


def main() -> None:
    """Parse arguments and run the online design loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_design_args(parser)
    common.add_epochs_arg(parser)
    common.add_seed_arg(parser)
    common.add_output_arg(parser, default="examples/outputs/online_design")
    args = parser.parse_args()

    common.configure_logging()
    common.set_seeds(args.seed)

    # GFP supplies the initial labelled sequences the search mutates from.
    dataset = common.build_dataset("gfp", args.seed)
    dataset.setup()

    surrogate = Surrogate(model=common.build_model("cnn", args.seed, epochs=args.epochs))
    # Generative search + model oracle are what make this "online".
    optimizer = Optimizer(acquisition_fn=Greedy(), search_fn=ProtocolSearch(SingleMutantSearch()))
    oracle = Oracle(scorer=SyntheticSequenceOracle(seed=args.seed))
    loggers = common.build_loggers(args.output_dir)

    task = DesignTask(num_acq_rounds=args.num_rounds, acq_batch_size=args.batch_size)
    state = task.setup(dataset=dataset, surrogate=surrogate)
    task.run(state=state, state_loggers=loggers, optimizer=optimizer, oracle=oracle)


if __name__ == "__main__":
    main()
