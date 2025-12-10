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

# You need to run the following command before:
# pip install pyrosetta-installer
# python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyrosetta
from core.alf_core import BaseModel, Candidate, LabeledCandidates, Predictions
from tools.alf_tools.utils.constants import PROTEIN_ALPHABET
from pyrosetta import rosetta
from pyrosetta.rosetta.core.pose import Pose


class PyRosetta(BaseModel):
    def __init__(
        self,
        dms_structure_name: str,
        dms_name: str,
        initial_relax_repeats: int = 20,
        repeats_per_prediction: int = 1,
        average_fn_over_repeats: str = "mean",
        alphabet: str = PROTEIN_ALPHABET,
        length: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.dms_structure_name = dms_structure_name
        self.dms_name = dms_name
        self.initial_relax_repeats = initial_relax_repeats
        self.repeats_per_prediction = repeats_per_prediction
        self.average_fn_over_repeats = average_fn_over_repeats
        self.alphabet = alphabet
        self.length = length
        self.seed = seed
        self._is_setup = False

    def setup(self) -> None:
        """Setup the PyRosetta model."""
        if self._is_setup:
            return

        pyrosetta.init(f"-constant_seed -jran {self.seed}")
        self.score_function = pyrosetta.get_fa_scorefxn()

        # Load the DMS structure from s3 bucket and relax structure
        remote_path = f"ProteinGym_AF2_structures/{self.dms_structure_name}.pdb"
        local_path = "structures"
        os.makedirs(local_path, exist_ok=True)
        # TODO: input_handler.download(remote_path, local_path)
        self.pose = pyrosetta.pose_from_pdb(
            f"{local_path}/{self.dms_structure_name}.pdb"
        )
        self.wt_sequence = self.pose.sequence()
        self.pose, relax_time_taken = self.relax_structure(
            self.pose, n_repeats=self.initial_relax_repeats
        )
        self.wt_score = self.score_function(self.pose)
        print(
            f"Initial relaxation took {relax_time_taken:.2f} seconds "
            f"with a score of {self.wt_score}."
        )
        self._is_setup = True

    def relax_structure(self, pose: Pose, n_repeats: int = 5) -> Tuple[Pose, float]:
        """Relaxes a Pose object using the FastRelax protocol. This involves rounds of
           side-chain packing and whole-atom-minimisations. For more detail on the protocol,
           refer to the Rosetta documentation. Saves the minimised structure as a PDB file.

        Args:
            pose: pose object to relax
            n_repeats: Number of pack-minimise rounds to perform. Larger
                values will significantly slow down the protocol.

        Returns:
            Tuple of (minimised Pose, time taken to perform the relaxation).
        """
        t0 = time.perf_counter()

        # Set up the FastRelax protocol with specific repeats.
        relax = rosetta.protocols.relax.FastRelax(n_repeats)
        relax.set_scorefxn(pyrosetta.get_fa_scorefxn())  # Score using default function
        # Don't deviate too much from starting coords as Xtal structure provided.
        relax.constrain_relax_to_start_coords(True)
        # Run protocol and save output
        relax.apply(pose)

        t1 = time.perf_counter()

        return pose, t1 - t0

    def relax_local(
        self,
        pose: Pose,
        rosetta_residue_num: List[int],
        distance_threshold: float = 8.0,
    ) -> Pose:
        """Locally minimises a structure around a given residue by moving both the backbone
            and side-chains.

        Args:
            pose: Input Pose object
            rosetta_residue_num: List of indices of residue in Pose around which miniisation
                should be performed. NB: This is likely to be different that the residue
                number in the PDB.
            distance_threshold: The radius within which a residue must be to the residue
                of interest for it to be indcluded in the minimisation. Value in Angstroms.

        Returns:
            Locally minimised Pose object.
        """

        # Setup the MoveMap to select which residues are included
        movemap = pyrosetta.MoveMap()
        movemap.set_bb(False)  # Initially set all backbone torsions to not move
        movemap.set_chi(False)  # Initially set all side chain torsions to not move

        # Identify neighbors and allow their movement
        # Since we are using CA distances here, we add 6A to the distance threshold as this
        # roughly ensures that residues with any heavy atom within the distance threshold move.
        for i in range(1, pose.total_residue() + 1):
            ca_dist = [
                (
                    pose.residue(rosetta_residue_num_i)
                    .xyz("CA")
                    .distance(pose.residue(i).xyz("CA"))
                )
                < distance_threshold + 6
                for rosetta_residue_num_i in rosetta_residue_num
            ]
            if (i in rosetta_residue_num) or any(ca_dist):
                movemap.set_bb(i, True)
                movemap.set_chi(i, True)

        # Set up the Minimization with the standard scoring function and apply
        min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover()
        min_mover.movemap(movemap)
        min_mover.score_function(self.score_function)
        min_mover.apply(pose)

        return pose

    def mutate_and_relax(self, sequence: str) -> float:
        """Mutates the wild-type structure to the given sequence and relaxes the structure.

        Args:
            sequence: Sequence to mutate the wild-type structure to.

        Returns:
            Score of the relaxed structure.
        """

        pose = self.pose.clone()

        mutants = [
            (i, b) for i, (a, b) in enumerate(zip(self.wt_sequence, sequence)) if a != b
        ]

        for index, residue in mutants:
            pyrosetta.toolbox.mutants.mutate_residue(pose, index + 1, residue)

        pose = self.relax_local(pose, [i + 1 for i, _ in mutants])
        score = self.score_function(pose)
        return -score

    def featurise(self, inputs: Union[LabeledCandidates, List[Candidate]]) -> Any:
        """Featurisation is not implemented for this model.

        Raises:
            NotImplementedError: Featurisation is not implemented for this model.
        """
        raise NotImplementedError("Featurisation is not implemented for this model.")

    def train(
        self,
        train_data: LabeledCandidates,
        val_data: LabeledCandidates | None = None,
    ) -> None:
        """Initialize the PyRosetta model by calling setup.

        For this physics-based model, training consists of setting up PyRosetta
        and relaxing the initial structure. The training data is not used.

        Args:
            train_data: Training data (not used for physics-based model).
            val_data: Validation data (not used for physics-based model).
        """
        self.setup()

    def predict(self, candidate_points: List[Candidate]) -> Predictions:
        """Predict fitness scores for the given candidate points.

        Procedure for scoring a candidate:
            - Mutate the wild-type structure to the given sequence
            - Relax the structure
            - Score the structure
            - Return the score

        Args:
            candidate_points: List of candidate points to predict.

        Returns:
            Predictions containing fitness scores.

        Raises:
            RuntimeError: If model has not been setup.
        """
        if not self._is_setup:
            raise RuntimeError("Model not setup. Call train() or setup() first.")

        t0 = time.perf_counter()

        fitness_scores = []
        variances = []
        for candidate in candidate_points:
            scores = [
                self.mutate_and_relax(candidate.data)
                for _ in range(self.repeats_per_prediction)
            ]
            if self.average_fn_over_repeats == "mean":
                score = np.mean(np.array(scores))
            elif self.average_fn_over_repeats == "median":
                score = np.median(np.array(scores))
            else:
                raise ValueError(
                    f"Invalid average function: {self.average_fn_over_repeats}"
                )
            fitness_scores.append(score)
            if self.repeats_per_prediction > 1:
                variances.append(np.var(scores))

        return Predictions(
            means=np.array(fitness_scores),
            variances=np.array(variances) if variances else None,
        )

    def sample(self, *args: Any, **kwargs: Any) -> List[Candidate]:
        """Sampling is not implemented for this model.

        Raises:
            NotImplementedError: Sampling is not implemented for this model.
        """
        raise NotImplementedError("Sampling is not implemented for this model.")

    def get_training_summary_metrics(self) -> Dict[str, Union[float, int, np.number]]:
        """Training summary metrics are not implemented for this model.

        Raises:
            NotImplementedError: Training metrics are not implemented for this model.
        """
        raise NotImplementedError("Training metrics are not implemented for this model.")
