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

import numpy as np
import pandas as pd
import pytest
import torch
from alf_core import (
    BaseDatasetConfig,
    DesignTask,
    FileStateLogger,
    Optimizer,
    Oracle,
    ProtocolSearch,
    Surrogate,
    TerminalStateLogger,
)
from alf_core.dataclasses.candidate import Candidate, Modality
from alf_core.dataclasses.labelled_candidates import LabelledCandidates
from alf_core.dataset.base_dataset import BaseDataset
from alf_tools.models import FeaturizerConfig, GPModel, GPModelConfig, GPTrainConfig
from alf_tools.models.guacamol_oracle import GuacaMolOracle, GuacaMolOracleConfig
from alf_tools.optimizer.acquisition_functions.greedy import Greedy
from alf_tools.optimizer.search.smiles_mutation_search import SmilesMutationSearch
from rdkit import Chem
from rdkit.Chem import AllChem

# A small round split is expected to under-run some regret-metric heuristics
# tuned for larger pools, which log a UserWarning rather than fail — matches
# the filter used in the sinusoidal e2e test for the same reason. The tiny
# test/candidate splits here can also make residual/target arrays near-constant,
# which scipy flags rather than errors on (same filter as test_cnn.py).
pytestmark = [
    pytest.mark.filterwarnings("ignore:num_acquisitions:UserWarning"),
    pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning"),
    pytest.mark.filterwarnings("ignore::scipy.stats.NearConstantInputWarning"),
]

# A dozen structurally diverse, drug-like seed molecules — enough for the GP
# surrogate to have real (if noisy) signal to fit, and long/varied enough that
# SmilesMutationSearch's character-substitution mutants stay valid in RDKit.
_SEED_SMILES = [
    "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34",  # osimertinib
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "c1ccc2c(c1)ccc3c2ccc4c3cccc4",  # pyrene
    "CCN(CC)CCOC(=O)c1ccc(N)cc1",  # procaine
    "Oc1ccc(Cl)cc1",  # chlorophenol
    "CC1=CC(=O)CC(C)(C)C1",  # dimethylcyclohexenone
    "COc1cc2c(cc1OC)nc(nc2N)N",  # trimethoprim-like scaffold
    "c1ccc(cc1)C(=O)Nc2ccccc2",  # benzanilide
    "CC(C)NCC(O)c1ccc(O)c(O)c1",  # isoproterenol
]


def _morgan_featurizer(smiles_list: list[str]) -> torch.Tensor:
    """Featurise SMILES as 128-bit Morgan fingerprints for the GP surrogate.

    Returns:
        Float tensor of shape (batch_size, 128).
    """
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=128)
        fingerprints.append(np.array(fp, dtype=np.float32))
    return torch.tensor(np.array(fingerprints), dtype=torch.float32)


class _SeedMoleculeDataset(BaseDataset):
    """Fixed seed corpus of drug-like molecules, labelled with osimertinib_mpo.

    Used only to bootstrap the online loop's initial train/test splits — unlike
    GuacaMol(BaseDataset), nothing here is looked up during acquisition; every
    acquired candidate is scored live by GuacaMolOracle instead.
    """

    def load_dataset(self) -> LabelledCandidates:
        """Load the fixed seed molecules with their osimertinib_mpo scores.

        Returns:
            LabelledCandidates over the seed SMILES.
        """
        scorer = GuacaMolOracle(GuacaMolOracleConfig(task_name="osimertinib_mpo"))
        candidates = [Candidate(data=s, modality=Modality.MOLECULE) for s in _SEED_SMILES]
        labels = scorer.predict(candidates).means
        return LabelledCandidates(candidates=candidates, labels=labels)


@pytest.fixture
def seed_dataset() -> _SeedMoleculeDataset:
    """Seed molecule dataset with validation_frac=0.0 so all acquisitions go to train.

    Returns:
        Configured _SeedMoleculeDataset (not yet split).
    """
    config = BaseDatasetConfig(
        name="seed_molecules",
        modality=Modality.MOLECULE,
        seed=0,
        train_ratio=0.6,
        validation_frac=0.0,
        test_ratio=0.2,
        split_type="random",
        problem_type="regression",
    )
    return _SeedMoleculeDataset(config)


@pytest.fixture
def gp_surrogate() -> Surrogate:
    """GPModel over Morgan fingerprint features.

    Returns:
        Surrogate wrapping a GPModel.
    """
    model_config = GPModelConfig(kernel_type="rbf", ard=False, mean_type="constant")
    train_config = GPTrainConfig(
        learning_rate=0.1,
        num_iterations=50,
        optimizer_type="adam",
        log_frequency=50,
    )
    featurizer_config = FeaturizerConfig(
        featurizer_type="custom",
        custom_featurizer=_morgan_featurizer,
    )
    gp = GPModel(
        name="gp_molecule_e2e",
        model_config=model_config,
        train_config=train_config,
        featurizer_config=featurizer_config,
        device="cpu",
    )
    return Surrogate(model=gp)


@pytest.fixture
def molecule_optimizer() -> Optimizer:
    """Greedy acquisition over candidates from SmilesMutationSearch.

    Returns:
        Optimizer generating and scoring novel SMILES each round.
    """
    return Optimizer(
        acquisition_fn=Greedy(),
        search_fn=ProtocolSearch(protocol=SmilesMutationSearch(max_candidates=30)),
    )


@pytest.fixture
def online_oracle() -> Oracle:
    """Online oracle scoring arbitrary SMILES via osimertinib_mpo.

    Returns:
        Oracle wrapping GuacaMolOracle.
    """
    return Oracle(scorer=GuacaMolOracle(GuacaMolOracleConfig(task_name="osimertinib_mpo")))


class TestDesignOnlineMoleculeOracle:
    """End-to-end smoke test: online molecule AL loop with a generative search."""

    def test_design_online_molecule_experiment(
        self,
        seed_dataset,
        gp_surrogate,
        molecule_optimizer,
        online_oracle,
        tmp_path,
    ):
        """Runs a 2-round online DesignTask and checks it produces sane metrics.

        Candidates are generated fresh each round by SmilesMutationSearch (not
        drawn from a fixed pool) and scored live by GuacaMolOracle (not
        looked up from precomputed labels) — this is the online counterpart to
        the offline GuacaMol(BaseDataset) usage.
        """
        save_path = tmp_path / "online_molecule_e2e"
        save_path.mkdir()

        state_loggers = [
            TerminalStateLogger(),
            FileStateLogger(output_path=save_path),
        ]

        seed_dataset.setup()
        task = DesignTask(num_acq_rounds=2, acq_batch_size=3)
        state = task.setup(dataset=seed_dataset, surrogate=gp_surrogate)
        initial_num_train = len(state.dataset.train_dataset)

        task.run(
            state=state,
            state_loggers=state_loggers,
            optimizer=molecule_optimizer,
            oracle=online_oracle,
        )

        metrics_file = save_path / "metrics.csv"
        assert metrics_file.exists(), "metrics.csv must be created"

        metrics = pd.read_csv(metrics_file)
        assert len(metrics) == 3, f"Expected round 0 + 2 acquisition rounds, got {len(metrics)}"
        assert "dataset/num_train" in metrics.columns
        assert "acquired_candidates/round_mean" in metrics.columns

        # validation_frac=0.0, so every acquired candidate goes to train.
        acq_batch_size = 3
        expected_num_train = [initial_num_train + i * acq_batch_size for i in range(3)]
        actual_num_train = metrics["dataset/num_train"].dropna().astype(int).tolist()
        assert actual_num_train == expected_num_train

        round_means = metrics["acquired_candidates/round_mean"].dropna()
        assert np.all(np.isfinite(round_means)), "All round_mean values must be finite"
        assert np.all(round_means >= 0.0) and np.all(round_means <= 1.0), (
            "osimertinib_mpo scores must stay within [0, 1]"
        )
