from alf.core.optimizer.search import SearchProtocol
from alf.core.dataclasses import TaskState, Candidate
from typing import List
from alf.core.utils.constants import PROTEIN_ALPHABET

class SingleMutantSearch(SearchProtocol):
    """Search protocol for single mutant search."""

    def __init__(self, alphabet: str = PROTEIN_ALPHABET):
        """Initialize the single mutant search protocol with the alphabet."""
        self.alphabet = alphabet

    def __call__(self, task_state: TaskState) -> List[Candidate]:
        """Apply the search protocol to return a pool of candidates."""
        train_dataset = task_state.dataset.train_dataset
        best_id = train_dataset.labels.argmax()
        best_sequence = train_dataset.candidates[best_id].data
        single_mutant_pool = []
        for i in range(len(best_sequence)):
            for j in range(len(self.alphabet)):
                if best_sequence[i] != self.alphabet[j]:
                    single_mutant_pool.append(Candidate(data=best_sequence[:i] + self.alphabet[j] + best_sequence[i+1:], label=None))
        return single_mutant_pool