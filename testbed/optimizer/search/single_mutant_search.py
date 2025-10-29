from core.optimizer.search import SearchProtocol
from core.dataclasses import TaskState, Candidate
from typing import List
from core.utils.constants import PROTEIN_ALPHABET

class SingleMutantSearch(SearchProtocol):
    """Search protocol for single mutant search."""

    def __init__(self, protein_alphabet: str = PROTEIN_ALPHABET):
        """Initialize the single mutant search protocol with the protein alphabet."""
        self.protein_alphabet = protein_alphabet

    def __call__(self, task_state: TaskState) -> List[Candidate]:
        """Apply the search protocol to return a pool of candidates."""
        train_dataset = task_state.dataset.train_dataset
        best_id = train_dataset.labels.argmax()
        best_sequence = train_dataset.candidates[best_id].data
        single_mutant_pool = []
        for i in range(len(best_sequence)):
            for j in range(len(self.protein_alphabet)):
                if best_sequence[i] != self.protein_alphabet[j]:
                    single_mutant_pool.append(Candidate(data=best_sequence[:i] + self.protein_alphabet[j] + best_sequence[i+1:], label=None))
        return single_mutant_pool