from core.optimizers.search.base_search import SearchProtocol
from core.dataclasses import TaskState, Candidate
from typing import List

class SingleMutantSearch(SearchProtocol):
    """Search protocol for single mutant search."""

    def __call__(self, task_state: TaskState) -> List[Candidate]:
        """Apply the search protocol to return a pool of candidates."""
        alphabet = task_state.alphabet
        assert alphabet is not None, "Alphabet must be provided to the task for single mutant search"
        assert type(alphabet) == str, "Alphabet must be a string"

        train_dataset = task_state.dataset.train_dataset
        best_id = train_dataset.labels.argmax()
        best_sequence = train_dataset.candidates[best_id].data
        single_mutant_pool = []
        for i in range(len(best_sequence)):
            for j in range(len(alphabet)):
                if best_sequence[i] != alphabet[j]:
                    single_mutant_pool.append(Candidate(data=best_sequence[:i] + alphabet[j] + best_sequence[i+1:], label=None))
        return single_mutant_pool