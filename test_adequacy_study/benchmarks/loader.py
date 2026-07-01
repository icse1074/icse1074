from abc import ABC, abstractmethod
from typing import Iterator
from test_adequacy_study.data_models.task import Task


class BenchmarkLoader(ABC):
    """Abstract base class for benchmark loaders.
    """

    @abstractmethod
    def load(self, no_tests = False) -> Iterator[Task]:
        """Yield List of Task objects for each benchmark
            if no_tests, tests are not loaded
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of tasks in the benchmark."""
        ...