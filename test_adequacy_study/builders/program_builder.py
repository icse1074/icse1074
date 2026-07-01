from abc import ABC, abstractmethod

from test_adequacy_study.data_models.task import Task
from test_adequacy_study.data_models.code_under_test import CUT
from test_adequacy_study.data_models.test_suite import TestSuite, TestSource


class ProgramBuilder(ABC):

    @abstractmethod
    def build_program(self, task: Task, code: str = None) -> CUT:
        """
        Build a CUT from generated code.
        Falls back to task.canonical_solution if code is None.
        """
        ...

    @abstractmethod
    def build_tests(
        self,
        task: Task,
        test_suite: TestSource = None,
    ) -> TestSuite:
        """
        Build a TestSuite for a task.
        Falls back to task.tests if test_suite is None.
        """
        ...