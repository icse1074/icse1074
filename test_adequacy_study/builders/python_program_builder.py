from test_adequacy_study.data_models.code_under_test import CUT
from test_adequacy_study.data_models.task import Task
from test_adequacy_study.builders.program_builder import ProgramBuilder
from test_adequacy_study.data_models.test_suite import TestSuite, TestSource, TestFramework


class PythonProgramBuilder(ProgramBuilder):
    def build_program(self, task: Task, code: str = None) -> CUT:
        if not code:
            code = task.canonical_solution  # default : reference solution
        return CUT(
            task=task,
            language="python",
            implementation=code,
        )

    def build_tests(
            self,
            task: Task,
            test_suite: TestSource = None,
    ) -> TestSuite:

        if test_suite:
            framework = TestFramework.PYTEST
        else:
            test_suite = task.tests  # default : humaneval tests
            framework = TestFramework.HUMANEVAL

        return TestSuite(
            task_id=task.task_id,
            language="python",
            source=test_suite,
            framework=framework

        )
