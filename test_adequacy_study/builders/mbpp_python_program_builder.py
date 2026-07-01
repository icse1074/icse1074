from test_adequacy_study.builders.python_program_builder import PythonProgramBuilder
from test_adequacy_study.data_models.code_under_test import CUT
from test_adequacy_study.data_models.task import Task
from test_adequacy_study.builders.program_builder import ProgramBuilder
from test_adequacy_study.data_models.test_suite import TestSuite, TestSource, TestFramework
from test_adequacy_study.helpers.parsers import is_syntactically_valid


class MbppPythonProgramBuilder(PythonProgramBuilder):
    def build_program(self, task: Task, code: str = None) -> CUT:
        if not code:
            code = task.canonical_solution  # default : reference solution
        cut = CUT(
            task=task,
            language="python",
            implementation=code,
        )
        cut.content = code
        cut.syntactically_valid = is_syntactically_valid(cut.content)

        return cut
