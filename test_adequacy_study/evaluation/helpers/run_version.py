from test_adequacy_study.builders.python_program_builder import PythonProgramBuilder
from test_adequacy_study.data_models.code_under_test import CUT
from test_adequacy_study.data_models.task import Task
from test_adequacy_study.runners.pytest_runner import PytestRunner


def run_tests_on_code(
        cut : CUT,
        task: Task,
        tests: str,
        builder: PythonProgramBuilder,
        runner: PytestRunner,
) -> dict[str, dict]:
    suite = builder.build_tests(task, test_suite=tests)
    result = runner.run(cut=cut, suite=suite)

    return {
        t.node_id.split("/", 1)[1]: {
            "outcome": t.outcome,
            "message": t.message,
        }
        for t in (result.detailed_test_results or [])
    }