import os
from pathlib import Path
import pytest

from test_adequacy_study.benchmarks.humaneval import HumanEvalLoader
from test_adequacy_study.builders.python_program_builder import PythonProgramBuilder
from test_adequacy_study.data_models.code_under_test import CUT
from test_adequacy_study.runners.humaneval_test_runner import HumanEvalTestRunner
from test_adequacy_study.data_models.test_suite import TestSuite, TestFramework
from test_adequacy_study.data_models.execution_report import Verdict
from dotenv import load_dotenv

load_dotenv()



@pytest.fixture(scope="module")
def humaneval_task():
    cache_dir = Path(os.environ["HF_DATASETS_CACHE"]) / "tmp"
    cache_dir.mkdir(parents=True, exist_ok=True)

    loader = HumanEvalLoader(cache_dir=str(cache_dir))

    task = next(iter(loader.load()))

    return task


@pytest.fixture(scope="module")
def builder():
    return PythonProgramBuilder()


@pytest.fixture(scope="module")
def cut_default(builder, humaneval_task):
    return builder.build_program(task=humaneval_task)


@pytest.fixture(scope="module")
def test_suite(humaneval_task):
    return TestSuite(
        task_id=humaneval_task.task_id,
        source=humaneval_task.tests,
        framework=TestFramework.HUMANEVAL,
        language="python",
    )

@pytest.fixture(scope="module")
def runner():
    return HumanEvalTestRunner(timeout=120)



#todo : check this later, it times out
def test_humaneval_small_example(runner, humaneval_task) :
    cut = CUT(
        task=humaneval_task,
        implementation="""
return False
    """,
        language="python",
    )

    suite = TestSuite(
        task_id="small/test",
        source="""
def check(function):
    assert function([1.0, 2.0, 3.0], 0.5) == True
    assert function([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True
""",
        framework=TestFramework.HUMANEVAL,
        language="python",
    )

    report = runner.run(cut, suite)
    assert report.verdict == Verdict.PASSED

def test_humaneval_runner_runs_default_cut(runner, cut_default, test_suite):
    report = runner.run(cut_default, test_suite)

    assert report.task_id == cut_default.task_id
    assert report.verdict == Verdict.PASSED

def test_humaneval_runner_detects_wrong_solution(humaneval_task, runner, test_suite):
    failing_cut = CUT(
        task=humaneval_task,
        implementation="return False",
        language="python"
    )
    report = runner.run(failing_cut, test_suite)

    assert report.verdict == Verdict.FAILED


def test_humaneval_runner_syntax_error(humaneval_task, runner, test_suite):
    bad_cut = CUT(
        task=humaneval_task,
        implementation="def broken(:",
        language="python"
    )

    report = runner.run(bad_cut, test_suite)

    assert report.verdict == Verdict.SYNTAX_ERROR

