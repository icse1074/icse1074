import os
from pathlib import Path
import pytest
from dotenv import load_dotenv

from test_adequacy_study.benchmarks.humaneval import HumanEvalLoader
from test_adequacy_study.builders.python_program_builder import PythonProgramBuilder
from test_adequacy_study.data_models.code_under_test import CUT
from test_adequacy_study.data_models.test_suite import TestSuite, TestFramework
from test_adequacy_study.data_models.execution_report import Verdict
from test_adequacy_study.runners.pytest_runner import PytestRunner

load_dotenv()


@pytest.fixture(scope="module")
def humaneval_task():
    cache_dir = Path(os.environ["HF_DATASETS_CACHE"]) / "tmp"
    cache_dir.mkdir(parents=True, exist_ok=True)

    loader = HumanEvalLoader(cache_dir=str(cache_dir))
    return next(iter(loader.load()))


@pytest.fixture(scope="module")
def builder():
    return PythonProgramBuilder()


@pytest.fixture(scope="module")
def cut_default(builder, humaneval_task):
    return builder.build_program(task=humaneval_task)


@pytest.fixture(scope="module")
def runner():
    return PytestRunner(timeout=5.0)

@pytest.fixture
def pytest_file(tmp_path):
    test_file = tmp_path / "test_sample.py"

    test_file.write_text(
        """
from solution import has_close_elements

def test_positive():
    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False

def test_negative():
    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True
""",
        encoding="utf-8",
    )

    return test_file


@pytest.fixture
def pytest_files(tmp_path):
    file1 = tmp_path / "test_part1.py"
    file2 = tmp_path / "test_part2.py"

    file1.write_text(
        """
from solution import has_close_elements

def test_positive():
    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False
""",
        encoding="utf-8",
    )

    file2.write_text(
        """
from solution import has_close_elements

def test_negative():
    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True
""",
        encoding="utf-8",
    )

    return [file1, file2]


def test_pytest_runner_passes_correct_solution(runner, cut_default):
    test_suite= TestSuite(
        task_id="simple_pytest_task",
        source="""
from solution import has_close_elements

def test_add_positive():
    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False

def test_add_negative():
    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True
""",
        framework=TestFramework.PYTEST,
        language="python",
    )
    report = runner.run(cut_default, test_suite)

    assert report.verdict == Verdict.PASSED
    assert len(report.detailed_test_results) == 2
    assert report.detailed_test_results[0].outcome == "passed"
    assert report.detailed_test_results[1].outcome == "passed"
    assert report.detailed_test_results[0].message is None
    assert report.detailed_test_results[1].message is None


def test_pytest_runner_detects_syntax_error(runner, humaneval_task):
    failed_cut = CUT(
        task=humaneval_task,
        implementation="""
    return False
        """,
        language="python",
    )

    test_suite = TestSuite(
        task_id="simple_pytest_task",
        source="""
    from solution import has_close_elements

    def test_add_positive():
        assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False

    def test_add_negative():
        assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True
    """,
        framework=TestFramework.PYTEST,
        language="python",
    )

    report = runner.run(failed_cut, test_suite)

    assert report.verdict == Verdict.SYNTAX_ERROR
    assert report.detailed_test_results is None


def test_pytest_runner_detects_syntax_error(runner, humaneval_task):
    test_suite = TestSuite(
        task_id="simple_pytest_task",
        source="""
    from solution import has_close_elements

    def test_add_positive():
        assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False

    def test_add_negative():
        assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True
    """,
        framework=TestFramework.PYTEST,
        language="python",
    )


    bad_cut = CUT(
        task=humaneval_task,
        implementation="def broken(:",
        language="python",
    )

    report = runner.run(bad_cut, test_suite)

    assert report.verdict == Verdict.SYNTAX_ERROR



def test_pytest_runner_with_multiple_test_files(
    runner,
    cut_default,
    pytest_files,
):
    test_suite = TestSuite(
        task_id="pytest_multi_path_task",
        source=pytest_files,
        framework=TestFramework.PYTEST,
        language="python",
    )

    report = runner.run(cut_default, test_suite)

    assert report.verdict == Verdict.PASSED
    assert len(report.detailed_test_results) == 2
    assert "test_part1.py" in report.detailed_test_results[0].node_id
    assert "test_part2.py" in report.detailed_test_results[1].node_id


def test_pytest_runner_multiple_files_detect_failure(
    runner,
    humaneval_task,
    pytest_files,
):
    failed_cut = CUT(
        task=humaneval_task,
        implementation="""
return False
""",
        language="python",
    )

    test_suite = TestSuite(
        task_id="pytest_multi_failure",
        source=pytest_files,
        framework=TestFramework.PYTEST,
        language="python",
    )

    report = runner.run(failed_cut, test_suite)

    assert report.verdict == Verdict.FAILED
    assert len(report.detailed_test_results) == 2
    failed_test = [result for result in report.detailed_test_results if result.outcome == "failed"]
    assert len(failed_test) == 1
    assert failed_test[0].message == "assert False == True\n +  where False = has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)"