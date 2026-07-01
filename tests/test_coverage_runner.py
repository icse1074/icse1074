import os
from pathlib import Path
import pytest

from test_adequacy_study.benchmarks.naturalcodebench import NaturalCodeBenchLoader
from test_adequacy_study.data_models.code_under_test import CUT
from test_adequacy_study.data_models.test_suite import TestSuite, TestFramework
from test_adequacy_study.runners.coverage_runner import CoverageRunner

WORK_DIR = os.environ.get("WORK_DIR")


def _read(filename: str) -> str:
    return (Path("resources") / filename).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def coverage_runner():
    return CoverageRunner(work_dir=WORK_DIR)


@pytest.fixture(scope="module")
def loader():
    return NaturalCodeBenchLoader(data_file=os.environ.get("NCB_DATASET"))


@pytest.fixture(scope="module")
def ncb_task(loader):
    return next(iter(loader.load()))


def _make_cut(ncb_task, impl_file: str) -> CUT:
    return CUT(task=ncb_task, implementation=_read(impl_file), language="python")


def _make_suite() -> TestSuite:
    return TestSuite(
        task_id="small/test",
        source=_read("suite_base.txt"),
        framework=TestFramework.PYTEST,
        language="python",
    )


def test_no_branch_coverage(coverage_runner, ncb_task):
    """Linear code with no control flow should report 0 branches."""
    cut = _make_cut(ncb_task, "no_branches.txt")
    report = coverage_runner.run(cut, _make_suite())

    assert report.missing_lines == []
    assert report.total_branches == 0
    assert report.missing_branches == []


def test_branch_coverage(coverage_runner, ncb_task):
    cut = _make_cut(ncb_task, "if_branches.txt")
    report = coverage_runner.run(cut, _make_suite())

    # Two if-statements → each with a true/false arm → 4 total
    assert report.total_branches == 4
    assert report.missing_branches == [[7, 8]]
    assert report.covered_branches == [[7, 9], [14, 15], [14, 16]]
    expected_per_test_branch_coverage = {
        'tests.test_generated.test_apostrophes_and_hyphens_removed': [(7, 9), (14, 16)],
        'tests.test_generated.test_basic_counts_and_sorting': [(7, 9), (14, 16)],
        'tests.test_generated.test_empty_file_returns_empty_dict': [(7, 9), (14, 15)],
        'tests.test_generated.test_numbers_and_unicode_words': [(7, 9), (14, 16)],
        'tests.test_generated.test_punctuation_and_case_handling': [(7, 9), (14, 16)],
    }
    assert report.per_test_branch_coverage == expected_per_test_branch_coverage


def test_nested_branch_coverage(coverage_runner, ncb_task):

    cut = _make_cut(ncb_task, "nested_branches.txt")
    report = coverage_runner.run(cut, _make_suite())

    # Outer if (file exists): 2 arms
    # Inner if (words non-empty): 2 arms
    # Innermost if (filtered non-empty): 2 arms
    # → 6 total branches
    assert report.total_branches == 6
    assert report.missing_branches == [[7, 8], [16, 20]]
    assert report.covered_branches == [[7, 9], [14, 15], [14, 22], [16, 17]]

    # (7, 9) is covered by every test
    for test, branches in report.per_test_branch_coverage.items():
        assert (7, 9) in branches

    # (7, 8) and (16, 20) are covered by no test
    for test, branches in report.per_test_branch_coverage.items():
        assert (7, 8) not in branches
        assert (16, 20) not in branches

    # (14, 22) is covered only by test_empty_file_returns_empty_dict
    for test, branches in report.per_test_branch_coverage.items():
        if test == "tests.test_generated.test_empty_file_returns_empty_dict":
            assert (14, 22) in branches
        else :
            assert (14, 22) not in branches

    # (14, 15) and (16, 17) are covered by all tests except test_empty_file_returns_empty_dict
    for test, branches in report.per_test_branch_coverage.items():
        if test == "tests.test_generated.test_empty_file_returns_empty_dict":

            assert (14, 15) not in branches
            assert (16, 17) not in branches
        else :
            assert (14, 15) in branches
            assert (16, 17) in branches

    assert (14, 15) not in report.per_test_branch_coverage['tests.test_generated.test_empty_file_returns_empty_dict']
    assert (16, 17) not in report.per_test_branch_coverage['tests.test_generated.test_empty_file_returns_empty_dict']


def test_loop_branch_coverage(coverage_runner, ncb_task):

    cut = _make_cut(ncb_task, "loop_branches.txt")
    report = coverage_runner.run(cut, _make_suite())

    # for loop: iterate + exit = 2 branches
    # if word in counts: true + false = 2 branches
    # if not os.path.exists: true + false = 2 branches
    # → 6 total
    assert report.total_branches == 6
    assert report.missing_branches == [[6, 7]]


