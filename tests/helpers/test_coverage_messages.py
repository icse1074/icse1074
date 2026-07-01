import pytest

from test_adequacy_study.data_models.code_under_test import CUT
from test_adequacy_study.data_models.coverage_report import CoverageReport
from test_adequacy_study.data_models.task import Task
from test_adequacy_study.helpers.coverage_messages import get_missing_branch_messages


@pytest.fixture
def sample_cut() -> CUT:
    """Provides a consistent mock source code structure for testing."""
    code = """if x > 10:
        print('Large')
    else:
        print('Small')
    return True"""

    return CUT(Task(0, "def check_value(x):", "", canonical_solution=code, tests=[]), implementation=code,
               language="python")


def test_standard_jump_branch(sample_cut):
    """Scenario 1: Branch from one line to another valid destination line."""
    report = CoverageReport(coverage_pct=0, missing_branches=[[2, 4]])

    result = get_missing_branch_messages(sample_cut, report)

    assert "In line 2, the branch that jumps to line 4 was not covered:" in result
    assert "if x > 10:" in result


def test_exit_jump_branch(sample_cut):
    """Scenario 2: Branch where destination is -1 (missing exit path)."""
    report = CoverageReport(coverage_pct=0, missing_branches=[[2, -1]])

    result = get_missing_branch_messages(sample_cut, report)

    assert "In line 2 the condition was never False" in result
    assert "the path exiting this block/function was missed" in result
    assert "if x > 10:" in result


def test_self_jump_branch(sample_cut):
    """Scenario 3: Branch where source equals destination (implicit loops/yields)."""
    report = CoverageReport(coverage_pct=0, missing_branches=[[5, 5]])

    result = get_missing_branch_messages(sample_cut, report)

    assert "In line 5 the implicit loop exit or internal branch path was never covered:" in result
    assert "print('Small')" in result


def test_multiple_missing_branches(sample_cut):
    """Ensure multiple branches are concatenated with newlines into a single string."""
    report = CoverageReport(coverage_pct=0, missing_branches=[[2, 4], [2, -1]])

    result = get_missing_branch_messages(sample_cut, report)

    # Split the result back down to verify both messages are present
    lines = result.splitlines()
    assert len(lines) == 2
    assert "jumps to line 4" in lines[0]
    assert "condition was never False" in lines[1]


def test_empty_missing_branches(sample_cut):
    """Edge Case: If coverage is perfect, it should return an empty string."""
    report = CoverageReport(coverage_pct=0, missing_branches=[])

    result = get_missing_branch_messages(sample_cut, report)

    assert result == ""
